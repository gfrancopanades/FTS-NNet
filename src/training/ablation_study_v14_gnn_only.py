#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study Pipeline V14: GNN-LSTM with RMSE as Optimization Metric
=======================================================================

V14 replaces the composite goal metric (0.40*MAE + 0.60*weighted(1-R²)) used
in V7–V13 with pure validation RMSE as the Optuna optimization objective.

Motivation:
-----------
The MAE + (1-R²) composite metric has two fundamental problems:
1. Scale mismatch: MAE is in target units, (1-R²) is dimensionless — the
   0.40/0.60 weights are not principled and shift implicitly with target scale.
2. Wrong penalty profile: for storm forecasting, 1-2% errors are acceptable
   but 30% deviations are critical. RMSE penalises large errors quadratically
   (30x error → 900x contribution vs 30x for MAE), making it a much better
   proxy for catching extreme event mispredictions.
3. R² is distribution-dependent: the same absolute error hurts R² more on
   low-variance targets, creating an implicit and unintended bias.

Key Changes from V13:
---------------------
- Optuna objective: val_rmse (pure RMSE, no R² term)
- GoalMetricCallback: removed — val_rmse is already logged by the model
- Early stopping: still monitors val_loss (task loss used in gradient training)
- Training config: unchanged from V13 (150 epochs, patience=40, 60 trials)
- All other hyperparameter ranges: unchanged

This should:
- Select hyperparameters that genuinely minimise large prediction errors
- Avoid the scale-mismatch artifact of the MAE+R² composite
- Provide a cleaner, more interpretable optimization signal

Trade-offs:
- RMSE is more sensitive to outliers in the validation set
- Small val_rmse does not guarantee good MAE (though they are correlated)

Author: Gerard Franco
Date: March 2026
Affiliation: Universitat Politècnica de Catalunya
"""

from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import argparse
import gc
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from datetime import datetime

import optuna
import torch
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.training import ablation_study_v5_gnn_only as v5


PIPELINE_VERSION = "v14"


# V14: Same training configuration as V13
v5.GNN_CONFIG["epochs"] = 150          # Same as V13
v5.GNN_CONFIG["patience"] = 40         # Same as V13
v5.GNN_CONFIG["min_delta"] = 0.00001   # Same as V13
v5.GNN_CONFIG["trials"] = 60           # Same as V13


def _to_float(x, default: float | None = None) -> float | None:
    if x is None:
        return default
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return default
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return default


def build_experiment_name() -> str:
    """Create experiment folder name using exclusions, resolution and job id."""
    prefix = v5.get_experiment_prefix()
    resolution = v5.get_time_resolution_token()
    job_id = v5.get_job_id()
    return f"v14_gnn_{prefix}_{resolution}_{job_id}"


def create_gnn_objective(
    input_size,
    static_size,
    output_size,
    num_pks,
    train_dataset,
    val_dataset,
    log_file,
    perf_log_file,
    model_time,
    checkpoint_dir,
    output_dir,
    n_gpus=1,
    csv_path=None,
):
    """V14 Optuna objective: minimize val_rmse (pure RMSE, no R² component)."""

    HYPERPARAMS = v5.get_gnn_hyperparams()
    best_tracker = {
        "val_rmse": float("inf"),
        "trial_number": -1,
        "lock": threading.Lock(),
    }
    dataloader_runtime = {"workers_override": None, "lock": threading.Lock()}

    def objective(trial: optuna.Trial):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        gpu_id = trial.number % n_gpus
        train_loader = None
        val_loader = None

        stagger_delay = gpu_id * 2.0
        if stagger_delay > 0:
            time.sleep(stagger_delay)

        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[ERROR] GPU setup failed: {e}")
            return 1e6

        try:
            hidden_size = trial.suggest_int("hidden_size", 64, 256, step=64)
            num_layers = trial.suggest_categorical("num_layers", HYPERPARAMS["num_layers"])
            dropout_prob = trial.suggest_float(
                "dropout_prob", *HYPERPARAMS["dropout_prob_range"], log=True
            )
            learning_rate = trial.suggest_float(
                "learning_rate", *HYPERPARAMS["learning_rate_range"], log=True
            )
            weight_decay = trial.suggest_float(
                "weight_decay", *HYPERPARAMS["weight_decay_range"], log=True
            )
            pk_embed_dim = trial.suggest_categorical("pk_embed_dim", HYPERPARAMS["pk_embed_dim"])
            graph_hidden_dim = trial.suggest_categorical(
                "graph_hidden_dim", HYPERPARAMS["graph_hidden_dim"]
            )
            num_graph_layers = trial.suggest_categorical(
                "num_graph_layers", HYPERPARAMS["num_graph_layers"]
            )

            target_weight_speed = trial.suggest_float(
                "target_weight_speed", *HYPERPARAMS["target_weight_speed_range"]
            )
            target_weight_intensity = trial.suggest_float(
                "target_weight_intensity", *HYPERPARAMS["target_weight_intensity_range"]
            )
            target_weights = [target_weight_speed, target_weight_intensity, target_weight_intensity]

            if torch.cuda.is_available():
                actual_gpu = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(actual_gpu)
                trial_start_msg = (
                    f"\n{'='*80}\n"
                    f"[TRIAL {trial.number + 1}/{v5.GNN_CONFIG['trials']}] STARTING on GPU {gpu_id} "
                    f"(actual device: {actual_gpu}, name: {gpu_name})\n"
                    f"{'='*80}"
                )
            else:
                trial_start_msg = (
                    f"\n{'='*80}\n"
                    f"[TRIAL {trial.number + 1}/{v5.GNN_CONFIG['trials']}] STARTING on GPU {gpu_id} "
                    f"(CUDA not available)\n"
                    f"{'='*80}"
                )
            print(trial_start_msg)
            v5.append_to_log(trial_start_msg, log_file)

            hyperparams_msg = (
                f"[HYPERPARAMS] Testing:\n"
                f"  LSTM: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout_prob:.4f}\n"
                f"  Optimizer: lr={learning_rate:.6f}, weight_decay={weight_decay:.6f}\n"
                f"  GNN: pk_embed={pk_embed_dim}, graph_hidden={graph_hidden_dim}, graph_layers={num_graph_layers}\n"
                f"  Task target_weights: speed={target_weight_speed:.3f}, intensity={target_weight_intensity:.3f}\n"
                f"  Training: max_epochs={v5.GNN_CONFIG['epochs']}, patience={v5.GNN_CONFIG['patience']}, "
                f"min_delta={v5.GNN_CONFIG['min_delta']}\n"
                f"  Optuna objective: val_rmse (pure RMSE — penalises large errors quadratically)"
            )
            print(hyperparams_msg)
            v5.append_to_log(hyperparams_msg, log_file)

            with dataloader_runtime["lock"]:
                workers_override = dataloader_runtime["workers_override"]

            train_loader, val_loader = v5._create_trial_dataloaders(
                train_dataset, val_dataset, v5.GNN_CONFIG["batch_size"], workers_override
            )

            model = v5.SpatioTemporalGNN_LSTM_V5(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout_prob=dropout_prob,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_pks=num_pks,
                static_feature_dim=static_size,
                pk_embed_dim=pk_embed_dim,
                graph_hidden_dim=graph_hidden_dim,
                num_graph_layers=num_graph_layers,
                target_weights=target_weights,
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"trial_{trial.number}_gnn-version={model_time}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=v5.GNN_CONFIG["patience"],
                mode="min",
                min_delta=v5.GNN_CONFIG["min_delta"],
            )
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

            def _build_trainer():
                return pl.Trainer(
                    max_epochs=v5.GNN_CONFIG["epochs"],
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=[gpu_id] if torch.cuda.is_available() else 1,
                    logger=False,
                    callbacks=[
                        checkpoint_callback,
                        early_stopping,
                        v5.ClearMemoryCallback(),
                        pruning_callback,
                        v5.PerformanceLoggerCallback(perf_log_file),
                    ],
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    precision="16-mixed" if torch.cuda.is_available() else 32,
                    gradient_clip_val=v5.GNN_CONFIG["gradient_clip"],
                )

            trainer = _build_trainer()

            print(f"[TRAINING] Starting training for Trial {trial.number + 1}...")
            print(f"[INFO] Deeper training: up to {v5.GNN_CONFIG['epochs']} epochs with patience={v5.GNN_CONFIG['patience']}")
            trial_train_start = time.time()
            fallback_used = False
            try:
                trainer.fit(model, train_loader, val_loader)
            except RuntimeError as fit_error:
                if not v5._is_dataloader_worker_crash(fit_error):
                    raise

                fallback_used = True
                fallback_msg = (
                    f"[WARN] Trial {trial.number + 1} hit a DataLoader worker crash "
                    f"({fit_error}). Retrying with num_workers=0 for stability."
                )
                print(fallback_msg)
                v5.append_to_log(fallback_msg, log_file)

                with dataloader_runtime["lock"]:
                    dataloader_runtime["workers_override"] = 0

                v5._shutdown_dataloaders(train_loader, val_loader)
                train_loader = None
                val_loader = None
                if "trainer" in locals() and trainer is not None:
                    del trainer
                torch.cuda.empty_cache()
                gc.collect()

                train_loader, val_loader = v5._create_trial_dataloaders(
                    train_dataset,
                    val_dataset,
                    v5.GNN_CONFIG["batch_size"],
                    num_workers_override=0,
                )
                trainer = _build_trainer()
                trainer.fit(model, train_loader, val_loader)

            trial_train_duration = time.time() - trial_train_start

            val_task_loss = _to_float(trainer.callback_metrics.get("val_loss"), default=None)
            if val_task_loss is None:
                raise ValueError("val_loss not found in callback_metrics - training may have failed")
            if not torch.isfinite(torch.tensor(val_task_loss)):
                warn_msg = f"Trial {trial.number + 1} produced NaN/Inf val_loss (task): {val_task_loss}"
                print(f"[WARNING] {warn_msg}")
                val_task_loss = 1e6

            train_loss = _to_float(trainer.callback_metrics.get("train_loss"), default=0.0) or 0.0
            if not torch.isfinite(torch.tensor(train_loss)):
                train_loss = 0.0

            val_r2 = _to_float(trainer.callback_metrics.get("val_r2"), default=0.0) or 0.0
            val_mae = _to_float(trainer.callback_metrics.get("val_mae"), default=0.0) or 0.0
            val_rmse = _to_float(trainer.callback_metrics.get("val_rmse"), default=None)

            if val_rmse is None or not torch.isfinite(torch.tensor(val_rmse)):
                warn_msg = f"Trial {trial.number + 1} produced NaN/Inf val_rmse: {val_rmse}"
                print(f"[WARNING] {warn_msg}")
                val_rmse = 1e6

            generalization_gap_task = val_task_loss - train_loss
            hit_max_epochs = (trainer.current_epoch + 1) >= v5.GNN_CONFIG["epochs"]

            minutes, seconds = divmod(trial_train_duration, 60)
            completion_msg = (
                f"\n[TRIAL {trial.number + 1}/{v5.GNN_CONFIG['trials']}] COMPLETED\n"
                f"  Epochs trained: {trainer.current_epoch + 1}/{v5.GNN_CONFIG['epochs']}"
            )
            if hit_max_epochs:
                completion_msg += " (HIT MAX - might benefit from more epochs!)\n"
            else:
                completion_msg += " (stopped early)\n"
            completion_msg += (
                f"  Training time: {int(minutes)}m {seconds:.1f}s\n"
                f"  Final metrics:\n"
                f"    - val_rmse: {val_rmse:.6f}  (OPTUNA objective)\n"
                f"    - val_loss_task: {val_task_loss:.6f}\n"
                f"    - val_r2: {val_r2:.4f}\n"
                f"    - val_mae: {val_mae:.4f}\n"
                f"    - train_loss: {train_loss:.6f}\n"
                f"    - generalization_gap_task: {generalization_gap_task:.6f}\n"
            )
            print(completion_msg)
            print("-" * 80)
            v5.append_to_log(completion_msg, log_file)

            if fallback_used:
                v5.append_to_log(
                    f"[INFO] Trial {trial.number + 1} completed after DataLoader fallback (num_workers=0).",
                    log_file,
                )

            if csv_path is not None:
                trial_results = {
                    "trial_number": trial.number + 1,
                    "gpu_id": gpu_id,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout_prob": dropout_prob,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "pk_embed_dim": pk_embed_dim,
                    "graph_hidden_dim": graph_hidden_dim,
                    "num_graph_layers": num_graph_layers,
                    "target_weight_speed": target_weight_speed,
                    "target_weight_intensity": target_weight_intensity,
                    "epochs_trained": trainer.current_epoch + 1,
                    "hit_max_epochs": hit_max_epochs,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_loss_task": val_task_loss,
                    "val_r2": val_r2,
                    "val_mae": val_mae,
                    "generalization_gap_task": generalization_gap_task,
                    "training_time_seconds": trial_train_duration,
                    "is_nan_loss": val_rmse >= 1e6,
                    "error_message": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                file_exists = os.path.exists(csv_path)
                with best_tracker["lock"]:
                    with open(csv_path, "a", newline="") as csvfile:
                        import csv as _csv

                        writer = _csv.DictWriter(csvfile, fieldnames=trial_results.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(trial_results)
                print(f"[CSV] Trial {trial.number + 1} results saved to: {csv_path}")

            is_new_best = False
            with best_tracker["lock"]:
                if val_rmse < best_tracker["val_rmse"]:
                    best_tracker["val_rmse"] = val_rmse
                    best_tracker["trial_number"] = trial.number
                    is_new_best = True

            if is_new_best:
                best_model_path = os.path.join(
                    output_dir, f"best-model_model=GNN-version={model_time}.pt"
                )
                checkpoint_files = v5.glob.glob(
                    os.path.join(checkpoint_dir, f"trial_{trial.number}_gnn-version={model_time}*.ckpt")
                )
                if checkpoint_files:
                    checkpoint = torch.load(checkpoint_files[0], map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    torch.save(state_dict, best_model_path)

                    metadata = {
                        "model_name": "SpatioTemporalGNN_LSTM_V5",
                        "model_version": "v14_rmse_objective",
                        "model_time": model_time,
                        "trial_number": trial.number + 1,
                        "input_size": input_size,
                        "static_size": static_size,
                        "output_size": output_size,
                        "num_pks": num_pks,
                        "best_val_rmse": val_rmse,
                        "best_val_loss_task": val_task_loss,
                        "epochs_trained": trainer.current_epoch + 1,
                        "hit_max_epochs": hit_max_epochs,
                        "goal_metric": {
                            "name": "val_rmse",
                            "description": "Pure RMSE — penalises large errors quadratically",
                        },
                        "training_config": {
                            "max_epochs": v5.GNN_CONFIG["epochs"],
                            "patience": v5.GNN_CONFIG["patience"],
                            "min_delta": v5.GNN_CONFIG["min_delta"],
                        },
                        "hyperparameters": {
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "dropout_prob": dropout_prob,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "pk_embed_dim": pk_embed_dim,
                            "graph_hidden_dim": graph_hidden_dim,
                            "num_graph_layers": num_graph_layers,
                            "target_weight_speed": target_weight_speed,
                            "target_weight_intensity": target_weight_intensity,
                            "target_weights": target_weights,
                        },
                        "ablation_settings": v5.ABLATION_SETTINGS,
                        "gnn_config": v5.GNN_CONFIG,
                        "test_mode": v5.TEST_MODE,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(
                        os.path.join(
                            output_dir, f"best-model-metadata_model=GNN-version={model_time}.json"
                        ),
                        "w",
                    ) as f:
                        json.dump(metadata, f, indent=2)

                    best_msg = (
                        f"\n{'='*80}\n"
                        f"NEW BEST MODEL FOUND! Trial {trial.number + 1}\n"
                        f"   val_rmse: {val_rmse:.6f} | val_loss_task: {val_task_loss:.6f} | "
                        f"val_r2: {val_r2:.4f} | val_mae: {val_mae:.4f}\n"
                        f"   Epochs: {trainer.current_epoch + 1}/{v5.GNN_CONFIG['epochs']}\n"
                        f"   Model saved to: {best_model_path}\n"
                        f"{'='*80}\n"
                    )
                    print(best_msg)
                    v5.append_to_log(
                        f"NEW BEST MODEL - Trial {trial.number + 1} - val_rmse: {val_rmse:.6f}",
                        log_file,
                    )

            del model, trainer
            v5._shutdown_dataloaders(train_loader, val_loader)
            del train_loader, val_loader
            train_loader = None
            val_loader = None
            torch.cuda.empty_cache()
            gc.collect()

            return val_rmse

        except optuna.TrialPruned:
            v5._shutdown_dataloaders(train_loader, val_loader)
            torch.cuda.empty_cache()
            gc.collect()
            raise

        except Exception as e:
            error_msg = f"[ERROR] Trial {trial.number + 1} failed: {e}"
            print(error_msg)
            v5.append_to_log(error_msg, log_file)

            v5._shutdown_dataloaders(train_loader, val_loader)
            train_loader = None
            val_loader = None
            torch.cuda.empty_cache()
            gc.collect()

            return 1e6

    return objective


def train_gnn_model(df_train_data, selected_features, output_dir, model_time):
    """Train GNN-LSTM model with Optuna optimization (V14: RMSE objective)."""

    print("\n" + "=" * 80)
    print("STAGE 1: GNN-LSTM HYPERPARAMETER OPTIMIZATION (V14: RMSE objective)")
    print("=" * 80)
    print(f"[CONFIG] Optuna trials: {v5.GNN_CONFIG['trials']}")
    print(f"[CONFIG] Max epochs: {v5.GNN_CONFIG['epochs']}")
    print(f"[CONFIG] Patience: {v5.GNN_CONFIG['patience']}")
    print(f"[CONFIG] Min delta: {v5.GNN_CONFIG['min_delta']}")
    print(f"[CONFIG] Optuna objective: val_rmse (pure RMSE)")
    sys.stdout.flush()

    log_file = os.path.join(output_dir, f"gnn_training_log_{model_time}.txt")
    perf_log_file = os.path.join(output_dir, f"gnn_perf_log_{model_time}.txt")
    checkpoint_dir = os.path.join(output_dir, "gnn_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    result = v5.prepare_gnn_data(df_train_data, selected_features, model_time, output_dir)
    (
        temporal_train,
        static_train,
        targets_train,
        temporal_val,
        static_val,
        targets_val,
        scaler_temporal,
        scaler_static,
        scaler_targets,
        df_train,
        df_val,
        cols_targets,
        temporal_cols,
        static_features,
    ) = result

    train_dataset, val_dataset, input_size, static_size, output_size, num_pks = v5.create_gnn_datasets(
        temporal_train,
        static_train,
        targets_train,
        temporal_val,
        static_val,
        targets_val,
        df_train,
        df_val,
        v5.GNN_CONFIG["batch_size"],
        v5.GNN_CONFIG["sequence_length"],
    )

    print("\n" + "=" * 80)
    print("[OPTUNA] Starting hyperparameter optimization...")
    print(f"[INFO] Input size: {input_size}, Static size: {static_size}, Output size: {output_size}")
    print(f"[INFO] Number of PKs: {num_pks}")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    csv_path = os.path.join(output_dir, f"training-evolution_model=GNN-version={model_time}.csv")
    print(f"[CSV] Training evolution will be saved to: {csv_path}")

    objective = create_gnn_objective(
        input_size,
        static_size,
        output_size,
        num_pks,
        train_dataset,
        val_dataset,
        log_file,
        perf_log_file,
        model_time,
        checkpoint_dir,
        output_dir,
        v5.N_GPUS,
        csv_path,
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=30,    # Same as V13
            interval_steps=3,
            n_min_trials=7,
        ),
    )

    print(f"\n[OPTUNA] Starting {v5.GNN_CONFIG['trials']} trials across {v5.N_GPUS} GPUs...")
    print("[OPTUNA] Using very relaxed MedianPruner with 30-epoch warmup")
    sys.stdout.flush()
    start_time = time.time()
    study.optimize(objective, n_trials=v5.GNN_CONFIG["trials"], n_jobs=v5.N_GPUS)
    duration = time.time() - start_time

    print("=" * 80)
    print("[COMPLETE] GNN OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"[GNN] Optuna optimization: {v5.format_duration(duration)}")
    print(f"[TRIALS] Completed: {len(study.trials)} trials")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(
        f"[TRIALS] Breakdown: {len(completed_trials)} completed, {len(pruned_trials)} pruned, {len(failed_trials)} failed"
    )

    if completed_trials:
        print("[BEST] Best hyperparameters found by Optuna:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print(f"[RESULT] Best validation RMSE: {study.best_value:.6f}")
    else:
        print("[WARNING] No trials completed successfully - cannot determine best hyperparameters")
    print("=" * 80)
    sys.stdout.flush()

    return study, {
        "input_size": input_size,
        "static_size": static_size,
        "output_size": output_size,
        "num_pks": num_pks,
        "cols_targets": cols_targets,
        "temporal_cols": temporal_cols,
        "static_features": static_features,
    }


def main():
    """Main execution pipeline - GNN-LSTM Only Training (V14: RMSE objective)."""

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    os.environ["LOCAL_RANK"] = "0"
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    experiment_name = build_experiment_name()
    output_dir = os.path.join(v5.OUTPUT_BASE_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("ABLATION STUDY V14 - GNN-LSTM WITH RMSE OPTIMIZATION OBJECTIVE")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Test Mode: {v5.TEST_MODE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"\n[NOTE] V14 uses pure val_rmse as Optuna objective (replaces MAE+R² composite from V7-V13)")
    print(f"[NOTE] Training config unchanged from V13: {v5.GNN_CONFIG['epochs']} epochs, patience={v5.GNN_CONFIG['patience']}")
    print("[NOTE] RMSE penalises 30% errors ~900x more than 1% errors (vs 30x for MAE).")
    print("[NOTE] Use v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR set to this output to continue.")
    print("=" * 80)
    sys.stdout.flush()

    v5.print_ablation_config()

    config = {
        "experiment_name": experiment_name,
        "model_time": model_time,
        "test_mode": v5.TEST_MODE,
        "ablation_settings": v5.ABLATION_SETTINGS,
        "gnn_config": v5.GNN_CONFIG,
        "train_dates": {"start": v5.TRAIN_START_DATE, "end": v5.TRAIN_END_DATE},
        "hyperparameter_search_space": v5.get_gnn_hyperparams(),
        "pipeline_version": "v14_gnn_rmse_objective",
        "goal_metric": {
            "name": "val_rmse",
            "description": "Pure RMSE — penalises large errors quadratically, no R² component",
        },
        "next_step": "Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR pointing to this output",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = os.path.join(output_dir, "experiment_config.json")
    config_versioned_path = os.path.join(output_dir, f"experiment_config_{PIPELINE_VERSION}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(config_versioned_path, "w") as f:
        json.dump(config, f, indent=2)

    total_start = time.time()
    stage_times = {}

    stage0_start = time.time()
    print("\n" + "=" * 80)
    print("STAGE 0: DATA LOADING")
    print("=" * 80)

    df_full, selected_features = v5.load_data()
    df_train = v5.filter_data_by_date(df_full, v5.TRAIN_START_DATE, v5.TRAIN_END_DATE)
    print(f"[INFO] Training data: {len(df_train):,} rows ({v5.TRAIN_START_DATE} to {v5.TRAIN_END_DATE})")
    sys.stdout.flush()

    del df_full
    gc.collect()

    stage_times["stage0_data_loading"] = time.time() - stage0_start
    v5.print_step_timing("STAGE 0: Data Loading Complete", stage0_start)

    stage1_start = time.time()
    gnn_study, _ = train_gnn_model(df_train, selected_features, output_dir, model_time)
    stage_times["stage1_gnn_training"] = time.time() - stage1_start
    v5.print_step_timing("STAGE 1: GNN-LSTM Training Complete", stage1_start)

    total_duration = time.time() - total_start

    print("\n" + "=" * 80)
    print("GNN-LSTM TRAINING COMPLETE (V14)")
    print("=" * 80)

    print("\nTIMING BREAKDOWN:")
    print("-" * 60)
    print(f"  Stage 0 - Data Loading:      {v5.format_duration(stage_times['stage0_data_loading']):>15}")
    print(f"  Stage 1 - GNN-LSTM Training: {v5.format_duration(stage_times['stage1_gnn_training']):>15}")
    print("-" * 60)
    print(f"  TOTAL:                       {v5.format_duration(total_duration):>15}")
    print("=" * 80)
    sys.stdout.flush()

    print(f"\nResults saved to: {output_dir}")

    summary = {
        "experiment_name": experiment_name,
        "total_duration_minutes": total_duration / 60,
        "total_duration_hours": total_duration / 3600,
        "stage_times_seconds": stage_times,
        "stage_times_formatted": {k: v5.format_duration(v) for k, v in stage_times.items()},
        "gnn_best_val_rmse": gnn_study.best_value if gnn_study.best_trial else None,
        "gnn_best_params": gnn_study.best_params if gnn_study.best_trial else None,
        "gnn_trials_completed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "gnn_trials_pruned": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "gnn_trials_failed": len(
            [t for t in gnn_study.trials if t.state == optuna.trial.TrialState.FAIL]
        ),
        "ablation_settings": v5.ABLATION_SETTINGS,
        "test_mode": v5.TEST_MODE,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "next_step": f"Run v5_xgboost_only with PREVIOUS_EXPERIMENT_DIR={output_dir}",
    }

    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    sys.stdout.flush()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study V14 - GNN-LSTM with RMSE Optimization Objective"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--time-resolution",
        choices=v5.TIME_RESOLUTION_CHOICES,
        help="Time resolution token to use",
    )
    for token in v5.TIME_RESOLUTION_CHOICES:
        minutes = token.replace("min", "")
        group.add_argument(
            f"--{minutes}min",
            dest="time_resolution",
            action="store_const",
            const=token,
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args()

    if getattr(args, "time_resolution", None):
        chosen = args.time_resolution
        new_data_file = re.sub(r"\d+min", chosen, v5.DATA_FILE, count=1)
        v5.DATA_FILE = new_data_file
        v5.DATA_PATH = os.path.join(_AP7_DATA, v5.DATA_FILE)
        print(f"[CONFIG] Using DATA_FILE={v5.DATA_FILE} for time resolution {chosen}")
        sys.stdout.flush()

    sys.exit(0 if main() else 1)
