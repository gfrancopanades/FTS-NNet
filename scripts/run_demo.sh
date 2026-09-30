#!/bin/bash
#SBATCH --job-name=ftsnnet_demo
#SBATCH --output=log_output/output_demo_%j.log
#SBATCH --error=log_error/error_demo_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
# ============================================================================
# FTS-NNet end to end on the synthetic dataset
# ============================================================================
# Runs the proposed configuration of the paper on data/synthetic/:
#
#   Stage 1  BL-F1 GeoLSTM traffic forecaster        (v62)
#   Stage 2  BL-C5 focal-loss MLP crash classifier   (v73)
#   Frozen 72-h rolling simulation of the held-out month (June)
#   Scoring  AUPRC / AUROC / lift, day-block bootstrap CI
#
# The search budgets are cut to a few trials and epochs so the demo finishes
# on a CPU in well under an hour; the paper's budgets are 30 Stage-1 trials of
# up to 150 epochs and 25 Stage-2 trials (see scripts/reproduce_paper.sh).
# Numbers obtained here describe the synthetic data, not the AP-7.
#
# Usage (from anywhere):
#   bash scripts/run_demo.sh            # or: sbatch scripts/run_demo.sh
# ============================================================================
set -euo pipefail
ROOT="${AP7_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
export AP7_PROJECT_ROOT="$ROOT"
export AP7_DATA_DIR="${AP7_DATA_DIR:-$ROOT/dades}"
export AP7_EXPERIMENTS_DIR="${AP7_EXPERIMENTS_DIR:-$ROOT/experiments}"
export AP7_GNN_TRIALS="${AP7_GNN_TRIALS:-2}"
export AP7_GNN_EPOCHS="${AP7_GNN_EPOCHS:-4}"
export AP7_GNN_PATIENCE="${AP7_GNN_PATIENCE:-2}"
export AP7_CLF_TRIALS="${AP7_CLF_TRIALS:-2}"
PY="${AP7_PYTHON:-python3}"
RID="${DEMO_RUN_ID:-demo}"
mkdir -p "$AP7_DATA_DIR" "$AP7_EXPERIMENTS_DIR" log_output log_error

DATA_FILE="CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv"
if [ ! -f "$AP7_DATA_DIR/$DATA_FILE" ]; then
    echo "### unpacking the synthetic dataset -> $AP7_DATA_DIR/$DATA_FILE"
    gunzip -c data/synthetic/ap7_synthetic_5min.csv.gz > "$AP7_DATA_DIR/$DATA_FILE"
fi

echo "### Stage 1: BL-F1 GeoLSTM (run id ${RID})"
bash scripts/run_ablation_study_gnn_only.sh --v62 --run-id="${RID}"

echo "### Stage 2: BL-C5 MLP + focal loss"
bash scripts/run_ablation_study_xgboost_only.sh --v73 \
     --gnn_v62 --gnn-run-id="${RID}" --xgb-run-id="${RID}c"

echo "### Frozen 72-h simulation of the held-out month"
bash scripts/run_ablation_study_simulation_only.sh --v73 \
     --gnn_v62 --gnn-run-id="${RID}" --xgb-run-id="${RID}c"

echo "### Scoring with the paper's harvester (day-block bootstrap CI)"
REG="$AP7_EXPERIMENTS_DIR/demo_registry.json"
cat > "$REG" <<EOF
[{"key": "C2_C5_Focal", "contrast": "C2", "version": "v73", "kind": "benchmark",
  "label": "BL-C5 MLP + Focal (proposed), synthetic demo",
  "gnn": "${RID}", "xgb": "${RID}c", "is_proposed": true}]
EOF
"$PY" -m src.evaluation.benchmark_harvest --registry-json "$REG" \
     --n-boot "${DEMO_N_BOOT:-200}" --output-dir "$AP7_EXPERIMENTS_DIR/demo_results"
echo
echo "Results: $AP7_EXPERIMENTS_DIR/demo_results/benchmark_accident_3day_full.csv"
