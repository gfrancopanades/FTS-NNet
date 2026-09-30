"""Table A — traffic-forecasting benchmark (RMSE / MAE / MAPE per channel).

Evaluates every C1 forecaster *as a forecaster* on the SAME June-2025 window
used by the accident benchmark (Table B), so the two tables cross-reference
row-for-row. It reuses the exact v51 stage-0 protocol — same `load_data`, same
V17-aligned window filter, same `_v5_generate_gnn_predictions` inference the
cascade sims consumed — so the `*_gnn` values scored here are the very values
the crash classifier saw in Table B.

CPU-only (v51 disables CUDA at import time; the cascades ran inference on CPU
too). Run standalone or via bash_files/run_benchmark_forecast_metrics.sh:

    python src/evaluation/benchmark_forecast_metrics.py \
        [--output-dir experiments/benchmark_results/final_run]

Outputs: benchmark_traffic_tableA.csv + benchmark_traffic_tableA.md
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

# Same env the cascade bash wrapper exports for the v17-style (no-w1d) runs.
os.environ.setdefault("INCLUDE_WEATHER_1D", "False")

PROD_ROOT = _AP7_ROOT
if PROD_ROOT not in sys.path:
    sys.path.insert(0, PROD_ROOT)

from src.training import ablation_study_v51_xgboost_only as v51  # noqa: E402
from src.evaluation.benchmark_harvest import (  # noqa: E402
    REGISTRY, EXPERIMENTS_ROOT)

CHANNELS = ["mean_speed", "intTot", "intP"]
EVAL_START, EVAL_END = "2025-06-01", "2025-07-01"  # = Table B eval window


def resolve_gnn_dir(gnn_run_id):
    cands = sorted(glob.glob(os.path.join(EXPERIMENTS_ROOT, f"*_gnn_*_{gnn_run_id}")))
    if not cands:
        raise FileNotFoundError(f"no GNN experiment dir for run-id {gnn_run_id}")
    return cands[-1]


def load_base_df():
    """Replicate v51 stage 0A + 0A-bis exactly (same rows the sims saw)."""
    date_min = min(pd.Timestamp(v51.TRAIN_START_DATE), pd.Timestamp(v51.SIM_START_DATE))
    date_max = max(pd.Timestamp(v51.TRAIN_END_DATE), pd.Timestamp(v51.SIM_END_DATE))
    t0 = time.time()
    df, _ = v51.load_data(pk_min=v51.PK_MIN, pk_max=v51.PK_MAX,
                          date_min=date_min, date_max=date_max)
    print(f"[A] base CSV loaded in {time.time()-t0:.0f}s — {len(df):,} rows")
    df["dat"] = pd.to_datetime(df["dat"])
    keep = pd.Series(False, index=df.index)
    for w_start, w_end in v51.GNN_TRAIN_WINDOWS:
        keep |= (df["dat"] >= pd.Timestamp(w_start)) & (df["dat"] < pd.Timestamp(w_end))
    keep |= (df["dat"] >= pd.Timestamp(v51.SIM_START_DATE)) & \
            (df["dat"] < pd.Timestamp(v51.SIM_END_DATE))
    df = df[keep].sort_values(["pk", "sen", "dat"]).reset_index(drop=True)
    print(f"[A] window filter -> {len(df):,} rows")
    # same module patch stage0 applies before _v5 helpers are used
    v51._v5_mod.DATA_FILE = v51.DATA_FILE
    v51._v5_mod.DATA_PATH = str(v51.DATA_PATH)
    return df


def channel_metrics(df_eval):
    out = {}
    for ch in CHANNELS:
        real = df_eval[ch].to_numpy(dtype=float)
        pred = df_eval[f"{ch}_gnn"].to_numpy(dtype=float)
        m = np.isfinite(real) & np.isfinite(pred)
        r, p = real[m], pred[m]
        err = p - r
        nz = np.abs(r) > 1e-9
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        out[ch] = dict(
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mae=float(np.mean(np.abs(err))),
            r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            bias=float(np.mean(err)),
            mape_nz=float(np.mean(np.abs(err[nz]) / np.abs(r[nz])) * 100.0),
            real_std=float(np.std(r)),
            n=int(m.sum()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",
                    default="experiments/benchmark_results/final_run")
    ap.add_argument("--registry-json", default=None,
                    help="Optional JSON list of registry entries replacing the "
                         "built-in REGISTRY, so forecasters trained after this "
                         "module was written still reach Table 1. Only `gnn` is "
                         "needed here -- a forecaster earns a Table 1 row without "
                         "any Stage-2 classifier or simulation behind it.")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    registry = REGISTRY
    if args.registry_json:
        import json
        with open(args.registry_json) as _f:
            registry = json.load(_f)
        print(f"[REGISTRY] Overridden from {args.registry_json}: {len(registry)} entries")

    # Every C1 row is a Stage-1 forecaster. `kind` only tells the harvester which
    # simulation subdirectory to glob, which is irrelevant to forecast error, so
    # filtering on it here silently drops arms (it dropped the adopted Stage-1).
    entries = [e for e in registry
               if e["contrast"] == "C1" and e.get("gnn")]
    if not entries:
        raise SystemExit("[FAIL] no C1 forecasters in the registry -- nothing to do")
    df_base = load_base_df()
    interval = v51._v5_get_time_resolution_minutes()
    seq_len = v51.SEQUENCE_LENGTH_BY_INTERVAL.get(interval, 8)
    print(f"[A] interval {interval}min -> sequence_length {seq_len}")

    rows = []
    for e in entries:
        gnn_dir = resolve_gnn_dir(e["gnn"])
        print("=" * 70)
        print(f"[A] {e['key']:16s} {e['label']:32s} gnn={e['gnn']} -> {os.path.basename(gnn_dir)}")
        model_time = v51._v5_detect_model_time(gnn_dir)
        (model, meta, _info, sc_t, sc_s, sc_y) = v51._v5_load_pretrained_gnn_model(
            gnn_dir, model_time)
        t0 = time.time()
        df = v51._v5_generate_gnn_predictions(
            model, df_base.copy(), sc_t, sc_s, sc_y, meta, sequence_length=seq_len,
            experiment_dir=gnn_dir)
        df = df.dropna(subset=["mean_speed_gnn"])
        ev = df[(df["dat"] >= pd.Timestamp(EVAL_START)) &
                (df["dat"] < pd.Timestamp(EVAL_END))]
        print(f"[A]   inference {time.time()-t0:.0f}s — eval rows {len(ev):,}")
        met = channel_metrics(ev)
        for ch in CHANNELS:
            rows.append(dict(key=e["key"], label=e["label"], version=e["version"],
                             architecture=meta.get("architecture") or "gnn_lstm_v5",
                             gnn=e["gnn"], channel=ch, **met[ch]))
        del df, ev, model

    out = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "benchmark_traffic_tableA.csv")
    out.to_csv(csv_path, index=False)

    md = ["# Table A — traffic forecasting benchmark (June 2025, same rows as Table B)",
          "",
          f"_Generated {time.strftime('%Y-%m-%d %H:%M')}. Eval window {EVAL_START} → {EVAL_END}; "
          "predictions are the identical `*_gnn` values consumed by the Table-B crash classifier "
          "(v51 stage-0 protocol, 1-step-ahead, covariate windows). MAPE over non-zero targets "
          "only. `real_std` = std of the real channel (scale reference)._", ""]
    for ch in CHANNELS:
        sub = out[out.channel == ch].sort_values("rmse")
        md += [f"## {ch}", "",
               "| Model | Ver | RMSE | MAE | R² | MAPE% (nz) | Bias | real_std | n |",
               "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            bold = "**" if "Proposed" in r["label"] else ""
            md.append(f"| {bold}{r['label']}{bold} | {r['version']} | {r['rmse']:.3f} "
                      f"| {r['mae']:.3f} | {r['r2']:.3f} | {r['mape_nz']:.1f} | {r['bias']:+.3f} "
                      f"| {r['real_std']:.3f} | {r['n']:,} |")
        md.append("")
    md_path = os.path.join(args.output_dir, "benchmark_traffic_tableA.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print(f"[DONE] {csv_path}\n       {md_path}")


if __name__ == "__main__":
    main()
