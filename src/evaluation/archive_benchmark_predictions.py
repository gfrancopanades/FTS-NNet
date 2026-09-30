"""Consolidate every benchmark model's per-row predictions into a stable store.

Why: the raw simulation CSVs (~150 MB each, 11k+ of them) live inside per-run
experiment folders and are discovered by "latest glob", so a re-run silently
shadows an earlier one. This script resolves each registry entry ONCE, copies the
prediction columns into a compact, explicitly-pinned Parquet store, and writes
a manifest recording exactly which job/file each row came from.

What is stored per model (full simulated range, not just June, so any month or
sub-slice can be re-analysed later):
    pk, sen, dat, y, prob, pred_binary
The label `y` is stored exactly as the simulation wrote it.

Outputs
    experiments/benchmark_results/predictions/<key>.parquet
    experiments/benchmark_results/predictions/manifest.json

Run:
    sbatch --mem=90G --cpus-per-task=4 -t 4:00:00 \
      -J pred_archive -o log_output/output_pred_archive_%j.log \
      -e log_error/error_pred_archive_%j.log \
      --wrap="cd <root> && export PYTHONPATH=<root> && \
              python -u src/evaluation/archive_benchmark_predictions.py"
"""
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

PROD_ROOT = _AP7_ROOT
if PROD_ROOT not in sys.path:
    sys.path.insert(0, PROD_ROOT)

from src.evaluation import benchmark_harvest as bh  # noqa: E402

OUT_DIR = os.path.join(PROD_ROOT, "experiments/benchmark_results/predictions")
KEEP = ["pk", "sen", "dat", "y", "prob", "pred_binary"]


def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    prob = cols.get(bh.SCORE_COL.lower()) or next(
        c for c in df.columns if "prob" in c.lower())
    ytrue = cols.get(bh.TRUE_COL.lower()) or next(
        c for c in df.columns if "real" in c.lower())
    pred = cols.get(bh.PRED_COL.lower())
    # cascade sims use pk/sen; the end-to-end sims use pk_id/sen_id
    def _col(*names):
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        return pd.Series(-1, index=df.index, dtype="float64")

    out = pd.DataFrame({
        "pk": _col("pk", "pk_id").fillna(-1).astype(np.int16),
        "sen": _col("sen", "sen_id").fillna(-1).astype(np.int8),
        "dat": pd.to_datetime(df["dat"]),
        "y": pd.to_numeric(df[ytrue], errors="coerce").fillna(0).astype(np.int8),
        "prob": pd.to_numeric(df[prob], errors="coerce").astype(np.float32),
    })
    out["pred_binary"] = (pd.to_numeric(df[pred], errors="coerce").fillna(0).astype(np.int8)
                          if pred else np.int8(0))
    out = out.sort_values(["pk", "sen", "dat"]).reset_index(drop=True)
    return out[KEEP]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest, ok, missing = [], 0, []

    # The registry moves whenever an arm is retrained, so honour the same
    # override the harvest uses -- otherwise the prediction store silently
    # keeps archiving the superseded runs the tables are meant to leave behind.
    registry = bh.REGISTRY
    _reg = os.environ.get("AP7_REGISTRY_JSON")
    if _reg:
        with open(_reg) as _f:
            registry = json.load(_f)
        print(f"[REGISTRY] Overridden from {_reg}: {len(registry)} entries")

    for e in registry:
        key = e["key"]
        csv = bh.discover_sim_csv(e, bh.EXPERIMENTS_ROOT, "two_stage")
        if not csv:
            missing.append(key)
            print(f"[MISS] {key:22s} no sim CSV")
            continue
        try:
            df = load_predictions(csv)
        except Exception as exc:                      # noqa: BLE001
            missing.append(key)
            print(f"[FAIL] {key:22s} {type(exc).__name__}: {exc}")
            continue

        path = os.path.join(OUT_DIR, f"{key}.parquet")
        df.to_parquet(path, index=False, compression="snappy")
        jun = df[(df.dat >= "2025-06-01") & (df.dat < "2025-07-01")]
        rec = dict(
            key=key, label=e["label"], version=e.get("version"),
            contrast=e.get("contrast"), kind=e.get("kind"),
            gnn_job=e.get("gnn"), xgb_job=e.get("xgb"), e2e_job=e.get("run"),
            source_csv=csv,
            source_md5=hashlib.md5(csv.encode()).hexdigest()[:12],
            parquet=os.path.relpath(path, PROD_ROOT),
            n_rows_total=int(len(df)),
            date_min=str(df.dat.min()), date_max=str(df.dat.max()),
            june_rows=int(len(jun)),
            june_pos=int(jun.y.sum()),
            size_mb=round(os.path.getsize(path) / 1e6, 1),
        )
        manifest.append(rec)
        ok += 1
        print(f"[OK]   {key:22s} n={rec['n_rows_total']:>9,} june={rec['june_rows']:>9,} "
              f"pos={rec['june_pos']:>5,} {rec['size_mb']:>5.1f}MB")

    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump({"n_models": ok, "missing": missing, "models": manifest}, fh, indent=2)
    print(f"\n[DONE] archived {ok} models -> {OUT_DIR}")
    if missing:
        print(f"[WARN] missing/failed: {missing}")
    print(f"[DONE] manifest -> {mpath}")


if __name__ == "__main__":
    main()
