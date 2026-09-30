#!/bin/bash
#SBATCH --job-name=stage2
#SBATCH --output=log_output/output_stage2_%j.log
#SBATCH --error=log_error/error_stage2_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=20:00:00
# ============================================================================
# Stage 2: train one crash classifier on a Stage-1 forecast
# ============================================================================
#   bash scripts/run_ablation_study_xgboost_only.sh --v73 --gnn_v62 \
#        --gnn-run-id=S1 [--xgb-run-id=S2] \
#        [--ini-train=YYYY-MM-DD --end-train=... --ini-test=... --end-test=...]
#
# --vNN selects src/training/ablation_study_vNN_xgboost_only.py (any model
# family), --gnn_vNN and --gnn-run-id the Stage-1 run it reads. The run is
# written to <Stage-1 dir>/xgboost_<S2>, where S2 is --xgb-run-id, else the
# SLURM job id, else a timestamp. The dates are used by the rolling-origin
# folds only. Set AP7_CLF_TRIALS=25 for the paper's search budget.
# ============================================================================
set -euo pipefail
ROOT="${AP7_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
export PROD_ENV_ROOT="$ROOT" AP7_PROJECT_ROOT="$ROOT" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${AP7_PYTHON:-python3}"
[ -n "${AP7_VENV:-}" ] && source "${AP7_VENV}/bin/activate"
[ -n "${AP7_CUDA_MODULE:-}" ] && module load "${AP7_CUDA_MODULE}"
EXPERIMENTS_ROOT="${AP7_EXPERIMENTS_DIR:-$ROOT/experiments}"
export AP7_EXPERIMENTS_DIR="$EXPERIMENTS_ROOT"

VERSION=""; GNN_VERSION=""; GNN_RUN_ID=""; XGB_RUN_ID=""
INI_TRAIN=""; END_TRAIN=""; INI_TEST=""; END_TEST=""
for arg in "$@"; do
    case "$arg" in
        --v[0-9]*)      VERSION="${arg#--}" ;;
        --gnn_v[0-9]*)  GNN_VERSION="${arg#--gnn_}" ;;
        --gnn-run-id=*) GNN_RUN_ID="${arg#*=}" ;;
        --xgb-run-id=*) XGB_RUN_ID="${arg#*=}" ;;
        --ini-train=*)  INI_TRAIN="${arg#*=}" ;;
        --end-train=*)  END_TRAIN="${arg#*=}" ;;
        --ini-test=*)   INI_TEST="${arg#*=}" ;;
        --end-test=*)   END_TEST="${arg#*=}" ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$VERSION" ] && [ -n "$GNN_VERSION" ] && [ -n "$GNN_RUN_ID" ] || {
    echo "[ERROR] --vNN, --gnn_vNN and --gnn-run-id=... are required" >&2; exit 2; }
XGB_RUN_ID="${XGB_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

PREVIOUS_EXPERIMENT_DIR="$EXPERIMENTS_ROOT/${GNN_VERSION}_gnn_no-w1d_5min_${GNN_RUN_ID}"
for f in "best-model_model=GNN-version=*.pt" "best-model-metadata_model=GNN-version=*.json" \
         "scaler-temporal_model=GNN-version=*.pkl" "scaler-static_model=GNN-version=*.pkl" \
         "scaler-targets_model=GNN-version=*.pkl"; do
    compgen -G "$PREVIOUS_EXPERIMENT_DIR/$f" >/dev/null || {
        echo "[ERROR] incomplete or missing Stage-1 run: $PREVIOUS_EXPERIMENT_DIR ($f)" >&2; exit 1; }
done

export ABLATION_VERSION="$VERSION" PREVIOUS_EXPERIMENT_DIR XGB_RUN_ID JOB_ID="$GNN_RUN_ID"
export INI_TRAIN END_TRAIN INI_TEST END_TEST
export ABLATION_NUM_GPUS=2
echo "Stage 2 $VERSION on $PREVIOUS_EXPERIMENT_DIR -> xgboost_$XGB_RUN_ID | $(date)"

SCRIPT="$(mktemp --suffix=.py)"; trap 'rm -f "$SCRIPT"' EXIT
cat > "$SCRIPT" <<'PY'
import importlib, os, sys
sys.path.insert(0, os.environ["PROD_ENV_ROOT"])
import src.training.ablation_study_v5_xgboost_only as base
import src.training.ablation_study_v51_xgboost_only as v51

ablation = importlib.import_module(
    f"src.training.ablation_study_{os.environ['ABLATION_VERSION']}_xgboost_only")
base.ABLATION_SETTINGS["include_weather_1d"] = False   # 3-day forecast only
if hasattr(ablation, "XGBOOST_CONFIG"):
    ablation.TRAIN_START_DATE, ablation.TRAIN_END_DATE = "2024-04-01", "2025-06-01"

# Rolling-origin folds: the dates must reach v51, whose globals the Stage-2
# feature preparation reads, not only the version module.
dates = {"TRAIN_START_DATE": "INI_TRAIN", "TRAIN_END_DATE": "END_TRAIN",
         "SIM_START_DATE": "INI_TEST", "SIM_END_DATE": "END_TEST",
         "_V17_ALIGNED_TRAIN_START": "INI_TRAIN", "_V17_ALIGNED_TRAIN_END": "END_TRAIN",
         "_V17_ALIGNED_SIM_START": "INI_TEST", "_V17_ALIGNED_SIM_END": "END_TEST"}
for mod in {ablation, base, v51}:
    for attr, env in dates.items():
        if os.environ.get(env) and hasattr(mod, attr):
            setattr(mod, attr, os.environ[env])

ablation.run_xgboost_only()
PY
"$PYTHON" -u "$SCRIPT"
echo "Stage 2 $VERSION finished | $(date)"
