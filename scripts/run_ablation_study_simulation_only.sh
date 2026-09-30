#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --output=log_output/output_simulation_%j.log
#SBATCH --error=log_error/error_simulation_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=08:00:00
# ============================================================================
# Frozen 72-h rolling simulation of a trained Stage-1 + Stage-2 pair
# ============================================================================
#   bash scripts/run_ablation_study_simulation_only.sh --v73 --gnn_v62 \
#        --gnn-run-id=S1 --xgb-run-id=S2 \
#        [--ini-train=YYYY-MM-DD --end-train=... --ini-test=... --end-test=...]
#
# --vNN selects src/training/ablation_study_vNN_simulation_only.py and must
# match the Stage-2 version. The dates are used by the rolling-origin folds
# and the 13-month simulation only.
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
[ -n "$VERSION" ] && [ -n "$GNN_VERSION" ] && [ -n "$GNN_RUN_ID" ] && [ -n "$XGB_RUN_ID" ] || {
    echo "[ERROR] --vNN, --gnn_vNN, --gnn-run-id=... and --xgb-run-id=... are required" >&2; exit 2; }

GNN_EXPERIMENT_DIR="$EXPERIMENTS_ROOT/${GNN_VERSION}_gnn_no-w1d_5min_${GNN_RUN_ID}"
XGBOOST_EXPERIMENT_DIR="$GNN_EXPERIMENT_DIR/xgboost_${XGB_RUN_ID}"
[ -f "$XGBOOST_EXPERIMENT_DIR/experiment_config_xgb_${VERSION}.json" ] || {
    echo "[ERROR] no Stage-2 run $VERSION at $XGBOOST_EXPERIMENT_DIR" >&2; exit 1; }

export SIM_VERSION="$VERSION" GNN_EXPERIMENT_DIR XGBOOST_EXPERIMENT_DIR JOB_ID="$XGB_RUN_ID"
export INI_TRAIN END_TRAIN INI_TEST END_TEST
echo "Simulation $VERSION of $XGBOOST_EXPERIMENT_DIR | $(date)"

SCRIPT="$(mktemp --suffix=.py)"; trap 'rm -f "$SCRIPT"' EXIT
cat > "$SCRIPT" <<'PY'
import importlib, json, os, sys
sys.path.insert(0, os.environ["PROD_ENV_ROOT"])
import src.training.ablation_study_v51_simulation_only as v51_sim
import src.training.ablation_study_v51_xgboost_only as v51

version = os.environ["SIM_VERSION"]
ablation = importlib.import_module(f"src.training.ablation_study_{version}_simulation_only")

# Settings and model timestamp recorded by the Stage-2 run
xgb_dir = os.environ["XGBOOST_EXPERIMENT_DIR"]
with open(os.path.join(xgb_dir, f"experiment_config_xgb_{version}.json")) as fh:
    config = json.load(fh)
ablation.ABLATION_SETTINGS.update(config.get("ablation_settings", {}))
if config.get("model_time"):
    os.environ["XGBOOST_MODEL_TIME"] = config["model_time"]
    ablation.XGBOOST_MODEL_TIME = config["model_time"]

ablation.SIMULATION_CONFIG["prediction_window_days"] = 7
ablation.SIMULATION_CONFIG["gnn_fine_tune_epochs"] = 30
ablation.SIMULATION_CONFIG["xgboost_fine_tune_rounds"] = 100
if not getattr(ablation, "_CUSTOM_SIM_DATES", False):
    ablation.SIM_START_DATE, ablation.SIM_END_DATE = "2025-06-01", "2025-10-01"

dates = {"TRAIN_START_DATE": "INI_TRAIN", "TRAIN_END_DATE": "END_TRAIN",
         "SIM_START_DATE": "INI_TEST", "SIM_END_DATE": "END_TEST",
         "_V17_ALIGNED_TRAIN_START": "INI_TRAIN", "_V17_ALIGNED_TRAIN_END": "END_TRAIN",
         "_V17_ALIGNED_SIM_START": "INI_TEST", "_V17_ALIGNED_SIM_END": "END_TEST"}
for mod in (ablation, v51_sim, v51):
    for attr, env in dates.items():
        if os.environ.get(env) and hasattr(mod, attr):
            setattr(mod, attr, os.environ[env])

ablation.run_simulation_only()
PY
"$PYTHON" -u "$SCRIPT"
echo "Simulation $VERSION finished | $(date)"
