#!/bin/bash
#SBATCH --job-name=stage1
#SBATCH --output=log_output/output_stage1_%j.log
#SBATCH --error=log_error/error_stage1_%j.log
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=168G
#SBATCH --time=165:00:00
# ============================================================================
# Stage 1: train one traffic forecaster
# ============================================================================
#   bash scripts/run_ablation_study_gnn_only.sh --v62 [--run-id=ID]
#
# --vNN selects src/training/ablation_study_vNN_gnn_only.py. The run is
# written to $AP7_EXPERIMENTS_DIR/vNN_gnn_no-w1d_5min_<ID>, where ID is
# --run-id, else the SLURM job id, else a timestamp. Every run uses the
# 5-minute data and the 3-day weather forecast only.
# ============================================================================
set -euo pipefail
ROOT="${AP7_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
export PROD_ENV_ROOT="$ROOT" AP7_PROJECT_ROOT="$ROOT" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${AP7_PYTHON:-python3}"
[ -n "${AP7_VENV:-}" ] && source "${AP7_VENV}/bin/activate"
[ -n "${AP7_CUDA_MODULE:-}" ] && module load "${AP7_CUDA_MODULE}"

VERSION=""; RUN_ID=""
for arg in "$@"; do
    case "$arg" in
        --v[0-9]*)  VERSION="${arg#--}" ;;
        --run-id=*) RUN_ID="${arg#*=}" ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$VERSION" ] || { echo "[ERROR] give a version, e.g. --v62" >&2; exit 2; }

export ABLATION_VERSION="$VERSION"
export JOB_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export ABLATION_NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
echo "Stage 1 $VERSION | run id $JOB_ID | node ${SLURM_NODELIST:-local} | $(date)"

SCRIPT="$(mktemp --suffix=.py)"; trap 'rm -f "$SCRIPT"' EXIT
cat > "$SCRIPT" <<'PY'
import importlib, os, sys
sys.path.insert(0, os.environ["PROD_ENV_ROOT"])
import src.training.ablation_study_v5_gnn_only as base

ablation = importlib.import_module(
    f"src.training.ablation_study_{os.environ['ABLATION_VERSION']}_gnn_only")
base.ABLATION_SETTINGS["include_weather_1d"] = False   # 3-day forecast only
base.TRAIN_START_DATE, base.TRAIN_END_DATE = "2024-04-01", "2025-06-01"
ok = ablation.main()
sys.exit(0 if ok in (None, True, 0) else 1)
PY
"$PYTHON" -u "$SCRIPT"
echo "Stage 1 $VERSION finished | $(date)"
