#!/bin/bash
#SBATCH --job-name=end2end
#SBATCH --output=log_output/output_end2end_%j.log
#SBATCH --error=log_error/error_end2end_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=72:00:00
# ============================================================================
# End-to-end baseline (no decomposition): train and frozen simulation, one job
# ============================================================================
#   bash scripts/run_benchmark_end2end.sh --v74
#
# --vNN selects src/training/ablation_study_vNN_end2end_only.py. The run id is
# the SLURM job id, else a timestamp. BENCH_TRAIN_WINDOWS / BENCH_SIM_START /
# BENCH_SIM_END set the windows of the Table B.1 months.
# ============================================================================
set -euo pipefail
ROOT="${AP7_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
export PROD_ENV_ROOT="$ROOT" AP7_PROJECT_ROOT="$ROOT" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${AP7_PYTHON:-python3}"
[ -n "${AP7_VENV:-}" ] && source "${AP7_VENV}/bin/activate"
[ -n "${AP7_CUDA_MODULE:-}" ] && module load "${AP7_CUDA_MODULE}"

VERSION=""
for arg in "$@"; do
    case "$arg" in
        --v[0-9]*) VERSION="${arg#--}" ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$VERSION" ] || { echo "[ERROR] give a version, e.g. --v74" >&2; exit 2; }

export ABLATION_VERSION="$VERSION"
export JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
echo "End-to-end $VERSION | run id $JOB_ID | $(date)"

SCRIPT="$(mktemp --suffix=.py)"; trap 'rm -f "$SCRIPT"' EXIT
cat > "$SCRIPT" <<'PY'
import importlib, os, sys
sys.path.insert(0, os.environ["PROD_ENV_ROOT"])
import src.training.ablation_study_v5_gnn_only as v5

v5.ABLATION_SETTINGS["include_weather_1d"] = False   # 3-day forecast only
ablation = importlib.import_module(
    f"src.training.ablation_study_{os.environ['ABLATION_VERSION']}_end2end_only")
sys.exit(ablation.main())
PY
"$PYTHON" -u "$SCRIPT"
echo "End-to-end $VERSION finished | $(date)"
