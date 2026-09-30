#!/bin/bash
#SBATCH --job-name=ftsnnet_tables
#SBATCH --output=log_output/output_tables_%j.log
#SBATCH --error=log_error/error_tables_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# ============================================================================
# Harvest the runs in experiments/registry_final.json and regenerate every
# table and figure of the paper into reports/
# ============================================================================
# Run after scripts/reproduce_paper.sh has finished and its registry has been
# copied over experiments/registry_final.json.
#
# Table B.4 (lead-time sweep) is pinned rather than read from the registry:
# set the job ids in SWEEP of src/evaluation/export_new_result_tables.py.
# ============================================================================
set -u
ROOT="${AP7_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
export AP7_PROJECT_ROOT="$ROOT"
PY="${AP7_PYTHON:-python3}"
REG=experiments/registry_final.json
export AP7_REGISTRY_JSON="$REG"
rc=0
run() { echo; echo "=== $* ==="; "$PY" -u -m "$@" || { echo "  ** FAILED: $* **"; rc=1; }; }

echo "### Harvest: June 2025 benchmark (Table 2) and the Table B.1 months"
run src.evaluation.benchmark_harvest --registry-json "$REG" --n-boot 1000
for m in May:2025-05-01:2025-06-01 Jul:2025-07-01:2025-08-01 Aug:2025-08-01:2025-09-01; do
    IFS=: read -r mon s e <<< "$m"
    f="experiments/registries/registry_${mon}-2025_reproduced.json"
    run src.evaluation.benchmark_harvest --registry-json "$f" --eval-start "$s" \
        --eval-end "$e" --output-dir "experiments/benchmark_results/rolling_${mon}-2025"
done

echo "### Prediction store"
run src.evaluation.archive_benchmark_predictions
# 13-month frozen simulation (Table B.2, Fig. 4): the BL-F7 + BL-C5 pair
read -r CYC_GNN CYC_XGB < <("$PY" -c "
import json; e = {x['key']: x for x in json.load(open('$REG'))}['C1_H4_MsgPass']
print(e['gnn'], e['xgb'])")
CYC_DIR="v86_gnn_no-w1d_5min_$CYC_GNN"
AP7_CYCLE_GNN_DIR="${AP7_CYCLE_GNN_DIR:-$CYC_DIR}" AP7_CYCLE_XGB="${AP7_CYCLE_XGB:-$CYC_XGB}" \
    run src.evaluation.archive_cycle_predictions

echo "### Tables"
run src.evaluation.benchmark_forecast_metrics --registry-json "$REG" \
    --output-dir experiments/benchmark_results/fcst_reproduced          # Table 1 data
run src.evaluation.export_latex_tables          # Tables 1, 2, 4
run src.evaluation.export_rolling_origin_table  # Table 3
run src.evaluation.export_multimonth_table      # Table B.1
run src.evaluation.event_level_metrics          # Table B.3 data
run src.evaluation.export_new_result_tables     # Tables B.3, B.4
run src.evaluation.descriptive_stats            # Tables A.1, A.2, A.4, A.5

echo "### Figures"
run src.evaluation.plot_quadrant_figure         # Fig. 1
run src.evaluation.plot_benchmark_pr_curves     # Fig. 3
run src.evaluation.plot_results_figures         # Figs. 4-6, Table B.2
run src.evaluation.operational_analysis         # Fig. 7
run src.evaluation.shap_analysis                # Figs. 8-9, Table 5 (SHAP)
run src.evaluation.safety_insights              # Table 5 (permutation dAUPRC)
run src.evaluation.spatial_error_map            # Fig. C.1

echo
[ $rc -eq 0 ] && echo "ALL STEPS COMPLETED -> reports/" || echo "SOME STEPS FAILED (see above)"
exit $rc
