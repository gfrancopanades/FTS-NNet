#!/bin/bash
# ============================================================================
# Re-run every experiment reported in the paper on a SLURM cluster
# ============================================================================
# Submits, in dependency order, the Stage-1 forecasters, the Stage-2
# classifiers trained on them, and the frozen 72-h simulation of the held-out
# month (June 2025) for every arm of Table 2, then the rolling-origin folds
# (Table 3), the multi-month panels (Table B.1), the 13-month frozen
# simulation (Table B.2, Fig. 4), the lead-time sweep (Table B.4) and the
# replicate seeds (Section 6.2). Budgets are the paper's: 30 Optuna trials per Stage-1
# forecaster (12 per rolling fold), 25 per Stage-2 classifier.
#
# Needs the real corridor data in $AP7_DATA_DIR (see README, "Data"). With the
# synthetic file every step runs, but the numbers describe the synthetic data.
#
# Output: experiments/jobmaps/reproduction_<timestamp>.csv (one row per arm)
# and experiments/registry_reproduced.json, a copy of registry_final.json with
# the new job ids, which the analysis scripts accept via --registry-json /
# AP7_REGISTRY_JSON.
#
# Usage (from the repository root):
#   bash scripts/reproduce_paper.sh [table2|inert|rolling|multimonth|cycle|horizon|seeds|all]
#   DRY_RUN=1 bash scripts/reproduce_paper.sh all     # print, submit nothing
#
# Site-specific sbatch options (partition, account, excluded nodes) go in
# SBATCH_EXTRA, e.g. SBATCH_EXTRA="-p gpu --account=myproj".
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WHAT="${1:-table2}"
X="${SBATCH_EXTRA:-}"
GNN_RES="--gres=gpu:4 --cpus-per-task=10 --mem=168G --time=165:00:00"
CLF_RES="--gres=gpu:1 --cpus-per-task=4 --mem=60G --time=20:00:00"
SIM_RES="--gres=gpu:1 --cpus-per-task=4 --mem=60G --time=08:00:00"
E2E_RES="--gres=gpu:1 --cpus-per-task=8 --mem=120G --time=72:00:00"
mkdir -p experiments/jobmaps log_output log_error
MAP="experiments/jobmaps/reproduction_$(date +%Y%m%d_%H%M%S).csv"
echo "key,paper_id,stage1_version,stage1_job,stage2_version,stage2_job,sim_job,env" > "$MAP"

submit() {  # sbatch args... -> prints the job id
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  sbatch $*" >&2; echo "DRY$RANDOM"; return
    fi
    sbatch --parsable $X "$@" | cut -d';' -f1
}
dep() { [ "${DRY_RUN:-0}" = "1" ] && echo "" || echo "--dependency=afterok:$1"; }

declare -A STAGE1   # stage-1 version -> job id (each forecaster trained once)
declare -A FOLD_S2  # month -> rolling-fold Stage-2 run (BL-C5 row of Table B.1)
stage1() {  # version -> sets S1_JOB; must not run in a subshell (it caches)
    local v=$1
    if [ -z "${STAGE1[$v]:-}" ]; then
        STAGE1[$v]=$(submit $GNN_RES -J "s1_$v" \
            scripts/run_ablation_study_gnn_only.sh --$v)
        echo "  stage 1 $v -> ${STAGE1[$v]}"
    fi
    S1_JOB="${STAGE1[$v]}"
}

cascade() {  # key paper_id stage1_version stage2_version [ENV=VAL,...]
    local key=$1 pid=$2 gv=$3 cv=$4 env=${5:-}
    local g t s ex="ALL,AP7_CLF_TRIALS=25${env:+,$env}"
    stage1 "$gv"; g=$S1_JOB
    t=$(submit $CLF_RES $(dep "$g") -J "s2_$key" --export="$ex" \
        scripts/run_ablation_study_xgboost_only.sh --$cv \
        --gnn_$gv --gnn-run-id="$g")
    s=$(submit $SIM_RES $(dep "$t") -J "sim_$key" --export="$ex" \
        scripts/run_ablation_study_simulation_only.sh --$cv \
        --gnn_$gv --gnn-run-id="$g" --xgb-run-id="$t")
    echo "$key,$pid,$gv,$g,$cv,$t,$s,$env" >> "$MAP"
    printf "  %-7s %-18s %s(%s) -> %s(%s) -> sim %s\n" "$pid" "$key" "$gv" "$g" "$cv" "$t" "$s"
}

end2end() {  # key paper_id version [ENV=VAL,...]
    local key=$1 pid=$2 v=$3 env=${4:-} j
    j=$(submit $E2E_RES -J "e2e_$key" --export="ALL${env:+,$env}" \
        scripts/run_benchmark_end2end.sh --$v)
    echo "$key,$pid,,,${v},$j,$j,$env" >> "$MAP"
    printf "  %-7s %-18s %s -> %s\n" "$pid" "$key" "$v" "$j"
}

table2() {
    echo "=== Table 2(a): Stage-1 forecaster swap, Stage 2 = BL-C5 (v73) ==="
    cascade C1_F1_GeoLSTM    BL-F1 v62  v73
    cascade C1_F0_HA         BL-F0 v61  v73
    cascade C1_F2b_DCRNN_MP  BL-F2 v104 v73
    cascade C1_H5_STGCN_MP   BL-F3 v105 v73
    cascade C1_F4b_GWNet_MP  BL-F4 v106 v73
    cascade C1_F5b_ASTGCN_MP BL-F5 v107 v73
    cascade C1_H4_MsgPass    BL-F7 v86  v73
    F7_STAGE2=$(tail -1 "$MAP" | cut -d, -f6)            # reused by cycle()
    cascade C1_FY_GeoLSTM    "Sec5.6" v78 v73        # full-year Stage 1

    echo "=== Table 2(b): Stage-2 classifier swap, Stage 1 = BL-F1 GeoLSTM (v62) ==="
    cascade C2_C5_Focal      BL-C5  v62 v73
    cascade C2_C0_LogReg     BL-C0  v62 v68
    cascade C2_C1_RF         BL-C1  v62 v69
    cascade C2_C2_SMOTE      BL-C2  v62 v70
    cascade C2_C3_Bagging    BL-C3  v62 v71
    cascade C2_C4_MLP2stage  BL-C4  v62 v72
    cascade C2_C6_GBDT1      BL-C6  v62 v76
    cascade C2_C7_GBDT2      BL-C7  v62 v77
    cascade C2_C8_MLPF2      BL-C8  v62 v79
    cascade C2_C9_GBDTMLP    BL-C9  v62 v80
    cascade C2_C10_HistRisk  BL-C10 v62 v81
    cascade C2_C11_LogRegPK  BL-C11 v62 v82
    cascade C2_C12_MLPGate   BL-C12 v62 v109
    cascade C2_C13_MLPCtx    BL-C13 v62 v110
    cascade C2_C14_MLPProx   BL-C14 v62 v111
    cascade C2_C15_MLPDevRes BL-C15 v62 v112
    cascade C2_proposed      BL-C16 v62 v58
    cascade C2_T1_GBDTProx   BL-T1  v62 v114
    cascade C2_T2_GBDTResid  BL-T2  v62 v115
    cascade C2_T3_GBDTMono   BL-T3  v62 v116
    cascade C2_T4_GBDTBlock  BL-T4  v62 v117
    cascade C2_T5_GBDTFocal  BL-T5  v62 v120

    echo "=== Table 2(c): no decomposition (72-h-available inputs only) ==="
    # The lagged/covariates-only classifier arms bypass the forecaster; the
    # Stage-1 run only anchors the experiment directory and the baselines.
    cascade C2_LAG_MLPF  BL-E7  v86 v73 BENCH_LAGGED_TRAFFIC=864
    cascade C2_LAG_GBDT  BL-E8  v86 v76 BENCH_LAGGED_TRAFFIC=864
    cascade C2_H2_GBDT   BL-E9  v86 v85
    cascade C2_H2_MLPF   BL-E10 v86 v84
    end2end C3_E4_LSTM_cov   BL-E4 v89 E2E_NO_TRAFFIC=1
    end2end C3_E6_LSTM_lag   BL-E6 v91 E2E_LAG_TRAFFIC=864
    end2end C3_E5_Transf_cov BL-E5 v90 E2E_NO_TRAFFIC=1

    echo "=== Table 2(d): complete systems ==="
    # The proposed row and BL-E2 reuse the BL-C5 and BL-C16 runs of 2(b).
    cascade H1_OBS_MLPF  BL-E11 v86 v73 BENCH_OBSERVED_TRAFFIC=1
    end2end C3_E0_Transformer BL-E0 v74
    end2end C3_E1_MSGNN       BL-E1 v75
    end2end C3_E3_LSTM        BL-E3 v83
}

rolling() {
    # Table 3. Stage 1 retrained per fold (v100-v103, 12 trials each); the
    # classifier and threshold are refitted on the two disjoint months that
    # precede the test month. June 2025 is the Table 2 reference run.
    echo "=== Table 3: rolling-origin folds ==="
    local M mon gv TW itr etr ts te g t s base dates
    for M in \
      "May-2025|v100|2024-05-01:2024-06-01;2025-04-01:2025-05-01|2024-05-01|2025-05-01|2025-05-01|2025-06-01" \
      "Jul-2025|v101|2024-07-01:2024-08-01;2025-06-01:2025-07-01|2024-07-01|2025-07-01|2025-07-01|2025-08-01" \
      "Aug-2025|v102|2024-08-01:2024-09-01;2025-07-01:2025-08-01|2024-08-01|2025-08-01|2025-08-01|2025-09-01" \
      "Sep-2025|v103|2024-09-01:2024-10-01;2025-08-01:2025-09-01|2024-09-01|2025-09-01|2025-09-01|2025-10-01"; do
        IFS="|" read -r mon gv TW itr etr ts te <<< "$M"
        # every flag in --flag=value form: the runners ignore space-separated values
        base="ALL,BENCH_TRAIN_WINDOWS=$TW,BENCH_SIM_START=$ts,BENCH_SIM_END=$te,AP7_CLF_TRIALS=25"
        dates="--ini-train=$itr --end-train=$etr --ini-test=$ts --end-test=$te"
        stage1 "$gv"; g=$S1_JOB
        t=$(submit $CLF_RES $(dep "$g") -J "fold_$mon" --export="$base" \
            scripts/run_ablation_study_xgboost_only.sh --v73 --gnn_$gv --gnn-run-id="$g" $dates)
        s=$(submit $SIM_RES $(dep "$t") -J "foldsim_$mon" --export="$base" \
            scripts/run_ablation_study_simulation_only.sh --v73 --gnn_$gv \
            --gnn-run-id="$g" --xgb-run-id="$t" $dates)
        echo "fold_$mon,BL-C5,$gv,$g,v73,$t,$s,BENCH_TRAIN_WINDOWS=$TW" >> "$MAP"
        FOLD_S2["${mon%%-*}"]="$t"                          # reused by multimonth()
        printf "  %-9s %s(%s) -> v73(%s) -> sim %s\n" "$mon" "$gv" "$g" "$t" "$s"
    done
}

inert() {
    # Section 5.6: the graph forecasters under the original location-major
    # batching, in which their graph layers never see a neighbour.
    echo "=== Section 5.6: graph-inert batching runs ==="
    cascade C1_proposed  "Sec5.6" v17 v73
    cascade C1_F2_DCRNN  "Sec5.6" v63 v73
    cascade C1_F3_STGCN  "Sec5.6" v64 v73
    cascade C1_F4_GWNet  "Sec5.6" v65 v73
    cascade C1_F5_ASTGCN "Sec5.6" v66 v73
}

multimonth() {
    # Table B.1: the classifier and end-to-end panels re-run for May, July and
    # August 2025 on the rolling-fold forecasters (June is Table 2). Writes one
    # registry per month for src.evaluation.benchmark_harvest.
    echo "=== Table B.1: multi-month panels ==="
    local M mon gv TW itr etr ts te base dates spec key ver env g t s reg
    for M in \
      "May|v100|2024-05-01:2024-06-01;2025-04-01:2025-05-01|2024-05-01|2025-05-01|2025-05-01|2025-06-01" \
      "Jul|v101|2024-07-01:2024-08-01;2025-06-01:2025-07-01|2024-07-01|2025-07-01|2025-07-01|2025-08-01" \
      "Aug|v102|2024-08-01:2024-09-01;2025-07-01:2025-08-01|2024-08-01|2025-08-01|2025-08-01|2025-09-01"; do
        IFS="|" read -r mon gv TW itr etr ts te <<< "$M"
        base="BENCH_TRAIN_WINDOWS=$TW,BENCH_SIM_START=$ts,BENCH_SIM_END=$te,AP7_CLF_TRIALS=25"
        dates="--ini-train=$itr --end-train=$etr --ini-test=$ts --end-test=$te"
        stage1 "$gv"; g=$S1_JOB
        reg="experiments/registries/registry_${mon}-2025_reproduced.json"; echo "[" > "$reg"
        # BL-C5 is the rolling-fold run of the month (Table 3); needs `rolling` in the same call
        [ -n "${FOLD_S2[$mon]:-}" ] && echo "  {\"key\": \"C2_C5_Focal\", \"contrast\": \"C2\", \"version\": \"v73\", \"kind\": \"benchmark\", \"gnn\": \"$g\", \"xgb\": \"${FOLD_S2[$mon]}\", \"label\": \"C2_C5_Focal\"}," >> "$reg"
        for spec in C2_C4_MLP2stage:v72: C2_C8_MLPF2:v79: C2_C11_LogRegPK:v82: \
                    C2_C0_LogReg:v68: C2_C10_HistRisk:v81: C2_LAG_MLPF:v73:BENCH_LAGGED_TRAFFIC=864; do
            IFS=: read -r key ver env <<< "$spec"
            t=$(submit $CLF_RES $(dep "$g") -J "mm_${mon}_$key" --export="ALL,$base${env:+,$env}" \
                scripts/run_ablation_study_xgboost_only.sh --$ver --gnn_$gv --gnn-run-id="$g" $dates)
            s=$(submit $SIM_RES $(dep "$t") -J "mmsim_${mon}_$key" --export="ALL,$base${env:+,$env}" \
                scripts/run_ablation_study_simulation_only.sh --$ver --gnn_$gv \
                --gnn-run-id="$g" --xgb-run-id="$t" $dates)
            echo "mm_${mon}_$key,B.1,$gv,$g,$ver,$t,$s,$env" >> "$MAP"
            echo "  {\"key\": \"$key\", \"contrast\": \"C2\", \"version\": \"$ver\", \"kind\": \"benchmark\", \"gnn\": \"$g\", \"xgb\": \"$t\", \"label\": \"$key\"}," >> "$reg"
        done
        for spec in C3_E4_LSTM_cov:v89:E2E_NO_TRAFFIC=1 C3_E5_Transf_cov:v90:E2E_NO_TRAFFIC=1; do
            IFS=: read -r key ver env <<< "$spec"
            t=$(submit $E2E_RES -J "mm_${mon}_$key" --export="ALL,$base,$env" \
                scripts/run_benchmark_end2end.sh --$ver)
            echo "mm_${mon}_$key,B.1,,,$ver,$t,$t,$env" >> "$MAP"
            echo "  {\"key\": \"$key\", \"contrast\": \"C3\", \"version\": \"$ver\", \"kind\": \"e2e\", \"run\": \"$t\", \"label\": \"$key\"}," >> "$reg"
        done
        sed -i '$ s/,$//' "$reg"; echo "]" >> "$reg"
        printf "  %-4s %s(%s) -> %s\n" "$mon" "$gv" "$g" "$reg"
    done
}

cycle() {
    # Table B.2 and Fig. 4: one Stage-1 + Stage-2 pair, trained once, simulated
    # frozen over Aug 2024 - Aug 2025. The published table used the BL-F7
    # (GNN-LSTM, v86) + BL-C5 pair of Table 2(a); reuses it when table2 ran in
    # the same call, otherwise trains it.
    echo "=== Table B.2 / Fig. 4: 13-month frozen simulation ==="
    local g t s
    stage1 v86; g=$S1_JOB
    t="${F7_STAGE2:-}"
    if [ -z "$t" ]; then
        t=$(submit $CLF_RES $(dep "$g") -J "s2_cycle" --export="ALL,AP7_CLF_TRIALS=25" \
            scripts/run_ablation_study_xgboost_only.sh --v73 --gnn_v86 --gnn-run-id="$g")
    fi
    s=$(submit $SIM_RES --time=16:00:00 $(dep "$t") -J "sim_cycle" \
        scripts/run_ablation_study_simulation_only.sh --v73 --gnn_v86 \
        --gnn-run-id="$g" --xgb-run-id="$t" --ini-test=2024-08-01 --end-test=2025-09-01)
    echo "cycle,B.2,v86,$g,v73,$t,$s," >> "$MAP"
    printf "  v86(%s) -> v73(%s) -> 13-month sim %s\n" "$g" "$t" "$s"
}

horizon() {
    # Table B.4: BL-E7 with the observation lagged by H hours (5-min bins).
    echo "=== Table B.4: lead-time sweep on the persistence baseline ==="
    local h
    for h in 1:12 3:36 8:96 24:288 48:576; do
        cascade "C2_LAG_MLPF_${h%%:*}h" "B.4" v86 v73 "BENCH_LAGGED_TRAFFIC=${h##*:}"
    done
}

seeds() {
    # Section 6.2: the proposed arm refitted under three further seeds, and
    # the extra arms of Sections 5.2 and 5.3(a).
    echo "=== Section 6.2 replicates and Section 5.2 / 5.3(a) checks ==="
    cascade E2_seed07  "Sec6.2"  v62 v73 AP7_CLF_SEED=7
    cascade E2_seed13  "Sec6.2"  v62 v73 AP7_CLF_SEED=13
    cascade E2_seed21  "Sec6.2"  v62 v73 AP7_CLF_SEED=21
    cascade E4_Union   "Sec5.2"  v62 v73 BENCH_UNION_TRAFFIC=1
    cascade E5_CovEng  "Sec5.3a" v86 v84 BENCH_COV_ENGINEERED=1
}

case "$WHAT" in
    table2)     table2 ;;
    inert)      inert ;;
    rolling)    rolling ;;
    multimonth) multimonth ;;
    cycle)      cycle ;;
    horizon)    horizon ;;
    seeds)      seeds ;;
    all)        table2; inert; rolling; multimonth; cycle; horizon; seeds ;;
    *) echo "usage: $0 [table2|inert|rolling|multimonth|cycle|horizon|seeds|all]" >&2; exit 2 ;;
esac

# Registry pointing every reported arm at the new runs
"${AP7_PYTHON:-python3}" - "$MAP" <<'EOF'
import csv, json, sys
rows = {r["key"]: r for r in csv.DictReader(open(sys.argv[1]))}
reg = json.load(open("experiments/registry_final.json"))
for e in reg:
    r = rows.get(e["key"])
    if not r:
        continue
    if e["kind"] == "e2e":
        e["run"] = r["stage2_job"]
    else:
        e["gnn"], e["xgb"] = r["stage1_job"], r["stage2_job"]
# Table 2(d) reads the Table 2(b) runs of BL-C5 and BL-C16
by = {e["key"]: e for e in reg}
for d, b in (("C3_proposed", "C2_C5_Focal"), ("C3_E2_GeoLSTM", "C2_proposed")):
    if b in rows:
        by[d]["gnn"], by[d]["xgb"] = by[b]["gnn"], by[b]["xgb"]
json.dump(reg, open("experiments/registry_reproduced.json", "w"), indent=2)
print(f"\njob map  -> {sys.argv[1]}\nregistry -> experiments/registry_reproduced.json")
# per-month registries of Table B.1: carry the paper's labels
import glob
labels = {e["key"]: e["label"] for e in reg}
for f in glob.glob("experiments/registries/registry_*-2025_reproduced.json"):
    month = json.load(open(f))
    for e in month:
        e["label"] = labels.get(e["key"], e["key"])
    json.dump(month, open(f, "w"), indent=2)
    print(f"registry -> {f}")
EOF
