# FTS-NNet: Forecast-Then-Supervise Neural Network

Code and synthetic data for

> Franco-Panadés, G., Llatje, Ò., Pagès, L., Jofre-Roca, L. **FTS-NNet:
> Forecast-Then-Supervise Neural Network, a high spatiotemporal resolution
> two-stage deep learning framework for multi-day operational highway
> crash-affected condition prediction.** Manuscript submitted for publication,
> 2026.

FTS-NNet issues a risk map of crash-affected conditions for a motorway corridor
72 hours ahead at 5-minute temporal and 1-km spatial resolution, using only
information available when the map is issued:

1. **Stage 1** forecasts the traffic state (mean speed, total and heavy-vehicle
   intensity) of every segment and interval from exogenous covariates:
   position and direction, calendar, holiday calendar, the 3-day weather
   forecast and road geometry.
2. **Stage 2** is a cost-sensitive classifier that recognises crash-affected
   conditions in the forecast traffic state, from 52 variables engineered on
   it.

The label of a segment-interval is its logged crash-affectation stamp
(`ACCIDENT`), used as recorded.

## Repository layout

```
src/
  paths.py          filesystem locations, overridable by AP7_* variables
  data/             pre-processing of the raw sources; synthetic-data generator
  models/           forecasters, classifiers, end-to-end models
  training/         Stage-1, Stage-2, simulation and end-to-end modules
  evaluation/       harvesting, bootstrap statistics, tables and figures
scripts/            runners: one per stage, demo, full reproduction, analysis
data/synthetic/     synthetic dataset with the study's schema (see its README)
experiments/        registry_final.json: every arm of the paper and its runs
```

## Installation

Python 3.10, with the versions used for the paper:

```bash
git clone https://github.com/gfrancopanades/FTS-NNet.git
cd FTS-NNet
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 11.7 build of PyTorch 2.0.1; on a
CPU-only machine install `torch==2.0.1` instead.

## Quick start on synthetic data

```bash
bash scripts/run_demo.sh          # or: sbatch scripts/run_demo.sh
```

This unpacks `data/synthetic/ap7_synthetic_5min.csv.gz` into `dades/` and runs
the proposed configuration end to end with small search budgets:

1. Stage 1, the GeoLSTM forecaster (BL-F1);
2. Stage 2, the focal-loss MLP (BL-C5);
3. the frozen 72-h simulation of the held-out month;
4. scoring (AUPRC, AUROC, lift, day-block bootstrap CI).

Results go to `experiments/demo_results/`. They describe the synthetic data
only.

## Data availability

The AP-7 traffic, crash-affectation and weather records were provided by the
Catalan Traffic Authority (Servei Català de Trànsit) and are not
redistributed. Researchers seeking access may contact the corresponding
author (gerard.franco@upc.edu); access can be arranged under the Catalan
Traffic Authority's data-sharing conditions.

`data/synthetic/` holds a synthetic dataset with the same variables and schema.
`src/data/make_synthetic_data.py` regenerates it, or builds larger versions:
the full reproduction needs the whole record, April 2024 to September 2025.

## Pre-processing

`src/data/initial_data_processing_5min_fund-propag.py` assembles the model
input from the raw operator files. It builds the 5-minute × kilometre-post ×
direction grid and merges onto it:

* speed, gaps filled by linear interpolation (`speed_imputation`);
* intensity, propagated from each loop detector to every kilometre post of its
  section with the fundamental diagram q = k·v, then interpolated
  (`intensity_imputation`);
* the crash-affectation log (`ACCIDENT`, `C_NIVELL_AFECTACIO`,
  `F_TEMPS_AFECTACIO`, `F_LONG_AFECTACIO`);
* the 1-day and 3-day weather forecasts;
* the special-mobility calendar and road geometry.

It reads `data/processed/` and `data/raw/` (file names in
`src/data_file_names.py`). Run it from the repository root:

```bash
python src/data/initial_data_processing_5min_fund-propag.py
```

Every stage reads the assembled input from
`$AP7_DATA_DIR/CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv`.

## Reproducing the paper

On a SLURM cluster, with the input file in `$AP7_DATA_DIR`:

```bash
bash scripts/reproduce_paper.sh all       # DRY_RUN=1 prints the jobs only
```

This submits every experiment in dependency order, with the paper's budgets
(30 Optuna trials per Stage-1 forecaster, 12 per rolling fold; 25 per Stage-2
classifier):

| Section | Experiments |
|---|---|
| `table2` | all arms of Table 2 and the full-year Stage 1 of Section 5.6 |
| `inert` | the graph-inert batching runs of Section 5.6 |
| `rolling` | the rolling-origin folds (Table 3) |
| `multimonth` | the May, July and August panels (Table B.1) |
| `cycle` | the 13-month frozen simulation (Table B.2, Fig. 4) |
| `horizon` | the lead-time sweep (Table B.4) |
| `seeds` | the replicate seeds and checks of Sections 5.2, 5.3(a) and 6.2 |

It writes `experiments/registry_reproduced.json` and, for Table B.1,
`experiments/registries/registry_<month>-2025_reproduced.json`. Site-specific
`sbatch` options go in `SBATCH_EXTRA`. Then build the tables and figures into
`reports/`:

```bash
cp experiments/registry_reproduced.json experiments/registry_final.json
bash scripts/make_tables_and_figures.sh
```

### One configuration

Each stage has a runner in `scripts/` that works under `sbatch` or `bash`.
The proposed configuration:

```bash
bash scripts/run_ablation_study_gnn_only.sh --v62 --run-id=S1                    # Stage 1
AP7_CLF_TRIALS=25 bash scripts/run_ablation_study_xgboost_only.sh --v73 \
     --gnn_v62 --gnn-run-id=S1 --xgb-run-id=S2                                  # Stage 2
bash scripts/run_ablation_study_simulation_only.sh --v73 \
     --gnn_v62 --gnn-run-id=S1 --xgb-run-id=S2                                  # simulation
```

`AP7_CLF_TRIALS` is the Stage-2 search budget; it defaults to 0 (no search), so
set it to 25 to match the paper.

## From the paper's arm IDs to the code

Each experiment is a module `src/training/ablation_study_vNN_<stage>_only.py`:
`gnn_only` is a Stage-1 forecaster of any architecture, `xgboost_only` a
Stage-2 classifier of any family, `simulation_only` the frozen simulation of a
Stage-1 + Stage-2 pair, and `end2end_only` a model without decomposition. The
lower-numbered modules (v5–v58) are the shared implementation the arms build
on. `experiments/registry_final.json` lists every arm (`paper_id`,
`reported_in`).

| Table 2 panel | Paper ID | Stage 1 | Stage 2 / model | Environment |
|---|---|---|---|---|
| (a) | BL-F1 GeoLSTM | v62 | v73 | |
| | BL-F0 Historical Average | v61 | v73 | |
| | BL-F2 DCRNN, BL-F3 STGCN, BL-F4 Graph WaveNet, BL-F5 ASTGCN | v104, v105, v106, v107 | v73 | |
| | BL-F7 GNN-LSTM | v86 | v73 | |
| (b) | BL-C0 … BL-C4 | v62 | v68 … v72 | |
| | BL-C5 MLP + Focal (proposed) | v62 | v73 | |
| | BL-C6 … BL-C11 | v62 | v76, v77, v79, v80, v81, v82 | |
| | BL-C12 … BL-C15 | v62 | v109 … v112 | |
| | BL-C16 GBDT Screener + GBDT Supervisor | v62 | v58 | |
| | BL-T1 … BL-T5 | v62 | v114 … v117, v120 | |
| (c) | BL-E7 / BL-E8, 72-h-lagged traffic | v86 (bypassed) | v73 / v76 | `BENCH_LAGGED_TRAFFIC=864` |
| | BL-E9 / BL-E10, covariates only | v86 (bypassed) | v85 / v84 | |
| | BL-E4, BL-E5, covariates only | | v89, v90 | `E2E_NO_TRAFFIC=1` |
| | BL-E6, 72-h-lagged traffic | | v91 | `E2E_LAG_TRAFFIC=864` |
| (d) | GeoLSTM + BL-C5 (proposed): the BL-C5 run of (b) | v62 | v73 | |
| | BL-E2: the BL-C16 run of (b), read as a complete pipeline | v62 | v58 | |
| | BL-E11, observed traffic | v86 (bypassed) | v73 | `BENCH_OBSERVED_TRAFFIC=1` |
| | BL-E0 Transformer, BL-E1 MSGNN, BL-E3 LSTM (observed traffic) | | v74, v75, v83 | |

Section 5.6 adds the full-year GeoLSTM (v78) and the graph-inert runs (v17,
v63–v66), each with v73 as Stage 2. In the registry, contrast `C1` is Table
2(a), `C2` is Table 2(b) plus BL-E7 to BL-E11, and `C3` the rows of 2(d) and the
end-to-end rows of 2(c).

The paper's `light_fraction` (Table A.3) is `hv_fraction` in the code, computed
as `(intTot - intP) / intTot` on the forecast channels.

## Configuration

| Variable | Default | Holds |
|---|---|---|
| `AP7_DATA_DIR` | `dades/` | input CSV |
| `AP7_EXPERIMENTS_DIR` | `experiments/` | model artefacts and simulations |
| `AP7_TABLES_DIR`, `AP7_FIGURES_DIR`, `AP7_RESULTS_DIR` | `reports/…` | tables, figures, result files |
| `AP7_PYTHON` | `python3` | interpreter used by the runners |
| `AP7_VENV`, `AP7_CUDA_MODULE` | unset | virtualenv to activate, module to load |

Experiment settings read from the environment: `AP7_CLF_TRIALS` and
`AP7_CLF_SEED` (Stage-2 search budget and seed); `BENCH_TRAIN_WINDOWS`,
`BENCH_SIM_START`, `BENCH_SIM_END` (rolling-fold windows); `BENCH_*_TRAFFIC`
and `E2E_*` (input mode of the no-decomposition and reference arms, see the
table). `AP7_GNN_TRIALS`, `AP7_GNN_EPOCHS` and `AP7_GNN_PATIENCE` shorten the
Stage-1 search for the demo; unset, they keep the paper's 30 / 150 / 40.

## Licence

MIT (`LICENSE`).
