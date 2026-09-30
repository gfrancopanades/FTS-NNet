# Synthetic dataset

`ap7_synthetic_5min.csv.gz` is a **synthetic** stand-in for the AP-7 corridor
data used in the paper. It has the same 39 columns, types, separator (`;`) and
row layout as the real 5-minute input, so every stage of the pipeline runs on
it unchanged. The real traffic, crash and weather records belong to the
Catalan Traffic Authority and are not redistributed (see the main README,
"Data availability").

**No value in this file comes from the real data.** Road geometry is drawn
from the summary statistics published in Table A.2 of the paper. Traffic,
weather forecasts and crashes come from simple parametric processes
(`src/data/make_synthetic_data.py`). Any model trained on this file learns
the synthetic generator, not the AP-7, so its scores say nothing about the
paper's results.

| | Committed file | Real data (paper) |
|---|---|---|
| Kilometre posts | 150–159 (10) | 120–219 (100) |
| Directions | 2 | 2 |
| Periods | Jun 2024, May–Jun 2025 | Apr 2024 – Sep 2025 |
| Segment-intervals | 524,160 | 26,323,200 (de-duplicated) |
| Crash-stamped intervals | 0.068 % | 0.0567 % |

The committed periods are those of the reference experiment: Stage 1 and
Stage 2 train on June 2024 and May 2025, and June 2025 is the held-out month.
Larger files, up to the full corridor and record, come from the generator:

```bash
python -m src.data.make_synthetic_data                      # default = committed file
python -m src.data.make_synthetic_data --pk-min 120 --pk-max 219 \
       --windows 2024-04-04:2025-10-01                      # full size, about 5 GB
```

Without `--out`, the generator writes straight to
`$AP7_DATA_DIR/CrashGNNLSTM_v1_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_20240404_to_20251001.csv`,
the filename every pipeline stage reads. `scripts/run_demo.sh` unpacks the
committed file to that path.

## Schema

One row per directional kilometre-post segment and 5-minute interval.

| Column | Type | Unit | Description |
|---|---|---|---|
| `dat` | str | local time | Interval start, `YYYY-MM-DD HH:MM:SS` |
| `via` | str | | Highway (`AP-7`) |
| `pk` | int | km | Kilometre post: lower bound of the 1-km segment |
| `sen` | str | | Direction: `dec` = south–north (coded 0), `cre` = north–south (coded 1), as in Table A.5 |
| `anyo`, `mes`, `dia` | int | | Year, month, day |
| `diaSem` | int | | Day of week, 0 = Monday |
| `hor`, `5min` | int | | Hour of day; minute of the hour (0, 5, …, 55) |
| `1d_fcst_*`, `3d_fcst_*` | float | | Weather forecast issued 1 or 3 days ahead, held constant within each hour: `temperature_2m` (°C), `precipitation` (mm), `snowfall` (cm), `cloud_cover` (%), `wind_speed_10m` and `wind_gusts_10m` (km/h) |
| `1d_fcst_rain_binary`, `3d_fcst_rain_binary` | int | | 1 if forecast precipitation ≥ 0.1 mm |
| `mean_speed` | int | km/h | Mean speed |
| `speed_imputation` | int | | 1 if `mean_speed` was imputed by linear interpolation |
| `car` | int | | Number of lanes |
| `intensity_imputation` | int | | 1 if the intensities were imputed |
| `intTot` | int | veh / 5 min | Total intensity |
| `intP` | int | veh / 5 min | Heavy-vehicle intensity |
| `ACCIDENT` | float | | 1 if the segment-interval lies inside a logged crash affectation |
| `C_NIVELL_AFECTACIO` | float | | Affectation level (0 = none, 1–5) |
| `F_TEMPS_AFECTACIO` | float | h | Affectation duration |
| `F_LONG_AFECTACIO` | float | km | Affectation length |
| `mob_esp` | float | | Special-mobility calendar flag (public holidays and holiday exit days) |
| `ang_curv` | float | deg | Trajectory curvature |
| `ang_pend_pos`, `ang_pend_neg` | float | deg | Positive and negative slope |
| `segment` | int | | Road-section identifier grouping consecutive posts |

The models use only the 3-day weather forecast; the 1-day columns exist in
the file but are dropped at load time.

## How the synthetic processes work

* **Traffic.** Demand follows weekday peaks (08:00 and 18:00, asymmetric by
  direction), a broad weekend profile, a summer uplift and the holiday
  calendar. Speed falls as intensity approaches lane capacity, and congestion
  spreads to neighbouring posts and persists over time. Heavy-vehicle share
  rises at night and falls at weekends.
* **Weather.** A hidden hourly series (seasonal and diurnal temperature, a
  two-state rain process, cloud and wind). The 1-day and 3-day forecasts are
  noisy copies of it; the 3-day copy is noisier and misses more rain events.
* **Crashes.** Onsets are rare events whose hazard rises with congestion,
  rain and curvature. Each onset stamps an affectation queue of 1–4 posts
  upstream for 15–90 minutes, sets `ACCIDENT` and the `*_AFECTACIO` fields
  there, and slows the traffic inside it. The label therefore describes the
  crash-affected state, as in the real affectation log (paper, Section 4.5.2).
