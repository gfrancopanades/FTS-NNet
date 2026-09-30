import pandas as pd
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
# Add parent directory to path to access data_file_names.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_file_names import *

def check_and_create_folder(path):
    # Check if the path exists
    if not os.path.exists(path):
        # Create the folder if it doesn't exist
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")


def get_first_decimal(column):
    """
    Extracts the first decimal digit from a numeric column.
    
    Parameters:
        column (pd.Series): The numeric column from which to extract the first decimal.
    
    Returns:
        pd.Series: A new column with the first decimal digit extracted.
    """
    return ((column * 10) % 10).astype(int)

def processing_geo_vies(folder_dades,file_geo_vies):
    """
    -------- PLOTTING THE HISTOGRAM TO DECIDE THE CURVE CATEGORY THRESHOLD -------
    import matplotlib.pyplot as plt

    # Select a column, for example 'mean_speed'
    column_name = 'radius'

    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Drop NaN values before plotting
        data = df[column_name][df[column_name]<10000][df[column_name]>-10000].dropna()
        
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=1000, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
    """
    print(f"> Importing road geometry BDD...")
    start_time = time.time()

    df = pd.read_csv(os.path.join(folder_dades, file_geo_vies), sep=",", decimal=".", encoding='latin-1')

    # Melt the DataFrame to restructure slope data
    df = df.melt(id_vars=['via', 'pk', 'x', 'y', 'Z', 'rad'], value_vars=['mAsc', 'mDes'], var_name='sen', value_name='slope')

    df['ang_curv'] = np.degrees(np.arctan(100 / df['rad'])).round(decimals=2).abs()
    df['ang_pend'] = np.degrees(np.arcsin(df['slope'] / 100)).round(decimals=2)
    df['ang_pend_pos'] = 0
    df['ang_pend_neg'] = 0
    df['ang_pend_pos'] = df.loc[df['ang_pend']>=0,'ang_pend']
    df['ang_pend_neg'] = df.loc[df['ang_pend']<0,'ang_pend']


    # Store original curvature and slope values
    df['orig_radius'] = df['rad']
    df['orig_slope'] = df['slope']

    # Add a segment number (0-9) to track original positions within each PK
    df['segment'] = get_first_decimal(df['pk'])

    # Modify 'sen' to categorical names
    df['sen'] = df['sen'].replace({'mAsc': 'cre', 'mDes': 'dec'})

    # Round PK values for aggregation
    df['pk_rounded'] = df['pk'].astype(float).round(0).astype(int)

    # Compute curvature cumulative measure
    df['curve_cum'] = (2 * math.pi) / df['rad'].abs()

    # Initialize curve categories
    df['curve_light'] = 0
    df['curve_regular'] = 0
    df['curve_heavy'] = 0

    # Assign curvature categories
    df.loc[df['orig_radius'].abs() <= 300, 'curve_heavy'] = 1
    df.loc[(df['orig_radius'].abs() > 300) & (df['orig_radius'].abs() <= 1300), 'curve_regular'] = 1
    df.loc[(df['orig_radius'].abs() > 1300), 'curve_light'] = 1

    # Assign slope values for aggregation
    df['slope_rise'] = df['slope']
    df['slope_drop'] = df['slope']
    df['slope_cum'] = df['slope'].abs()


    # Aggregate per PK while keeping original values in lists
    df_agg = df.groupby(['via', 'pk_rounded', 'sen']).agg({
        'slope_cum': 'sum',
        'slope_rise': 'max',
        'slope_drop': 'min',
        'curve_light': 'sum',
        'curve_regular': 'sum',
        'curve_heavy': 'sum',
        'curve_cum': 'sum',
        'ang_curv': 'sum',
        'ang_pend_pos': 'sum',
        'ang_pend_neg': 'sum',
        'orig_radius': lambda x: list(x),  # Store list of 10 values
        'orig_slope': lambda x: list(x)
    }).reset_index()

    # Count the number of original segments per PK (each PK should have 10)
    df_num = df.groupby(['via', 'pk_rounded', 'sen'], as_index=False)['segment'].nunique()
    # Merge this count back into df_agg
    df_agg = df_agg.merge(df_num[['via', 'pk_rounded', 'sen', 'segment']], on=['via', 'pk_rounded', 'sen'], how='left')

    # Divide curvature values by the number of original segments
    df_agg[['curve_light', 'curve_regular', 'curve_heavy']] = df_agg[['curve_light', 'curve_regular', 'curve_heavy']].div(df_agg['segment'], axis=0).round(decimals=2)

    # Convert slopes to absolute values
    df_agg['slope_drop'] = df_agg['slope_drop'].abs()
    df_agg['slope_rise'] = df_agg['slope_rise'].abs()
    df_agg['slope_cum'] = df_agg['slope_cum'].abs()

    # Ensure lists have exactly 10 elements per PK (pad with NaN if needed)
    df_agg['orig_radius'] = df_agg['orig_radius'].apply(lambda x: x[:10] if len(x) >= 10 else x + [float('nan')] * (10 - len(x)))
    df_agg['orig_slope'] = df_agg['orig_slope'].apply(lambda x: x[:10] if len(x) >= 10 else x + [float('nan')] * (10 - len(x)))

    # Convert lists to separate columns (pivoting)
    df_pivot = df_agg[['via', 'pk_rounded', 'sen']].copy()
    for i in range(10):
        df_pivot[f'orig_radius_{i}'] = df_agg['orig_radius'].apply(lambda x: x[i])
        df_pivot[f'orig_slope_{i}'] = df_agg['orig_slope'].apply(lambda x: x[i])

    # Merge with aggregated features
    df_final = df_agg.drop(columns=['orig_radius', 'orig_slope']).merge(df_pivot, on=['via', 'pk_rounded', 'sen'], how='left')

    # Rename PK column and return the processed DataFrame
    df_final = df_final.rename(columns={'pk_rounded': 'pk'})

    df_final = df_final.fillna(0)

    return df_final

def processing_cal_mob(folder_dades,file_cal_mob):
    print(f"> Importing special mobility calendar BDD...")
    start_time = time.time()

    df = pd.read_csv(os.path.join(folder_dades,file_cal_mob), sep=",", decimal=".", encoding='latin-1')
    
    df['dat']=pd.to_datetime(df['dat'])
    df['Any']=df.dat.dt.year
    df['mes']=df.dat.dt.month
    df['dia']=df.dat.dt.day
    df=df[['Any','mes','dia','senOpe','pont']].copy()
    df['mob_esp']=1
    df['senOpe']=df['senOpe'].fillna('ambdos')
    df = pd.get_dummies(df, columns=['senOpe', 'pont'])
    df = df.astype(int)
    df.drop_duplicates(inplace=True)

    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")

    return df


# ==============================================================================
# ACCIDENT PRECURSOR LABELLING
# ==============================================================================
#
# The source retentions/accidents database flags ACCIDENT=1 for the DURATION of
# the retention that follows a crash — i.e. the label the model has always been
# trained on is "is there currently a retention caused by an accident", not
# "is an accident about to happen". `analyze_accident_speed_mismatch.py`
# (2026-03-24, job acc_speed_mismatch) established, on the real 5-min dataset,
# that this is detectable in advance: of 8,781 highway-speed accident events,
# 72.0% show mean_speed dropping below 70% of its local baseline BEFORE the
# ACCIDENT flag activates (median lead 20 min, 81.7% negative pre-flag slope).
# That result is the empirical basis for the window-picking method below.
#
# `tag_accident_precursor_window` re-derives that same per-location event
# structure and, instead of just measuring the lead time, USES it to place a
# 30-minute (configurable) precursor tag: it walks backward from each
# accident's onset, finds where the speed anomaly actually starts (rather than
# blindly using the 30 minutes immediately before onset), and tags that window.
# Events with no detectable speed anomaly (already-congested baseline, or a
# sudden onset with no gradual precursor) fall back to the naive
# "last N minutes before onset" window, so every accident still gets tagged.
#
# Designed to run on the FINAL merged frame (same columns
# `analyze_accident_speed_mismatch.py` already validates against: via, pk, sen,
# dat, mean_speed, ACCIDENT), so it is usable both inside a fresh ETL run
# (`initial_data_processing_5min_fund-propag_precursor30min.py`) and as a fast
# relabelling pass over an already-built dataset
# (`relabel_accident_precursor_30min.py`) — the two callers stay identical by
# construction because they share this one function.
# ==============================================================================

def _close_short_gaps(acc, max_gap_steps):
    """Bridge runs of 0s no longer than `max_gap_steps` that sit between two
    1-runs, so run-length episode detection sees one merged block instead of
    fragments. Gaps at the series' start/end (not bounded by a 1 on both
    sides) are left alone -- there is nothing to bridge them to.

    `zero_starts` (1->0 transitions) has no entry for a LEADING gap (nothing
    precedes it); `zero_ends` (0->1 transitions) has no entry for a TRAILING
    gap (nothing follows it). Both are dropped before pairing so `zip` lines
    up entries belonging to the same interior gap, not adjacent-but-unrelated
    ones -- caught by a synthetic leading/trailing/interior-gap test before
    this ever touched real data.
    """
    if max_gap_steps <= 0:
        return acc.copy()
    acc = acc.astype(np.int8).copy()
    padded = np.concatenate(([0], acc, [0]))
    diff = np.diff(padded)
    zero_starts = np.where(diff == -1.0)[0]   # 1 -> 0 transition, gap begins here
    zero_ends = np.where(diff == 1.0)[0] - 1  # 0 -> 1 transition, gap ends here (inclusive)
    if len(acc) and acc[0] == 0 and len(zero_ends):
        zero_ends = zero_ends[1:]     # leading gap's end has no matching start
    if len(acc) and acc[-1] == 0 and len(zero_starts):
        zero_starts = zero_starts[:-1]  # trailing gap's start has no matching end
    for zs, ze in zip(zero_starts, zero_ends):
        if (ze - zs + 1) <= max_gap_steps:
            acc[zs:ze + 1] = 1
    return acc


def tag_accident_precursor_window(
    df,
    time_res_min: int = 5,
    lookback_minutes: int = 60,
    tag_window_minutes: int = 30,
    speed_drop_ratio: float = 0.70,
    min_highway_speed: float = 60.0,
    min_persistence_steps: int = 2,
    min_baseline_steps: int = 4,
    merge_gap_minutes: int = 20,
    anchor_before_drop: bool = True,
    adaptive_only: bool = False,
    use_multisignal: bool = False,
    cv_spike_mult: float = 2.0,
    grad_drop_kmh: float = 20.0,
    occ_spike_mult: float = 1.5,
    loc_cols=("via", "pk", "sen"),
    accident_col: str = "ACCIDENT",
    speed_col: str = "mean_speed",
    time_col: str = "dat",
    new_accident_col: str = "ACCIDENT",
    orig_col: str = "ACCIDENT_RETENTION_ORIG",
):
    """Re-tag ACCIDENT from "retention in progress" to "precursor window".

    For every contiguous ACCIDENT==1 block (a "retention episode") at a given
    (via, pk, sen) location, walk backward from the block's first row (the
    retention's onset -- the best available proxy for when the accident became
    disruptive) and pick a `tag_window_minutes` window that becomes the NEW
    ACCIDENT=1 label. Two ways that window gets picked:

      * ADAPTIVE (preferred): scan up to `lookback_minutes` before onset for the
        first point sustained for `min_persistence_steps` consecutive readings
        where `mean_speed` drops below `speed_drop_ratio` x baseline_speed,
        where baseline_speed is the mean of the earlier half of the lookback
        window (same estimator `analyze_accident_speed_mismatch.py`
        validated -- that script used a single-step, i.e. persistence=1,
        crossing test). The tag window starts there and runs
        `tag_window_minutes` forward, clipped so it never reaches the
        retention onset itself.

        `min_persistence_steps` defaults to 2 (10 min sustained), not 1: a
        unit test (`verify_precursor_tag.py`, case 6) constructed a series
        with one noisy single-step dip sitting well before the real ramp-down
        and confirmed persistence=1 latches onto that blip as the "onset" --
        an artificially early, unrepresentative window -- while persistence=2
        correctly skips it and lands on the true sustained drop. This is a
        deliberate departure from the validated report's exact methodology,
        made for label quality; pass `min_persistence_steps=1` to reproduce
        that report's numbers exactly.
      * FALLBACK (when adaptive finds nothing -- baseline already congested,
        i.e. <= `min_highway_speed`, or no threshold crossing in the lookback):
        the literal last `tag_window_minutes` before onset, clipped to
        available history.

    In both cases the window is also clipped so it never overlaps the tail of a
    PRECEDING episode at the same location -- otherwise recovery from a prior
    incident could get mislabelled as the precursor to the next one.

    BEFORE any of that: episodes separated by a gap of <= `merge_gap_minutes`
    at the SAME location are merged into one episode. This is not an edge
    case -- direct inspection of pk=120/cre, 2024-06-20 08:30-10:30 (speed
    122 -> 92 -> 71 -> 62 -> 25 -> 15 km/h at 08:55-09:10, then pinned at
    7-24 km/h until 10:30) showed the raw ACCIDENT flag pulses on for single
    5-min ticks at 09:15, 09:35 and 10:20 while the disrupted minutes between
    them are flagged 0 -- fragments of the same underlying disruption, not
    independent events. Unmerged, a fragment's own "precursor" window is
    spurious -- it's really mid-incident, and gets truncated almost to nothing
    by the previous-episode clipping above.

    The default of 20 min is evidence-based, not a round-number guess: a
    histogram of gap-to-next-episode-at-the-same-location over 678 real
    June-2024 episode pairs (85 accident-bearing PKs) has a sharp mode at
    10-20 min (270 pairs, the single largest bucket) that falls off 3.5x to 77
    pairs at 20-30 min, then decays slowly and genuinely ambiguously out to
    180 min, then a clean empty band (180-300 min, 2 pairs) before a large
    300+min cluster (166) that is almost certainly unrelated later incidents.
    20 min captures the dominant immediate-re-flag mode without reaching into
    that ambiguous middle tail -- e.g. it merges the pk=120 example's first
    gap (tick1->tick2, exactly 20 min) but correctly leaves its second gap
    (tick2->tick3, 45 min) unmerged, matching the real distribution's own
    break point rather than assuming every same-location cluster is one
    incident. See `verify_precursor_tag.py` test 8 for the worked example.
    Set `merge_gap_minutes=0` to disable and use raw episode boundaries
    exactly.
    Merging only affects where episode boundaries (s, e) are computed for
    WINDOW PLACEMENT; `{orig_col}` always reflects the true unmerged raw flag.

    The retention block itself, and everything outside the chosen window, is
    labelled 0 in the new scheme: the model is meant to learn the precursor
    signature, not the "retention already happening" state it already knows how
    to detect from raw speed alone.

    Parameters
    ----------
    df : DataFrame with at least [loc_cols, time_col, speed_col, accident_col].
        Must be one row per (location, timestep) on a REGULAR `time_res_min`
        grid (as produced by the ETL's spatiotemporal base) -- irregular gaps
        will desynchronise the step-index arithmetic used for lookback/window
        sizing. Row order does not matter; the function sorts internally.
    Returns
    -------
    (df_out, events) :
        df_out  -- copy of `df`, original row order restored, with:
            {new_accident_col}                 the new precursor label (0/1)
            {orig_col}                         the original retention flag (0/1)
            accident_precursor_method          'adaptive' | 'fallback' | NaN
            accident_precursor_lead_min        minutes from window start to onset
            accident_precursor_anchor_dat      onset timestamp the tag belongs to
        events  -- one row per retention episode, full diagnostics (baseline
            speed, threshold, method, lead time, clipping applied) for
            reporting/plotting -- same shape of information as
            `analyze_accident_speed_mismatch.identify_accident_events`.
    """
    loc_cols = list(loc_cols)
    lookback_steps = max(1, round(lookback_minutes / time_res_min))
    tag_window_steps = max(1, round(tag_window_minutes / time_res_min))
    min_persistence_steps = max(1, min_persistence_steps)

    df = df.copy()
    df["__orig_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(loc_cols + [time_col], kind="stable").reset_index(drop=True)

    n = len(df)
    new_label = np.zeros(n, dtype=np.int8)
    method_arr = np.full(n, np.nan, dtype=object)
    lead_arr = np.full(n, np.nan)
    anchor_arr = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    acc_all = df[accident_col].fillna(0).to_numpy()
    speed_all = df[speed_col].to_numpy(dtype=float)
    dat_all = pd.to_datetime(df[time_col]).to_numpy()
    merge_gap_steps = max(0, round(merge_gap_minutes / time_res_min))

    # Occupancy proxy (flow / speed ~ density) for the multi-signal detector.
    # Only built when needed and when the volume column is actually present.
    occ_all = None
    if use_multisignal and "intTot" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            occ_all = df["intTot"].to_numpy(dtype=float) / np.clip(speed_all, 1e-3, None)

    event_records = []
    n_adaptive = 0
    n_fallback = 0
    n_no_history = 0
    n_merged_gaps = 0

    for _, idx in df.groupby(loc_cols, sort=False).indices.items():
        idx = np.asarray(idx)
        # groupby(sort=False).indices preserves within-group ORDER from the
        # sorted frame, i.e. already chronological for this location.
        acc_raw = acc_all[idx]
        speed = speed_all[idx]
        dat = dat_all[idx]
        occ = occ_all[idx] if occ_all is not None else None

        # Episode BOUNDARIES are computed on the gap-bridged series; the raw
        # flag (acc_raw, -> ACCIDENT_RETENTION_ORIG) is never altered by this.
        acc = _close_short_gaps(acc_raw, merge_gap_steps)
        n_merged_gaps += int(acc.sum() - acc_raw.sum())  # rows bridged (0 -> 1 for boundary purposes)

        padded = np.concatenate(([0], acc, [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1.0)[0]
        ends = np.where(diff == -1.0)[0] - 1  # inclusive

        prev_end = -1  # local index (within this location) of the previous episode's last row
        for s, e in zip(starts, ends):
            lookback_left = max(0, s - lookback_steps, prev_end + 1)
            pre_speeds = speed[lookback_left:s]
            pre_dat = dat[lookback_left:s]
            pre_occ = occ[lookback_left:s] if occ is not None else None

            baseline_speed = np.nan
            if len(pre_speeds) >= min_baseline_steps:
                baseline_speed = float(np.mean(pre_speeds[: max(1, len(pre_speeds) // 2)]))

            drop_local_idx = None  # index into pre_speeds of the detected anomaly onset
            if (
                not np.isnan(baseline_speed)
                and baseline_speed > min_highway_speed
                and len(pre_speeds) > 0
            ):
                threshold = baseline_speed * speed_drop_ratio
                below = pre_speeds < threshold
                # first index with `min_persistence_steps` consecutive True
                if min_persistence_steps == 1:
                    hits = np.where(below)[0]
                    if len(hits) > 0:
                        drop_local_idx = int(hits[0])
                else:
                    run = 0
                    for i, b in enumerate(below):
                        run = run + 1 if b else 0
                        if run >= min_persistence_steps:
                            drop_local_idx = i - min_persistence_steps + 1
                            break
            else:
                threshold = np.nan

            # ---- multi-signal detection -------------------------------------
            # The mean-speed rule above only fires once speed has already
            # fallen 30%. The crash-precursor literature (e.g. Abdel-Aty et al.)
            # weights speed VARIABILITY and upstream-downstream speed DIFFERENCE
            # at least as heavily -- turbulence and shockwave formation precede
            # the mean-speed collapse. Firing on the EARLIEST of several
            # detectors both catches precursors the speed rule misses (turning
            # would-be `fallback` episodes into evidence-anchored `adaptive`
            # ones) and, because the window is anchored to that onset, pushes
            # the label further ahead of the incident.
            if use_multisignal and len(pre_speeds) >= max(4, min_baseline_steps):
                cands = [] if drop_local_idx is None else [drop_local_idx]
                base_n = max(2, len(pre_speeds) // 2)

                # (a) speed-variability spike: rolling std over a 2-step window
                #     exceeding `cv_spike_mult` x its own baseline level.
                sd = pd.Series(pre_speeds).rolling(2).std().to_numpy()
                sd_base = np.nanmean(sd[:base_n])
                if np.isfinite(sd_base) and sd_base > 0:
                    hits = np.where(sd > cv_spike_mult * max(sd_base, 1.0))[0]
                    if len(hits):
                        cands.append(int(hits[0]))

                # (b) abrupt step-to-step deceleration (>= grad_drop_kmh in one
                #     5-min step) -- the leading edge of a shockwave.
                dv = np.diff(pre_speeds, prepend=pre_speeds[0])
                hits = np.where(dv <= -abs(grad_drop_kmh))[0]
                if len(hits):
                    cands.append(int(hits[0]))

                # (c) occupancy proxy (flow/speed ~ density) spiking above its
                #     baseline -- congestion building while speed still looks OK.
                if pre_occ is not None and len(pre_occ) == len(pre_speeds):
                    ob = np.nanmean(pre_occ[:base_n])
                    if np.isfinite(ob) and ob > 0:
                        hits = np.where(pre_occ > occ_spike_mult * ob)[0]
                        if len(hits):
                            cands.append(int(hits[0]))

                if cands:
                    drop_local_idx = int(min(cands))   # earliest signal wins

            if drop_local_idx is not None:
                if anchor_before_drop:
                    # ADAPTIVE (corrected): the window ENDS where the anomaly
                    # starts, so it covers the `tag_window_minutes` of
                    # still-normal-looking traffic immediately BEFORE any
                    # visible degradation -- the actual precursor.
                    win_end_local = drop_local_idx
                    win_start_local = max(0, win_end_local - tag_window_steps)
                else:
                    # LEGACY: window STARTS at the anomaly and runs forward,
                    # i.e. it covers the developing congestion rather than what
                    # preceded it. Retained only to reproduce the first
                    # precursor dataset; not the intended semantics.
                    win_start_local = drop_local_idx
                    win_end_local = min(win_start_local + tag_window_steps, len(pre_speeds))
                method = "adaptive"
            elif len(pre_speeds) > 0 and not adaptive_only:
                # FALLBACK: literal last `tag_window_steps` before onset. No
                # anomaly was detected, so this window is not anchored to any
                # evidence -- see `adaptive_only` to exclude these instead.
                win_start_local = max(0, len(pre_speeds) - tag_window_steps)
                win_end_local = len(pre_speeds)
                method = "fallback"
            else:
                win_start_local = win_end_local = 0
                method = None

            if win_end_local > win_start_local:
                global_start = idx[lookback_left + win_start_local]
                global_end = idx[lookback_left + win_end_local]  # exclusive
                new_label[global_start:global_end] = 1
                onset_dat = dat[s]
                lead_min = (win_end_local - win_start_local) * time_res_min \
                    if method == "fallback" \
                    else (len(pre_speeds) - win_start_local) * time_res_min
                method_arr[global_start:global_end] = method
                lead_arr[global_start:global_end] = lead_min
                anchor_arr[global_start:global_end] = onset_dat
                if method == "adaptive":
                    n_adaptive += 1
                else:
                    n_fallback += 1
            else:
                n_no_history += 1
                lead_min = np.nan

            event_records.append({
                **{c: df[c].to_numpy()[idx[0]] for c in loc_cols},
                "accident_start": pd.Timestamp(dat[s]),
                "accident_end": pd.Timestamp(dat[e]),
                "duration_min": int((e - s + 1) * time_res_min),
                "baseline_speed": baseline_speed,
                "threshold": threshold if not np.isnan(baseline_speed) else np.nan,
                "method": method,
                "lead_min": lead_min,
                "lookback_available_min": len(pre_speeds) * time_res_min,
            })

            prev_end = e

    df[new_accident_col] = new_label
    df[orig_col] = acc_all.astype(np.int8)
    df["accident_precursor_method"] = method_arr
    df["accident_precursor_lead_min"] = lead_arr
    df["accident_precursor_anchor_dat"] = anchor_arr

    # Restore caller's original row order.
    df = df.sort_values("__orig_order", kind="stable").drop(columns="__orig_order").reset_index(drop=True)

    events = pd.DataFrame(event_records)

    n_events = len(events)
    print(f"[PRECURSOR-TAG] merge_gap_minutes={merge_gap_minutes} bridged "
          f"{n_merged_gaps:,} rows before episode detection "
          f"(fragments of the same incident, not independent events)")
    print(f"[PRECURSOR-TAG] {n_events:,} retention episodes -> "
          f"{n_adaptive:,} adaptive ({n_adaptive/max(n_events,1)*100:.1f}%), "
          f"{n_fallback:,} fallback ({n_fallback/max(n_events,1)*100:.1f}%), "
          f"{n_no_history:,} untaggable (no pre-history, {n_no_history/max(n_events,1)*100:.1f}%)")
    if n_events:
        tagged_leads = events["lead_min"].dropna()
        if len(tagged_leads):
            print(f"[PRECURSOR-TAG] lead time (window-start -> onset): "
                  f"mean={tagged_leads.mean():.1f} min, median={tagged_leads.median():.1f} min, "
                  f"max={tagged_leads.max():.0f} min")
    print(f"[PRECURSOR-TAG] new ACCIDENT positives: {int(new_label.sum()):,} rows "
          f"(was {int(acc_all.sum()):,} rows under the retention-duration definition)")

    return df, events 