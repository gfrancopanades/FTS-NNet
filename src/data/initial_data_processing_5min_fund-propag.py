#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data integration and preprocessing pipeline for 5-minute velocity and intensity features
along the AP-7 highway corridor.

This script constructs a complete spatiotemporal dataset from multiple sources:
5-minute aggregated velocity records (2022-2025), 5-minute aggregated traffic intensity (2022-2025), 
special mobility calendar, and road geometry features. The result is a structured dataset 
ready for training spatiotemporal predictive models.

Data Sources:
- Speed: ap7_5min_speed_aggregations_2022_2025.csv
- Intensity: ap7_5min_intensity_aggregations_2022_2025.csv
- Mobility Calendar: Special events and holidays
- Road Geometry: Curvature, slope, and segment information

Author: Gerard Franco-Panadés
Affiliation: Universitat Politècnica de Catalunya
"""

import sys
import os
# Ensure project root is on sys.path so `import src...` works when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product

from src.data_file_names import *  # Path configuration  
from src.data.data_utils import *  # Custom data treatment functions

# Disable MKL optimizations to prevent potential TensorFlow issues (if used downstream)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure output folder exists
check_and_create_folder(folder_model)

print("=" * 70)
print(">> Spatiotemporal data preprocessing: 5-MINUTE AGGREGATION")
print("=" * 70)

# ------------------------------------------------------------------------------
# CONFIGURATION: Boolean flags to control which datasets are loaded/processed
# ------------------------------------------------------------------------------

# Dataset inclusion flags
LOAD_FORECAST_WEATHER = False      # Forecast weather data (1d and 3d forecasts)
LOAD_VELOCITY_DATA = True         # 5-minute velocity/speed data (REQUIRED for intensity propagation)
LOAD_INTENSITY_DATA = True        # 5-minute traffic intensity data (REQUIRED for fundamental diagram)
LOAD_ACCIDENTS_DATA = True        # Accidents and retentions data
LOAD_MOBILITY_CALENDAR = True     # Special mobility calendar (holidays, events)
LOAD_GEOMETRY_DATA = True         # Road geometry data (curvature, slope, segments)

# Print configuration summary
print("\n[DATASET CONFIGURATION]")
print(f"  Forecast Weather    : {'✓' if LOAD_FORECAST_WEATHER else '✗'}")
print(f"  Velocity Data       : {'✓' if LOAD_VELOCITY_DATA else '✗'}")
print(f"  Intensity Data      : {'✓' if LOAD_INTENSITY_DATA else '✗'}")
print(f"  Accidents Data      : {'✓' if LOAD_ACCIDENTS_DATA else '✗'}")
print(f"  Mobility Calendar   : {'✓' if LOAD_MOBILITY_CALENDAR else '✗'}")
print(f"  Geometry Data       : {'✓' if LOAD_GEOMETRY_DATA else '✗'}")

# Validate required datasets
if not LOAD_VELOCITY_DATA:
    print("\n  ⚠️  WARNING: Velocity data is required for intensity propagation!")
if not LOAD_INTENSITY_DATA:
    print("\n  ⚠️  WARNING: Intensity data is required for fundamental diagram propagation!")

# Track which datasets were actually loaded (for final summary)
datasets_loaded = {
    'forecast_weather': False,
    'velocity': False,
    'intensity': False,
    'accidents': False,
    'mobility_calendar': False,
    'geometry': False
}

# ------------------------------------------------------------------------------
# Step 1: Construct a full 5-minute spatiotemporal grid for the AP-7 corridor
# ------------------------------------------------------------------------------

start_date = '2023-01-01'
end_date = '2025-10-01'  # New data from April to October 2025
start_date_dt = pd.to_datetime(start_date)
end_date_dt = pd.to_datetime(end_date)
via_values = ['AP-7']
pk_values = range(120, 220)
sen_values = ['dec','cre']

date_range = pd.date_range(start=start_date, end=end_date, freq='5min')

# Create DataFrame efficiently by chunking to avoid memory issues with 28M+ rows
print(f"  Creating grid: {len(date_range)} timestamps × {len(pk_values)} PKs × {len(sen_values)} directions")
total_expected = len(date_range) * len(pk_values) * len(sen_values)
print(f"  Expected rows: {total_expected:,}")

# Process in daily chunks to reduce memory footprint
chunk_size = 288  # One day = 288 5-minute intervals
chunks = []

print(f"  Processing in chunks of {chunk_size} timestamps...")
for i in range(0, len(date_range), chunk_size):
    date_chunk = date_range[i:i+chunk_size]
    chunk_grid = list(product(date_chunk, via_values, pk_values, sen_values))
    df_chunk = pd.DataFrame(chunk_grid, columns=['dat', 'via', 'pk', 'sen'])
    chunks.append(df_chunk)
    
    if (i // chunk_size + 1) % 50 == 0:  # Progress update every 50 days
        print(f"    Processed {i // chunk_size + 1} days...")

# Concatenate all chunks
df_base = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory immediately
df_base['Any'] = df_base['dat'].dt.year
df_base['mes'] = df_base['dat'].dt.month
df_base['dia'] = df_base['dat'].dt.day
df_base['diaSem'] = df_base['dat'].dt.weekday
df_base['hor'] = df_base['dat'].dt.hour
df_base['5min'] = (df_base['dat'].dt.minute // 5) * 5  # 5-minute interval: 0, 5, 10, 15, ..., 50, 55

print(f"> Spatiotemporal base created: {df_base.shape[0]:,} rows")

# Clean up grid construction variables
del date_range, via_values, pk_values, sen_values, chunk_size, total_expected

# ------------------------------------------------------------------------------
# Step 2: Merge with forecast weather data (hourly -> 5-minute grid)
# ------------------------------------------------------------------------------

if LOAD_FORECAST_WEATHER:
    forecast_file = os.path.join(folder_dades, "forecast_weather_1d_3d_clean.csv")
    print(f"> Loading forecast weather data from {forecast_file}")
    if os.path.exists(forecast_file):
        df_weather = pd.read_csv(forecast_file)
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_weather['Any'] = df_weather['time'].dt.year
        df_weather['mes'] = df_weather['time'].dt.month
        df_weather['dia'] = df_weather['time'].dt.day
        df_weather['hor'] = df_weather['time'].dt.hour
        df_weather['diaSem'] = df_weather['time'].dt.weekday
        df_weather['via'] = df_weather['via'].fillna('AP-7')
        df_weather['sen'] = df_weather['sen'].astype(str)

        forecast_merge_keys = ['Any', 'mes', 'dia', 'hor', 'via', 'sen', 'pk']
        excluded_cols = set(forecast_merge_keys + ['time', 'dat', 'diaSem'])
        forecast_cols = [col for col in df_weather.columns if col not in excluded_cols]

        df_base = df_base.merge(df_weather[forecast_merge_keys + forecast_cols],
                                on=forecast_merge_keys,
                                how='left')
        print(f"  Forecast data merged: {len(forecast_cols)} columns added")
        datasets_loaded['forecast_weather'] = True

        if forecast_cols:
            nan_counts = df_base[forecast_cols].isna().sum()
            total_missing = nan_counts.sum()
            print(f"  Total missing values after forecast merge: {total_missing:,}")
            missing_cols = nan_counts[nan_counts > 0].sort_values(ascending=True)
            if not missing_cols.empty:
                os.makedirs(folder_visualizations, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, max(4, len(missing_cols) * 0.4)))
                missing_cols.plot.barh(ax=ax, color='tab:blue', edgecolor='black')
                ax.set_xlabel('Missing rows')
                ax.set_title('Missing values per forecast column after merge')
                ax.grid(True, axis='x', alpha=0.35)
                plt.tight_layout()
                nan_plot_path = os.path.join(folder_visualizations, 'forecast_nan_counts.png')
                plt.savefig(nan_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Forecast NaN counts plotted at: {nan_plot_path}")
        del df_weather
        gc.collect()
    else:
        print(f"  ⚠️ Forecast file not found at {forecast_file}; skipping merge.")
else:
    print(f"> Skipping forecast weather data (LOAD_FORECAST_WEATHER = False)")

# ------------------------------------------------------------------------------
# Step 3: Merge with 5-minute velocity data
# ------------------------------------------------------------------------------

if LOAD_VELOCITY_DATA:
    df_vel = pd.read_csv(os.path.join(folder_dades, "ap7_5min_speed_aggregations_extinrix.csv"),
                         sep=",", decimal=".", encoding='latin-1')

    # Filter by date range
    df_vel['date'] = pd.to_datetime(df_vel[['Any', 'mes', 'dia']].rename(columns={'Any': 'year', 'mes': 'month', 'dia': 'day'}))
    df_vel = df_vel[(df_vel['date'] >= start_date_dt) & (df_vel['date'] < end_date_dt)].drop(columns=['date']).copy()

    # For 5-minute data, we only have mean_speed (no percentiles or std_dev)
    # If the file has different column names, rename them
    if 'mean_speed' not in df_vel.columns and 'speed' in df_vel.columns:
        df_vel = df_vel.rename(columns={'speed': 'mean_speed'})

    # Ensure we have the 5-minute interval column in minute format (0, 5, 10, ..., 55)
    if '5min' not in df_vel.columns:
        # Assuming there's a datetime column or minute column to derive from
        if 'datetime' in df_vel.columns:
            df_vel['datetime'] = pd.to_datetime(df_vel['datetime'])
            df_vel['5min'] = (df_vel['datetime'].dt.minute // 5) * 5  # Minute format: 0, 5, 10, ..., 55
        elif 'minute' in df_vel.columns:
            df_vel['5min'] = (df_vel['minute'] // 5) * 5  # Minute format: 0, 5, 10, ..., 55
    else:
        # If 5min exists, ensure it's in minute format (0, 5, 10, ..., 55)
        df_vel['5min'] = pd.to_numeric(df_vel['5min'], errors='coerce')
        # Convert from slot format (0-11) to minute format if needed
        if len(df_vel) > 0 and df_vel['5min'].notna().any():
            if df_vel['5min'].max() <= 11:
                df_vel['5min'] = (df_vel['5min'] * 5).astype(int)  # Convert slot (0-11) to minutes (0, 5, 10, ..., 55)
            else:
                # Already in minute format, ensure it's rounded to 0, 5, 10, ..., 55
                df_vel['5min'] = ((df_vel['5min'] // 5) * 5).astype(int)
        else:
            df_vel['5min'] = df_vel['5min'].astype(int)

    # Aggregate by temporal and spatial keys (in case there are duplicates)
    df_vel = df_vel.groupby(['Any', 'mes', 'dia', 'hor', '5min', 'via', 'pk', 'sen']).agg({
        'mean_speed': 'mean'
    }).reset_index()

    df_merge = df_base.merge(df_vel, on=['Any', 'mes', 'dia', 'hor', '5min', 'via', 'pk', 'sen'], how='left')
    datasets_loaded['velocity'] = True

    del df_vel
    del df_base
    gc.collect()

    df_merge = df_merge.sort_values(['via', 'sen', 'pk', 'Any', 'mes', 'dia', 'hor', '5min']).copy()
    df_merge['speed_imputation'] = df_merge['mean_speed'].isna().astype(int)
    df_merge['mean_speed'] = df_merge['mean_speed'].interpolate(method='linear', limit_direction='both')

    df_merge['pk'] = df_merge['pk'].astype(int)
else:
    print(f"> Skipping velocity data (LOAD_VELOCITY_DATA = False)")
    print(f"  ⚠️  ERROR: Velocity data is required! Creating empty merge...")
    df_merge = df_base.copy()
    df_merge['mean_speed'] = np.nan
    df_merge['speed_imputation'] = 1
    df_merge['pk'] = df_merge['pk'].astype(int)
    del df_base
    gc.collect()

# ------------------------------------------------------------------------------
# Step 3: Merge with 5-minute traffic intensity data
# ------------------------------------------------------------------------------

if LOAD_INTENSITY_DATA:
    df_int = pd.read_csv(os.path.join(folder_dades, "ap7_5min_intensity_aggregations_2022_2025.csv"),
                         sep=",", decimal=".", encoding='latin-1')

    # Filter by date range
    df_int['date'] = pd.to_datetime(df_int[['Any', 'mes', 'dia']].rename(columns={'Any': 'year', 'mes': 'month', 'dia': 'day'}))
    df_int = df_int[(df_int['date'] >= start_date_dt) & (df_int['date'] < end_date_dt)].drop(columns=['date']).copy()
                         
    # Ensure '5min' is in minute format (0, 5, 10, ..., 55)
    df_int['5min'] = pd.to_numeric(df_int['5min'], errors='coerce')
    if len(df_int) > 0 and df_int['5min'].notna().any():
        if df_int['5min'].max() <= 11 and df_int['5min'].max() > 0:
            # Convert slot format (0-11) to minutes (0, 5, 10, ..., 55)
            df_int['5min'] = (df_int['5min'] * 5).astype(int)
        else:
            # Already minute-like; coerce to multiples of 5
            df_int['5min'] = ((df_int['5min'] // 5) * 5).astype(int)
    else:
        df_int['5min'] = df_int['5min'].astype(int)

    # Create datetime column from components (now 5min is minutes)
    df_int['datetime_dt'] = pd.to_datetime(
        df_int[['Any', 'mes', 'dia', 'hor']].rename(columns={'Any': 'year', 'mes': 'month', 'dia': 'day', 'hor': 'hour'})
    ) + pd.to_timedelta(df_int['5min'], unit='m')

    # Map columns to match expected structure (light vs heavy, and lanes)
    df_int = df_int.rename(columns={
        'total': 'intTot',
        'vehicles_lleugers': 'intL',
        'vehicles_pesats': 'intP',
        'carr': 'car'  # Rename 'carr' (lane number) to 'car'
    })

    # Keep decimal PKs for ETD mapping (don't truncate yet!)
    # Store original decimal PK for later reference
    df_int['pk_decimal'] = df_int['pk'].copy()
    # Filter to PK range 120-220 (using decimal values)
    df_int = df_int[(df_int['pk'] >= 120) & (df_int['pk'] <= 220)].copy()

    # Create a temporary integer PK column for merging with speed data
    df_int['pk_for_speed'] = np.trunc(df_int['pk']).astype(int)
    df_int = df_int.merge(df_merge[['Any','mes','dia','hor','5min','via','pk','sen','mean_speed']], 
                          left_on=['Any','mes','dia','hor','5min','via','pk_for_speed','sen'],
                          right_on=['Any','mes','dia','hor','5min','via','pk','sen'], 
                          how='left', suffixes=('', '_speed'))
    # Drop the temporary columns from merge
    df_int = df_int.drop(columns=['pk_speed', 'pk_for_speed'])

    # Compute delta_x (distance to next pk within each timestamp and direction)
    df_int = df_int.sort_values(['datetime_dt', 'sen', 'pk']).copy()

    # Fundamental-diagram based intensity propagation with +/-50% restriction
    # We will propagate intensity values across PKs within each ETD range using
    # a fundamental diagram approach, but cap deviations at +/-50% of the sensor value.

    df=df_int.copy()

    # Free memory from df_int after copying
    del df_int
    gc.collect()

    # Load ETD ranges and map PK values
    df_etds = pd.read_csv(os.path.join(folder_dades, pkini_pkfi_etds), sep=";", decimal=".", encoding="latin-1")
    interval_index = pd.IntervalIndex.from_arrays(df_etds['pkIni'], df_etds['pkFi'], closed='both')

    df['ETD'] = df['pk'].map(lambda x: df_etds['ETD'][interval_index.contains(x)].iloc[0] if any(interval_index.contains(x)) else None)

    # Merge ETD data and filter ranges
    df = pd.merge(df, df_etds, on=['ETD'], how='left')

    # Store min/max before deleting df_etds
    etd_pkini_min = float(df_etds['pkIni'].min())
    etd_pkfi_max = float(df_etds['pkFi'].max())

    # Free memory from df_etds after merge
    del df_etds
    gc.collect()

    df = df[(df['pkIni'] >= etd_pkini_min) & (df['pkFi'] <= etd_pkfi_max)]
    df = df[(df['pk'] >= 120) & (df['pk'] <= 220)].sort_values(by='pk')

    # IMPORTANT: Store original sensor PK (still decimal at this point) and intensity values before exploding
    # NOTE: Keeping decimal PKs until after ETD mapping ensures sensors like 144.95 and 195.9
    #       are correctly mapped to their ETD ranges, then propagated to all integer PKs in that range.
    #       Previous bug: truncating to 144 and 195 before ETD mapping caused them to map to wrong ranges,
    #       leaving integer PKs 145, 196, 197, 198 with no data.
    df['pk_sensor_original'] = df['pk'].copy()  # Original sensor PK location (decimal)
    df['intTot_sensor_original'] = df['intTot'].copy()  # Original intTot value at sensor
    df['intP_sensor_original'] = df['intP'].copy()  # Original intP value at sensor
    df['mean_speed_sensor'] = df['mean_speed'].copy()  # Speed at sensor location

    # Explode PK ranges into individual rows (this creates INTEGER PKs)
    pk_ranges = df.apply(lambda row: list(range(int(np.ceil(row['pkIni'])), int(np.floor(row['pkFi']) + 1))), axis=1)
    df = df.loc[df.index.repeat(pk_ranges.str.len())]
    df['pk'] = np.concatenate(pk_ranges.values)  # NOW pk becomes integer (after explosion)

    # Clean up pk_ranges immediately after use
    del pk_ranges
    gc.collect()

    # Calculate distance from sensor PK (in km)
    df['distance_from_sensor'] = np.abs(df['pk'] - df['pk_sensor_original'])

# ============================================================================
# FUNDAMENTAL DIAGRAM APPROXIMATION FOR INTENSITY PROPAGATION
# ============================================================================
# 
# APPROACH: Use fundamental traffic flow equation q = k * v
# where q = flow (vehicles/hour), k = density (vehicles/km), v = speed (km/h)
#
# DATA SOURCES:
# - Intensity (q): ETD sensors at specific PKs only
# - Speed (v): GPS data available at all PKs
# 
# STRATEGY:
# 1. Calculate density at sensor: k_sensor = q_sensor / v_sensor
# 2. Propagate density within ETD section (density is more conserved than flow)
# 3. Reconstruct intensity at target PK: q_target = k_propagated * v_target
#
# ============================================================================
# APPROXIMATIONS AND ASSUMPTIONS:
# ============================================================================
#
# A1. HOMOGENEOUS ETD SECTIONS
#     - Assumes road characteristics (lanes, geometry) are constant within ETD
#     - Reality: Minor variations exist (curvature, slope changes)
#     - Impact: Moderate - ETDs are designed to be homogeneous
#
# A2. EXPONENTIAL DENSITY DECAY
#     - Model: k(x) = k_sensor * exp(-alpha * distance)
#     - Assumes vehicles enter/exit at constant rate with distance
#     - Reality: On/off ramps are discrete, not continuous
#     - alpha = 0.02 km^-1 means ~2% density reduction per km
#     - Impact: Small within ETD sections (typically < 5 km)
#
# A3. CONSERVATION OF VEHICLE TYPE RATIO
#     - Assumes heavy/total ratio stays constant: intP/intTot = constant
#     - Reality: Heavy vehicles may have different entry/exit patterns
#     - Impact: Small - heavy vehicle ratio is relatively stable
#
# A4. INSTANTANEOUS EQUILIBRIUM
#     - Assumes traffic is in steady state (no shockwaves or congestion waves)
#     - Uses q = k * v without considering temporal dynamics
#     - Reality: Traffic waves propagate, especially in congestion
#     - Impact: Large during incidents/congestion formation
#
# A5. GPS SPEED REPRESENTS ETD SENSOR SPEED
#     - Uses GPS speed at sensor PK as proxy for ETD sensor speed
#     - Reality: GPS may have different sample, coverage, or timing
#     - Impact: Moderate - introduces measurement uncertainty
#
# A6. SPATIAL INTERPOLATION VALIDITY
#     - Assumes measurements at one PK are valid for nearby PKs
#     - Reality: Local disturbances, lane closures, incidents
#     - Mitigation: ±15% bounds limit unrealistic extrapolations
#
# ============================================================================
# LIMITATIONS:
# ============================================================================
#
# L1. NO CONGESTION DYNAMICS
#     - Cannot model stop-and-go waves, phantom jams, or shockwaves
#     - Fundamental diagram is static; real traffic has hysteresis
#
# L2. NO LANE-SPECIFIC MODELING
#     - Aggregates across all lanes; ignores lane changing, truck restrictions
#     - Reality: Heavy vehicles concentrate in right lanes
#
# L3. NO INCIDENT PROPAGATION
#     - Cannot model how accidents affect upstream/downstream flow
#     - Would need wave equation: ∂k/∂t + ∂q/∂x = 0
#
# L4. ASSUMES FREE FLOW OR STEADY CONGESTION
#     - Works well for: uncongested flow, stable congestion
#     - Fails for: transitional states, capacity drop, queue formation
#
# L5. IGNORES WEATHER, VISIBILITY, ROAD CONDITIONS
#     - Speed-flow relationship changes with conditions
#     - Would need conditional fundamental diagrams
#
# L6. NO RAMP METERING OR TRAFFIC CONTROL
#     - Assumes natural flow; ignores active traffic management
#
# ============================================================================

    # Small constant to avoid division by zero
    EPSILON = 1e-6

    # ============================================================================
    # SEGMENT-SPECIFIC CALIBRATION OF ALPHA_DENSITY
    # ============================================================================
    # 
    # For interurban highways, alpha varies by segment characteristics:
    # - Road geometry (curvature, slope)
    # - Traffic patterns (intensity variability)
    # - Segment type (through section vs. near interchange)
    #
    # Strategy: Calculate alpha for each ETD segment based on:
    # 1. Geometric complexity (higher curvature/slope → higher alpha)
    # 2. Flow variability (CV of intensity over time → reflects entry/exit activity)
    #
    # Expected range for interurban highway: 0.015 - 0.04 km^-1
    # ============================================================================

    print("\n> Calibrating segment-specific density decay parameters...")

    # Group by ETD and sensor location to calculate flow statistics
    # Use a sample of timestamps to avoid computational burden
    sample_timestamps = df['datetime_dt'].drop_duplicates().sample(
        n=min(1000, df['datetime_dt'].nunique()), 
        random_state=42
    )
    df_sample = df[df['datetime_dt'].isin(sample_timestamps)].copy()

    # Calculate flow variability at each sensor location (proxy for entry/exit activity)
    flow_stats = df_sample[df_sample['distance_from_sensor'] == 0].groupby(['ETD', 'pk_sensor_original', 'sen']).agg({
        'intTot_sensor_original': ['mean', 'std'],
        'mean_speed_sensor': 'mean'
    }).reset_index()

    flow_stats.columns = ['ETD', 'pk_sensor', 'sen', 'intTot_mean', 'intTot_std', 'speed_mean']

    # Calculate Coefficient of Variation (CV) as measure of flow variability
    flow_stats['flow_cv'] = flow_stats['intTot_std'] / (flow_stats['intTot_mean'] + EPSILON)

    # Merge geometry data (if available in df)
    if 'ang_curv' in df.columns and 'ang_pend_pos' in df.columns:
        geo_stats = df_sample[df_sample['distance_from_sensor'] == 0].groupby(['ETD', 'sen']).agg({
            'ang_curv': 'max',  # Maximum curvature in section
            'ang_pend_pos': 'max',  # Maximum positive slope
            'ang_pend_neg': 'max'   # Maximum negative slope
        }).reset_index()
        
        flow_stats = flow_stats.merge(geo_stats, on=['ETD', 'sen'], how='left')
        flow_stats[['ang_curv', 'ang_pend_pos', 'ang_pend_neg']] = flow_stats[['ang_curv', 'ang_pend_pos', 'ang_pend_neg']].fillna(0)
        
        # Calculate geometric complexity score (normalized 0-1)
        geo_complexity = (
            0.5 * (flow_stats['ang_curv'] / (flow_stats['ang_curv'].max() + EPSILON)) +
            0.25 * (flow_stats['ang_pend_pos'] / (flow_stats['ang_pend_pos'].max() + EPSILON)) +
            0.25 * (flow_stats['ang_pend_neg'] / (flow_stats['ang_pend_neg'].max() + EPSILON))
        )
    else:
        geo_complexity = 0

    # Calculate traffic complexity score from flow variability (normalized 0-1)
    # Higher CV indicates more entry/exit activity or traffic pattern changes
    flow_complexity = flow_stats['flow_cv'] / (flow_stats['flow_cv'].quantile(0.95) + EPSILON)
    flow_complexity = flow_complexity.clip(upper=1.0)

    # Combine geometric and traffic complexity
    # Weight: 60% traffic variability, 40% geometry (traffic is more direct indicator)
    combined_complexity = 0.6 * flow_complexity + 0.4 * geo_complexity

    # Map complexity to alpha range for interurban highway
    # Base: 0.015 km^-1 (rural/stable sections)
    # Max:  0.040 km^-1 (complex/high-activity sections)
    alpha_base = 0.015
    alpha_range = 0.025  # (0.040 - 0.015)
    flow_stats['alpha_calibrated'] = alpha_base + alpha_range * combined_complexity

    # Create lookup dictionary: (ETD, sen) -> alpha
    alpha_lookup = flow_stats.set_index(['ETD', 'sen'])['alpha_calibrated'].to_dict()

    # Apply segment-specific alpha to dataframe
    df['alpha_segment'] = df.apply(lambda row: alpha_lookup.get((row['ETD'], row['sen']), 0.025), axis=1)

    # Summary statistics
    print(f"  Calibrated alpha_density for {len(alpha_lookup)} segments")
    print(f"  Alpha range: [{flow_stats['alpha_calibrated'].min():.4f}, {flow_stats['alpha_calibrated'].max():.4f}] km^-1")
    print(f"  Alpha mean:  {flow_stats['alpha_calibrated'].mean():.4f} km^-1")
    print(f"  Alpha median: {flow_stats['alpha_calibrated'].median():.4f} km^-1")

    # Show examples of different alpha values by segment type
    print(f"\n  Sample calibration by segment:")
    for quantile, label in [(0.0, 'Most stable'), (0.5, 'Typical'), (1.0, 'Most variable')]:
        idx = flow_stats['alpha_calibrated'].quantile(quantile)
        sample = flow_stats.loc[flow_stats['alpha_calibrated'].abs().sub(idx).abs().idxmin()]
        print(f"    {label:15s}: ETD {sample['ETD']}, alpha={sample['alpha_calibrated']:.4f} km^-1, " +
              f"Flow CV={sample['flow_cv']:.3f}")

    # Clean up calibration dataframes
    del df_sample, sample_timestamps, flow_stats, flow_complexity, combined_complexity, alpha_lookup
    if 'geo_stats' in locals():
        del geo_stats
    if 'geo_complexity' in locals():
        del geo_complexity

    # ============================================================================
    # DENSITY PROPAGATION WITH SEGMENT-SPECIFIC ALPHA
    # ============================================================================

    # Step 1: Calculate traffic density at sensor location
    # Density [veh/km] = Flow [veh/h] / Speed [km/h]
    df['density_sensor'] = df['intTot_sensor_original'] / (df['mean_speed_sensor'] + EPSILON)

    # Step 2: Propagate density with SEGMENT-SPECIFIC exponential decay
    # Uses calibrated alpha for each ETD section based on its characteristics
    df['density_propagated'] = df['density_sensor'] * np.exp(-df['alpha_segment'] * df['distance_from_sensor'])

    # Step 3: Reconstruct intensity at target PK using GPS speed at that location
    # Flow [veh/h] = Density [veh/km] × Speed [km/h]
    # KEY ADVANTAGE: Uses actual GPS speed field, captures speed variations
    df['intTot_propagated'] = df['density_propagated'] * df['mean_speed']

    # Step 4: Propagate heavy vehicle intensity maintaining vehicle type ratio
    # Assumes heavy vehicle percentage stays constant within ETD section
    heavy_vehicle_ratio = df['intP_sensor_original'] / (df['intTot_sensor_original'] + EPSILON)
    df['intP_propagated'] = heavy_vehicle_ratio * df['intTot_propagated']

    # Apply +/-50% restriction relative to original sensor values
    max_deviation = 0.50  # 50% maximum deviation

    # Calculate bounds
    df['intTot_lower_bound'] = df['intTot_sensor_original'] * (1 - max_deviation)
    df['intTot_upper_bound'] = df['intTot_sensor_original'] * (1 + max_deviation)
    df['intP_lower_bound'] = df['intP_sensor_original'] * (1 - max_deviation)
    df['intP_upper_bound'] = df['intP_sensor_original'] * (1 + max_deviation)

    # Clip propagated values to stay within +/-50% bounds
    df['intTot'] = df['intTot_propagated'].clip(
        lower=df['intTot_lower_bound'],
        upper=df['intTot_upper_bound']
    )
    df['intP'] = df['intP_propagated'].clip(
        lower=df['intP_lower_bound'],
        upper=df['intP_upper_bound']
    )

    # Update intL to maintain consistency (intTot = intL + intP)
    df['intL'] = df['intTot'] - df['intP']
    df['intL'] = df['intL'].clip(lower=0)  # Ensure non-negative

    # Clean up temporary columns used for propagation
    df.drop(columns=['pk_sensor_original', 'intTot_sensor_original', 'intP_sensor_original', 
                     'mean_speed_sensor', 'distance_from_sensor', 'alpha_segment',
                     'density_sensor', 'density_propagated', 'intTot_propagated', 'intP_propagated', 
                     'intTot_lower_bound', 'intTot_upper_bound', 
                     'intP_lower_bound', 'intP_upper_bound'], inplace=True)

    # Clean up ETD-related columns and pk_decimal (no longer needed)
    df.drop(columns=['ETD', 'pkIni', 'pkFi', 'pk_decimal'], inplace=True, errors='ignore')

    print(f"> Applied fundamental-diagram (q=k×v) intensity propagation with segment-specific calibration")
    print(f"  - Segment-specific alpha calibrated from traffic and geometry data")
    print(f"  - Applied +/-50% bounds to limit extrapolation errors")
    print(f"  - Using GPS speed field for reconstruction at all PKs")

    df_int=df.copy()

    # Clean up temporary dataframes
    del df
    gc.collect()

    # Ensure all temporal columns are integers (same as intensity data)
    for c in ['Any','mes','dia','hor','5min']: 
        df_int[c] = df_int[c].astype(int)
        
    # Ensure df_merge 5min is also integer (in minute format)
    df_merge['5min'] = df_merge['5min'].astype(int)

    # Intensity check BEFORE aggregation (per lane the sum should hold)
    if (df_int['intTot'] == (df_int['intL'] + df_int['intP'])).all():
        print("Intensity check Total = Light + Heavy passed successfully!!")
    else:
        print("Intensity check Total = Light + Heavy failed.")

    df_int['car'] = df_int['car'].fillna(1).astype(int)

    # Extract max car value per PK before merging (we'll need this later)
    df_car = df_int.groupby(['via','pk','sen']).agg({'car': 'max'}).reset_index()

    df_merge = df_merge.merge(
        df_int,
        on=['Any', 'mes', 'dia', 'hor', '5min', 'via', 'pk', 'sen'],
        how='left',
        suffixes=('', '_int')
    )

    # Free memory from df_int after merge
    del df_int
    gc.collect()

    df_merge = df_merge.sort_values(['via', 'sen', 'pk', 'Any', 'mes', 'dia', 'hor', '5min']).copy()

    # Drop all unnecessary columns from the merge (including duplicates and ETD-related columns)
    df_merge.drop(columns=['etd', 'car', 'hour', 'datetime_dt', 'pk_decimal', 'pkIni', 'pkFi', 
                           'ETD', 'mean_speed_int'], inplace=True, errors='ignore')

    # Merge back the car column
    df_merge = df_merge.merge(df_car, on=['via','pk','sen'], how='left')

    # Clean up df_car immediately after use
    del df_car

    # Interpolate only essential columns (no fundamental-diagram related fields)
    intensity_cols = ['intTot', 'intP', 'intL', 'car']
    df_merge['intensity_imputation'] = df_merge[intensity_cols].isna().any(axis=1).astype(int)
    df_merge[intensity_cols] = df_merge[intensity_cols].interpolate(method='linear', limit_direction='both')

    # Clean up intensity_cols list
    del intensity_cols

    # Directly assign the same intensity across PKs in ETD range (already replicated by explode)
    df_merge['intTot_fund'] = df_merge['intTot']
    df_merge['intP_fund'] = df_merge['intP']

    df_merge['intTot_fund'] = df_merge['intTot_fund'][df_merge['intTot_fund'].notna()].astype(int)
    df_merge['intP_fund'] = df_merge['intP_fund'][df_merge['intP_fund'].notna()].astype(int)

    # Clean up original intensity columns (we only export _fund versions)
    df_merge.drop(columns=['intTot', 'intP', 'intL'], inplace=True, errors='ignore')
    
    datasets_loaded['intensity'] = True
else:
    print(f"> Skipping intensity data (LOAD_INTENSITY_DATA = False)")
    print(f"  ⚠️  WARNING: Intensity data is required for fundamental diagram propagation!")
    print(f"  Creating empty intensity columns...")
    df_merge['intTot_fund'] = np.nan
    df_merge['intP_fund'] = np.nan
    df_merge['car'] = 1
    df_merge['intensity_imputation'] = 1

# ------------------------------------------------------------------------------
# Step 4: Merge with retentions/accidents data
# ------------------------------------------------------------------------------

if LOAD_ACCIDENTS_DATA:
    df_accidents = pd.read_csv(os.path.join(folder_dades, "bdd_retencions_accidents_AP7_pk115-225.csv"),
                              sep=",", decimal=".", encoding='latin-1')

    print(f"> Loaded {len(df_accidents):,} accident rows")
    print(f"  Accident columns: {df_accidents.columns.tolist()}")
    print(f"  Unique 'sen' values in accidents: {df_accidents['sen'].unique() if 'sen' in df_accidents.columns else 'N/A'}")

    # IMPORTANT: Keep 'sen' values as 'dec' and 'cre' to match df_merge
    # Do NOT convert to 'Sud'/'Nord' - that causes merge mismatch!
     # Map sen values from ['Est', 'Nord', 'Sud'] to ['cre', 'dec']
    # Est/Sud -> cre (increasing direction)
    # Nord -> dec (decreasing direction)
    if 'sen' in df_accidents.columns:
        df_accidents['sen'] = df_accidents['sen'].replace('Sud', 'cre').replace('Est', 'cre').replace('Nord', 'dec')
        print(f"  Mapped 'sen' values: Sud/Est -> cre, Nord -> dec")

    # Ensure column names match for merge
    if 'Any' not in df_accidents.columns and 'anyo' in df_accidents.columns:
        df_accidents = df_accidents.rename(columns={'anyo': 'Any'})

    # Ensure '5min' is in minute format (0, 5, 10, ..., 55) - same as df_merge
    if '5min' in df_accidents.columns:
        df_accidents['5min'] = pd.to_numeric(df_accidents['5min'], errors='coerce')
        # Ensure it's in minute format (0, 5, 10, ..., 55), not slot format (0-11)
        if len(df_accidents) > 0 and df_accidents['5min'].notna().any():
            if df_accidents['5min'].max() <= 11 and df_accidents['5min'].max() > 0:
                # It's in slot format (0-11), convert to minute format (0, 5, 10, ..., 55)
                df_accidents['5min'] = (df_accidents['5min'] * 5).astype(int)
                print(f"  Converted '5min' from slot format (0-11) to minute format (0, 5, 10, ..., 55)")
            else:
                # Already in minute format, but ensure it's rounded to 0, 5, 10, ..., 55
                df_accidents['5min'] = ((df_accidents['5min'] // 5) * 5).astype(int)
                print(f"  Ensured '5min' in minute format (0, 5, 10, ..., 55)")

    # Ensure 'pk' is integer - same conversion as intensity data (truncate, not round)
    if 'pk' in df_accidents.columns:
        df_accidents['pk'] = pd.to_numeric(df_accidents['pk'], errors='coerce')
        df_accidents['pk'] = np.trunc(df_accidents['pk']).astype(int)  # Truncate like intensity data
        print(f"  Converted 'pk' to integer (truncated like intensity data)")

    # Ensure all temporal columns are integers (same as intensity data)
    # Note: '5min' already handled above, so exclude it from this loop
    for c in ['Any','mes','dia','hor']:
        if c in df_accidents.columns:
            df_accidents[c] = pd.to_numeric(df_accidents[c], errors='coerce')
            df_accidents[c] = df_accidents[c].astype(int)
            print(f"  Converted '{c}' to integer (same as intensity data)")

    # Filter by date range if needed
    if 'date' not in df_accidents.columns:
        df_accidents['date'] = pd.to_datetime(df_accidents[['Any', 'mes', 'dia']].rename(columns={'Any': 'year', 'mes': 'month', 'dia': 'day'}))
    df_accidents = df_accidents[(df_accidents['date'] >= start_date_dt) & (df_accidents['date'] < end_date_dt)].drop(columns=['date']).copy()

    print(f"> After date filtering: {len(df_accidents):,} accident rows")
    if len(df_accidents) > 0:
        min_date = df_accidents[['Any', 'mes', 'dia']].min()
        max_date = df_accidents[['Any', 'mes', 'dia']].max()
        print(f"  Date range: {int(min_date['Any'])}-{int(min_date['mes']):02d}-{int(min_date['dia']):02d} to {int(max_date['Any'])}-{int(max_date['mes']):02d}-{int(max_date['dia']):02d}")
    else:
        print(f"  WARNING: No accident rows after date filtering!")

    # Identify the columns to merge (excluding the key columns)
    merge_keys = ['Any', 'mes', 'dia', 'hor', '5min', 'via', 'sen', 'pk']
    accident_cols = [col for col in df_accidents.columns if col not in merge_keys]

    print(f"\n> Merge keys: {merge_keys}")
    print(f"> Accident columns to merge: {accident_cols}")

    # Diagnostic: Check merge key values and data types before merge
    print(f"\n> ===== DETAILED MERGE KEY DIAGNOSTICS =====")

    # Check data types
    print(f"\n> Data Types Comparison:")
    for key in merge_keys:
        if key in df_merge.columns and key in df_accidents.columns:
            print(f"  {key}:")
            print(f"    df_merge dtype: {df_merge[key].dtype}")
            print(f"    df_accidents dtype: {df_accidents[key].dtype}")
            print(f"    Match: {df_merge[key].dtype == df_accidents[key].dtype}")
        elif key in df_merge.columns:
            print(f"  {key}: EXISTS in df_merge, MISSING in df_accidents")
        elif key in df_accidents.columns:
            print(f"  {key}: MISSING in df_merge, EXISTS in df_accidents")
        else:
            print(f"  {key}: MISSING in both dataframes")

    # Check unique values
    print(f"\n> Unique Values Comparison:")
    print(f"  df_merge 'sen' unique: {sorted(df_merge['sen'].unique())}")
    print(f"  df_accidents 'sen' unique: {sorted(df_accidents['sen'].unique()) if 'sen' in df_accidents.columns else 'N/A'}")
    print(f"  df_merge 'via' unique: {sorted(df_merge['via'].unique())}")
    print(f"  df_accidents 'via' unique: {sorted(df_accidents['via'].unique()) if 'via' in df_accidents.columns else 'N/A'}")

    # Check PK ranges and sample values
    print(f"\n> PK (Point Kilometric) Comparison:")
    print(f"  df_merge 'pk' range: {df_merge['pk'].min()} to {df_merge['pk'].max()}")
    print(f"  df_merge 'pk' sample values: {sorted(df_merge['pk'].unique())[:10]}")
    if 'pk' in df_accidents.columns:
        print(f"  df_accidents 'pk' range: {df_accidents['pk'].min()} to {df_accidents['pk'].max()}")
        print(f"  df_accidents 'pk' sample values: {sorted(df_accidents['pk'].unique())[:10]}")
        # Check if PKs overlap
        df_merge_pks = set(df_merge['pk'].unique())
        df_accidents_pks = set(df_accidents['pk'].unique())
        overlap_pks = df_merge_pks.intersection(df_accidents_pks)
        print(f"  PK overlap count: {len(overlap_pks):,} out of {len(df_accidents_pks):,} accident PKs")
        if len(overlap_pks) == 0:
            print(f"  ⚠️ WARNING: NO PK OVERLAP! This will cause merge to fail!")
    else:
        print(f"  df_accidents 'pk': N/A (column not found)")

    # Check 5min values
    print(f"\n> 5min (5-minute interval) Comparison:")
    print(f"  df_merge '5min' unique: {sorted(df_merge['5min'].unique())}")
    if '5min' in df_accidents.columns:
        print(f"  df_accidents '5min' unique: {sorted(df_accidents['5min'].unique())}")
    else:
        print(f"  df_accidents '5min': N/A (column not found)")

    # Check sample rows for both dataframes
    print(f"\n> Sample Rows (first 3):")
    print(f"\n  df_merge sample:")
    sample_keys = [k for k in merge_keys if k in df_merge.columns]
    print(df_merge[sample_keys].head(3).to_string())

    print(f"\n  df_accidents sample:")
    sample_keys_acc = [k for k in merge_keys if k in df_accidents.columns]
    if len(sample_keys_acc) > 0:
        print(df_accidents[sample_keys_acc].head(3).to_string())
        
        # Try to find a matching row manually
        print(f"\n> Testing manual row match:")
        if len(df_accidents) > 0:
            test_row = df_accidents.iloc[0]
            test_keys = {k: test_row[k] for k in merge_keys if k in df_accidents.columns}
            print(f"  Testing accident row: {test_keys}")
            
            # Try to find matching row in df_merge
            match_mask = pd.Series([True] * len(df_merge))
            for key, value in test_keys.items():
                if key in df_merge.columns:
                    match_mask = match_mask & (df_merge[key] == value)
            
            matches = match_mask.sum()
            print(f"  Matching rows in df_merge: {matches}")
            if matches > 0:
                print(f"  ✅ Found {matches} matching rows!")
                print(f"  Sample match:")
                print(df_merge[match_mask][sample_keys].head(3).to_string())
            else:
                print(f"  ❌ NO MATCHES FOUND!")
                print(f"  Checking each key individually:")
                for key, value in test_keys.items():
                    if key in df_merge.columns:
                        count = (df_merge[key] == value).sum()
                        print(f"    {key} = {value}: {count:,} rows in df_merge")
                        if count == 0:
                            print(f"      ⚠️ This key value doesn't exist in df_merge!")
                            print(f"      Sample df_merge values: {sorted(df_merge[key].unique())[:5]}")
    else:
        print("  No merge keys found in df_accidents!")

    print(f"\n> ===== END DIAGNOSTICS =====")

    # Clean up diagnostic variables
    if 'df_merge_pks' in locals():
        del df_merge_pks, df_accidents_pks, overlap_pks
    if 'sample_keys' in locals():
        del sample_keys, sample_keys_acc
    if 'test_row' in locals():
        del test_row, test_keys, match_mask, matches

    # Merge with df_merge
    print(f"\n> Merging accidents data...")
    df_merge_before_merge = len(df_merge)
    df_merge = df_merge.merge(df_accidents, on=merge_keys, how='left')

    # Check merge success
    merged_count = df_merge[accident_cols].notna().any(axis=1).sum() if accident_cols else 0
    print(f"> Merge completed:")
    print(f"  Rows in df_merge: {len(df_merge):,}")
    print(f"  Rows with accident data: {merged_count:,} ({merged_count/len(df_merge)*100:.4f}%)")

    # Fill NaN values with 0 for the new accident columns
    df_merge[accident_cols] = df_merge[accident_cols].fillna(0)

    print(f"> Accidents data merged: {len(accident_cols)} new columns added")
    print(f"  New columns: {accident_cols}")

    # Clean up accident-related temporary variables
    del df_accidents, df_merge_before_merge, merged_count, merge_keys
    gc.collect()
    datasets_loaded['accidents'] = True
else:
    print(f"> Skipping accidents data (LOAD_ACCIDENTS_DATA = False)")

# ------------------------------------------------------------------------------
# Step 5: Merge with mobility calendar (e.g., holidays or high traffic)
# ------------------------------------------------------------------------------

if LOAD_MOBILITY_CALENDAR:
    df_cal = processing_cal_mob(folder_dades_raw, file_cal_mob)
    df_cal = df_cal.groupby(['Any', 'mes', 'dia']).agg({'mob_esp': 'max'}).reset_index()

    df_merge = df_merge.merge(df_cal, on=['Any', 'mes', 'dia'], how='left')
    df_merge['mob_esp'] = df_merge['mob_esp'].fillna(0)
    datasets_loaded['mobility_calendar'] = True

    # Clean up calendar dataframe
    del df_cal
    gc.collect()
else:
    print(f"> Skipping mobility calendar (LOAD_MOBILITY_CALENDAR = False)")
    df_merge['mob_esp'] = 0

# ------------------------------------------------------------------------------
# Step 6: Merge with road geometry data
# ------------------------------------------------------------------------------

if LOAD_GEOMETRY_DATA:
    df_geo = processing_geo_vies(folder_dades_raw, file_geo_vies)
    df_geo = df_geo.groupby(['via', 'pk', 'sen']).agg({
        'ang_curv': 'max',
        'ang_pend_pos': 'max',
        'ang_pend_neg': 'max',
        'segment': 'max'
    }).reset_index()

    df_merge = df_merge.merge(df_geo, on=['via', 'pk', 'sen'], how='left')
    datasets_loaded['geometry'] = True
    del df_geo
    gc.collect()
else:
    print(f"> Skipping geometry data (LOAD_GEOMETRY_DATA = False)")
    df_merge['ang_curv'] = 0.0
    df_merge['ang_pend_pos'] = 0.0
    df_merge['ang_pend_neg'] = 0.0
    df_merge['segment'] = None

# Clean up date variables (no longer needed after data loading/filtering)
del start_date_dt, end_date_dt

# ------------------------------------------------------------------------------
# Step 7: Mark imputed values and interpolate missing features
# ------------------------------------------------------------------------------

df_merge=df_merge.rename(columns={'intTot_fund':'intTot','intP_fund':'intP'})

output_path = os.path.join(folder_dades, f"{data_version}_vel-extinrix_int_geo_mob_wthr_5min_fund-propag-ltd_from_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv")

print(f"> Data exported to: {output_path}")
print("Final DataFrame shape:", df_merge.shape)
print("Missing values per column:")
print(df_merge.isna().sum())
print("Duplicate rows:", df_merge.duplicated().sum())


# ------------------------------------------------------------------------------
# Step 8: Standardize types and round numerical precision
# ------------------------------------------------------------------------------

columns_operations = {
    'mean_speed': (int, 0),
    'car': (int, None),
    'intTot': (int, None),
    'intP': (int, None),
    'speed_imputation': (int, None),
    'intensity_imputation': (int, None),
    'ang_curv': (float, 2),
    'ang_pend_pos': (float, 2),
    'ang_pend_neg': (float, 2)
}

for column, (dtype, decimals) in columns_operations.items():
    if decimals is not None:
        df_merge[column] = df_merge[column].astype(dtype).round(decimals=decimals)
    else:
        df_merge[column] = df_merge[column].astype(dtype)

# Clean up columns_operations dictionary
del columns_operations

# ------------------------------------------------------------------------------
# Step 9: Save final dataset to CSV
# ------------------------------------------------------------------------------

# Rename columns before export; no column trimming so weather features stay
df_merge = df_merge.rename(columns={'Any':'anyo','intTot_fund':'intTot','intP_fund':'intP'}).copy()

# Base columns used later for diagnostics
base_columns = ['dat', 'via', 'pk', 'sen', 'anyo', 'mes', 'dia', 'diaSem', 'hor', '5min', 'mean_speed', 
                'intTot', 'intP', 'car', 'mob_esp', 'ang_curv', 'ang_pend_pos', 
                'ang_pend_neg', 'segment', 'speed_imputation', 'intensity_imputation']

# ------------------------------------------------------------------------------
# Step 10: Data quality checks and summary statistics before saving
# ------------------------------------------------------------------------------

print("\n" + "=" * 70)
print(">> DATA QUALITY CHECKS AND SUMMARY STATISTICS")
print("=" * 70)

# Unique values for categorical/discrete columns
print("\n[UNIQUE VALUES]")
categorical_cols = ['via', 'sen', 'pk', 'anyo', 'mes', 'diaSem', 'hor', '5min', 'car', 'mob_esp', 'segment']
for col in categorical_cols:
    if col in df_merge.columns:
        unique_count = df_merge[col].nunique()
        print(f"  {col:20s}: {unique_count:6d} unique values")
        if unique_count <= 20:  # Show values if there aren't too many
            print(f"    → Values: {sorted(df_merge[col].unique())}")

# Summary statistics for numerical columns
print("\n[NUMERICAL FEATURES - SUMMARY STATISTICS]")
numerical_cols = ['mean_speed', 'intTot', 'intP', 'ang_curv', 'ang_pend_pos', 'ang_pend_neg']
for col in numerical_cols:
    if col in df_merge.columns:
        print(f"\n  {col}:")
        print(f"    Mean   : {df_merge[col].mean():.2f}")
        print(f"    Std    : {df_merge[col].std():.2f}")
        print(f"    Min    : {df_merge[col].min():.2f}")
        print(f"    25%    : {df_merge[col].quantile(0.25):.2f}")
        print(f"    Median : {df_merge[col].median():.2f}")
        print(f"    75%    : {df_merge[col].quantile(0.75):.2f}")
        print(f"    Max    : {df_merge[col].max():.2f}")
        print(f"    Zeros  : {(df_merge[col] == 0).sum()} ({(df_merge[col] == 0).sum() / len(df_merge) * 100:.2f}%)")

# Forecast columns statistics
forecast_stats_cols = [col for col in df_merge.columns if col.startswith('1d_fcst_') or col.startswith('3d_fcst_')]
if forecast_stats_cols:
    print("\n[FORECAST FEATURES - SUMMARY STATISTICS]")
    for col in sorted(forecast_stats_cols):
        data = df_merge[col].dropna()
        if data.empty:
            continue
        print(f"\n  {col}:")
        print(f"    Mean   : {data.mean():.2f}")
        print(f"    Std    : {data.std():.2f}")
        print(f"    Min    : {data.min():.2f}")
        print(f"    25%    : {data.quantile(0.25):.2f}")
        print(f"    Median : {data.median():.2f}")
        print(f"    75%    : {data.quantile(0.75):.2f}")
        print(f"    Max    : {data.max():.2f}")
        if set(data.unique()).issubset({0, 1}):
            print(f"    Values : {data.value_counts().to_dict()}")
        else:
            zeros = (data == 0).sum()
            print(f"    Zeros  : {zeros} ({zeros / len(data) * 100:.2f}%)")

# Imputation flags summary
print("\n[IMPUTATION FLAGS]")
imputation_cols = ['speed_imputation', 'intensity_imputation']
for col in imputation_cols:
    if col in df_merge.columns:
        imputed_count = df_merge[col].sum()
        imputed_pct = imputed_count / len(df_merge) * 100
        print(f"  {col:25s}: {imputed_count:10d} rows ({imputed_pct:6.2f}%)")

# Accident columns summary (if present)
accident_summary_cols = [col for col in df_merge.columns if col not in base_columns and col != 'anyo']
if accident_summary_cols:
    print("\n[ACCIDENT/RETENTION FEATURES]")
    for col in accident_summary_cols:
        non_zero = (df_merge[col] != 0).sum()
        non_zero_pct = non_zero / len(df_merge) * 100
        print(f"  {col:35s}: {non_zero:8d} non-zero ({non_zero_pct:6.3f}%)")
        if non_zero > 0:
            print(f"    → Range: [{df_merge[col].min():.0f}, {df_merge[col].max():.0f}], Mean: {df_merge[col].mean():.4f}")

# Date range coverage
print("\n[TEMPORAL COVERAGE]")
print(f"  Start date: {df_merge['dat'].min()}")
print(f"  End date  : {df_merge['dat'].max()}")
print(f"  Total days: {(pd.to_datetime(df_merge['dat'].max()) - pd.to_datetime(df_merge['dat'].min())).days}")

# PK coverage by direction
print("\n[SPATIAL COVERAGE]")
for sen in df_merge['sen'].unique():
    pk_range = df_merge[df_merge['sen'] == sen]['pk'].agg(['min', 'max'])
    pk_count = df_merge[df_merge['sen'] == sen]['pk'].nunique()
    print(f"  Direction {sen:4s}: PK {pk_range['min']:.0f} to {pk_range['max']:.0f} ({pk_count} PKs)")

print("\n" + "=" * 70)

# Dataset loading summary
print("\n[DATASET LOADING SUMMARY]")
print(f"  Forecast Weather    : {'✓ Loaded' if datasets_loaded['forecast_weather'] else '✗ Skipped'}")
print(f"  Velocity Data       : {'✓ Loaded' if datasets_loaded['velocity'] else '✗ Skipped'}")
print(f"  Intensity Data      : {'✓ Loaded' if datasets_loaded['intensity'] else '✗ Skipped'}")
print(f"  Accidents Data      : {'✓ Loaded' if datasets_loaded['accidents'] else '✗ Skipped'}")
print(f"  Mobility Calendar   : {'✓ Loaded' if datasets_loaded['mobility_calendar'] else '✗ Skipped'}")
print(f"  Geometry Data       : {'✓ Loaded' if datasets_loaded['geometry'] else '✗ Skipped'}")
loaded_count = sum(datasets_loaded.values())
total_count = len(datasets_loaded)
print(f"\n  Total: {loaded_count}/{total_count} datasets loaded")

print("\n" + "=" * 70)

# Clean up summary statistics variables
del categorical_cols, numerical_cols, imputation_cols, accident_summary_cols
del base_columns, accident_cols

# Save final dataset
df_merge.to_csv(output_path, sep=";", decimal=".", encoding="latin-1", index=False)

final_columns = df_merge.columns.tolist()
print(f"\n✅ Data processing complete! File saved to: {output_path}")
print("\n[FINAL DATAFRAME COLUMNS]")
print(f"  Total columns: {len(final_columns)}")
print(f"  Columns: {final_columns}")

# Final memory cleanup
del df_merge
gc.collect()
print(f"Memory cleanup: Released final dataframe from memory")

