"""File names and folders read by the pre-processing script.

Paths are relative to the working directory, so run
`src/data/initial_data_processing_5min_fund-propag.py` from the repository root.
"""
import os

data_version = 'preMARIA_v2_geo-cal-vel-int-ret_v1'   # prefix of the assembled CSV
model_version = 'from-gen23-to-dec24'

folder_path = os.getcwd()
folder_dades = os.path.join(folder_path, 'data', 'processed')
folder_dades_raw = os.path.join(folder_path, 'data', 'raw')
folder_visualizations = os.path.join(folder_path, 'visualizations')
folder_model = os.path.join(folder_path, 'model_' + model_version)

pkini_pkfi_etds = 'pkini_pkfi_etds_unique.csv'    # loop-detector section ranges
file_geo_vies = 'pksCur_AP7_120_220.csv'          # road geometry per kilometre post
file_cal_mob = 'dataFest_2021_2024.csv'           # special-mobility calendar
