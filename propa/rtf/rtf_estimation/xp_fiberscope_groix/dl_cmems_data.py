#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dl_cmems_data.py
@Time    :   2025/09/02 08:23:34
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

from get_data.cmems import load_data_from_cmems as cmems
from misc import dms_to_deg

# if __name__ == "__main__":

folder_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\xp_fiberscope_groix"
data_folder_path = os.path.join(folder_root, "data")

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

# Root
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\cmems"


# https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services
# dataset_id = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
# dataset_version = "202406"
# start_datetime = "2022-09-15T00:00:00"
# end_datetime = "2022-09-16T00:00:00"

dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
dataset_version = "202311"
start_datetime = "2020-09-15T00:00:00"
end_datetime = "2020-09-16T00:00:00"

# Set file name
fname = f"cmems_thetao_so_{dataset_id.split('_')[-1]}_{start_datetime[:10]}_{end_datetime[:10]}.nc"
# fname = "data_cmems.nc"

process_pos = False
if process_pos:
    # Load position of the observation
    df_pos = pd.read_csv(
        os.path.join(data_folder_path, "pos_dm.csv"),
        header=0,
        sep=",",
    )
    lat_s = df_pos.lat_m * 0
    lat_deg = dms_to_deg(df_pos.lat_d, df_pos.lat_m, lat_s)
    lon_s = df_pos.lon_m * 0
    lon_deg = -dms_to_deg(-df_pos.lon_d, df_pos.lon_m, lon_s)

    df_pos_deg = pd.DataFrame(
        {
            "id": df_pos.id,
            "lat": lat_deg,
            "lon": lon_deg,
        }
    )
    # Set id as index
    df_pos_deg = df_pos_deg.set_index("id")

    # Save to csv
    df_pos_deg.to_csv(os.path.join(data_folder_path, "pos_deg.csv"))
else:
    # Load pos
    df_pos_deg = pd.read_csv(
        os.path.join(data_folder_path, "pos_deg.csv"),
        header=0,
        sep=",",
        index_col=0,
    )

dlat_box = 0.1
dlon_box = 0.1

# Delete existing file
fpath = os.path.join(data_folder_path, fname)
if os.path.exists(fpath):
    os.remove(fpath)

# Set parameters
data_request = dict(
    dataset_id=dataset_id,
    dataset_version=dataset_version,
    variables=[
        "so",
        "thetao",
    ],
    minimum_longitude=df_pos_deg.loc["obs3"].lon - dlon_box,
    maximum_longitude=df_pos_deg.loc["obs3"].lon + dlon_box,
    minimum_latitude=df_pos_deg.loc["obs3"].lat - dlat_box,
    maximum_latitude=df_pos_deg.loc["obs3"].lat + dlat_box,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    minimum_depth=0,
    maximum_depth=100,
    output_dir=data_folder_path,
    output_filename=fname,
    force_download=True,
)

ds_cmems = cmems.load_data(data_request)

print(ds_cmems)
