#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils_bathy.py
@Time    :   2026/05/18 16:22:24
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import xarray as xr

PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
ROOT_BATHY_DATA = os.path.join(PROJECT_ROOT, "data", "bathy")


def load_bathy(
    box_center_lon,
    box_center_lat,
    dlon_box=0.25,
    dlat_box=0.25,
    root_bathy_data=ROOT_BATHY_DATA,
):
    # input_data_root = os.path.join(project_root, "data")
    bathy_fpath = os.path.join(root_bathy_data, "GEBCO_2021_sub_ice_topo.nc")

    # Load bathy data
    ds_bathy = xr.open_dataset(bathy_fpath)

    # Slice data to get the area of interest
    lat0 = box_center_lat
    lon0 = box_center_lon
    ds_bathy = ds_bathy.sel(
        lat=slice(
            lat0 - dlat_box / 2,
            lat0 + dlat_box / 2,
        ),
        lon=slice(
            lon0 - dlon_box / 2,
            lon0 + dlon_box / 2,
        ),
    )

    return ds_bathy


if __name__ == "__main__":
    pass
