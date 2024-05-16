#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   add_event.py
@Time    :   2024/05/16 16:19:10
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr

from localisation.verlinden.verlinden_utils import (
    init_event_src_traj,
    init_grid_around_event_src_traj,
    get_bathy_grid_size,
    load_rhumrum_obs_pos,
)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def add_event(xr_dataset):
    pass


def load_subset(fpath, src_info, grid_info, dt):
    """
    Load a subset of the dataset around the source to be localized.
    """
    # Load the dataset
    ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    # Define limits of the subset area
    init_event_src_traj(src_info, dt)
    init_grid_around_event_src_traj(src_info, grid_info)

    # Extract area around the source
    ds_subset = ds.sel(
        lon=slice(grid_info["min_lon"], grid_info["max_lon"]),
        lat=slice(grid_info["min_lat"], grid_info["max_lat"]),
    )

    return ds_subset


if __name__ == "__main__":
    fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1\propa\propa_65.5973_65.8993_-27.6673_-27.3979.zarr"
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    dt = 7
    v_knots = 20  # 20 knots
    v_ship = v_knots * 1852 / 3600  # m/s

    z_src = 5
    route_azimuth = 45  # North-East route

    fs = 100
    duration = 200  # 1000 s
    nmax_ship = 1
    src_stype = "ship"

    rcv_info = {
        # "id": ["RR45", "RR48", "RR44"],
        # "id": ["RRpftim0", "RRpftim1", "RRpftim2"],
        "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)

    initial_ship_pos = {
        "lon": rcv_info["lons"][0],
        "lat": rcv_info["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    src_info = {
        "speed": v_ship,
        "depth": z_src,
        "duration": duration,
        "signal_type": src_stype,
        "max_nb_of_pos": nmax_ship,
        "route_azimuth": route_azimuth,
        "initial_pos": initial_ship_pos,
    }

    # Event
    f0_event = 1.5  # Fundamental frequency of the ship signal
    event_src_info = {
        "sig_type": "ship",
        "f0": f0_event,
        "std_fi": f0_event * 10 / 100,
        "tau_corr_fi": 1 / f0_event,
        "fs": fs,
    }

    src_info["event"] = event_src_info

    lon, lat = rcv_info["lons"][0], rcv_info["lats"][0]
    dlon, dlat = get_bathy_grid_size(lon, lat)

    grid_offset_cells = 35

    grid_info = dict(
        offset_cells_lon=grid_offset_cells,
        offset_cells_lat=grid_offset_cells,
        dx=100,
        dy=100,
        dlat_bathy=dlat,
        dlon_bathy=dlon,
    )

    load_subset(fpath, src_info, grid_info, dt)
