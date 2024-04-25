#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   build_misc.py
@Time    :   2024/04/23 10:42:59
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import pandas as pd
import dask.array

from pyproj import Geod
from localisation.verlinden.verlinden_utils import (
    get_range_from_rcv,
    get_azimuth_rcv,
    get_populated_path,
    set_azimuths,
    set_dataset_attrs,
    build_rcv_pairs,
    get_max_kraken_range,
    get_dist_between_rcv,
    get_bathy_grid_size,
)
from cst import C0


ROOT_DATASET_PATH = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset"


def init_dataset(
    rcv_info,
    testcase,
):
    """
    Initialize the dataset to be used by build_dataset().

    Parameters
    ----------
    grid_info : dict
        Grid information.
    rcv_info : dict
        Receiver information.
    src_info : dict
        Source information.

    Returns
    -------
    xr.Dataset
        Initialized dataset.

    """

    # Define grid
    grid_info = init_grid(rcv_info, minimum_distance_around_rcv=20 * 1e3)

    # Derive max distance to be used in kraken for each receiver
    get_max_kraken_range(rcv_info, grid_info)

    # Derive dist between rcvs
    get_dist_between_rcv(rcv_info)

    n_rcv = len(rcv_info["id"])

    # Compute range from each receiver
    rr_rcv = get_range_from_rcv(grid_info, rcv_info)

    xr_dataset = xr.Dataset(
        data_vars=dict(
            lon_rcv=(["idx_rcv"], rcv_info["lons"]),
            lat_rcv=(["idx_rcv"], rcv_info["lats"]),
            rcv_id=(["idx_rcv"], rcv_info["id"]),
            r_from_rcv=(["idx_rcv", "lat", "lon"], rr_rcv),
        ),
        coords=dict(
            lon=grid_info["lons"],
            lat=grid_info["lats"],
            idx_rcv=np.arange(n_rcv),
        ),
        attrs=dict(
            title=testcase.title,
            description=testcase.desc,
            dx=grid_info["dx"],
            dy=grid_info["dy"],
        ),
    )

    xr_dataset["delay_rcv"] = xr_dataset.r_from_rcv / C0

    # Associate azimuths to grid cells
    set_azimuths(xr_dataset, grid_info, rcv_info, n_rcv)

    # Create arrays to store transfer functions
    all_az = np.unique(xr_dataset.azimuths_rcv.values.flatten())
    xr_dataset.coords["all_az"] = all_az
    # nf = 1000
    tf_dims = ["idx_rcv", "all_az"]
    tf_shape = [xr_dataset.sizes[dim] for dim in tf_dims]
    tf_arr = np.empty(tf_shape)
    xr_dataset["tf"] = (tf_dims, tf_arr)
    xr_dataset.chunk({"idx_rcv": 1})

    zarr_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\saves\test.zarr"
    xr_dataset.to_zarr(zarr_path, compute=False)

    # Build rcv pairs
    build_rcv_pairs(xr_dataset)

    # Set attributes
    set_attrs(xr_dataset, grid_info, testcase)

    return xr_dataset


def init_grid(rcv_info, minimum_distance_around_rcv, dx=100, dy=100):
    """
    Initialize the grid around the source trajectory.

    Parameters
    ----------
    src_info : dict
        Source information.
    grid_info : dict
        Grid information.

    Returns
    -------
    None

    """

    dlon_bathy, dlat_bathy = get_bathy_grid_size(
        lon=rcv_info["lons"][0], lat=rcv_info["lats"][0]
    )
    grid_info = dict(
        dx=dx,
        dy=dy,
        dlat_bathy=dlat_bathy,
        dlon_bathy=dlon_bathy,
    )

    min_lon, max_lon = np.min(rcv_info["lons"]), np.max(rcv_info["lons"])
    min_lat, max_lat = np.min(rcv_info["lats"]), np.max(rcv_info["lats"])
    rcv_barycentre_lon, rcv_barycentre_lat = np.mean(rcv_info["lons"]), np.mean(
        rcv_info["lats"]
    )

    geod = Geod(ellps="WGS84")
    min_lon_grid, _, _ = geod.fwd(
        lons=min_lon,
        lats=rcv_barycentre_lat,
        az=270,
        dist=minimum_distance_around_rcv,
    )
    max_lon_grid, _, _ = geod.fwd(
        lons=max_lon,
        lats=rcv_barycentre_lat,
        az=90,
        dist=minimum_distance_around_rcv,
    )

    _, min_lat_grid, _ = geod.fwd(
        lons=rcv_barycentre_lon,
        lats=min_lat,
        az=180,
        dist=minimum_distance_around_rcv,
    )
    _, max_lat_grid, _ = geod.fwd(
        lons=rcv_barycentre_lon,
        lats=max_lat,
        az=0,
        dist=minimum_distance_around_rcv,
    )

    grid_lons = np.array(
        geod.inv_intermediate(
            lat1=rcv_barycentre_lat,
            lon1=min_lon_grid,
            lat2=rcv_barycentre_lat,
            lon2=max_lon_grid,
            del_s=grid_info["dx"],
        ).lons
    )
    grid_lats = np.array(
        geod.inv_intermediate(
            lat1=min_lat_grid,
            lon1=rcv_barycentre_lon,
            lat2=max_lat_grid,
            lon2=rcv_barycentre_lon,
            del_s=grid_info["dy"],
        ).lats
    )
    grid_info["lons"] = grid_lons
    grid_info["lats"] = grid_lats
    grid_info["min_lat"] = np.min(grid_lats)
    grid_info["max_lat"] = np.max(grid_lats)
    grid_info["min_lon"] = np.min(grid_lons)
    grid_info["max_lon"] = np.max(grid_lons)

    return grid_info


def set_attrs(xr_dataset, grid_info, testcase):
    """
    Add attributes to the dataset.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    grid_info : dict
        Grid information.
    testcase : Testcase object.
        Tescase.

    Returns
    -------
    xr.Dataset
    """
    # Set attributes
    var_unit_mapping = {
        "°": [
            "lon_rcv",
            "lat_rcv",
            "lon",
            "lat",
        ],
        "m": ["r_from_rcv"],
        "": ["idx_rcv"],
        "s": ["delay_rcv"],
    }
    for unit in var_unit_mapping.keys():
        for var in var_unit_mapping[unit]:
            xr_dataset[var].attrs["units"] = unit

    xr_dataset["lon_rcv"].attrs["long_name"] = "Receiver longitude"
    xr_dataset["lat_rcv"].attrs["long_name"] = "Receiver latitude"
    xr_dataset["r_from_rcv"].attrs["long_name"] = "Range from receiver"
    xr_dataset["lon"].attrs["long_name"] = "Longitude"
    xr_dataset["lat"].attrs["long_name"] = "Latitude"
    xr_dataset["idx_rcv"].attrs["long_name"] = "Receiver index"
    xr_dataset["delay_rcv"].attrs["long_name"] = "Propagation delay from receiver"

    # Initialisation time
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    xr_dataset.attrs["init_time"] = now

    xr_dataset.attrs["fullpath_dataset"] = get_dataset_path(grid_info, testcase)

    if not os.path.exists(os.path.dirname(xr_dataset.fullpath_dataset)):
        os.makedirs(os.path.dirname(xr_dataset.fullpath_dataset))

    return xr_dataset


def get_dataset_path(grid_info, testcase):
    """
    Build dataset path.
    Parameters
    ----------
    grid_info : dict
        Grid information.
    testcase : Testcase object.
        Tescase.

    Returns
    -------
    str
        Fullpath to dataset.

    """
    boundaries = "_".join(
        [
            f"{v:.4f}"
            for v in [
                grid_info["min_lon"],
                grid_info["max_lon"],
                grid_info["min_lat"],
                grid_info["max_lat"],
            ]
        ]
    )
    populated_path = os.path.join(
        ROOT_DATASET_PATH,
        testcase.name,
        f"propa_dataset_{boundaries}.zarr",
    )
    return populated_path


if __name__ == "__main__":
    from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos

    rcv_info_dw = {
        "id": ["RR45", "RR48", "RR44"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)
