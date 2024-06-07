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
import dask.array as da

from pyproj import Geod
from localisation.verlinden.plateform.utils import (
    set_attrs,
    init_simu_info_dataset,
    set_simu_unique_id,
)

from propa.kraken_toolbox.utils import get_rcv_pos_idx
from propa.kraken_toolbox.run_kraken import runkraken
from localisation.verlinden.verlinden_utils import (
    get_range_from_rcv,
    set_azimuths,
    build_rcv_pairs,
    get_max_kraken_range,
    get_dist_between_rcv,
    get_bathy_grid_size,
)
from cst import C0


def init_dataset(
    rcv_info,
    testcase,
    minimum_distance_around_rcv=1 * 1e3,
    dx=100,
    dy=100,
    nfft=1024,
    fs=100,
    max_range=None,
):
    """
    Initialize the dataset to be used by build_dataset().

    Parameters
    ----------
    rcv_info : dict
        Receiver information.
    testcase: TestCase
        Testcase.

    Returns
    -------
    xr.Dataset
        Initialized dataset.

    """

    # Define grid
    grid_info = init_grid(rcv_info, minimum_distance_around_rcv, dx, dy)

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
    xr_dataset = set_azimuths(xr_dataset, grid_info, rcv_info, n_rcv)

    # Build rcv pairs
    build_rcv_pairs(xr_dataset)

    # Set attributes
    set_attrs(xr_dataset, grid_info, testcase)

    # Create arrays to store transfer functions
    positive_freq = np.fft.rfftfreq(nfft, 1 / fs)
    xr_dataset.coords["kraken_freq"] = positive_freq

    # Dummy kraken run to get field grid
    if max_range is None:
        max_range = xr_dataset.r_from_rcv.max().values
    testcase_varin = dict(
        max_range_m=max_range,
        azimuth=0,
        rcv_lon=rcv_info["lons"][0],
        rcv_lat=rcv_info["lats"][0],
        freq=[1],
    )
    testcase.update(testcase_varin)

    # Run kraken
    pressure_field, field_pos = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=testcase.env.freq,
        parallel=True,
        verbose=False,
    )


    kraken_range = field_pos["r"]["r"]
    kraken_depth = field_pos["r"]["z"]

    # TODO : pass z_src as param 
    z_src = np.array([5])
    rr, zz, field_pos = get_rcv_pos_idx(shd_fpath=testcase.env.shd_fpath, rcv_depth=z_src, rcv_range=kraken_range)
    pressure_field = pressure_field[:, zz, rr]

    # List of all azimuts
    xr_dataset.coords["all_az"] = np.unique(xr_dataset.azimuths_rcv.values.flatten())
    # kraken grid coords
    xr_dataset.coords["kraken_range"] = kraken_range
    xr_dataset.coords["kraken_depth"] = kraken_depth

    # Chunk dataset 
    # chunk_dict = {}
    # for coord in list(xr_dataset.coords):
    #     chunk_dict[coord] = xr_dataset.sizes[coord]
    # for coord in ["idx_rcv"]:
    #     chunk_dict[coord] = 1

    # xr_dataset = xr_dataset.chunk(chunk_dict)

    # Save dataset before adding tf array 
    xr_dataset.to_zarr(xr_dataset.fullpath_dataset_propa, compute=True, mode="w")

    # Transfert function array
    tf_dims = ["idx_rcv", "all_az", "kraken_freq", "kraken_depth", "kraken_range"]
    tf_shape = [xr_dataset.sizes[dim] for dim in tf_dims]

    # Chunk new variable tf
    tf_chunksize = {}
    for d in ["idx_rcv", "all_az"]:
        tf_chunksize[d] = 1
    for d in tf_dims[2:]:
        tf_chunksize[d] = xr_dataset.sizes[d]

    tf_arr = da.empty(tf_shape, dtype=np.complex64)
    xr_dataset["tf"] = (tf_dims, tf_arr)
    xr_dataset["tf"] = xr_dataset.tf.chunk(tf_chunksize)

    # Store zarr without computing
    xr_dataset.to_zarr(xr_dataset.fullpath_dataset_propa, compute=False, mode="a")

    # # Save simu info in netcdf
    # folder = os.path.dirname(os.path.dirname(xr_dataset.fullpath_dataset_propa))
    # ds_info_path = os.path.join(folder, "simu_index.nc")

    # if os.path.exists(ds_info_path):
    #     # Load ds
    #     ds_info = xr.open_dataset(ds_info_path)

    # else:
    #     # Create ds
    #     ds_info = init_simu_info_dataset()

    # # Add new simu
    # id_to_write_in = (~ds_info.launched).idxmax()
    # ds_info.launched.loc[id_to_write_in] = True
    # ds_info.min_dist.loc[id_to_write_in] = minimum_distance_around_rcv

    # nf = xr_dataset.sizes["kraken_freq"]
    # ds_info.nfreq.loc[id_to_write_in] = nf
    # ds_info.freq.loc[id_to_write_in][:nf] = xr_dataset.kraken_freq.values

    # nr = xr_dataset.sizes["idx_rcv"]
    # ds_info.nrcv.loc[id_to_write_in] = nr
    # ds_info.rcv_id.loc[id_to_write_in][:nr] = xr_dataset.rcv_id.values

    # ds_info.boundaries_label.loc[id_to_write_in] = xr_dataset.attrs["boundaries_label"]
    # set_simu_unique_id(ds_info, id_to_write_in)

    # # Save ds
    # ds_info.to_netcdf(ds_info_path)

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
