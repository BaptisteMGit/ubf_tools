#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   build_dataset.py
@Time    :   2024/04/23 10:35:21
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
import dask.array as da

from time import sleep
from dask.distributed import Client
from propa.kraken_toolbox.run_kraken import runkraken, clear_kraken_parallel_working_dir
from localisation.verlinden.plateform.init_dataset import init_dataset


# ======================================================================================================================
# Functions
# ======================================================================================================================
def build_dataset(
    rcv_info,
    testcase,
):
    """
    Build the dataset to be used by the localisation algorithm.

    Returns
    -------
    xr.Dataset
    """
    client = Client(n_workers=6, threads_per_worker=1)
    print(client.dashboard_link)
    # Create zarr
    ds = init_dataset(
        rcv_info,
        testcase,
    )

    print(f"Dataset sizes : {ds.sizes}")
    clear_kraken_parallel_working_dir(root=testcase.env.root)
    rmax = ds.r_from_rcv.max().values
    lon_rcv = ds.lon_rcv.values
    lat_rcv = ds.lat_rcv.values
    all_az = ds.all_az.values
    freq = ds.kraken_freq.values

    # Looop over dataset regions
    nregion = 10
    # regions are defined by azimuth slices
    az_limits = np.linspace(0, ds.sizes["all_az"], nregion, dtype=int)
    regions_az_slices = [
        slice(az_limits[i], az_limits[i + 1]) for i in range(len(az_limits) - 1)
    ]

    for r_az_slice in regions_az_slices:
        region_ds = ds.isel(all_az=r_az_slice)

        # Compute transfer functions (tf) using `map_blocks`
        rmax = 10000

        array_template = region_ds.tf.isel(idx_rcv=0, all_az=0).data
        tf_dask = region_ds.tf.data

        region_tf = tf_dask.map_blocks(
            compute_tf_chunk_dask,
            testcase,
            rmax,
            lon_rcv,
            lat_rcv,
            all_az,
            freq,
            meta=array_template,
        )
        region_tf_data = region_tf.compute()

        ds.tf[dict(all_az=r_az_slice)] = region_tf_data
        region_to_save = ds.tf[dict(all_az=r_az_slice)]

        region_to_save.to_zarr(
            ds.fullpath_dataset,
            mode="r+",
            region={
                "idx_rcv": slice(0, ds.sizes["idx_rcv"]),
                "all_az": r_az_slice,
                "kraken_freq": slice(0, ds.sizes["kraken_freq"]),
                "kraken_depth": slice(0, ds.sizes["kraken_depth"]),
                "kraken_range": slice(0, ds.sizes["kraken_range"]),
            },
        )
        sleep(1)

    client.close()


def compute_tf_chunk_dask(
    chunk, testcase, rmax, lon_rcv, lat_rcv, all_az, freq, block_info=None
):
    # print(block_info[0]["array-location"])
    i_rcv = block_info[0]["array-location"][0][0]
    i_az = block_info[0]["array-location"][1][0]
    testcase_varin = dict(
        max_range_m=rmax,
        azimuth=all_az[i_az],
        rcv_lon=lon_rcv[i_rcv],
        rcv_lat=lat_rcv[i_rcv],
        freq=freq,
        called_by_subprocess=True,
    )
    testcase.update(testcase_varin)

    # Run kraken
    tf_chunk, _ = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=testcase.env.freq,
        parallel=False,
        verbose=False,
        clear=False,
    )

    dask_tf_chunk = da.from_array(
        np.squeeze(tf_chunk, (1, 2)), chunks=block_info[None]["chunk-shape"][2:]
    )
    return dask_tf_chunk


def compute_tf_chunk(chunk, testcase, rmax, lon_rcv, lat_rcv):
    # Assuming `external_function` is the function you want to call for each chunk
    # This function might be replaced with your actual computation logic
    # print(chunk.sizes)
    # print(chunk.coords)
    testcase_varin = dict(
        # max_range_m=chunk.r_from_rcv.max().values,
        max_range_m=rmax,  # TODO
        azimuth=chunk.all_az.values[0],
        rcv_lon=lon_rcv[chunk.idx_rcv],
        rcv_lat=lat_rcv[chunk.idx_rcv],
        freq=chunk.kraken_freq.values,
    )
    testcase.update(testcase_varin)

    # Run kraken
    tf_chunk, field_pos = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=testcase.env.freq,
        parallel=False,
        verbose=False,
    )

    ds_tf_chunk = chunk
    ds_tf_chunk[dict(idx_rcv=0, all_az=0)] = np.squeeze(tf_chunk, (1, 2))
    # return da.from_array(tf_chunk, chunks=tf_chunk.shape)
    return ds_tf_chunk


if __name__ == "__main__":
    from localisation.verlinden.testcases.testcase_envs import TestCase3_1
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

    initial_ship_pos_dw = {
        "lon": rcv_info_dw["lons"][0],
        "lat": rcv_info_dw["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    tc = TestCase3_1()

    build_dataset(rcv_info=rcv_info_dw, testcase=tc)

    # path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1\propa_dataset_65.4003_66.1446_-27.8377_-27.3528.zarr"
    # ds = xr.open_dataset(path, engine="zarr")
    # print(ds)
