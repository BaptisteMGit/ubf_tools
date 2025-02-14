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
import os
import gc
import zarr
import numpy as np
import dask.array as da

from time import sleep, time
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from propa.kraken_toolbox.run_kraken import runkraken, clear_kraken_parallel_working_dir
from localisation.verlinden.plateform.init_dataset import init_dataset
from propa.kraken_toolbox.utils import waveguide_cutoff_freq

from localisation.verlinden.plateform.utils import update_info_status
from localisation.verlinden.plateform.params import N_WORKERS, MAX_RAM_GB


# ======================================================================================================================
# Functions
# ======================================================================================================================
def build_dataset(
    rcv_info,
    testcase,
    **kwargs,
):
    """
    Build the dataset to be used by the localisation algorithm.

    Returns
    -------
    xr.Dataset
    """
    print("Building dataset.....")

    # Create zarr
    ds = init_dataset(
        rcv_info,
        testcase,
        **kwargs,
    )

    print(f"Dataset sizes : {ds.sizes}")
    print(f"Dataset memory size = {ds.nbytes * 1e-9} GB")
    clear_kraken_parallel_working_dir(root=testcase.env.root)

    # Params common to regions
    rmax = ds.r_from_rcv.max().values
    lon_rcv = ds.lon_rcv.values
    lat_rcv = ds.lat_rcv.values
    all_az = ds.all_az.values
    freq = ds.kraken_freq.values

    # Loop over dataset regions
    nregion_workers = int(
        np.ceil(((ds.sizes["all_az"] * ds.sizes["idx_rcv"]) / N_WORKERS))
    )

    nregion_memory = ds.sizes["all_az"]
    max_size_bytes = 1 * 1e9  # 1 Go
    size = ds.tf.nbytes / nregion_memory
    while size <= max_size_bytes and nregion_memory > 1:  # At least 1 region
        nregion_memory -= 1
        size = ds.tf.nbytes / nregion_memory

    nregion = max(nregion_memory, nregion_workers)

    # Regions are defined by azimuth slices
    az_limits = np.linspace(0, ds.sizes["all_az"], nregion + 1, dtype=int)
    regions_az_slices = [
        slice(az_limits[i], az_limits[i + 1]) for i in range(len(az_limits) - 1)
    ]
    n_az_max = max([az_limits[i + 1] - az_limits[i] for i in range(len(az_limits) - 1)])
    n_eff_workers = ds.sizes["idx_rcv"] * n_az_max
    size = ds.tf.nbytes / nregion

    if not testcase.isotropic:
        print(f"Start building propa dataset using {n_eff_workers} workers")
        with Client(n_workers=n_eff_workers, threads_per_worker=1) as client:
            print(client.dashboard_link)

            for r_az_slice in regions_az_slices:

                t0 = time()
                region_ds = ds.isel(all_az=r_az_slice)

                # Compute transfer functions (tf) using `map_blocks`
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

                # with ProgressBar():
                region_tf_data = region_tf.compute()

                ds.tf[dict(all_az=r_az_slice)] = region_tf_data
                region_to_save = ds.tf[dict(all_az=r_az_slice)]

                print(f"Region ({size * 1e-9} Go) processed in {time() - t0:.2f} s")
                t1 = time()
                region_to_save.to_zarr(
                    ds.fullpath_dataset_propa,
                    mode="r+",
                    region={
                        "idx_rcv": slice(None),
                        "all_az": r_az_slice,
                        "kraken_freq": slice(None),
                        "kraken_depth": slice(None),
                        "kraken_range": slice(None),
                    },
                )

                print(f"Region ({size * 1e-9} Go) saved in {time() - t1:.2f} s")

                # Ensure memory is freed
                del region_ds, region_tf, region_tf_data, region_to_save
                gc.collect()

                sleep(1)  # Seems to solve unstable behavior ...

    else:
        for i_rcv in ds.idx_rcv.values:
            # Propagating freq
            fc = waveguide_cutoff_freq(waveguide_depth=testcase.min_depth) + 0.1
            f = freq[freq > fc]

            testcase_varin = dict(
                max_range_m=rmax,
                rcv_lon=lon_rcv[i_rcv],
                rcv_lat=lat_rcv[i_rcv],
                freq=f,
                called_by_subprocess=True,
            )
            testcase.update(testcase_varin)

            # Run kraken
            tf_prof, field_pos = runkraken(
                env=testcase.env,
                flp=testcase.flp,
                frequencies=f,
                parallel=testcase.run_parallel,
                verbose=False,
                clear=False,
            )

            if freq.size > f.size:
                # Add zeros to match original freq size
                tf_prof = np.concatenate(
                    [
                        tf_prof,
                        np.zeros(
                            (freq.size - f.size, *tf_prof.shape[1:]), dtype=complex
                        ),
                    ]
                )
            for i_az in range(len(all_az)):
                ds["tf"][dict(idx_rcv=i_rcv, all_az=i_az)] = tf_prof.squeeze(((1, 2)))

        region_to_save = ds.tf
        region_to_save.to_zarr(
            ds.fullpath_dataset_propa,
            mode="r+",
            region={
                "idx_rcv": slice(None),
                "all_az": slice(None),
                "kraken_freq": slice(None),
                "kraken_depth": slice(None),
                "kraken_range": slice(None),
            },
        )

    # Open the Zarr store and update attrs
    zarr_store = zarr.open(ds.fullpath_dataset_propa)
    zarr_store.attrs.update({"propa_done": True})
    zarr.consolidate_metadata(ds.fullpath_dataset_propa)

    print("..................Done")

    return ds.fullpath_dataset_propa


def compute_tf_chunk_dask(
    chunk, testcase, rmax, lon_rcv, lat_rcv, all_az, freq, block_info=None
):
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

    # Propagating freq
    fc = waveguide_cutoff_freq(waveguide_depth=testcase.min_depth) + 0.1
    f = freq[freq > fc]

    # Run kraken
    tf_chunk, field_pos = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=f,
        parallel=testcase.run_parallel,
        verbose=False,
        clear=False,
    )

    if freq.size > f.size:
        # Add zeros to match original freq size
        tf_chunk = np.concatenate(
            [
                tf_chunk,
                np.zeros((freq.size - f.size, *tf_chunk.shape[1:]), dtype=complex),
            ]
        )

        dask_tf_chunk = da.from_array(
            np.squeeze(tf_chunk, (1, 2)), chunks=block_info[None]["chunk-shape"][2:]
        )

    return dask_tf_chunk


if __name__ == "__main__":
    # import xarray as xr
    from localisation.verlinden.testcases.testcase_envs import TestCase3_1
    from localisation.verlinden.misc.verlinden_utils import load_rhumrum_obs_pos

    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        "id": ["RR45", "RR48"],
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
