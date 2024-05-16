#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_plateform.py
@Time    :   2024/05/03 11:32:50
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import xarray as xr

from signals import pulse
from localisation.verlinden.plateform.utils import *
from localisation.verlinden.plateform.init_dataset import init_grid
from localisation.verlinden.AcousticComponent import AcousticSource
from localisation.verlinden.plateform.build_dataset import build_dataset
from localisation.verlinden.plateform.populate_dataset import (
    populate_dataset,
    grid_synthesis,
)
from localisation.verlinden.testcases.testcase_envs import TestCase3_1
from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos


def run_plateform_test():
    """
    Run the plateform test.
    """

    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        # "id": ["RR45", "RR48"],
        "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)

    tc = TestCase3_1()
    min_dist = 15 * 1e3
    dx, dy = 100, 100

    # Define source signal
    dt = 1
    min_waveguide_depth = 5000
    f0, fs = 5, 10
    nfft = int(fs * dt)
    src_sig, t_src_sig = pulse(T=dt, f=f0, fs=fs)

    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="debug_pulse",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    fullpath_dataset = build_dataset(
        rcv_info=rcv_info_dw,
        testcase=tc,
        minimum_distance_around_rcv=min_dist,
        dx=dx,
        dy=dy,
        nfft=src.nfft,
        fs=fs,
    )

    # Load dataset
    ds = xr.open_dataset(fullpath_dataset, engine="zarr", chunks={})

    # Get src label
    ds.attrs["src_label"] = build_src_label(src_name=src.name)

    # Populate dataset
    ds = populate_dataset(ds, src, rcv_info=rcv_info_dw)

    print("Done")

    return ds


def run_on_plateform(rcv_info, testcase, min_dist, dx, dy, src):

    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)

    grid_info = init_grid(rcv_info, min_dist, dx, dy)
    boundaries_label = build_boundaries_label(grid_info)
    fullpath_dataset_propa = build_propa_path(testcase.name, boundaries_label)

    root_dir = build_root_dir(testcase.name)
    grid_label = build_grid_label(dx, dy)
    fullpath_dataset_propa_grid = build_propa_grid_path(
        root_dir,
        boundaries_label, grid_label
    )
    
    if not os.path.exists(fullpath_dataset_propa):
        steps = [0, 1]  # All steps required
    else:
        ds = xr.open_dataset(fullpath_dataset_propa, engine="zarr", chunks={})
        if ds.propa_done:
            print(f"Propa dataset already exists at {fullpath_dataset_propa}")


            if not os.path.exists(fullpath_dataset_propa_grid):
                steps = [1]  # Gridding and synthesis required
            else:
                ds = xr.open_dataset(
                    fullpath_dataset_propa_grid, engine="zarr", chunks={}
                )
                if ds.propa_grid_done:
                    print(
                        f"Grid dataset already exists at {fullpath_dataset_propa_grid}"
                    )
                    steps = [2]  # Only synthesis required
                else:
                    steps = [1]  # Gridding and synthesis required

        else:
            steps = [0, 1]

    if 0 in steps:
        print("Step 0")
        build_dataset(
            rcv_info=rcv_info,
            testcase=testcase,
            minimum_distance_around_rcv=min_dist,
            dx=dx,
            dy=dy,
            nfft=src.nfft,
            fs=src.fs,
        )
    if 1 in steps:
        print("Step 1")
        ds = xr.open_dataset(fullpath_dataset_propa, engine="zarr", chunks={})
        ds = populate_dataset(
            ds,
            src,
            rcv_info=rcv_info,
            dx=dx,
            dy=dy,
        )
        fullpath_dataset_propa_grid_src = ds.fullpath_dataset_propa_grid_src

    if 2 in steps:
        print("Step 2")
        ds = xr.open_dataset(fullpath_dataset_propa_grid, engine="zarr", chunks={})
        ds = grid_synthesis(ds, src)
        fullpath_dataset_propa_grid_src = ds.fullpath_dataset_propa_grid_src

    return (
        fullpath_dataset_propa,
        fullpath_dataset_propa_grid,
        fullpath_dataset_propa_grid_src,
    )


if __name__ == "__main__":
    import numpy as np

    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        # "id": ["RR45", "RR48"],
        "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }
    tc = TestCase3_1()
    min_dist = 15 * 1e3
    dx, dy = 100, 100

    # Define source signal
    dt = 1
    min_waveguide_depth = 5000
    f0, fs = 5, 10
    nfft = int(fs * dt)
    src_sig, t_src_sig = pulse(T=dt, f=f0, fs=fs)

    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="debug_pulse",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    # src_sig = src_sig + 10 * np.random.randn(*src_sig.shape)
    src_sig = np.ones_like(src_sig)
    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="debug_pulse",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )
    src.display_source()

    run_on_plateform(
        rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
    )
    # ds = run_plateform_test()

    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # from localisation.verlinden.plateform.populate_dataset import grid_synthesis

    # # fname = "propa_grid_65.0965_66.4484_-28.1084_-27.0821_100_100.zarr"
    # # froot = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1\propa_grid"
    # # fpath = os.path.join(froot, fname)
    # # ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    # # # Define source signal
    # # dt = 1
    # # min_waveguide_depth = 5000
    # # f0, fs = 10, 20
    # # nfft = int(fs * dt)
    # # src_sig, t_src_sig = pulse(T=dt, f=f0, fs=fs)

    # # src = AcousticSource(
    # #     signal=src_sig,
    # #     time=t_src_sig,
    # #     name="debug_pulse",
    # #     waveguide_depth=min_waveguide_depth,
    # #     nfft=nfft,
    # # )

    # # # Grid synthesis
    # # ds = grid_synthesis(ds, src)

    # fname = "propa_grid_src_65.5973_65.8993_-27.6673_-27.3979_100_100_debug_pulse.zarr"
    # froot = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1\propa_grid_src"
    # fpath = os.path.join(froot, fname)
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    # # Plot
    # f = 2.5
    # # lon, lat = np.mean(ds.lon_rcv.values[1:]), np.mean(ds.lat_rcv.values[1:])
    # lon, lat = 65.3, -27.17
    # for i in range(ds.sizes["idx_rcv"]):
    #     to_plot = 10 * np.log10(
    #         np.abs(ds.tf_gridded.isel(idx_rcv=i).sel(kraken_freq=f))
    #     )
    #     plt.figure()
    #     to_plot.plot(cmap="jet", vmin=-120, vmax=-40)
    #     plt.scatter(lon, lat, color="red", label="src_pos")
    #     for ii in range(ds.sizes["idx_rcv"]):
    #         plt.scatter(
    #             ds.lon_rcv.isel(idx_rcv=ii).values,
    #             ds.lat_rcv.isel(idx_rcv=ii).values,
    #             label=f"rcv_{ii}",
    #         )

    #     plt.title(f"Transfer function at {f} Hz")
    #     plt.legend()

    # plt.figure()
    # for i in range(ds.sizes["idx_rcv"]):
    #     ds.rcv_signal_library.isel(idx_rcv=i).sel(
    #         lon=lon, lat=lat, method="nearest"
    #     ).plot(label=f"rcv_{i}")
    # plt.title(f"Receiver signal at lon={lon}, lat={lat}")
    # plt.legend()
    # plt.show()

    # print(ds)
