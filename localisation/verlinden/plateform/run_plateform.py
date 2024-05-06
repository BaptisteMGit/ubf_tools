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
from localisation.verlinden.AcousticComponent import AcousticSource
from localisation.verlinden.plateform.build_dataset import build_dataset
from localisation.verlinden.plateform.populate_dataset import populate_dataset


def run_plateform_test():
    """
    Run the plateform test.
    """
    from localisation.verlinden.testcases.testcase_envs import TestCase3_1
    from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos

    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        "id": ["RR45", "RR48"],
        # "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)

    tc = TestCase3_1()
    min_dist = 2 * 1e3
    dx, dy = 100, 100

    # Define source signal
    dt = 1
    min_waveguide_depth = 5000
    f0, fs = 10, 20
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

    # Populate dataset
    ds = populate_dataset(ds, src, rcv_info=rcv_info_dw)

    print("Done")

    return ds


if __name__ == "__main__":
    # ds = run_plateform_test()

    import os

    fname = "propa_dataset_65.5827_65.9622_-27.6752_-27.5621.zarr"
    froot = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1"
    fpath = os.path.join(froot, fname)
    ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    print(ds)
