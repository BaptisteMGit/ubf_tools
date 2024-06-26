#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   to_run.py
@Time    :   2024/05/07 11:31:36
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
from signals import pulse, generate_ship_signal
from localisation.verlinden.AcousticComponent import AcousticSource
from localisation.verlinden.testcases.testcase_envs import TestCase3_1
from localisation.verlinden.plateform.run_plateform import run_on_plateform


def test():
    rcv_info_dw = {
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

    (
        fullpath_dataset_propa,
        fullpath_dataset_propa_grid,
        fullpath_dataset_propa_grid_sr,
    ) = run_on_plateform(
        rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
    )


def run_swir():
    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        # "id": ["RRpftim0", "RRpftim1", "RRpftim2"],
        "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }
    tc = TestCase3_1()
    min_dist = 22 * 1e3
    dx, dy = 100, 100

    # Define source signal
    min_waveguide_depth = 5000
    dt = 10
    fs = 100  # Sampling frequency
    f0_lib = 1  # Fundamental frequency of the ship signal
    src_info = {
        "sig_type": "ship",
        "f0": f0_lib,
        "std_fi": f0_lib * 1 / 100,
        "tau_corr_fi": 1 / f0_lib,
        "fs": fs,
    }
    src_sig, t_src_sig = generate_ship_signal(
        Ttot=dt,
        f0=src_info["f0"],
        std_fi=src_info["std_fi"],
        tau_corr_fi=src_info["tau_corr_fi"],
        fs=src_info["fs"],
    )

    src_sig *= np.hanning(len(src_sig))
    # nfft = None
    nfft = 2**3
    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="ship",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    print(f"nfft = {src.nfft}")
    (
        fullpath_dataset_propa,
        fullpath_dataset_propa_grid,
        fullpath_dataset_propa_grid_sr,
    ) = run_on_plateform(
        rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
    )


if __name__ == "__main__":

    run_swir()
    # test()
