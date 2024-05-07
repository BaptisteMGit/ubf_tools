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
from signals import pulse
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
        "id": ["RR45", "RR48", "RR44"],
        "lons": [],
        "lats": [],
    }
    tc = TestCase3_1()
    min_dist = 50 * 1e3
    dx, dy = 100, 100

    # Define source signal
    dt = 7
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


if __name__ == "__main__":

    test()
