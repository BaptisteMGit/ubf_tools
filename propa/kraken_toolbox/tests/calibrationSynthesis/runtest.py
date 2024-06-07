#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run.py
@Time    :   2024/03/18 08:59:34
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
import matplotlib.pyplot as plt

from signals import pulse
from propa.kraken_toolbox.plot_utils import plotmode
from propa.kraken_toolbox.run_kraken import runkraken
from localisation.verlinden.AcousticComponent import AcousticSource
from localisation.verlinden.testcases.testcase_envs import TestCase
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal_from_broadband_pressure_field,
)

ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\calibrationSynthesis"


class CalibTestCase(TestCase):
    def __init__(self, src):

        name = "calib_fourier_synthesis"
        testcase_varin = {"freq": src.kraken_freq}
        title = "Calibration test case"
        desc = "Fourier synthesis processor calibration test case"
        mode = "prod"
        super().__init__(name, testcase_varin, title, desc, mode)

        self.env_dir = ROOT
        self.isotropic = True
        self.src_depth = 25

        # Config : Computational Ocean Acoustics
        self.flp_n_rcv_z = 9
        self.flp_rcv_z_min = 10
        self.flp_rcv_z_max = 90

        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 5,
            "nb_modes": 100,
            "mode_addition": "coupled",
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value

        self.process()


if __name__ == "__main__":

    run_k = True

    # Source
    T = 7.2
    fc = 50
    fs = 200
    s, t = pulse(T=T, f=fc, fs=fs)
    src = AcousticSource(
        signal=s,
        time=t,
        name="Pulse",
        waveguide_depth=100,
        window="hanning",
        nfft=2 ** int(np.log2(s.size) + 1),
    )

    calib = CalibTestCase(src)
    calib.env.plot_env(plot_src=True, src_depth=calib.src_depth)
    plt.ylim([110, 0])
    plt.plot()
    plt.savefig(os.path.join(ROOT, "calib_medium.png"))

    z_src = calib.src_depth
    c0 = 1500
    rcv_range = np.array([30000])  # 30km
    # rcv_depth = np.array([z_src + i * 10 for i in range(-10, 10)])
    rcv_depth = np.linspace(calib.flp_rcv_z_min, calib.flp_rcv_z_max, calib.flp_n_rcv_z)
    # rcv_depth = np.array([z_src + i * 10 for i in [-10, -5, 0, 5, 10]])

    delays = rcv_range / c0

    if run_k:
        pf, field_pos = runkraken(
            env=calib.env, flp=calib.flp, frequencies=src.kraken_freq
        )

        # Synthesis
        (
            t_obs,
            s_obs,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=calib.env.shd_fpath,
            broadband_pressure_field=pf,
            frequencies=src.kraken_freq,
            source=src,
            rcv_range=rcv_range,
            rcv_depth=rcv_depth,
            apply_delay=True,
            delay=delays,
            minimum_waveguide_depth=100,
        )

        ds = xr.Dataset(
            {
                "s_at_rcv_pos": (
                    ["time", "rcv_depth", "rcv_range"],
                    s_obs.astype(np.float32),
                ),
            },
            coords={
                "time": t_obs.astype(np.float32),
                "rcv_depth": rcv_depth,
                "rcv_range": rcv_range,
            },
        )

        ds.to_netcdf(os.path.join(ROOT, "rcv_sig.nc"))

    else:
        # Load rcv sig
        ds = xr.open_dataset(os.path.join(ROOT, "rcv_sig.nc"))

    # Plot
    # rcv_depth = np.array([z_src + i * 10 for i in [-10, -5, 0, 5, 10]])

    # Scale to unity
    max_amplitude = ds.s_at_rcv_pos.max().values
    ds["s_at_rcv_pos"] = ds.s_at_rcv_pos / max_amplitude

    # z_offset = max_amplitude + 1e-5
    z_offset = 1.5
    for ir, r in enumerate(rcv_range):
        plt.figure(figsize=(8, 12))
        for iz, z in enumerate(rcv_depth):
            sig = (
                ds.s_at_rcv_pos.sel(rcv_depth=z, rcv_range=r, method="nearest")
                - iz * z_offset
            )
            if z == z_src:
                sig.plot(color="r", label="Source depth")
            else:
                sig.plot(color="k")

        depth_label = [f"{z:.0f}" for z in ds.rcv_depth.values]
        depth_pos = [-iz * z_offset for iz in range(ds.dims["rcv_depth"])]
        ax = plt.gca()
        ax.set_yticks(depth_pos[::2])
        ax.set_yticklabels(depth_label[::2])
        plt.xlim([0, 1])
        plt.xlabel(f"Time [s] - r/{c0}")
        plt.ylabel("Depth [m]")
        plt.title(
            f"Received signal (r={r}m)\n",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, f"rcv_sig_r{r}m.png"))

    # Plot modes
    plotmode(calib.env.env_fpath, freq=50, modes=[1, 2])
    f = plt.gcf()
    f.set_size_inches(8, 12)
    plt.savefig(os.path.join(ROOT, f"modes.png"))
