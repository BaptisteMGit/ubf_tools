#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   runtest.py
@Time    :   2024/06/12 14:46:12
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

from misc import mult_along_axis
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.utils import waveguide_cutoff_freq
from localisation.verlinden.testcases.testcase_envs import TestCase3_1, TestCase1_0
from localisation.verlinden.misc.AcousticComponent import AcousticSource


def build_pressure_field(nf):
    f = np.linspace(0, 50, nf)

    lambda_min = 1500 / max(f)
    d_inter_rcv = lambda_min / 4  # d < lambda / 2
    nrcv = 16
    z_min = 100
    z_max = z_min + nrcv * d_inter_rcv

    tc_varin = {
        "freq": f,
        "max_range_m": 60 * 1e3,
        "azimuth": 0,
        "rcv_lon": 65.943,
        "rcv_lat": -27.5792,
        "mode_theory": "coupled",
        "flp_n_rcv_z": 16,  # ALMA
        "flp_rcv_z_min": z_min,
        "flp_rcv_z_max": z_max,
        "min_depth": 300,
    }
    # tc = TestCase3_1(mode="prod", testcase_varin=tc_varin)
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)

    fc = waveguide_cutoff_freq(waveguide_depth=tc.min_depth) + 0.1
    f = f[f > fc]

    u_varin = {"freq": f}
    tc.update(u_varin)

    tc.env.write_env()
    tc.flp.write_flp()

    # tc.env.write_envt
    pressure_field, field_pos = runkraken(
        env=tc.env,
        flp=tc.flp,
        frequencies=tc.env.freq,
        parallel=True,
        verbose=False,
    )

    p = pressure_field.squeeze((1, 2))
    ds = xr.Dataset(
        data_vars=dict(
            press_real=(["f", "z", "r"], np.real(p)),
            press_img=(["f", "z", "r"], np.imag(p)),
        ),
        coords=dict(
            f=list(f),
            z=field_pos["r"]["z"],
            r=field_pos["r"]["r"],
        ),
    )

    ds.to_netcdf(os.path.join(root, "pressure.nc"))

    return pressure_field, field_pos


def compute_ri(p):
    C0 = 1500
    nfft_inv = 6 * p.sizes["f"]
    k0 = 2 * np.pi * p.f.values / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    ri_f = mult_along_axis(p, norm_factor, axis=0)
    ri_t = np.fft.irfft(ri_f, axis=0, n=nfft_inv)

    # Build data set
    T_tot = 1 / (p.f.values[1] - p.f.values[0])
    dt = T_tot / nfft_inv
    t = np.arange(0, T_tot, dt)

    ri_t = xr.Dataset(
        data_vars=dict(
            ri_t=(["t", "z", "r"], ri_t),
        ),
        coords=dict(
            t=t,
            z=p.z,
            r=p.r,
        ),
    )

    return ri_t


def plot_ri(ri_t, r):
    ri_t_plot = ri_t.ri_t.sel(r=r, method="nearest")
    # Subplots
    f, axs = plt.subplots(3, 1, sharex=True)
    for ii, iz in enumerate([2, 4, 6]):
        # axs[ii].set_title(f"{}")
        ri_t_plot.isel(z=iz).plot(ax=axs[ii])
    plt.savefig(os.path.join(root, "ri_subplots.png"))
    # plt.show()

    # Matrix
    tau = ri_t_plot.r.values / 1500 % ri_t.t.max()
    ri_t_plot = ri_t_plot[np.logical_and(ri_t_plot.t > tau, ri_t_plot.t < tau + 0.5)]
    ri_t_plot = ri_t_plot / ri_t_plot.max()
    # ri_t.isel()
    plt.figure()
    ri_ = 20 * np.log10(np.abs(ri_t_plot))
    ri_.plot(x="t", y="z", cmap="gray", yincrease=False)
    plt.savefig(os.path.join(root, f"ri_z_t_r{r}.png"))
    # plt.show()


if __name__ == "__main__":

    nf = 513
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\test_RI"

    # Build
    # build_pressure_field(nf)

    # Load pressure field
    ds = xr.load_dataset(os.path.join(root, "pressure.nc"))

    p = ds.press_real + 1j * ds.press_img

    ri_t = compute_ri(p)

    # Plot
    # r_target = 10 * 1e3  # 10 km
    for r_target in [10, 20, 30, 40, 50]:
        plot_ri(ri_t, r=r_target * 1e3)
