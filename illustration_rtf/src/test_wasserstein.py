#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_wasserstein.py
@Time    :   2026/09/18 09:57:19
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance


def calc_rtf_wasserstein_dist(gamma_a, gamma_b):
    """Wasserstein (earth mover's) distance between two RTF magnitude
    spectra, each treated as a 1D distribution over FREQUENCY-BIN
    INDEX (not the physical frequency values -- both curves share the
    same frequency grid, so the bin index alone is enough to compare
    them; see scipy.stats.wasserstein_distance()'s own 'u_values'/
    'v_values', here just np.arange(n_freq) for both curves). Only the
    RTF MAGNITUDE at each bin -- used as that bin's WEIGHT -- differs
    between the two.

    NOTE: scipy.stats.wasserstein_distance() only accepts 1D inputs --
    'gamma_b' can carry several swept values at once (each compared
    against the same baseline 'gamma_a'), so this loops over that
    dimension internally rather than vectorizing (there is no
    vectorized wasserstein_distance in scipy as of this writing).

    Args:
        gamma_a (np.ndarray): baseline gamma, shape (n_freq, 1) (a
            single reference curve, broadcast against every swept
            value below).
        gamma_b (np.ndarray): swept-value(s) gamma, shape
            (n_freq, n_values).

    Returns:
        np.ndarray: shape (n_values,) -- one Wasserstein distance per
        swept value.
    """

    # rtf_a = 10 ** (gamma_a / 20.0)
    # rtf_b = 10 ** (gamma_b / 20.0)
    water_level = min(np.min(gamma_a), np.min(gamma_b)) + 1e-5
    rtf_a = gamma_a + water_level
    rtf_b = gamma_b + water_level
    rtf_a = np.broadcast_to(rtf_a, rtf_b.shape)

    n_freq, n_values = rtf_b.shape
    distribution_support = np.arange(n_freq)

    dist = np.empty(n_values)
    for i in range(n_values):
        u = np.nan_to_num(rtf_a[:, i], nan=0.0)
        v = np.nan_to_num(rtf_b[:, i], nan=0.0)
        dist[i] = wasserstein_distance(
            u_values=distribution_support,
            v_values=distribution_support,
            u_weights=u,
            v_weights=v,
        )
    return dist


r0 = 15 * 1e3
d12 = 1000


def build_gamma(ds, r0=r0, d12=d12):
    g_fr_1 = ds.gf.sel(r=r0, method="nearest")
    g_fr_2 = ds.gf.sel(r=r0 + d12, method="nearest")
    gamma_r0 = 20 * np.log10(np.abs(g_fr_2.values / g_fr_1.values))
    return gamma_r0


# Import SW
fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\resilience\result\sw\depth\depth\depth_0001.nc"
ds_sw_1 = xr.open_dataset(fpath)
gamma_sw_1 = build_gamma(ds=ds_sw_1.isel(depth=0))

fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\resilience\result\sw\depth\depth\depth_0010.nc"
ds_sw_2 = xr.open_dataset(fpath)
gamma_sw_2 = build_gamma(ds=ds_sw_2.isel(depth=0))

# Import DW
fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\resilience\result\dw\depth\depth\depth_0001.nc"
ds_dw_1 = xr.open_dataset(fpath)
gamma_dw_1 = build_gamma(ds=ds_dw_1.isel(depth=0))

fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\resilience\result\dw\depth\depth\depth_0010.nc"
ds_dw_2 = xr.open_dataset(fpath)
gamma_dw_2 = build_gamma(ds=ds_dw_2.isel(depth=0))


# Visualize
f = ds_sw_1.f.values
fig, axs = plt.subplots(2, 1, figsize=(16, 8))
axs[0].plot(f, gamma_sw_1, label=f"D = {ds_sw_1.depth.values} m")
axs[0].plot(f, gamma_sw_2, label=f"D = {ds_sw_2.depth.values} m")
axs[1].plot(f, gamma_dw_1, label=f"D = {ds_dw_1.depth.values} m")
axs[1].plot(f, gamma_dw_2, label=f"D = {ds_dw_2.depth.values} m")

axs[0].legend()
axs[1].legend()

# plt.savefig("test")

# Step 2 : transform into positive only (distribution like)


def shift_to_positive(gamma_a, gamma_b):
    water_level = min(np.nanmin(gamma_a), np.nanmin(gamma_b)) + 1e-5
    gamma_a_shift = gamma_a + water_level
    gamma_b_shift = gamma_b + water_level
    gamma_a_shift = np.broadcast_to(gamma_a_shift, gamma_b_shift.shape)

    return gamma_a_shift, gamma_b_shift


gamma_sw_1_shift, gamma_sw_2_shift = shift_to_positive(
    gamma_a=gamma_sw_1, gamma_b=gamma_sw_2
)
gamma_dw_1_shift, gamma_dw_2_shift = shift_to_positive(
    gamma_a=gamma_dw_1, gamma_b=gamma_dw_2
)

fig, axs = plt.subplots(2, 1, figsize=(16, 8))
axs[0].plot(f, gamma_sw_1_shift, label=f"D = {ds_sw_1.depth.values} m")
axs[0].plot(f, gamma_sw_2_shift, label=f"D = {ds_sw_2.depth.values} m")
axs[1].plot(f, gamma_dw_1_shift, label=f"D = {ds_dw_1.depth.values} m")
axs[1].plot(f, gamma_dw_2_shift, label=f"D = {ds_dw_2.depth.values} m")

axs[0].legend()
axs[1].legend()

plt.savefig("test")

print()
