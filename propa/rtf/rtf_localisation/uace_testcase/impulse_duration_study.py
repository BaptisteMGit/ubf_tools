#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   impulse_duration_study.py
@Time    :   2025/05/15 11:45:23
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Study impulse duration distribution to choose appropriate frame length for RTF estimation
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import xarray as xr

from misc import cast_matrix_to_target_shape

root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\uace_testcase\data\dw_real_env_impulsive_response"
fname = "library_dx20m_dy20m.nc"
fpath = os.path.join(root, fname)

ds = xr.open_dataset(fpath)
iy = np.random.randint(0, ds.sizes["y"], size=15)
ix = np.random.randint(0, ds.sizes["x"], size=15)
ds = ds.isel(x=ix, y=iy)

# sl_roll_var = ds.s_l.rolling(t=20, center=True).var()
# sl_roll_var = sl_roll_var / sl_roll_var.max()
# flat_shape = ds.sizes["idx_rcv"] * ds.sizes["x"] * ds.sizes["y"]
# var_flat = sl_roll_var.values.reshape(ds.sizes["t"], flat_shape)

# # Single plot
# plt.figure()
# sl_roll_var.isel(idx_rcv=0, y=0, x=0).plot()
# plt.imshow(var_flat.T)
# # plt.show()

# plt.figure()
# sl_roll_var.isel(idx_rcv=0, y=0).plot(x="t")

# var = sl_roll_var.isel(idx_rcv=0)
# var_flat = np.vstack([var.sel(x=x, y=y) for x in ds.x for y in ds.y])
# xy_flat = np.vstack([ds.x.values, ds.y.values]).flatten()
# # Plot var
# plt.figure()
# plt.imshow(var_flat, aspect="equal")
# plt.show()

# Derive impulse duration
eps = 1e-20
window_length = 20
spl = ds.s_l.rolling(t=window_length, center=True).var()
t = spl.t.values
ts = t[1] - t[0]
th_dB = -30

t_mat = cast_matrix_to_target_shape(t, spl.shape)
max_spl = spl.max(dim="t")
spl_norm = spl / max_spl
spl_norm_dB = 10 * np.log10(spl_norm + eps)

mask_30dB = spl_norm_dB > th_dB
mask_30dB = mask_30dB.astype(int)

t_mask = t_mat * mask_30dB
t_mask = np.where(t_mask == 0, np.nan, t_mask)
t_first_arrival = np.nanmin(t_mask, axis=1)
t_last_arrival = np.nanmax(t_mask, axis=1)
tau_ir = t_last_arrival - t_first_arrival

ds_tau_ir = xr.Dataset(
    {
        "tau_ir": (("idx_rcv", "x", "y"), tau_ir),
    },
    coords={
        "idx_rcv": ds.idx_rcv,
        "x": ds.x,
        "y": ds.y,
    },
)

# Save netcdf
fpath = os.path.join(root, "tau_ir.nc")
ds_tau_ir.to_netcdf(fpath)

med_tau = ds_tau_ir.tau_ir.median()
print(med_tau)

# plt.figure()
# tau_ir.isel(idx_rcv=0).plot()
# plt.show()

plt.figure()
ds_tau_ir.tau_ir.plot.hist(bins=200)
plt.axvline(med_tau, color="r", linestyle="--")

plt.figure()
spl_norm_dB.isel(idx_rcv=0, x=0, y=0).plot()
plt.show()


# spl_dB = [
#     10
#     * np.log10(
#         (spl.sel(idx_rcv=ircv, x=x, y=y) + eps)
#         / (spl.sel(idx_rcv=ircv, x=x, y=y).max() + eps)
#     )
#     for ircv in ds.idx_rcv
#     for x in ds.x
#     for y in ds.y
# ]

# plt.figure()
# spl_dB[0].plot()
# plt.show()
# ir_durations =
# if __name__ == "__main__":
#     pass
