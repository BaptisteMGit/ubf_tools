#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_implementation.py
@Time    :   2024/07/11 17:10:43
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
import scipy.fft as sp_fft
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load dataset
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process"
dir_path = r"testcase1_4_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240711_161507.zarr"

fpath = os.path.join(root, dir_path)
ds = xr.open_dataset(fpath, engine="zarr", chunks={})
ds = ds.isel(snr=0)

lon_s, lat_s = ds.lon_src.values[0], ds.lat_src.values[0]

# Extract transfert function at ship position
tf_at_ship_pos = ds.tf_gridded.sel(lon=lon_s, lat=lat_s, method="nearest")

# Extract corr
lib_corr_s = ds.library_corr.sel(lon=lon_s, lat=lat_s, method="nearest")
event_corr_s = ds.event_corr.isel(src_trajectory_time=0)

# Plot correlation for each receivers pair
f, ax = plt.subplots(ds.sizes["idx_rcv_pairs"], 1, figsize=(10, 5), sharex=True)
for i_rcv_p in ds.idx_rcv_pairs.values:
    # Event
    event_corr_s.sel(idx_rcv_pairs=i_rcv_p).plot(ax=ax[i_rcv_p], label=f"event")
    # Library
    lib_corr_s.sel(idx_rcv_pairs=i_rcv_p).plot(ax=ax[i_rcv_p], label=f"lib")
    ax[i_rcv_p].legend()
    ax[i_rcv_p].set_title("")
    ax[i_rcv_p].set_title(f"Receiver pair {i_rcv_p}", loc="left")

# plt.show()


# Derive impulse response
ir = sp_fft.irfft(tf_at_ship_pos.values, n=ds.sizes["library_signal_time"], axis=1)

plt.figure()
for i_rcv in ds.idx_rcv.values:
    plt.plot(ds.library_signal_time, ir[i_rcv, :], label=f"Receiver {i_rcv}")

plt.legend()
# plt.show()

# Derive theoretical intercorr
nlag = ds.sizes["library_corr_lags"]
th_corr_f = []
th_corr_sig = []
# th_corr_f = np.array([])
for i_rcv_p in ds.idx_rcv_pairs.values:
    pair = ds.rcv_pairs[i_rcv_p].values

    num = np.abs(tf_at_ship_pos.sel(idx_rcv=pair[0])) * np.abs(
        tf_at_ship_pos.sel(idx_rcv=pair[1])
    )
    num = np.where(num == 0, 1e-20, num)

    # cij_f = signal.fftconvolve(in1, in2[..., ::-1], mode="full", axes=-1)

    cij_f = (
        tf_at_ship_pos.sel(idx_rcv=pair[0])
        * np.conj(tf_at_ship_pos.sel(idx_rcv=pair[1]))
        / num
    )
    th_corr_f.append(cij_f.values)

    # From signal
    in1 = ds.rcv_signal_library.sel(idx_rcv=pair[0]).values
    in2 = ds.rcv_signal_library.sel(idx_rcv=pair[1]).values
    r_12 = signal.fftconvolve(in1, in2[..., ::-1], mode="full", axes=-1)
    r_11 = signal.fftconvolve(in1, in1[..., ::-1], mode="full", axes=-1)
    r_22 = signal.fftconvolve(in2, in2[..., ::-1], mode="full", axes=-1)

    n0 = r_12.shape[-1] // 2
    norm = np.sqrt(r_11[..., n0] * r_22[..., n0])
    norm = np.repeat(np.expand_dims(norm, axis=-1), nlag, axis=-1)
    c_12 = r_12 / norm
    th_corr_sig.append(c_12)

th_corr_f = np.array(th_corr_f)
# Derive theoretical intercorr in lag domain
th_corr = sp_fft.irfft(th_corr_f, n=nlag, axis=1)


# Plot th_corr
plt.figure()
for i_rcv_p in ds.idx_rcv_pairs.values:
    plt.plot(ds.library_corr_lags, th_corr[i_rcv_p, :], label=f"Pair {i_rcv_p}")
plt.legend()
plt.show()
