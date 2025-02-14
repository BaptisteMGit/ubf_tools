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

from misc import mult_along_axis
from signals.signals import generate_ship_signal
from signals.AcousticComponent import AcousticSource

# Load dataset
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process"
# dir_path = r"testcase1_4_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240711_161507.zarr"
# dir_path = r"testcase1_4_AC198EBFF716\65.6544_66.0574_-27.7358_-27.3766_ship\20240717_164140.zarr"
# dir_path = r"testcase1_4_AC198EBFF716\65.6544_66.0574_-27.7358_-27.3766_ship\20240718_101313.zarr"
dir_path = r"testcase3_1_AC198EBFF716\65.4157_65.8190_-27.8330_-27.4739_ship\20240718_155452.zarr"

fpath = os.path.join(root, dir_path)
ds = xr.open_dataset(fpath, engine="zarr", chunks={})
ds = ds.isel(snr=0)

# Extract source position
lon_s, lat_s = ds.lon_src.values[0], ds.lat_src.values[0]
# lon_s = 65.775
# lat_s = -27.56

# Extract transfert function at ship position
tf_at_ship_pos = ds.tf_gridded.sel(lon=lon_s, lat=lat_s, method="nearest")

# Extract corr
lib_feature = ds.library_feature.sel(lon=lon_s, lat=lat_s, method="nearest")
event_feature = ds.event_feature.isel(src_trajectory_time=0)

# Extract signals
lib_sig = ds.rcv_signal_library.sel(lon=lon_s, lat=lat_s, method="nearest")
event_sig = ds.rcv_signal_event.isel(src_trajectory_time=0)

# Extract delays
tau = ds.delay_rcv.sel(lon=lon_s, lat=lat_s, method="nearest")

f_th = ds.kraken_freq.values

# Params
fs = 100

dt = ds.library_signal_time.values[1] - ds.library_signal_time.values[0]
fs_mes = 1 / dt
#######################
# Test t_ij computation
#######################
f0_event = 5
duration = 10.24
event_sig_info = {
    "sig_type": "ship",
    "f0": f0_event,
    "std_fi": f0_event * 1 / 100,
    "tau_corr_fi": 1 / f0_event,
    "fs": fs,
}

src_sig, t_src_sig = generate_ship_signal(
    Ttot=duration,
    f0=event_sig_info["f0"],
    std_fi=event_sig_info["std_fi"],
    tau_corr_fi=event_sig_info["tau_corr_fi"],
    fs=event_sig_info["fs"],
)

src_sig *= np.hanning(len(src_sig))
min_waveguide_depth = 5000
src = AcousticSource(
    signal=src_sig,
    time=t_src_sig,
    name="ship",
    waveguide_depth=min_waveguide_depth,
    nfft=len(src_sig),
)

C0 = 1500
nfft = src.nfft
propagating_freq = src.positive_freq
propagating_spectrum = src.positive_spectrum
k0 = 2 * np.pi * propagating_freq / C0
norm_factor = np.exp(1j * k0) / (4 * np.pi)


# Relative transfert function
rtf = []
rtf_th = []
rtf_other = []

lib_feature_other = ds.library_feature.sel(
    lon=lon_s + 0.02, lat=lat_s + 0.02, method="nearest"
)

alpha = 1
nfft *= alpha
fs *= alpha

tau = tau.min(dim="idx_rcv").values

# Plot correlation for each receivers pair
for i_rcv_p in ds.idx_rcv_pairs.values:

    pair = ds.rcv_pairs[i_rcv_p].values

    # Features
    lib_feat = lib_feature.sel(idx_rcv_pairs=i_rcv_p).values
    # event_feat = event_feature.sel(idx_rcv_pairs=i_rcv_p).values

    # Signals
    in1 = lib_sig.sel(idx_rcv=pair[0]).values
    in2 = lib_sig.sel(idx_rcv=pair[1]).values

    # Transfert function
    h_p1 = tf_at_ship_pos.sel(idx_rcv=pair[0]).values
    h_p2 = tf_at_ship_pos.sel(idx_rcv=pair[1]).values

    # Derive delay factor
    tau_vec = tau * propagating_freq
    delay_f = np.exp(1j * 2 * np.pi * tau_vec)

    in1_f = h_p1 * propagating_spectrum * norm_factor * delay_f
    in1_p = sp_fft.irfft(in1_f, n=nfft)
    in2_f = h_p2 * propagating_spectrum * norm_factor * delay_f
    in2_p = sp_fft.irfft(in2_f, n=nfft)

    sp1 = sp_fft.rfft(in1)
    h_dec = h_p1 * norm_factor * delay_f
    z_idx = h_dec != 0
    sp1 = sp1[z_idx]
    sinv = sp_fft.irfft(sp1 / h_dec[z_idx], n=nfft)

    s1 = in1_p
    s2 = in2_p
    # s1, s2 = in1, in2
    # Easy way : from PSD and cross PSD
    # f_, s_11 = signal.welch(in1, fs=fs)
    nperseg = 256 * alpha
    noverlap = nperseg // 100 * 50
    # f_, s_22 = signal.welch(s2, fs=fs)
    # f_, s_12 = signal.csd(s1, s2, fs=fs)

    s_22 = sp_fft.rfft(s2) * np.conj(sp_fft.rfft(s2))
    s_12 = sp_fft.rfft(s1) * np.conj(sp_fft.rfft(s2))
    f_ = sp_fft.rfftfreq(n=in1.size, d=1 / fs)

    # RTF
    # t_12 = s_12 / s_22
    # t_12.append(d)
    rtf.append(lib_feat)
    rtf_other.append(lib_feature_other.sel(idx_rcv_pairs=i_rcv_p).values)

    # Theoretical RTF
    t_12_th = h_p1 / h_p2
    # t_12_th = s_12 / s_22
    rtf_th.append(t_12_th)

t = ds.library_signal_time.values
# plt.figure()
# plt.plot(np.abs(h_p1))

# plt.figure()
# plt.plot(src.time, src.signal, label="s")
# plt.plot(t, sinv, label="sinv")
# plt.legend()

plt.figure()
plt.plot(t, in1, label="s1 (sim)")
plt.plot(t, in1_p, label="s1 (inplace)")
plt.legend()

plt.figure()
plt.plot(t, in2, label="s2 (sim)")
plt.plot(t, in2_p, label="s2 (inplace)")
plt.legend()

# plt.figure()
# c11p = signal.correlate(in1, in1_p, mode="full")
# c22p = signal.correlate(in2, in2_p, mode="full")
# plt.plot(c11p, label="c11p")
# plt.plot(c22p, label="c22p")
# plt.legend()
# plt.show()

# Plot
plt.figure()
plt.plot(f_th, np.abs(tf_at_ship_pos.sel(idx_rcv=0)), label=rf"$H_{0}$")
plt.plot(f_th, np.abs(tf_at_ship_pos.sel(idx_rcv=1)), label=rf"$H_{1}$")
plt.plot(f_th, np.abs(tf_at_ship_pos.sel(idx_rcv=2)), label=rf"$H_{2}$")
plt.legend()

# Plot rtfs
rtf = np.array(rtf)
rtf_th = np.array(rtf_th)
rtf_other = np.array(rtf_other)
for i_rcv_p in ds.idx_rcv_pairs.values:
    pair = ds.rcv_pairs[i_rcv_p].values
    plt.figure()
    plt.plot(f_, np.abs(rtf[i_rcv_p, :]), label=rf"$T_{{{pair[0]} {pair[1]}}}$")
    plt.plot(
        f_th,
        np.abs(rtf_th[i_rcv_p, :]),
        label=rf"$T_{{{pair[0]} {pair[1]}}} th$",
        linestyle="--",
    )
    plt.plot(
        f_,
        np.abs(rtf_other[i_rcv_p, :]),
        label=rf"$T_{{{pair[0]} {pair[1]}}} other$",
        linestyle="--",
    )

    plt.ylim([0, 6])
    plt.legend()

plt.figure()
plt.plot(f_th, np.abs(tf_at_ship_pos.sel(idx_rcv=0)), label=rf"$H_{0}$")
plt.plot(f_th, np.abs(tf_at_ship_pos.sel(idx_rcv=1)), label=rf"$H_{1}$")

for i_rcv_p in ds.idx_rcv_pairs.values:
    pair = ds.rcv_pairs[i_rcv_p].values
    h_1 = tf_at_ship_pos.sel(idx_rcv=pair[1])
    h_1_interp_real = np.interp(f_, f_th, np.real(h_1))
    h_1_interp_imag = np.interp(f_, f_th, np.imag(h_1))
    h0_tild_real = np.real(rtf[i_rcv_p, :]) * h_1_interp_real
    h0_tild_imag = np.imag(rtf[i_rcv_p, :]) * h_1_interp_imag
    h0_tild = h0_tild_real + 1j * h0_tild_imag

    plt.plot(
        f_,
        np.abs(h0_tild),
        label=rf"$\tilde{{H}}_{pair[0]} (T_{{{pair[0]} {pair[1]}}})$",
    )

plt.legend()

plt.show()


# lon_s, lat_s = ds.lon_src.values[0], ds.lat_src.values[0]

# # Extract transfert function at ship position
# tf_at_ship_pos = ds.tf_gridded.sel(lon=lon_s, lat=lat_s, method="nearest")

# # Extract corr
# lib_corr_s = ds.library_corr.sel(lon=lon_s, lat=lat_s, method="nearest")
# event_corr_s = ds.event_corr.isel(src_trajectory_time=0)

# # Plot correlation for each receivers pair
# f, ax = plt.subplots(ds.sizes["idx_rcv_pairs"], 1, figsize=(10, 5), sharex=True)
# for i_rcv_p in ds.idx_rcv_pairs.values:
#     # Event
#     event_corr_s.sel(idx_rcv_pairs=i_rcv_p).plot(ax=ax[i_rcv_p], label=f"event")
#     # Library
#     lib_corr_s.sel(idx_rcv_pairs=i_rcv_p).plot(ax=ax[i_rcv_p], label=f"lib")
#     ax[i_rcv_p].legend()
#     ax[i_rcv_p].set_title("")
#     ax[i_rcv_p].set_title(f"Receiver pair {i_rcv_p}", loc="left")

# plt.show()


# # Derive impulse response
# ir = sp_fft.irfft(tf_at_ship_pos.values, n=ds.sizes["library_signal_time"], axis=1)

# plt.figure()
# for i_rcv in ds.idx_rcv.values:
#     plt.plot(ds.library_signal_time, ir[i_rcv, :], label=f"Receiver {i_rcv}")

# plt.legend()
# # plt.show()

# # Derive theoretical intercorr
# nlag = ds.sizes["library_corr_lags"]
# th_corr_f = []
# th_corr_sig = []
# # th_corr_f = np.array([])
# for i_rcv_p in ds.idx_rcv_pairs.values:
#     pair = ds.rcv_pairs[i_rcv_p].values

#     num = np.abs(tf_at_ship_pos.sel(idx_rcv=pair[0])) * np.abs(
#         tf_at_ship_pos.sel(idx_rcv=pair[1])
#     )
#     num = np.where(num == 0, 1e-20, num)

#     # cij_f = signal.fftconvolve(in1, in2[..., ::-1], mode="full", axes=-1)

#     cij_f = (
#         tf_at_ship_pos.sel(idx_rcv=pair[0])
#         * np.conj(tf_at_ship_pos.sel(idx_rcv=pair[1]))
#         / num
#     )
#     th_corr_f.append(cij_f.values)

#     # From signal
#     in1 = ds.rcv_signal_library.sel(idx_rcv=pair[0]).values
#     in2 = ds.rcv_signal_library.sel(idx_rcv=pair[1]).values
#     r_12 = signal.fftconvolve(in1, in2[..., ::-1], mode="full", axes=-1)
#     r_11 = signal.fftconvolve(in1, in1[..., ::-1], mode="full", axes=-1)
#     r_22 = signal.fftconvolve(in2, in2[..., ::-1], mode="full", axes=-1)

#     n0 = r_12.shape[-1] // 2
#     norm = np.sqrt(r_11[..., n0] * r_22[..., n0])
#     norm = np.repeat(np.expand_dims(norm, axis=-1), nlag, axis=-1)
#     c_12 = r_12 / norm
#     th_corr_sig.append(c_12)

# th_corr_f = np.array(th_corr_f)
# # Derive theoretical intercorr in lag domain
# th_corr = sp_fft.irfft(th_corr_f, n=nlag, axis=1)


# # Plot th_corr
# plt.figure()
# for i_rcv_p in ds.idx_rcv_pairs.values:
#     plt.plot(ds.library_corr_lags, th_corr[i_rcv_p, :], label=f"Pair {i_rcv_p}")
# plt.legend()
# plt.show()
