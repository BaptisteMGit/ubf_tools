#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf.py
@Time    :   2024/09/17 14:49:14
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Show transfert function structure and rtf verctor to study potential feature to use for localisation
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 0) Define usefull params
root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf"
# 1) Load data from the simulation database (xarray)
fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process\testcase3_1_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240709_091228.zarr"
ds = xr.open_dataset(fpath, engine="zarr", chunks={})

# Position of the source to be localized
src_lon = ds.lon_src.values[0]
src_lat = ds.lat_src.values[0]

# 1.1) Show the transfert functions corresponding to the source position
tf_at_src_pos = ds.tf_gridded.sel(lon=src_lon, lat=src_lat, method="nearest")
tf_at_src_pos["mod"] = (["idx_rcv", "kraken_freq"], np.abs(tf_at_src_pos).data)
tf_at_src_pos["arg"] = (["idx_rcv", "kraken_freq"], np.angle(tf_at_src_pos))


plt.figure()
for i in range(ds.sizes["idx_rcv"]):
    tf_at_src_pos.mod.isel(idx_rcv=i).plot(label=f"rcv_{i}")
plt.title("Transfer function at source position")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

fpath = os.path.join(root_img, "tf_mod.png")
plt.savefig(fpath)

plt.figure()
for i in range(ds.sizes["idx_rcv"]):
    tf_at_src_pos.arg.isel(idx_rcv=i).plot(label=f"rcv_{i}")
plt.title("Transfer function at source position")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.legend()
plt.grid()

fpath = os.path.join(root_img, "tf_arg.png")
plt.savefig(fpath)


# 1.2) Derive impulse response from the transfer function
ir_at_src_pos = np.fft.irfft(tf_at_src_pos, axis=1)
Ts = ds.library_signal_time.diff(dim="library_signal_time").values[0]
time = np.arange(0, ir_at_src_pos.shape[1]) * Ts

plt.figure()
for i in range(ds.sizes["idx_rcv"]):
    plt.plot(time, ir_at_src_pos[i, :], label=f"rcv_{i}")
plt.title("Impulse response at source position")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

fpath = os.path.join(root_img, "ir.png")
plt.savefig(fpath)


# 1.3) Plot associated tl profiles to make sure the impulse response is correct

# Select tf profile closest to the source position
fpath_tl = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_dataset\testcase3_1_AC198EBFF716\propa\propa_65.5103_65.6797_-27.7342_-27.5758.zarr"
ds_tl = xr.open_dataset(fpath_tl, engine="zarr", chunks={})
az = ds_tl.az_propa.sel(lon=src_lon, lat=src_lat, method="nearest")
mod_az = np.abs(ds_tl.tf.sel(all_az=az, method="nearest"))
# Replace 0 by 1e-10 to avoid log(0) issue
mod_az = mod_az.where(mod_az > 0, 1e-10)

tl_az = 20 * np.log10(mod_az)

freq_of_interest = 10

tl_cyl = -30 * np.log10(tl_az.kraken_range.values[1:])
# ds.sizes["idx_rcv"]
plt.figure()
plt.plot(
    tl_az.kraken_range.values[1:],
    tl_cyl,
    label=r"$-30log_{10}(r)$",
    color="k",
    linestyle="--",
)

for i in range(2):
    tl_az.isel(idx_rcv=i).sel(kraken_freq=freq_of_interest, method="nearest").plot(
        label="rcv_" + str(i)
    )

plt.title(f"Transmission loss at {freq_of_interest} Hz for rcv_{i}")
plt.xlabel("Range [m]")
plt.ylabel("Amplitude [dB]")
plt.legend()
plt.grid()

fpath = os.path.join(root_img, "tl.png")
plt.savefig(fpath)


# 2) Build rtf vector
idx_rcv_ref = 0
rtf = tf_at_src_pos / tf_at_src_pos.isel(idx_rcv=idx_rcv_ref)

# 2.1) Show the rtf vector
rtf["mod"] = (["idx_rcv", "kraken_freq"], np.abs(rtf).data)
rtf["arg"] = (["idx_rcv", "kraken_freq"], np.angle(rtf))

plt.figure()
for i in range(ds.sizes["idx_rcv"]):
    rtf.mod.isel(idx_rcv=i).plot(label=f"rcv_{i}")
plt.title("RTF at source position")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

fpath = os.path.join(root_img, "rtf_mod.png")
plt.savefig(fpath)


# plt.figure()
# for i in range(ds.sizes["idx_rcv"]):
#     rtf.arg.isel(idx_rcv=i).plot(label=f"rcv_{i}")
# plt.title("RTF at source position")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Phase [rad]")
# plt.grid()
# plt.legend()


# 2.2) Compare to the rtf vector for a close position to the source
src_lon_close = src_lon + 0.01
src_lat_close = src_lat + 0.01
tf_at_src_pos_close = ds.tf_gridded.sel(
    lon=src_lon_close, lat=src_lat_close, method="nearest"
)
rtf_close = tf_at_src_pos_close / tf_at_src_pos_close.isel(idx_rcv=idx_rcv_ref)

rtf_close["mod"] = (["idx_rcv", "kraken_freq"], np.abs(rtf_close).data)
rtf_close["arg"] = (["idx_rcv", "kraken_freq"], np.angle(rtf_close))

for i in range(ds.sizes["idx_rcv"]):
    plt.figure()
    rtf_close.mod.isel(idx_rcv=i).plot(label=f"rcv_{i} (close)")
    rtf.mod.isel(idx_rcv=i).plot(label=f"rcv_{i} (ref)")
    plt.ylim(
        0,
        max(np.max(rtf.mod.isel(idx_rcv=i)), np.max(rtf_close.mod.isel(idx_rcv=i)))
        * 1.2,
    )
    plt.title("RTF at close source position")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    fpath = os.path.join(root_img, f"rtf_mod_close_{i}.png")
    plt.savefig(fpath)


# plt.figure()
# for i in range(ds.sizes["idx_rcv"]):
#     rtf_close.arg.isel(idx_rcv=i).plot(label=f"rcv_{i}")
# plt.title("RTF at close source position")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Phase [rad]")
# plt.grid()
# plt.legend()

# Evaluate distance between the two rtf vectors for a set of close positions
src_lon_close = src_lon + np.arange(0, 0.05, 0.01)
src_lat_close = src_lat + np.arange(0, 0.05, 0.01) * 0
npos_close = len(src_lon_close)

rtf_dist = np.empty((ds.sizes["idx_rcv"], npos_close))

fig, ax = plt.subplots(
    nrows=ds.sizes["idx_rcv"] - 1, ncols=1, figsize=(10, 10), sharex=True
)
k = 0
for i_rcv in range(ds.sizes["idx_rcv"]):
    if i_rcv != idx_rcv_ref:
        ax[k].plot(rtf.mod.isel(idx_rcv=i_rcv), label=f"rcv_{i_rcv} (ref)")
        ax[k].legend()
        ax[k].grid()
        k += 1
fig.supxlabel("Frequency [Hz]")
fig.supylabel("Amplitude")


for i_pos in range(npos_close):
    lon_i = src_lon_close[i_pos]
    lat_i = src_lat_close[i_pos]

    # Derive rtf
    tf_at_src_pos_i = ds.tf_gridded.sel(lon=lon_i, lat=lat_i, method="nearest")
    # Replace 0 by nan to avoid division by 0
    tf_at_src_pos_i = tf_at_src_pos_i.where(np.abs(tf_at_src_pos_i) > 0, np.nan)
    rtf_i = tf_at_src_pos_i / tf_at_src_pos_i.isel(idx_rcv=idx_rcv_ref)

    # Plot rtf

    rtf_i["mod"] = (["idx_rcv", "kraken_freq"], np.abs(rtf_i).data)
    rtf_i["arg"] = (["idx_rcv", "kraken_freq"], np.angle(rtf_i))
    k = 0
    for i_rcv in range(ds.sizes["idx_rcv"]):
        if i_rcv != idx_rcv_ref:
            ax[k].plot(
                rtf_i.mod.isel(idx_rcv=i_rcv), label=f"rcv_{i_rcv} (pos n°{i_pos})"
            )
            ax[k].legend()
            ax[k].set_ylim([0, 10])
            k += 1

    # Derive distance between rtf_i and rtf (true source position)
    # dist = np.linalg.norm(rtf_i - rtf)
    # dist = np.sqrt(np.sum((rtf_i - rtf) ** 2))
    d1 = np.sum(np.abs(rtf_i - rtf) ** 2, axis=1)

    # Store distance
    rtf_dist[:, i_pos] = d1

fpath = os.path.join(root_img, "rtf_mod_close_pos.png")
plt.savefig(fpath)

plt.close()


# 2.2) Compare to the rtf vector for a  position far from the source
src_lon_far = src_lon + 0.1
src_lat_far = src_lat + 0.1
tf_at_src_pos_far = ds.tf_gridded.sel(
    lon=src_lon_far, lat=src_lat_far, method="nearest"
)
rtf_far = tf_at_src_pos_far / tf_at_src_pos_far.isel(idx_rcv=idx_rcv_ref)

rtf_far["mod"] = (["idx_rcv", "kraken_freq"], np.abs(rtf_far).data)
rtf_far["arg"] = (["idx_rcv", "kraken_freq"], np.angle(rtf_far))

for i in range(ds.sizes["idx_rcv"]):
    plt.figure()
    rtf_far.mod.isel(idx_rcv=i).plot(label=f"rcv_{i} (far)")
    rtf.mod.isel(idx_rcv=i).plot(label=f"rcv_{i} (ref)")
    plt.ylim(
        0,
        max(np.max(rtf.mod.isel(idx_rcv=i)), np.max(rtf_far.mod.isel(idx_rcv=i))) * 1.2,
    )
    plt.title("RTF at far source position")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    fpath = os.path.join(root_img, f"rtf_mod_far_{i}.png")
    plt.savefig(fpath)


# plt.figure()
# for i in range(ds.sizes["idx_rcv"]):
#     rtf_far.arg.isel(idx_rcv=i).plot(label=f"rcv_{i}")
# plt.title("RTF at far source position")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Phase [rad]")
# plt.grid()
# plt.legend()

# Evaluate distance between the two rtf vectors for a set of far positions
src_lon_far = src_lon + np.arange(0.2, 0.5, 0.01)
src_lat_far = src_lat + np.arange(0.2, 0.5, 0.01) * 0
npos_far = len(src_lon_far)

rtf_dist = np.empty((ds.sizes["idx_rcv"], npos_far))

fig, ax = plt.subplots(
    nrows=ds.sizes["idx_rcv"] - 1, ncols=1, figsize=(10, 10), sharex=True
)
k = 0
for i_rcv in range(ds.sizes["idx_rcv"]):
    if i_rcv != idx_rcv_ref:
        ax[k].plot(rtf.mod.isel(idx_rcv=i_rcv), label=f"rcv_{i_rcv} (ref)")
        ax[k].legend()
        ax[k].grid()
        k += 1
fig.supxlabel("Frequency [Hz]")
fig.supylabel("Amplitude")

for i_pos in range(npos_far):
    lon_i = src_lon_far[i_pos]
    lat_i = src_lat_far[i_pos]

    # Derive rtf
    tf_at_src_pos_i = ds.tf_gridded.sel(lon=lon_i, lat=lat_i, method="nearest")
    # Replace 0 by nan to avoid division by 0
    tf_at_src_pos_i = tf_at_src_pos_i.where(np.abs(tf_at_src_pos_i) > 0, np.nan)
    rtf_i = tf_at_src_pos_i / tf_at_src_pos_i.isel(idx_rcv=idx_rcv_ref)

    # Plot rtf

    rtf_i["mod"] = (["idx_rcv", "kraken_freq"], np.abs(rtf_i).data)
    rtf_i["arg"] = (["idx_rcv", "kraken_freq"], np.angle(rtf_i))
    k = 0
    for i_rcv in range(ds.sizes["idx_rcv"]):
        if i_rcv != idx_rcv_ref:
            ax[k].plot(
                rtf_i.mod.isel(idx_rcv=i_rcv), label=f"rcv_{i_rcv} (pos n°{i_pos})"
            )
            ax[k].legend()
            ax[k].set_ylim([0, 10])
            k += 1

    # Derive distance between rtf_i and rtf (true source position)
    # dist = np.linalg.norm(rtf_i - rtf)
    # dist = np.sqrt(np.sum((rtf_i - rtf) ** 2))
    d1 = np.sum(np.abs(rtf_i - rtf) ** 2, axis=1)

    # Store distance
    rtf_dist[:, i_pos] = d1

fpath = os.path.join(root_img, "rtf_mod_far_pos.png")
plt.savefig(fpath)

plt.close()


# Plot rtf_dist against index
plt.figure()
for i in range(ds.sizes["idx_rcv"]):
    plt.plot(rtf_dist[i, :], label=f"rcv_{i}")
plt.title("Distance between RTF vectors")
plt.xlabel("Index")
plt.ylabel("Distance")
plt.grid()
plt.legend()

plt.show()
