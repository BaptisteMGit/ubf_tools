# %% [markdown]
# # Objectif
#
# L'objectif de jupyter notebook est d'étudier la structure de la fonction d'intercorrélation pour deux trajectoires de navires circulant au Nord de la sous anetnne SWIR
# %%
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   real_data_analysis_1.ipynb
@Time    :   2024/09/06 12:01:19
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
# %matplotlib ipympl

import sys

sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd")

import os
import pandas as pd
import numpy as np
import scipy.fft as sf
import scipy.signal as sp
import matplotlib.pyplot as plt

from get_data.ais.ais_tools import *
from get_data.wav.signal_processing_utils import *
from publication.PublicationFigure import PubFigure
from localisation.verlinden.misc.verlinden_utils import load_rhumrum_obs_pos

# ## AIS data

# Define usefulls params, load and pre-process AIS data
img_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\real_data\intercorrelation"
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ais\extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
fname = "extract-ais-pos-for-zone-ecole-navale-by-month-201305.csv"
fpath = os.path.join(root, fname)

# Ensure img_root exists
if not os.path.exists(img_root):
    os.makedirs(img_root)

lon_min = 65
lon_max = 66
lat_min = -28
lat_max = -27

# Load and pre-filter
df = extract_ais_area(fpath, lon_min, lon_max, lat_min, lat_max)
# Remove ships with less than 2 points
df = df.groupby("mmsi").filter(lambda x: len(x) > 1)
# Interpolate trajectories to have a point every 5 minutes
df_interp = interpolate_trajectories(df, time_step="5min")


# station_id = ["RR43", "RR44"]
station_id = ["RR41", "RR47", "RR43", "RR44"]

rcv_info = {"id": station_id, "lons": [], "lats": []}
for obs_id in rcv_info["id"]:
    pos_obs = load_rhumrum_obs_pos(obs_id)
    rcv_info["lons"].append(pos_obs.lon)
    rcv_info["lats"].append(pos_obs.lat)

# Selected ships
# mmsi1 = 403508000
# mmsi2 = 353766000
# NAVIOS / SHUI HO
mmsi1 = 416004485
mmsi2 = 373759000
mmsi_selected = [mmsi1, mmsi2]

# Select given ships
ais_data = df_interp[df_interp["mmsi"].isin(mmsi_selected)]
# Reset index
ais_data.reset_index(drop=True, inplace=True)

# Folder name composed of the selected ships names, the receiver ids
name1 = ais_data[ais_data["mmsi"] == mmsi1]["shipName"].values[0]
name2 = ais_data[ais_data["mmsi"] == mmsi2]["shipName"].values[0]
img_folder_shipname = f"{name1}_{name2}"
img_folder_rcv_ids = "_".join(station_id)
img_path = os.path.join(img_root, img_folder_shipname, img_folder_rcv_ids)

# Create folder if no exist
if not os.path.exists(img_path):
    os.makedirs(img_path)

# Convert to ENU
project_to_enu(df=ais_data)

# Compute intersection
intersection_wgs84, intersection_time_s1, intersection_time_s2 = (
    intersection_time_2ships(df_2ships=ais_data)
)

# Store intersection time in a dictionary for simplicity
intersection_time = {mmsi1: intersection_time_s1[0], mmsi2: intersection_time_s2[0]}
print(intersection_time)

# Plot selected ships
intersection_data = {
    "intersection_wgs84": intersection_wgs84,
    "intersection_time": intersection_time,
    "intersection_time_s1": intersection_time_s1,
    "intersection_time_s2": intersection_time_s2,
}
plt.figure()
plot_traj_over_bathy(
    ais_data,
    rcv_info,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    intersection_data=intersection_data,
)

# Save figure
fig_name = "ship_routes_over_bathy"
plt.savefig(os.path.join(img_path, f"{fig_name}.png"))


duration_seconds, start_times, end_times = get_available_time_window(
    ais_data=ais_data, intersection_time=intersection_time
)
ais_data = restrict_ais_data_to_time_window(
    ais_data=ais_data, start_times=start_times, end_times=end_times
)


# We are only interested in the hydrophone channel
ch = ["BDH"]

# Select frequency properties
fmin = 4
fmax = 18
filter_type = "bandpass"
filter_corners = 20

wav_data = load_data(
    station_id,
    mmsi_selected,
    start_times,
    ch=ch,
    fmin=fmin,
    fmax=fmax,
    filter_type=filter_type,
    filter_corners=filter_corners,
    duration_seconds=duration_seconds,
)


# Plot spectrograms
plot_spectrograms(
    wav_data=wav_data,
    ais_data=ais_data,
    intersection_time=intersection_time,
    fmin=fmin,
    fmax=fmax,
)

# Save figure
fig_name = "stft"
plt.savefig(os.path.join(img_path, f"{fig_name}.png"))

# ## Distance ship to receiver

# Compute distance
distance = compute_distance_ship_rcv(ais_data, rcv_info)

# Derive CPA for each each ship and each receiver
cpa = get_cpa(ais_data=ais_data, distance=distance, rcv_info=rcv_info)
print(cpa)

# ### Plot distances

plot_distance_ship_rcv(ais_data=ais_data, distance=distance, cpa=cpa, rcv_info=rcv_info)

# Save figure
fig_name = "r_ship_rcv"
plt.savefig(os.path.join(img_path, f"{fig_name}.png"))

# Derive time delay

# Derive propagation time from each ship position to each receiver asssuming constant speed of sound
c0 = 1500  # Speed of sound in water in m/s
propagation_time = {}

for mmsi in ais_data["mmsi"].unique():
    propagation_time[mmsi] = {}
    for rcv_id in rcv_info["id"]:
        propagation_time[mmsi][rcv_id] = distance[mmsi][rcv_id] / c0

print(f"Propagation time: {propagation_time}")

# Derive time delay between receiver couples for each ship position
time_delay = {}

for i, rcv_id1 in enumerate(rcv_info["id"]):
    for j, rcv_id2 in enumerate(rcv_info["id"]):
        # Rcvi - Rcvj is equivalent to Rcvj - Rcvi, only one couple needs to be evaluated
        if i < j:
            rcv_couple = f"{rcv_id1}_{rcv_id2}"
            time_delay[rcv_couple] = {}
            for mmsi in ais_data["mmsi"].unique():

                time_delay[rcv_couple][mmsi] = (
                    propagation_time[mmsi][rcv_id1] - propagation_time[mmsi][rcv_id2]
                )


# Plot time delay between receiver couples as a function of time for each ship
n_ships = len(mmsi_selected)
fig, ax = plt.subplots(n_ships, 1, sharex=False)

# Reshape ax if only one ship
if n_ships == 1:
    ax = np.array([ax])

for j, mmsi in enumerate(mmsi_selected):
    df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
    for rcv_couple in time_delay.keys():
        ax[j].plot(df_mmsi["time"], time_delay[rcv_couple][mmsi], label=f"{rcv_couple}")

    ship_name = df_mmsi["shipName"].values[0]
    ax[j].set_title(f"{ship_name}")
    ax[j].legend(loc="upper right", fontsize=12)

    ax[j].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[j].xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Get the last date in the time data
    end_date = df_mmsi["time"].iloc[0].strftime("%Y-%m-%d")

    # Add the date as text at the end of the x-axis
    ax[j].text(
        0.99,
        0.05,
        f"Date: {end_date}",
        transform=ax[j].transAxes,
        ha="right",
        fontsize=12,
    )
    ax[j].tick_params(axis="x", rotation=0)

fig.supxlabel("Time [h]")
fig.supylabel(r"$\tau [s]$")

# Save figure
fig_name = "tdoa"
plt.savefig(os.path.join(img_path, f"{fig_name}.png"))

# ### Derive correlation window duration
# Correlation window must be sufficiently large to include signal from both ship
# A sufficient condition is given by $T_{corr} \geq \tau + \max{T_{r1}, T_{r2}}$ where $\tau$ is the delay between R1 and R2 and $T_{r1}$ and $T_{r2}$ represents the
# impulse response duration corresponding to waveguides between the ship position and the receivers

# Derive maximum time delay for each ship
Tau = {}
for rcv_couple in time_delay.keys():
    Tau[rcv_couple] = {}
    for j, mmsi in enumerate(mmsi_selected):
        Tau[rcv_couple][mmsi] = np.max(np.abs(time_delay[rcv_couple][mmsi]))


# First approach : lets try some values for Tri
impulse_response_max_duration = 5
# delta_t_stat = 25 * 60
# delta_t_stat = 1 * 60
delta_t_stat = 0

Tcorr = {}
for i, rcv_id1 in enumerate(rcv_info["id"]):
    for j, rcv_id2 in enumerate(rcv_info["id"]):
        if i < j:
            rcv_couple = f"{rcv_id1}_{rcv_id2}"
            Tau_max_between_ships = np.max(
                np.fromiter(Tau[rcv_couple].values(), dtype=float)
            )
            window_corr = (
                Tau_max_between_ships + impulse_response_max_duration + delta_t_stat
            )
            # Round to closest minute
            # window_corr = np.ceil(window_corr / 60) * 60
            window_corr = np.ceil(window_corr)
            Tcorr[rcv_id1] = window_corr
            Tcorr[rcv_id2] = window_corr

# for j, mmsi in enumerate(mmsi_selected):
#     tau = np.max(np.fromiter(Tau[mmsi].values(), dtype=float))
#     Tcorr[mmsi] = tau + impulse_response_max_duration + delta_t_stat

print(Tcorr)
img_path_corr = os.path.join(img_path, f"tcorr_{window_corr}s")
if not os.path.exists(img_path_corr):
    os.makedirs(img_path_corr)

# ### Derive correlation using appropriate window size

# Compute the cross-correlation between the signals received by the two receivers for each ship
# First : derive STFT using appropriate window length derived before

stft_data_crosscorr = {}
for i, rcv_id in enumerate(station_id):
    stft_data_crosscorr[rcv_id] = {}
    for mmsi in mmsi_selected:
        data = wav_data[rcv_id][mmsi]
        fs = data["sig"].meta.sampling_rate

        max_tcorr = np.max(np.fromiter(Tcorr.values(), dtype=float))
        nperseg = np.floor(max_tcorr * fs).astype(int)  # Window length
        noverlap = int(nperseg * 1 / 2)  # Overlap length
        # print(f"Window length: {nperseg}")
        # print(f"Window duration: {nperseg / fs}s")

        # Derive stft
        f, tt, stft = sp.stft(
            data["data"],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft_data_crosscorr[rcv_id][mmsi] = {}
        stft_data_crosscorr[rcv_id][mmsi]["f"] = f
        stft_data_crosscorr[rcv_id][mmsi]["tt"] = tt
        stft_data_crosscorr[rcv_id][mmsi]["stft"] = stft

# Plot spectrograms

plot_spectrograms(
    wav_data=stft_data_crosscorr,
    ais_data=ais_data,
    intersection_time=intersection_time,
    fmin=fmin,
    fmax=fmax,
)

# Save figure
fig_name = f"stft_crosscorr"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


# Second : compute the cross-correlation between the two signals for each ship and each time window
s_xy = {}
coh_xy = {}
crosscorr_data = {}
for i, rcv_id1 in enumerate(rcv_info["id"]):
    for j, rcv_id2 in enumerate(rcv_info["id"]):
        # Intercorr for Rcvi / Rcvj is equivalent to Rcvj / Rcvi, only one couple needs to be evaluated
        if i < j:
            rcv_couple_id = f"{rcv_id1}_{rcv_id2}"
            s_xy[rcv_couple_id] = {}
            coh_xy[rcv_couple_id] = {}
            crosscorr_data[rcv_couple_id] = {}

            for mmsi in mmsi_selected:
                stft_1 = stft_data_crosscorr[rcv_id1][mmsi]["stft"]
                stft_2 = stft_data_crosscorr[rcv_id2][mmsi]["stft"]
                f = stft_data_crosscorr[rcv_id1][mmsi]["f"]
                tt = stft_data_crosscorr[rcv_id1][mmsi]["tt"]

                # Compute cross spectrum
                s_12 = stft_1 * np.conj(stft_2)
                s_11 = stft_1 * np.conj(stft_1)
                s_22 = stft_2 * np.conj(stft_2)

                # Store cross spectrum
                s_xy[rcv_couple_id][mmsi] = {}
                s_xy[rcv_couple_id][mmsi]["f"] = f
                s_xy[rcv_couple_id][mmsi]["tt"] = tt
                s_xy[rcv_couple_id][mmsi]["s_xy"] = s_12

                # Derive coherence
                coh_12 = np.abs(s_12) ** 2 / (np.abs(s_11) * np.abs(s_22))
                coh_xy[rcv_couple_id][mmsi] = {}
                coh_xy[rcv_couple_id][mmsi]["f"] = f
                coh_xy[rcv_couple_id][mmsi]["tt"] = tt
                coh_xy[rcv_couple_id][mmsi]["coh_xy"] = coh_12

                # Compute cross-correlation
                # Derive correlation lags
                lags = sp.correlation_lags(
                    len(np.fft.irfft(stft_1[:, 0])),
                    len(np.fft.irfft(stft_2[:, 0])),
                    mode="full",
                )

                c_xy = np.zeros((stft_1.shape[1], len(lags)), dtype=float)
                for k in range(len(tt)):
                    x = np.fft.irfft(stft_1[:, k])
                    y = np.fft.irfft(stft_2[:, k])
                    c_xy_k = sp.correlate(x, y, mode="full")
                    c_xy[k, :] = c_xy_k / np.max(np.abs(c_xy_k))

                # crosscorr = np.fft.irfft(s_12, axis=0)
                crosscorr_data[rcv_couple_id][mmsi] = {}
                crosscorr_data[rcv_couple_id][mmsi]["tt"] = tt
                crosscorr_data[rcv_couple_id][mmsi]["c_xy"] = c_xy
                crosscorr_data[rcv_couple_id][mmsi]["lags"] = lags * 1 / fs


# Third : plot the cross-correlation for each ship and each time window
n_couple = len(crosscorr_data.keys())
n_ships = len(crosscorr_data[list(crosscorr_data.keys())[0]].keys())
fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])
if n_ships == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(crosscorr_data.keys()):
    for j, mmsi in enumerate(mmsi_selected):

        df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
        ship_name = df_mmsi["shipName"].values[0]
        ax[0, j].set_title(f"{ship_name}")

        crosscorr = crosscorr_data[rcv_couple][mmsi]["c_xy"]
        lags = crosscorr_data[rcv_couple][mmsi]["lags"]
        tt = crosscorr_data[rcv_couple][mmsi]["tt"]
        tt_hour = tt / 3600
        im = ax[i, j].pcolormesh(
            lags, tt_hour, np.abs(crosscorr), cmap="jet", shading="gouraud"
        )

        ax[i, j].legend(loc="upper right", fontsize=12)

        # Add theoretical time delay
        tau_th = time_delay[rcv_couple][mmsi]
        time_along_traj = df_mmsi["time"].values

        # Convert time to seconds from the beginning of the trajectory
        time_along_traj = (
            (time_along_traj - time_along_traj[0])
            .astype("timedelta64[s]")
            .astype(float)
        )

        # Convert time to hours
        time_along_traj = time_along_traj / 3600

        # Plot the theoretical time delay
        ax[i, j].plot(
            tau_th,
            time_along_traj,
            color="r",
            linestyle="--",
            label="Theoretical time delay",
            zorder=2,
        )
        # Set xlims
        lag_max = 0.9 * np.max(np.abs(lags))
        ax[i, j].set_xlim(-lag_max, lag_max)

        # Add colorbar
        fig.colorbar(im, ax=ax[i, j])

fig.supxlabel("Lags [s]")
fig.supylabel("Time along trajectory [h]")

# Save figure
fig_name = f"crosscorr_vs_time"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


# Derive max and min tdoa instants
min_tdoa, max_tdoa = get_maxmin_tdoa(
    ais_data=ais_data, wav_data=crosscorr_data, tdoa=time_delay, verbose=True
)

# Plot one cross-correlation for each ship corresponding to the minimum time delay
n_ships = len(mmsi_selected)
n_couple = len(crosscorr_data.keys())
fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])
if n_ships == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(crosscorr_data.keys()):
    for j, mmsi in enumerate(crosscorr_data[rcv_couple].keys()):
        df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
        ship_name = df_mmsi["shipName"].values[0]
        lags = crosscorr_data[rcv_couple][mmsi]["lags"]

        ax[0, j].set_title(f"{ship_name}")
        ax[i, j].plot(
            lags,
            crosscorr_data[rcv_couple][mmsi]["c_xy"][
                min_tdoa[rcv_couple][mmsi]["idx_tt"], :
            ],
            # label=f"Ship {mmsi}",
        )

        ax[i, j].text(
            0.015,
            0.85,
            rcv_couple,
            transform=ax[i, j].transAxes,
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )

        ax[i, j].axvline(
            time_delay[rcv_couple][mmsi][min_tdoa[rcv_couple][mmsi]["idx"]],
            color="r",
            linestyle="--",
            label="Theoretical min TDOA",
        )

        # Set xlims
        lag_max = 0.9 * np.max(np.abs(lags))
        ax[i, j].set_xlim(-lag_max, lag_max)

    ax[0, j].legend(loc="upper right", fontsize=12)

fig.supxlabel("Lags [s]")
fig.supylabel(r"$C_{xy}$")

# Save figure
fig_name = "crosscorr_min_tdoa"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))

# Plot spectrograms
vmin = -100
vmax = 0

n_ships = len(mmsi_selected)
n_stations = len(rcv_info["id"])

fig, ax = plt.subplots(n_ships, n_stations, sharex=True, sharey=True)
for i, s_id in enumerate(station_id):
    for j, mmsi in enumerate(mmsi_selected):
        stft = stft_data_crosscorr[s_id][mmsi]["stft"]
        f = stft_data_crosscorr[s_id][mmsi]["f"]
        tt = stft_data_crosscorr[s_id][mmsi]["tt"]

        ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

        tt_hour = tt / 3600
        ax[j, i].pcolormesh(
            tt_hour,
            f,
            20 * np.log10(np.abs(stft)),
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )

        # ax[j, 0].set_ylabel("Frequency [Hz]")
        ax[j, i].text(
            0.02,
            0.9,
            ship_name,
            transform=ax[j, i].transAxes,
            ha="left",
            fontsize=12,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )
        ax[j, i].legend(loc="upper right", fontsize=7)

        # Add arrow pointing the center of the cross-corr window
        ax[j, i].annotate(
            "",
            xy=(tt_hour[min_tdoa[rcv_couple][mmsi]["idx_tt"]], 2),
            xytext=(tt_hour[min_tdoa[rcv_couple][mmsi]["idx_tt"]], 0),
            arrowprops=dict(facecolor="red", width=2),
            ha="center",
        )

    ax[0, i].set_title(f"Station {station_id[i]}")
    # ax[-1, i].set_xlabel("Time [h]")

fig.supxlabel("Time [h]")
fig.supylabel("Frequency [Hz]")
plt.ylim([fmin - 2, fmax + 2])
# plt.tight_layout()

# Save figure
fig_name = "cross_corr_min_tdoa_spectro"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


# Plot one cross-correlation for each ship corresponding to the maximum time delay
n_couple = len(crosscorr_data.keys())
n_ships = len(mmsi_selected)
fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(crosscorr_data.keys()):
    for j, mmsi in enumerate(crosscorr_data[rcv_couple].keys()):
        df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
        ship_name = df_mmsi["shipName"].values[0]
        ax[0, j].set_title(f"{ship_name}")

        lags = crosscorr_data[rcv_couple][mmsi]["lags"]
        ax[i, j].plot(
            lags,
            crosscorr_data[rcv_couple][mmsi]["c_xy"][
                max_tdoa[rcv_couple][mmsi]["idx_tt"], :
            ],
            label=f"Ship {mmsi}",
        )
        ax[i, j].text(
            0.015,
            0.85,
            rcv_couple,
            transform=ax[i, j].transAxes,
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )

        ax[i, j].axvline(
            time_delay[rcv_couple][mmsi][max_tdoa[rcv_couple][mmsi]["idx"]],
            color="r",
            linestyle="--",
            label="Theoretical max TDOA",
        )

        # Set xlims
        lag_max = 0.9 * np.max(np.abs(lags))
        ax[i, j].set_xlim(-lag_max, lag_max)

fig.supxlabel("Lags [s]")
fig.supylabel(r"$C_{xy}$")

# Save figure
fig_name = "crosscorr_max_tdoa"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))

# Plot corresponding time window over spectrogram for each ship and each receiver
# Plot spectrograms
vmin = -100
vmax = 0

fig, ax = plt.subplots(n_ships, n_stations, sharex=True, sharey=True)
for i, s_id in enumerate(station_id):
    for j, mmsi in enumerate(mmsi_selected):
        stft = stft_data_crosscorr[s_id][mmsi]["stft"]
        f = stft_data_crosscorr[s_id][mmsi]["f"]
        tt = stft_data_crosscorr[s_id][mmsi]["tt"]

        ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

        tt_hour = tt / 3600
        ax[j, i].pcolormesh(
            tt_hour,
            f,
            20 * np.log10(np.abs(stft)),
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )

        # ax[j, 0].set_ylabel("Frequency [Hz]")
        ax[j, i].text(
            0.02,
            0.9,
            ship_name,
            transform=ax[j, i].transAxes,
            ha="left",
            fontsize=12,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )
        ax[j, i].legend(loc="upper right", fontsize=7)

        # Add arrow pointing the center of the cross-corr window
        ax[j, i].annotate(
            "",
            xy=(tt_hour[max_tdoa[rcv_couple][mmsi]["idx"]], 2),
            xytext=(tt_hour[max_tdoa[rcv_couple][mmsi]["idx"]], 0),
            arrowprops=dict(facecolor="red", edgecolor="red", width=2),
            ha="center",
        )

    ax[0, i].set_title(f"Station {station_id[i]}")
    # ax[-1, i].set_xlabel("Time [h]")

fig.supxlabel("Time [h]")
fig.supylabel("Frequency [Hz]")
plt.ylim([fmin - 2, fmax + 2])
# plt.tight_layout()

# Save figure
fig_name = "cross_corr_max_tdoa_spectro"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


# Plot one cross-correlation for each ship corresponding to a given time along the trajectory
idx_time_trajectory = np.random.randint(
    0, crosscorr_data[rcv_couple][mmsi]["c_xy"].shape[0]
)

fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])
if n_ships == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(crosscorr_data.keys()):
    for j, mmsi in enumerate(crosscorr_data[rcv_couple].keys()):
        df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
        ship_name = df_mmsi["shipName"].values[0]
        ax[0, j].set_title(f"{ship_name}")

        lags = crosscorr_data[rcv_couple][mmsi]["lags"]
        ax[i, j].plot(
            lags,
            crosscorr_data[rcv_couple][mmsi]["c_xy"][idx_time_trajectory, :],
            label=f"Ship {mmsi}",
        )
        # ax[i, j].axvline(
        #     time_delay[rcv_couple][mmsi][idx_time_trajectory],
        #     color="r",
        #     linestyle="--",
        #     label="Theoretical TDOA",
        # )
        ax[i, j].text(
            0.015,
            0.85,
            rcv_couple,
            transform=ax[i, j].transAxes,
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )

fig.supxlabel("Lags [s]")
fig.supylabel(r"$C_{xy}$")

# Save figure
fig_name = "crosscorr_random_time"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))

# Plot corresponding time window over spectrogram for each ship and each receiver
# Plot spectrograms
vmin = -100
vmax = 0

fig, ax = plt.subplots(n_ships, n_stations, sharex=True, sharey=True)

for i, s_id in enumerate(station_id):
    for j, mmsi in enumerate(mmsi_selected):
        stft = stft_data_crosscorr[s_id][mmsi]["stft"]
        f = stft_data_crosscorr[s_id][mmsi]["f"]
        tt = stft_data_crosscorr[s_id][mmsi]["tt"]

        ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

        tt_hour = tt / 3600
        ax[j, i].pcolormesh(
            tt_hour,
            f,
            20 * np.log10(np.abs(stft)),
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )

        # ax[j, 0].set_ylabel("Frequency [Hz]")
        ax[j, i].text(
            0.02,
            0.9,
            ship_name,
            transform=ax[j, i].transAxes,
            ha="left",
            fontsize=12,
            bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
        )
        ax[j, i].legend(loc="upper right", fontsize=7)

        # Add arrow pointing the center of the cross-corr window
        ax[j, i].annotate(
            "",
            xy=(tt_hour[idx_time_trajectory], 2),
            xytext=(tt_hour[idx_time_trajectory], 0),
            arrowprops=dict(facecolor="red", edgecolor="k", width=2),
            ha="center",
        )

    ax[0, i].set_title(f"Station {station_id[i]}")
    # ax[-1, i].set_xlabel("Time [h]")

fig.supxlabel("Time [h]")
fig.supylabel("Frequency [Hz]")
plt.ylim([fmin - 2, fmax + 2])
# plt.tight_layout()

# Save figure
fig_name = "cross_corr_random_time_spectro"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


# Plot cross correlations for the intersection time (this is the most interesting time since we hope intercorrelation will have similar structure for both ships)
idx_time_trajectory = np.argmin(np.abs(tt - intersection_time[mmsi_selected[0]]))


# ======================================================================================================================
# DCF analysis
# ======================================================================================================================

# 1) Derive Double Cross-correlation Function (DCF) for two ships and a given couple of receivers
dcf_data = {}

for i, rcv_couple in enumerate(crosscorr_data.keys()):
    dcf_data[rcv_couple] = {}

    # So far lets assume we only have two ships selected for simplicity
    # We further asssume that ships have been selected properly so that they have intersecting routes
    mmsi1 = mmsi_selected[0]
    mmsi2 = mmsi_selected[1]
    corr_map_1 = crosscorr_data[rcv_couple][mmsi1]["c_xy"]
    corr_map_2 = crosscorr_data[rcv_couple][mmsi2]["c_xy"]

    dcf = np.zeros((corr_map_1.shape[0], corr_map_2.shape[0]), dtype=float)
    for itt_1 in range(corr_map_1.shape[0]):
        for itt_2 in range(corr_map_2.shape[0]):
            dcf_ij = np.sum(corr_map_1[itt_1, :] * corr_map_2[itt_2, :])

            autocorr_corr1 = np.sum(np.abs(corr_map_1[itt_1, :]) ** 2)
            autocorr_corr2 = np.sum(np.abs(corr_map_2[itt_2, :]) ** 2)
            norm = np.sqrt(autocorr_corr1 * autocorr_corr2)
            dcf_ij = dcf_ij / norm  # Values in [-1, 1]
            dcf_ij = (dcf_ij + 1) / 2  # Values in [0, 1]

            dcf[itt_1, itt_2] = dcf_ij

    dcf_data[rcv_couple]["dcf"] = dcf
    dcf_data[rcv_couple]["tt1"] = crosscorr_data[rcv_couple][mmsi1]["tt"]
    dcf_data[rcv_couple]["tt2"] = crosscorr_data[rcv_couple][mmsi2]["tt"]


# 2) Plot DCF
n_couple = len(dcf_data.keys())
fig, ax = plt.subplots(n_couple, 1, sharex=True, sharey=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(dcf_data.keys()):
    dcf = dcf_data[rcv_couple]["dcf"]
    tt1 = dcf_data[rcv_couple]["tt1"]
    tt2 = dcf_data[rcv_couple]["tt2"]

    # Derive vmin and vmax to have a better constrat
    dcf_no_nan = dcf[~np.isnan(dcf)]
    vmin = np.percentile(dcf_no_nan, 20)
    vmax = np.percentile(dcf_no_nan, 99.9)

    # im = ax[i].pcolormesh(tt1, tt2, dcf, cmap="jet", shading="gouraud")
    # Plot with imshow
    im = ax[i].imshow(
        dcf,
        cmap="jet",
        aspect="auto",
        origin="lower",
        extent=[tt1[0], tt1[-1], tt2[0], tt2[-1]],
        vmin=vmin,
        vmax=vmax,
    )

    ax[i].text(
        0.02,
        0.9,
        rcv_couple,
        transform=ax[i].transAxes,
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
    )

    # Add colorbar
    fig.colorbar(im, ax=ax[i])

fig.supxlabel(
    f"Time along trajectory of {ais_data[ais_data['mmsi'] == mmsi1]['shipName'].values[0]} [s]"
)
fig.supylabel(
    f"Time along trajectory of {ais_data[ais_data['mmsi'] == mmsi2]['shipName'].values[0]} [s]"
)


# Save figure
fig_name = "dcf"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))


n_couple = len(dcf_data.keys())
fig, ax = plt.subplots(n_couple, 1, sharex=True, sharey=True)

# Reshape ax if only one couple
if n_couple == 1:
    ax = np.array([ax])

for i, rcv_couple in enumerate(dcf_data.keys()):
    dcf = dcf_data[rcv_couple]["dcf"]
    tt1 = dcf_data[rcv_couple]["tt1"]
    tt2 = dcf_data[rcv_couple]["tt2"]

    # Derive vmin and vmax to have a better constrat
    dcf_no_nan = dcf[~np.isnan(dcf)]
    vmin = np.percentile(10 * np.log10(dcf_no_nan), 20)
    vmax = np.percentile(10 * np.log10(dcf_no_nan), 99.9)
    # vmax = 0

    # im = ax[i].pcolormesh(tt1, tt2, dcf, cmap="jet", shading="gouraud")
    # Plot with imshow
    im = ax[i].imshow(
        10 * np.log10(dcf),
        cmap="jet",
        aspect="auto",
        origin="lower",
        extent=[tt1[0], tt1[-1], tt2[0], tt2[-1]],
        vmin=vmin,
        vmax=vmax,
    )

    ax[i].text(
        0.02,
        0.9,
        rcv_couple,
        transform=ax[i].transAxes,
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
    )

    # Add colorbar
    fig.colorbar(im, ax=ax[i], label="DCF [dB]")

fig.supxlabel(
    f"Time along trajectory of {ais_data[ais_data['mmsi'] == mmsi1]['shipName'].values[0]} [s]"
)
fig.supylabel(
    f"Time along trajectory of {ais_data[ais_data['mmsi'] == mmsi2]['shipName'].values[0]} [s]"
)


# Save figure
fig_name = "dcf_log"
plt.savefig(os.path.join(img_path_corr, f"{fig_name}.png"))
