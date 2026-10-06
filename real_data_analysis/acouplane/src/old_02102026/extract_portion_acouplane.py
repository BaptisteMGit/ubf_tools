#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   Untitled-1
@Time    :   2026/09/29 08:57:41
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from publication.publication_figure import PubFigure, set_subfigures_abc_labels
from real_data_analysis.acouplane.src.old_02102026.utils import *

PubFigure()

project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
root_data = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\acouplane\data"
)
root_bathy_data = os.path.join(project_root, "data", "bathy")


# ======================================================================================================================
# Chargement des données
# ======================================================================================================================
# Read wav dataset
ds_wav = xr.open_dataset(wav_dataset_fpath)
ds_wav

# Read source pos dataset
ds_src_pos = xr.open_dataset(source_position_dataset_fpath)
ds_src_pos

# Read OBS pos
df_obs_pos = pd.read_csv(
    obs_position_fpath,
    dtype={
        "obs_id": str,
        "lon": np.float64,
        "lat": np.float64,
    },
)
df_obs_pos

# Tracé des positions de la source pour identification d'une portion favorable

plt.figure()
for obs_id in df_obs_pos.obs_id:
    obs = df_obs_pos.loc[df_obs_pos["obs_id"] == obs_id]
    plt.scatter(obs.lon, obs.lat, marker="d", label=f"{obs.obs_id.values[0].upper()}")

sc = plt.scatter(
    ds_src_pos.lon,
    ds_src_pos.lat,
    cmap="jet",
    c=np.arange(ds_src_pos.time.size),
    marker="x",
    s=10,
)
plt.colorbar(sc, label="Source position")
plt.legend()
plt.xlabel("Longitude [°]")
plt.ylabel("Latitude [°]")

plt.close("all")

# L'objectif est de reproduire un cadre similaire à celui de l'article de Verlinden et al. 2015 :
#
# * Une trajectoire *library*
# * Une ou plusieurs trajectoires *event*

# OPTION 1 : croisement le plus proche (pour l'instant pas la donnée dispo)
lib_traj = {
    "start": 21900,
    "end": 22500,
    # "start": 35000,
    # "end": 4500,
}
event_1_traj = {
    "start": 2000,
    "end": 2800,
}
event_2_traj = {
    "start": 79300,
    "end": 80200,
}

# # OPTION 2 : croisement un peu plus loin
# lib_traj = {
#     "start": 54000,
#     "end": 55000,
# }
# event_1_traj = {
#     "start": 69800,
#     "end": 70800,
# }
# # event_1_traj = {
# #     "start": 65000,
# #     "end": 66000,
# # }
# event_2_traj = {
#     "start": 72300,
#     "end": 73300,
# }

plt.figure()
for obs_id in df_obs_pos.obs_id:
    obs = df_obs_pos.loc[df_obs_pos["obs_id"] == obs_id]
    plt.scatter(obs.lon, obs.lat, marker="d", label=f"{obs.obs_id.values[0].upper()}")

sc = plt.scatter(
    ds_src_pos.lon,
    ds_src_pos.lat,
    cmap="jet",
    c=np.arange(ds_src_pos.time.size),
    marker="x",
    s=10,
)
# Library
lib_pos = ds_src_pos.isel(time=slice(lib_traj["start"], lib_traj["end"]))
plt.scatter(lib_pos.lon, lib_pos.lat, marker="x", color="k", label="library")
# Event
event_1_pos = ds_src_pos.isel(time=slice(event_1_traj["start"], event_1_traj["end"]))
plt.scatter(event_1_pos.lon, event_1_pos.lat, marker="o", color="r", label="event 1")
# Event 2
event_2_pos = ds_src_pos.isel(time=slice(event_2_traj["start"], event_2_traj["end"]))
plt.scatter(event_2_pos.lon, event_2_pos.lat, marker="o", color="m", label="event 2")
plt.colorbar(sc, label="Source position")
plt.legend()
plt.xlabel("Longitude [°]")
plt.ylabel("Latitude [°]")
# plt.show()


plt.figure()
for obs_id in df_obs_pos.obs_id:
    obs = df_obs_pos.loc[df_obs_pos["obs_id"] == obs_id]
    plt.scatter(obs.lon, obs.lat, marker="d", label=f"{obs.obs_id.values[0].upper()}")

plt.scatter(lib_pos.lon, lib_pos.lat, marker="x", color="k", label="library")
plt.scatter(event_1_pos.lon, event_1_pos.lat, marker="o", color="r", label="event 1")
plt.scatter(event_2_pos.lon, event_2_pos.lat, marker="o", color="m", label="event 2")
plt.legend()
plt.xlabel("Longitude [°]")
plt.ylabel("Latitude [°]")
plt.show()
# plt.close("all")

# # Représentation des signaux associés à chaque segment

# ## Library

start_dt = lib_pos.time.values[0]
end_dt = lib_pos.time.values[-1]
print(f"Library sequence : from {start_dt} to {end_dt}")
# analysis_duration_s = 10 * 60

# plot_sequence(
#     ds_wav, start_analysis_dt, analysis_duration_s, nperseg=2**7, alpha_overlap=0.75
# )


# ## Event 1
start_dt = event_1_pos.time.values[0]
start_dt = pd.to_datetime(str(start_dt)).strftime(datetime_format)
end_dt = event_1_pos.time.values[-1]
end_dt = pd.to_datetime(str(end_dt)).strftime(datetime_format)
print(f"Event sequence 1 : from {start_dt} to {end_dt}")
# analysis_duration_s = 10 * 60

# plot_sequence(ds_wav, start_dt, analysis_duration_s, nperseg=2**7, alpha_overlap=0.75)

# ## Event 2
start_dt = event_2_pos.time.values[0]
start_dt = pd.to_datetime(str(start_dt)).strftime(datetime_format)
end_dt = event_2_pos.time.values[-1]
end_dt = pd.to_datetime(str(end_dt)).strftime(datetime_format)
print(f"Event sequence 2 : from {start_dt} to {end_dt}")
analysis_duration_s = 60 * 60

start_dt = "2026-02-24_21-30-00"
ds_wav_ = ds_wav.sel(obs_id=[4, 7])
plot_sequence(ds_wav_, start_dt, analysis_duration_s, nperseg=2**7, alpha_overlap=0.75)

# plt.show()
plt.close("all")
# ## Détection des arrivées

# ### STA / LTA


# interval_inter_pulse_s = 3
# sta_duration = 0.45
# lta_duration = 3
# sta_lta_threshold_obs4 = 0.5
# sta_lta_threshold_obs7 = 0.65

# sig_dt, signals, arrivals_dt, arrivals_dt_idx = apply_sta_lta(
#     ds_wav,
#     sta_duration,
#     lta_duration,
#     start_analysis_dt,
#     analysis_duration_s,
#     sta_lta_threshold_obs4,
#     sta_lta_threshold_obs7,
# )

interval_inter_pulse_s = 3
sta_duration = 0.45
lta_duration = 3
sta_lta_threshold_obs4 = 0.5
sta_lta_threshold_obs7 = 0.5
sta_lta_threshold_obsa = 0.5
sta_lta_threshold_obsb = 0.5
sta_lta_threshold_obsc = 0.5
sta_lta_threshold_obsd = 0.5

thresholds = [
    sta_lta_threshold_obs4,
    sta_lta_threshold_obs7,
    sta_lta_threshold_obsa,
    sta_lta_threshold_obsb,
    sta_lta_threshold_obsc,
    sta_lta_threshold_obsd,
]

sig_dt, signals, arrivals_dt, arrivals_dt_idx = apply_sta_lta_only_first(
    ds_wav,
    sta_duration,
    lta_duration,
    start_dt,
    analysis_duration_s,
    thresholds,
    pulse_interval_s=interval_inter_pulse_s,
    fs=ds_wav.fs.isel(obs_id=0).values,
)
plt.show()
# plt.close("all")


# Attention, il y a une ambiguïté pour associer les paires de réceptions, il faut vérifier à l'aide des données de positions.

# # NOTE : Fix temporaire à modifier --> marche pas vraiment il faut attendre d'avoir les temps d'émission
min_size = np.min([len(arrivals_dt[i]) for i in range(len(arrivals_dt))])
for i in range(len(arrivals_dt)):
    if len(arrivals_dt[i]) > min_size:
        arrivals_dt[i] = arrivals_dt[i][len(arrivals_dt[i]) - min_size :]

# Conversion en dataset xarray
arr_dt = np.array(arrivals_dt)
ds_arr = xr.Dataset(
    data_vars=dict(arr_dt=(["obs_id", "pulse_id"], arr_dt)),
    coords=dict(obs_id=ds_wav.obs_id.values, pulse_id=np.arange(arr_dt.shape[1])),
)


obs_ids = ds_wav.obs_id.values  #
signals_d = {4: signals[0], 7: signals[1], 1: signals[2], 6: signals[5]}
times_d = {4: sig_dt[0], 7: sig_dt[1], 1: sig_dt[2], 6: sig_dt[5]}

common_kwargs = dict(
    num_obs=[1, 6, 7],
    den_obs=4,
    fs=2000,
    offset_s=0.1,
    pulse_len_s=interval_inter_pulse_s,
    ref_obs=[4],
    nfft=2**13,
)

# Option 1: common window / Option 2: shifted windows
ds_opt1 = compute_rtf(
    ds_arr,
    signals_d,
    times_d,
    window_mode="common",
    **common_kwargs,
)
ds_opt2 = compute_rtf(
    ds_arr,
    signals_d,
    times_d,
    window_mode="shifted",
    **common_kwargs,
)

# Stack both options along a new dimension
ds_rtf = xr.concat(
    [ds_opt1, ds_opt2], dim=pd.Index(["common", "shifted"], name="window_mode")
)

# Magnitude difference between the two options (should be ~0 dB)
diff_db = ds_rtf["gamma"].sel(window_mode="shifted") - ds_rtf["gamma"].sel(
    window_mode="common"
)

# Plot RTFs of pulse 0 for both options
ds_rtf["gamma"].sel(pulse_id=0, rtf_id="7/4").plot.line(
    x="freq", hue="window_mode", xlim=(200, 500)
)
# plt.show()


# Plot 2D map and difference
# plot_rtf_map(ds_rtf, "4/6", "shifted")
# plot_rtf_diff(ds_rtf, "4/6", "shifted", "common")

for rtf_id in ds_rtf.rtf_id.values:
    plot_rtf_map(ds_rtf, str(rtf_id), "shifted")

plt.show()
