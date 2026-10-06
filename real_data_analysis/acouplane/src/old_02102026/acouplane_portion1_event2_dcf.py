# %%
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

project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
sys.path.append(
    project_root
)  # \real_data_analysis\fiberscope_groix\src\data_processing
root_data = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\acouplane\data"
)
root_bathy_data = os.path.join(project_root, "data", "bathy")
from publication.publication_figure import PubFigure, set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.old_02102026.utils import *

PubFigure()

# %% [markdown]
# # Illustration de la portion1 event2

# %% [markdown]
# ## Chargement des données

# %%
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

# %%
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

# %%
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

# %%
start_dt = event_2_pos.time.values[0]
start_dt = pd.to_datetime(str(start_dt)).strftime(datetime_format)
end_dt = event_2_pos.time.values[-1]
end_dt = pd.to_datetime(str(end_dt)).strftime(datetime_format)
print(f"Event sequence 2 : from {start_dt} to {end_dt}")
analysis_duration_s = 30 * 60

start_dt = "2026-02-24_22-00-00"
# ds_wav_ = ds_wav.sel(obs_id=[4, 7])
plot_sequence(ds_wav, start_dt, analysis_duration_s, nperseg=2**7, alpha_overlap=0.75)

# %% [markdown]
# ## Calcul des temps d'arrivée

# %%
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
# plt.show()
# plt.close("all")

# %% [markdown]
# Les données des capteurs OBS b et c ont des très faibles SNR, en particulier en début de séquence.

# %%
# Conversion en dataset xarray
arr_dt = np.array(arrivals_dt)
ds_arr = xr.Dataset(
    data_vars=dict(arr_dt=(["obs_id", "pulse_id"], arr_dt)),
    coords=dict(obs_id=ds_wav.obs_id.values, pulse_id=np.arange(arr_dt.shape[1])),
)

# %% [markdown]
# ## Calcul Sxy SCOT


# %%
def compute_sxy_scot(
    ds_arr,
    signals,
    times,
    num_obs,
    den_obs,
    ref_obs=None,
    window_mode="common",
    fs=2000,
    offset_s=0.1,
    pulse_len_s=None,
    nfft=2**16,
    n_smooth=5,
):
    """Estimate the Sxy_scot = S_{num,den} / sqrt(S_{num,num} S_{den,den}) for each pulse.


    Parameters
    ----------
    ds_arr, signals, times, fs, offset_s, pulse_len_s, window_mode, ref_obs
        See `extract_pulse_windows`.
    num_obs : list of str
        Sensors in the numerator (e.g. [4, 1, 6]).
    den_obs : str
        Sensor in the denominator (e.g. 7).
    nfft : int
        FFT length (zero-padding if larger than the window).
    n_smooth : int
        Length of the moving average applied to the RTF magnitude in dB.

    Returns
    -------
    _type_
        _description_

    """

    all_obs = list(dict.fromkeys([*num_obs, den_obs, *(ref_obs or [])]))
    windows, idx_start = extract_pulse_windows(
        ds_arr,
        {o: signals[o] for o in all_obs},
        times,
        fs,
        offset_s,
        pulse_len_s,
        window_mode=window_mode,
        ref_obs=ref_obs,
    )

    freq = np.fft.rfftfreq(nfft, d=1 / fs)
    fft = {o: np.fft.rfft(windows[o], n=nfft, axis=-1) for o in all_obs}

    # Auto-spectrums
    s_den = fft[den_obs] * np.conj(fft[den_obs])
    sxy_scot = np.stack(
        [
            (fft[o] * np.conj(fft[den_obs])) / np.sqrt(s_den * fft[o] * np.conj(fft[o]))
            for o in num_obs
        ]
    )

    rtf_id = [f"{o}/{den_obs}" for o in num_obs]
    dims = ("rtf_id", "pulse_id", "freq")
    return xr.Dataset(
        data_vars=dict(
            sxy_scot=(dims, sxy_scot),
            idx_start=(
                ("obs_id", "pulse_id"),
                np.stack([idx_start[o] for o in all_obs]),
            ),
        ),
        coords=dict(
            rtf_id=rtf_id, obs_id=all_obs, pulse_id=ds_arr.pulse_id.values, freq=freq
        ),
        attrs=dict(
            fs=fs,
            offset_s=offset_s,
            nfft=nfft,
            den_obs=den_obs,
            window_mode=window_mode,
            n_smooth=n_smooth,
        ),
    )


# %%
def plot_sxy_map(
    ds_dcf,
    rtf_id,
    window_mode,
    freq_lim=(10, 800),
    percentiles=(5, 95),
    cmap="magma",
    ax=None,
):
    """Plot the sxy magnitude [dB] as a (pulse_id, frequency) map.

    Parameters
    ----------
    ds_dcf : xr.Dataset
        Output of `compute_sxy_scot`, concatenated along "window_mode".
    rtf_id : str
        RTF to display (e.g. "4/7").
    window_mode : str
        Window option to display ("common" or "shifted").
    freq_lim : tuple of float
        Frequency limits [Hz] of the plot.
    percentiles : tuple of float
        Percentiles of the data used as color limits.
    """
    da_real = np.real(ds_dcf["sxy_scot"].sel(rtf_id=rtf_id))
    da_imag = np.imag(ds_dcf["sxy_scot"].sel(rtf_id=rtf_id))

    vmin, vmax = np.nanpercentile(da_real.values, percentiles)

    _, axs = plt.subplots(1, 2, figsize=(16, 8))

    im = axs[0].pcolormesh(
        da_real["freq"],
        da_real["pulse_id"],
        da_real.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    im = axs[1].pcolormesh(
        da_imag["freq"],
        da_imag["pulse_id"],
        da_imag.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    for ax in axs.flatten():
        ax.set_xlim(*freq_lim)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Replica ID")
        ax.set_title(f"RTF {rtf_id}")
        ax.figure.colorbar(im, ax=ax, label="Magnitude [dB]")

    return ax


# %%
obs_ids = ds_wav.obs_id.values  #
signals_d = {4: signals[0], 7: signals[1], 1: signals[2], 6: signals[5]}
times_d = {4: sig_dt[0], 7: sig_dt[1], 1: sig_dt[2], 6: sig_dt[5]}


# # Ref = 1
# common_kwargs = dict(
#     num_obs=[7, 6, 4],
#     den_obs=1,
#     fs=2000,
#     offset_s=0.1,
#     pulse_len_s=interval_inter_pulse_s,
#     ref_obs=[4],
#     nfft = 2**14
# )

# # Ref = 7
# common_kwargs = dict(
#     num_obs=[1, 6, 4],
#     den_obs=7,
#     fs=2000,
#     offset_s=0.1,
#     pulse_len_s=interval_inter_pulse_s,
#     ref_obs=[4],
#     nfft=2**14,
# )

# Ref = 4
common_kwargs = dict(
    num_obs=[1, 6, 7],
    den_obs=4,
    fs=2000,
    offset_s=0.1,
    pulse_len_s=interval_inter_pulse_s,
    ref_obs=[4],
    nfft=2**14,
)

# # Ref = 6
# common_kwargs = dict(
#     num_obs=[1, 4, 7],
#     den_obs=6,
#     fs=2000,
#     offset_s=0.1,
#     pulse_len_s=interval_inter_pulse_s,
#     ref_obs=[4],
#     nfft=2**14,
# )

# Option 1: common window / Option 2: shifted windows
ds_dcf = compute_sxy_scot(
    ds_arr,
    signals_d,
    times_d,
    window_mode="shifted",
    **common_kwargs,
)


# %%
plt.close("all")
for rtf_id in ds_dcf.rtf_id.values:
    plot_sxy_map(ds_dcf, str(rtf_id), "shifted")

plt.show()

# %%
np.real(ds_dcf.sxy_scot.sel(rtf_id="7/4").sel(pulse_id=0)).plot(x="freq")
