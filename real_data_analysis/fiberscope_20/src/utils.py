#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   read_tdms.py
@Time    :   2024/11/12 15:43:22
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import nptdms
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

import real_data_analysis.fiberscope_20.src.params as prms

from publication.publication_figure import PubFigure
from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
    FiberscopeSweep1,
    FiberscopeDynamicRecording,
)

PubFigure()


def load_fiberscope_data(file_path, subsampling_factor=None, calc_stft=False):
    # Load data from tdms file
    group_of_interrest = "Acquisition Hydros - Données"

    # Load data into dataframe
    usefull_props = {
        "channel": [
            "wf_increment",
            "wf_start_offset",
            "wf_samples",
            "wf_start_time",
        ],
        "group": ["Freq. ech", "Freq. ech. Unité", "Gamme", "Gamme Unité"],
    }
    with nptdms.TdmsFile.open(file_path) as tdms_file:
        df = tdms_file[group_of_interrest].as_dataframe(time_index=True)
        usefull_attrs = {
            prop: tdms_file[group_of_interrest].properties[prop]
            for prop in usefull_props["group"]
        }
        usefull_attrs.update(
            {
                prop: tdms_file[group_of_interrest]["Hydro1"].properties[prop]
                for prop in usefull_props["channel"]
            }
        )

    # Rename columns
    df.columns = [f"H{i}" for i in range(1, len(df.columns) + 1)]

    # Convert to xarray
    ds = df.to_xarray()

    # Rename index dimension into time
    ds = ds.rename({"index": "time"})

    # ds.drop_attrs("wf_increment")

    # Concatenate H1 to H5 into a single new varaible 'signal'
    ds = xr.concat([ds.H1, ds.H2, ds.H3, ds.H4, ds.H5], dim="h_index")
    ds = ds.to_dataset(name="signal")
    # Add new coordinates
    ds["h_index"] = [1, 2, 3, 4, 5]

    # Remove mean from signal to ensure that the signal is centered on 0
    ds["signal"] = ds.signal - ds.signal.mean("time")

    # Store usefull attributes in the dataset
    for key, value in usefull_attrs.items():
        ds.attrs[key] = value

    # Convert start time from datetime to str (For serialization to netCDF files)
    ds.attrs["wf_start_time"] = pd.to_datetime(ds.attrs["wf_start_time"]).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )

    # Rename attrs
    ds.attrs["ts"] = np.float64(ds.attrs["wf_increment"])
    ds.attrs["fs"] = 1 / ds.ts

    # Add attributes to dimensions
    ds["time"].attrs["long_name"] = r"$t$"
    ds["time"].attrs["unit"] = r"$\textrm{s}$"
    ds["h_index"].attrs["long_name"] = "Hydrophone index"
    ds["h_index"].attrs["unit"] = "index"

    # Add attributes to variables
    ds["signal"].attrs["long_name"] = r"$u$"
    ds["signal"].attrs["unit"] = r"$\textrm{V}$"

    # Apply subsampling is required
    if subsampling_factor is not None:
        # Subsample the signal
        ds = ds.isel(time=slice(0, None, subsampling_factor))
        # Update sampling frequency
        ds.attrs["ts"] = ds.ts * subsampling_factor
        ds.attrs["fs"] = 1 / ds.ts

    # Derive stft of the signal
    if calc_stft:
        stft = []
        nperseg = 2**14
        noverlap = 2**13
        for idx in ds.h_index:
            ff, tt, stft_i = sp.stft(
                ds.signal.sel(h_index=idx),
                fs=ds.fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
            )
            stft.append(stft_i)

        # Add stft to dataset as two new variables (amplitude and phase to avoid complex values)
        stft = np.array(stft)
        ds["ff"] = ff
        ds["tt"] = tt
        ds["stft_amp"] = (
            ["h_index", "ff", "tt"],
            np.abs(stft),
        )
        ds["stft_phase"] = (
            ["h_index", "ff", "tt"],
            np.angle(stft),
        )

    return ds


# Define the detection function
def direct_arrival_env(sig, fs, c, plot=False, save=False, root_img=None):
    # Calcul de l'enveloppe du signal
    sig_h = sp.hilbert(sig)
    sig_env = np.abs(sig_h)
    # sig_phase = np.angle(sig_h)

    # On défini un seuil à 200 % du niveau de bruit
    d_min = 2
    td_min = d_min / c
    noise_lvl = sig_env[0 : int(td_min * fs)].max()
    env_detection_th = 2 * noise_lvl

    # Recherche du premier pic au dessus du seuil
    idx_peak = np.where(sig_env > env_detection_th)[0][0]
    # On définit un intervalle de reherche autour du pic
    delta_idx_peak = 10
    idx_search_start = max(0, idx_peak - delta_idx_peak)
    t_win_inf = sig.time[idx_search_start]
    idx_search_stop = min(len(sig_env), idx_peak + delta_idx_peak)
    t_win_max = sig.time[idx_search_stop]

    # On calcul la dérivée de l'enveloppe pour trouver le maximum
    ts = 1 / fs
    sig_env_deriv = np.gradient(sig_env, ts)
    idx_peak_refined = (
        np.argmax(sig_env_deriv[idx_search_start:idx_search_stop]) + idx_search_start
    )
    t_arrival = sig.time[idx_peak_refined]

    if plot:
        plt.figure()
        plt.plot(sig.time, sig_env)
        plt.axhline(env_detection_th, color="red", linestyle="--")
        plt.xlim([0, sig.time[idx_search_stop] * 1.2])
        # plt.axvline(sig.time[idx_peak], color="green", linestyle="--")
        plt.axvline(sig.time[idx_peak_refined], color="green", linestyle="--")
        plt.axvline(sig.time[idx_search_start], color="orange", linestyle="--")
        plt.axvline(sig.time[idx_search_stop], color="orange", linestyle="--")
        plt.title("Enveloppe")

        if save:
            fpath = os.path.join(root_img, f"H{sig.h_index.values}_env.png")
            plt.savefig(fpath)

        plt.figure()
        sig.plot()
        plt.xlim([0, sig.time[idx_search_stop] * 1.2])
        # plt.axvline(sig.time[idx_peak], color="green", linestyle="--")
        plt.axvline(t_arrival, color="green", linestyle="--")
        plt.axvline(t_win_inf, color="orange", linestyle="--")
        plt.axvline(t_win_max, color="orange", linestyle="--")
        plt.title("Temps d'arrivée")

        if save:
            fpath = os.path.join(root_img, f"H{sig.h_index.values}_td.png")
            plt.savefig(fpath)

    return t_arrival


def all_arrivals_static(
    mode="first_pulse",
    records_to_process=[],
    fs_sweep_info=FiberscopeSweep1(),
    hydro_apriori_pos=None,
    reject_outlayers=False,
    outlayer_tol=0.1,
    c=1500,
):
    # Estimation du temps de propagation direct pour chacune des positions de la sources e

    # Define the source signal
    tr = fs_sweep_info.interp_pulse_period
    n_sweep = fs_sweep_info.n_sweep

    # Init dictionnary
    ds_rtf_fpath = os.path.join(
        prms.root_data, "static", f"{records_to_process[0]}_rtf.nc"
    )
    ds_rtf = xr.open_dataset(ds_rtf_fpath)

    src_pos_labels = np.unique(
        [record_name.split("_")[-4] for record_name in records_to_process]
    )
    rcv_labels = [f"H{idx_rcv}" for idx_rcv in ds_rtf.h_index.values]
    obs = {
        rcv_l: {
            src_pos_l: {"td": [], "td_var": [], "d2": [], "d2_var": []}
            for src_pos_l in src_pos_labels
        }
        for rcv_l in rcv_labels
    }

    dict_th_pos = prms.dict_th_pos
    all_dists = np.array([dict_th_pos[pos] for pos in dict_th_pos.keys()])
    idx_pos_sort = np.argsort(all_dists)
    src_labels = list(dict_th_pos.keys())
    src_pos_coords = {
        src_pos_l: [dict_th_pos[src_pos_l], 0] for src_pos_l in src_labels
    }
    src_label_sorted = [src_labels[i] for i in idx_pos_sort]

    for idx_rcv in ds_rtf.h_index.values:

        # Iterate overs source positions
        for record_name in records_to_process:
            # Load data
            ds_rtf_fpath = os.path.join(
                prms.root_data, "static", f"{record_name}_rtf.nc"
            )
            ds_rtf = xr.open_dataset(ds_rtf_fpath)

            # Get received signal
            sig = ds_rtf.signal.sel(h_index=idx_rcv)

            src_pos_label = record_name.split("_")[-4]
            rcv_label = f"H{idx_rcv}"

            # Approach 0 : Use only first LFM
            if mode == "first_pulse":
                sig_pulse = sig.sel(
                    time=slice(0, tr)
                )  # Extract first pulse (t in [0, tr])

                td = direct_arrival_env(
                    sig=sig_pulse, fs=fs_sweep_info.fs, c=c, plot=False
                )
                d2 = (td * c) ** 2
                # print(f"Time of arrival (direct) = {td} s")
                # print(f"Distance (direct) = {d} m")

                # Store results
                obs[rcv_label][src_pos_label]["td"].append(td)
                obs[rcv_label][src_pos_label]["td_var"].append(1)
                obs[rcv_label][src_pos_label]["d2"].append(d2)
                obs[rcv_label][src_pos_label]["d2_var"].append(1)

            # Approach 1 : detect each pulse arrival and derive the mean td
            if mode == "mean_over_all_pulses":
                t_dir = []
                if reject_outlayers:
                    d_apriori = np.sqrt(
                        (
                            hydro_apriori_pos[f"H{idx_rcv}"][0]
                            - src_pos_coords[src_pos_label][0]
                        )
                        ** 2
                        + (
                            hydro_apriori_pos[f"H{idx_rcv}"][1]
                            - src_pos_coords[src_pos_label][1]
                        )
                        ** 2
                    )
                    td_apriori = d_apriori / c

                for ipulse in range(n_sweep):
                    sig_pulse = sig.sel(time=slice(ipulse * tr, (ipulse + 1) * tr))
                    # print(sig_pulse.time)

                    td = direct_arrival_env(
                        sig=sig_pulse, fs=fs_sweep_info.fs, c=c, plot=False
                    )
                    td = td.values - ipulse * tr

                    if reject_outlayers:
                        if np.abs(td - td_apriori) / td_apriori < outlayer_tol:
                            t_dir.append(td)
                        else:
                            print(np.abs(td - td_apriori) / td_apriori)
                    else:
                        t_dir.append(td)

                t_dir = np.array(t_dir)
                if t_dir.shape[0] == 0:
                    print(f"H{idx_rcv}, {record_name}: {t_dir.shape}")
                    print(f"td : {td}, td_apriori : {td_apriori}")
                td = np.mean(t_dir)
                td_var = np.var(t_dir)
                d2 = (td * c) ** 2
                d2_var = np.var((t_dir * c) ** 2)
                # Store results
                obs[rcv_label][src_pos_label]["td"].append(td)
                obs[rcv_label][src_pos_label]["td_var"].append(td_var)
                obs[rcv_label][src_pos_label]["d2"].append(d2)
                obs[rcv_label][src_pos_label]["d2_var"].append(d2_var)

            # Approach 2 : detect each pulse arrival and derive the mean td
            if mode == "all_pulses":
                t_dir = []
                for ipulse in range(n_sweep):
                    sig_pulse = sig.sel(time=slice(ipulse * tr, (ipulse + 1) * tr))
                    # print(sig_pulse.time)

                    td = direct_arrival_env(
                        sig=sig_pulse, fs=fs_sweep_info.fs, c=c, plot=False
                    )
                    td = td.values - ipulse * tr
                    t_dir.append(td)

                td = np.array(t_dir)
                # td = np.mean(t_dir)
                # td_var = np.var(t_dir) * np.ones_like(td)
                td_var = (t_dir - np.mean(t_dir)) ** 2
                d2 = (td * c) ** 2
                # d2_var = np.var(d2) * np.ones_like(d2)
                d2_var = (d2 - np.mean(d2)) ** 2

                # Store results
                obs[rcv_label][src_pos_label]["td"].append(td)
                obs[rcv_label][src_pos_label]["td_var"].append(td_var)
                obs[rcv_label][src_pos_label]["d2"].append(d2)
                obs[rcv_label][src_pos_label]["d2_var"].append(d2_var)

            # Approach 3
            if mode == "single_pulse":
                t_dir = []
                d_apriori = np.sqrt(
                    (
                        hydro_apriori_pos[f"H{idx_rcv}"][0]
                        - src_pos_coords[src_pos_label][0]
                    )
                    ** 2
                    + (
                        hydro_apriori_pos[f"H{idx_rcv}"][1]
                        - src_pos_coords[src_pos_label][1]
                    )
                    ** 2
                )
                td_apriori = d_apriori / c

                for ipulse in range(n_sweep):
                    sig_pulse = sig.sel(time=slice(ipulse * tr, (ipulse + 1) * tr))
                    # print(sig_pulse.time)

                    td = direct_arrival_env(
                        sig=sig_pulse, fs=fs_sweep_info.fs, c=c, plot=False
                    )
                    td = td.values - ipulse * tr

                    if np.abs(td - td_apriori) / td_apriori < outlayer_tol:
                        t_dir.append(td)
                    else:
                        print(np.abs(td - td_apriori) / td_apriori)

                td = t_dir[0]
                td_var = 1
                d2 = (td * c) ** 2
                d2_var = 1
                # Store results
                obs[rcv_label][src_pos_label]["td"].append(td)
                obs[rcv_label][src_pos_label]["td_var"].append(td_var)
                obs[rcv_label][src_pos_label]["d2"].append(d2)
                obs[rcv_label][src_pos_label]["d2_var"].append(d2_var)

    # print(obs)

    # Convert to np arrays in the right order in order to use it in the least-squares
    d2_obs = {rcv_label: [] for rcv_label in obs.keys()}
    d2_obs_var = {rcv_label: [] for rcv_label in obs.keys()}
    t_obs = {rcv_label: [] for rcv_label in obs.keys()}
    t_obs_var = {rcv_label: [] for rcv_label in obs.keys()}

    # Define the order based on the source position : from closest to farthest

    for rcv_label in d2_obs.keys():
        d2_obs_rcv = []
        d2_obs_var_rcv = []
        t_obs_rcv = []
        t_obs_var_rcv = []
        for src_label in src_label_sorted:
            d2_obs_rcv.append(obs[rcv_label][src_label]["d2"])
            d2_obs_var_rcv.append(obs[rcv_label][src_label]["d2_var"])
            t_obs_rcv.append(obs[rcv_label][src_label]["td"])
            t_obs_var_rcv.append(obs[rcv_label][src_label]["td_var"])

        d2_obs[rcv_label] = np.array(d2_obs_rcv)
        d2_obs_var[rcv_label] = np.array(d2_obs_var_rcv)
        t_obs[rcv_label] = np.array(t_obs_rcv)
        t_obs_var[rcv_label] = np.array(t_obs_var_rcv)

    return t_obs, d2_obs, t_obs_var, d2_obs_var


def all_arrivals_dynamic(
    xr_data,
    fs_sweep_info=FiberscopeDynamicRecording().signal,
    c=1500,
):
    """Estimation du temps de propagation direct pour chacune des positions successives de la sources en mouvement"""

    # Paramètres de la source
    tr = fs_sweep_info.interp_pulse_period
    n_sweep = fs_sweep_info.n_sweep

    # Init dictionnary
    rcv_labels = [f"H{idx_rcv}" for idx_rcv in xr_data.h_index.values]
    d2_obs = {rcv_label: [] for rcv_label in rcv_labels}
    t_obs = {rcv_label: [] for rcv_label in rcv_labels}

    for idx_rcv in xr_data.h_index.values:

        rcv_label = f"H{idx_rcv}"
        # Get received signal for current hydrophone
        sig = xr_data.signal.sel(h_index=idx_rcv)

        # Process each pulse
        t_dir = []
        dist2 = []
        for ipulse in range(n_sweep):

            sig_pulse = sig.sel(time=slice(ipulse * tr, (ipulse + 1) * tr))
            td = direct_arrival_env(sig=sig_pulse, fs=fs_sweep_info.fs, c=c, plot=False)
            td = td.values - ipulse * tr
            d2 = (td * c) ** 2

            # Store results
            t_dir.append(td)
            dist2.append(d2)

        # Store results
        t_obs[rcv_label] = np.array(t_dir)
        d2_obs[rcv_label] = np.array(dist2)

    return t_obs, d2_obs


if __name__ == "__main__":
    data_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\Fiberscope_campagne_oct_2024"
    date = "09-10-2024"
    data_path = os.path.join(data_root, f"Campagne_{date}")

    file_name = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34.tdms"
    file_path = os.path.join(data_path, file_name)

    data = load_fiberscope_data(file_path)
    plt.figure()
    data.signal.plot(x="time", hue="h_index")
    plt.show()

    print(data)
