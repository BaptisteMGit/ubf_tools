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
import shutil
import nptdms
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy import stats

from misc import progression_bar
from publication.publication_figure import PubFigure
from real_data_analysis.fiberscope_20.src.fiberscope_manager import FiberscopeManager
from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
    FiberscopeSweep1,
    FiberscopeDynamicRecording,
)
from propa.rtf.rtf_utils import D_hermitian_angle_fast

import real_data_analysis.fiberscope_20.src.params as prms

PubFigure()


# ======================================================================================================================
# General
# ======================================================================================================================
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


# ======================================================================================================================
# RTF-MFP localisation
# ======================================================================================================================
def optimum_stft_params(
    fs_sweep,
    fsm,
    fs,
    root_results,
    signal_duration=1,
    rtf_estimators=["cs", "cs-evd"],
    h_index_ref=5,  # Index of the receference receiver (H1 to H5)
    idx_rcv_ref=4,  # Corresponding position in the datasets (usually it should be h_index_ref -1 because index list starts at 0)
    theta_statistics="mean",
):
    """
    The goal of this function is to find the optimum couple of STFT params (nperseg, alpha_overlap) to estimate RTF
    vectors.
    To do so each potential params combination is tested to estimate RTF and the distance between estimated RTF and
    reference one (given by deconvolution) is computed.
    The couple of params minimizing the distance can then be used to compute RTF for the dynamic recording.
    """
    # Define the range of the params to study
    ns = int(signal_duration * fs)  # Number of sample in the signal
    # Arbitrary but we now that stft snapshots should be longer than impulse response to ensure the multiplicative
    # transfert function assumption holds
    n_stft_pow2_min = 10
    # STFT snapshot length can not exceed the total siganl snapshot duration
    n_stft_pow2_max = int(np.log2(ns))
    # Test all snapshot length between min and max
    n_stft_pow2 = np.arange(n_stft_pow2_min, n_stft_pow2_max + 1)
    # Overlap factor to test (usual sample cov matrix estimation assumes alpha_ov=0 for independence of segments)
    ov_factors = np.array([0, 0.25, 0.5, 0.75, 0.9])

    # To src position label in sorted order (dummy it just from 1 to 6)
    dict_th_pos = prms.dict_th_pos
    all_dists = np.array([dict_th_pos[pos] for pos in dict_th_pos.keys()])
    idx_pos_sort = np.argsort(all_dists)
    src_labels = list(dict_th_pos.keys())
    src_label_sorted = [src_labels[i] for i in idx_pos_sort]

    # Use recordings with higher SNR (is it the most representative ? )
    static_records = fs_sweep.records_N1

    # Distance to use settings
    dist_func = D_hermitian_angle_fast
    ax_f = 1
    ax_rcv = 0
    # Use mean value assuming normal distribution of thetas
    if theta_statistics == "mean":
        apply_mean = True
    # Use real theta distribution to derive the statstical expectation of the distribution
    elif theta_statistics == "expectation":
        apply_mean = False

    dist_kwargs = {
        "ax_f": ax_f,
        "unit": "deg",
        "ax_rcv": ax_rcv,
        "apply_mean": apply_mean,
        "apply_median": False,
    }

    # Prep variable to store results
    theta_datasets = {rtf_e: None for rtf_e in rtf_estimators}
    n_test = n_stft_pow2.size * ov_factors.size

    # Folder to store temporary files
    root_tmp = os.path.join(root_results, "tmp")
    if not os.path.exists(root_tmp):
        os.makedirs(root_tmp)

    for rtf_e in rtf_estimators:
        i_test = 0
        prev_progress = 0
        print("\n--------------------------------")
        print(f"{rtf_e.upper()} estimator")
        # Define the file path to save results
        res_fpath = os.path.join(
            root_results, f"theta_all_static_pos_Tw{signal_duration}_{rtf_e}.nc"
        )

        theta_dict = {s_lab: [] for s_lab in src_label_sorted}

        first_iter = True
        fs_sweep.records_folder = root_tmp

        # Iterate over n_stft values
        for n_stft_p2 in n_stft_pow2:
            # Iterate over overlapping factor
            for ov_factor in ov_factors:

                prev_progress = progression_bar(
                    index=i_test,
                    index0=0,
                    indexf=n_test - 1,
                    prev_progress=prev_progress,
                )

                fsm.nperseg = 2**n_stft_p2
                fsm.noverlap = int(fsm.nperseg * ov_factor)
                # print(f"Test {i_test+1}/{n_test} : (nperseg = 2**{int(np.log2(fsm.nperseg))}, nov = {fsm.noverlap})")

                # Update the covariance and rtf managers using current nperseg and noverlap params
                fsm.set_managers(fs=fs, idx_rcv_ref=idx_rcv_ref)

                if first_iter:
                    # t0 = time()

                    fsm.process_static_analysis(
                        static_signal=fs_sweep,
                        static_records_names=static_records,
                        set_stft_props=False,
                        rtf_estimator=rtf_e,
                    )

                    # Compare RTF estimated by deconvolution and by the RTF estimator (CS or CS-EVD)
                    for record_name in static_records:

                        ds_rtf_fpath = os.path.join(
                            fs_sweep.records_folder, f"{record_name}_rtf.nc"
                        )
                        ds_rtf = xr.open_dataset(ds_rtf_fpath)

                        # Build RTF ref (from deconvolution)
                        rtf_ref = ds_rtf.rtf_amp * np.exp(1j * ds_rtf.rtf_phase)

                        # Build RTF hat
                        rtf_hat = ds_rtf.rtf_amp_hat * np.exp(1j * ds_rtf.rtf_phase_hat)

                        # Interp RTF ref at RTF hat frequencies (nearest neigbor)
                        rtf_ref = rtf_ref.sel(
                            f_ir=rtf_hat.f_rtf.values, method="nearest"
                        )

                        # Compute theta distance over the frequency band of interest
                        theta = dist_func(
                            rtf_ref=rtf_ref.values,
                            rtf=rtf_hat.values,
                            **dist_kwargs,
                        )

                        theta_c = get_theta_c(val=theta)

                        # Store distance
                        src_pos_label = record_name.split("_")[-4]
                        theta_dict[src_pos_label].append(theta_c)

                    first_iter = False

                else:
                    # t0 = time()

                    for (
                        record_name
                    ) in (
                        static_records
                    ):  # Avoid re-computing the ref rtf by deconvolution
                        ds_rtf = fsm.derive_feature(
                            recording_name=record_name,
                            records_folder=fs_sweep.records_folder,
                            signal=fs_sweep,
                            save=False,
                            rtf_estimator=rtf_e,
                        )

                        # Build RTF ref (from deconvolution)
                        rtf_ref = ds_rtf.rtf_amp * np.exp(1j * ds_rtf.rtf_phase)

                        # Build RTF hat
                        rtf_hat = ds_rtf.rtf_amp_hat * np.exp(1j * ds_rtf.rtf_phase_hat)

                        # Interp RTF ref at RTF hat frequencies (nearest neigbor)
                        rtf_ref = rtf_ref.sel(
                            f_ir=rtf_hat.f_rtf.values, method="nearest"
                        )

                        # Compute mean theta distance over the frequency band of interest
                        theta = dist_func(
                            rtf_ref=rtf_ref.values,
                            rtf=rtf_hat.values,
                            **dist_kwargs,
                        )

                        theta_c = get_theta_c(val=theta)

                        # Store distance
                        src_pos_label = record_name.split("_")[-4]
                        theta_dict[src_pos_label].append(theta_c)

                # To unasynchroneous issues
                ds_rtf.close()
                del ds_rtf

                i_test += 1

        # Save
        theta_all_pos = np.array([theta_dict[l] for l in theta_dict.keys()])
        alpha_ovv, n_pow2_pp = np.meshgrid(ov_factors, n_stft_pow2)

        # pos_id is the source position index (sorted by increasing distance)
        # test_id is the index of the STFT parameter test (combination of nperseg and overlap)
        xr_theta = xr.Dataset(
            data_vars=dict(
                theta=(["pos_id", "test_id"], theta_all_pos),
                n_stft_pow2=(["test_id"], n_pow2_pp.flatten()),
                alpha_ov=(["test_id"], alpha_ovv.flatten()),
            ),
            coords=dict(
                pos_id=[int(pl[1]) for pl in theta_dict.keys()],
                test_id=np.arange(n_pow2_pp.size),
            ),
        )

        # Add attributes
        rtf_estimator_minimization_label = rtf_e.upper().replace("-", "_")
        xr_theta.attrs = {
            "description": f"Theta distance between RTF estimated by deconvolution and by {rtf_estimator_minimization_label}, for different STFT parameters",
            "rtf_estimator": rtf_e,
            "h_index_ref": h_index_ref,
            "signal_level": static_records[0].split("_")[2],
        }

        xr_theta.to_netcdf(res_fpath)

        # Store in dict
        theta_datasets[rtf_e] = xr_theta
        xr_theta.close()
        del xr_theta

    # Delete root_tmp and its content to save space
    shutil.rmtree(root_tmp)

    return theta_datasets


def get_theta_c(val, apply_mean):
    # We dont have anything to do we can store the mean value directly
    if apply_mean:
        theta_c = val

    # We need to derive expectation
    else:
        # Step 1: estimate the probability density function associate to the observed distribution
        kde = stats.gaussian_kde(val)
        # Step 2: derive expectation    (note: kde.evaluate(x) is 10 times faster than kde.pdf(x))
        expectation = np.sum(val * kde.evaluate(val)) / np.sum(kde.evaluate(val))
        theta_c = expectation

    return theta_c


def apply_rtf_mfp(
    n_lfm,
    nperseg,
    noverlap,
    root_data,
    root_results,
    rtf_estimator="cs-evd",
    theta_statistics="mean",
    h_index_ref=5,
    verbose=False,
):

    # Initialisation
    fs_sweep1 = FiberscopeSweep1()
    fs_dynamic = FiberscopeDynamicRecording()
    fsm = FiberscopeManager(
        root_processed_data=root_data,
        h_index_ref=h_index_ref,
        plot_feature=False,
        theta_statistics=theta_statistics,
    )

    # Set folders to store processed files
    fs_sweep1.records_folder = os.path.join(root_data, rtf_estimator, "static_to_loc")
    if not os.path.exists(fs_sweep1.records_folder):
        os.makedirs(fs_sweep1.records_folder)

    fsm.root_processed_data = os.path.join(root_data, rtf_estimator)
    if not os.path.exists(fsm.root_processed_data):
        os.makedirs(fsm.root_processed_data)

    data_dynamic = xr.open_dataset(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_20\data\dynamic\10-10-2024T16-53-43-200271_PR_N1_346.nc"
    )

    # Set stft params
    fsm.nperseg = nperseg
    fsm.noverlap = noverlap
    if verbose:
        print(f"(nperseg = 2**{int(np.log2(fsm.nperseg))}, nov = {fsm.noverlap})")

    # Calcul des RTFs pour les positions statiques -> vecteurs de la librairie
    fsm.process_static_analysis(
        static_signal=fs_sweep1,
        static_records_names=fs_sweep1.records_N1,
        set_stft_props=False,
        rtf_estimator=rtf_estimator,
    )

    # Définition des segments du signal dynamique à analyser
    fsm.presplit_dynamic_record(
        fs_dynamic_recording=fs_dynamic,
        n_sweep=n_lfm,
        t_max=np.max(data_dynamic.time.values),
    )

    # Découpage du signal et création des datasets associés à chaque segment
    fsm.split_dynamic_record(fs_dynamic_recording=fs_dynamic, force_reload=False)

    # Calcul des RTFs de chaque segment
    fsm.process_dyn_analysis(
        fs_dynamic_recording=fs_dynamic,
        use_global_noise_csdm=False,
        set_stft_props=False,
        rtf_estimator=rtf_estimator,
    )

    # Localisation : calcul de l'angle hermitien pour chaque segment et chaque position de la librairie
    d = fsm.localize_dyn_recording(
        static_signal=fs_sweep1,
        static_records_names=fs_sweep1.records_N1,
        fs_dynamic_recording=fs_dynamic,
    )

    # Order d
    dict_th_pos = prms.dict_th_pos
    all_dists = np.array([dict_th_pos[pos] for pos in dict_th_pos.keys()])
    idx_pos_sort = np.argsort(all_dists)

    distmap = d[idx_pos_sort, :]
    # Normalize each line
    distmap_q = -distmap
    axis_norm = 1
    d_max = np.tile(
        np.max(distmap_q, axis=axis_norm), (distmap_q.shape[axis_norm], 1)
    )  # Cast to d shape
    d_min = np.tile(np.min(distmap_q, axis=axis_norm), (distmap_q.shape[axis_norm], 1))
    d_max = d_max.T
    d_min = d_min.T
    q = (distmap_q - d_min) / (d_max - d_min)
    # In dB
    q[q == 0] = 1e-6
    q_dB = 10 * np.log10(q)

    rtf_mfp_distance_map = xr.Dataset(
        data_vars=dict(
            theta=(["idx_ref_position", "time"], distmap),
            q_dB=(["idx_ref_position", "time"], q_dB),
        ),
        coords=dict(
            idx_ref_position=np.arange(d.shape[0]),
            time=np.arange(0, d.shape[1]) * fs_dynamic.time_step,
        ),
        attrs=dict(
            rtf_estimator=rtf_estimator,
            theta_statistics=theta_statistics,
        ),
    )

    rtf_mfp_distance_map.theta.attrs["unit"] = "°"
    rtf_mfp_distance_map.theta.attrs["long_name"] = r"$\theta$"
    rtf_mfp_distance_map.q_dB.attrs["unit"] = "dB"
    rtf_mfp_distance_map.q_dB.attrs["long_name"] = "q"
    rtf_mfp_distance_map.time.attrs["unit"] = "s"
    rtf_mfp_distance_map.time.attrs["long_name"] = "Time"

    # Save
    signal_duration = n_lfm * fs_sweep1.interp_pulse_period
    fpath = os.path.join(
        root_results,
        f"rtf_mfp_distance_map_Tw{signal_duration}_{theta_statistics}_{rtf_estimator}.nc",
    )
    rtf_mfp_distance_map.to_netcdf(fpath)

    return rtf_mfp_distance_map


# ======================================================================================================================
# TOA localisation
# ======================================================================================================================


# Define the detection function
def direct_arrival_env(sig, fs, c, plot=False, save=False, root_img=None, idx_rcv="?"):
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
            fpath = os.path.join(root_img, f"H{idx_rcv}_env.png")
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
            fpath = os.path.join(root_img, f"H{idx_rcv}_td.png")
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


# Apply to our problem
def f_d2(X, k, dp=5):
    """
    Parameters
    ----------
    X : array
        [r, z] position of the hydrophone to estimate
    k : array
        index of the position of the source from 0 to N-1 (N-1 = 5 here)
    dp : float
        distance between two consecutive source positions
    """
    r, z = X
    alpha_k = k * dp
    # y = (r + alpha_k)**2 + z**2
    y = (alpha_k - r) ** 2 + z**2
    return y


def f_d2_no_z(X, k, dp=5):
    """
    Parameters
    ----------
    X : array
        [r] position of the hydrophone to estimate
    k : array
        index of the position of the source from 0 to N-1 (N-1 = 5 here)
    dp : float
        distance between two consecutive source positions
    """
    r = X
    alpha_k = k * dp
    # y = (r + alpha_k)**2 + z**2
    y = (alpha_k - r) ** 2
    return y


def jac_d2(X, k, dp=5):
    """
    Parameters
    ----------
    X : array
        [r, z] position of the hydrophone to estimate
    k : array
        index of the position of the source from 0 to N-1 (N-1 = 5 here)
    dp : float
        distance between two consecutive source positions
    """

    r, z = X
    alpha_k = k * dp
    # df_dr = 2*(r + alpha_k)
    df_dr = 2 * (r - alpha_k)
    df_dz = 2 * z * np.ones_like(k)
    J = np.array([df_dr, df_dz]).T
    return J


def jac_d2_no_z(X, k, dp=5):
    """
    Parameters
    ----------
    X : array
        [r] position of the hydrophone to estimate
    k : array
        index of the position of the source from 0 to N-1 (N-1 = 5 here)
    dp : float
        distance between two consecutive source positions
    """

    r = X
    alpha_k = k * dp
    # df_dr = 2*(r + alpha_k)
    df_dr = 2 * (r - alpha_k)
    J = np.array([df_dr]).T
    return J


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
