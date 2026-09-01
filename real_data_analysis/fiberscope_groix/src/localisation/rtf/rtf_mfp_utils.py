#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_utils.py
@Time    :   2026/02/09 16:11:11
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp

from datetime import datetime, timedelta
from propa.rtf.rtf_utils import D_hermitian_angle_fast

# from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from real_data_analysis.fiberscope_groix.src.fiberscope_groix_manager import (
    ActiveFiberscopeManager,
    PassiveFiberscopeManager,
    BandFilter,
)

from real_data_analysis.fiberscope_groix.src.localisation.rtf.rtf_mfp_animation_utils import (
    rtf_mfp_animation,
)

# def get_dists(fsm, df_arr, seq_id_ref, seq_id, fmin=600, fmax=800):

#     dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}

#     fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
#     xr_seq_ref = xr.open_dataset(fpath)

#     df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
#     # Define the pulse id to use (easiest case where we only consider a single pulse within the selected sequence)
#     psnr_tot = (
#         df_seq_ref["psnr_obs1"] + df_seq_ref["psnr_obs2"] + df_seq_ref["psnr_obs3"]
#     )
#     pulse_id_to_use = df_seq_ref["pulse_id"].iloc[np.nanargmax(psnr_tot)]
#     print(f"Library {seq_id_ref} -> used pulse = {pulse_id_to_use}")

#     ref_pos_e = (
#         df_seq_ref["emission_interp_e_gps"]
#         .loc[df_seq_ref["pulse_id"] == pulse_id_to_use]
#         .values
#     )
#     ref_pos_n = (
#         df_seq_ref["emission_interp_n_gps"]
#         .loc[df_seq_ref["pulse_id"] == pulse_id_to_use]
#         .values
#     )

#     # Load rtf data
#     fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id}_rtf.nc")
#     xr_seq_i = xr.open_dataset(fpath)
#     df_seq_i = df_arr.loc[df_arr["sequence_id"] == seq_id]
#     df_seq_i = df_seq_i.loc[df_seq_i["pulse_id"].isin(xr_seq_i.pulse_id.values)]

#     # -----------------------------------------
#     # 1) Evaluate spatial variations
#     # -----------------------------------------
#     # Emission positions in the sequence
#     pos_e_pulse = df_seq_i["emission_interp_e_gps"]
#     pos_n_pulse = df_seq_i["emission_interp_n_gps"]

#     # Compute distance to all
#     spatial_dist = np.sqrt(
#         (ref_pos_e - pos_e_pulse) ** 2 + (ref_pos_n - pos_n_pulse) ** 2
#     )

#     # -----------------------------------------
#     # 2) Evaluate RTF variations
#     # -----------------------------------------
#     # Reference RTF <-> first pulse in sequence
#     xr_seq_i_ref = xr_seq_ref.sel(pulse_id=pulse_id_to_use)
#     rtf_ref = xr_seq_i_ref.rtf_amp_hat * np.exp(1j * xr_seq_i_ref.rtf_phase_hat)

#     # Rtf for each pulse
#     rtf_pulse = xr_seq_i.rtf_amp_hat * np.exp(1j * xr_seq_i.rtf_phase_hat)

#     # Limit frequency band
#     rtf_ref = rtf_ref.sel(f_rtf=slice(fmin, fmax))
#     rtf_pulse = rtf_pulse.sel(f_rtf=slice(fmin, fmax))

#     # Compute hermitian angle distance
#     theta_dist = []
#     for pulse_id in xr_seq_i.pulse_id.values:
#         rtf_pulse_i = rtf_pulse.sel(pulse_id=pulse_id)
#         theta = D_hermitian_angle_fast(
#             rtf_ref=rtf_ref.values, rtf=rtf_pulse_i.values, **dist_kwargs
#         )

#         theta_dist.append(theta)

#     # print(spatial_dist, theta_dist)
#     spatial_dist = np.array(spatial_dist)
#     theta_dist = np.array(theta_dist)

#     return theta_dist, spatial_dist, xr_seq_i, df_seq_i


def derive_doppler(ds_gps, df_seq, f0):

    # Get receiver positions
    rcv_keys = ["obs1", "obs2", "obs3"]
    rcv_e = np.array([ds_gps.attrs[f"{k}_e_apriori"] for k in rcv_keys])
    rcv_n = np.array([ds_gps.attrs[f"{k}_n_apriori"] for k in rcv_keys])
    x_rcv = np.vstack((rcv_e, rcv_n))

    x_src_rcv = []
    v_src = []
    v_src_to_rcv = []
    v_src_to_rcv_norm = []
    shifted_freq = []

    # Iterate over successive positions of the sequence to evaluate
    pulse_ids = df_seq["pulse_id"].values
    for pulse_id in pulse_ids:
        df_seq_i_pulse = df_seq.loc[df_seq["pulse_id"] == pulse_id]

        # -----------------------------------------
        # 1) Derive vector connecting source to receiver
        # -----------------------------------------
        # Emission positions for the current pulse
        pos_i_e = df_seq_i_pulse["emission_interp_e_gps"].values
        pos_i_n = df_seq_i_pulse["emission_interp_n_gps"].values
        x_s = np.vstack((pos_i_e, pos_i_n))

        # Vector from source to receiver (for each receiver)
        x_sr = x_rcv - x_s
        x_sr_norm = np.linalg.norm(x_sr, axis=0, keepdims=True)

        # -----------------------------------------
        # 2) Extract source speed vector
        # -----------------------------------------
        # Source speed for the current pulse
        pos_i_ve = df_seq_i_pulse["emission_interp_ve_gps"].values
        pos_i_vn = df_seq_i_pulse["emission_interp_vn_gps"].values
        v_s = np.vstack((pos_i_ve, pos_i_vn))

        # -----------------------------------------
        # 3) Projection of Vs on X_sr
        # -----------------------------------------
        v_sr = x_sr.T @ v_s
        v_sr_norm = v_sr * 1 / x_sr_norm.T

        # -----------------------------------------
        # 3) Compute doppler shifted freq
        # -----------------------------------------
        c0 = 1500
        f_p = f0 * 1 / (1 - v_sr_norm / c0)

        # print(v_sr_norm)
        x_src_rcv.append(x_sr)
        v_src.append(v_s)
        v_src_to_rcv.append(v_sr)
        v_src_to_rcv_norm.append(v_sr_norm)
        shifted_freq.append(f_p)

    x_src_rcv = np.array(x_src_rcv)
    v_src = np.array(v_src)
    v_src_to_rcv = np.array(v_src_to_rcv)
    v_src_to_rcv_norm = np.array(v_src_to_rcv_norm)
    shifted_freq = np.array(shifted_freq)

    # # print("ok")
    # # Array of shape (n_pulse, n_pulse_ref)
    # rtf_distances = np.array(rtf_distances)
    # spatial_distances = np.array(spatial_distances)

    return x_src_rcv, v_src, v_src_to_rcv, v_src_to_rcv_norm, shifted_freq


def derive_doppler_jules(ds_gps, f0):

    # Get receiver positions
    rcv_keys = ["obs1", "obs2", "obs3"]
    rcv_e = np.array([ds_gps.attrs[f"{k}_e_apriori"] for k in rcv_keys])
    rcv_n = np.array([ds_gps.attrs[f"{k}_n_apriori"] for k in rcv_keys])
    x_rcv = np.vstack((rcv_e, rcv_n))

    x_src_rcv = []
    v_src = []
    v_src_to_rcv = []
    v_src_to_rcv_norm = []
    shifted_freq = []

    # Iterate over successive positions the jules

    for t in ds_gps.time.values:

        # -----------------------------------------
        # 1) Derive vector connecting source to receiver
        # -----------------------------------------
        # Emission positions for the current pulse
        pos_i_e = ds_gps.e.sel(time=t).values
        pos_i_n = ds_gps.n.sel(time=t).values
        x_s = np.vstack((pos_i_e, pos_i_n))

        # Vector from source to receiver (for each receiver)
        x_sr = x_rcv - x_s
        x_sr_norm = np.linalg.norm(x_sr, axis=0, keepdims=True)

        # -----------------------------------------
        # 2) Extract source speed vector
        # -----------------------------------------
        # Source speed for the current pulse
        pos_i_ve = ds_gps.v_e.sel(time=t).values
        pos_i_vn = ds_gps.v_n.sel(time=t).values
        v_s = np.vstack((pos_i_ve, pos_i_vn))

        # -----------------------------------------
        # 3) Projection of Vs on X_sr
        # -----------------------------------------
        v_sr = x_sr.T @ v_s
        v_sr_norm = v_sr * 1 / x_sr_norm.T

        # -----------------------------------------
        # 3) Compute doppler shifted freq
        # -----------------------------------------
        c0 = 1500
        f_p = f0 * 1 / (1 - v_sr_norm / c0)

        # print(v_sr_norm)
        x_src_rcv.append(x_sr)
        v_src.append(v_s)
        v_src_to_rcv.append(v_sr)
        v_src_to_rcv_norm.append(v_sr_norm)
        shifted_freq.append(f_p)

    x_src_rcv = np.array(x_src_rcv)
    v_src = np.array(v_src)
    v_src_to_rcv = np.array(v_src_to_rcv)
    v_src_to_rcv_norm = np.array(v_src_to_rcv_norm)
    shifted_freq = np.array(shifted_freq)

    return x_src_rcv, v_src, v_src_to_rcv, v_src_to_rcv_norm, shifted_freq


def get_dists_2(
    fsm, df_arr, seq_id_ref, seq_id, fmin=600, fmax=800, dist_type="hermitian_angle"
):

    if dist_type == "hermitian_angle":
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        print("Compute distance using hermitian angle distance (in C^N)")
    elif dist_type == "euclidean":
        print("Compute distance using euclidean distance (in C^N)")
    elif dist_type == "euclidean_module":
        print(
            "Compute distance using euclidean distance on the module of RTF vectors (in R^N)"
        )
    elif dist_type == "euclidean_phase":
        print(
            "Compute distance using euclidean distance on the phase of RTF vectors (in R^N)"
        )
    else:
        print("Warning : unknown distance, set to default -> hermitian angle")
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        print("Compute distance using hermitian angle distance (in C^N)")

    fpath = os.path.join(
        fsm.root_data_sequence, "active", f"sequence_{seq_id_ref}_rtf.nc"
    )
    xr_seq_ref = xr.open_dataset(fpath)

    df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
    df_seq_ref = df_seq_ref.loc[
        df_seq_ref["pulse_id"].isin(xr_seq_ref.pulse_id.values)
    ]  # keep only detected pulse

    ref_pos_e = df_seq_ref["emission_interp_e_gps"].values
    ref_pos_n = df_seq_ref["emission_interp_n_gps"].values

    # Load rtf data
    fpath = os.path.join(fsm.root_data_sequence, "active", f"sequence_{seq_id}_rtf.nc")
    xr_seq_i = xr.open_dataset(fpath)
    df_seq_i = df_arr.loc[df_arr["sequence_id"] == seq_id]
    df_seq_i = df_seq_i.loc[df_seq_i["pulse_id"].isin(xr_seq_i.pulse_id.values)]

    # Reference RTFs
    rtf_ref = xr_seq_ref.rtf_amp_hat * np.exp(1j * xr_seq_ref.rtf_phase_hat)

    rtf_distances = []
    spatial_distances = []

    # Iterate over successive positions of the sequence to evaluate
    for pulse_id in xr_seq_i.pulse_id.values:
        df_seq_i_pulse = df_seq_i.loc[df_seq_i["pulse_id"] == pulse_id]

        # -----------------------------------------
        # 1) Evaluate spatial variations
        # -----------------------------------------
        # Emission positions for the current pulse
        pos_e_pulse = df_seq_i_pulse["emission_interp_e_gps"].values
        pos_n_pulse = df_seq_i_pulse["emission_interp_n_gps"].values

        # Compute distance to all reference positions
        spatial_dist = np.sqrt(
            (ref_pos_e - pos_e_pulse) ** 2 + (ref_pos_n - pos_n_pulse) ** 2
        )

        # -----------------------------------------
        # 2) Evaluate RTF variations
        # -----------------------------------------

        # Rtf for current pulse
        xr_seq_i_pulse = xr_seq_i.sel(pulse_id=pulse_id)
        rtf_pulse = xr_seq_i_pulse.rtf_amp_hat * np.exp(
            1j * xr_seq_i_pulse.rtf_phase_hat
        )

        # Limit frequency band
        rtf_ref = rtf_ref.sel(f_rtf=slice(fmin, fmax))
        rtf_pulse = rtf_pulse.sel(f_rtf=slice(fmin, fmax))

        rtf_dist = []

        if dist_type == "hermitian_angle":
            # Compute hermitian angle distance
            for pulse_id_ref in xr_seq_ref.pulse_id.values:
                rtf_ref_pulse_i = rtf_ref.sel(pulse_id=pulse_id_ref)
                theta_pulse_i = D_hermitian_angle_fast(
                    rtf_ref=rtf_ref_pulse_i.values, rtf=rtf_pulse.values, **dist_kwargs
                )

                rtf_dist.append(theta_pulse_i)

        elif dist_type == "euclidean":
            # Compute eulcidean distance
            for pulse_id_ref in xr_seq_ref.pulse_id.values:
                rtf_ref_pulse_i = rtf_ref.sel(pulse_id=pulse_id_ref)
                d_euc = np.mean(
                    np.linalg.norm(rtf_ref_pulse_i.values - rtf_pulse.values, axis=1)
                )
                rtf_dist.append(d_euc)

        elif dist_type == "euclidean_module":
            # Compute eulcidean distance using only the module
            for pulse_id_ref in xr_seq_ref.pulse_id.values:
                rtf_ref_pulse_i = rtf_ref.sel(pulse_id=pulse_id_ref)
                d_euc = np.mean(
                    np.linalg.norm(
                        np.abs(rtf_ref_pulse_i.values) - np.abs(rtf_pulse.values),
                        axis=1,
                    )
                )
                rtf_dist.append(d_euc)

        elif dist_type == "euclidean_phase":
            # Compute eulcidean distance using only the phase
            for pulse_id_ref in xr_seq_ref.pulse_id.values:
                rtf_ref_pulse_i = rtf_ref.sel(pulse_id=pulse_id_ref)
                d_euc = np.mean(
                    np.linalg.norm(
                        np.angle(rtf_ref_pulse_i.values) - np.angle(rtf_pulse.values),
                        axis=1,
                    )
                )
                rtf_dist.append(d_euc)

        rtf_dist = np.array(rtf_dist)

        # Store
        spatial_distances.append(spatial_dist)
        rtf_distances.append(rtf_dist)

    # print("ok")
    # Array of shape (n_pulse, n_pulse_ref)
    rtf_distances = np.array(rtf_distances)
    spatial_distances = np.array(spatial_distances)

    return (
        rtf_distances,
        spatial_distances,
        xr_seq_ref,
        df_seq_ref,
        xr_seq_i,
        df_seq_i,
    )


def get_dists_2_passive_event(
    fsm,
    df_arr_library,
    seq_id_ref,
    ds_gps,
    t_start,
    t_end,
    fmin=600,
    fmax=800,
    dist_type="hermitian_angle",
    use_weighted_mean=False,
    verbose=False,
):

    if dist_type == "hermitian_angle":
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        comment = "Compute distance using hermitian angle distance (in C^N)"
        # print("Compute distance using hermitian angle distance (in C^N)")
    elif dist_type == "euclidean":
        comment = "Compute distance using euclidean distance (in C^N)"
        # print("Compute distance using euclidean distance (in C^N)")
    elif dist_type == "euclidean_module":
        comment = "Compute distance using euclidean distance on the module of RTF vectors (in R^N)"
        # print(
        #     "Compute distance using euclidean distance on the module of RTF vectors (in R^N)"
        # )
    elif dist_type == "euclidean_phase":
        comment = "Compute distance using euclidean distance on the phase of RTF vectors (in R^N)"
        # print(
        #     "Compute distance using euclidean distance on the phase of RTF vectors (in R^N)"
        # )
    else:
        print("Warning : unknown distance, set to default -> hermitian angle")
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        comment = "Compute distance using hermitian angle distance (in C^N)"
        # print("Compute distance using hermitian angle distance (in C^N)")

    if verbose:
        print(comment)

    fpath = os.path.join(
        fsm.root_data_sequence, "active", f"sequence_{seq_id_ref}_rtf.nc"
    )
    xr_seq_ref = xr.open_dataset(fpath)
    df_seq_ref = df_arr_library.loc[df_arr_library["sequence_id"] == seq_id_ref]
    df_seq_ref = df_seq_ref.loc[
        df_seq_ref["pulse_id"].isin(xr_seq_ref.pulse_id.values)
    ]  # keep only detected pulse
    xr_seq_ref = xr_seq_ref.sel(pulse_id=df_seq_ref["pulse_id"].values)

    ref_pos_e = df_seq_ref["emission_interp_e_gps"].values
    ref_pos_n = df_seq_ref["emission_interp_n_gps"].values

    # Load rtf data
    record_id = f"passive_{datetime.strftime(t_start, fsm.datetime_fmt)}_to_{datetime.strftime(t_end, fsm.datetime_fmt)}"
    fpath = os.path.join(
        fsm.root_data_sequence, "passive", f"sequence_{record_id}_rtf.nc"
    )
    xr_passive = xr.open_dataset(fpath)

    # Define comon frequency band to use
    fmin_common_band = max(fmin, max(xr_seq_ref.f_rtf.min(), xr_passive.f_rtf.min()))
    fmax_common_band = min(fmax, min(xr_seq_ref.f_rtf.max(), xr_passive.f_rtf.max()))

    # Slice common band
    xr_seq_ref = xr_seq_ref.sel(f_rtf=slice(fmin_common_band, fmax_common_band))
    xr_passive = xr_passive.sel(f_rtf=slice(fmin_common_band, fmax_common_band))

    # Reference RTFs
    rtf_ref = xr_seq_ref.rtf_amp_hat * np.exp(1j * xr_seq_ref.rtf_phase_hat)

    rtf_distances = []
    spatial_distances = []

    # t_start = datetime.strptime(xr_passive.t_start, xr_passive.datetime_format)

    rtf_passive = xr_passive.rtf_amp_hat * np.exp(1j * xr_passive.rtf_phase_hat)
    rtf_passive_4d = rtf_passive.values[..., np.newaxis]  # Shape (nrcv, nf, nseg, 1)

    if use_weighted_mean:
        start_seg = 0
        end_seg = xr_passive.analysis_segment_duration
        segment_shift = xr_passive.analysis_segment_duration * (
            1 - xr_passive.analysis_segment_alpha_overlap
        )

        event_weights = []

        for i_seg in range(xr_passive.sizes["segment_dt"]):
            # Get signal corresponding to the current segment
            passive_sig_seg = xr_passive.signal.sel(time=slice(start_seg, end_seg))
            # Compute weights using signal from the reference receiver
            passive_sig_seg_rcv_ref = passive_sig_seg.sel(
                h_index=xr_passive.h_index_ref
            )

            # Compute PSD of the signal
            ff, Pxx_seg = sp.welch(
                passive_sig_seg_rcv_ref.values,
                fs=xr_passive.fs,
                nperseg=2
                ** 13,  # TODO save those params in xr_passive to avoid hardcoding and potential errors
                noverlap=int(2**13 * 0.75),
                window="hann",
            )

            # Select frequency band of interest
            idx_ff_in_band = np.logical_and(
                (ff >= fmin_common_band), (ff <= fmax_common_band)
            )
            ff = ff[idx_ff_in_band]
            Pxx_seg = Pxx_seg[idx_ff_in_band]
            # Convert to dB
            Pxx_seg = 10 * np.log10(Pxx_seg)

            # Compute weights (normalized PSD)
            w_k = (Pxx_seg + np.min(np.abs(Pxx_seg))) / np.max(
                Pxx_seg + np.min(np.abs(Pxx_seg))
            )
            event_weights.append(w_k)

            end_seg += segment_shift
            start_seg += segment_shift

        event_weights = np.array(event_weights)  # Shape (n_seg, nf)

        # TODO adapt the section below for the library signal

        # start_seg = 0
        # end_seg = xr_passive.analysis_segment_duration
        # segment_shift = xr_passive.analysis_segment_duration * (
        #     1 - xr_passive.analysis_segment_alpha_overlap
        # )
        # library_weights = []
        # for i_pulse in range(xr_seq_ref.sizes["pulse_id"]):
        #     # TODO : implement this
        #     # We need to compute the weights to use to compute the weighted mean hermitian angle

        #     # Get signal corresponding to the current segment
        #     passive_sig_seg = xr_passive.signal.sel(time=slice(start_seg, end_seg))
        #     # Compute weights using signal from the reference receiver
        #     passive_sig_seg_rcv_ref = passive_sig_seg.sel(
        #         h_index=xr_passive.h_index_ref
        #     )

        #     # Compute PSD of the signal
        #     ff, Pxx_seg = sp.welch(
        #         passive_sig_seg_rcv_ref.values,
        #         fs=xr_passive.fs,
        #         nperseg=2
        #         ** 13,  # TODO save those params in xr_passive to avoid hardcoding and potential errors
        #         noverlap=int(2**13 * 0.75),
        #         window="hann",
        #     )

        #     # Select frequency band of interest
        #     idx_ff_in_band = np.logical_and(
        #         (ff >= fmin_common_band), (ff <= fmax_common_band)
        #     )
        #     ff = ff[idx_ff_in_band]
        #     Pxx_seg = Pxx_seg[idx_ff_in_band]
        #     # Convert to dB
        #     Pxx_seg = 10 * np.log10(Pxx_seg)

        #     # Compute weights (normalized PSD)
        #     w_k = (Pxx_seg + np.min(np.abs(Pxx_seg))) / np.max(
        #         Pxx_seg + np.min(np.abs(Pxx_seg))
        #     )
        #     event_weights.append(w_k)

        #     end_seg += segment_shift
        #     start_seg += segment_shift

        # library_weights = np.array(library_weights)  # Shape ?

    if dist_type == "hermitian_angle":
        # Iterate of each pulse of the library replica
        for pulse_id_ref in xr_seq_ref.pulse_id.values:

            rtf_ref_pulse_i = rtf_ref.sel(pulse_id=pulse_id_ref)
            rtf_dist = D_hermitian_angle_fast(
                rtf_ref=rtf_ref_pulse_i.values,
                rtf=rtf_passive_4d,
                **dist_kwargs,
            )

            rtf_distances.append(rtf_dist)
    rtf_distances = np.array(rtf_distances)
    rtf_distances = rtf_distances.T

    # Build distances
    for segment_dt in xr_passive.segment_dt.values:
        gps_pos_segment = ds_gps.sel(time=segment_dt, method="nearest")

        pos_e_segment = gps_pos_segment.e.values
        pos_n_segment = gps_pos_segment.n.values

        # Compute distance to all reference positions
        spatial_dist = np.sqrt(
            (ref_pos_e - pos_e_segment) ** 2 + (ref_pos_n - pos_n_segment) ** 2
        )
        spatial_distances.append(spatial_dist)

    spatial_distances = np.array(spatial_distances)

    return (
        rtf_distances,
        spatial_distances,
        xr_seq_ref,
        df_seq_ref,
        xr_passive,
    )


def filter_ais(ais_event, verbose=False):
    if verbose:
        print("\tFiltering AIS data ...")
    # Filter AIS data in area
    e_min = -1000
    e_max = +1000
    n_min = -1000
    n_max = +1000

    mmsi_in_box = []
    for mmsi in ais_event.mmsi.values:
        ship = ais_event.sel(mmsi=mmsi)
        ship_in_e_range = np.logical_and(ship.e.values >= e_min, ship.e.values <= e_max)
        ship_in_n_range = np.logical_and(ship.n.values >= n_min, ship.n.values <= n_max)
        ship_in_box = np.logical_and(ship_in_e_range, ship_in_n_range)

        if np.any(ship_in_box):
            mmsi_in_box.append(mmsi)

    # print(mmsi_in_box)
    ais_event = ais_event.sel(mmsi=mmsi_in_box)

    return ais_event


def plot_traj(df_library, gps_event, ais_event, root_fig):

    print(f"\tPlotting ships trajectories ...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)
    # fig.suptitle(f"{str(t_start.day)}")       # TODO adapt title

    keys = ["obs1", "obs2", "obs3", "t1", "t2", "t3", "t4", "t5"]
    for k in keys:
        e = gps_event.attrs[f"{k}_e_apriori"]
        n = gps_event.attrs[f"{k}_n_apriori"]
        ax.scatter(
            e,
            n,
            marker="D",
            label=k,
            zorder=1,
            s=150,
        )

    times = mdates.date2num(gps_event.time)
    im = ax.scatter(
        gps_event.e,
        gps_event.n,
        c=times,
        s=20,
        cmap="hsv",
        zorder=4,
    )
    cbar = plt.colorbar(im)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Add ships in the area using AIS
    for mmsi in ais_event.mmsi.values:
        ship = ais_event.sel(mmsi=mmsi)
        plt.plot(ship.e.values, ship.n.values, label=mmsi)
        plt.scatter(
            ship.e.values[0], ship.n.values[0], marker="s", s=100, color="k", zorder=5
        )
        plt.scatter(
            ship.e.values[-1], ship.n.values[-1], marker="*", s=100, color="r", zorder=5
        )

    # Série de positions successives
    library_sequence = df_library["sequence_id"].iloc[0]
    im = plt.scatter(
        df_library["emission_interp_e_gps"],
        df_library["emission_interp_n_gps"],
        marker="+",
        label=f"Library ({library_sequence})",
        c=np.arange(df_library["emission_interp_e_gps"].size),
        cmap="magma",
    )
    cbar = plt.colorbar(im)

    # plt.xlim(-1500, 3000)
    # plt.ylim(-1500, 3000)

    ax.legend(fontsize=12)
    ax.grid()
    ax.set_title("")
    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")

    fpath = os.path.join(root_fig, "situation_ais_gps.png")
    plt.savefig(fpath)
    plt.close("all")


def plot_spectro(ds_wav, t_start, t_end, root_fig, **spectro_kwargs):
    print(f"\tPlotting spectrograms ...")
    # Paramètre STFT
    nperseg = spectro_kwargs.get("nperseg", 2**12)
    noverlap = spectro_kwargs.get("noverlap", int(nperseg * 0.5))
    save = spectro_kwargs.get("save", True)

    rcv_sxx = []
    for i, obs_id in enumerate([1, 2, 3]):

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        fs = ds_wav.attrs[f"fs_obs{obs_id}"]

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})

        # Derive stft
        ff, tt, stft = sp.stft(
            sig_win.values,  # .values -> ici on charge les données en mémoire (un tout petit subset seulement)
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # U^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft))  # dB re 1uPa**2 / Hz ou dB re 1 (m/s)^2 / Hz
        rcv_sxx.append(sxx)

    # Define datetime borders
    t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
    t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

    # Associated datetime vector
    tt_datetime = pd.date_range(
        t0_slice,
        t0_slice + timedelta(seconds=tt[-1]),
        freq=f"{tt[1]-tt[0]}s",
        inclusive="both",
    )

    cmap = "magma"
    vmin = np.percentile(sxx, 30)
    vmax = np.percentile(sxx, 95)
    # vmin = 25
    # vmax = 45

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    for i, obs_id in enumerate([1, 2, 3]):

        sxx = rcv_sxx[i]
        # Plot

        im = axs[i].pcolormesh(tt_datetime, ff, sxx, cmap=cmap, vmin=vmin, vmax=vmax)

        clabel = r"dB re 1$\mu$Pa$^2$ / Hz"
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="2%", pad=0.10)
        fig.colorbar(im, cax=cax, orientation="vertical", label=clabel)

        # axs[ich].colorbar(im, label=clabel)

        axs[i].set_title(f"OBS{obs_id}")

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    # nticks = 10
    # xmin, xmax = axs[-1].get_xlim()
    # axs[-1].set_xticks(np.linspace(xmin, xmax, nticks))
    # plt.setp(axs[-1].get_xticklabels(), rotation=25, ha="center")

    fig.supylabel("Fréquence [Hz]")
    fig.supxlabel("Temps UTC")
    # fig.suptitle(f"OBS{obs_id}")

    if save:
        fpath = os.path.join(root_fig, "spectro_3obs.png")
        plt.savefig(fpath, bbox_inches="tight")
        # plt.close("all")
    else:
        plt.show()

    return fig, axs


def compute_spatial_dist(gps_event, ais_event, df_library, segment_dt):
    print(f"\tComputing distance from ships to library replicas positions ...")
    gps_event.time.attrs["long_name"] = "Temps UTC"

    spatial_dist = {}
    spatial_dist_ais = {mmsi: {} for mmsi in ais_event.mmsi.values}
    replica_ids = df_library.index.to_numpy()
    for idx in replica_ids:
        df_lib_replica_i = df_library.iloc[idx]

        lib_pos_e = df_lib_replica_i["emission_interp_e_gps"]
        lib_pos_n = df_lib_replica_i["emission_interp_n_gps"]

        # Compute distance
        # GPS
        spatial_dist_i = np.sqrt(
            (lib_pos_e - gps_event.e) ** 2 + (lib_pos_n - gps_event.n) ** 2
        )

        spatial_dist_i.attrs["long_name"] = "Distance"
        spatial_dist[idx] = spatial_dist_i

        # AIS
        for mmsi in ais_event.mmsi.values:
            ship = ais_event.sel(mmsi=mmsi)

            dist_i_ais = np.sqrt((lib_pos_e - ship.e) ** 2 + (lib_pos_n - ship.n) ** 2)
            dist_i_ais.attrs["long_name"] = "Distance"
            spatial_dist_ais[mmsi][idx] = dist_i_ais

    spatial_dist_ais_mat = np.zeros(
        (
            ais_event.sizes["mmsi"],
            replica_ids.size,
            gps_event.sizes["time"],
        )
    )
    for i, mmsi in enumerate(ais_event.mmsi.values):
        for j, idx in enumerate(replica_ids):
            spatial_dist_ais_mat[i, j, :] = np.array(spatial_dist_ais[mmsi][idx].values)

    spatial_dist_mat = np.array([spatial_dist[idx].values for idx in replica_ids])

    ds_spatial_dist = xr.Dataset(
        data_vars=dict(spatial_dist=(["replica_id", "time"], spatial_dist_mat)),
        coords=dict(time=gps_event.time, replica_id=replica_ids),
    )
    ds_spatial_dist["spatial_dist"].attrs = {"units": "m", "long_name": "Distance"}
    ds_spatial_dist["replica_id"].attrs = {"long_name": "Replica ID"}

    ds_spatial_dist_ais = xr.Dataset(
        data_vars=dict(
            spatial_dist=(["mmsi", "replica_id", "time"], spatial_dist_ais_mat)
        ),
        coords=dict(
            mmsi=ais_event.mmsi.values,
            time=gps_event.time,
            replica_id=replica_ids,
        ),
    )
    ds_spatial_dist_ais["spatial_dist"].attrs = {"units": "m", "long_name": "Distance"}
    ds_spatial_dist_ais["replica_id"].attrs = {"long_name": "Replica ID"}
    ds_spatial_dist_ais["mmsi"].attrs = {"long_name": "MMSI"}

    # Align theta_dist and spatial_dist on the same time vector
    dist_interp = ds_spatial_dist.spatial_dist.interp(time=segment_dt)
    dist_interp_ais = ds_spatial_dist_ais.spatial_dist.interp(time=segment_dt)
    # Update ds_spatial_dist with the interpolated distance
    ds_spatial_dist = ds_spatial_dist.assign(spatial_dist=dist_interp)
    ds_spatial_dist_ais = ds_spatial_dist_ais.assign(spatial_dist=dist_interp_ais)

    return ds_spatial_dist, ds_spatial_dist_ais


def plot_spatial_dist(ds_spatial_dist, ds_spatial_dist_ais):
    print(f"\tPlotting distance from ships to library replicas positions ...")
    ds_spatial_dist.spatial_dist.plot(
        x="segment_dt", y="replica_id", cmap="magma", vmin=0, vmax=1500
    )

    if ds_spatial_dist_ais is not None:
        for mmsi in ds_spatial_dist_ais.mmsi.values:
            plt.figure()
            ds_spatial_dist_ais.sel(mmsi=mmsi).spatial_dist.plot(
                x="time", y="replica_id", cmap="magma", vmin=0, vmax=1500
            )


def process_event(ds_wav, t_start, t_end, **process_event_kwargs):

    # Unpack kwargs
    compute_rtf_event = process_event_kwargs.get("compute_rtf_event", True)
    root_processed_data = process_event_kwargs.get("root_processed_data", None)
    h_index_ref = process_event_kwargs.get("h_index_ref", 1)
    plot_feature = process_event_kwargs.get("plot_feature", False)
    rtf_estimator = process_event_kwargs.get("rtf_estimator", "cs-evd")
    analysis_segment_duration = process_event_kwargs.get(
        "analysis_segment_duration", 10
    )
    analysis_segment_alpha_overlap = process_event_kwargs.get(
        "analysis_segment_alpha_overlap", 0.5
    )
    verbose = process_event_kwargs.get("verbose", False)

    fs = 2000
    tau_rtf_analysis = 3

    # Number of samples corresponding to the assumed impulse response duration
    n_rtf_analysis = int(tau_rtf_analysis * fs)
    # Get closer power of 2
    nperseg = 2 ** int(
        np.log2(n_rtf_analysis) + 1
    )  # Number of sample per snapshot to use = closest power of two
    alpha_overlap = 0.75
    noverlap = int(nperseg * alpha_overlap)

    # h_index_ref = 1  # -> OBS 3 has the higher snr
    # root_rtf_data = os.path.join(data_folder, "rtf")
    # plot_feature = False
    # process_pulse_one_by_one = True
    # estimate_ir_duration = False

    fsm = PassiveFiberscopeManager(
        ds_wav=ds_wav,
        root_processed_data=root_processed_data,
        h_index_ref=h_index_ref,
        plot_feature=plot_feature,
        # bandfilter=bandfilter,
        # tau_ir=tau_ir,
        # process_pulse_one_by_one=process_pulse_one_by_one,
        # estimate_ir_duration=estimate_ir_duration,
        rtf_estimator=rtf_estimator,
        verbose=verbose,
        analysis_segment_duration=analysis_segment_duration,
        analysis_segment_alpha_overlap=analysis_segment_alpha_overlap,
    )

    fsm.nperseg = nperseg
    fsm.noverlap = noverlap

    print(f"\tProcessing passive event (derive RTF)...")
    if compute_rtf_event:
        # Comment below if already computed
        fsm.process_analysis(
            t_start,
            t_end,
            set_stft_props=False,
        )

    return fsm


def compute_theta_dist(df_library, gps_event, t_start, t_end, fsm, **theta_dist_kwargs):
    print(f"\tMatching (computing hermitian angle theta) ...")
    # Analyse rtf variations
    fmin = theta_dist_kwargs.get("fmin", 300)
    fmax = theta_dist_kwargs.get("fmax", 900)

    use_weighted_mean = theta_dist_kwargs.get("use_weighted_mean", False)

    # seq_library = df_library["sequence_id"].unique()
    # seq_event = df_event["sequence_id"].unique()[0]
    replica_ids = []

    library_seq_ids = df_library["sequence_id"].unique()
    theta_dist_mat = []
    for lib_seq in library_seq_ids:
        print(f"\t\tProcessing library sequence {lib_seq} ...")
        theta_dist, spatial_dist, xr_seq_ref, df_seq_ref, xr_passive = (
            get_dists_2_passive_event(
                fsm=fsm,
                df_arr_library=df_library,
                seq_id_ref=lib_seq,
                ds_gps=gps_event,
                t_start=t_start,
                t_end=t_end,
                fmin=fmin,
                fmax=fmax,
                dist_type="hermitian_angle",
                use_weighted_mean=use_weighted_mean,
            )
        )
        # theta_dist.shape = (segment_dt, pulse_id)
        if len(theta_dist_mat) == 0:
            theta_dist_mat = theta_dist
        else:
            theta_dist_mat = np.hstack([theta_dist_mat, theta_dist])

        replica_ids.extend(df_seq_ref.index.to_numpy())

    # Ensure we keep only the replicas corresponding to detected pulses in the library for the final dataset (in case some pulses were not detected and thus not included in the theta_dist matrix)
    df_library = df_library.iloc[replica_ids]
    df_library = df_library.reset_index(drop=True)

    # theta_dist_mat = np.array(theta_dist_mat)
    replica_ids = df_library.index.to_numpy()

    segment_dt = xr_passive.segment_dt.values
    segment_dt = pd.to_datetime(
        segment_dt
    ).to_pydatetime()  # Convert to datetime objects

    ds_theta_dist = xr.Dataset(
        data_vars=dict(
            theta=(
                [
                    "segment_dt",
                    "replica_id",
                ],
                theta_dist_mat.astype(np.float32),
            )
        ),
        coords=dict(
            replica_id=replica_ids,
            segment_dt=segment_dt,
            # pulse_id=xr_seq_ref.pulse_id.values,
        ),
    )
    ds_theta_dist["segment_dt"].attrs = xr_passive.segment_dt.attrs
    ds_theta_dist["replica_id"].attrs = {"long_name": "Replica ID"}
    # ds_theta_dist["pulse_id"].attrs = {"long_name": "Pulse ID"}
    ds_theta_dist["theta"].attrs = {
        "units": "°",
        "long_name": r"$\theta$",
    }

    return ds_theta_dist, df_library


def plot_theta_vs_sorted_dist(
    ds_theta_dist, ds_spatial_dist, ds_spatial_dist_ais, root_fig
):

    print(f"\tPlotting theta vs distance from ships to library replicas positions ...")

    # Get replica closest to ship trajectory  (GPS Jules)
    cpa_idx = ds_spatial_dist.spatial_dist.argmin(...)
    cpa_idx_segment_dt = cpa_idx["segment_dt"].values
    # cpa_segment_dt = ds_spatial_dist.isel(segment_dt=cpa_idx_segment_dt).segment_dt.values
    cpa_idx_replica_id = cpa_idx["replica_id"].values
    cpa_replica_id = ds_spatial_dist.isel(
        replica_id=cpa_idx_replica_id
    ).replica_id.values

    # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(16, 12))
    plt.figure()

    # Extract theta and dist variation for the selected replica
    ds_theta_dist_cpa = ds_theta_dist.sel(replica_id=cpa_replica_id)
    ds_spatial_dist_cpa = ds_spatial_dist.sel(replica_id=cpa_replica_id)

    dist_to_cpa_argsort = ds_spatial_dist_cpa.spatial_dist.argsort()
    sorted_spatial_dist = ds_spatial_dist_cpa.spatial_dist.values[dist_to_cpa_argsort]
    sorted_theta_dist = ds_theta_dist_cpa.theta.values[dist_to_cpa_argsort]

    # Theta distance
    plt.scatter(
        sorted_spatial_dist,
        sorted_theta_dist,
        # label=f"{pos} ({theta_dist_obs[pos]['id']})",
    )

    # ds_theta_dist.theta.plot(
    #     x="segment_dt",
    #     y="replica_id",
    #     vmin=vmin,
    #     vmax=vmax,
    #     cmap="magma",
    #     ax=axs[0],
    # )
    # # Annotate CPA point
    # axs[0].annotate(
    #     "CPA",
    #     xy=(cpa_time, cpa_replica_id),
    #     xytext=(cpa_time + np.timedelta64(0, "s"), cpa_replica_id + 20),
    #     arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
    #     fontsize=14,
    #     color="cyan",
    #     ha="center",
    # )

    # # Spatial distance
    # ds_spatial_dist.spatial_dist.plot(
    #     x="time", y="replica_id", cmap="magma", vmin=0, vmax=1000, ax=axs[1]
    # )
    # # Annotate CPA point
    # axs[1].annotate(
    #     "CPA",
    #     xy=(cpa_time, cpa_replica_id),
    #     xytext=(cpa_time + np.timedelta64(0, "s"), cpa_replica_id + 20),
    #     arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
    #     fontsize=14,
    #     color="cyan",
    #     ha="center",
    # )

    fpath = os.path.join(
        root_fig, f"theta_vs_sorted_distance_closest_replica_gps_Jules.png"
    )
    plt.savefig(fpath, bbox_inches="tight")

    # AIS
    for mmsi in ds_spatial_dist_ais.mmsi.values:
        spatial_dist_mmsi = ds_spatial_dist_ais.sel(mmsi=mmsi)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
        # plt.figure()

        cpa_idx = spatial_dist_mmsi.spatial_dist.argmin(...)
        cpa_idx_segment_dt = cpa_idx["segment_dt"].values
        # cpa_segment_dt = spatial_dist_mmsi.isel(
        #     segment_dt=cpa_idx_segment_dt
        # ).segment_dt.values
        cpa_idx_replica_id = cpa_idx["replica_id"].values
        cpa_replica_id = spatial_dist_mmsi.isel(
            replica_id=cpa_idx_replica_id
        ).replica_id.values

        # Extract theta and dist variation for the selected replica
        rep_offset = 2
        min_rep = max(0, cpa_replica_id - rep_offset)
        max_rep = min(
            spatial_dist_mmsi.replica_id.max().values, cpa_replica_id + rep_offset
        )
        ds_theta_dist_cpa = ds_theta_dist.sel(replica_id=slice(min_rep, max_rep))
        spatial_dist_mmsi_cpa = spatial_dist_mmsi.sel(
            replica_id=slice(min_rep, max_rep)
        )
        ds_theta_dist_cpa.theta.plot(ax=axs[0], hue="replica_id")
        spatial_dist_mmsi_cpa.spatial_dist.plot(ax=axs[1], hue="replica_id")

        fpath = os.path.join(
            root_fig,
            f"theta_vs_and_distance_{rep_offset*2+1}_closest_replicas_ais_{mmsi}.png",
        )
        plt.savefig(fpath, bbox_inches="tight")

        plt.figure()

        # cpa_idx = spatial_dist_mmsi.spatial_dist.argmin(...)
        # cpa_idx_time = cpa_idx["time"].values
        # cpa_time = spatial_dist_mmsi.isel(time=cpa_idx_time).time.values
        # cpa_idx_replica_id = cpa_idx["replica_id"].values
        # cpa_replica_id = spatial_dist_mmsi.isel(
        #     replica_id=cpa_idx_replica_id
        # ).replica_id.values

        # Extract theta and dist variation for the selected replica
        ds_theta_dist_cpa = ds_theta_dist.sel(replica_id=cpa_replica_id)
        spatial_dist_mmsi_cpa = spatial_dist_mmsi.sel(replica_id=cpa_replica_id)

        # dist_to_cpa_argsort = spatial_dist_mmsi_cpa.spatial_dist.argsort()
        dist_to_cpa_argsort = np.argsort(spatial_dist_mmsi_cpa.spatial_dist.values)

        sorted_spatial_dist = spatial_dist_mmsi_cpa.spatial_dist.values[
            dist_to_cpa_argsort
        ]
        sorted_theta_dist = ds_theta_dist_cpa.theta.values[dist_to_cpa_argsort]

        # Theta distance
        plt.scatter(
            sorted_spatial_dist,
            sorted_theta_dist,
            # label=f"{pos} ({theta_dist_obs[pos]['id']})",
        )

        # ds_theta_dist.theta.plot(
        #     x="segment_dt",
        #     y="replica_id",
        #     vmin=vmin,
        #     vmax=vmax,
        #     cmap="magma",
        #     ax=axs[0],
        # )

        # # Annotate CPA point
        # axs[0].annotate(
        #     "CPA",
        #     xy=(cpa_time, cpa_replica_id),
        #     xytext=(cpa_time + np.timedelta64(0, "s"), cpa_replica_id + 20),
        #     arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
        #     fontsize=14,
        #     color="cyan",
        #     ha="center",
        # )

        # spatial_dist_mmsi.spatial_dist.plot(
        #     x="time", y="replica_id", cmap="magma", vmin=0, vmax=1000, ax=axs[1]
        # )
        # # Annotate CPA point

        # axs[1].annotate(
        #     "CPA",
        #     xy=(cpa_time, cpa_replica_id),
        #     xytext=(cpa_time + np.timedelta64(0, "s"), cpa_replica_id + 20),
        #     arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
        #     fontsize=14,
        #     color="cyan",
        #     ha="center",
        # )

        fpath = os.path.join(
            root_fig, f"theta_vs_sorted_distance_closest_replica_ais_{mmsi}.png"
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_theta_vs_spatial_dist(
    ds_theta_dist, ds_spatial_dist, ds_spatial_dist_ais, root_fig
):

    print(f"\tPlotting theta vs distance from ships to library replicas positions ...")

    # Define colorbar limits
    vmin = np.percentile(ds_theta_dist.theta.values, 0.1)
    vmax = np.percentile(ds_theta_dist.theta.values, 50)

    # GPS Jules
    cpa_idx = ds_spatial_dist.spatial_dist.argmin(...)
    cpa_idx_segment_dt = cpa_idx["segment_dt"].values
    # cpa_idx = ds_spatial_dist.spatial_dist.argmin(...)
    # cpa_idx_time = cpa_idx["time"].values
    cpa_segment_dt = ds_spatial_dist.isel(
        segment_dt=cpa_idx_segment_dt
    ).segment_dt.values
    cpa_idx_replica_id = cpa_idx["replica_id"].values
    cpa_replica_id = ds_spatial_dist.isel(
        replica_id=cpa_idx_replica_id
    ).replica_id.values

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # Theta distance
    ds_theta_dist.theta.plot(
        x="segment_dt",
        y="replica_id",
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        ax=axs[0],
    )
    # # Add marker at CPA time and replica id
    # axs[0].scatter(
    #     cpa_time,
    #     cpa_replica_id,
    #     marker="X",
    #     s=100,
    #     color="cyan",
    #     # label="CPA",
    #     zorder=5,
    # )
    # Annotate CPA point
    axs[0].annotate(
        "CPA",
        xy=(cpa_segment_dt, cpa_replica_id),
        xytext=(cpa_segment_dt + np.timedelta64(0, "s"), cpa_replica_id + 20),
        arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
        fontsize=14,
        color="cyan",
        ha="center",
    )

    # Spatial distance
    ds_spatial_dist.spatial_dist.plot(
        x="segment_dt", y="replica_id", cmap="magma", vmin=0, vmax=1000, ax=axs[1]
    )
    # axs[1].scatter(
    #     cpa_time,
    #     cpa_replica_id,
    #     marker="X",
    #     s=100,
    #     color="cyan",
    #     # label="CPA",
    #     zorder=5,
    # )
    # Annotate CPA point
    axs[1].annotate(
        "CPA",
        xy=(cpa_segment_dt, cpa_replica_id),
        xytext=(cpa_segment_dt + np.timedelta64(0, "s"), cpa_replica_id + 20),
        arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
        fontsize=14,
        color="cyan",
        ha="center",
    )

    fpath = os.path.join(root_fig, f"theta_vs_distance_gps_Jules.png")
    plt.savefig(fpath, bbox_inches="tight")

    # AIS
    for mmsi in ds_spatial_dist_ais.mmsi.values:
        spatial_dist_mmsi = ds_spatial_dist_ais.sel(mmsi=mmsi)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

        cpa_idx_replica_id = cpa_idx["replica_id"].values
        cpa_replica_id = spatial_dist_mmsi.isel(
            replica_id=cpa_idx_replica_id
        ).replica_id.values

        cpa_idx = spatial_dist_mmsi.spatial_dist.argmin(...)
        # cpa_idx_time = cpa_idx["time"].values
        cpa_idx_segment_dt = cpa_idx["segment_dt"].values

        # cpa_time = spatial_dist_mmsi.isel(time=cpa_idx_time).time.values
        cpa_segment_dt = spatial_dist_mmsi.isel(
            segment_dt=cpa_idx_segment_dt
        ).segment_dt.values

        cpa_idx_replica_id = cpa_idx["replica_id"].values
        cpa_replica_id = spatial_dist_mmsi.isel(
            replica_id=cpa_idx_replica_id
        ).replica_id.values

        ds_theta_dist.theta.plot(
            x="segment_dt",
            y="replica_id",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            ax=axs[0],
        )

        # Annotate CPA point
        # axs[0].scatter(
        #     cpa_time,
        #     cpa_replica_id,
        #     marker="X",
        #     s=100,
        #     color="cyan",
        #     # label="CPA",
        #     zorder=5,
        # )
        axs[0].annotate(
            "CPA",
            xy=(cpa_segment_dt, cpa_replica_id),
            xytext=(cpa_segment_dt + np.timedelta64(0, "s"), cpa_replica_id + 20),
            arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
            fontsize=14,
            color="cyan",
            ha="center",
        )

        spatial_dist_mmsi.spatial_dist.plot(
            x="segment_dt", y="replica_id", cmap="magma", vmin=0, vmax=1000, ax=axs[1]
        )
        # # Annotate CPA point
        # axs[1].scatter(
        #     cpa_time,
        #     cpa_replica_id,
        #     marker="X",
        #     s=100,
        #     color="cyan",
        #     # label="CPA",
        #     zorder=5,
        # )
        axs[1].annotate(
            "CPA",
            xy=(cpa_segment_dt, cpa_replica_id),
            xytext=(cpa_segment_dt + np.timedelta64(0, "s"), cpa_replica_id + 20),
            arrowprops=dict(facecolor="cyan", shrink=0.05, width=4, headwidth=8),
            fontsize=14,
            color="cyan",
            ha="center",
        )

        fpath = os.path.join(root_fig, f"theta_vs_distance_ais_{mmsi}.png")
        plt.savefig(fpath, bbox_inches="tight")


def process_batch(
    ds_wav,
    ds_gps,
    ds_ais,
    df_library,
    t_start,
    t_end,
    root_results_fig,
    root_results_data,
    compute_animation=True,
    theta_dist_kwargs={},
    process_event_kwargs={},
    spectro_kwargs={},
):
    print(f"\nProcessing batch for event from {t_start} to {t_end} ...")

    gps_event = ds_gps.sel(time=slice(t_start, t_end))
    ais_event = ds_ais.sel(time=slice(t_start, t_end))

    ais_event = filter_ais(ais_event)

    # Plot Traj on map
    plot_traj(
        df_library=df_library,
        gps_event=gps_event,
        ais_event=ais_event,
        root_fig=root_results_fig,
    )

    # Plot signal spectrograms
    plot_spectro(
        ds_wav=ds_wav,
        t_start=t_start,
        t_end=t_end,
        root_fig=root_results_fig,
        **spectro_kwargs,
    )

    # Compute RTF
    fsm = process_event(ds_wav, t_start, t_end, **process_event_kwargs)

    # Match processing : derive hermitian angle theta and associated spatial dist
    ds_theta_dist, df_library = compute_theta_dist(
        df_library=df_library,
        gps_event=gps_event,
        t_start=t_start,
        t_end=t_end,
        fsm=fsm,
        **theta_dist_kwargs,
    )

    # Save theta_dist
    fpath = os.path.join(
        root_results_data,
        f"theta_{datetime.strftime(t_start, fsm.datetime_fmt)}_to_{datetime.strftime(t_end, fsm.datetime_fmt)}.nc",
    )
    ds_theta_dist.to_netcdf(fpath)

    # Compute spatial dist
    ds_spatial_dist, ds_spatial_dist_ais = compute_spatial_dist(
        gps_event=gps_event,
        ais_event=ais_event,
        df_library=df_library,
        segment_dt=ds_theta_dist.segment_dt,
    )

    # Plot spatial dist
    # plot_spatial_dist(ds_spatial_dist, ds_spatial_dist_ais)

    # Compare theta and spatial dist
    plot_theta_vs_spatial_dist(
        ds_theta_dist=ds_theta_dist,
        ds_spatial_dist=ds_spatial_dist,
        ds_spatial_dist_ais=ds_spatial_dist_ais,
        root_fig=root_results_fig,
    )

    plot_theta_vs_sorted_dist(
        ds_theta_dist=ds_theta_dist,
        ds_spatial_dist=ds_spatial_dist,
        ds_spatial_dist_ais=ds_spatial_dist_ais,
        root_fig=root_results_fig,
    )

    # Animate RTF/MFP variations and ship trajectories
    if (
        compute_animation
    ):  # We can set to False if we just want to compute the theta_dist for multiple events in batch, without spending time on the animation for each event
        rtf_mfp_animation(
            ds_theta=ds_theta_dist,
            df_library=df_library,
            ds_gps_event=gps_event,
            ds_ais_event=ais_event,
            normalization_percentile=50,
            apply_roll_avg=True,
            roll_avg_window=3,
            step=4,
            save=True,
            output_fname="rtf_mfp_localisation",
            fps=10,
            dpi=100,
            root_img=root_results_fig,
        )


if __name__ == "__main__":
    pass
