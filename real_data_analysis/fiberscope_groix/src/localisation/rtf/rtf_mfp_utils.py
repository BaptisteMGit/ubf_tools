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
from datetime import datetime, timedelta
from propa.rtf.rtf_utils import D_hermitian_angle_fast


def get_dists(fsm, df_arr, seq_id_ref, seq_id, fmin=600, fmax=800):

    dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}

    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
    xr_seq_ref = xr.open_dataset(fpath)

    df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
    # Define the pulse id to use (easiest case where we only consider a single pulse within the selected sequence)
    psnr_tot = (
        df_seq_ref["psnr_obs1"] + df_seq_ref["psnr_obs2"] + df_seq_ref["psnr_obs3"]
    )
    pulse_id_to_use = df_seq_ref["pulse_id"].iloc[np.nanargmax(psnr_tot)]
    print(f"Library {seq_id_ref} -> used pulse = {pulse_id_to_use}")

    ref_pos_e = (
        df_seq_ref["emission_interp_e_gps"]
        .loc[df_seq_ref["pulse_id"] == pulse_id_to_use]
        .values
    )
    ref_pos_n = (
        df_seq_ref["emission_interp_n_gps"]
        .loc[df_seq_ref["pulse_id"] == pulse_id_to_use]
        .values
    )

    # Load rtf data
    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id}_rtf.nc")
    xr_seq_i = xr.open_dataset(fpath)
    df_seq_i = df_arr.loc[df_arr["sequence_id"] == seq_id]
    df_seq_i = df_seq_i.loc[df_seq_i["pulse_id"].isin(xr_seq_i.pulse_id.values)]

    # -----------------------------------------
    # 1) Evaluate spatial variations
    # -----------------------------------------
    # Emission positions in the sequence
    pos_e_pulse = df_seq_i["emission_interp_e_gps"]
    pos_n_pulse = df_seq_i["emission_interp_n_gps"]

    # Compute distance to all
    spatial_dist = np.sqrt(
        (ref_pos_e - pos_e_pulse) ** 2 + (ref_pos_n - pos_n_pulse) ** 2
    )

    # -----------------------------------------
    # 2) Evaluate RTF variations
    # -----------------------------------------
    # Reference RTF <-> first pulse in sequence
    xr_seq_i_ref = xr_seq_ref.sel(pulse_id=pulse_id_to_use)
    rtf_ref = xr_seq_i_ref.rtf_amp_hat * np.exp(1j * xr_seq_i_ref.rtf_phase_hat)

    # Rtf for each pulse
    rtf_pulse = xr_seq_i.rtf_amp_hat * np.exp(1j * xr_seq_i.rtf_phase_hat)

    # Limit frequency band
    rtf_ref = rtf_ref.sel(f_rtf=slice(fmin, fmax))
    rtf_pulse = rtf_pulse.sel(f_rtf=slice(fmin, fmax))

    # Compute hermitian angle distance
    theta_dist = []
    for pulse_id in xr_seq_i.pulse_id.values:
        rtf_pulse_i = rtf_pulse.sel(pulse_id=pulse_id)
        theta = D_hermitian_angle_fast(
            rtf_ref=rtf_ref.values, rtf=rtf_pulse_i.values, **dist_kwargs
        )

        theta_dist.append(theta)

    # print(spatial_dist, theta_dist)
    spatial_dist = np.array(spatial_dist)
    theta_dist = np.array(theta_dist)

    return theta_dist, spatial_dist, xr_seq_i, df_seq_i


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

    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
    xr_seq_ref = xr.open_dataset(fpath)

    df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
    df_seq_ref = df_seq_ref.loc[
        df_seq_ref["pulse_id"].isin(xr_seq_ref.pulse_id.values)
    ]  # keep only detected pulse

    ref_pos_e = df_seq_ref["emission_interp_e_gps"].values
    ref_pos_n = df_seq_ref["emission_interp_n_gps"].values

    # Load rtf data
    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id}_rtf.nc")
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
    df_arr,
    seq_id_ref,
    ds_gps,
    t_start,
    t_end,
    fmin=600,
    fmax=800,
    dist_type="hermitian_angle",
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

    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
    xr_seq_ref = xr.open_dataset(fpath)
    df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
    df_seq_ref = df_seq_ref.loc[
        df_seq_ref["pulse_id"].isin(xr_seq_ref.pulse_id.values)
    ]  # keep only detected pulse

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
    rtf_passive_4d = rtf_passive.values[..., np.newaxis]

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
    # TODO store those information in xr_passive to avoid duplication and potential errors
    # window_duration = 10
    # window_overlap = 0.5
    # for segment_id in xr_passive.segment_id.values:
    #     # Emission positions for the current sugment
    #     t_end_segment = window_duration * (1 + (segment_id - 1) * (1 - window_overlap))
    #     t_centre_segment_s = t_end_segment - 0.5 * window_duration
    #     t_centre_segment = t_start + timedelta(seconds=t_centre_segment_s)

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


if __name__ == "__main__":
    pass
