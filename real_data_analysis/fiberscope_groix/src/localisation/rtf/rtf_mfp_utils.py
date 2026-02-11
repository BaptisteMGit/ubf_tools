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
import numpy as np
import xarray as xr
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


if __name__ == "__main__":
    pass
