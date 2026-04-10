#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_utils.py
@Time    :   2024/10/20 12:20:48
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.interpolate as sp_int
from misc import cast_matrix_to_target_shape

# def D_frobenius(g_ref, g):
#     """Derive the generalised distance combining all receivers."""
#     # Expand g_ref to the same shape as g_r
#     tile_shape = tuple([g.shape[i] - g_ref.shape[i] + 1 for i in range(g.ndim)])
#     g_ref_expanded = np.tile(g_ref, tile_shape)

#     nb_pos_r = g.shape[1]
#     nb_pos_z = g.shape[3]

#     Df_shape = (nb_pos_r, nb_pos_z)

#     D_frobenius = np.zeros(Df_shape)
#     for i_r in range(nb_pos_r):
#         for i_z in range(nb_pos_z):
#             Gamma = g_ref_expanded[:, i_r, :, i_z] - g[:, i_r, :, i_z]
#             D_frobenius[i_r, i_z] = np.linalg.norm(Gamma, ord="fro")

#     if nb_pos_z == 1 or nb_pos_r == 1:
#         D_frobenius = D_frobenius.flatten()

#     return D_frobenius


def D_frobenius_module(rtf_ref, rtf, **kwargs):
    """Derive distance combining all receivers but using only RTF modules."""

    apply_mean = kwargs.get("apply_mean", True)
    apply_median = kwargs.get("apply_median", False)
    ax_rcv = kwargs.get("ax_rcv", 1)
    ax_f = kwargs.get("ax_f", 0)

    # Moveaxis to fit with the reference order (nf, nrcv, ...)
    rtf = np.moveaxis(rtf, [ax_f, ax_rcv], [0, 1])
    rtf_ref = np.moveaxis(rtf_ref, [ax_f, ax_rcv], [0, 1])

    # Case: 4D input for variation
    if rtf.ndim == 4:

        # Expand rtf_ref along the necessary axes for broadcasting
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Take modules
        rtf = np.abs(rtf)
        rtf_ref_expanded = np.abs(rtf_ref_expanded)

        # Compute the difference
        diff = rtf_ref_expanded - rtf
        dist_f2 = np.sum(
            np.abs(diff) ** 2, axis=1
        )  # Sum over receiver axis to get dist squared per frequency
        dist_f = np.sqrt(dist_f2)

        if apply_mean:
            dist = np.nanmean(dist_f, axis=0)
        elif apply_median:
            dist = np.nanmedian(dist_f, axis=0)

        # Flatten if only one range or one depth
        dist = np.squeeze(dist)

    # Case: 2D input for simple distance evaluation
    elif rtf.ndim == 2:

        # Take modules
        rtf = np.abs(rtf)
        rtf_ref_expanded = np.abs(rtf_ref_expanded)

        # Compute the difference
        diff = rtf_ref_expanded - rtf
        dist_f2 = np.sum(
            np.abs(diff) ** 2, axis=1
        )  # Sum over receiver axis to get dist squared per frequency
        dist_f = np.sqrt(dist_f2)

        if apply_mean:
            dist = np.nanmean(dist_f)
        elif apply_median:
            dist = np.nanmedian(dist_f)

    return dist


def D_frobenius_module_phase(rtf_ref, rtf, **kwargs):
    """Derive distance combining all receivers but using only RTF modules."""

    apply_mean = kwargs.get("apply_mean", True)
    apply_median = kwargs.get("apply_median", False)
    ax_rcv = kwargs.get("ax_rcv", 1)
    ax_f = kwargs.get("ax_f", 0)

    # Moveaxis to fit with the reference order (nf, nrcv, ...)
    rtf = np.moveaxis(rtf, [ax_f, ax_rcv], [0, 1])
    rtf_ref = np.moveaxis(rtf_ref, [ax_f, ax_rcv], [0, 1])

    # Case: 4D input for variation
    if rtf.ndim == 4:

        # Expand rtf_ref along the necessary axes for broadcasting
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute the difference
        diff = rtf_ref_expanded - rtf
        dist_f2 = np.sum(
            np.abs(diff) ** 2, axis=1
        )  # Sum over receiver axis to get dist squared per frequency
        dist_f = np.sqrt(dist_f2)

        if apply_mean:
            dist = np.nanmean(dist_f, axis=0)
        elif apply_median:
            dist = np.nanmedian(dist_f, axis=0)

        # Flatten if only one range or one depth
        dist = np.squeeze(dist)

    # Case: 2D input for simple distance evaluation
    elif rtf.ndim == 2:

        # Compute the difference
        diff = rtf_ref_expanded - rtf
        dist_f2 = np.sum(
            np.abs(diff) ** 2, axis=1
        )  # Sum over receiver axis to get dist squared per frequency
        dist_f = np.sqrt(dist_f2)

        if apply_mean:
            dist = np.nanmean(dist_f)
        elif apply_median:
            dist = np.nanmedian(dist_f)

    return dist


def D_frobenius(rtf_ref, rtf, **kwargs):
    """Derive the generalised distance combining all receivers."""

    # For variation studies
    if rtf.ndim == 4:
        # Expand rtf_ref to the same shape as rtf

        tile_shape = tuple(
            [rtf.shape[i] - rtf_ref.shape[i] + 1 for i in range(rtf.ndim)]
        )
        rtf_ref_expanded = np.tile(rtf_ref, tile_shape)

        nb_pos_r = rtf.shape[1]
        nb_pos_z = rtf.shape[3]

        Df_shape = (nb_pos_r, nb_pos_z)

        dist = np.zeros(Df_shape)
        for i_r in range(nb_pos_r):
            for i_z in range(nb_pos_z):
                Gamma = rtf_ref_expanded[:, i_r, :, i_z] - rtf[:, i_r, :, i_z]
                dist[i_r, i_z] = np.linalg.norm(Gamma, ord="fro")

        if nb_pos_z == 1 or nb_pos_r == 1:
            dist = dist.flatten()

    # For simple distance evaluation between two rtf vector
    elif rtf.ndim == 2:
        # Make sure to remove all nan values that can occure due to the 0 division (0 in the transfert function)
        idx_nan = np.isnan(rtf_ref)
        Gamma = rtf_ref - rtf
        Gamma[idx_nan] = 0
        dist = np.linalg.norm(Gamma, ord="fro")

    return dist


def D_hermitian_angle(rtf_ref, rtf, **kwargs):
    """Derive hermitian angle distance between two RTF."""

    unit = kwargs.get("unit", "deg")
    apply_mean = kwargs.get("apply_mean", True)

    # For variation studies
    if rtf.ndim == 4:
        # Expand rtf_ref to the same shape as rtf
        tile_shape = tuple(
            [rtf.shape[i] - rtf_ref.shape[i] + 1 for i in range(rtf.ndim)]
        )
        rtf_ref_expanded = np.tile(rtf_ref, tile_shape)

        nb_pos_r = rtf.shape[1]
        nb_pos_z = rtf.shape[3]

        dist_shape = (nb_pos_r, nb_pos_z)
        dist = np.zeros(dist_shape)
        for i_r in range(nb_pos_r):
            for i_z in range(nb_pos_z):
                dist[i_r, i_z] = D_hermitian_angle(
                    rtf_ref_expanded[:, i_r, :, i_z], rtf[:, i_r, :, i_z], **kwargs
                )

        if nb_pos_z == 1 or nb_pos_r == 1:
            dist = dist.flatten()

    # For simple distance evaluation between two rtf vector
    if rtf.ndim == 2:
        dist = np.empty(rtf.shape[0])
        # Ugly loop but it makes the job so far
        for i_omega in range(rtf.shape[0]):
            x = np.abs(np.sum(rtf_ref[i_omega].conj() * rtf[i_omega])) / (
                np.linalg.norm(rtf_ref[i_omega]) * np.linalg.norm(rtf[i_omega])
            )
            # Set max min to avoid problems due to round errors
            x = max(-1.0, min(1.0, x))
            dist[i_omega] = np.arccos(x)
            # print(f"x = {x}, d = {dist[i_omega]}, d_deg = {np.rad2deg(dist[i_omega])}")

        if unit == "deg":
            dist = np.rad2deg(dist)

        if apply_mean:
            dist = np.nanmean(dist)

    # print(f"dist = {dist} °")

    return dist


def D_hermitian_angle_fast(rtf_ref, rtf, **kwargs):
    """Derive Hermitian angle distance between two RTFs."""

    unit = kwargs.get("unit", "deg")
    apply_mean = kwargs.get("apply_mean", True)
    apply_median = kwargs.get("apply_median", False)
    weights = kwargs.get("weights", None)
    apply_sum = kwargs.get("apply_sum", False)
    ax_rcv = kwargs.get("ax_rcv", 3 if rtf.ndim == 4 else 1)
    ax_f = kwargs.get("ax_f", 1)
    data_space = kwargs.get("data_space", "complex")

    # Moveaxis to fit with the reference order (nf, nrcv, ...)
    rtf = np.moveaxis(rtf, [ax_f, ax_rcv], [0, 1])
    rtf_ref = np.moveaxis(rtf_ref, [ax_f, ax_rcv], [0, 1])

    # Case: 4D input for variation
    if rtf.ndim == 4:

        # Expand rtf_ref along the necessary axes for broadcasting
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Calculate inner product along the receiver axis
        if data_space == "real":
            # In real space (R^n) we use the traditionnal angle definition in euclidian space
            # Thus, we should not use abs values to define the angle.
            # Otherwise vectors with opposite directions leads to theta = 0° (<=> perfect match) which does not make sense
            inner_prod = np.sum(rtf_ref_expanded.conj() * rtf, axis=1)
        elif data_space == "complex":
            # Traditionnal definition of hermitian angle in C^n
            inner_prod = np.abs(np.sum(rtf_ref_expanded.conj() * rtf, axis=1))

        # Calculate norms along the receiver axis
        norm_ref = np.linalg.norm(rtf_ref_expanded, axis=1)
        norm_rtf = np.linalg.norm(rtf, axis=1)

        if data_space == "real":
            # Clip to [-1, 1] for stability
            cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)

        elif data_space == "complex":
            # Calculate cosine of Hermitian angle, clipped to [0, 1] for stability
            cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), 0, 1.0)

        dist = np.arccos(cos_angle)

        if unit == "deg":
            dist = np.rad2deg(dist)

        # Take mean along frequency axis if needed
        if apply_mean:
            # Check if weights are provided
            if weights is None:
                # If no weights are provided, use uniform weights
                weights = np.ones_like(dist)
            if weights.ndim == 1:
                # If weights are 1D, expand them to match the shape of dist
                weights = cast_matrix_to_target_shape(weights, dist.shape)

            # We can either use ma.average or do it by manually
            idx_nan = np.isnan(dist)
            weights[idx_nan] = np.nan
            dist = np.nansum(dist * weights, axis=0) * 1 / (np.nansum(weights, axis=0))

            # # Convert to mask array to handle NaN values with the ma.average function
            # dist = np.ma.MaskedArray(dist, mask=np.isnan(dist))
            # # Derive weighted average
            # dist = np.ma.average(dist, axis=0, weights=weights)
            # # Convert back to regular numpy array
            # dist = dist.filled(np.nan)

        elif apply_median:
            dist = np.nanmedian(dist, axis=0)
        elif apply_sum:
            dist = np.nansum(dist, axis=0)

        # Flatten if only one range or one depth
        dist = np.squeeze(dist)

    # Case: 2D input for simple distance evaluation
    elif rtf.ndim == 2:
        # Calculate inner product and norms along the receiver axis (axis=1)
        # ax_rcv = 1

        # Calculate inner product along the receiver axis
        if data_space == "real":
            # In real space (R^n) we use the traditionnal angle definition in euclidian space
            # Thus, we should not use abs values to define the angle.
            # Otherwise vectors with opposite directions leads to theta = 0° (<=> perfect match) which does not make sense
            inner_prod = np.sum(rtf_ref.conj() * rtf, axis=1)
        elif data_space == "complex":
            # Traditionnal definition of hermitian angle in C^n
            inner_prod = np.abs(np.sum(rtf_ref.conj() * rtf, axis=1))

        # inner_prod = np.abs(np.sum(rtf_ref.conj() * rtf, axis=1))
        norm_ref = np.linalg.norm(rtf_ref, axis=1)
        norm_rtf = np.linalg.norm(rtf, axis=1)

        if data_space == "real":
            # Clip to [-1, 1] for stability
            cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)

        elif data_space == "complex":
            # Calculate cosine of Hermitian angle, clipped to [0, 1] for stability
            cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), 0, 1.0)

        # Cosine of Hermitian angle, clipped for stability
        # cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)
        dist = np.arccos(cos_angle)

        if unit == "deg":
            dist = np.rad2deg(dist)

        if apply_mean:
            # Check if weights are provided
            if weights is None:
                # If no weights are provided, use uniform weights
                weights = np.ones_like(dist)
            # Ensure weights is a 1D array
            # if weights.shape

            # We can either use ma.average or do it by manually
            idx_nan = np.isnan(dist)
            weights[idx_nan] = np.nan
            dist = np.nansum(dist * weights, axis=0) * 1 / (np.nansum(weights, axis=0))

            dist = np.nanmean(dist)
        elif apply_median:
            dist = np.nanmedian(dist)
        elif apply_sum:
            dist = np.nansum(dist)

    return dist


def D_euclidian(rtf_ref, rtf, **kwargs):
    """Derive Euclidian distance between two RTFs."""

    apply_mean = kwargs.get("apply_mean", True)
    apply_median = kwargs.get("apply_median", False)
    weights = kwargs.get("weights", None)
    apply_sum = kwargs.get("apply_sum", False)
    ax_rcv = kwargs.get("ax_rcv", 3 if rtf.ndim == 4 else 1)
    ax_f = kwargs.get("ax_f", 1)

    # Moveaxis to fit with the reference order (nf, nrcv, ...)
    rtf = np.moveaxis(rtf, [ax_f, ax_rcv], [0, 1])
    rtf_ref = np.moveaxis(rtf_ref, [ax_f, ax_rcv], [0, 1])

    # Case: 4D input for variation
    if rtf.ndim == 4:

        # Expand rtf_ref along the necessary axes for broadcasting
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Calculate euclidian distance
        # d_euc = np.sqrt(np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=1))
        dist = np.linalg.norm(rtf_ref_expanded - rtf, axis=1)

        # Take mean along frequency axis if needed
        if apply_mean:
            # Check if weights are provided
            if weights is None:
                # If no weights are provided, use uniform weights
                weights = np.ones_like(dist)
            if weights.ndim == 1:
                # If weights are 1D, expand them to match the shape of dist
                weights = cast_matrix_to_target_shape(weights, dist.shape)

            # We can either use ma.average or do it by manually
            idx_nan = np.isnan(dist)
            weights[idx_nan] = np.nan
            dist = np.nansum(dist * weights, axis=0) * 1 / (np.nansum(weights, axis=0))

            # # Convert to mask array to handle NaN values with the ma.average function
            # dist = np.ma.MaskedArray(dist, mask=np.isnan(dist))
            # # Derive weighted average
            # dist = np.ma.average(dist, axis=0, weights=weights)
            # # Convert back to regular numpy array
            # dist = dist.filled(np.nan)

        elif apply_median:
            dist = np.nanmedian(dist, axis=0)
        elif apply_sum:
            dist = np.nansum(dist, axis=0)

        # Flatten if only one range or one depth
        dist = np.squeeze(dist)

    # Case: 2D input for simple distance evaluation
    elif rtf.ndim == 2:
        # Calculate euclidian along the receiver axis (axis=1)
        dist = np.linalg.norm(rtf_ref_expanded - rtf, axis=1)

        if apply_mean:
            # Check if weights are provided
            if weights is None:
                # If no weights are provided, use uniform weights
                weights = np.ones_like(dist)
            # Ensure weights is a 1D array
            # if weights.shape

            # We can either use ma.average or do it by manually
            idx_nan = np.isnan(dist)
            weights[idx_nan] = np.nan
            dist = np.nansum(dist * weights, axis=0) * 1 / (np.nansum(weights, axis=0))

            dist = np.nanmean(dist)
        elif apply_median:
            dist = np.nanmedian(dist)
        elif apply_sum:
            dist = np.nansum(dist)

    return dist


def D1(rtf_ref, rtf):
    d = np.sum(np.abs(rtf_ref - rtf), axis=0)
    return d


def D2(rtf_ref, rtf):
    d = np.sum(np.abs(rtf_ref - rtf) ** 2, axis=0)
    return d


def true_rtf(kraken_data):
    tf_ref = kraken_data[f"rcv{0}"]["h_f"]
    rtf = np.zeros((len(kraken_data["f"]), kraken_data["n_rcv"]), dtype=complex)
    for i in range(kraken_data["n_rcv"]):
        rtf[:, i] = kraken_data[f"rcv{i}"]["h_f"] / tf_ref

    return kraken_data["f"], rtf


def interp_true_rtf(kraken_data, f_interp):
    f_true, rtf_true = true_rtf(kraken_data)
    rtf_true = np.nan_to_num(rtf_true)
    nrcv = rtf_true.shape[1]
    nf = len(f_interp)
    rtf_true_interp = np.empty((nf, nrcv), dtype=complex)
    # Interpolate rtf_true to f_cs / f_cw
    for i_rcv in range(rtf_true.shape[1]):
        interp_real = sp_int.interp1d(f_true, np.real(rtf_true[:, i_rcv]))
        interp_imag = sp_int.interp1d(f_true, np.imag(rtf_true[:, i_rcv]))
        rtf_true_interp[:, i_rcv] = interp_real(f_interp) + 1j * interp_imag(f_interp)

    return f_interp, rtf_true_interp


def normalize_metric_contrast(d):
    """Build constrast from a metric d so that q lies in [0, 1]."""
    d_max = np.max(d)
    d_min = np.min(d)
    q = (d - d_min) / (d_max - d_min)
    return q


if __name__ == "__main__":
    pass
