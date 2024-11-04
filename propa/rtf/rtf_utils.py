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

    # Case: 4D input for variation studies
    if rtf.ndim == 4:
        # Expand rtf_ref along the necessary axes for broadcasting
        # rtf_ref_expanded = np.expand_dims(rtf_ref, axis=(1, 3))
        tile_shape = tuple(
            [rtf.shape[i] - rtf_ref.shape[i] + 1 for i in range(rtf.ndim)]
        )
        rtf_ref_expanded = np.tile(rtf_ref, tile_shape)

        # Calculate inner product and norms along the receiver axis (axis=2)
        ax_rcv = 2
        inner_prod = np.abs(np.sum(rtf_ref_expanded.conj() * rtf, axis=ax_rcv))
        norm_ref = np.linalg.norm(rtf_ref_expanded, axis=ax_rcv)
        norm_rtf = np.linalg.norm(rtf, axis=ax_rcv)

        # Calculate cosine of Hermitian angle, clipped to [-1, 1] for stability
        cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)
        dist = np.arccos(cos_angle)

        if unit == "deg":
            dist = np.rad2deg(dist)

        # Take mean along frequency axis if needed
        if apply_mean:
            dist = np.nanmean(dist, axis=0)

        # Flatten if only one receiver or one depth
        dist = np.squeeze(dist)

    # Case: 2D input for simple distance evaluation
    elif rtf.ndim == 2:
        # Calculate inner product and norms along the receiver axis (axis=1)
        ax_rcv = 1
        inner_prod = np.abs(np.sum(rtf_ref_expanded.conj() * rtf, axis=ax_rcv))
        norm_ref = np.linalg.norm(rtf_ref_expanded, axis=ax_rcv)
        norm_rtf = np.linalg.norm(rtf, axis=ax_rcv)

        # Cosine of Hermitian angle, clipped for stability
        cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)
        dist = np.arccos(cos_angle)

        if unit == "deg":
            dist = np.rad2deg(dist)

        if apply_mean:
            dist = np.nanmean(dist)

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


if __name__ == "__main__":
    pass
