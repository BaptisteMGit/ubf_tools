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


def D_frobenius(rtf_ref, rtf):
    """Derive the generalised distance combining all receivers."""

    # For variation studies
    if rtf.ndim == 4:
        # Expand g_ref to the same shape as g_r
        tile_shape = tuple(
            [rtf.shape[i] - rtf_ref.shape[i] + 1 for i in range(rtf.ndim)]
        )
        rtf_ref_expanded = np.tile(rtf_ref, tile_shape)

        nb_pos_r = rtf.shape[1]
        nb_pos_z = rtf.shape[3]

        Df_shape = (nb_pos_r, nb_pos_z)

        D_frobenius = np.zeros(Df_shape)
        for i_r in range(nb_pos_r):
            for i_z in range(nb_pos_z):
                Gamma = rtf_ref_expanded[:, i_r, :, i_z] - rtf[:, i_r, :, i_z]
                D_frobenius[i_r, i_z] = np.linalg.norm(Gamma, ord="fro")

        if nb_pos_z == 1 or nb_pos_r == 1:
            D_frobenius = D_frobenius.flatten()

    # For simple distance evaluation between two rtf vector
    elif rtf.ndim == 2:
        # Make sure to remove all nan values that can occure due to the 0 division (0 in the transfert function)
        idx_nan = np.isnan(rtf_ref)
        Gamma = rtf_ref - rtf
        Gamma[idx_nan] = 0
        D_frobenius = np.linalg.norm(Gamma, ord="fro")

    return D_frobenius
