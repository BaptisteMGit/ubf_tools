#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_utils.py
@Time    :   2024/10/17 10:11:34
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from misc import *
from propa.rtf.ideal_waveguide import *
from real_data_analysis.real_data_utils import *


def rtf_covariance_whitening(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the RTF using covariance whitening method described in  Markovich-Golan, S., & Gannot, S. (2015).
    """
    # Derive usefull params
    x = rcv_sig + rcv_noise
    n_rcv = x.shape[1]
    ts = t[1] - t[0]
    fs = 1 / ts

    f, Rx, Rs, Rv = get_csdm(t, rcv_sig, rcv_noise, nperseg, noverlap)

    ff, tt, stft_list_x = get_stft_list(x, fs, nperseg, noverlap)
    stft_x = np.array(stft_list_x)

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        Rv_f = Rv[i]
        # Rs_f = Rs[i]
        # Rx_f = Rx[i]
        stft_x_f = stft_x[i, :]

        # Cholesky decomposition of the noise csdm and its inverse : Equation (25a) and (25b)
        Rv_half = sp.linalg.cholesky(Rv_f, lower=False)
        Rv_inv = np.linalg.inv(Rv_f)
        Rv_half_inv = sp.linalg.cholesky(Rv_inv, lower=False)

        # Compute the whitened signal csdm : Equation (26)
        y = Rv_half_inv @ stft_x_f

        # Compute the whitened signal csdm : Equation (31)
        ff, Ry_f = get_csdm_from_signal(t, y)

        # Eigenvalue decomposition of Ry_f to get q (major eingenvector) : Equation (32)
        eig_val, eig_vect = np.linalg.eig(Ry_f)
        i_max_eig = np.argmax(eig_val)
        q = eig_vect[:, i_max_eig]

        rtf_f = (Rv_half @ q) / (e1.T @ Rv_half @ q)  # Equation (32)
        rtf[i, :] = rtf_f

    return f, rtf, Rx, Rs, Rv


def rtf_covariance_substraction(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the RTF using covariance substraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    """
    # Derive usefull params
    x = rcv_sig + rcv_noise
    n_rcv = x.shape[1]
    ts = t[1] - t[0]
    fs = 1 / ts

    # x = rcv_sig + rcv_noise
    f, Rx, Rs, Rv = get_csdm(t, rcv_sig, rcv_noise, nperseg, noverlap)

    # Check that Rs is of rank 1
    # for i in range(len(f)):
    #     rank = np.linalg.matrix_rank(Rs[i])
    #     print(f"Rank of Rs at f = {f[i]} Hz : {rank}")

    # Rx : CSDM of signal + noise
    # Rs : CSDM of signal
    # Rv : CSDM of noise
    R_delta = Rx - Rv  # Equation (9)

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        R_delta_f = R_delta[i]
        rtf_f = (R_delta_f @ e1) / (e1.T @ R_delta_f @ e1)
        rtf[i, :] = rtf_f

    return f, rtf, Rx, Rs, Rv


def get_csdm(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the CSDM of the received signal and noise.
    """
    x = rcv_sig + rcv_noise
    ff, csdm_x = get_csdm_from_signal(t, x, nperseg, noverlap)
    ff, csdm_sig = get_csdm_from_signal(t, rcv_sig, nperseg, noverlap)
    ff, csdm_noise = get_csdm_from_signal(t, rcv_noise, nperseg, noverlap)

    return ff, csdm_x, csdm_sig, csdm_noise


def get_csdm_from_signal(t, y, nperseg=2**12, noverlap=2**11):
    """
    Derive the CSDM of y.
    """
    fs = 1 / (t[1] - t[0])
    ff, _, stft_list = get_stft_list(y, fs, nperseg, noverlap)
    csdm_y = compute_csd_matrix_fast(stft_list, n_seg_cov="all")

    return ff, csdm_y


def get_stft_list(y, fs, nperseg, noverlap):
    """
    Derive the STFT of each component of y.
    """

    stft_list = []
    n_rcv = y.shape[1]

    for i in range(n_rcv):
        ff, tt, stft = sp.stft(
            y[:, i],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft_list.append(stft)

    return ff, tt, stft_list


if __name__ == "__main__":
    pass