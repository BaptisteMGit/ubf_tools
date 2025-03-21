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
import scipy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from misc import *
from propa.rtf.ideal_waveguide import *
from propa.rtf.rtf_estimation_const import *
from real_data_analysis.real_data_utils import (
    compute_csd_matrix_fast,
)


def rtf_covariance_whitening(
    t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11
):
    """
    Derive the RTF using covariance whitening method described in  Markovich-Golan, S., & Gannot, S. (2015).
    """
    # Derive usefull params
    n_rcv = noisy_signal.shape[1]
    ts = t[1] - t[0]
    fs = 1 / ts

    # Derive CSDM
    f, Rx, Rs, Rv = get_csdm(
        t,
        noisy_signal=noisy_signal,
        noise_only=noise_only,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    # Derive noisy_signal STFT
    ff, tt, stft_x = get_stft_array(noisy_signal, fs, nperseg, noverlap)
    # Estimate RTF
    f, rtf = rtf_cw(f, n_rcv, stft_x, Rv)

    return f, rtf, Rx, Rs, Rv


def rtf_cw(f, n_rcv, stft_x, Rv):

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        Rv_f = Rv[i]
        # Rs_f = Rs[i]
        # Rx_f = Rx[i]
        stft_x_f = stft_x[:, i, :]

        # Cholesky decomposition of the noise csdm and its inverse : Equation (25a) and (25b)
        Rv_half = scipy.linalg.cholesky(Rv_f, lower=False)
        # Rv_half_inv = np.linalg.inv(Rv_half).T        # Theoreticaly equivalent but leads to greater numerical errors
        Rv_inv_f = np.linalg.inv(Rv_f)
        Rv_half_inv = scipy.linalg.cholesky(Rv_inv_f, lower=False)

        # Compute the whitened signal csdm : Equation (26)
        stft_y_f = Rv_half_inv @ stft_x_f

        # Compute the whitened signal csdm : Equation (31)
        # Reshape to the required shape for the computation
        stft_y_f = [
            stft_y_f[i, np.newaxis, :] for i in range(n_rcv)
        ]  # List of stft at frequency f : n_rcv element of shape (n_freq=1, n_seg)
        Ry_f = compute_csd_matrix_fast(
            stft_y_f, n_seg_cov=0
        )  # Covariance matrix at frequency f
        Ry_f = (
            Ry_f.squeeze()
        )  # Remove useless frequency dimension to get shape (n_rcv, n_rcv)

        # Eigenvalue decomposition of Ry_f to get q (major eingenvector) : Equation (32)
        eig_val, eig_vect = np.linalg.eig(Ry_f)
        # We can check that the Ry_f can be diagonalized np.round(np.abs(np.linalg.inv(eig_vect) @ Ry_f @ eig_vect), 5)

        i_max_eig = np.argmax(np.abs(eig_val))
        q = eig_vect[:, i_max_eig]

        rtf_f = (Rv_half @ q) / (e1.T @ Rv_half @ q)  # Equation (32)
        rtf[i, :] = rtf_f

    return f, rtf


def rtf_covariance_substraction(
    t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11
):
    """
    Derive the RTF using covariance substraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    Reference receiver is assumed to be the first one.
    """

    # Derive usefull params
    n_rcv = noisy_signal.shape[1]
    # Derive CSDM
    f, Rx, Rs, Rv = get_csdm(
        t,
        noisy_signal=noisy_signal,
        noise_only=noise_only,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    # Estimate RTF
    f, rtf = rtf_cs(f, n_rcv, Rx, Rv)

    # x = rcv_sig + rcv_noise
    # n_rcv = x.shape[1]
    # ts = t[1] - t[0]
    # fs = 1 / ts

    # Check that Rs is of rank 1
    # for i in range(len(f)):
    #     rank = np.linalg.matrix_rank(Rs[i])
    #     print(f"Rank of Rs at f = {f[i]} Hz : {rank}")

    return f, rtf, Rx, Rs, Rv


def rtf_cs(f, n_rcv, Rx, Rv):
    """
    Derive RTF vector using covariance subtraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    Reference receiver is assumed to be the first one.

    Parameters:
    f : ndarray
        Frequencies vector.
    n_rcv : int
        Number of receivers
    Rx : ndarray
        3D CSD matrix for signal at receiver positions (frequency bins x num_receivers x num_receivers).
    Rv : ndarray
        3D CSD matrix for noise at receiver positions (frequency bins x num_receivers x num_receivers).

    Returns:
    f : ndarray
        Frequencies vector.
    rtf : ndarray
        Relative Transfer Function (RTF) matrix (len(f) x num_receivers).
    """

    R_delta = Rx - Rv  # Equation (9)

    # Faster implementation
    # Reference receiver is assumed to be the first one
    e1 = np.eye(n_rcv)[:, 0]

    # Vectorized computation of rtf across all frequencies
    R_delta_e1 = R_delta @ e1
    e1_TR_delta_e1 = e1.T @ R_delta @ e1
    rtf = R_delta_e1 / e1_TR_delta_e1[:, None]

    # print(f"Ellapsed time (fast) = {time()-t0}")

    return f, rtf


def get_csdm(
    t,
    noisy_signal,
    noise_only,
    signal_only=None,
    nperseg=2**12,
    noverlap=2**11,
):
    """
    Derive the CSDM of the received signal and noise.
    Shape of received signal and noise must be (ns, nrcv) where ns is the number of samples and nrcv is the number of receivers
    """

    ff, csdm_x = get_csdm_from_signal(t, noisy_signal, nperseg, noverlap)
    ff, csdm_noise = get_csdm_from_signal(t, noise_only, nperseg, noverlap)

    if signal_only is not None:
        ff, csdm_sig = get_csdm_from_signal(t, signal_only, nperseg, noverlap)
    else:
        csdm_sig = None

    return ff, csdm_x, csdm_sig, csdm_noise


def get_csdm_from_signal(t, y, nperseg=2**12, noverlap=2**11):
    """
    Derive the CSDM of y.
    Shape of y must be (ns, nrcv) where ns is the number of samples and nrcv is the number of receivers
    """
    fs = 1 / (t[1] - t[0])
    ff, _, stft_list = get_stft_array(y, fs, nperseg, noverlap)

    # t0 = time()
    csdm_y = compute_csd_matrix_fast(stft_list, n_seg_cov=0)
    # print(f"Ellapsed time (first) : {time() - t0}s")

    return ff, csdm_y


def get_stft_array(y, fs, nperseg, noverlap):
    """
    Derive the STFT of each component of y.

    Args:
    - y (np.ndarray): A 2D array with shape (ns, nrcv), where ns is the number of samples and nrcv is the number of receivers.
    - fs (float): Sampling frequency of the signal.
    - nperseg (int): Length of each segment for the STFT.
    - noverlap (int): Number of overlapping samples between consecutive segments.

    Returns:
    - ff (np.ndarray): Array of frequency bins for the STFT (shape: (n_freq_bins,)).
    - tt (np.ndarray): Array of time bins for the STFT (shape: (n_time_bins,)).
    - stft_list (np.ndarray): 3D array with the STFT of each receiver (shape: (n_rcv, n_freq_bins, n_time_bins)), where n_rcv is the number of receivers.
    """

    # t0 = time()
    ff, tt, stft_array = sp.stft(
        y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, axis=0
    )
    stft_array = np.moveaxis(stft_array, 1, 0)
    # print(f"Ellapsed time (direct) : {time() - t0}s")

    return ff, tt, stft_array


if __name__ == "__main__":
    pass
