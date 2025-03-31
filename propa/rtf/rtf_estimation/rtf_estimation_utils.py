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
from source.rtf_estimator import RTFEstimator

# from real_data_analysis.real_data_utils import (
#     compute_csd_matrix_fast,
# )


def rtf_covariance_whitening(
    t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11
):
    """
    Derive the RTF using covariance whitening method described in  Markovich-Golan, S., & Gannot, S. (2015).
    """

    f, Rv = get_csdm_from_signal(t, noise_only, nperseg, noverlap, add_identity=True)
    f, Rx = get_csdm_from_signal(t, noisy_signal, nperseg, noverlap, add_identity=True)
    Rs = None
    re = RTFEstimator()
    rtf = re.estimate_rtf_covariance_whitening(Rx, Rv)

    return f, rtf, Rx, Rs, Rv


def rtf_covariance_subtraction(
    t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11, first_column=False
):
    """
    Derive the RTF using covariance subtraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    Reference receiver is assumed to be the first one.
    """

    if first_column:
        add_identity_noise = False
    else:
        add_identity_noise = True

    f, Rx = get_csdm_from_signal(t, noisy_signal, nperseg, noverlap, add_identity=False)
    f, Rv = get_csdm_from_signal(
        t, noise_only, nperseg, noverlap, add_identity=add_identity_noise
    )
    Rs = None
    Rs_tild = Rx - Rv

    re = RTFEstimator()
    rtf = re.estimate_rtf_covariance_subtraction(Rs_tild, use_first_column=first_column)

    return f, rtf, Rx, Rs, Rv


def get_csdm(
    t,
    noisy_signal,
    noise_only,
    signal_only=None,
    nperseg=2**12,
    noverlap=2**11,
    add_identity=False,
):
    """
    Derive the CSDM of the received signal and noise.
    Shape of received signal and noise must be (ns, nrcv) where ns is the number of samples and nrcv is the number of receivers
    """

    ff, csdm_x = get_csdm_from_signal(
        t, noisy_signal, nperseg, noverlap, add_identity=add_identity
    )
    ff, csdm_noise = get_csdm_from_signal(
        t, noise_only, nperseg, noverlap, add_identity=add_identity
    )

    if signal_only is not None:
        ff, csdm_sig = get_csdm_from_signal(
            t, signal_only, nperseg, noverlap, add_identity=add_identity
        )
    else:
        csdm_sig = None

    return ff, csdm_x, csdm_sig, csdm_noise


def get_csdm_from_signal(t, y, nperseg=2**12, noverlap=2**11, add_identity=False):
    """
    Derive the CSDM of y.
    Shape of y must be (ns, nrcv) where ns is the number of samples and nrcv is the number of receivers
    """
    fs = 1 / (t[1] - t[0])
    ff, _, stft_list = get_stft_array(y, fs, nperseg, noverlap)

    # t0 = time()
    csdm_y = compute_csd_matrix_fast(stft_list, n_seg_cov=0)
    # print(f"Ellapsed time (first) : {time() - t0}s")

    if add_identity:
        diagonal_loading = 1e-8  # amount of diagonal loading when adding identity matrix to covariance matrix
        csdm_y = (
            csdm_y + diagonal_loading * np.identity(csdm_y.shape[-1])[np.newaxis, ...]
        )
    return ff, csdm_y


def compute_csd_matrix_fast(stfts, n_seg_cov):
    """
    Compute the Cross Spectral Density (CSD) matrix for a set of receivers using matrix operations.

    Args:
    - stfts: list of 2D STFT matrices (frequency bins x time snapshots), one per receiver.
    - n_seg_cov: Number of time snapshots to average over (number of segments per block).

    Returns:
    - csd_matrix: 3D CSD matrix (frequency bins x num_receivers x num_receivers).
    """
    num_receivers = len(stfts)
    num_freq_bins, num_snapshots = stfts[0].shape

    if n_seg_cov == 0:
        n_seg_cov = num_snapshots

    n_available_segments = num_snapshots // n_seg_cov

    # Convert list of arrays into a single array
    stacked_stfts = np.asarray(
        stfts
    )  # Shape: (num_receivers, num_freq_bins, num_snapshots)
    stacked_stfts = np.moveaxis(
        stacked_stfts, 0, -1
    )  # (num_freq_bins, num_snapshots, num_receivers)

    # Preallocate CSD matrix
    csd_matrix = np.empty(
        (num_freq_bins, num_receivers, num_receivers, n_available_segments),
        dtype=np.complex128,
    )

    # Compute CSD matrix using batch operations
    for k in range(n_available_segments):
        idx_start = k * n_seg_cov
        stft_block = stacked_stfts[
            :, idx_start : idx_start + n_seg_cov, :
        ]  # View-based slicing
        stft_block_conj = np.conj(stft_block)  # Precompute conjugate

        csd_matrix[..., k] = (
            np.einsum("ftr,fts->frs", stft_block, stft_block_conj) / n_seg_cov
        )

    return np.squeeze(csd_matrix, axis=-1) if n_available_segments == 1 else csd_matrix


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
