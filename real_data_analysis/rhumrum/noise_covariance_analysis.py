#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   noise_covariance_analysis.py
@Time    :   2024/10/04 13:52:27
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
import matplotlib.pyplot as plt
from real_data_analysis.real_data_utils import *

# ======================================================================================================================
# Functions
# ======================================================================================================================

# 1) Load data
# 2) Compute covariance matrix
# 3) Plot covariance matrix
# 4) Compute eigenvalues and eigenvectors of the covariance matrix
# 5) Plot eigenvalues
# 6) Plot eigenvectors


def load_data_vector(
    date,
    rcv_id,
    duration_s,
    freq_properties,
    ch=["BDH"],
):
    y_data = {}
    for rcv_id in rcv_ids:
        data = load_wav_data(
            date=date,
            rcv_id=rcv_id,
            duration_s=duration_s,
            ch=ch,
            freq_properties=freq_properties,
        )

        y_data[rcv_id] = data

    return y_data


def compute_covariance_matrix(y_data, n_seg_cov):
    """
    Derive spatial covariance matrix also usually refered to as the Cross Spectral Density Matrix (CSDM).

    The derivation adopted follows the scheme proposed by to Polichetti, M., Varray, F., Gilles, B., Béra, J.-C., & Nicolas, B. (2021). Use of the Cross-Spectral Density Matrix for Enhanced Passive Ultrasound Imaging of Cavitation. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 68(4), 910–925. https://doi.org/10.1109/TUFFC.2020.3032345

    The estimated CSDM M(f) is given by :

    M(f) = 1/K * Sum( 1 / T_snap * Y_k(f) * Y_k(f)^H )

    Where :

        K is the number of snapshots
        T_snap is the length of the snapshots

    One can consider overlapping snapshots to artificially increase K. Then,

    K = T / ( (1 - alpha_overlap) * T_snap )

    Where:

        T is the total length of the recording
    """

    rcv_ids = list(y_data.keys())

    (nfreq, nseg) = y_data[rcv_ids[0]]["stft"].shape
    n_available_segments = np.floor(nseg / n_seg_cov).astype(int)
    cov_mat = np.zeros(
        (len(rcv_ids), len(rcv_ids), nfreq, n_available_segments), dtype=np.complex128
    )

    for j in range(nfreq):
        for k in range(n_available_segments):
            cov_fj = np.zeros((len(rcv_ids), len(rcv_ids)), dtype=np.complex128)

            # Create y_fj
            y_fj = []
            for i, rcv_id in enumerate(rcv_ids):
                y_fj_i = y_data[rcv_id]["stft"][j, :]
                y_fj.append(y_fj_i)
            y_fj = np.array(y_fj)

            idx_start = k * n_seg_cov
            for l in range(n_seg_cov):
                y_fj_l = y_fj[:, idx_start + l : idx_start + l + 1]
                cov_fj += y_fj_l @ np.conjugate(y_fj_l).T
            cov_fj /= n_seg_cov

            cov_mat[..., j, k] = cov_fj

    return cov_mat


def compute_csd_matrix(stfts):
    """
    Compute the CSD matrix from a list of STFT matrices, one for each receiver.
    Each STFT matrix is 2D (frequency bins x time snapshots).

    Args:
    - stfts: list of 2D STFT matrices, one per receiver.

    Returns:
    - csd_matrix: 3D CSD matrix (frequency bins x num_receivers x num_receivers).
    """
    num_receivers = len(stfts)  # Number of receivers
    num_freq_bins, num_snapshots = stfts[0].shape  # Frequency bins and time snapshots
    csd_matrix = np.zeros(
        (num_freq_bins, num_receivers, num_receivers), dtype=np.complex128
    )

    # Loop over time snapshots to compute the CSD for each snapshot and average it
    for t in range(num_snapshots):
        # For each time snapshot, compute outer products for all pairs of receivers
        for i in range(num_receivers):
            for j in range(i, num_receivers):
                csd_matrix[:, i, j] += (
                    np.conj(stfts[i][:, t]) * stfts[j][:, t]
                )  # Cross-spectral density
                if i != j:
                    csd_matrix[:, j, i] = np.conj(
                        csd_matrix[:, i, j]
                    )  # Ensure the matrix is Hermitian

    # Normalize by the number of snapshots
    csd_matrix /= num_snapshots
    return csd_matrix




def compute_covariance_matrix_from_time_series(y_data, n_seg_cov):

    rcv_ids = list(y_data.keys())

    (nfreq, nseg) = y_data[rcv_ids[0]]["stft"].shape
    n_available_segments = np.floor(nseg / n_seg_cov).astype(int)
    cov_mat = np.zeros(
        (len(rcv_ids), len(rcv_ids), n_available_segments), dtype=np.complex128
    )

    for k in range(n_available_segments):

        # Create y vector
        y = []
        for i, rcv_id in enumerate(rcv_ids):
            y_i = y_data[rcv_id]["data"]
            y.append(y_i)
        y = np.array(y)

        idx_start = k * n_seg_cov
        y_seg = y[:, idx_start : idx_start + n_seg_cov + 1]
        cov_seg = y_seg @ np.conjugate(y_seg).T

        cov_mat[..., k] = cov_seg

    return cov_mat


def plot_covariance_matrix(cov_matrix):
    pass


def compute_eigenvalues_and_eigenvectors(cov_matrix):
    pass


def plot_eigenvalues(eigenvalues):
    pass


def plot_eigenvectors(eigenvectors):
    pass


# ======================================================================================================================
# Main
# ======================================================================================================================

if __name__ == "__main__":
    from time import time

    ch = ["BDH"]
    rcv_id = "RR46"
    duration_seconds = 1 * 60 * 60
    date = "2013-05-26 04:15:00"
    duration_seconds = 2 * 60 * 60
    date = "2013-05-04 12:10:00"

    yyyy_mm = "2013-04"

    save = True
    # Def frequency properties
    nperseg = 2**13
    noverlap = int(nperseg * 3 / 4)

    fmin = 4
    fmax = 46
    filter_type = "bandpass"
    filter_corners = 4

    freq_properties = {
        "fmin": fmin,
        "fmax": fmax,
        "filter_type": filter_type,
        "filter_corners": filter_corners,
        "noverlap": noverlap,
        "nperseg": nperseg,
    }
    # rcv_ids = ["RR41", "RR43", "RR44", "RR46", "RR47", "RR48"]
    rcv_ids = ["RR41", "RR43", "RR46"]

    # Load data
    y_data = load_data_vector(date, rcv_ids, duration_seconds, freq_properties)

    n_seg_cov = 210  # Number of segments to compute the covariance matrix
    t0 = time()
    cov_mat = compute_covariance_matrix(y_data, n_seg_cov)
    print(f"Time to compute covariance matrix: {time() - t0:.2f} s")

    # Fast computation
    t0 = time()
    stfts = [y_data[rcv_id]["stft"] for rcv_id in rcv_ids]
    csd_matrix = compute_csd_matrix_fast(stfts, n_seg_cov)
    print(f"Time to compute CSD matrix: {time() - t0:.2f} s")

    # Cov mat from temporal data
    cov_mat_time = compute_covariance_matrix_from_time_series(y_data, n_seg_cov)

    print(cov_mat.shape)

    # Plot covariance matrix components evolution
    freq_to_plot = 35
    idx_freq = np.argmin(np.abs(y_data[rcv_ids[0]]["f"] - freq_to_plot))
    fplot = y_data[rcv_ids[0]]["f"][idx_freq]
    plt.figure()
    for i in range(csd_matrix.shape[1]):
        for j in range(csd_matrix.shape[2]):
            # plt.plot(
            #     np.abs(cov_mat[i, j, idx_freq, :]),
            #     label=f"{rcv_ids[i]}-{rcv_ids[j]}",
            #     marker=".",
            # )
            plt.plot(
                np.abs(csd_matrix[idx_freq, i, j, :]),
                label=f"CSD_{rcv_ids[i]}-{rcv_ids[j]}",
                marker=".",
            )
            # diff = cov_mat[i, j, idx_freq, :] - np.real(csd_matrix[idx_freq, i, j, :])
            # plt.plot(
            #     diff,
            #     label=f"Diff_{rcv_ids[i]}-{rcv_ids[j]}",
            #     marker=".",
            # )
    plt.legend()
    plt.xlabel("Segment")
    plt.ylabel("Covariance")
    plt.title(f"Covariance matrix components evolution for f={fplot:.2f} Hz")
    plt.grid()

    # # Compare frequency cov to time cov
    # plt.figure()
    # for i in range(cov_mat.shape[0]):
    #     j = i
    #     plt.plot(
    #         cov_mat[i, j, idx_freq, :],
    #         label=f"Cov_freq_{rcv_ids[i]}-{rcv_ids[j]}",
    #         marker=".",
    #     )
    #     plt.plot(
    #         cov_mat_time[i, j, :],
    #         label=f"Cov_time_{rcv_ids[i]}-{rcv_ids[j]}",
    #         marker=".",
    #     )
    #     plt.plot(
    #         np.abs(csd_matrix[idx_freq, i, 2 - j, :]),
    #         label=f"CSD_{rcv_ids[i]}-{rcv_ids[j]}",
    #         marker=".",
    #     )
    # plt.legend()
    # plt.xlabel("Segment")
    # plt.ylabel("Covariance")
    # plt.title(f"Covariance matrix components evolution for f={fplot:.2f} Hz")
    # plt.grid()

    for i_seg in range(1):
        plt.figure()
        im = plt.imshow(
            np.abs(csd_matrix[:, :, :, i_seg].mean(axis=0)),
            cmap="jet",
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im)
        plt.title(f"CSD matrix at f={fplot:.2f} Hz for segment {i_seg}")

        plt.figure()
        im = plt.imshow(
            np.abs(cov_mat_time[:, :, i_seg]),
            cmap="jet",
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im)
        plt.title(f"Cov mat matrix at f={fplot:.2f} Hz for segment {i_seg}")

    plt.show()
