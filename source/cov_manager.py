#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   cov_manager.py
@Time    :   2025/03/31 13:58:16
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Handle covariance matrices.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import source.global_constants as g

from scipy.signal import stft


class CovManager:
    """
    Class to manage covariance matrices.
    """

    def __init__(self, nperseg=2**12, noverlap=2**11, window="hann"):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window

    def get_signal_csdm(
        self, y: np.ndarray, fs: float, add_identity: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Derive the CSDM of y.
        :param y: A 2D signals array with shape (ns, nrcv), where ns is the number of samples and nrcv is the number of receivers.
        :param fs: Sampling frequency of the signal.
        :param add_identity: Add identity matrix to the covariance matrix.
        :return: Frequency bins and corresponding CSDM.
        """
        ff, _, stft_list = self.get_stft_array(
            y, fs, self.nperseg, self.noverlap, self.window
        )
        csdm_y = self.compute_csdm_fast(stft_list, n_seg_cov=0)

        if add_identity:
            csdm_y = (
                csdm_y
                + g.diagonal_loading * np.identity(csdm_y.shape[-1])[np.newaxis, ...]
            )
        return ff, csdm_y

    @staticmethod
    def get_stft_array(
        y: np.ndarray,
        fs: float,
        nperseg: int = 2**12,
        noverlap: int = 2**11,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive the STFT of each component of y.
        :param y: A 2D signals array with shape (ns, nrcv), where ns is the number of samples and nrcv is the number of receivers.
        :param fs: Sampling frequency of the signal.
        :param nperseg: Length of each segment for the STFT.
        :param noverlap: Number of overlapping samples between consecutive segments.
        :param window: Window function to apply to each segment.
        :return: Frequency bins, time bins and STFT array.
        """

        ff, tt, stft_array = stft(
            y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, axis=0
        )
        stft_array = np.moveaxis(stft_array, 1, 0)

        return ff, tt, stft_array

    @staticmethod
    def compute_csdm_fast(stfts: list[np.ndarray], n_seg_cov: int = 0) -> np.ndarray:
        """
        Compute the Cross Spectral Density Matrix (CSDM) from STFTs.
        :param stfts: list of 2D STFT matrices (frequency bins x time snapshots), one per receiver.
        :param n_seg_cov: Number of time snapshots to average over (number of segments per block).
        :return: 3D CSDM (frequency bins x num_receivers x num_receivers).
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

        return (
            np.squeeze(csd_matrix, axis=-1) if n_available_segments == 1 else csd_matrix
        )
