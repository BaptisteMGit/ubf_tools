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

    Attributes
    ----------
    nperseg : int
        Length of each segment for the STFT.
    noverlap : int
        Number of overlapping samples between consecutive segments.
    window : str
        Window function to apply to each segment.

    Methods
    -------
    get_signal_csdm(y, fs, add_identity=False)
        Derive the CSDM of y.
    get_stft_array(y, fs, nperseg=2**12, noverlap=2**11, window="hann")
        Derive the STFT of each component of y.
    compute_csdm_fast(stfts, n_seg_cov=0)
        Compute the Cross Spectral Density Matrix (CSDM) from STFTs.

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

        Parameters
        ----------
        y : np.ndarray
            A 2D signals array with shape (num_samples, num_receivers).
        fs : float
            Sampling frequency of the signal.
        add_identity : bool
            Add identity matrix to the covariance matrix.

        Returns
        -------
        ff : np.ndarray
            Frequency bins.
        csdm_y : np.ndarray
            Corresponding CSDM (num_frequency_bins x num_receivers x num_receivers).
        """

        ff, _, stft_arr = self.get_stft_array(
            y, fs, self.nperseg, self.noverlap, self.window
        )
        csdm_y = self.compute_csdm_fast(stft_arr, n_seg_cov=0)

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

        Parameters
        ----------
        y : np.ndarray
            A 2D signals array with shape (num_samples, num_receivers).
        fs : float
            Sampling frequency of the signal.
        nperseg : int
            Length of each segment for the STFT.
        noverlap : int
            Number of overlapping samples between consecutive segments.
        window : str
            Window function to apply to each segment.

        Returns
        -------
        ff : np.ndarray
            Frequency bins.
        tt : np.ndarray
            Time bins.
        stft_array : np.ndarray
            STFTs array (num_receivers, num_frequency_bins, num_snapshots).
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

        Parameters
        ----------
        stfts : np.ndarray
            STFTs array (num_receivers, num_frequency_bins, num_snapshots).
        n_seg_cov : int
            Number of time snapshots to average over (number of segments per block).

        Returns
        -------
        np.ndarray
            3D CSDM (num_frequency_bins x num_receivers x num_receivers).
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

        # We can check the optimal path for the einsum operation
        # path_info = np.einsum_path(
        #     "ftr,fts->frs", stft_block, stft_block_conj, optimize="greedy"
        # )

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

    @staticmethod
    def csdm_5D(
        stft_5D: np.ndarray,
        dims_order: dict = {"r": 0, "f": 1, "y": 2, "x": 3, "t": 4},
    ) -> np.ndarray:
        """
        Compute the Cross Spectral Density Matrix (CSDM) from a multi-dimensional STFT array.

        Parameters
        ----------
        stft_5D : np.ndarray
            5D STFTs array, default shape is (num_receivers, num_frequency_bins, num_y, num_x, num_snapshots),
            one can use a different input shape by specifying dims_order.
        dims_order : dict
            Dictionary specifying the order of the dimensions in the input array.

        Returns
        -------
        np.ndarray
            5D array containing the CSDMs at all x, y positions (num_frequency_bins x num_receivers x num_receivers x num_y x num_x).
        """

        # Stft is a (nrcv, nf, ny, nx, nt) array -> for coherence with the compute_csdm_fast implementation of the
        # we can reshape the array into (nrcv, nf, nt, ny, nx)

        # 1) Reshape the stft array to ensure order is (nf, nt, nr, ny, nx)
        # Get the axis order
        dims = dims_order.keys()
        target_order = {"r": 2, "f": 0, "t": 1, "y": 3, "x": 4}
        axis_src = [dims_order[dim] for dim in dims]
        axis_dst = [target_order[dim] for dim in dims]
        # Reshape
        stft_5D = np.moveaxis(stft_5D, axis_src, axis_dst)

        # 2) Compute CSDMs
        # We can compute the CSDMs on the whole dataset before estimating the RTFs by applying same einsum
        # operations as in the CovManager class
        # stfts dimensions are (nf, nt, nrcv, ny, nx) = (f,t,r,y,x)
        # conjuagted stfts dimensions are (nrcv, nf, nt, ny, nx) = (f,t,s,y,x)  (r, s are the receiver indices)
        csdm = np.einsum(
            "ftryx,ftsyx->frsyx", stft_5D, np.conj(stft_5D)
        )  # (nf,nrcv,nrcv,ny,nx)

        # In the previous line t indices disappear as we sum over them to compute the CSDM
        # We can check that no there is no optimized version of the previous  path_info = np.einsum_path(
        #     "ftryx,ftsyx->frsyx", stfts, np.conj(stfts), optimize="greedy"
        # )
        csdm = (
            csdm / stft_5D.shape[1]
        )  # Normalization by the number of time samples to get the average

        return csdm

    @staticmethod
    def get_major_eigve_5D(
        csdm_5D: np.ndarray,
        dims_order: dict = {"f": 0, "r1": 1, "r2": 2, "y": 3, "x": 4},
    ):

        #  1) Reshape the csdm array to ensure order is (nf, ny, nx, nrcv, nrcv) as required by np.linalg.eigh
        # Get the axis order
        dims = dims_order.keys()
        target_order = {"f": 0, "r1": 3, "r2": 4, "y": 1, "x": 2}
        axis_src = [dims_order[dim] for dim in dims]
        axis_dst = [target_order[dim] for dim in dims]
        # Reshape
        csdm_5D = np.moveaxis(csdm_5D, axis_src, axis_dst)

        # 2) Compute eigen decomposition
        eigva, eigve = np.linalg.eigh(csdm_5D)

        # 3) Sort eigenvalues and eigenvectors to get the major eigenvector
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(np.real(eigva), axis=-1)[::-1]
        # eigva_sorted = np.take_along_axis(eigva, idx, axis=-1)
        eigve_sorted = np.take_along_axis(
            eigve, idx[..., np.newaxis, :], axis=-1
        )  # (nf, ny, nx, nrcv, nrcv)

        # # Assert it is still a valid eigendecomposition
        # assert np.alltrue(
        #     [
        #         np.allclose(
        #             np.dot(Rdelta_[i, j, k, ...], eigve_sorted[i, j, k, :, iv]),
        #             eigva_sorted[i, j, k, iv] * eigve_sorted[i, j, k, :, iv],
        #         )
        #         for i in range(Rdelta_.shape[0])
        #         for j in range(Rdelta_.shape[1])
        #         for k in range(Rdelta_.shape[2])
        #         for iv in range(eigva.shape[-1])
        #     ]
        # )

        # Extract major eigenvector
        major_eigve = eigve_sorted[..., -1]  # (nf, ny, nx, nrcv)
        # major_eigva = eigva_sorted[..., -1]

        # Move receiver axis in first position
        major_eigve = np.moveaxis(major_eigve, -1, 0)  # (nrcv, nf, ny, nx)

        return major_eigve
