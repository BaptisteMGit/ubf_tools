#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimator.py
@Time    :   2025/03/31 09:25:47
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class dedicated to the estimation of the RTF.

Some of the functions are adapted from the SVD-direct package available at https://github.com/Screeen/SVD-direct

G. Bologni, R. C. Hendriks and R. Heusdens, "Wideband Relative Transfer Function (RTF) Estimation Exploiting Frequency Correlations," in IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 731-747, 2025, doi: 10.1109/TASLPRO.2025.3533371.

"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import scipy
import numpy as np

import propa.rtf.rtf_global_constants as g_rtf

# ======================================================================================================================
# Class
# ======================================================================================================================


class RTFEstimator:
    """
    Class dedicated to the estimation of the RTF.
    Two methods are available:
    - covariance subtraction
    - covariance whitening

    Covariance subtraction method is available with two options:
    - using the first column of the CSDM
    - using the major eigen vector of the CSDM

    Both methods are described in:
    Markovich-Golan, S., & Gannot, S. (2015). Performance analysis of the covariance subtraction method for relative
    transfer function estimation and comparison to the covariance whitening method. 2015 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), 544-548. https://doi.org/10.1109/ICASSP.2015.7178028

    """

    def __init__(self, idx_rcv_ref=0):
        self.index_reference_rcv = idx_rcv_ref

    def estimate_rtf_covariance_subtraction(
        self, clean_signal_csdm: np.array, use_first_column: bool = False
    ) -> np.array:
        """
        Estimate the RTF using the covariance subtraction method.
        :param clean_signal_csdm: Clean speech CSDM at all frequency bins
        (3D array: frequency bins x num_receivers x num_receivers).
        :param use_first_column: Boolean indicating if estimation should be performed using the clean_signal_csdm first
        column. Otherwise, rtf is estimated from the major eigen vector of the clean_signal_csdm.
        :return: Estimated RTF.
        """
        if use_first_column:
            rtf = self.covariance_subtraction_first_column(clean_signal_csdm)
        else:
            rtf = self.covariance_subtraction_major_eigen_vector(
                clean_signal_csdm, self.index_reference_rcv
            )

        return rtf

    def estimate_rtf_covariance_whitening(
        self, noisy_cpsd: np.array, noise_cpsd: np.array
    ) -> np.array:
        """
        Estimate the RTF using the covariance whitening method.
        :param noisy_cpsd: Noisy CSDMs (frequency bins x num_receivers x num_receivers).
        :param noise_cpsd: Noise CSDMs covariances (frequency bins x num_receivers x num_receivers).
        :return: Estimated RTF.
        """
        rtf = self.covariance_whitening_cholesky(noisy_cpsd, noise_cpsd)

        return rtf

    @staticmethod
    def covariance_subtraction_first_column(clean_signal_csdm: np.array) -> np.array:
        """
        Estimate the RTF using the covariance subtraction method with the first column of the CSDM.
        :param clean_signal_csdm: Clean speech CSDM (frequency bins x num_receivers x num_receivers).
        :return: Estimated RTF.
        """
        e1 = np.eye(clean_signal_csdm.shape[-1])[:, 0]

        # Vectorized computation of rtf across all frequencies
        clean_signal_csdm_e1 = (
            clean_signal_csdm @ e1
        )  # First columns of CSDMs (for all freqs)
        clean_signal_csdm_e11 = (
            e1.T @ clean_signal_csdm @ e1
        )  # First entry of first column of CSDMs (for all freqs)

        rtf = clean_signal_csdm_e1 / (clean_signal_csdm_e11[:, np.newaxis] + g_rtf.eps)

        return rtf

    @classmethod
    def covariance_subtraction_major_eigen_vector(
        cls, clean_signal_csdm: np.array, idx_rcv_ref: int = 0
    ) -> np.array:
        """
        Estimate the RTF using the covariance subtraction method with the major eigen vector of the CSDM.
        :param clean_signal_csdm: Clean speech CSDM (frequency bins x num_receivers x num_receivers).
        :return: Estimated RTF.
        """

        # from time import time
        # t0 = time()
        def covariance_subtraction_major_eigen_vector_f(
            clean_signal_csdm_single_freq, nr, idx_rcv_ref=0
        ):
            _, major_eigve = scipy.linalg.eigh(
                clean_signal_csdm_single_freq,
                check_finite=False,
                subset_by_index=[nr - 1, nr - 1],  # Only get the major eigenvector
            )

            rtf_f = cls.normalize_eigve_to_1(major_eigve, idx_rcv_ref)

            return rtf_f

        nf, nr, _ = clean_signal_csdm.shape
        # Using map is slighly faster than using an explicit for loop
        rtf = list(
            map(
                covariance_subtraction_major_eigen_vector_f,
                [clean_signal_csdm[k, ...] for k in range(nf)],
                [nr] * nf,
                [idx_rcv_ref] * nf,
            )
        )
        rtf = np.array(rtf)
        # print(f"Ellapsed time (map) = {time()-t0}")

        return np.squeeze(rtf)

    @staticmethod
    def sort_eigenvectors_get_major(
        eigva: np.array, eigve: np.array, num_to_keep: int = 1
    ) -> tuple:
        """
        Sort eigenvectors and eigenvalues in descending order and return the major eigenvector.
        This function is adapted from the original code in the SVD-direct package and is replaced when possible
        here by the subset_by_index args of scipy.linalg.eigh.
        :param eigva: Eigenvalues.
        :param eigve: Eigenvectors.
        :return: Sorted eigenvalues and eigenvectors and major eigenvector.
        """

        if num_to_keep == -1:
            num_to_keep = len(eigva)  # keep all eigenvectors

        if not np.all(np.isfinite(eigva)):
            return (
                np.ones_like(eigva)[:num_to_keep] * np.nan,
                np.ones_like(eigve)[:, :num_to_keep] * np.nan,
            )

        # Sort eigenvalues and eigenvectors in descending order
        descending_indices = np.argsort(np.real(eigva))[::-1]
        eigva, eigve = (
            eigva[descending_indices],
            eigve[:, descending_indices],
        )

        return eigva[:num_to_keep], eigve[:, :num_to_keep]

    @staticmethod
    def normalize_eigve_to_1(
        eigve_single_column: np.array, idx_rcv_ref: int = 0
    ) -> np.array:
        """
        Normalize eigenvector to 1 at the reference microphone.
        :param eigve_single_column: Eigenvector.
        :param idx_rcv_ref: Index of the reference microphone.
        :return: Normalized eigenvector.
        """

        # Normalize input vector at the reference microphone
        if np.abs(eigve_single_column[idx_rcv_ref]) < g_rtf.eps:
            eigve_normalized = np.zeros_like(eigve_single_column)
        else:
            eigve_normalized = eigve_single_column / eigve_single_column[idx_rcv_ref]

        return eigve_normalized

    @classmethod
    def covariance_whitening_cholesky(
        cls, noisy_cpsd: np.array, noise_cpsd: np.array
    ) -> np.array:
        """
        Estimate the RTF using the covariance whitening method.
        Noise CSDM must be Hermitian (symmetric if real-valued) and positive-definite.
        :param noisy_cpsd: Noisy CSDMs (frequency bins x num_receivers x num_receivers).
        :param noise_cpsd: Noise CSDMs (frequency bins x num_receivers x num_receivers).
        :return: Estimated RTF.
        """

        rtf = np.zeros((noisy_cpsd.shape[0], noisy_cpsd.shape[1]), dtype=complex)
        for k in range(noise_cpsd.shape[0]):

            # Get major eigenvector of whitened noisy covariance
            _, maj_eigve_noisy_whitened, noise_cpsd_sqrt = (
                cls.get_eigenvectors_whitened_noisy_cov(
                    noisy_cpsd[k, ...], noise_cpsd[k, ...]
                )
            )
            # Transform back from whitened domain
            rtf_ = noise_cpsd_sqrt @ maj_eigve_noisy_whitened
            rtf[k, :] = np.squeeze(cls.normalize_eigve_to_1(rtf_))

        return rtf

    @classmethod
    def get_eigenvectors_whitened_noisy_cov(
        cls, noisy_cpsd: np.array, noise_cpsd: np.array
    ) -> np.array:
        """
        Get the major eigenvector of the whitened noisy CSDM.
        :param noisy_cpsd: Noisy CSDMs (frequency bins x num_receivers x num_receivers).
        :param noise_cpsd: Noise CSDMs (frequency bins x num_receivers x num_receivers).
        :return: Major eigenvalue, major eigenvector of the whitened noisy covariance, Cholesky factor of the noise
        spatial covariance.
        """

        noise_cpsd_sqrt, noisy_cpsd_whitened = cls.whiten_covariance(
            noisy_cpsd, noise_cpsd
        )
        nr = noisy_cpsd.shape[-1]
        maj_eigva, maj_eigve_whitened = scipy.linalg.eigh(
            noisy_cpsd_whitened,
            subset_by_index=[nr - 1, nr - 1],  # Only get the major eigenvector
        )

        return maj_eigva, maj_eigve_whitened, noise_cpsd_sqrt

    @staticmethod
    def whiten_covariance(noisy_cpsd: np.array, noise_cpsd: np.array) -> np.array:
        """
        1) Perform Cholesky decomposition on noise_cpsd: noise_cpsd = L @ L.conj().T
        2) Calculate whitened covariance R_white = L^-1 @  noisy_cpsd @ (L^(-1))^H
        :param noisy_cpsd: Noisy CSDMs (frequency bins x num_receivers x num_receivers).
        :param noise_cpsd: Noise CSDMs (frequency bins x num_receivers x num_receivers).
        :return: Cholesky factor L, whitened noisy spatial covariance
        """
        noise_cpsd_sqrt = np.linalg.cholesky(noise_cpsd)
        noise_cpsd_sqrt_inv = np.linalg.inv(noise_cpsd_sqrt)
        noisy_cpsd_whitened = (
            noise_cpsd_sqrt_inv @ noisy_cpsd @ noise_cpsd_sqrt_inv.conj().T
        )
        # assert u.is_hermitian(noisy_cpsd_whitened)
        return noise_cpsd_sqrt, noisy_cpsd_whitened
