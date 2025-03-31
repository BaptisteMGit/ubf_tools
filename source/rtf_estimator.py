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
import numpy as np
import source.global_constants as g
from scipy.linalg import eigh
from source.cov_manager import CovManager


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

    def covariance_subtraction(
        self,
        t: np.ndarray,
        noisy_signal: np.ndarray,
        noise_only: np.ndarray,
        nperseg: int = 2**12,
        noverlap: int = 2**11,
        window: str = "hann",
        use_first_column=False,
    ):
        """
        Derive the RTF using covariance subtraction method described in Markovich-Golan, S., & Gannot, S. (2015).
        :param t: Time vector.
        :param noisy_signal: Noisy signal.
        :param noise_only: Noise only signal.
        :param nperseg: Number of samples per segment used to derive CSDM.
        :param noverlap: Number of overlapping samples between consecutive segments used to derived CSDM.
        :param window: Window function used to derive CSDM.
        :param first_column: Boolean indicating if estimation should be performed using the first column of the CSDM.
        Otherwise, rtf is estimated from the major eigen vector of the CSDM.
        :return: Frequencies vector, RTF, CSDM of the noisy signal, CSDM of the noise signal.

        """

        if use_first_column:
            add_identity_noise = False
        else:
            add_identity_noise = True

        cm = CovManager(nperseg=nperseg, noverlap=noverlap, window=window)
        fs = 1 / (t[1] - t[0])
        f, Rx = cm.get_signal_csdm(y=noisy_signal, fs=fs, add_identity=False)
        f, Rv = cm.get_signal_csdm(y=noise_only, fs=fs, add_identity=add_identity_noise)

        rtf = self.estimate_rtf_covariance_subtraction(
            Rx - Rv, use_first_column=use_first_column
        )

        return f, rtf, Rx, Rv

    def covariance_whitening(
        self,
        t: np.ndarray,
        noisy_signal: np.ndarray,
        noise_only: np.ndarray,
        nperseg: int = 2**12,
        noverlap: int = 2**11,
        window: str = "hann",
    ) -> np.ndarray:
        """
        Derive the RTF using covariance whitening method described in Markovich-Golan, S., & Gannot, S. (2015).
        :param t: Time vector.
        :param noisy_signal: Noisy signal.
        :param noise_only: Noise only signal.
        :param nperseg: Number of samples per segment used to derive CSDM.
        :param noverlap: Number of overlapping samples between consecutive segments used to derived CSDM.
        :param window: Window function used to derive CSDM.
        :return: Frequencies vector, RTF, CSDM of the noisy signal, CSDM of the speech signal.
        """

        cm = CovManager(nperseg=nperseg, noverlap=noverlap, window=window)
        fs = 1 / (t[1] - t[0])
        f, Rx = cm.get_signal_csdm(y=noisy_signal, fs=fs, add_identity=False)
        f, Rv = cm.get_signal_csdm(y=noise_only, fs=fs, add_identity=True)

        rtf = self.estimate_rtf_covariance_whitening(Rx, Rv)

        return f, rtf, Rx, Rv

    def estimate_rtf_covariance_subtraction(
        self, clean_signal_csdm: np.ndarray, use_first_column: bool = False
    ) -> np.ndarray:
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
        self, noisy_cpsd: np.ndarray, noise_cpsd: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the RTF using the covariance whitening method.
        :param noisy_cpsd: Noisy CSDMs (frequency bins x num_receivers x num_receivers).
        :param noise_cpsd: Noise CSDMs covariances (frequency bins x num_receivers x num_receivers).
        :return: Estimated RTF.
        """
        rtf = self.covariance_whitening_cholesky(noisy_cpsd, noise_cpsd)

        return rtf

    @staticmethod
    def covariance_subtraction_first_column(
        clean_signal_csdm: np.ndarray,
    ) -> np.ndarray:
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

        rtf = clean_signal_csdm_e1 / (clean_signal_csdm_e11[:, np.newaxis] + g.eps)

        return rtf

    @classmethod
    def covariance_subtraction_major_eigen_vector(
        cls, clean_signal_csdm: np.ndarray, idx_rcv_ref: int = 0
    ) -> np.ndarray:
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
            _, major_eigve = eigh(
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
        eigva: np.ndarray, eigve: np.ndarray, num_to_keep: int = 1
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
        eigve_single_column: np.ndarray, idx_rcv_ref: int = 0
    ) -> np.ndarray:
        """
        Normalize eigenvector to 1 at the reference microphone.
        :param eigve_single_column: Eigenvector.
        :param idx_rcv_ref: Index of the reference microphone.
        :return: Normalized eigenvector.
        """

        # Normalize input vector at the reference microphone
        if np.abs(eigve_single_column[idx_rcv_ref]) < g.eps:
            eigve_normalized = np.zeros_like(eigve_single_column)
        else:
            eigve_normalized = eigve_single_column / eigve_single_column[idx_rcv_ref]

        return eigve_normalized

    @classmethod
    def covariance_whitening_cholesky(
        cls, noisy_cpsd: np.ndarray, noise_cpsd: np.ndarray
    ) -> np.ndarray:
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
        cls, noisy_cpsd: np.ndarray, noise_cpsd: np.ndarray
    ) -> np.ndarray:
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
        maj_eigva, maj_eigve_whitened = eigh(
            noisy_cpsd_whitened,
            subset_by_index=[nr - 1, nr - 1],  # Only get the major eigenvector
        )

        return maj_eigva, maj_eigve_whitened, noise_cpsd_sqrt

    @staticmethod
    def whiten_covariance(noisy_cpsd: np.ndarray, noise_cpsd: np.ndarray) -> np.ndarray:
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


if __name__ == "__main__":
    pass
