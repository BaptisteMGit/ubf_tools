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

    Attributes
    ----------
    index_reference_rcv : int
        Index of the reference receiver.

    Methods
    -------
    covariance_subtraction(t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11, window="hann", use_first_column=False)
        Derive the RTF using covariance subtraction method.
    covariance_whitening(t, noisy_signal, noise_only, nperseg=2**12, noverlap=2**11, window="hann")
        Derive the RTF using covariance whitening method.
    estimate_rtf_covariance_subtraction(clean_signal_csdm, use_first_column=False)
        Estimate the RTF using the covariance subtraction method.
    estimate_rtf_covariance_whitening(noisy_cpsd, noise_cpsd)
        Estimate the RTF using the covariance whitening method.
    covariance_subtraction_first_column(clean_signal_csdm)
        Estimate the RTF using the covariance subtraction method with the first column of the CSDM.
    covariance_subtraction_major_eigen_vector(clean_signal_csdm, idx_rcv_ref=0)
        Estimate the RTF using the covariance subtraction method with the major eigen vector of the CSDM.
    sort_eigenvectors_get_major(eigva, eigve, num_to_keep=1)
        Sort eigenvectors and eigenvalues in descending order and return the major eigenvector.
    normalize_eigve_to_1(eigve_single_column, idx_rcv_ref=0)
        Normalize eigenvector to 1 at the reference microphone.
    covariance_whitening_cholesky(noisy_cpsd, noise_cpsd)
        Estimate the RTF using the covariance whitening method.
    get_eigenvectors_whitened_noisy_cov(noisy_cpsd, noise_cpsd)
        Get the major eigenvector of the whitened noisy CSDM.
    whiten_covariance(noisy_cpsd, noise_cpsd)
        Perform Cholesky decomposition on noise_cpsd and calculate whitened covariance

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive the RTF using covariance subtraction method described in Markovich-Golan, S., & Gannot, S. (2015).

        Parameters
        ----------
        t : np.ndarray
            Time vector.
        noisy_signal : np.ndarray
            Noisy signal.
        noise_only : np.ndarray
            Noise only signal.
        nperseg : int, optional
            Number of samples per segment used to derive CSDM.
        noverlap : int, optional
            Number of overlapping samples between consecutive segments used to derived CSDM.
        window : str, optional
            Window function used to derive CSDM.
        use_first_column : bool, optional
            Boolean indicating if estimation should be performed using the first column of the CSDM.
            Otherwise, rtf is estimated from the major eigen vector of the CSDM.

        Returns
        -------
        f : np.ndarray
            Frequencies vector.
        rtf : np.ndarray
            RTF.
        Rx : np.ndarray
            CSDM of the noisy signal.
        Rv : np.ndarray
            CSDM of the noise signal.

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive the RTF using covariance whitening method described in Markovich-Golan, S., & Gannot, S. (2015).

        Parameters
        ----------
        t : np.ndarray
            Time vector.
        noisy_signal : np.ndarray
            Noisy signal.
        noise_only : np.ndarray
            Noise only signal.
        nperseg : int, optional
            Number of samples per segment used to derive CSDM.
        noverlap : int, optional
            Number of overlapping samples between consecutive segments used to derived CSDM.
        window : str, optional
            Window function used to derive CSDM.

        Returns
        -------
        f : np.ndarray
            Frequencies vector.
        rtf : np.ndarray
            RTF.
        Rx : np.ndarray
            CSDM of the noisy signal.
        Rv : np.ndarray
            CSDM of the noise signal.

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

        Parameters
        ----------
        clean_signal_csdm : np.ndarray
            Clean speech CSDM at all frequency bins (3D array: num_frequency_bins x num_receivers x num_receivers).
        use_first_column : bool, optional
            Boolean indicating if estimation should be performed using the clean_signal_csdm first column.
            Otherwise, rtf is estimated from the major eigen vector of the clean_signal_csdm.

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF.

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

        Parameters
        ----------
        noisy_cpsd : np.ndarray
            Noisy CSDMs (num_frequency_bins x num_receivers x num_receivers).
        noise_cpsd : np.ndarray
            Noise CSDMs (num_frequency_bins x num_receivers x num_receivers).

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF (num_frequency_bins x num_receivers).

        """
        rtf = self.covariance_whitening_cholesky(noisy_cpsd, noise_cpsd)

        return rtf

    @staticmethod
    def covariance_subtraction_first_column(
        clean_signal_csdm: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the RTF using the covariance subtraction method with the first column of the CSDM.

        Parameters
        ----------
        clean_signal_csdm : np.ndarray
            Clean speech CSDM (num_frequency_bins x num_receivers x num_receivers).

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF (num_frequency_bins x num_receivers).

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

        Parameters
        ----------
        clean_signal_csdm : np.ndarray
            Clean speech CSDM (num_frequency_bins x num_receivers x num_receivers).
        idx_rcv_ref : int, optional
            Index of the reference receiver.

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF (num_frequency_bins x num_receivers).

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
    def covariance_subtraction_major_eigen_vector_5D(
        clean_signal_csdm_5D: np.ndarray,
        idx_rcv_ref: int = 0,
        dims_order: dict = {"f": 0, "r1": 1, "r2": 2, "y": 3, "x": 4},
    ) -> np.ndarray:
        """
        Estimate the RTF using the covariance subtraction method with the major eigen vector of the CSDM in the case of
        a 5D CSDM array (CSDM at all positions on a y, x grid).

        Parameters
        ----------
        clean_signal_csdm_5D : np.ndarray
            Clean speech CSDM at all positions (num_frequency_bins x num_receivers x num_receivers x num_y x num_x).
        idx_rcv_ref : int, optional
            Index of the reference receiver.

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF (num_receivers x num_frequency_bins x num_y x num_x).

        """

        # 1) Reshape the csdm array to ensure order is (nf, ny, nx, nrcv, nrcv) as required by np.linalg.eigh
        # Get the axis order
        dims = dims_order.keys()
        target_order = {"f": 0, "r1": 3, "r2": 4, "y": 1, "x": 2}
        axis_src = [dims_order[dim] for dim in dims]
        axis_dst = [target_order[dim] for dim in dims]
        # Reshape
        clean_signal_csdm_5D = np.moveaxis(clean_signal_csdm_5D, axis_src, axis_dst)

        # 2) Compute eigen decomposition
        eigva, eigve = np.linalg.eigh(clean_signal_csdm_5D)

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

        # Normalize to 1 at idx_rcv_ref
        rtf = major_eigve / np.broadcast_to(
            major_eigve[idx_rcv_ref : idx_rcv_ref + 1, ...], major_eigve.shape
        )  # (nrcv, nf, ny, nx)

        return rtf

    @staticmethod
    def sort_eigenvectors_get_major(
        eigva: np.ndarray, eigve: np.ndarray, num_to_keep: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sort eigenvectors and eigenvalues in descending order and return the major eigenvector.
        This function is adapted from the original code in the SVD-direct package and is replaced when possible
        here by the subset_by_index args of scipy.linalg.eigh

        Parameters
        ----------
        eigva : np.ndarray
            Eigenvalues.
        eigve : np.ndarray
            Eigenvectors.
        num_to_keep : int, optional
            Number of major eigenvectors and eigvalues to return.

        Returns
        -------
        eigva : np.ndarray
            num_to_keep sorted eigenvalues.
        eigve : np.ndarray
            num_to_keep sorted eigenvectors.
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

        Parameters
        ----------
        eigve_single_column : np.ndarray
            Eigenvector.
        idx_rcv_ref : int, optional
            Index of the reference microphone.

        Returns
        -------
        eigve_normalized : np.ndarray
            Normalized eigenvector.

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

        Parameters
        ----------
        noisy_cpsd : np.ndarray
            Noisy CSDMs (num_frequency_bins x num_receivers x num_receivers).
        noise_cpsd : np.ndarray
            Noise CSDMs (num_frequency_bins x num_receivers x num_receivers).

        Returns
        -------
        rtf : np.ndarray
            Estimated RTF (num_frequency_bins x num_receivers).

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the major eigenvector of the whitened noisy CSDM.

        Parameters
        ----------
        noisy_cpsd : np.ndarray
            Noisy CSDMs (num_frequency_bins x num_receivers x num_receivers).
        noise_cpsd : np.ndarray
            Noise CSDMs (num_frequency_bins x num_receivers x num_receivers).

        Returns
        -------
        maj_eigva : np.ndarray
            Major eigenvalue.
        maj_eigve_whitened : np.ndarray
            Major eigenvector of the whitened noisy covariance.
        noise_cpsd_sqrt : np.ndarray
            Cholesky factor of the noise covariance.

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
    def whiten_covariance(
        noisy_cpsd: np.ndarray, noise_cpsd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1) Perform Cholesky decomposition on noise_cpsd: noise_cpsd = L @ L.conj().T
        2) Calculate whitened covariance R_white = L^-1 @  noisy_cpsd @ (L^(-1))^H

        Parameters
        ----------
        noisy_cpsd : np.ndarray
            Noisy CSDMs (num_frequency_bins x num_receivers x num_receivers).
        noise_cpsd : np.ndarray
            Noise CSDMs (num_frequency_bins x num_receivers x num_receivers).

        Returns
        -------
        noise_cpsd_sqrt : np.ndarray
            Cholesky factor L.
        noisy_cpsd_whitened : np.ndarray
            Whitened noisy spatial covariance.

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
