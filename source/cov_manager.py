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
        self,
        y: np.ndarray,
        fs: float,
        add_identity: bool = False,
        mask_tt: np.ndarray = None,
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
        mask_tt : np.ndarray
            Mask to apply on the time snapshots (num_time_snapshots,). If none the whole signal is used.
            The purpose of this parameter is to derive signal + noise CSDM or noise only CSDM directly from
            the complete signal by applying the right mask to the STFTs.

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
        csdm_y = self.compute_csdm_fast(stft_arr, n_seg_cov=0, mask_tt=mask_tt)

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
    def compute_csdm_fast(
        stfts: list[np.ndarray], n_seg_cov: int = 0, mask_tt: np.ndarray = None
    ) -> np.ndarray:
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

        # Convert list of arrays into a single array
        stacked_stfts = np.asarray(
            stfts
        )  # Shape: (num_receivers, num_freq_bins, num_snapshots)
        num_receivers, num_freq_bins, num_snapshots = stacked_stfts.shape

        # Derive the number of CSDM to compute according to the total number of FFT snapshots and the the number of snapshots to use per block
        if n_seg_cov == 0:
            n_seg_cov = num_snapshots
        n_available_segments = num_snapshots // n_seg_cov

        # Apply mask to stfts if provided
        mask_applied = False
        if mask_tt is not None and mask_tt.shape[0] == num_snapshots:
            # Cast to (num_freq_bins, num_snapshots)
            mask = np.repeat(mask_tt[np.newaxis, :], num_freq_bins, axis=0)
            # Stack the mask along the receiver dimension
            stacked_mask = np.repeat(mask[np.newaxis], num_receivers, axis=0)
            # Apply mask
            stacked_stfts *= stacked_mask
            # Store flag to indicate that the mask was applied
            mask_applied = True

        # Reshape the stacked STFTs to (num_freq_bins, num_snapshots, num_receivers)
        stacked_stfts = np.moveaxis(stacked_stfts, 0, -1)

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
        n_mean = n_seg_cov
        for k in range(n_available_segments):
            idx_start = k * n_seg_cov
            stft_block = stacked_stfts[
                :, idx_start : idx_start + n_seg_cov, :
            ]  # View-based slicing
            stft_block_conj = np.conj(stft_block)  # Precompute conjugate

            # Derive the number of non-zero snapshots within the k-th block
            if mask_applied:
                n_mean = np.sum(mask_tt[idx_start : idx_start + n_seg_cov])

            csd_matrix[..., k] = (
                np.einsum("ftr,fts->frs", stft_block, stft_block_conj) / n_mean
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
    ) -> np.ndarray:
        """
        Compute the major eigenvector of CSDMs from a mutli-dimensional CSDM array.

        Parameters
        ----------
        csdm_5D : np.ndarray
            5D CSDM array, default shape is (num_frequency_bins, num_receivers, num_receivers, num_y, num_x),
            one can use a different input shape by specifying dims_order.
        dims_order : dict
            Dictionary specifying the order of the dimensions in the input array.
        Returns
        -------
        major_eigve : np.ndarray
            4D array containing the major eigenvectors at all x, y positions and all frequencies (num_receivers x num_frequency_bins x num_y x num_x).
        """

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


if __name__ == "__main__":
    # Quick test on signal
    import scipy.signal as sp
    import matplotlib.pyplot as plt
    from source.signal_generator import SignalGenerator

    sg = SignalGenerator()
    cm = CovManager()

    duration = 60
    f0 = 15
    f1 = 20
    fs = 1000

    nperseg = 2**7
    noverlap = int(nperseg * 0.75)

    s0, t = sg.lfm_chirp_train(
        f0=f0,
        f1=f1,
        fs=fs,
        T_chirp=1,
        T=duration,
        interpulse_delay=2,
        start_delay=1,
    )
    # sg.plot_signal(t, s)

    ff, tt, stft_array = stft(
        s0, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    # sg.plot_spectrogram(t=tt, f=ff, S_tf=stft_array)
    # plt.show()

    # # Generate signal
    # s0, t = sg.pulse_train(T=duration, f=f0, fs=fs, interpulse_delay=5)
    s0 *= 2
    # # sg.plot_signal(t, s0)
    # Generate noise
    t, v0 = sg.colored_noise(T=duration, fs=fs, noise_color="white")
    # v0 /= np.max(np.abs(v0))
    v0 /= np.var(v0)

    n_rcv = 3
    roll_shift = 200
    shift = roll_shift
    x = s0 + v0
    x = x[np.newaxis, ...]

    v = v0
    v = v[np.newaxis, ...]

    s = s0
    s = s[np.newaxis, ...]

    # Generate a signal with multiple receivers
    for ircv in range(n_rcv - 1):
        si = np.roll(s0, shift=shift)

        # Generate noise
        t, vi = sg.colored_noise(T=duration, fs=fs, noise_color="white")
        # vi /= np.max(np.abs(vi))
        vi /= np.var(vi)
        # sg.plot_signal(t, v)

        # Signal plus noise
        xi = si + vi
        x = np.concatenate((x, xi[np.newaxis, ...]), axis=0)
        v = np.concatenate((v, vi[np.newaxis, ...]), axis=0)
        s = np.concatenate((s, si[np.newaxis, ...]), axis=0)
        shift += roll_shift

    # Transpose to required shape
    x = x.T
    v = v.T
    s = s.T
    # Stft
    nperseg = 2**8
    noverlap = int(nperseg * 0.75)
    ff, tt, stft_x = cm.get_stft_array(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, stft_v = cm.get_stft_array(v, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, stft_s = cm.get_stft_array(s, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # sg.plot_spectrogram(t=tt, f=ff, S_tf=stft_x[0, ...])
    # sg.plot_spectrogram(t=tt, f=ff, S_tf=stft_x[1, ...])

    # Define a mask to apply on the signal
    threshold = 1.5
    mask_tt = np.zeros_like(tt, dtype=int)
    for ircv in range(n_rcv):
        energy = np.sum(np.abs(stft_x[ircv, ...]) ** 2, axis=0)
        mask_tt_i = energy > threshold
        mask_tt = np.logical_or(mask_tt, mask_tt_i)

    # mask_tt = None
    sg.plot_signal(tt, energy)

    # Plot stfts
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    for ircv in range(n_rcv):
        sg.plot_spectrogram(t=tt, f=ff, S_tf=stft_x[ircv, ...] * mask_tt, ax=axs[ircv])
        axs[ircv].set_title(f"Rcv n°{ircv}")
        axs[ircv].set_ylim([0, 100])
    plt.suptitle("X")
    # plt.show()

    # Derive CSDM of the noisy signal
    # mask_tt = None
    csdm_x = cm.compute_csdm_fast(stft_x, mask_tt=mask_tt)
    mean_csdm_x = np.mean(csdm_x, axis=0)

    # Derive CSDM of the noise
    csdm_vx = cm.compute_csdm_fast(stft_x, mask_tt=~mask_tt)
    mean_csdm_vx = np.mean(csdm_vx, axis=0)

    # Derive CSDM of the noise
    csdm_v = cm.compute_csdm_fast(stft_v)
    mean_csdm_v = np.mean(csdm_v, axis=0)

    # Derive CSDM of the noise
    csdm_s = cm.compute_csdm_fast(stft_s)
    mean_csdm_s = np.mean(csdm_s, axis=0)

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    # axs[0].imshow(np.abs(mean_csdm_x) / np.max(np.abs(mean_csdm_x)), aspect="auto")
    # axs[1].imshow(np.abs(mean_csdm_s) / np.max(np.abs(mean_csdm_s)), aspect="auto")
    # imv = axs[2].imshow(
    #     np.abs(mean_csdm_v) / np.max(np.abs(mean_csdm_v)), aspect="auto"
    # )

    # plt.colorbar(imv)
    # plt.title("CSDM")
    # plt.xlabel("Receiver index")
    # plt.ylabel("Receiver index")

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    fmid = f0 + (f1 - f0) / 2
    idx_fmid = np.argmin(np.abs(ff - fmid))
    # imx = axs[0].imshow(
    #     np.abs(csdm_x[idx_fmid]) / np.max(np.abs(csdm_x[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )
    # plt.colorbar(imx)

    # imv = axs[1].imshow(
    #     np.abs(csdm_v[idx_fmid]) / np.max(np.abs(csdm_v[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )
    # plt.colorbar(imv)

    # ims = axs[2].imshow(
    #     np.abs(csdm_s[idx_fmid]) / np.max(np.abs(csdm_s[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )

    # plt.colorbar(ims)
    # plt.suptitle("CSDM")
    # plt.xlabel("Receiver index")
    # plt.ylabel("Receiver index")

    # Compare estimated CSDMs
    # Signal only
    csdm_s = cm.compute_csdm_fast(stft_s, mask_tt=mask_tt)

    # Noise only
    csdm_v = cm.compute_csdm_fast(stft_v, mask_tt=None)

    # Signal plus noise
    csdm_x = cm.compute_csdm_fast(stft_x, mask_tt=mask_tt)

    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # imx = axs[0].imshow(
    #     np.abs(csdm_x[idx_fmid]) / np.max(np.abs(csdm_x[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )
    # plt.colorbar(imx)
    # imv = axs[1].imshow(
    #     np.abs(csdm_v[idx_fmid]) / np.max(np.abs(csdm_v[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )
    # plt.colorbar(imv)
    # ims = axs[2].imshow(
    #     np.abs(csdm_s[idx_fmid]) / np.max(np.abs(csdm_s[idx_fmid])),
    #     aspect="auto",
    #     vmin=0,
    #     vmax=0.1,
    # )
    # plt.colorbar(ims)
    # plt.suptitle("CSDM")
    # plt.xlabel("Receiver index")
    # plt.ylabel("Receiver index")

    # Compare csdm_s versus estimated csdm_s (cov subtraction)
    csdm_hat = csdm_x - csdm_v
    # Normalize to 1
    # csdm_s = csdm_s / np.max(np.abs(csdm_s))
    # csdm_hat = csdm_hat / np.max(np.abs(csdm_hat))
    # Comput diff
    csdm_delta = csdm_s - csdm_hat

    fig, axs = plt.subplots(nrows=1, ncols=3)
    ims = axs[0].imshow(
        np.abs(csdm_s[idx_fmid]),
        aspect="auto",
        # vmin=0,
        # vmax=0.1,
        cmap="jet",
    )
    axs[0].set_title("$R_s$")
    plt.colorbar(ims)

    ims_hat = axs[1].imshow(
        np.abs(csdm_hat[idx_fmid]),
        aspect="auto",
        # vmin=0,
        # vmax=0.1,
        cmap="jet",
    )
    axs[1].set_title("$R_x - R_v$")
    plt.colorbar(ims_hat)

    imd = axs[2].imshow(
        np.abs(csdm_delta[idx_fmid]) / np.abs(csdm_s[idx_fmid]) * 100,
        aspect="auto",
        vmin=0,
        vmax=100,
        cmap="jet",
    )
    axs[2].set_title(r"$(R_s - (R_x - R_v)) / R_s \times 100$")
    plt.colorbar(imd)
    plt.suptitle(f"CSDM at f = {fmid} Hz")
    plt.xlabel("Receiver index")
    plt.ylabel("Receiver index")

    plt.show()

    # print()
