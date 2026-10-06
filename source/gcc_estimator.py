#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   gcc_estimator.py
@Time    :   2025/04/07 13:36:38
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.signal as sp


class GCCEstimator:
    """Class to estimate GCC."""

    def __init__(
        self,
        fs: float,
        idx_rcv_ref: int = 0,
        nperseg: int = 2**12,
        noverlap: int = 2**11,
        window: str = "hann",
    ):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency.
        idx_rcv_ref : int, optional
            Index of the reference receiver.
        nperseg : int, optional
            Number of samples per segment used to derive CSDM.
        noverlap : int, optional
            Number of overlapping samples between consecutive segments used to derived CSDM.
        window : str, optional
            Window function used to derive CSDM.
        """
        self.fs = fs
        self.index_reference_rcv = idx_rcv_ref
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window

    def gcc_4D(self, x_4D, idx_rcv_refs):
        # Power spectral density of signals
        ff, sxx = sp.welch(
            x_4D,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window=self.window,
            axis=1,
        )  # (nrcv, nf, ny, nx)

        iref = 0

        gcc = np.empty((sxx.shape[0],) + sxx.shape, dtype=complex)
        for iref in idx_rcv_refs:
            # Cross power spectral density of signals between the reference receiver and other receivers
            x_4D_ref = x_4D[
                iref : iref + 1, ...
            ]  # Keep first dimension to allow broadcasting
            ff, Sxy_library = sp.csd(
                x_4D_ref,
                x_4D,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                window=self.window,
                axis=1,
            )  # (nrcv, nf, ny, nx)

            # Compute GCC-SCOT weights
            sxx_ref = sxx[
                iref : iref + 1, ...
            ]  # Keep first dimension to allow broadcasting
            w = 1 / np.abs(np.sqrt(sxx_ref * sxx))  # (nrcv, nf, ny, nx)

            # Apply GCC-SCOT
            gcc[iref, ...] = w * Sxy_library

        return ff, gcc
