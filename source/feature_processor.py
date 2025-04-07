#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   features_processor.py
@Time    :   2025/04/07 13:34:48
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to process localisation features.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from source.rtf_estimator import RTFEstimator
from source.gcc_estimator import GCCEstimator


class FeatureProcessor:
    """Class to process localisation features."""

    def __init__(
        self,
        fs: float,
        idx_rcv_ref: int = 0,
        nperseg: int = 2**12,
        noverlap: int = 2**11,
        window: str = "hann",
    ):
        """Init the class."""
        self.rtf_estimator = RTFEstimator(
            fs=fs,
            idx_rcv_ref=idx_rcv_ref,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        self.gcc_estimator = GCCEstimator(
            fs=fs,
            idx_rcv_ref=idx_rcv_ref,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )

    def dask_rtf_4D(self, x_4D, v_4D, idx_rcv_refs):
        """Wrapper function for rtf_4D with dask."""

        # x_4D is a (nrcv, nt, ny, nx) array
        dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
        _, rtf = self.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
            x_4D,
            v_4D,
            dims_order,
            idx_rcv_refs,
            return_csdm=False,
        )
        return rtf

    def dask_gcc_4D(self, x_4D, idx_rcv_refs):
        """Wrapper function for gcc_4D with dask."""
        _, gcc = self.gcc_estimator.gcc_4D(x_4D, idx_rcv_refs)
        return gcc
