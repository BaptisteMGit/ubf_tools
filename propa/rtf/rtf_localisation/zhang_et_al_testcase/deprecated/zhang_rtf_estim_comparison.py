#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_rtf_estim_comparison.py
@Time    :   2025/03/27 09:57:56
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
import xarray as xr
import matplotlib.pyplot as plt


from matplotlib import rcParams

# rcParams["backend"] = "tkagg"
from propa.rtf.rtf_utils import (
    D_hermitian_angle_fast,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.deprecated.zhang_params import (
    ROOT_DATA,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.deprecated.zhang_build_datasets import (
    build_features_from_time_signal,
    build_signal,
)
import propa.rtf.rtf_localisation.zhang_et_al_testcase.src.params as p


# print(rcParams["backend"])

# print(rcParams["backend"])


# ======================================================================================================================
# Functions
# ======================================================================================================================
def run_simu(
    snrs, estimators, debug=True, check=True, antenna_type="zhang", verbose=True
):

    for estimator in estimators:
        root_name = f"zhang_features_{estimator}"
        for snr in snrs:

            # Build features for given snr and estimator
            build_features_from_time_signal(
                snr_dB=snr,
                debug=debug,
                check=check,
                use_welch_estimator=True,
                antenna_type=antenna_type,
                rtf_estimator=estimator,
                root_name=root_name,
                verbose=verbose,
            )


def evaluate_performances(snrs, estimators):

    dx = dy = 20

    # Load true RTF dataset
    fname = f"tf_zhang_grid_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_tf = xr.open_dataset(fpath)
    # Build complex tf
    tf = ds_tf.tf_real + 1j * ds_tf.tf_imag
    # Define reference receiver to use
    i_rcv_ref = 0

    # Define distance to use
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 0,
        "unit": "deg",
        "apply_mean": True,
        "ax_f": 1,
    }

    res = {}
    res["snr"] = snrs

    first_iter_flag = True
    for estimator in estimators:
        root_name = f"zhang_features_{estimator}"
        mean = []
        median = []
        std = []
        for snr in snrs:
            fpath = os.path.join(
                ROOT_DATA, f"{root_name}_dx{dx}m_dy{dy}m_snr{snr:.1f}dB.nc"
            )
            ds = xr.open_dataset(fpath)
            # Select reference receiver
            ds = ds.sel(idx_rcv_ref=i_rcv_ref)
            rtf_hat = ds.rtf_real + 1j * ds.rtf_imag

            if first_iter_flag:
                # Extract tf between fmin and fmax from ds_rtf_cs
                tf = tf.sel(f=slice(ds.f_rtf.min(), ds.f_rtf.max()))

                # Define tf_ref
                tf_ref = tf.sel(idx_rcv=i_rcv_ref)

                # Build "true" RTF
                rtf = tf / tf_ref

                # Select common frequencies
                rtf = rtf.sel(f=ds.f_rtf, method="nearest")

            # Compute hermitian angle
            dist_grid = dist_func(rtf.values, rtf_hat.values, **dist_kwargs)

            # Derive median, mean and std over entire grid
            mean_dist = np.mean(dist_grid)
            median_dist = np.median(dist_grid)
            std_dist = np.std(dist_grid)

            mean.append(mean_dist)
            median.append(median_dist)
            std.append(std_dist)

            print(f"Estimator: {estimator}, SNR = {snr}dB")
            print(f"Mean, Median, Std : {mean_dist, median_dist, std_dist}")

        res[estimator] = {"mean": mean, "median": median, "std": std}

    return res


def plot_performances(snrs, estimators):

    perf = evaluate_performances(snrs=snrs, estimators=estimators)

    plt.figure()

    for estimator in estimators:
        plt.errorbar(
            snrs, perf[estimator]["mean"], yerr=perf[estimator]["std"], label=estimator
        )

    plt.xlabel("SNR [dB]")
    plt.ylabel(r"$\theta$" + " [°]")
    plt.legend()
    plt.savefig("test")


if __name__ == "__main__":

    snrs = [40]
    estimators = ["cs", "cw"]

    # build_signal(debug=True, antenna_type="zhang", event_stype="wn")

    # snrs = [-15, -10, -5, 0, 10, 15, 20, 25, 30, 35, 40]
    snrs = [-15, 0, 15]
    run_simu(snrs=snrs, estimators=estimators)
    # evaluate_performances(snrs=snrs, estimators=estimators)
    plot_performances(snrs=snrs, estimators=estimators)
