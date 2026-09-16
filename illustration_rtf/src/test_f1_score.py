#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_f1_score.py
@Time    :   2026/09/14 10:21:32
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
from scipy.ndimage import convolve


def get_min_max_idx(arr: np.ndarray, axs: int = 1, pad: bool = True) -> np.ndarray:
    """Find local minima and maxima in array."""
    grad = np.diff(arr, axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.abs(np.sign(np.diff(grad_sign, axis=axs)))
    if pad:
        pad_shape = list(min_max.shape)
        pad_shape[axs] = 1
        min_max = np.concatenate(
            [np.zeros(pad_shape), min_max, np.zeros(pad_shape)], axis=axs
        )
    return min_max


def get_f1_score(
    min_max_idx_truth: np.ndarray,
    min_max_idx_ae: np.ndarray,
    axs: int = 1,
    kernel_size: int = 10,
) -> np.ndarray:
    """Compute F1 score for extremum detection."""
    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = kernel_size
    kernel = np.ones(kernel_shape)
    truth_expanded = convolve(min_max_idx_truth, kernel, mode="constant", cval=0.0)
    ae_expanded = convolve(min_max_idx_ae, kernel, mode="constant", cval=0.0)

    true_positives = (truth_expanded > 0) & (min_max_idx_ae > 0)
    num_true_positives = np.sum(true_positives, axis=axs)
    false_positives = (truth_expanded == 0) & (min_max_idx_ae > 0)
    num_false_positives = np.sum(false_positives, axis=axs)
    false_negatives = (min_max_idx_truth > 0) & (ae_expanded == 0)
    num_false_negatives = np.sum(false_negatives, axis=axs)

    precision_den = num_true_positives + num_false_positives
    recall_den = num_true_positives + num_false_negatives
    precision_score = np.where(
        precision_den == 0, 0, num_true_positives / precision_den
    )
    recall_score = np.where(recall_den == 0, 0, num_true_positives / recall_den)
    sum_scores = precision_score + recall_score
    f1_score = np.where(
        sum_scores == 0, 0, 2 * (precision_score * recall_score) / sum_scores
    )

    return f1_score


if __name__ == "__main__":
    import xarray as xr
    from illustration_rtf.src.sensitivity import (
        load_synthetic_celerity_profiles,
        load_mean_celerity_profile,
        _ssp_filename,
        _celerity_profile_rmse,
        _synthetic_ssp_filename,
    )

    env_type = "sw"
    situation = "summer"
    target_depth = 100
    z_baseline, c_p_baseline = load_mean_celerity_profile(
        _ssp_filename(env_type, situation), target_depth=target_depth
    )

    ssp_filename = _synthetic_ssp_filename(env_type, situation, n_samples=1000)
    z, c_p_all = load_synthetic_celerity_profiles(ssp_filename, target_depth)

    rmse = _celerity_profile_rmse(
        z_baseline,
        c_p_baseline,
        z,
        c_p_all,
    )

    min_max_idx_truth = get_min_max_idx(c_p_baseline[np.newaxis, :], axs=1, pad=False)
    min_max_idx_ae = get_min_max_idx(c_p_all, axs=1, pad=False)
    f1_score = get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=5)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.plot(c_p_all.T, z, label="Synthetic Samples", color="blue", alpha=0.1)
    plt.plot(c_p_baseline, z_baseline, label="Baseline", color="black", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Celerity (m/s)")
    plt.ylabel("Depth (m)")

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].plot(rmse, label="RMSE")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("RMSE")

    axs[1].plot(f1_score, label="F1 Score", color="orange")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("F1 Score")

    plt.figure()
    plt.plot(rmse, f1_score, "o")
    plt.xlabel("RMSE")
    plt.ylabel("F1 Score")
    # plt.savefig("test_f1_score.png")

    plt.figure()
    plt.plot(c_p_baseline, z_baseline, label="Baseline", color="black", linewidth=2)
    # Profile of max RMSE
    idx_max_rmse = np.argmax(rmse)
    plt.plot(
        c_p_all[idx_max_rmse, :],
        z,
        label="Max RMSE Sample",
        color="red",
        linestyle="--",
    )
    # Profile of max F1 score
    idx_max_f1 = np.argmax(f1_score)
    plt.plot(c_p_all[idx_max_f1, :], z, label="Max F1 Score Sample", color="red")
    # Profile of min RMSE
    idx_min_rmse = np.argmin(rmse)
    plt.plot(
        c_p_all[idx_min_rmse, :],
        z,
        label="Min RMSE Sample",
        color="green",
        linestyle="--",
    )
    # Profile of min F1 score
    idx_min_f1 = np.argmin(f1_score)
    plt.plot(c_p_all[idx_min_f1, :], z, label="Min F1 Score Sample", color="green")
    plt.gca().invert_yaxis()
    plt.legend()

    plt.show()
