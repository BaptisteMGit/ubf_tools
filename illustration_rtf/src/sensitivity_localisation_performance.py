#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/08/26 14:00:58
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import glob
import shutil
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from misc import progression_bar
from illustration_rtf.src.sensitivity import *
from source.global_constants import ROOT_PROGRAM, ROOT_DATA
from publication.publication_figure import PubFigure, color

from illustration_rtf.src.sensitivity import (
    _celerity_profile_rmse,
    _celerity_profile_f1_score,
    _celerity_profile_std,
    _select_r0_pair,
)

# NOTE :
# THIS CODE IS INTENDED TO BE RUN ON THE PLATEFORM ONLY, LOADING PREVIOUSLY COMPUTED DATASET OF GREEN FUNCTION FOR A SET
# OF CELERITY PROFILES

if os.name == "nt":  # Windows
    raise ValueError("CODE IS INTENDED TO BE RUN ON THE PLATEFORM !")

else:  # Linux
    # Input data
    ROOT_RTF_RESILIENCE_DATA_DIR = os.path.join(ROOT_DATA, "RTF_RESILIENCE")
    IMG_DIR = os.path.join(ROOT_PROGRAM, "illustration_rtf", "img")
    OUTPUT_DATA_DIR = os.path.join(ROOT_PROGRAM, "illustration_rtf", "data")


def dist_from_baseline_loc_performance(ds_baseline, ds_test, d12, r0):
    """Compute the RTF distance (L1/L2/theta/wasserstein) between the
    baseline's gamma at all r and each swept value's gamma at r0.

    """
    # Baseline gamma (along r axis) -- already precomputed for every r at
    # build_baseline()
    gamma_baseline = ds_baseline.gamma  # (nf, nr)
    gamma_a = gamma_baseline.values[:, :, np.newaxis]  # (nf, nr, 1)

    # 'ds_test.gf' only has r=[r0, r0+d12] left (2 points)
    g_fr_1 = ds_test.gf.sel(r=r0, method="nearest")
    g_fr_2 = ds_test.gf.sel(r=r0 + d12, method="nearest")
    gamma_test_r0 = 20 * np.log10(np.abs(g_fr_2.values / g_fr_1.values))
    gamma_b = gamma_test_r0.T[:, np.newaxis, :]  # (nf, 1, nprofile)

    # Compute distance
    dist_L1 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1")
    dist_L2 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2")
    dist_theta = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta")

    # NOTE : wasserstein implementation does not support vectorisation
    # It could be by-passed by ugly loop over position if required
    # dist_wasserstein = calc_gamma_dist(
    #     gamma_a=gamma_a, gamma_b=gamma_b, dist_type="wasserstein"
    # )
    dist_wasserstein = None

    return dist_L1, dist_L2, dist_theta, dist_wasserstein


def dist_from_baseline_loc_performance_baseline(ds_baseline, r0):
    """Compute the RTF distance (L1/L2/theta/wasserstein) between the
    baseline's gamma at all r and baseline's gamma at r0

    """
    # Baseline gamma (along r axis) -- already precomputed for every r at
    # build_baseline()
    gamma_baseline = ds_baseline.gamma  # (nf, nr)
    gamma_a = gamma_baseline.values  # (nf, nr)
    gamma_b = gamma_baseline.sel(r=r0, method="nearest").values[
        :, np.newaxis
    ]  # (nf, 1)

    # Compute distance
    dist_L1 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L1")
    dist_L2 = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="L2")
    dist_theta = calc_gamma_dist(gamma_a=gamma_a, gamma_b=gamma_b, dist_type="theta")

    # NOTE : wasserstein implementation does not support vectorisation
    # It could be by-passed by ugly loop over position if required
    # dist_wasserstein = calc_gamma_dist(
    #     gamma_a=gamma_a, gamma_b=gamma_b, dist_type="wasserstein"
    # )
    dist_wasserstein = None

    return dist_L1, dist_L2, dist_theta, dist_wasserstein


def save_sensitivity_distance_results(
    test_arg_name,
    test_values,
    dist_L1,
    dist_L2,
    dist_theta,
    dist_wasserstein=None,
    result_dir=RESULT_DIR,
    file_prefix="dist_",
):
    """Save the RTF distance-from-baseline results (L1/L2/theta[/
    wasserstein]) for ONE swept parameter to a small, dedicated CSV
    file -- small enough (a handful of floats per swept value) that a
    plain text format is the simplest, most portable choice; no need
    for NetCDF/xarray here, unlike the much larger raw Green's
    function datasets this is derived from (see
    build_sensitivity_dataset()).

    Args:
        test_arg_name (str): the swept parameter's name.
        test_values, dist_L1, dist_L2, dist_theta (array-like): equal-
            length 1D arrays, as returned by dist_from_baseline() (plus
            the parameter values themselves) -- or, for
            process_sensitivity_mainlobe_width(), the mainlobe WIDTHS
            (see single_sensitivity_test_calc_dist_width()) rather than
            raw distances.
        dist_wasserstein (array-like|None): the Wasserstein distance
            (see dist_from_baseline()'s own docstring), same length as
            the others. None (the default): omitted from the saved
            file entirely -- callers that only ever have L1/L2/theta
            (e.g. process_sensitivity_mainlobe_width()/
            process_sensitivity_intrinsic_mainlobe_width(), whose
            "distance AS A FUNCTION OF r" this hasn't been extended to
            -- see calc_gamma_dist()'s own NOTE) keep writing the
            original 4-column ("<test_arg_name>,L1,L2,theta") file
            unchanged; only dist_from_baseline()'s own callers
            (process_sensitivity()/process_celerity_sensitivity())
            pass it.
        result_dir (str): directory to write into, as
            '<result_dir>/<file_prefix><test_arg_name>.csv'.
        file_prefix (str): distinguishes what these metrics actually
            are when several kinds of results are saved side by side
            in the same 'result_dir' -- e.g. "dist_" for the raw RTF
            distance at r0 (process_sensitivity()) vs
            "mainlobe_width_" for the -3dB mainlobe width
            (process_sensitivity_mainlobe_width()). Using the same
            prefix for two DIFFERENT quantities would silently
            overwrite one with the other.

    Returns:
        str: path to the written file.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{file_prefix}{test_arg_name}.csv")
    if dist_wasserstein is None:
        data = np.column_stack([test_values, dist_L1, dist_L2, dist_theta])
        header = f"{test_arg_name},L1,L2,theta"
    else:
        data = np.column_stack(
            [test_values, dist_L1, dist_L2, dist_theta, dist_wasserstein]
        )
        header = f"{test_arg_name},L1,L2,theta,wasserstein"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


def get_msr(ds_single_profile, dist="L1"):
    ds_res_p = ds_single_profile
    d_norm = (ds_res_p[f"dist_{dist}"] - ds_res_p[f"dist_{dist}"].min()) / (
        ds_res_p[f"dist_{dist}"].max() - ds_res_p[f"dist_{dist}"].min()
    )
    # Get main lobe domain
    try:
        _, r_d_neg_sup_3dB, r_d_pos_sup_3dB = calc_mainlobe_width_3dB(
            d_norm, ds_res_p.r, ds_res_p[f"r_hat_{dist}"]
        )
    except:
        return np.nan

    # Exclude main lobe
    idx_main_lobe = np.logical_and(
        ds_res_p.r >= ds_res_p[f"r_hat_{dist}"] + r_d_neg_sup_3dB,
        ds_res_p.r <= ds_res_p[f"r_hat_{dist}"] + r_d_pos_sup_3dB,
    )
    r_outside_main_lobe = ds_res_p.r.values[~idx_main_lobe]
    d_norm_outside_mainlobe = d_norm.sel(r=r_outside_main_lobe)
    first_side_lobe_level = d_norm_outside_mainlobe.min()
    msr = 10 * np.log10(
        1 - first_side_lobe_level
    )  # Distance is normalized to 1 -> derive constrast in dB

    return msr


def process_celerity_sensitivity_loc_performance(
    env_type, result_dir, situations, save_dir=None
):
    """
    TODO complete
    """
    # NOTE: env_type must be a valid key -- fail fast with a clear
    # KeyError (matching build_celerity_baseline()'s own behaviour)
    # rather than a confusing FileNotFoundError further down.
    CELERITY_ENV_TYPES[env_type]
    env_config = CELERITY_ENV_TYPES[env_type]

    import functools

    baseline_src_rcv_param = baseline_src_rcv()
    d12 = baseline_src_rcv_param["d12"]
    r0 = baseline_src_rcv_param["r0"]
    preprocess = functools.partial(_select_r0_pair, r0=r0, d12=d12)

    for situation in situations:
        situation_dir = os.path.join(result_dir, situation)
        value_files = sorted(
            glob.glob(os.path.join(situation_dir, f"{situation}_*.nc"))
        )
        if not value_files:
            continue

        # NOTE: THIS situation's own baseline (see
        # build_celerity_baseline()'s own NOTE for why -- "winter" is
        # compared against winter's own mean profile, not "all"'s).
        fpath_baseline = os.path.join(situation_dir, "gf_dataset_baseline.nc")

        with xr.open_dataset(fpath_baseline) as ds_baseline:
            z_baseline, c_p_baseline = (
                ds_baseline.z_ssp.values,
                ds_baseline.c_p_ssp.values,
            )

            # Derive theta(r) for baseline
            baseline_dist_L1, baseline_dist_L2, baseline_dist_theta, _ = (
                dist_from_baseline_loc_performance_baseline(ds_baseline, r0)
            )

            # NOTE: concat_dim="profile" -- matches
            # build_celerity_sensitivity_dataset()'s own saved
            # dimension name (every swept "value" here is a profile,
            # not a scalar named after 'situation' the way
            # build_sensitivity_dataset()'s own 'test_arg_name' is).
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim="profile",
                join="override",
                preprocess=preprocess,
            ) as ds_test:

                # NOTE : wasserstein not handled yet
                dist_L1, dist_L2, dist_theta, _ = dist_from_baseline_loc_performance(
                    ds_baseline, ds_test, d12, r0
                )

                # RMSE
                rmse = _celerity_profile_rmse(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                )
                # F1 score
                f1_score = _celerity_profile_f1_score(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                    kernel_size=5,
                )
                # Std
                std = _celerity_profile_std(
                    z_baseline,
                    c_p_baseline,
                    ds_test["z"].values,
                    ds_test["c_p_ssp"].values,
                )

        ### Analyse results ###
        # Build result dataset
        ds_res = xr.Dataset(
            data_vars=dict(
                dist_L1=(["r", "profile"], dist_L1),
                dist_L2=(["r", "profile"], dist_L2),
                dist_theta=(["r", "profile"], dist_theta),
                # dist_wasserstein=(["r", "profile"], dist_wasserstein),        # NOTE : not handled yet
                baseline_dist_L1=(["r"], baseline_dist_L1),
                baseline_dist_L2=(["r"], baseline_dist_L2),
                baseline_dist_theta=(["r"], baseline_dist_theta),
                ssp_rmse=(["profile"], rmse),
                ssp_std=(["profile"], std),
                ssp_f1=(["profile"], f1_score),
            ),
            coords=dict(
                r=ds_baseline.r.values,
                profile=ds_test.profile.values,
            ),
        )
        # Copy attributes
        ds_res.attrs = ds_baseline.attrs
        # Restrict range
        ds_res = ds_res.sel(r=slice(ds_baseline.r_rcv.min(), ds_baseline.r_rcv.max()))

        # Perf indicator 1: estimated position
        idx_r_hat_L1 = ds_res.dist_L1.argmin(dim="r")
        r_hat_L1 = ds_res.r.values[idx_r_hat_L1]
        idx_r_hat_L2 = ds_res.dist_L2.argmin(dim="r")
        r_hat_L2 = ds_res.r.values[idx_r_hat_L2]
        idx_r_hat_theta = ds_res.dist_theta.argmin(dim="r")
        r_hat_theta = ds_res.r.values[idx_r_hat_theta]
        # Add to dataset
        ds_res["r_hat_L1"] = (["profile"], r_hat_L1)
        ds_res["r_hat_L2"] = (["profile"], r_hat_L2)
        ds_res["r_hat_theta"] = (["profile"], r_hat_theta)

        # Perf indicator 2: MSR (Main lobe to Side lobe ratio)
        msr_L1 = []
        msr_L2 = []
        msr_theta = []

        # NOTE it could be vectorized
        for prof in ds_res.profile.values:
            ds_res_p = ds_res.sel(profile=prof)

            msr_L1.append(get_msr(ds_single_profile=ds_res_p, dist="L1"))
            msr_L2.append(get_msr(ds_single_profile=ds_res_p, dist="L2"))
            msr_theta.append(get_msr(ds_single_profile=ds_res_p, dist="theta"))

        # Add to dataset
        ds_res["msr_L1"] = (["profile"], msr_L1)
        ds_res["msr_L2"] = (["profile"], msr_L2)
        ds_res["msr_theta"] = (["profile"], msr_theta)

        # Save results
        output_fname = f"resilience_loc_performance__{env_type}_{situation}.nc"
        output_fpath = os.path.join(OUTPUT_DATA_DIR, output_fname)
        ds_res.to_netcdf(output_fpath)

    return


def generate_celerity_diag_loc_performance(
    distance=["theta"],
    celerity_env_types=None,
    celerity_situations=None,
    process_sensi=True,
    build_baseline=False,
):
    """
    Generate celerity diagnostic in term of localisation performance.

    # Parameters
    # ----------
    # distance : list, optional
    #     _description_, by default ["theta"]
    # celerity_env_types : _type_, optional
    #     _description_, by default None
    # celerity_situations : _type_, optional
    #     _description_, by default None
    # process_sensi : bool, optional
    #     _description_, by default True
    # build_baseline : bool, optional
    #     _description_, by default False
    """
    if build_baseline:
        build_celerity_baselines(
            env_types=celerity_env_types, situations=celerity_situations
        )

    for env_type in celerity_env_types:
        result_dir = os.path.join(
            RTF_CELERITY_RESILIENCE_DATA_DIR, env_type, "celerity"
        )

        if not os.path.isdir(result_dir) or not any(
            os.path.isdir(os.path.join(result_dir, name))
            for name in os.listdir(result_dir)
        ):
            print(
                f"generate_all_diagnostics: no celerity sweep data found for "
                f"env_type={env_type!r} yet (see build_celerity_sensitivity_dataset()) "
                f"-- skipping its diagnostics."
            )
            continue

        # 4.1) read all profile files and compute distance from baseline
        if process_sensi:
            process_celerity_sensitivity_loc_performance(
                env_type,
                result_dir=result_dir,
                situations=celerity_situations,
                save_dir=IMG_DIR,
            )
        # # 4.2) the key diagnostic: distance vs. RMSE deviation from the
        # # baseline profile (see plot_celerity_distance_vs_rmse()'s own
        # # docstring for why this, rather than the raw profile index).
        # # RMSE
        # plot_celerity_distance_vs_rmse(
        #     env_type,
        #     situations=celerity_situations,
        #     metric=distance[0],
        #     result_dir=result_dir,
        #     save_dir=img_dir,
        # )
        # # F1 score
        # plot_celerity_distance_vs_f1_score(
        #     env_type,
        #     situations=celerity_situations,
        #     metric=distance[0],
        #     result_dir=result_dir,
        #     save_dir=img_dir,
        # )
        # # STD
        # plot_celerity_distance_vs_std(
        #     env_type,
        #     situations=celerity_situations,
        #     metric=distance[0],
        #     result_dir=result_dir,
        #     save_dir=img_dir,
        # )

        # # 4.3) gamma AND the profiles themselves, for the smallest-/
        # # largest-distance configurations.
        # plot_extremal_celerity_configs(
        #     env_type,
        #     situations=celerity_situations,
        #     metric=distance[0],
        #     result_dir=result_dir,
        #     save_dir=img_dir,
        # )
        # plt.close("all")


if __name__ == "__main__":

    run_name = "resilience_12102026"
    RTF_CELERITY_RESILIENCE_DATA_DIR = os.path.join(
        ROOT_RTF_RESILIENCE_DATA_DIR, run_name
    )

    env_type = "dw"
    situation = "winter"
    generate_celerity_diag_loc_performance(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=[situation],
        process_sensi=True,
        build_baseline=False,
    )
    env_type = "dw"
    situation = "summer"
    generate_celerity_diag_loc_performance(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=[situation],
        process_sensi=True,
        build_baseline=False,
    )

    env_type = "sw"
    situation = "winter"
    generate_celerity_diag_loc_performance(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=[situation],
        process_sensi=True,
        build_baseline=False,
    )
    env_type = "sw"
    situation = "summer"
    generate_celerity_diag_loc_performance(
        distance=["theta"],
        celerity_env_types=[env_type],
        celerity_situations=[situation],
        process_sensi=True,
        build_baseline=False,
    )

    output_fname = f"resilience_loc_performance__{env_type}_{situation}.nc"
    output_fpath = os.path.join(OUTPUT_DATA_DIR, output_fname)
    ds_res = xr.open_dataset(output_fpath)

    # Compute pos rmse
    r0 = 15000
    rmse_pos_theta = np.sqrt(np.mean((ds_res.r_hat_theta - r0) ** 2))

    # plt.figure()
    # ds_res.r_hat_theta.plot.hist(bins=50)

    err_theta = ds_res.r_hat_theta - r0
    err_L2 = ds_res.r_hat_L2 - r0
    # plt.figure()
    # plt.scatter(ds_res.ssp_std, err_theta, marker=".")
    # plt.scatter(ds_res.ssp_std, err_L2, marker="x")

    # plt.figure()
    # plt.scatter(ds_res.ssp_std, -ds_res.msr_L1, label="L1")
    # plt.scatter(ds_res.ssp_std, -ds_res.msr_L2, label="L2")
    # plt.scatter(ds_res.ssp_std, -ds_res.msr_theta, label="theta")
    # plt.legend()
    # plt.scatter(ds_res.ssp_rmse, err_theta)
    # print()

    # Plot theta sorted by ascending ssp std and compare to baseline distance
    idx_sort = np.argsort(ds_res.ssp_std)
    fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(16, 8))
    # Sort by ascending ssp_std
    axs[0].pcolormesh(
        ds_res.r,
        ds_res.ssp_std.values[idx_sort],
        ds_res.dist_theta.values[:, idx_sort].T,
    )
    ds_res.baseline_dist_theta.plot(ax=axs[1])

    # Plot theta sorted by ascending ssp rmse and compare to baseline distance
    idx_sort = np.argsort(ds_res.ssp_rmse)
    fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1], figsize=(16, 8))
    # Sort by ascending ssp_std
    axs[0].pcolormesh(
        ds_res.r,
        ds_res.ssp_std.values[idx_sort],
        ds_res.dist_theta.values[:, idx_sort].T,
    )
    ds_res.baseline_dist_theta.plot(ax=axs[1])

    plt.show()
