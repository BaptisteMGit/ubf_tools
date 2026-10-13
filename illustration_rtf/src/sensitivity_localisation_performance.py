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

# NOTE :
# THIS CODE IS INTENDED TO BE RUN ON THE PLATEFORM ONLY, LOADING PREVIOUSLY COMPUTED DATASET OF GREEN FUNCTION FOR A SET
# OF CELERITY PROFILES

if os.name == "nt":  # Windows
    raise ValueError("CODE IS INTENDED TO BE RUN ON THE PLATEFORM !")

else:  # Linux
    # Input data
    ROOT_RTF_RESILIENCE_DATA_DIR = os.path.join(ROOT_DATA, "RTF_RESILIENCE")
    IMG_DIR = os.path.join(ROOT_PROGRAM, "illustration_rtf", "img")

run_name = "resilience_12102026"
RTF_CELERITY_RESILIENCE_DATA_DIR = os.path.join(ROOT_RTF_RESILIENCE_DATA_DIR, run_name)


def _select_r0_pair(ds, r0, d12):
    """xr.open_mfdataset's 'preprocess' callback (see
    process_sensitivity()): reduce a single per-value file's 'gf'
    variable down to just the 2 r-values the RTF at r0 actually needs
    (r0 and r0+d12), IMMEDIATELY as that one file is opened, before any
    multi-file concatenation.

    This is what actually fixes process_sensitivity()'s slowness/memory
    usage (see dist_from_baseline()'s docstring for the full
    explanation): 'preprocess' runs on each file while it is still
    small and separate, so the (f, r) array with r spanning the whole,
    possibly huge receiver-range grid -- built by
    build_sensitivity_dataset(), one such file per swept value -- gets
    cut down to (f, 2) before dask ever needs to combine it with any
    other file's data. Selecting r0/r0+d12 AFTER concatenation instead
    (as a plain '.sel()' on the combined dataset) would still require
    reading every file's FULL r range from disk first, since
    xr.open_mfdataset's own lazy dask arrays are chunked per-file --
    'preprocess' avoids that entirely by never letting the full arrays
    become part of the combined dataset's dask graph in the first place.
    """
    return ds.sel(r=[r0, r0 + d12], method="nearest")


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

        ds_baseline = xr.open_dataset(fpath_baseline)
        z_baseline, c_p_baseline = load_mean_celerity_profile(
            _ssp_filename(env_type, situation), target_depth=env_config["depth"]
        )

        with xr.open_dataset(fpath_baseline) as ds_baseline:
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
                dist_L1, dist_L2, dist_theta, dist_wasserstein = dist_from_baseline(
                    ds_baseline, ds_test, d12, r0
                )
                profile_idx = ds_test["profile"].values

                # NOTE: 'c_p_ssp'/'z' have no "r" dimension, so
                # 'preprocess' (_select_r0_pair(), which only ever
                # touches "r") leaves them fully intact here -- no
                # extra file read needed to also get the RMSE-from-
                # baseline-profile diagnostic (see
                # plot_celerity_distance_vs_rmse(), the main reason
                # this exists at all per user request).
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

        save_sensitivity_distance_results(
            situation,
            profile_idx,
            dist_L1,
            dist_L2,
            dist_theta,
            dist_wasserstein,
            result_dir=result_dir,
        )
        # RMSE
        _save_celerity_rmse_results(situation, profile_idx, rmse, result_dir)
        # F1 score
        _save_celerity_f1_score_results(situation, profile_idx, f1_score, result_dir)
        # STD
        _save_celerity_std_results(situation, profile_idx, std, result_dir)

    return plot_sensitivity_curves(situations, result_dir=result_dir, save_dir=save_dir)


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
        # 4.2) the key diagnostic: distance vs. RMSE deviation from the
        # baseline profile (see plot_celerity_distance_vs_rmse()'s own
        # docstring for why this, rather than the raw profile index).
        # RMSE
        plot_celerity_distance_vs_rmse(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        # F1 score
        plot_celerity_distance_vs_f1_score(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        # STD
        plot_celerity_distance_vs_std(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )

        # 4.3) gamma AND the profiles themselves, for the smallest-/
        # largest-distance configurations.
        plot_extremal_celerity_configs(
            env_type,
            situations=celerity_situations,
            metric=distance[0],
            result_dir=result_dir,
            save_dir=img_dir,
        )
        plt.close("all")


if __name__ == "__main__":

    # run_all_celerity(env_type="sw", n_profiles=1000, celerity_profile_source="real")
    # run_all_celerity(env_type="dw", n_profiles=1000, celerity_profile_source="real")

    generate_celerity_diag_loc_performance(
        distance=["theta"],
        celerity_env_types=["sw"],
        celerity_situations=["all"],
        process_sensi=True,
        build_baseline=False,
    )
