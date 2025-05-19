#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   window_props_study.py
@Time    :   2025/05/19 08:18:24
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
import matplotlib.pyplot as plt

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import (
    LocalizationProcessor,
)
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
    DeepWaterRealEnv,
)
from publication.publication_figure import PubFigure

if __name__ == "__main__":

    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    # antenna.plot_antenna()
    # plt.savefig("antenna")

    debug = False
    check = True
    n_mc = 20
    use_weighted_rtf = True
    name = "dw_real_env"

    search_area_length = 0.7 * 1e3
    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    ### Test different window sizes ###
    import shutil

    # mode = "analysis"
    mode = "run"
    snrs = [-8, -6, -4, -2, 0, 2]

    # real_env_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\uace_testcase\data\dw_real_env"
    real_env_root = simu.data_folder

    # shutil.copyfile(
    #     os.path.join(real_env_root + "_impulsive_response", "tf.nc"),
    #     os.path.join(real_env_root, "tf.nc"),
    # )
    # db = DataBuilder(simulation=simu)
    # db.grid_dataset()
    # db.build_signal()

    ideal_nperseg = 7 * 200
    npersegs = [ideal_nperseg, 2**11, 2**10, 2**9]
    alpha_ov = [0.5, 0.75, 0.9]

    if mode == "run":

        for nperseg in npersegs:
            for alpha in alpha_ov:
                name = f"dw_real_env_nperseg{nperseg}_aov{alpha}"
                simu = Simulation(
                    name=name,
                    debug=debug,
                    antenna=antenna,
                    check_features=check,
                    feature_nperseg=nperseg,
                    feature_overlap_ratio=alpha,
                    monte_carlo_iterations=n_mc,
                    use_weighted_rtf=use_weighted_rtf,
                    search_area_length=search_area_length,
                )
                test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)
                lp = LocalizationProcessor(simulation=simu, use_dask=False)

                # Copy dataset from real_env to avoid rerunning DataBuilder
                # shutil.copyfile(os.path.join(real_env_root, "tf.nc"), simu.tf_dataset_fpath)

                shutil.copyfile(
                    os.path.join(real_env_root, "tf_grid_dx20m_dy20m.nc"),
                    simu.tf_grid_dataset_fpath,
                )

                shutil.copyfile(
                    os.path.join(real_env_root, "library_dx20m_dy20m.nc"),
                    simu.library_dataset_fpath,
                )

                lp.process_multiple_snrs(snrs=snrs, run_mode="a")
        mode = "analysis"

    msr_ = [[] for i in range(len(npersegs))]
    rmse_ = [[] for i in range(len(npersegs))]
    if mode == "analysis":
        for i, nperseg in enumerate(npersegs):
            msr_.append([])
            for j, alpha in enumerate(alpha_ov):
                name = f"dw_real_env_nperseg{nperseg}_aov{alpha}"
                simu = Simulation(
                    name=name,
                    debug=debug,
                    antenna=antenna,
                    check_features=check,
                    feature_nperseg=nperseg,
                    feature_overlap_ratio=alpha,
                    monte_carlo_iterations=n_mc,
                    use_weighted_rtf=use_weighted_rtf,
                    search_area_length=search_area_length,
                )
                test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)
                lp = LocalizationProcessor(simulation=simu, use_dask=False)

                # Load metric for the current simu
                # Build subarray list
                subarrays_list = np.atleast_2d(simu.antenna.rcv_idx)  # Fullarray
                # Load metrics
                msr, dr, rmse = lp.load_msr_rmse_res_subarrays(subarrays_list)
                # Store metrics
                fa_key = list(msr.keys())[0]
                msr = msr[fa_key]
                rmse = rmse[fa_key]
                msr_[i].append(msr)
                rmse_[i].append(rmse)

        pfig = PubFigure(legend_fontsize=10)
        for i, nperseg in enumerate(npersegs):
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            # RMSE
            ax_rmse = axs[0]
            # MSR
            ax_msr = axs[1]
            for j, alpha in enumerate(alpha_ov):
                test_id = (
                    r"$n_{perseg} = "
                    + str(nperseg)
                    + r", \, \alpha_{overlap} = "
                    + str(alpha)
                    + r"$"
                )
                msr = msr_[i][j]
                rmse = rmse_[i][j]
                ax_msr.errorbar(
                    msr.index,
                    msr.rtf_mean,
                    yerr=msr.rtf_std,
                    fmt="o-",
                    label=f"RTF - {test_id}",
                )
                ax_msr.legend()

                # Plot rmse
                ax_rmse.plot(rmse.index, rmse["rtf"], "o-", label=f"RTF - {test_id}")

            # Save figures for each nperseg
            # RMSE
            ax_rmse.legend()
            ax_rmse.set_xlabel("SNR [dB]")
            ax_rmse.set_ylabel("RMSE [m]")
            ax_rmse.grid()

            # MSR
            ax_msr.set_xlabel("SNR [dB]")
            ax_msr.set_ylabel("MSR [dB]")
            ax_msr.legend()
            ax_msr.grid()

            rcv_ids = [f"{id[0]}_{id[1]}" for id in fa_key.split("_")]
            rcv_str = "$" + ", \,".join(rcv_ids) + "$"
            plt.suptitle(f"Receivers = ({rcv_str})")

            img_folder = os.path.join(simu.root_img, "window_params_comparison")
            fpath = os.path.join(img_folder, f"nperseg_{nperseg}")
            plt.savefig(fpath)

    for j, alpha in enumerate(alpha_ov):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        # RMSE
        ax_rmse = axs[0]
        # MSR
        ax_msr = axs[1]
        for i, nperseg in enumerate(npersegs):
            test_id = (
                r"$n_{perseg} = "
                + str(nperseg)
                + r", \, \alpha_{overlap} = "
                + str(alpha)
                + r"$"
            )
            msr = msr_[i][j]
            rmse = rmse_[i][j]

            ax_msr.errorbar(
                msr.index,
                msr.rtf_mean,
                yerr=msr.rtf_std,
                fmt="o-",
                label=f"RTF - {test_id}",
            )
            ax_msr.legend()

            # Plot rmse
            ax_rmse.plot(rmse.index, rmse["rtf"], "o-", label=f"RTF - {test_id}")

        # Save figures for each nperseg
        # RMSE
        ax_rmse.legend()
        ax_rmse.set_xlabel("SNR [dB]")
        ax_rmse.set_ylabel("RMSE [m]")
        ax_rmse.grid()

        # MSR
        ax_msr.set_xlabel("SNR [dB]")
        ax_msr.set_ylabel("MSR [dB]")
        ax_msr.legend()
        ax_msr.grid()

        rcv_ids = [f"{id[0]}_{id[1]}" for id in fa_key.split("_")]
        rcv_str = "$" + ", \,".join(rcv_ids) + "$"
        plt.suptitle(f"Receivers = ({rcv_str})")

        img_folder = os.path.join(simu.root_img, "window_params_comparison")
        fpath = os.path.join(img_folder, f"alpha_ov_{alpha}.png")
        plt.savefig(fpath)
