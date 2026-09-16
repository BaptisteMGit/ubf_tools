#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sensitivity_illustration.py
@Time    :   2026/09/14 14:12:48
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
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from illustration_rtf.src.sensitivity import (
    load_mean_celerity_profile,
    _ssp_filename,
    baseline_env,
    baseline_src_rcv,
    load_sensitivity_distance_results,
    _resilience_study_dirs,
    derive_gamma,
    CELERITY_ENV_TYPES,
    SSP_DATA_DIR,
    ARG_LABEL,
)
from propa.kraken_toolbox.plot_utils import plot_ssp
from publication.publication_figure import set_subfigures_abc_labels, color

# ======================================================================================================================
# Dedicated plots
# ======================================================================================================================


def plot_resilience_depth_baseline_celerity_profiles():
    env = baseline_env()
    src_rcv = baseline_src_rcv()

    situation = "all"

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Shallow water
    env_type = "sw"
    env_config = CELERITY_ENV_TYPES[env_type]
    z_ssp_sw, c_p_ssp_sw = load_mean_celerity_profile(
        _ssp_filename(env_type, situation),
        target_depth=env_config["depth"],
        ssp_data_dir=SSP_DATA_DIR,
    )
    z_bottom = z_ssp_sw[-1]
    z_in_bottom = np.array([0.0, 0.2 * z_bottom])
    z_env = np.append(z_ssp_sw, z_in_bottom + z_bottom)

    cp_env = np.append(
        c_p_ssp_sw,
        # np.array([env["c2"], env["c2"]]),  # With c in sediment
        np.array([np.nan, np.nan]),  # Without c in sediment
    )
    cs_env = np.zeros_like(cp_env)
    plot_ssp(cp_ssp=cp_env, cs_ssp=cs_env, z=z_env, z_bottom=z_bottom, ax=axs[0])
    axs[0].set_ylim(z_env.max() * 0.95, 0)

    env_type = "dw"
    env_config = CELERITY_ENV_TYPES[env_type]
    z_ssp_dw, c_p_ssp_dw = load_mean_celerity_profile(
        _ssp_filename(env_type, situation),
        target_depth=env_config["depth"],
        ssp_data_dir=SSP_DATA_DIR,
    )

    z_bottom = z_ssp_dw[-1]
    z_in_bottom = np.array([0.0, 0.2 * z_bottom])

    z_env = np.append(z_ssp_dw, z_in_bottom + z_bottom)
    cp_env = np.append(
        c_p_ssp_dw,
        # np.array([env["c2"], env["c2"]]),  # With c in sediment
        np.array([np.nan, np.nan]),  # Without c in sediment
    )

    # z_env = z_ssp_dw
    # cp_env = c_p_ssp_dw

    cs_env = np.zeros_like(cp_env)

    plot_ssp(cp_ssp=cp_env, cs_ssp=cs_env, z=z_env, z_bottom=z_bottom, ax=axs[1])
    axs[1].set_ylim(z_env.max() * 0.95, 0)

    # Plot source pos
    src_depth = src_rcv["z_s"]
    for ax in axs.flatten():
        xmin = ax.get_xlim()[0]
        ax.scatter(xmin, src_depth, s=30, color="k")
        for s in [200, 500]:
            ax.scatter(
                xmin,
                src_depth,
                s=s,
                facecolors="None",
                edgecolors="k",
                linewidths=0.5,
            )

    # Remove legend labels
    for ax in axs.flatten():
        ax.legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("Célérité [m s$^{-1}$]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    plt.show()


def plot_resilience_depth_results():
    test_arg_name = "depth"
    D_sw = CELERITY_ENV_TYPES["sw"]["depth"]
    D_dw = CELERITY_ENV_TYPES["dw"]["depth"]

    # Load results for sw
    resilience_result_dir, resilience_img_dir = _resilience_study_dirs(
        env_type="sw", kind="depth"
    )
    test_values, dist_L1, dist_L2, dist_theta_sw = load_sensitivity_distance_results(
        test_arg_name, result_dir=resilience_result_dir
    )
    delta_D = np.array(test_values) - D_sw
    delta_D_perc_sw = delta_D / D_sw * 100
    # Load results for dw
    resilience_result_dir, resilience_img_dir = _resilience_study_dirs(
        env_type="dw", kind="depth"
    )
    test_values, dist_L1, dist_L2, dist_theta_dw = load_sensitivity_distance_results(
        test_arg_name, result_dir=resilience_result_dir
    )
    delta_D_perc_dw = delta_D / D_dw * 100

    # Plots for each env type
    plt.figure(figsize=(16, 8))
    plt.plot(delta_D, dist_theta_sw, "o-", label="D = 100 m", color=color(0))
    plt.plot(delta_D, dist_theta_dw, "o-", label="D = 2000 m ", color=color(1))
    plt.xlabel(r"$\delta_D$ [m]")
    plt.ylabel(r"$\theta$")
    plt.legend()
    plt.ylim(0, 2)

    # Plots for each env type as a function of percentage depth perturbation
    print(f"delta_D_perc_sw: {delta_D_perc_sw}")
    print(f"delta_D_perc_dw: {delta_D_perc_dw}")
    plt.figure(figsize=(16, 8))
    plt.plot(delta_D_perc_sw, dist_theta_sw, "o-", label="D = 100 m", color=color(0))
    plt.plot(delta_D_perc_dw, dist_theta_dw, "o-", label="D = 2000 m ", color=color(1))
    plt.xlabel(r"$\delta_D$ [%]")
    plt.ylabel(r"$\theta$")
    plt.legend()
    plt.ylim(0, 2)

    plt.show()


def plot_resilience_depth_results_associated_extrema_gamma():

    test_arg_name = "depth"
    d12 = baseline_src_rcv()["d12"]
    r0 = baseline_src_rcv()["r0"]

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for i, env_type in enumerate(["sw", "dw"]):
        ax = axs[i]

        result_dir, resilience_img_dir = _resilience_study_dirs(
            env_type=env_type, kind=test_arg_name
        )

        with xr.open_dataset(
            os.path.join(result_dir, "gf_dataset_baseline.nc")
        ) as ds_baseline:
            # The baseline doesn't depend on which parameter is being swept
            # -- computed once, reused for every test_arg_name below.
            baseline_gamma_r0 = ds_baseline.gamma.sel(r=r0, method="nearest")

            test_values, dist_L1, dist_L2, dist_theta = (
                load_sensitivity_distance_results(
                    test_arg_name, result_dir=result_dir, file_prefix="dist_"
                )
            )
            dist = dist_theta
            # idx_min = int(np.nanargmin(dist))
            # idx_max = int(np.nanargmax(dist))
            idx_max = -1
            # value_min = test_values[idx_min]
            value_max = test_values[idx_max]

            # param_label = ARG_LABEL.get(test_arg_name, test_arg_name)
            # label_min = f"{param_label} = {value_min:g} (min dist)"
            # label_max = f"{param_label} = {value_max:g} (max dist)"

            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
            ) as ds_test_full:

                # ds_test_sel = ds_test_full.sel(
                #     **{test_arg_name: [value_min, value_max]}, method="nearest"
                # )

                ds_test_sel = ds_test_full.sel(
                    **{test_arg_name: [value_max]}, method="nearest"
                )
                gamma_sel = derive_gamma(ds_test_sel, d12)  # dims (test_arg_name, f, r)

                baseline_gamma_r0.plot(ax=ax, color="k")
                gamma_sel.isel(**{test_arg_name: 0}).sel(r=r0, method="nearest").plot(
                    ax=ax, color=color(0)
                )
                # gamma_sel.isel(**{test_arg_name: 1}).sel(r=r0, method="nearest").plot(
                #     ax=ax, color=color(1)
                # )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    legend_handles = []
    legend_handles.append(Line2D([0], [0], color="k", linestyle="-", label="Référence"))
    # legend_handles.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         color=color(0),
    #         linestyle="-",
    #         label=r"$\theta = \theta_{\text{min}}$",
    #     )
    # )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=color(0),
            linestyle="-",
            # label=r"$\theta = \theta_{\text{max}}$",
            label=r"$\delta_D = 10~\text{m}$ ",
        )
    )

    fig.legend(handles=legend_handles, loc="outside center right")
    fig.supxlabel("Fréquence [Hz]")
    fig.supylabel(r"$\gamma$")

    # Ensure y axis limits are symetrical for both subplots
    ylim = max(np.abs(axs[0].get_ylim()))
    axs[0].set_ylim(-ylim, ylim)
    ylim = max(np.abs(axs[1].get_ylim()))
    axs[1].set_ylim(-ylim, ylim)

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    plt.show()


def plot_resilience_depth_results_associated_extrema_rtf():
    # plot_extremal_resilience_dist_configs(
    #     ["depth"],
    #     metric=distance[0],
    #     result_dir=resilience_result_dir,
    #     save_dir=resilience_img_dir,
    # )
    test_arg_name = "depth"
    d12 = baseline_src_rcv()["d12"]
    r0 = baseline_src_rcv()["r0"]

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for i, env_type in enumerate(["sw", "dw"]):
        ax = axs[i]

        result_dir, resilience_img_dir = _resilience_study_dirs(
            env_type=env_type, kind=test_arg_name
        )

        with xr.open_dataset(
            os.path.join(result_dir, "gf_dataset_baseline.nc")
        ) as ds_baseline:
            # The baseline doesn't depend on which parameter is being swept
            # -- computed once, reused for every test_arg_name below.
            baseline_gamma_r0 = ds_baseline.gamma.sel(r=r0, method="nearest")
            baseline_rtf_r0 = 10 ** (baseline_gamma_r0 / 20)

            test_values, dist_L1, dist_L2, dist_theta, wd = (
                load_sensitivity_distance_results(
                    test_arg_name, result_dir=result_dir, file_prefix="dist_"
                )
            )
            # dist = dist_theta
            # idx_min = int(np.nanargmin(dist))
            # idx_max = int(np.nanargmax(dist))
            idx_max = -1
            # value_min = test_values[idx_min]
            value_max = test_values[idx_max]

            # param_label = ARG_LABEL.get(test_arg_name, test_arg_name)
            # label_min = f"{param_label} = {value_min:g} (min dist)"
            # label_max = f"{param_label} = {value_max:g} (max dist)"

            value_files = sorted(
                glob.glob(
                    os.path.join(result_dir, test_arg_name, f"{test_arg_name}_*.nc")
                )
            )
            with xr.open_mfdataset(
                value_files,
                combine="nested",
                concat_dim=test_arg_name,
                join="override",
            ) as ds_test_full:

                # ds_test_sel = ds_test_full.sel(
                #     **{test_arg_name: [value_min, value_max]}, method="nearest"
                # )

                ds_test_sel = ds_test_full.sel(
                    **{test_arg_name: [value_max]}, method="nearest"
                )
                gamma_sel = derive_gamma(ds_test_sel, d12)  # dims (test_arg_name, f, r)
                rtf_sel = 10 ** (gamma_sel / 20)

                baseline_rtf_r0.plot(ax=ax, color="k")
                rtf_sel.isel(**{test_arg_name: 0}).sel(r=r0, method="nearest").plot(
                    ax=ax, color=color(0)
                )

        # Compute wasserstein distance between baseline and perturbed rtf
        from scipy.stats import wasserstein_distance

        distribution_support = np.arange(baseline_rtf_r0.size)
        u = baseline_rtf_r0.values
        u[np.isnan(u)] = 0.0
        v = rtf_sel.isel(**{test_arg_name: 0}).sel(r=r0, method="nearest").values
        v[np.isnan(v)] = 0.0
        wd_ = wasserstein_distance(
            u_values=distribution_support,
            v_values=distribution_support,
            u_weights=u,
            v_weights=v,
        )
        print(
            f"Wasserstein distance between baseline and perturbed rtf for {env_type}: {wd}, {wd_}"
        )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    legend_handles = []
    legend_handles.append(Line2D([0], [0], color="k", linestyle="-", label="Référence"))
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=color(0),
            linestyle="-",
            # label=r"$\theta = \theta_{\text{max}}$",
            label=r"$\delta_D = 10~\text{m}$ ",
        )
    )

    fig.legend(handles=legend_handles, loc="outside center right")
    fig.supxlabel("Fréquence [Hz]")
    fig.supylabel(r"$\lvert \Pi \rvert $")

    # # Ensure y axis limits are symetrical for both subplots
    # ylim = max(np.abs(axs[0].get_ylim()))
    # axs[0].set_ylim(-ylim, ylim)
    # ylim = max(np.abs(axs[1].get_ylim()))
    # axs[1].set_ylim(-ylim, ylim)
    for ax in axs.flatten():
        ax.set_xscale("log")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    plt.show()


if __name__ == "__main__":

    # plot_resilience_depth_baseline_celerity_profiles()
    # plot_resilience_depth_results()
    # plot_resilience_depth_results_associated_extrema_gamma()
    plot_resilience_depth_results_associated_extrema_rtf()
