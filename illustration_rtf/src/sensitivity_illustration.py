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
from matplotlib.ticker import MaxNLocator

from matplotlib.lines import Line2D

from illustration_rtf.src.sensitivity import (
    load_mean_celerity_profile,
    _ssp_filename,
    baseline_env,
    baseline_src_rcv,
    load_sensitivity_distance_results,
    _resilience_study_dirs,
    derive_gamma,
    _load_celerity_std_results,
    plot_celerity_distance_vs_std,
    plot_sensitivity_curves,
    CELERITY_ENV_TYPES,
    SSP_DATA_DIR,
    ARG_LABEL,
    METRIC_LABEL,
    RESULT_DIR,
)
from propa.kraken_toolbox.plot_utils import plot_ssp
from publication.publication_figure import (
    set_subfigures_abc_labels,
    color,
    set_round_xticks,
    PubFigure,
)

PubFigure(label_fontsize=24, ticks_fontsize=20)

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


def _load_res(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    dist_wasserstein = data[:, 4] if data.shape[1] > 4 else None
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], dist_wasserstein


def plot_resilience_depth_results(distance="theta", use_plateform_res=False):
    test_arg_name = "depth"
    D_sw = CELERITY_ENV_TYPES["sw"]["depth"]
    D_dw = CELERITY_ENV_TYPES["dw"]["depth"]

    # Load results for sw

    # Load from pc
    if use_plateform_res:
        # Load from plateform results
        result_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\result_plateform_tim\resilience_depth"
        # path = os.path.join(result_dir, f"dist_depth_resilience_sw.csv")
        # path = os.path.join(result_dir, f"dist_depth_resilience_sw_wasserstein_dB.csv")
        path = os.path.join(
            result_dir, f"dist_depth_resilience_sw_wasserstein_dB_mass_norm.csv"
        )

        test_values, dist_L1, dist_L2, dist_theta_sw, dist_wass_sw = _load_res(path)
    else:
        resilience_result_dir, resilience_img_dir = _resilience_study_dirs(
            env_type="sw", kind="depth"
        )
        test_values, dist_L1, dist_L2, dist_theta_sw, dist_wass_sw = (
            load_sensitivity_distance_results(
                test_arg_name, result_dir=resilience_result_dir
            )
        )

    delta_D = np.array(test_values) - D_sw

    # Load results for dw
    if use_plateform_res:
        # Load from plateform results
        result_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\result_plateform_tim\resilience_depth"
        # path = os.path.join(result_dir, f"dist_depth_resilience_dw.csv")
        # path = os.path.join(result_dir, f"dist_depth_resilience_dw_wasserstein_dB.csv")
        path = os.path.join(
            result_dir, f"dist_depth_resilience_dw_wasserstein_dB_mass_norm.csv"
        )

        test_values, dist_L1, dist_L2, dist_theta_dw, dist_wass_dw = _load_res(path)
    else:
        resilience_result_dir, resilience_img_dir = _resilience_study_dirs(
            env_type="dw", kind="depth"
        )
        test_values, dist_L1, dist_L2, dist_theta_dw, dist_wass_dw = (
            load_sensitivity_distance_results(
                test_arg_name, result_dir=resilience_result_dir
            )
        )

    if distance == "theta":
        dist_sw = dist_theta_sw
        dist_dw = dist_theta_dw
        ylabel = r"$\theta$"
        use_ylim = True
    elif distance == "wasserstein":
        dist_sw = dist_wass_sw
        dist_dw = dist_wass_dw
        ylabel = "Distance de Wasserstein"
        use_ylim = False

    # Plots for each env type
    plt.figure(figsize=(16, 8))
    plt.plot(delta_D, dist_sw, label="D = 100 m", color=color(0))
    plt.plot(delta_D, dist_dw, label="D = 2000 m ", color=color(1))
    plt.xlabel(r"$\delta_D$ [m]")
    plt.ylabel(ylabel)
    plt.legend()
    if use_ylim:
        plt.ylim(0, 1)

    # # Plots for each env type as a function of percentage depth perturbation
    # delta_D_perc_sw = delta_D / D_sw * 100
    # delta_D_perc_dw = delta_D / D_dw * 100
    # print(f"delta_D_perc_sw: {delta_D_perc_sw}")
    # print(f"delta_D_perc_dw: {delta_D_perc_dw}")
    # plt.figure(figsize=(16, 8))
    # plt.plot(delta_D_perc_sw, dist_theta_sw, "o-", label="D = 100 m", color=color(0))
    # plt.plot(delta_D_perc_dw, dist_theta_dw, "o-", label="D = 2000 m ", color=color(1))
    # plt.xlabel(r"$\delta_D$ [\%]")
    # plt.ylabel(r"$\theta$")
    # plt.legend()
    # plt.ylim(0, 2)

    plt.show()


def plot_resilience_depth_results_associated_extrema_gamma(use_plateform_res=False):

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

            if use_plateform_res:
                plt_result_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\result_plateform_tim\resilience_depth"
                path = os.path.join(
                    plt_result_dir, f"dist_depth_resilience_{env_type}.csv"
                )
                test_values, dist_L1, dist_L2, dist_theta, _ = _load_res(path)
            else:
                test_values, dist_L1, dist_L2, dist_theta, _ = (
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


def plot_ssp_all():
    fpath_sw = os.path.join(SSP_DATA_DIR, "ssp_profiles_sw.nc")
    ds_sw = xr.open_dataset(fpath_sw)
    fpath_dw = os.path.join(SSP_DATA_DIR, "ssp_profiles_dw.nc")
    ds_dw = xr.open_dataset(fpath_dw)

    # Plot profiles
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharey=False)
    axs_sw, axs_dw = axs[0], axs[1]
    for it in range(ds_sw.sizes["time"]):
        # Shallow water profiles
        ds_sw.ssp.isel(time=it).plot(
            y="depth", yincrease=False, alpha=0.1, color="b", ax=axs_sw
        )

        # Deep water profiles
        ds_dw.ssp.isel(time=it).plot(
            y="depth", yincrease=False, alpha=0.1, color="b", ax=axs_dw
        )

    ds_sw.ssp.mean(dim="time").plot(y="depth", yincrease=False, color="k", ax=axs_sw)
    ds_dw.ssp.mean(dim="time").plot(y="depth", yincrease=False, color="k", ax=axs_dw)

    axs_sw.set_ylabel("")
    axs_dw.set_ylabel("")
    axs_sw.set_xlabel("")
    axs_dw.set_xlabel("")
    axs_sw.set_title("")
    axs_dw.set_title("")

    from matplotlib.ticker import MaxNLocator

    axs_dw.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axs_sw.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.suptitle("")
    fig.supxlabel("Célérité [m s$^{-1}$]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    plt.show()


def plot_seasonal_profiles(
    ssp_season, axs, season_name, max_number_of_profiles_to_plot=500
):
    print(f"Season {season_name}: {ssp_season.sizes['time']} profiles")
    selected_time_indices = np.random.choice(
        ssp_season.sizes["time"], size=max_number_of_profiles_to_plot, replace=False
    )  # random.choice with replace=False to avoid duplica

    for it in selected_time_indices:
        ssp_season.isel(time=it).plot(
            y="depth", yincrease=False, alpha=0.25, color="b", ax=axs
        )
    ssp_season.mean(dim="time").plot(
        y="depth", yincrease=False, color="k", linewidth=2, ax=axs
    )
    axs.set_title(f"{season_name}")
    axs.set_xlabel("")
    axs.set_ylabel("")


def plot_ssp_seasons(add_all=False, max_number_of_profiles_to_plot=500):

    fnames = [
        "ssp_profiles_sw_winter.nc",
        "ssp_profiles_sw_spring.nc",
        "ssp_profiles_sw_summer.nc",
        "ssp_profiles_sw_automn.nc",
        "ssp_profiles_dw_winter.nc",
        "ssp_profiles_dw_spring.nc",
        "ssp_profiles_dw_summer.nc",
        "ssp_profiles_dw_automn.nc",
    ]

    if add_all:
        fnames = (
            ["ssp_profiles_sw.nc"] + fnames[:4] + ["ssp_profiles_dw.nc"] + fnames[4:]
        )

    ncols = len(fnames) // 2
    fig, axs = plt.subplots(2, ncols, figsize=(14, 10), sharey="row", sharex="row")

    for k, fname in enumerate(fnames):
        j = k % ncols
        i = k // ncols
        # print(i, j)
        ssp_season = xr.open_dataset(os.path.join(SSP_DATA_DIR, fname)).ssp
        season_name = fname.split("_")[3][:-3] if len(fname.split("_")) > 3 else "all"
        plot_seasonal_profiles(
            ssp_season,
            axs[i, j],
            season_name=season_name,
            max_number_of_profiles_to_plot=max_number_of_profiles_to_plot,
        )

    for row in range(axs.shape[0]):
        set_round_xticks(axs[row], base=10, n_ticks=4)

    for ax in axs.flatten():
        ax.set_title("")
        # ax.xaxis.set_major_locator(LinearLocator(numticks=4))
        # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    fig.supxlabel("Célérité [m s$^{-1}$]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    # plt.show()


def plot_temp_salinity_seasons():
    # Load dataset
    fname = "cmems_data_1993_2026.nc"
    ds = xr.open_dataset(os.path.join(SSP_DATA_DIR, fname))
    # Load ssp to get pos
    fpath_sw = os.path.join(SSP_DATA_DIR, "ssp_profiles_sw.nc")
    ssp_sw = xr.open_dataset(fpath_sw)
    fpath_dw = os.path.join(SSP_DATA_DIR, "ssp_profiles_dw.nc")
    ssp_dw = xr.open_dataset(fpath_dw)

    ds_sw = ds.sel(
        longitude=ssp_sw.longitude.values,
        latitude=ssp_sw.latitude.values,
        method="nearest",
    )
    ds_dw = ds.sel(
        longitude=ssp_dw.longitude.values,
        latitude=ssp_dw.latitude.values,
        method="nearest",
    )

    # Extract profiles for each season
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    automn_months = [9, 10, 11]

    ds_sw_winter = ds_sw.sel(time=ds_sw.time.dt.month.isin(winter_months))
    ds_sw_spring = ds_sw.sel(time=ds_sw.time.dt.month.isin(spring_months))
    ds_sw_summer = ds_sw.sel(time=ds_sw.time.dt.month.isin(summer_months))
    ds_sw_automn = ds_sw.sel(time=ds_sw.time.dt.month.isin(automn_months))

    ds_dw_winter = ds_dw.sel(time=ds_dw.time.dt.month.isin(winter_months))
    ds_dw_spring = ds_dw.sel(time=ds_dw.time.dt.month.isin(spring_months))
    ds_dw_summer = ds_dw.sel(time=ds_dw.time.dt.month.isin(summer_months))
    ds_dw_automn = ds_dw.sel(time=ds_dw.time.dt.month.isin(automn_months))

    ds_list = [
        ds_sw_winter,
        ds_sw_spring,
        ds_sw_summer,
        ds_sw_automn,
        ds_dw_winter,
        ds_dw_spring,
        ds_dw_summer,
        ds_dw_automn,
    ]

    # Temperature
    fig, axs = plt.subplots(2, 4, figsize=(14, 10), sharey="row", sharex="row")

    for k, ds_ in enumerate(ds_list):
        j = k % 4
        i = k // 4
        # print(i, j)
        plot_seasonal_profiles(ds_.thetao, axs[i, j], season_name="")

    from matplotlib.ticker import MaxNLocator

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title("")

    fig.supxlabel("Température [°C]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    # Salinité
    fig, axs = plt.subplots(2, 4, figsize=(14, 10), sharey="row", sharex="row")

    for k, ds_ in enumerate(ds_list):
        j = k % 4
        i = k // 4
        # print(i, j)
        plot_seasonal_profiles(ds_.so, axs[i, j], season_name="")

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title("")

    fig.supxlabel("Salinité [psu]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    # plt.show()


def plot_ssp_acp_process():

    fpath_sw = os.path.join(SSP_DATA_DIR, "ssp_profiles_sw.nc")
    ssp_original_sw = xr.open_dataset(fpath_sw).ssp
    fpath_dw = os.path.join(SSP_DATA_DIR, "ssp_profiles_dw.nc")
    ssp_original_dw = xr.open_dataset(fpath_dw).ssp

    # fpath_sw = os.path.join(SSP_DATA_DIR, "synthetic_ssp_profiles_sw_1000.nc")
    # ds_synthetic_sw = xr.open_dataset(fpath_sw)
    # fpath_dw = os.path.join(SSP_DATA_DIR, "synthetic_ssp_profiles_dw_1000.nc")
    # ds_synthetic_dw = xr.open_dataset(fpath_dw)

    from illustration_rtf.src.ssp.ssp_process_eof import (
        get_ssp_eof,
        generate_new_ssp_profiles,
        convert_synthetic_to_xarray,
    )

    # Get EOFs
    n_new_samples = 50
    cumulative_variance_threshold = 0.999

    def get_eof_and_synthetic(ssp):
        ssp, eof, X_pca, pca, scaler = get_ssp_eof(
            ssp,
            cumulative_variance_threshold=cumulative_variance_threshold,
            verbose=True,
        )

        # Generate synthetic profiles
        X_ssp_synthetic = generate_new_ssp_profiles(
            pca, scaler, n_new_samples=n_new_samples
        )
        ssp_synthetic = convert_synthetic_to_xarray(
            X_ssp_synthetic=X_ssp_synthetic,
            ssp_original=ssp,
            n_components=pca.n_components_,
        )

        return eof, ssp_synthetic

    eof_sw, ssp_synthetic_sw = get_eof_and_synthetic(ssp=ssp_original_sw)
    eof_dw, ssp_synthetic_dw = get_eof_and_synthetic(ssp=ssp_original_dw)

    # Plot EOFs and original profiles
    fig, axs = plot_synthetic_profiles_and_eofs(
        ssp_original_sw,
        ssp_synthetic_sw,
        eof_sw,
        ssp_original_dw,
        ssp_synthetic_dw,
        eof_dw,
        n_eof_max=5,
        n_profiles_max=50,
    )

    # fig.supxlabel("Célérité [m s$^{-1}$]")
    fig.supylabel("Profondeur [m]")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    # plt.show()


def plot_synthetic_profiles_and_eofs(
    ssp_original_sw,
    ssp_synthetic_sw,
    eof_sw,
    ssp_original_dw,
    ssp_synthetic_dw,
    eof_dw,
    n_eof_max=4,
    n_profiles_max=50,
):

    # n_components = eof.shape[0]
    n_components = min(eof_sw.shape[0], n_eof_max)
    fig, axs = plt.subplots(
        2,
        n_components + 1,
        figsize=(14, 10),
        sharey="row",
        # sharex="row",
        gridspec_kw={"width_ratios": [1] + [0.6] * n_components},
    )

    # Share x axis by row among the EOF columns only (column 0 stays independent)
    for row in range(2):
        for ic in range(2, n_components + 1):
            axs[row, ic].sharex(axs[row, 1])

    ### Shallow water ###
    # Plot synthetic profiles
    if ssp_synthetic_sw.sizes["time"] > n_profiles_max:
        ssp_synthetic_sw = ssp_synthetic_sw.isel(
            time=np.random.choice(
                ssp_synthetic_sw.sizes["time"], n_profiles_max, replace=False
            )
        )

    for it in range(ssp_synthetic_sw.sizes["time"]):
        ssp_synthetic_sw.isel(time=it).plot(y="depth", yincrease=False, ax=axs[0, 0])

    # Plot original mean profile
    ssp_original_sw.mean(dim="time").plot(
        y="depth", yincrease=False, color="k", lw=2, ax=axs[0, 0]
    )

    # Plot EOFs
    for ic in range(n_components):
        idx_comp = ic + 1
        axs[0, idx_comp].plot(
            eof_sw[ic, :], ssp_synthetic_sw.depth.values, color="k", lw=2
        )
        # axs[0, idx_comp].set_title(rf"$v_{{{idx_comp}}}$")

    axs[0, 0].set_xlabel(r"")
    axs[0, 0].set_ylabel("")
    axs[0, 0].set_title("")

    ### Deep water ###
    # Plot synthetic profiles
    if ssp_synthetic_dw.sizes["time"] > n_profiles_max:
        ssp_synthetic_dw = ssp_synthetic_dw.isel(
            time=np.random.choice(
                ssp_synthetic_dw.sizes["time"], n_profiles_max, replace=False
            )
        )

    for it in range(ssp_synthetic_dw.sizes["time"]):
        ssp_synthetic_dw.isel(time=it).plot(y="depth", yincrease=False, ax=axs[1, 0])

    # Plot original mean profile
    ssp_original_dw.mean(dim="time").plot(
        y="depth", yincrease=False, color="k", lw=2, ax=axs[1, 0]
    )

    # Plot EOFs
    for ic in range(n_components):
        idx_comp = ic + 1
        axs[1, idx_comp].plot(
            eof_dw[ic, :], ssp_synthetic_dw.depth.values, color="k", lw=2
        )
        # axs[1, idx_comp].set_title(rf"$v_{{{idx_comp}}}$")

    axs[1, 0].set_xlabel(r"[m~s$^{-1}$]")
    axs[1, 0].set_ylabel("")
    axs[1, 0].set_title("")

    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    fig.supylabel("Profondeur [m]")

    return fig, axs


def _add_celerity(
    ax, result_dir, situation, metric="theta", set_label=False, linear_reg=False
):
    profile_idx_dist, dist_L1, dist_L2, dist_theta, dist_wasserstein = (
        load_sensitivity_distance_results(
            situation, result_dir=result_dir, file_prefix="dist_"
        )
    )
    profile_idx_std, std = _load_celerity_std_results(situation, result_dir)

    # Sort
    order_dist = np.argsort(profile_idx_dist)
    order_std = np.argsort(profile_idx_std)
    dist = {
        "L1": dist_L1,
        "L2": dist_L2,
        "theta": dist_theta,
        "wasserstein": dist_wasserstein,
    }[metric][order_dist]

    # # TODO remove that
    # if metric == "theta":
    #     dist /= 2
    # else:
    #     dist /= np.max(dist)

    # print(situation, np.max(dist))

    std = std[order_std]

    if set_label:
        ax.scatter(std, dist, label=f"{METRIC_LABEL[metric]}", s=12)
        ax.legend()

    else:
        ax.scatter(std, dist, s=12)

    if metric == "theta":
        ax.set_ylim(0, 1)

    # Linear regression
    if linear_reg:
        from scipy.stats import linregress

        slope, intercept, r, p, se = linregress(x=std, y=dist)
        x_std = np.linspace(0, std.max(), 100)
        y_dist = intercept + slope * x_std
        ax.plot(
            x_std,
            y_dist,
            color="k",
            label=f"{intercept:.2f} + {slope:.2f}$\sigma_c$ ($r^2$ = {r**2:.2f})",
        )

        ax.legend(fontsize=12)

    # # Remove percentile
    # # axs[0].axvline(np.percentile(std, 50))
    # std_max = np.percentile(std, 75)
    # slope, intercept, r, p, se = linregress(x=std[std < std_max], y=dist[std < std_max])
    # x_std = np.linspace(0, std_max, 100)
    # y_dist = intercept + slope * x_std
    # ax.plot(
    #     x_std,
    #     y_dist,
    #     color="k",
    #     linestyle="--",
    #     label=f"{intercept:.2f} + {slope:.2f}$\sigma$ ($r^2$ = {r**2:.2f})",
    # )


def _plot_celerity_distance_vs_std(
    result_dir,
    situation=None,
    metric="theta",
):

    result_dir_ = result_dir
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), squeeze=False, sharey=True)
    axs = axs.flatten()

    if np.asarray(metric).size > 1:
        for met in metric:
            result_dir = os.path.join(result_dir_, "sw")
            _add_celerity(
                ax=axs[0],
                result_dir=result_dir,
                situation=situation,
                metric=met,
                set_label=True,
            )

            result_dir = os.path.join(result_dir_, "dw")
            _add_celerity(
                ax=axs[1],
                result_dir=result_dir,
                situation=situation,
                metric=met,
                set_label=True,
            )

        # fig.supxlabel(r"STD from baseline profile [m s$^{-1}$]")
        fig.supxlabel(r"$\sigma_c$ [m s$^{-1}$]")

        fig.supylabel(f"Distance")
    else:
        result_dir = os.path.join(result_dir_, "sw")
        _add_celerity(
            ax=axs[0],
            result_dir=result_dir,
            situation=situation,
            metric=metric,
            set_label=False,
        )

        result_dir = os.path.join(result_dir_, "dw")
        _add_celerity(
            ax=axs[1],
            result_dir=result_dir,
            situation=situation,
            metric=metric,
            set_label=False,
        )

        # fig.supxlabel(r"STD from baseline profile [m s$^{-1}$]")
        fig.supxlabel(r"$\sigma_c$ [m s$^{-1}$]")
        fig.supylabel(f"Distance {METRIC_LABEL[metric]}")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    return fig


def plot_resilience_celerity_results(metric="theta", use_plateform_res=True):

    # Load from pc
    if use_plateform_res:
        # Load from plateform results
        result_dir = RESULT_PLATEFORM_DIR_RESILIENCE_SSP
    else:
        result_dir = None

    situation = "all"
    fig = _plot_celerity_distance_vs_std(
        situation=situation,
        metric=metric,
        result_dir=result_dir,
    )

    situation = "winter"
    fig = _plot_celerity_distance_vs_std(
        situation=situation,
        metric=metric,
        result_dir=result_dir,
    )

    situation = "summer"
    fig = _plot_celerity_distance_vs_std(
        situation=situation,
        metric=metric,
        result_dir=result_dir,
    )

    situation = "automn"
    fig = _plot_celerity_distance_vs_std(
        situation=situation,
        metric=metric,
        result_dir=result_dir,
    )

    situation = "spring"
    fig = _plot_celerity_distance_vs_std(
        situation=situation,
        metric=metric,
        result_dir=result_dir,
    )


def plot_sensitivity_at_r0(
    distance=["theta"],
    params=["c1", "rho2", "attn2"],
):
    result_dir = RESULT_PLATEFORM_DIR_SENSITIVITY

    fig = plot_sensitivity_curves(
        test_arg_names=params,
        distance=distance,
        ylabel=f"Distance ({METRIC_LABEL[distance[0]]})",
        result_dir=result_dir,
        save_dir=None,
    )

    axs = fig.get_axes()
    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )


def plot_sensitivity_mainlobe_width(distance=["theta"], use_plateform_res=True):

    # Load from pc
    if use_plateform_res:
        # Load from plateform results
        result_dir = RESULT_PLATEFORM_DIR_SENSITIVITY
    else:
        result_dir = RESULT_DIR

    fig = plot_sensitivity_curves(
        distance=distance,
        file_prefix="intrinsic_mainlobe_width_",
        ylabel=r"$2 r_{\text{-3dB}}$ [m]",
        result_dir=result_dir,
        save_dir=None,
    )

    axs = fig.get_axes()

    for ax in axs:
        ax.set_ylim(0, 2000)
    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )


def plot_celerity_distance_vs_std_seasons(
    metric="theta", use_plateform_res=True, linear_reg=False, add_all=False
):

    # Load from pc
    if use_plateform_res:
        # Load from plateform results
        result_dir = RESULT_PLATEFORM_DIR_RESILIENCE_SSP
    else:
        result_dir = None

    seasons = [
        "winter",
        "spring",
        "summer",
        "automn",
    ] * 2

    if add_all:
        seasons = ["all"] + seasons[:4] + ["all"] + seasons[4:]
    print(seasons)

    result_dir_ = result_dir
    result_dir_sw = os.path.join(result_dir_, "sw")
    result_dir_dw = os.path.join(result_dir_, "dw")
    result_dirs = [result_dir_sw, result_dir_dw]

    ncol = len(seasons) // 2
    fig, axs = plt.subplots(
        2, ncol, figsize=(16, 8), squeeze=False, sharey=True, sharex=True
    )

    if np.asarray(metric).size > 1:
        for met in metric:

            for k, season in enumerate(seasons):
                j = k % ncol
                i = k // ncol
                # print(i, j)

                result_dir = result_dirs[i]

                _add_celerity(
                    ax=axs[i, j],
                    result_dir=result_dir,
                    situation=season,
                    metric=met,
                    set_label=True,
                    linear_reg=linear_reg,
                )

        # fig.supxlabel(r"STD from baseline profile [m s$^{-1}$]")
        fig.supxlabel(r"$\sigma_c$ [m s$^{-1}$]")
        fig.supylabel(f"Distance")

    else:
        for k, season in enumerate(seasons):
            j = k % ncol
            i = k // ncol
            # print(i, j)

            result_dir = result_dirs[i]

            _add_celerity(
                ax=axs[i, j],
                result_dir=result_dir,
                situation=season,
                metric=metric,
                set_label=False,
                linear_reg=linear_reg,
            )

        # fig.supxlabel(r"STD from baseline profile [m s$^{-1}$]")
        fig.supxlabel(r"$\sigma_c$ [m s$^{-1}$]")
        fig.supylabel(f"Distance {METRIC_LABEL[metric]}")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )


if __name__ == "__main__":

    RESULT_PLATEFORM_DIR_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\result_plateform_tim"
    # RESULT_PLATEFORM_DIR_RESILIENCE_SSP = os.path.join(
    #     RESULT_PLATEFORM_DIR_ROOT, "resilience_ssp_17092026"
    # )

    RESULT_PLATEFORM_DIR_RESILIENCE_SSP = os.path.join(
        RESULT_PLATEFORM_DIR_ROOT, "resillience_ssp_06102026"
    )

    RESULT_PLATEFORM_DIR_SENSITIVITY = os.path.join(
        RESULT_PLATEFORM_DIR_ROOT, "sensitivity_17092026", "result"
    )

    # result_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\result_plateform_tim\resilience_ssp"

    # plot_resilience_depth_baseline_celerity_profiles()
    # plot_resilience_depth_results_associated_extrema_rtf()

    # plot_resilience_celerity_results(metric=["theta", "wasserstein"])

    # plot_sensitivity_mainlobe_width(distance=["theta"], use_plateform_res=False)

    # # Validé
    plot_sensitivity_at_r0(distance=["theta"])
    # plot_resilience_depth_results(distance="wasserstein", use_plateform_res=True)
    # plot_resilience_depth_results(distance="theta", use_plateform_res=True)
    # plot_resilience_depth_results_associated_extrema_gamma(use_plateform_res=True)
    # plot_ssp_all()
    # plot_ssp_seasons(add_all=True, max_number_of_profiles_to_plot=100)
    # plot_temp_salinity_seasons()
    # plot_ssp_acp_process()
    # plot_resilience_celerity_results(metric="theta")
    # plot_resilience_celerity_results(metric="wasserstein")

    # plot_celerity_distance_vs_std_seasons(metric="theta", linear_reg=True, add_all=True)

    plt.show()
