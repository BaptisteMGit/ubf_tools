#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_figures.py
@Time    :   2025/02/26 11:37:40
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Create figures for JASA publication
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr


from publication.publication_figure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    get_rcv_couples,
)

import propa.rtf.rtf_localisation.zhang_et_al_testcase.src.params as p

from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    perf_threshold,
    study_perf_vs_snr_publi,
    study_perf_vs_snr_publi_lfm_wgn,
    study_perf_vs_snr_compare_arrays_publi,
    plot_fullarray_ambiguity_surfaces_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi_violin,
)

# pfig = PubFigure()


def no_noise_amb_surf():
    # Params
    antenna_type = "zhang"
    _, _, source, grid, _, _ = params(antenna_type=antenna_type)
    dx = grid["dx"]
    dy = grid["dy"]

    # Full simu
    folder = f"fullsimu_dx{dx}m_dy{dy}m"
    root_data = os.path.join(p.root_data, folder)

    array_label = "s1_s2_s3_s4_s5_s6"
    data_fname_fa = f"loc_zhang_dx{dx}m_dy{dy}m_fullarray_{array_label}.nc"
    fpath = os.path.join(root_data, data_fname_fa)
    ds_fa = xr.open_dataset(fpath)
    vmax = 0
    vmin = -8
    x_src = source["x"]
    y_src = source["y"]

    # Root img
    root_img = os.path.join(p.root_img_publi, "hexagonal_array_noise_free")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_fullarray_ambiguity_surfaces_publi(
        ds_fa,
        root_img,
        x_src,
        y_src,
        vmin,
        vmax,
        cmap="jet",
    )


def perf_vs_nb_rcv(snrs=[0], root_data=p.root_data, n_rcv=[3, 4, 5, 6]):
    # Root img
    data_name = os.path.split(os.path.split(root_data)[0])[1]
    root_img = os.path.join(
        p.root_img_publi, "performance_against_number_of_receivers", data_name
    )
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_performance_vs_number_of_rcv_in_subarray_publi(
        root_img=root_img, snrs=snrs, root_data=root_data, n_rcv=n_rcv
    )
    # plot_performance_vs_number_of_rcv_in_subarray_publi_violin(
    #     root_img=root_img, snrs=snrs, root_data=root_data
    # )


def perf_vs_snr(root_data):
    # Root img
    data_name = os.path.split(os.path.split(root_data)[0])[1]
    root_img = os.path.join(p.root_img_publi, "performance_against_snr", data_name)
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    best_subarray = [0, 1, 4]
    worst_subarray = [0, 1, 3]
    # subarrays_list = [best_subarray, worst_subarray]
    # subarrays_list = [[0, 1, 2, 3, 4, 5]]
    subarrays_list = [[0, 1, 2, 3, 4, 5], worst_subarray, best_subarray]

    perf_threshold(subarrays_list, root_data, root_res=root_img)

    study_perf_vs_snr_compare_arrays_publi(
        subarrays_list, root_img, root_data=root_data
    )
    study_perf_vs_snr_publi(subarrays_list, root_img, root_data=root_data)


def study_lfm_vs_wgn_snr(root_data_lfm, root_data_wgn):
    # Root img
    root_img = os.path.join(p.root_img_publi, "performance_against_snr", "lfm_vs_wgn")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    best_subarray = [0, 1, 4]
    worst_subarray = [0, 1, 3]
    subarrays_list = [[0, 1, 2, 3, 4, 5], worst_subarray, best_subarray]

    # study_perf_vs_snr_publi(subarrays_list, root_img, root_data=root_data)
    study_perf_vs_snr_publi_lfm_wgn(
        subarrays_list, root_img, root_data_lfm, root_data_wgn
    )


# ======================================================================================================================
# Clean functions to generate JASA EL figures (13/11/2025)
# ======================================================================================================================
import matplotlib.pyplot as plt
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    # params,
    # get_array_label,
    # get_rcv_couples,
    # estimate_msr,
    # find_mainlobe,
    get_subarrays,
    load_msr_rmse_res_subarrays,
    # get_estimated_src_pos,
    # get_axis_order,
    # get_hull_points,
    # get_mainlobe_mask,
    # get_mainlobe_contours,
)


def generate_jasael_fig_3a_13112025(root_data, root_img):

    subarrays_list = [[0, 1, 2, 3, 4, 5]]

    # Load results (all available snrs)
    msr, dr, rmse = load_msr_rmse_res_subarrays(subarrays_list, root_data=root_data)

    for sa_key in msr.keys():
        # Extract info dataframes for current subarray
        rcv_ids = [f"{id[0]}_{id[1]}" for id in sa_key.split("_")]
        rcv_str = "$" + ", \,".join(rcv_ids) + "$"
        dr_sa = dr[sa_key]
        msr_sa = msr[sa_key]
        rmse_sa = rmse[sa_key]

        ## Combined plot ##
        # Select fewer snrs to plot
        rmse_sa = rmse_sa.loc[rmse_sa.index >= -15]
        msr_sa = msr_sa.loc[msr_sa.index >= -15]

        # Plot RMSE
        fig, ax1 = plt.subplots(figsize=(8, 6))
        rmse_gcc = ax1.plot(
            rmse_sa.index,
            rmse_sa["dcf"],
            "s--",
            label="RMSE DCF",
            color="tab:blue",
            markersize=4,
        )
        rmse_rtf = ax1.plot(
            rmse_sa.index,
            rmse_sa["rtf"],
            "o-",
            label="RMSE RTF-MFP",
            color="tab:blue",
            markersize=4,
        )
        ax1.set_xlabel("SNR [dB]")
        ax1.set_ylabel("RMSE [m]", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Create a second y-axis for MSR
        ax2 = ax1.twinx()
        msr_gcc = ax2.errorbar(
            msr_sa.index,
            msr_sa.dcf_mean,
            yerr=msr_sa.dcf_std,
            color="tab:red",
            capsize=4,
            fmt="s--",
            label="MSR DCF",
            markersize=4,
        )
        msr_rtf = ax2.errorbar(
            msr_sa.index,
            msr_sa.rtf_mean,
            yerr=msr_sa.rtf_std,
            color="tab:red",
            capsize=4,
            fmt="o-",
            label="MSR RTF-MFP",
            markersize=4,
        )

        ax2.set_ylabel("MSR [dB]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        handles = [rmse_gcc[0], rmse_rtf[0], msr_gcc, msr_rtf]
        labels = [h.get_label() for h in handles]
        fig.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            # bbox_to_anchor=(0.81, 0.7),
            bbox_to_anchor=(0.85, 0.98),
            ncol=1,
        )

        # Save the combined figure
        # fpath = os.path.join(root_img, f"rmse_msr_combined_vs_snr_{sa_key}")
        fpath = os.path.join(root_img, f"figure_3a")

        plt.savefig(f"{fpath}.eps", dpi=300)
        plt.savefig(f"{fpath}.png", dpi=300)
        plt.savefig(f"{fpath}.pdf", dpi=600)


def generate_jasael_fig_3b_13112025(root_data, root_img):
    # Set snrs
    snrs = [-10]

    # For the JASA EL we consider all the potential subarrays containing from 3 to 6 receivers -> n_rcv=[3,4,5,6]
    n_rcv = [3, 4, 5, 6]
    subarrays_list = []

    for i in n_rcv:
        subarrays_list += list(get_subarrays(nr_fullarray=6, nr_subarray=i))
    # Build associated labels
    subarray_sizes = [len(sa) for sa in subarrays_list]
    subarray_sizes_unique = np.unique(subarray_sizes)

    ### Load results and reorganize them ###
    msr, dr, rmse = load_msr_rmse_res_subarrays(subarrays_list, root_data=root_data)

    # Init list to store all results
    rmse_gcc_mean_allsnrs = []
    rmse_rtf_mean_allsnrs = []
    dr_gcc_mean_allsnrs = []
    dr_rtf_mean_allsnrs = []
    msr_gcc_mean_allsnrs = []
    msr_rtf_mean_allsnrs = []
    rmse_gcc_std_allsnrs = []
    rmse_rtf_std_allsnrs = []
    dr_gcc_std_allsnrs = []
    dr_rtf_std_allsnrs = []
    msr_gcc_std_allsnrs = []
    msr_rtf_std_allsnrs = []

    for snr in snrs:
        # Plot RMSE vs nr

        rmse_rtf = []
        rmse_gcc = []
        dr_rtf = []
        dr_gcc = []
        msr_rtf = []
        msr_gcc = []

        # Group by number of receivers in subarray
        for sa_size in subarray_sizes_unique:
            idx_required_size_sa = np.where(subarray_sizes == sa_size)[0]
            key_required_size_sa = [
                list(msr.keys())[idx] for idx in idx_required_size_sa
            ]
            # Get rmse, dr and msr for subarrays of size sa_size
            rmse_for_required_size_sa = [
                rmse[key].loc[snr] for key in key_required_size_sa
            ]
            dr_mu_for_required_size_sa = [
                dr[key].loc[snr][["rtf_mean", "dcf_mean"]]
                for key in key_required_size_sa
            ]
            msr_mu_for_required_size_sa = [
                msr[key].loc[snr][["rtf_mean", "dcf_mean"]]
                for key in key_required_size_sa
            ]

            # RMSE
            rmse_gcc_for_required_size_sa = [
                rmse_for_required_size_sa[i]["dcf"]
                for i in range(len(rmse_for_required_size_sa))
            ]
            rmse_rtf_for_required_size_sa = [
                rmse_for_required_size_sa[i]["rtf"]
                for i in range(len(rmse_for_required_size_sa))
            ]
            rmse_gcc.append(rmse_gcc_for_required_size_sa)
            rmse_rtf.append(rmse_rtf_for_required_size_sa)

            # DR
            dr_gcc_for_required_size_sa = [
                # dr["dr_gcc"].loc[snr] for dr in dr_mu_for_required_size_sa
                dr_mu_for_required_size_sa[i]["dcf_mean"]
                for i in range(len(dr_mu_for_required_size_sa))
            ]
            dr_rtf_for_required_size_sa = [
                dr_mu_for_required_size_sa[i]["rtf_mean"]
                for i in range(len(dr_mu_for_required_size_sa))
            ]
            dr_gcc.append(dr_gcc_for_required_size_sa)
            dr_rtf.append(dr_rtf_for_required_size_sa)

            # MSR
            msr_gcc_for_required_size_sa = [
                msr_mu_for_required_size_sa[i]["dcf_mean"]
                for i in range(len(msr_mu_for_required_size_sa))
            ]
            msr_rtf_for_required_size_sa = [
                msr_mu_for_required_size_sa[i]["rtf_mean"]
                for i in range(len(msr_mu_for_required_size_sa))
            ]
            msr_gcc.append(msr_gcc_for_required_size_sa)
            msr_rtf.append(msr_rtf_for_required_size_sa)

        # Derive mean of each metric per subarray size
        rmse_gcc_mean = np.array([np.mean(rmse) for rmse in rmse_gcc])
        rmse_rtf_mean = np.array([np.mean(rmse) for rmse in rmse_rtf])
        dr_gcc_mean = np.array([np.mean(dr) for dr in dr_gcc])
        dr_rtf_mean = np.array([np.mean(dr) for dr in dr_rtf])
        msr_gcc_mean = np.array([np.mean(msr) for msr in msr_gcc])
        msr_rtf_mean = np.array([np.mean(msr) for msr in msr_rtf])

        # Derive std for each metric per subarray size
        rmse_gcc_std = np.array([np.std(rmse) for rmse in rmse_gcc])
        rmse_rtf_std = np.array([np.std(rmse) for rmse in rmse_rtf])
        dr_gcc_std = np.array([np.std(dr) for dr in dr_gcc])
        dr_rtf_std = np.array([np.std(dr) for dr in dr_rtf])
        msr_gcc_std = np.array([np.std(msr) for msr in msr_gcc])
        msr_rtf_std = np.array([np.std(msr) for msr in msr_rtf])

        # Add to global list
        rmse_gcc_mean_allsnrs.append(rmse_gcc_mean)
        rmse_rtf_mean_allsnrs.append(rmse_rtf_mean)
        dr_gcc_mean_allsnrs.append(dr_gcc_mean)
        dr_rtf_mean_allsnrs.append(dr_rtf_mean)
        msr_gcc_mean_allsnrs.append(msr_gcc_mean)
        msr_rtf_mean_allsnrs.append(msr_rtf_mean)
        rmse_gcc_std_allsnrs.append(rmse_gcc_std)
        rmse_rtf_std_allsnrs.append(rmse_rtf_std)
        dr_gcc_std_allsnrs.append(dr_gcc_std)
        dr_rtf_std_allsnrs.append(dr_rtf_std)
        msr_gcc_std_allsnrs.append(msr_gcc_std)
        msr_rtf_std_allsnrs.append(msr_rtf_std)

        ### Plot results ###
        snr = snrs[0]

        # Plot RMSE and MSR on the same figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot RMSE
        rmse_gcc = ax1.errorbar(
            subarray_sizes_unique,
            rmse_gcc_mean,
            yerr=rmse_gcc_std,
            color="tab:blue",
            capsize=4,
            fmt="o--",
            label="RMSE DCF",
            markersize=4,
        )
        rmse_rtf = ax1.errorbar(
            subarray_sizes_unique,
            rmse_rtf_mean,
            yerr=rmse_rtf_std,
            color="tab:blue",
            capsize=4,
            fmt="o-",
            label="RMSE RTF-MFP",
            markersize=4,
        )
        ax1.set_xlabel("Number of receivers in subarray")
        ax1.set_ylabel("RMSE [m]", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Create a second y-axis for MSR
        ax2 = ax1.twinx()
        msr_gcc = ax2.errorbar(
            subarray_sizes_unique,
            msr_gcc_mean,
            yerr=msr_gcc_std,
            color="tab:red",
            capsize=4,
            fmt="o--",
            label="MSR DCF",
            markersize=4,
        )
        msr_rtf = ax2.errorbar(
            subarray_sizes_unique,
            msr_rtf_mean,
            yerr=msr_rtf_std,
            color="tab:red",
            capsize=4,
            fmt="o-",
            label="MSR RTF-MFP",
            markersize=4,
        )
        ax2.set_ylabel("MSR [dB]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Collect handles and labels from both axes
        handles = [rmse_gcc, rmse_rtf, msr_gcc, msr_rtf]
        labels = [h.get_label() for h in handles]
        fig.legend(
            handles,
            labels,
            loc="lower left",
            frameon=True,
            # bbox_to_anchor=(0.81, 0.7),
            bbox_to_anchor=(0.16, 0.17),
            ncol=1,
        )

        # Save the combined figure
        # fpath = os.path.join(root_img, f"rmse_msr_combined_snr{snr}")
        fpath = os.path.join(root_img, f"figure_3b")

        plt.savefig(f"{fpath}.eps", dpi=300)
        plt.savefig(f"{fpath}.png", dpi=300)
        plt.savefig(f"{fpath}.pdf", dpi=600)


def generate_jasael_fig_2_13112025(root_img):
    antenna_type = "zhang"
    _, _, source, grid, _, _ = params(antenna_type=antenna_type)
    dx = grid["dx"]
    dy = grid["dy"]

    # Full simu
    folder = f"fullsimu_dx{dx}m_dy{dy}m"
    root_data = os.path.join(p.root_data, folder)

    array_label = "s1_s2_s3_s4_s5_s6"
    data_fname_fa = f"loc_zhang_dx{dx}m_dy{dy}m_fullarray_{array_label}.nc"
    fpath = os.path.join(root_data, data_fname_fa)
    ds_fa = xr.open_dataset(fpath)
    vmax = 0
    vmin = -8
    x_src = source["x"]
    y_src = source["y"]

    true_pos_label = (
        r"$X_{src} = ( "
        + f"{x_src:.0f}\,"
        + r"\textrm{m},\,"
        + f"{y_src:.0f}\,"
        + r"\textrm{m})$"
    )

    # Plot d_gcc and d_rtf
    cmap = "jet"
    lab = ["a", "b"]

    # Convert to km for plotting
    x_src = x_src / 1e3
    y_src = y_src / 1e3

    for i, dist in enumerate(["d_gcc", "d_rtf"]):

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        amb_surf = ds_fa[dist]

        # Convert to km for plotting
        amb_surf = amb_surf.assign_coords(
            x=amb_surf.x / 1e3,
            y=amb_surf.y / 1e3,
        )

        im = amb_surf.plot(
            x="x",
            y="y",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            extend="neither",
            cbar_kwargs={"label": r"$q$ [dB]"},
            rasterized=True,
        )

        ax.scatter(
            x_src,
            y_src,
            color="k",
            label=true_pos_label,
            marker="o",
            facecolors="none",
            s=900,
            linewidths=6,
        )

        ax.set_xlabel(r"$x$" + " [km]")
        ax.set_ylabel(r"$y$" + " [km]")
        # ax.set_xlabel(r"$x$" + " [m]")
        # ax.set_ylabel(r"$y$" + " [m]")

        # Save figure
        # fpath = os.path.join(root_img, f"amb_surf_{dist}")
        fpath = os.path.join(root_img, f"figure_2{lab[i]}")

        plt.savefig(f"{fpath}.eps", dpi=300)
        plt.savefig(f"{fpath}.png", dpi=300)
        plt.savefig(f"{fpath}.pdf", dpi=600)


def generate_jasael_fig_13112025():
    """Generate all figures for JASA EL publication (13/11/2025)"""

    pfig = PubFigure(
        label_fontsize=27,
        ticks_fontsize=25,
        labelpad=15,
        legend_fontsize=18,
        title_fontsize=14,
    )

    root_backup = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\backups"

    # Library source = LFM
    data_name = "rtf_zhang_backup_07041041"
    root_data_lfm = os.path.join(root_backup, data_name, "data")

    # Set root img
    root_img = os.path.join(p.root_img_publi, "jasa_el_figures_13112025")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Figure 2 (a and b) : No noise ambiguity surface for full array
    # no_noise_amb_surf()
    generate_jasael_fig_2_13112025(root_img=root_img)

    # Figure 3a : MSR, and RMSE vs SNR for full array (LFM source)
    generate_jasael_fig_3a_13112025(root_data=root_data_lfm, root_img=root_img)

    # Figure 3b : Performance vs number of receivers at SNR = -10 dB (LFM source)
    generate_jasael_fig_3b_13112025(root_data=root_data_lfm, root_img=root_img)


if __name__ == "__main__":
    # no_noise_amb_surf()

    # root_backup = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\backups"

    # # Library source = LFM
    # data_name = "rtf_zhang_backup_07041041"
    # root_data_lfm = os.path.join(root_backup, data_name, "data")

    # Library source = WGN
    # data_name = "rtf_zhang_backup_11042025"
    # data_name = "rtf_zhang_backup_05052025"
    # root_data_wgn = os.path.join(root_backup, data_name, "data")

    # p1 = np.arange(-15, -10, 1)
    # p2 = np.arange(-5, 5, 1)
    # p3 = np.arange(5, 15, 1)
    # p4 = np.arange(-5, 15, 1)
    # p5 = np.arange(-10, 1, 1)

    # perf_vs_nb_rcv(root_data=root_data, snrs=p1)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p2)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p3)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p4)

    # study_lfm_vs_wgn_snr(root_data_lfm=root_data_lfm, root_data_wgn=root_data_wgn)

    ##### FIGURES RETENUES POUR LA PUBLIE JASA EL #####
    generate_jasael_fig_13112025()
