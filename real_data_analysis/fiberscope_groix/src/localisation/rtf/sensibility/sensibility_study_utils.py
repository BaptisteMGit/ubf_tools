#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sensibility_study_utils.py
@Time    :   2026/04/15 13:33:43
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================


import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.special import kl_div
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import (
    wasserstein_distance,
    wasserstein_distance_nd,
    entropy,
    gaussian_kde,
)
from datetime import timedelta, datetime
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

from publication.publication_figure import (
    PubFigure,
    LargeFigure,
    color,
    set_subfigures_abc_labels,
)
from propa.rtf.rtf_utils import D_hermitian_angle_fast, D_euclidian


from real_data_analysis.fiberscope_groix.src.fiberscope_groix_manager import (
    # ActiveFiberscopeManager,
    # PassiveFiberscopeManager,
    BandFilter,
)

from real_data_analysis.fiberscope_groix.src.localisation.rtf.rtf_mfp import (
    RTF_MFP_Processor,
)
from misc import progression_bar


# ======================================================================================================================
# Plotting routines
# ======================================================================================================================
# Tracer des positions d'émission et des positions des OBS sur la carte
def plot_seq_replica_positions(df_seq, ds_gps, root_fig):
    """
    Plot the interpolated GPS positions of the source along the sequence, as well as the a priori positions of the OBS.
    Parameters
    ----------
    df_seq : pandas.DataFrame
        DataFrame containing the interpolated GPS positions of the source along the sequence.
    ds_gps : xarray.Dataset
        Dataset containing the a priori positions of the OBS.
    root_fig : str
        Root directory to save the figure.
    """
    pfig = PubFigure(size=(10, 8), legend_fontsize=16)
    # Plot
    plt.figure()

    seq_id = df_seq["sequence_id"].iloc[0]
    # Série de positions successives
    plt.scatter(
        df_seq["emission_interp_e_gps"],
        df_seq["emission_interp_n_gps"],
        marker="+",
        label=f"Event ({seq_id})",
        c=np.arange(df_seq["emission_interp_e_gps"].size),
        cmap="jet",
    )
    plt.colorbar(label="Replica ID")

    keys = ["obs1", "obs2", "obs3"]
    label = {
        "obs1": "1S",
        "obs2": "2S",
        "obs3": "3S",
        "t1": "1",
        "t2": "2",
        "t3": "3",
        "t4": "4",
        "t5": "5",
    }
    for ik, k in enumerate(keys):
        e = ds_gps.attrs[f"{k}_e_apriori"]
        n = ds_gps.attrs[f"{k}_n_apriori"]
        plt.scatter(
            e,
            n,
            marker="D",
            label=label[k],
            zorder=10,
            color=color(ik),
            s=40,
        )

    plt.legend()
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    fpath = os.path.join(root_fig, "emission_positions.png")
    plt.savefig(fpath)


def plot_speed_along_seq(df_seq, root_fig):
    """
    Plot the speed along the sequence, computed from the interpolated GPS positions.

    Parameters
    ----------
    df_seq : pandas.DataFrame
        DataFrame containing the interpolated GPS positions of the source along the sequence.
    root_fig : str
        Root directory to save the figure.
    """

    ve = df_seq["emission_interp_ve_gps"]
    vn = df_seq["emission_interp_vn_gps"]
    vs = np.vstack((ve, vn))
    vs_norm = np.linalg.norm(vs, axis=0)

    plt.figure()
    plt.plot(vs_norm)
    plt.xlabel("Replica ID")
    plt.ylabel(r"$\lVert \vec{v_{ship}} \rVert$")
    fpath = os.path.join(root_fig, "speed_along_seq.png")
    plt.savefig(fpath, dpi=300)

    print(f"Median source speed along traj : vs = {np.median(vs_norm):.2f} m.s-1")
    print(f"Std source speed along traj : vs = {np.std(vs_norm):.2f} m.s-1")


def plot_sequence_spectrogram(
    ds_sig, ds_wav, fig_folder, nperseg=2**12, noverlap=2**11, fmin=200, fmax=900
):
    """
    Plot the spectrogram of the signal recorded by each OBS during the sequence.
    Parameters
    ----------
    ds_sig : xarray.Dataset
        Dataset containing the signal metadata (start datetime, datetime format, etc.).
    ds_wav : xarray.Dataset
        Dataset containing the raw signal recorded by each OBS.
    fig_folder : str
        Root directory to save the figure.
    """
    datetime_fmt = ds_sig.attrs["datetime_format"]
    t_start = ds_sig.attrs[f"start_datetime"]
    t_start = datetime.strptime(t_start, datetime_fmt)
    t_end = t_start + timedelta(seconds=int(np.ceil(np.max(ds_sig.time.values))))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    for i, obs_id in enumerate([1, 2, 3]):

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        fs = ds_wav.attrs[f"fs_obs{obs_id}"]

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Derive stft
        ff, tt, stft = sp.stft(
            sig_win.values,  # .values -> ici on charge les données en mémoire (un tout petit subset seulement)
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # U^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft))  # dB re 1uPa**2 / Hz ou dB re 1 (m/s)^2 / Hz
        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t0_slice + timedelta(seconds=tt[-1]),
            freq=f"{tt[1]-tt[0]}s",
            inclusive="both",
        )

        # Plot
        cmap = "magma"
        # vmin = np.percentile(sxx, 10)
        # vmax = np.percentile(sxx, 99)
        vmin = 25
        vmax = 45

        im = axs[i].pcolormesh(tt_datetime, ff, sxx, cmap=cmap, vmin=vmin, vmax=vmax)

        clabel = r"dB re 1$\mu$Pa$^2$ / Hz"
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="2%", pad=0.10)
        fig.colorbar(im, cax=cax, orientation="vertical", label=clabel)

        # axs[ich].colorbar(im, label=clabel)

        axs[i].set_title(f"OBS{obs_id}")
        axs[i].set_ylim([fmin, fmax])

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    fig.supylabel("Fréquence [Hz]")
    fig.supxlabel("Temps UTC")
    # fig.suptitle(f"OBS{obs_id}")

    fpath = os.path.join(fig_folder, "spectro_3obs.png")
    plt.savefig(fpath, bbox_inches="tight")


def plot_rtf_mod_along_sequence(
    ds,
    dist_rcv,
    obs_cpa_idx,
    reps,
    fig_folder,
    replica_id_slice=slice(0, 10000),
    fmin=200,
    fmax=900,
):
    """
    Plot the module of the RTF along the sequence, for each OBS. The CPA positions of the OBS are also plotted as horizontal lines.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the RTF estimates and metadata.
    dist_rcv : numpy.ndarray
        Array containing the distance from each replica to each receiver.
    obs_cpa_idx : numpy.ndarray
        Array containing the index of the replica corresponding to the CPA position of each OBS.
    reps : numpy.ndarray
        Array containing the replica IDs.
    fig_folder : str
        Root directory to save the figure.
    """

    n_rcv = ds.h_index.size
    idx_rcv_plot = [idx for idx in ds.h_index.values if idx != ds.reference_receiver_id]

    fig, axs = plt.subplots(
        nrows=1, ncols=len(idx_rcv_plot), sharex=True, sharey="row", figsize=(16, 10)
    )
    ax_mod = axs

    rtf_cs_evd_amp = ds.rtf_amp
    # Select frequency range
    rtf_cs_evd_amp = rtf_cs_evd_amp.sel(f_rtf=slice(fmin, fmax))

    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(ds.h_index.values - id_rcv))

        rtf_cs_evd_amp_rcv = rtf_cs_evd_amp.sel(h_index=id_rcv).sel(
            replica_id=replica_id_slice
        )
        log_mod = 10 * np.log10(rtf_cs_evd_amp_rcv)
        log_mod.plot(
            x="f_rtf",
            cmap="magma",
            ax=ax_mod[i_ax],
            vmin=np.percentile(log_mod, 5),
            vmax=np.percentile(log_mod, 95),
            cbar_kwargs={"label": r"$\lvert \hat{\Pi} \rvert$ [dB]"},
        )

        for i_rcv in range(dist_rcv.shape[0]):
            if (reps[obs_cpa_idx[i_rcv]] <= replica_id_slice.stop) and (
                reps[obs_cpa_idx[i_rcv]] >= replica_id_slice.start
            ):
                ax_mod[i_ax].axhline(
                    reps[obs_cpa_idx[i_rcv]],
                    color=color(2 + i_rcv),
                    label=f"CPA OBS{i_rcv+1}",
                    linestyle="--",
                    zorder=10,
                )

        ax_mod[i_ax].set_title(f"OBS {id_rcv}")
        ax_mod[i_ax].set_xlabel("")
        ax_mod[i_ax].set_ylabel("")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    fig.supxlabel("Frequency [Hz]")
    fig.supylabel("Replica ID")

    fpath = os.path.join(fig_folder, "rtf_along_traj_cs_evd.png")
    plt.savefig(fpath)


def plot_log_mod_distribution_along_sequence(
    log_mod_distribution, log_mod_distribution_distance, bin_centers, replica_ids
):
    """
    Plot the distribution of the log-module of the RTF along the sequence, for each OBS. The distance to a reference distribution is also plotted as a function of the replica ID.

    Parameters
    ----------
    log_mod_distribution : dict
        Dictionary containing the distribution of the log-module of the RTF for each OBS. The keys are the OBS IDs and the values are 2D arrays of shape (n_reps, n_bins).
    log_mod_distribution_distance : dict
        Dictionary containing the distance to a reference distribution for each OBS. The keys are the OBS IDs and the values are 1D arrays of shape (n_reps,).

    """

    idx_rcv_plot = list(log_mod_distribution.keys())
    fig, axs = plt.subplots(
        nrows=2, ncols=len(idx_rcv_plot), sharex=False, sharey="row", figsize=(16, 10)
    )
    ax_distribution, ax_distance = axs

    # bin_centers = log_mod_distribution["bin_centers"]
    # reps = log_mod_distribution["replica_id"]
    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:

        im = ax_distribution[i_ax].pcolormesh(
            bin_centers,
            replica_ids,
            log_mod_distribution[id_rcv],
            shading="auto",
            cmap="jet",
            vmin=0,
            vmax=np.percentile(log_mod_distribution[id_rcv], 90),
        )
        plt.colorbar(im, ax=ax_distribution[i_ax], label=r"$\mu$")

        ax_distribution[i_ax].set_xlabel(r"$\lvert \Pi_2(f, r) \rvert$")
        ax_distribution[i_ax].set_ylabel("")
        ax_distribution[i_ax].set_title(f"OBS {id_rcv}")

        # Plot distances
        idist = 0
        for d_name, d_arr in log_mod_distribution_distance.items():
            ax_distance[i_ax].plot(
                d_arr[id_rcv], replica_ids, label=d_name, color=color(idist)
            )
            idist += 1

        # ax_distance[i_ax].plot(wasserstein_dist_arr, reps, label="W", color=color(0))
        # ax_distance[i_ax].plot(jensenshannon_dist_arr, reps, label="JS", color=color(1))
        ax_distance[i_ax].set_xlabel("Distance")
        ax_distance[i_ax].set_ylabel("")
        ax_distance[i_ax].legend()
        ax_distance[i_ax].set_title(f"OBS {id_rcv}")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    # fig.supxlabel(r"$\lvert \Pi(f, r) \rvert$")
    fig.supylabel("Replica ID")

    # fpath = os.path.join(fig_folder, "rtf_along_traj_cs_evd.png")
    # plt.savefig(fpath)


def plot_distance_matrix_comparison_subroutine(
    spatial_dist_mat, dist_mat, cpa_idx=None
):
    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(18, 10), constrained_layout=True
    )
    ax1, ax2 = axs

    # Spatial distance matrix
    vmax = np.nanpercentile(spatial_dist_mat, 75)
    im = ax1.pcolormesh(spatial_dist_mat, cmap="magma", vmax=vmax)
    fig.colorbar(im, ax=ax1, orientation="vertical", label=r"$d$ [m]")
    ax1.set_title(r"$M_{d}$")

    # Data distance matrix
    vmax = np.nanpercentile(dist_mat, 75)
    im = ax2.pcolormesh(dist_mat, cmap="magma", vmax=vmax)
    fig.colorbar(im, ax=ax2, orientation="vertical", label="Shannon entropy S")
    ax2.set_title(r"$M_{S}$")

    # Add CPA indication
    if cpa_idx is not None:
        for ax in axs:
            for i in range(len(cpa_idx)):
                ax.scatter(
                    cpa_idx[i],
                    cpa_idx[i],
                    marker="o",
                    s=150,
                    color=color(i),
                    label=f"OBS{i+1}",
                )

    fig.supxlabel("Replica i")
    fig.supylabel("Replica j")

    return fig, axs


def plot_distance_matrix_comparison(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    cpa_idx=None,
):
    # Plot matrix comparison
    fig, axs = plot_distance_matrix_comparison_subroutine(
        spatial_dist_mat=spatial_dist_mat, dist_mat=dist_mat, cpa_idx=cpa_idx
    )

    # Add legend for CPA position
    ax1, ax2 = axs
    ax1.legend()
    ax2.legend()

    fpath = os.path.join(
        fig_folder,
        f"distance_matrix_comparison_{dist_type}_{rcv_combinaison_strategy}.png",
    )
    plt.savefig(fpath)


def plot_distance_matrix_comparison_line_selected(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    cpa_idx=None,
    selected_lines=None,
):
    # Plot matrix comparison
    fig, axs = plot_distance_matrix_comparison_subroutine(
        spatial_dist_mat=spatial_dist_mat, dist_mat=dist_mat, cpa_idx=cpa_idx
    )

    if selected_lines is None:
        i_ls = np.random.choice(
            np.arange(spatial_dist_mat.shape[0]), size=3, replace=False
        )
    else:
        i_ls = selected_lines

    for ax in axs:
        for k, i_l in enumerate(i_ls):
            ax.axhline(i_l + 0.5, color=color(k), label=rf"$l_{k}$ (j = {i_l})")

    # Add legend for CPA position
    ax1, ax2 = axs
    ax1.legend()
    ax2.legend()

    fpath = os.path.join(
        fig_folder,
        f"distance_matrix_comparison_{dist_type}_{rcv_combinaison_strategy}_selected_lines.png",
    )
    plt.savefig(fpath)


def plot_distance_selected_lines(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    selected_lines=None,
):

    if selected_lines is None:
        i_ls = np.random.choice(
            np.arange(spatial_dist_mat.shape[0]), size=3, replace=False
        )
    else:
        i_ls = selected_lines

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))
    ax1, ax2 = axs

    for k, i_l in enumerate(i_ls):
        ax1.plot(
            spatial_dist_mat[i_l, :],
            marker="+",
            color=color(k),
            label=rf"$l_{k}$ (j = {i_l})",
        )
        ax2.plot(
            dist_mat[i_l, :], marker="+", color=color(k), label=rf"$l_{k}$ (j = {i_l})"
        )

    ax1.set_ylabel(r"$d$ [m]")
    ax2.set_ylabel("S")
    ax1.legend()
    ax2.legend()
    fig.supxlabel("Replica ID")

    fpath = os.path.join(
        fig_folder,
        f"distance_selected_lines_{dist_type}_{rcv_combinaison_strategy}.png",
    )
    plt.savefig(fpath)


def get_combine_distance(distance_dict, combine_method="product"):

    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_replicas = len(distance_dict[d_name_0][idx_rcv_plot[0]])
    if combine_method == "product":
        combined_distance_dict = {
            d_name: np.ones(n_replicas) for d_name in distance_dict.keys()
        }
    elif combine_method == "sum":
        combined_distance_dict = {
            d_name: np.zeros(n_replicas) for d_name in distance_dict.keys()
        }

    for id_rcv in idx_rcv_plot:
        for d_name in distance_dict.keys():
            if combine_method == "product":
                combined_distance_dict[d_name] *= distance_dict[d_name][id_rcv]
            elif combine_method == "sum":
                combined_distance_dict[d_name] += distance_dict[d_name][id_rcv]

    for d_name in combined_distance_dict.keys():
        if combine_method == "product":
            # Smallest distance corresponds to all distance being small, which means that the product of 1/d will be large, and thus 1/product will be small
            # combined_distance_dict[d_name] =
            pass
        elif combine_method == "sum":
            combined_distance_dict[d_name] *= 1 / len(
                idx_rcv_plot
            )  # Average over receivers

        # Normalize to fall into [0, 1] for comparison purpose
        combined_distance_dict[d_name] = normalize(arr=combined_distance_dict[d_name])

    return combined_distance_dict


def plot_dist_and_combined_dist(
    distance_dict, product_distance_dict, sum_distance_dict, replica_ids
):

    def plot_distance(ax, reps, distance, label, color, marker=None):
        ax.plot(reps, distance, label=label, color=color, marker=marker)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend()

    # Setup
    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_axes = len(idx_rcv_plot) + 2

    fig, axs = plt.subplots(
        nrows=1,
        ncols=n_axes,
        sharex=False,
        sharey="row",
        figsize=(16, 10),
    )

    i_ax = 0
    # --- Individual OBS plots ---
    for id_rcv in idx_rcv_plot:

        for i_d, d_name in enumerate(distance_dict.keys()):
            plot_distance(
                axs[i_ax],
                replica_ids,
                distance_dict[d_name][id_rcv],
                d_name,
                color(i_d),
                marker="+",
            )
        axs[i_ax].set_title(f"OBS {id_rcv}")

        i_ax += 1

    # Sum distances
    for i_d, d_name in enumerate(sum_distance_dict.keys()):
        plot_distance(
            axs[i_ax],
            replica_ids,
            sum_distance_dict[d_name],
            d_name,
            color(i_d),
            marker="+",
        )
    axs[i_ax].set_title(" + ".join([f"OBS {id_rcv}" for id_rcv in idx_rcv_plot]))

    i_ax += 1
    # Product distances
    for i_d, d_name in enumerate(product_distance_dict.keys()):
        plot_distance(
            axs[i_ax],
            replica_ids,
            product_distance_dict[d_name],
            d_name,
            color(i_d),
            marker="+",
        )
    axs[i_ax].set_title(" x ".join([f"OBS {id_rcv}" for id_rcv in idx_rcv_plot]))

    # --- Global formatting ---
    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.99, ha="left", va="top"
    )

    fig.supylabel("Distance")
    fig.supxlabel("Replica ID")
    plt.yscale("log")


# ======================================================================================================================
# Miscellaneous routines
# ======================================================================================================================
def get_dist_to_rcv(ds):
    e = ds["e_replica"].values
    n = ds["n_replica"].values

    rep_pos = np.column_stack((e, n))
    dist_to_rcv = []
    for i_rcv in ds.h_index.values:
        e_rcv = ds.attrs[f"obs{i_rcv}_e_apriori"]
        n_rcv = ds.attrs[f"obs{i_rcv}_n_apriori"]
        rcv_pos = np.column_stack((e_rcv, n_rcv))

        spatial_dist = cdist(rep_pos, rcv_pos, metric="euclidean")
        dist_to_rcv.append(spatial_dist)

    return np.array(dist_to_rcv)


def get_distribution_arr(rtf_module, n_bins=50, kde=False):
    log_mod = 10 * np.log10(rtf_module)

    # Derive histogram
    # Define a common range of values for the histograms
    log_mod_min = np.percentile(log_mod.values, 0.5)
    log_mod_max = np.percentile(log_mod.values, 99.5)

    reps = rtf_module.replica_id.values

    log_mod_hist_arr = np.zeros((len(reps), n_bins))
    for irep, rep in enumerate(reps):
        log_mod_r = log_mod.sel(replica_id=rep)
        log_mod_r_hist, bin_edges = np.histogram(
            log_mod_r.values,
            bins=n_bins,
            range=(log_mod_min, log_mod_max),
            density=True,
        )

        log_mod_hist_arr[irep, :] = log_mod_r_hist

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if kde:

        log_mod_kde_arr = np.zeros((len(reps), n_bins))

        for irep, rep in enumerate(reps):
            log_mod_r = log_mod.sel(replica_id=rep)
            # Ensure a common grid of values for the KDE
            log_mod_r_sorted = np.sort(log_mod_r.values)
            log_mod_r_idx_min = np.where(log_mod_r_sorted >= log_mod_min)[0][0]
            log_mod_r_idx_max = np.where(log_mod_r_sorted <= log_mod_max)[0][-1]
            log_mod_r = log_mod_r_sorted[log_mod_r_idx_min : log_mod_r_idx_max + 1]

            # Apply Gaussian KDE
            kde = gaussian_kde(log_mod_r)
            log_mod_kde_arr[irep, :] = kde.pdf(bin_centers)
    else:
        log_mod_kde_arr = None

    return log_mod_hist_arr, bin_centers, log_mod_kde_arr


def get_distribution_distance(
    log_mod_distribution, idx_rep_ref, distance=["shannon_entropy"]
):

    # Reference distribution to compare with
    mu_ref = log_mod_distribution[idx_rep_ref, :]

    nreps = log_mod_distribution.shape[0]
    distance_dict = {d_name: np.zeros(nreps) for d_name in distance}

    for irep in range(nreps):
        mu_r = log_mod_distribution[irep, :]

        if "wasserstein" in distance:
            wd = wasserstein_distance(u_values=mu_ref, v_values=mu_r)
            distance_dict["wasserstein"][irep] = wd

        if "jensen_shannon" in distance:
            js = jensenshannon(mu_ref, mu_r)
            distance_dict["jensen_shannon"][irep] = js

        if "shannon_entropy" in distance:
            mu_ref_s = mu_ref.copy()
            mu_ref_s[mu_ref_s == 0] = np.nan
            mu_r_s = mu_r.copy()
            mu_r_s[mu_r_s == 0] = np.nan
            s = entropy(mu_ref_s, mu_r_s, nan_policy="omit")
            distance_dict["shannon_entropy"][irep] = s

        # kld = kl_div(mu_ref, mu_r)

    return distance_dict


def get_distribution_distance_all_rcv(
    log_mod_distribution, idx_rep_ref, distances=["jensen_shannon"]
):

    distance_dict = {d_name: {} for d_name in distances}

    for id_rcv, log_mod_distribution_rcv in log_mod_distribution.items():
        distance_dict_id_rcv = get_distribution_distance(
            log_mod_distribution=log_mod_distribution_rcv,
            idx_rep_ref=idx_rep_ref,
            distance=distances,
        )
        for d_name in distances:
            d = distance_dict_id_rcv[d_name]
            d_norm = normalize(arr=d)
            # d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d))
            distance_dict[d_name][id_rcv] = d_norm

    return distance_dict


def get_log_module_distributions(ds, fmin, fmax, n_bins=50):

    idx_rcv_plot = [idx for idx in ds.h_index.values if idx != ds.reference_receiver_id]

    log_mod_hist = {}
    log_mod_kde = {}
    for id_rcv in idx_rcv_plot:
        rtf_cs_evd_rcv = ds.rtf_amp.sel(h_index=id_rcv).sel(f_rtf=slice(fmin, fmax))

        log_mod_hist_arr, bin_centers, log_mod_kde_arr = get_distribution_arr(
            rtf_module=rtf_cs_evd_rcv, n_bins=n_bins, kde=True
        )

        log_mod_hist[id_rcv] = log_mod_hist_arr
        log_mod_kde[id_rcv] = log_mod_kde_arr

    return (log_mod_hist, log_mod_kde, bin_centers)


def get_bootstrap_dist_matrix(
    ds,
    distances=["wasserstein", "jensen_shannon", "shannon_entropy"],
    input="histogram",
    fmin=400,
    fmax=800,
    n_bins=100,
):

    log_mod_hist, log_mod_kde, bin_centers = get_log_module_distributions(
        ds, fmin=fmin, fmax=fmax, n_bins=n_bins
    )

    # Derive distributions distance matrix
    dist_mat_prod = {d_name: [] for d_name in distances}
    dist_mat_sum = {d_name: [] for d_name in distances}
    for i, id in enumerate(ds.replica_id.values):
        # Histogram
        if input == "histogram":
            distance_dict_hist = get_distribution_distance_all_rcv(
                log_mod_distribution=log_mod_hist,
                idx_rep_ref=id,
                distances=distances,
            )
            product_distance_dict = get_combine_distance(
                distance_dict=distance_dict_hist, combine_method="product"
            )
            sum_distance_dict = get_combine_distance(
                distance_dict=distance_dict_hist, combine_method="sum"
            )

        # KDE
        elif input == "kde":
            distance_dict_kde = get_distribution_distance_all_rcv(
                log_mod_distribution=log_mod_kde,
                idx_rep_ref=id,
                distances=distances,
            )
            product_distance_dict = get_combine_distance(
                distance_dict=distance_dict_kde, combine_method="product"
            )
            sum_distance_dict = get_combine_distance(
                distance_dict=distance_dict_kde, combine_method="sum"
            )

        else:
            raise ValueError(f"Unknown input {input}, must be 'histogram' or 'kde'.")

        for d_name in distances:
            dist_mat_prod[d_name].append(product_distance_dict[d_name])
            dist_mat_sum[d_name].append(sum_distance_dict[d_name])

    # Convert to np array
    for d_name in distances:
        dist_mat_prod[d_name] = np.array(dist_mat_prod[d_name])
        dist_mat_sum[d_name] = np.array(dist_mat_sum[d_name])

    # Normalize distance matrix to fall into [0, 1] for comparison purpose
    for d_name in distances:
        dist_mat_prod[d_name] = normalize(arr=dist_mat_prod[d_name])
        dist_mat_sum[d_name] = normalize(arr=dist_mat_sum[d_name])

    return dist_mat_prod, dist_mat_sum


def build_dist_matrix(ds):
    e = ds["e_replica"].values
    n = ds["n_replica"].values

    rep_pos = np.column_stack((e, n))
    spatial_dist = cdist(rep_pos, rep_pos, metric="euclidean")

    return spatial_dist


def normalize(arr):
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))


def get_combine_distance(distance_dict, combine_method="product"):

    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_replicas = len(distance_dict[d_name_0][idx_rcv_plot[0]])
    if combine_method == "product":
        combined_distance_dict = {
            d_name: np.ones(n_replicas) for d_name in distance_dict.keys()
        }
    elif combine_method == "sum":
        combined_distance_dict = {
            d_name: np.zeros(n_replicas) for d_name in distance_dict.keys()
        }

    for id_rcv in idx_rcv_plot:
        for d_name in distance_dict.keys():
            if combine_method == "product":
                combined_distance_dict[d_name] *= distance_dict[d_name][id_rcv]
            elif combine_method == "sum":
                combined_distance_dict[d_name] += distance_dict[d_name][id_rcv]

    for d_name in combined_distance_dict.keys():
        if combine_method == "product":
            # Smallest distance corresponds to all distance being small, which means that the product of 1/d will be large, and thus 1/product will be small
            # combined_distance_dict[d_name] =
            pass
        elif combine_method == "sum":
            combined_distance_dict[d_name] *= 1 / len(
                idx_rcv_plot
            )  # Average over receivers

        # Normalize to fall into [0, 1] for comparison purpose
        combined_distance_dict[d_name] = normalize(arr=combined_distance_dict[d_name])

    return combined_distance_dict


if __name__ == "__main__":
    pass
