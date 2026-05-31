#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_plot_utils.py
@Time    :   2026/05/18 13:34:15
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import timedelta, datetime
from scipy.spatial.distance import cdist

from propa.rtf.rtf_utils import D_hermitian_angle_fast
from publication.publication_figure import set_subfigures_abc_labels, color


def plot_sequence_spectrogram(
    ds_wav,
    sequence_start_dt,
    sequence_end_dt,
    savefig=False,
    fig_folder=None,
    nperseg=2**12,
    noverlap=2**11,
    fmin=200,
    fmax=900,
    fname="spectro_3obs",
):
    """
    Plot the spectrogram of the signal recorded by each OBS during the sequence.
    Parameters
    ----------
    ds_wav : xarray.Dataset
        Dataset containing the raw signal recorded by each OBS.
    fig_folder : str
        Root directory to save the figure.
    """
    # datetime_fmt = ds_sig.attrs["datetime_format"]
    # t_start = ds_sig.attrs[f"start_datetime"]
    # t_start = datetime.strptime(t_start, datetime_fmt)
    # t_end = t_start + timedelta(seconds=int(np.ceil(np.max(ds_sig.time.values))))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    sxx_plot = []

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
        t_from_t0_start_s = (sequence_start_dt - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (sequence_end_dt - t0).total_seconds()
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

        sxx_plot.append(sxx)

    # Plot
    cmap = "magma"
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    for i, obs_id in enumerate([1, 2, 3]):
        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(f"OBS{obs_id}")
        axs[i].set_ylim([fmin, fmax])

    clabel = r"dB re 1$\mu$Pa$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    fig.supylabel("Frequency [Hz]")
    fig.supxlabel("Time (UTC)")
    # fig.suptitle(f"OBS{obs_id}")

    set_subfigures_abc_labels(
        axs=axs,
        fontsize=14,
        x_pos=0.015,
        y_pos=1.02,
        ha="left",
        va="top",
    )

    if savefig:
        if fig_folder is not None:
            fpath = os.path.join(fig_folder, f"{fname}.png")
            plt.savefig(fpath, bbox_inches="tight")
        else:
            warnings.warn(
                "No folder selected to save figure (fig_folder = None) while setting savefig to True. Figure will not be saved ! "
            )


def plot_mfp_datasets(ds_library, ds_event, root_img: str = None):

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True
    else:
        save_fig = False

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot receiver positions
    keys = ["obs1", "obs2", "obs3"]
    for k in keys:
        e = ds_library.attrs[f"{k}_e_apriori"]
        n = ds_library.attrs[f"{k}_n_apriori"]
        ax.scatter(
            e,
            n,
            marker="D",
            label=k,
            zorder=1,
            s=150,
        )

    # Plot library replicas positions
    e_library = ds_library["e_replica"].values
    n_library = ds_library["n_replica"].values
    im_lib = ax.scatter(
        e_library,
        n_library,
        marker="+",
        label=f"{ds_library.type.capitalize()} ({ds_library.id})",
        c=np.arange(e_library.size),
        cmap="managua",
    )

    # Plot event replicas positions
    e_event = ds_event["e_replica"].values
    n_event = ds_event["n_replica"].values
    im_event = ax.scatter(
        e_event,
        n_event,
        marker="x",
        label=f"{ds_event.type.capitalize()} ({ds_event.id})",
        c=np.arange(e_event.size),
        cmap="vanimo",
    )

    # Add colorbars
    plt.colorbar(im_lib, label="Library replica index")
    plt.colorbar(im_event, label="Event replica index")

    plt.legend(fontsize=12)
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"{ds_library.type}_{ds_library.id}_and_{ds_event.type}_{ds_event.id}_positions.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_mfp_dataset(ds, cmap="jet", root_img: str = None):

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True
    else:
        save_fig = False

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot receiver positions
    keys = ["obs1", "obs2", "obs3"]
    for k in keys:
        e = ds.attrs[f"{k}_e_apriori"]
        n = ds.attrs[f"{k}_n_apriori"]
        ax.scatter(
            e,
            n,
            marker="D",
            label=k,
            zorder=1,
            s=150,
        )

    # Plot replicas positions
    e_library = ds["e_replica"].values
    n_library = ds["n_replica"].values
    im = ax.scatter(
        e_library,
        n_library,
        marker="+",
        label=f"{ds.type.capitalize()} ({ds.id})",
        c=np.arange(e_library.size),
        cmap=cmap,
    )
    plt.colorbar(im, label="Replica index")

    plt.legend(fontsize=12)
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    if save_fig:
        fpath = os.path.join(root_img, f"{ds.type}_{ds.id}_positions.png")
        plt.savefig(fpath, bbox_inches="tight")


def plot_results_dist(ds_results, ds_event, ds_library, root_img: str = None):

    print(f"\tPlotting RTF distance vs spatial distance")

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True
    else:
        save_fig = False

    # Find crossing point between library replicas and event replicas in the spatial domain
    min_dist_ids = ds_results.spatial_dist.argmin(...)
    min_dist_event_replica_id = min_dist_ids["event_replica_id"].values
    min_dist_library_replica_id = min_dist_ids["library_replica_id"].values
    min_dist_results = ds_results.isel(
        event_replica_id=min_dist_event_replica_id,
        library_replica_id=min_dist_library_replica_id,
    )

    # Find minimum dist in rtf space
    min_rtf_dist_ids = ds_results.rtf_dist.argmin(...)
    min_rtf_dist_event_replica_id = min_rtf_dist_ids["event_replica_id"].values
    min_rtf_dist_library_replica_id = min_rtf_dist_ids["library_replica_id"].values
    min_rtf_dist_results = ds_results.isel(
        event_replica_id=min_rtf_dist_event_replica_id,
        library_replica_id=min_rtf_dist_library_replica_id,
    )
    # min_rtf_dist_event = ds_event.sel(replica_id=min_rtf_dist_event_replica_id)
    # min_rtf_dist_library = ds_library.sel(replica_id=min_rtf_dist_library_replica_id)

    # Get CPA of lib and event traj for each receiver
    e_lib = ds_library["e_replica"].values
    n_lib = ds_library["n_replica"].values
    e_event = ds_event["e_replica"].values
    n_event = ds_event["n_replica"].values

    lib_pos = np.column_stack((e_lib, n_lib))
    event_pos = np.column_stack((e_event, n_event))

    # lib_dist_to_rcv = []
    cpa_idx_lib = []
    # event_dist_to_rcv = []
    cpa_idx_event = []

    for i_rcv in ds_library.h_index.values:
        e_rcv = ds_library.attrs[f"obs{i_rcv}_e_apriori"]
        n_rcv = ds_library.attrs[f"obs{i_rcv}_n_apriori"]
        rcv_pos = np.column_stack((e_rcv, n_rcv))

        spatial_dist_lib = cdist(lib_pos, rcv_pos, metric="euclidean")
        cpa_rcv_idx_lib = np.nanargmin(spatial_dist_lib)
        cpa_idx_lib.append(cpa_rcv_idx_lib)

        spatial_dist_event = cdist(event_pos, rcv_pos, metric="euclidean")
        cpa_rcv_idx_event = np.nanargmin(spatial_dist_event)
        cpa_idx_event.append(cpa_rcv_idx_event)

        # dist_to_rcv.append(spatial_dist)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # Plot CPAs
    for i, i_rcv in enumerate(ds_library.h_index.values):
        axs[0].axhline(cpa_idx_lib[i], linestyle="--", color=color(i))
        axs[0].axvline(
            cpa_idx_event[i], linestyle="--", color=color(i), label=f"CPA OBS{i_rcv}"
        )
        # axs[0].scatter()

    # Define colorbar limits
    vmin = np.percentile(ds_results.rtf_dist.values, 0.1)
    vmax = np.percentile(ds_results.rtf_dist.values, 50)
    # vmin = np.percentile(ds_results.rtf_dist.values, 0.5)
    # vmax = np.percentile(ds_results.rtf_dist.values, 90)

    # Theta distance
    ds_results.rtf_dist.plot(
        x="event_replica_id",
        y="library_replica_id",
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        ax=axs[0],
    )
    axs[0].set_xlabel("")

    # Add marker at minimum spatial distance point
    axs[0].scatter(
        min_dist_results.event_replica_id,
        min_dist_results.library_replica_id,
        marker="X",
        s=80,
        color="cyan",
        label="Minimum spatial distance",
        zorder=5,
    )

    # Add marker at minimum rtf distance point
    axs[0].scatter(
        min_rtf_dist_results.event_replica_id,
        min_rtf_dist_results.library_replica_id,
        marker="o",
        s=80,
        color="cyan",
        label="Minimum theta distance",
        zorder=5,
    )

    # Spatial distance
    ds_results.spatial_dist.plot(
        x="event_replica_id",
        y="library_replica_id",
        cmap="magma",
        vmin=0,
        vmax=250,
        ax=axs[1],
    )

    # Add marker at minimum spatial distance point
    axs[1].scatter(
        min_dist_results.event_replica_id,
        min_dist_results.library_replica_id,
        marker="X",
        s=80,
        color="cyan",
        label="Minimum spatial distance",
        zorder=5,
    )
    # Add marker at minimum rtf distance point
    axs[1].scatter(
        min_rtf_dist_results.event_replica_id,
        min_rtf_dist_results.library_replica_id,
        marker="o",
        s=80,
        color="cyan",
        label="Minimum theta distance",
        zorder=5,
    )

    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_distances.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_results_sorted_dist(
    ds_results: xr.Dataset, offset_around_min_dist: int = 2, root_img: str = None
):

    print(f"\tPlotting RTF distance vs spatial distance (sorted by distance)")

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True
    else:
        save_fig = False

    # Find crossing point between library replicas and event replicas in the spatial domain
    min_dist_ids = ds_results.spatial_dist.argmin(...)
    min_dist_event_replica_id = min_dist_ids["event_replica_id"].values
    min_dist_library_replica_id = min_dist_ids["library_replica_id"].values
    min_dist_results = ds_results.isel(
        event_replica_id=min_dist_event_replica_id,
        library_replica_id=min_dist_library_replica_id,
    )

    # Compare RTF distance and spatial distance for a few replicas around the minimum spatial distance point
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # Extract rtf distance for a few replicas around the minimum spatial distance point
    min_rep = max(0, min_dist_results.library_replica_id - offset_around_min_dist)
    max_rep = min(
        ds_results.library_replica_id.max().values,
        min_dist_results.library_replica_id + offset_around_min_dist,
    )
    ds_results_around_min_dist = ds_results.sel(
        library_replica_id=slice(min_rep, max_rep)
    )

    ds_results_around_min_dist.rtf_dist.plot(ax=axs[0], hue="library_replica_id")
    ds_results_around_min_dist.spatial_dist.plot(ax=axs[1], hue="library_replica_id")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_var_around_min_dist.png",
        )
        plt.savefig(fpath, bbox_inches="tight")

    # Extract theta and dist variation for the selected replica
    ds_results_min_dist = ds_results.sel(
        library_replica_id=min_dist_results.library_replica_id.values
    )
    dist_to_cpa_argsort = np.argsort(ds_results_min_dist.spatial_dist.values)
    sorted_spatial_dist = ds_results_min_dist.spatial_dist.values[dist_to_cpa_argsort]
    sorted_rtf_dist = ds_results_min_dist.rtf_dist.values[dist_to_cpa_argsort]

    plt.figure()
    plt.scatter(
        sorted_spatial_dist,
        sorted_rtf_dist,
        # label=f"{pos} ({theta_dist_obs[pos]['id']})",
    )
    plt.xlabel("Spatial distance to closest replica [m]")
    plt.ylabel(r"$\theta$ [°]")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_sorted_var_at_min_dist.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_features(
    ds_results: xr.Dataset,
    ds_library: xr.Dataset,
    ds_event: xr.Dataset,
    root_img: str = None,
    plot_module: bool = True,
    plot_phase: bool = True,
    plot_theta: bool = True,
):

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True

    # Find crossing point between library replicas and event replicas in the spatial domain -> main lobe should be here
    min_dist_ids = ds_results.spatial_dist.argmin(...)
    min_dist_event_replica_id = min_dist_ids["event_replica_id"].values
    min_dist_event = ds_event.sel(replica_id=min_dist_event_replica_id)
    # min_dist_p1_event = ds_event.sel(replica_id=min_dist_event_replica_id + 1)
    min_dist_library_replica_id = min_dist_ids["library_replica_id"].values
    min_dist_library = ds_library.sel(replica_id=min_dist_library_replica_id)
    # min_dist_p1_library = ds_library.sel(replica_id=min_dist_library_replica_id + 1)

    # Find minimum dist in rtf space
    min_rtf_dist_ids = ds_results.rtf_dist.argmin(...)
    min_rtf_dist_event_replica_id = min_rtf_dist_ids["event_replica_id"].values
    min_rtf_dist_event = ds_event.sel(replica_id=min_rtf_dist_event_replica_id)
    min_rtf_dist_library_replica_id = min_rtf_dist_ids["library_replica_id"].values
    min_rtf_dist_library = ds_library.sel(replica_id=min_rtf_dist_library_replica_id)

    fmin = ds_results.fmin
    fmax = ds_results.fmax

    n_rcv = ds_library.h_index.size
    # Compare module
    if plot_module:
        fig, axs = plt.subplots(
            nrows=n_rcv, ncols=2, sharex=True, sharey="row", figsize=(16, 3 * n_rcv)
        )
        ax_mod_dist, ax_mod_rtf_dist = axs[:, 0], axs[:, 1]

        for i_rcv in range(n_rcv):
            h_plot = i_rcv + 1

            # Module
            # Event minimum spatial distance
            min_dist_event.rtf_amp.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            ).plot(ax=ax_mod_dist[i_rcv], label="event (d = d_min)", color=color(0))
            # Library minimum spatial distance
            min_dist_library.rtf_amp.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            ).plot(ax=ax_mod_dist[i_rcv], label="library (d = d_min)", color=color(1))

            # ax_mod_dist[i_rcv].set_title(
            #     f"theta = {theta_along_rcv_at_min_spatial_dist:.1f}°, d_euc = {euc_dist_at_min_spatial_dist:.1f}, d_euc_mod = {euc_mod_dist_at_min_spatial_dist:.1f} "
            # )

            dist_name = ds_results.rtf_dist.attrs["long_name"]
            dist_unit = ds_results.rtf_dist.attrs["units"]
            ax_mod_dist[i_rcv].set_title(
                f"{dist_name} = {ds_results.rtf_dist.sel(event_replica_id=min_dist_event_replica_id, library_replica_id=min_dist_library_replica_id):.2f} {dist_unit}"
            )

            # Event minimum rtf distance
            min_rtf_dist_event.rtf_amp.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            ).plot(
                ax=ax_mod_rtf_dist[i_rcv],
                label="event (d_rtf = d_rtf_min)",
                color=color(2),
            )
            # Library minimum rtf distance
            min_rtf_dist_library.rtf_amp.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            ).plot(
                ax=ax_mod_rtf_dist[i_rcv],
                label="library (d_rtf = d_rtf_min)",
                color=color(3),
            )
            # ax_mod_rtf_dist[i_rcv].set_title(
            #     f"theta = {theta_along_rcv_at_min_theta_dist:.1f}°, d_euc = {euc_dist_at_min_theta_dist:.1f}, d_euc_mod = {euc_mod_dist_at_min_theta_dist:.1f} "
            # )
            ax_mod_rtf_dist[i_rcv].set_title(
                f"{dist_name} = {ds_results.rtf_dist.sel(event_replica_id=min_rtf_dist_event_replica_id, library_replica_id=min_rtf_dist_library_replica_id):.2f} {dist_unit}"
            )
            # ax_mod_dist[i_rcv].set_ylim([0.01, 100])
            ax_mod_dist[i_rcv].set_yscale("log")
            ax_mod_dist[i_rcv].set_xlabel("")
            ax_mod_dist[i_rcv].set_ylabel(r"$|\Pi|$")
            # ax_mod_dist[i_rcv].set_ylabel("")
            # ax_mod_dist[i_rcv].set_title("")
            ax_mod_dist[i_rcv].legend(fontsize=10, ncol=3, loc="lower right")

            # ax_mod_rtf_dist[i_rcv].set_ylim([0.01, 100])
            ax_mod_rtf_dist[i_rcv].set_yscale("log")
            ax_mod_rtf_dist[i_rcv].set_xlabel("")
            ax_mod_rtf_dist[i_rcv].set_ylabel(r"$|\Pi|$")
            # ax_mod_rtf_dist[i_rcv].set_ylabel("")
            # ax_mod_rtf_dist[i_rcv].set_title("")
            ax_mod_rtf_dist[i_rcv].legend(fontsize=10, ncol=3, loc="lower right")

        set_subfigures_abc_labels(
            axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
        )

        fig.supxlabel("Frequency [Hz]")

        if save_fig:
            fpath = os.path.join(
                root_img,
                f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_feature_comparison_mod.png",
            )
            plt.savefig(fpath, bbox_inches="tight")

    # Compare unwrapped phase
    if plot_phase:
        fig, axs = plt.subplots(
            nrows=n_rcv, ncols=2, sharex=True, figsize=(16, 3 * n_rcv)
        )
        ax_phase_dist, ax_phase_rtf_dist = axs[:, 0], axs[:, 1]

        for i_rcv in range(n_rcv):
            h_plot = i_rcv + 1

            # Unwrapped phase
            # Event minimum spatial distance
            rtf_event_dist = min_dist_event.rtf_phase.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            )
            rtf_event_dist_unwrap = np.unwrap(rtf_event_dist)
            ax_phase_dist[i_rcv].plot(
                rtf_event_dist.f_rtf,
                rtf_event_dist_unwrap,
                color=color(0),
                label=r"event (d = d_min)",
            )
            # Library minimum spatial distance
            rtf_lib_dist = min_dist_library.rtf_phase.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            )
            rtf_lib_dist_unwrap = np.unwrap(rtf_lib_dist)
            ax_phase_dist[i_rcv].plot(
                rtf_lib_dist.f_rtf,
                rtf_lib_dist_unwrap,
                color=color(1),
                label=r"library (d = d_min)",
                # marker="o",
                # linewidth=1,
                # markersize=3,
                # alpha=0.7,
            )
            # Event minimum rtf distance
            rtf_event_rtf_dist = min_rtf_dist_event.rtf_phase.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            )
            rtf_event_rtf_dist_unwrap = np.unwrap(rtf_event_rtf_dist)
            ax_phase_rtf_dist[i_rcv].plot(
                rtf_event_rtf_dist.f_rtf,
                rtf_event_rtf_dist_unwrap,
                color=color(2),
                label=r"event (d_rtf = d_rtf_min)",
            )

            # Library minimum rtf distance
            rtf_lib_rtf_dist = min_rtf_dist_library.rtf_phase.sel(h_index=h_plot).sel(
                f_rtf=slice(fmin, fmax)
            )
            rtf_lib_rtf_dist_unwrap = np.unwrap(rtf_lib_rtf_dist)
            ax_phase_rtf_dist[i_rcv].plot(
                rtf_lib_rtf_dist.f_rtf,
                rtf_lib_rtf_dist_unwrap,
                color=color(3),
                label=r"library (d_rtf = d_rtf_min)",
                # marker="o",
                # linewidth=1,
                # markersize=3,
                # alpha=0.7,
            )

            ax_phase_dist[i_rcv].set_xlabel("")
            ax_phase_dist[i_rcv].set_ylabel(r"$\Phi$")
            ax_phase_dist[i_rcv].set_title("")
            # ax_phase_dist[i_rcv].set_ylim([-30, 30])
            ax_phase_dist[i_rcv].legend(fontsize=10, ncol=3, loc="lower right")

            ax_phase_rtf_dist[i_rcv].set_xlabel("")
            ax_phase_rtf_dist[i_rcv].set_ylabel(r"$\Phi$")
            ax_phase_rtf_dist[i_rcv].set_title("")
            # ax_phase_rtf_dist[i_rcv].set_ylim([-30, 30])
            ax_phase_rtf_dist[i_rcv].legend(fontsize=10, ncol=3, loc="lower right")

        set_subfigures_abc_labels(
            axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
        )

        fig.supxlabel("Frequency [Hz]")

        if save_fig:
            fpath = os.path.join(
                root_img,
                f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_feature_comparison_phase.png",
            )
            plt.savefig(fpath, bbox_inches="tight")

    # Hermitian angle vs frequency
    if plot_theta:
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 8))

        rtf_1_mod = min_dist_event.rtf_amp.sel(f_rtf=slice(fmin, fmax))
        rtf_2_mod = min_dist_library.rtf_amp.sel(f_rtf=slice(fmin, fmax))
        rtf_1_phase = min_dist_event.rtf_phase.sel(f_rtf=slice(fmin, fmax))
        rtf_2_phase = min_dist_library.rtf_phase.sel(f_rtf=slice(fmin, fmax))
        rtf_1 = rtf_1_mod * np.exp(1j * rtf_1_phase)
        rtf_2 = rtf_2_mod * np.exp(1j * rtf_2_phase)
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": False}
        theta_along_rcv_at_min_spatial_dist = D_hermitian_angle_fast(
            rtf_ref=rtf_1.values,
            rtf=rtf_2.values,
            **dist_kwargs,
        )

        rtf_1_mod = min_rtf_dist_event.rtf_amp.sel(f_rtf=slice(fmin, fmax))
        rtf_2_mod = min_rtf_dist_event.rtf_amp.sel(f_rtf=slice(fmin, fmax))
        rtf_1_phase = min_rtf_dist_event.rtf_phase.sel(f_rtf=slice(fmin, fmax))
        rtf_2_phase = min_rtf_dist_library.rtf_phase.sel(f_rtf=slice(fmin, fmax))
        rtf_1 = rtf_1_mod * np.exp(1j * rtf_1_phase)
        rtf_2 = rtf_2_mod * np.exp(1j * rtf_2_phase)
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": False}
        theta_along_rcv_at_min_theta_dist = D_hermitian_angle_fast(
            rtf_ref=rtf_1.values,
            rtf=rtf_2.values,
            **dist_kwargs,
        )

        axs[0].plot(
            rtf_1.f_rtf.values,
            theta_along_rcv_at_min_spatial_dist,
            label="d = d_min",
            color=color(0),
        )
        axs[1].plot(
            rtf_2.f_rtf.values,
            theta_along_rcv_at_min_theta_dist,
            label="theta = theta_min",
            color=color(2),
        )

        axs[0].axhline(
            np.percentile(theta_along_rcv_at_min_spatial_dist, 25),
            label="25 th percentile",
            color=color(4),
        )
        axs[0].axhline(
            np.percentile(theta_along_rcv_at_min_spatial_dist, 50),
            label="50 th percentile",
            color=color(5),
        )
        axs[0].axhline(
            np.percentile(theta_along_rcv_at_min_spatial_dist, 75),
            label="75 th percentile",
            color=color(6),
        )
        axs[0].axhline(
            np.mean(theta_along_rcv_at_min_spatial_dist),
            label="mean",
            color=color(7),
        )

        axs[1].axhline(
            np.percentile(theta_along_rcv_at_min_theta_dist, 25),
            label="25 th percentile",
            color=color(4),
        )
        axs[1].axhline(
            np.percentile(theta_along_rcv_at_min_theta_dist, 50),
            label="50 th percentile",
            color=color(5),
        )
        axs[1].axhline(
            np.percentile(theta_along_rcv_at_min_theta_dist, 75),
            label="75 th percentile",
            color=color(6),
        )
        axs[1].axhline(
            np.mean(theta_along_rcv_at_min_theta_dist),
            label="mean",
            color=color(7),
        )
        axs[0].legend()
        axs[1].legend()

        set_subfigures_abc_labels(
            axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
        )

        fig.supxlabel("Frequency [Hz]")
        fig.supylabel(
            f"{ds_results.rtf_dist.attrs['long_name']} [{ds_results.rtf_dist.attrs['units']}]"
        )

        if save_fig:
            fpath = os.path.join(
                root_img,
                f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_feature_comparison_d_rtf_vs_freq.png",
            )
            plt.savefig(fpath, bbox_inches="tight")


if __name__ == "__main__":
    pass
