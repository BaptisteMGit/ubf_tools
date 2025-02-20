import os
import io
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

plt.switch_backend("agg")

import scipy.signal as signal
import moviepy.editor as mpy

from PIL import Image
from cst import (
    LIBRARY_COLOR,
    EVENT_COLOR,
    LABEL_FONTSIZE,
    TICKS_FONTSIZE,
    TITLE_FONTSIZE,
    LEGEND_FONTSIZE,
    SUPLABEL_FONTSIZE,
)
from misc import confidence_ellipse

from pyproj import Geod
from propa.kraken_toolbox.plot_utils import plotshd
from localisation.verlinden.verlinden_analysis_report import (
    plot_localisation_performance,
)

from localisation.verlinden.misc.plot_utils import plot_localisation_moviepy
from localisation.verlinden.misc.params import (
    VERLINDEN_OUTPUT_FOLDER,
    VERLINDEN_ANALYSIS_FOLDER,
)

from publication.PublicationFigure import PubFigure

PFIG = PubFigure(
    label_fontsize=40,
    title_fontsize=40,
    ticks_fontsize=40,
    legend_fontsize=30,
)
PFIG.set_all_params()


def plot_emmited_signal(xr_dataset, img_root):
    """
    Plot emmited signal for each source position.

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        Dataset containing the emmited signal.
    img_root : str
        Path to the folder where the images will be saved.

    Returns
    -------
    None

    """
    # Init folders
    img_folder = os.path.join(img_root, "emmited_signals")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    def build_label_detail(xr_dataset, src):
        detail_label = ""
        for p, val_p in xr_dataset[src].attrs.items():
            if p not in ["short_name", "long_name", "sig_type"]:
                if p == "fs":
                    p_tag = r"$f_s$"
                elif p == "fc":
                    p_tag = r"$f_c$"
                elif p == "std_fi":
                    p_tag = r"$\sigma_{fi}$"
                elif p == "tau_corr_fi":
                    p_tag = r"$\tau_{corr_{fi}}$"
                else:
                    p_tag = p
                detail_label += f"{p_tag}={val_p}, "
        return detail_label[:-2]

    # Plot time series
    plt.figure()
    for src in ["library_src", "event_src"]:
        label = xr_dataset[src].attrs["short_name"]
        detail_label = build_label_detail(xr_dataset, src)
        label += f"({detail_label})"
        xr_dataset[src].plot(label=label)

    plt.ylabel("Amplitude")
    plt.legend()
    img_fpath = os.path.join(img_folder, "emmited_signals.png")
    plt.savefig(img_fpath)
    plt.close()

    # Plot spectrogram
    # for src in ["library_src", "event_src"]:
    #     plt.figure()
    #     label = xr_dataset[src].attrs["long_name"]
    #     f, t, Sxx = signal.spectrogram(
    #         xr_dataset[src].values,
    #         fs=xr_dataset[src].attrs["fs"],
    #         window="hamming",
    #         # nperseg=256,
    #         # noverlap=int(0.75 * 256),
    #     )
    #     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    #     plt.ylabel("Frequency [Hz]")
    #     plt.xlabel("Time [s]")
    #     plt.title(f"Spectrogram of {label}")
    #     plt.colorbar(label="Intensity [dB]")
    #     # plt.tight_layout()
    #     img_fpath = os.path.join(img_folder, f"spectrogram_{src}.png")
    #     plt.savefig(img_fpath)
    #     plt.close()

    # Plot PSD
    plt.figure()
    for src in ["library_src", "event_src"]:
        label = xr_dataset[src].attrs["short_name"]
        detail_label = build_label_detail(xr_dataset, src)
        label += f"({detail_label})"
        f, Pxx = signal.welch(xr_dataset[src].values, fs=xr_dataset[src].attrs["fs"])
        # plt.semilogy(f, Pxx, label=label)
        plt.plot(f, 10 * np.log(Pxx), label=label)

    plt.ylabel(r"$S_{xx}(f)$")
    plt.xlabel("Frequency [Hz]")
    plt.title(f"PSD")
    # plt.tight_layout()
    plt.legend()
    img_fpath = os.path.join(img_folder, f"psd.png")
    plt.savefig(img_fpath)
    plt.close()

    # Plot auto-correlation
    plt.figure()
    for src in ["library_src", "event_src"]:
        label = xr_dataset[src].attrs["short_name"]
        detail_label = build_label_detail(xr_dataset, src)
        label += f"({detail_label})"
        s = xr_dataset[src].values
        ns = len(s)
        Rxx = signal.correlate(s, s)
        tau_idx = signal.correlation_lags(ns, ns)

        Ts = 1 / xr_dataset[src].attrs["fs"]
        tau = tau_idx * Ts

        plt.plot(tau, Rxx, label=label)

    plt.ylabel(r"$R_{xx}(\tau)$")
    plt.xlabel(r"$\tau \, [s]$")
    plt.title(f"Auto-correlation")
    # plt.tight_layout()
    plt.legend()
    img_fpath = os.path.join(img_folder, f"auto_correlation.png")
    plt.savefig(img_fpath)
    plt.close()

    # # Plot auto-correlation psd
    # plt.figure()
    # for src in ["library_src", "event_src"]:
    #     label = xr_dataset[src].attrs["short_name"]
    #     detail_label = build_label_detail(xr_dataset, src)
    #     label += f"({detail_label})"
    #     s = xr_dataset[src].values
    #     ns = len(s)
    #     Rxx = signal.correlate(s, s)
    #     tau = signal.correlation_lags(ns, ns)
    #     f, Pxx = signal.welch(Rxx, fs=xr_dataset[src].attrs["fs"])
    #     plt.plot(f, 10 * np.log(Pxx), label=label)
    # plt.ylabel(r"$S_{xx}(\tau)$")
    # plt.xlabel("Frequency [Hz]")
    # plt.title(f"Auto-correlation PSD")
    # # plt.tight_layout()
    # plt.legend()
    # img_fpath = os.path.join(img_folder, f"auto_correlation_psd.png")
    # plt.savefig(img_fpath)
    # plt.close()


def plot_received_signal(xr_dataset, img_root, n_instant_to_plot=None):
    """Plot received signal for each receiver pair and each source position."""
    # Init folders
    img_folder = os.path.join(img_root, "received_signals")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    if n_instant_to_plot is None:
        n_instant_to_plot = xr_dataset.dims["src_trajectory_time"]
    else:
        n_instant_to_plot = min(
            n_instant_to_plot, xr_dataset.dims["src_trajectory_time"]
        )

    for i_ship in range(n_instant_to_plot):
        fig, axes = plt.subplots(
            xr_dataset.dims["idx_rcv"],
            1,
            sharex=True,
        )

        for i_rcv in range(xr_dataset.dims["idx_rcv"]):
            lib_sig = xr_dataset.rcv_signal_library.sel(
                lon=xr_dataset.lon_src.isel(src_trajectory_time=i_ship),
                lat=xr_dataset.lat_src.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_rcv=i_rcv)
            event_sig = xr_dataset.rcv_signal_event.isel(
                src_trajectory_time=i_ship, idx_rcv=i_rcv
            )
            # Plot the signal with the smallest std on top
            if event_sig.std() <= lib_sig.std():
                lib_zorder = 1
                event_zorder = 2
            else:
                event_zorder = 1
                lib_zorder = 2

            lib_sig.plot(
                ax=axes[i_rcv],
                label=f"library",
                color=LIBRARY_COLOR,
                zorder=lib_zorder,
            )
            event_sig.plot(
                ax=axes[i_rcv],
                label=f"event",
                color=EVENT_COLOR,
                zorder=event_zorder,
            )

            axes[i_rcv].set_xlabel("")
            axes[i_rcv].set_ylabel("")
            axes[i_rcv].set_title("")
            axes[i_rcv].legend(loc="upper right")

        for ax, col in zip(axes, [f"Receiver {i}" for i in xr_dataset.idx_rcv.values]):
            ax.set_title(col, loc="right")

        fig.supylabel("Received signal")
        fig.supxlabel("Time [s]")
        img_fpath = os.path.join(img_folder, f"received_signals_time_{i_ship}.png")
        plt.savefig(img_fpath)
        plt.close()

    # Plot auto-correlation
    for i_ship in range(n_instant_to_plot):
        fig, axes = plt.subplots(
            xr_dataset.dims["idx_rcv"],
            1,
            sharex=True,
        )

        for i_rcv in range(xr_dataset.dims["idx_rcv"]):
            lib_sig = xr_dataset.rcv_signal_library.sel(
                lon=xr_dataset.lon_src.isel(src_trajectory_time=i_ship),
                lat=xr_dataset.lat_src.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_rcv=i_rcv)
            event_sig = xr_dataset.rcv_signal_event.isel(
                src_trajectory_time=i_ship, idx_rcv=i_rcv
            )
            # plt.tight_layout()

            lib_corr = signal.correlate(lib_sig.values, lib_sig.values)
            event_corr = signal.correlate(event_sig.values, event_sig.values)
            tau_idx = signal.correlation_lags(len(lib_sig), len(lib_sig))
            Ts = xr_dataset.library_signal_time.diff(dim="library_signal_time").values[
                0
            ]
            tau = tau_idx * Ts

            axes[i_rcv].plot(tau, lib_corr, label="library", color=LIBRARY_COLOR)
            axes[i_rcv].plot(tau, event_corr, label="event", color=EVENT_COLOR)
            axes[i_rcv].set_xlabel("")
            axes[i_rcv].set_ylabel("")
            axes[i_rcv].legend(loc="upper right")
            # axes[i_rcv].tick_params(labelsize=TICKS_FONTSIZE)

        for ax, col in zip(axes, [f"Receiver {i}" for i in xr_dataset.idx_rcv.values]):
            ax.set_title(col, loc="right")

        fig.supylabel(r"$R_{xx}(\tau)$")
        fig.supxlabel(r"$\tau$ [s]")
        # # plt.tight_layout()
        img_fpath = os.path.join(img_folder, f"auto_correlation_time_{i_ship}.png")
        plt.savefig(img_fpath)
        # plt.show()
        plt.close()


def get_ambiguity_surface(ds):
    # Avoid singularity for S = 0
    amb_surf_not_0 = ds.ambiguity_surface.values[ds.ambiguity_surface > 0]
    ds.ambiguity_surface.values[ds.ambiguity_surface == 0] = amb_surf_not_0.min()
    amb_surf = 10 * np.log10(ds.ambiguity_surface)  # dB scale

    amb_surf_combined_not_0 = ds.ambiguity_surface_combined.values[
        ds.ambiguity_surface_combined > 0
    ]
    ds.ambiguity_surface_combined.values[ds.ambiguity_surface_combined == 0] = (
        amb_surf_combined_not_0.min()
    )
    amb_surf_combined = 10 * np.log10(ds.ambiguity_surface_combined)  # dB scale

    return amb_surf, amb_surf_combined


def init_plot_folders(img_root, var_to_plot, similarity_metric):
    img_folder = os.path.join(img_root, var_to_plot, str(similarity_metric))
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    return img_folder


def plot_ambiguity_surface_dist(ds, img_root, n_instant_to_plot=1):
    """Plot ambiguity surface distribution."""
    # Init folers
    img_folder = init_plot_folders(
        img_root, "ambiguity_surface_dist", ds.similarity_metric.values
    )

    amb_surf, __ = get_ambiguity_surface(ds)

    # Limit nb of plots
    n_instant_to_plot = min(n_instant_to_plot, 10)

    # Hist
    for i_src_time in range(n_instant_to_plot):
        for i_rcv_pair in range(ds.dims["idx_rcv_pairs"]):

            # Plot hist
            amb_surf_i = amb_surf.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            )
            bins = int(np.ceil(np.abs(amb_surf_i).max()) * amb_surf_i.size // 100)
            plt.figure(figsize=(16, 10))
            amb_surf_i.plot.hist(bins=bins)
            plt.axvline(
                amb_surf_i.max(),
                color="red",
                linestyle="--",
                label=f"Max = {amb_surf_i.max().round(1).values}dB",
            )
            plt.ylabel("Number of points", fontsize=LABEL_FONTSIZE)
            plt.xlabel("Ambiguity surface [dB]", fontsize=LABEL_FONTSIZE)
            plt.yticks(fontsize=TICKS_FONTSIZE)
            plt.xticks(fontsize=TICKS_FONTSIZE)
            plt.legend(loc="best")

            img_fpath = os.path.join(
                img_folder,
                f"ambiguity_surface_dist_time_{i_src_time}_pair_{i_rcv_pair}.png",
            )
            plt.savefig(img_fpath)
            plt.close()

            # Cumulative hist
            amb_surf_i = amb_surf.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            )
            bins = int(np.ceil(np.abs(amb_surf_i).max()) * amb_surf_i.size // 100)
            plt.figure(figsize=(16, 10))
            amb_surf_i.plot.hist(bins=bins, cumulative=True, density=True)
            plt.ylabel("Number of points", fontsize=LABEL_FONTSIZE)
            plt.xlabel("Ambiguity surface [dB]", fontsize=LABEL_FONTSIZE)
            plt.yticks(fontsize=TICKS_FONTSIZE)
            plt.xticks(fontsize=TICKS_FONTSIZE)
            img_fpath = os.path.join(
                img_folder,
                f"ambiguity_surface_cumul_dist_tim_{i_src_time}_pair_{i_rcv_pair}.png",
            )
            plt.savefig(img_fpath)
            plt.close()

    for i_rcv_pair in range(ds.dims["idx_rcv_pairs"]):
        plt.figure(figsize=(16, 10))
        bins_all = int(np.ceil(np.abs(amb_surf).max()) * amb_surf.size // 100)
        amb_surf.isel(idx_rcv_pairs=i_rcv_pair).plot.hist(bins=bins_all)
        plt.axvline(
            amb_surf.max(),
            color="red",
            linestyle="--",
            label=f"Max = {amb_surf.max().round(1).values}dB",
        )
        plt.ylabel("Number of points", fontsize=LABEL_FONTSIZE)
        plt.xlabel("Ambiguity surface [dB]", fontsize=LABEL_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.legend(loc="best")
        img_fpath = os.path.join(
            img_folder, f"ambiguity_surface_dist_allpos_pair_{i_rcv_pair}.png"
        )
        plt.savefig(img_fpath)
        plt.close()

        # Cumulative hist
        plt.figure(figsize=(16, 10))
        amb_surf.isel(idx_rcv_pairs=i_rcv_pair).plot.hist(
            bins=bins_all, cumulative=True, density=True
        )
        plt.ylabel("Number of points", fontsize=LABEL_FONTSIZE)
        plt.xlabel("Ambiguity surface [dB]", fontsize=LABEL_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        img_fpath = os.path.join(
            img_folder,
            f"ambiguity_surface_cumul_dist_allpos_pair_{i_rcv_pair}.png",
        )
        plt.savefig(img_fpath)
        plt.close()


def get_grid_arrays(ds, grid_info={}):
    grid_x = np.arange(
        -grid_info["Lx"] / 2,
        grid_info["Lx"] / 2,
        grid_info["dx"],
        dtype=np.float32,
    )
    grid_y = np.arange(
        -grid_info["Ly"] / 2,
        grid_info["Ly"] / 2,
        grid_info["dy"],
        dtype=np.float32,
    )
    xx, yy = np.meshgrid(grid_x, grid_y)

    rr_obs = np.array(
        [
            np.sqrt(
                (xx - ds.lon_rcv.sel(idx_rcv=i_rcv).values) ** 2
                + (yy - ds.lat_rcv.sel(idx_rcv=i_rcv).values) ** 2
            )
            for i_rcv in range(ds.dims["idx_rcv"])
        ]
    )

    delta_rr = rr_obs[0, ...] - rr_obs[1, ...]
    delta_rr_ship = ds.r_obs_ship.sel(idx_rcv=0) - ds.r_obs_ship.sel(idx_rcv=1)

    return xx, yy, delta_rr, delta_rr_ship


def plot_ambiguity_surface(
    ds,
    img_root,
    nb_instant_to_plot=10,
    grid_info={},
    plot_info={},
    zoom=False,
    plot_hyperbol=False,
):
    """Plot ambiguity surface for each source position."""
    # Init folders
    sim_metric = ds.similarity_metric.values
    img_folder = init_plot_folders(img_root, "ambiguity_surface", sim_metric)

    amb_surf, amb_surf_combined = get_ambiguity_surface(ds)

    for i_src_time in range(nb_instant_to_plot):
        for i_rcv_pair in ds["idx_rcv_pairs"].values:
            rcv_pair = ds["rcv_pairs"].sel(idx_rcv_pairs=i_rcv_pair).values
            pair_id = ds.rcv_pair_id.isel(idx_rcv_pairs=i_rcv_pair).values

            plt.figure()

            vmin = (
                amb_surf.isel(idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time)
                .quantile(0.35)
                .values
            )
            vmax = amb_surf.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            ).max()
            amb_surf.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            ).plot(x="lon", y="lat", zorder=0, vmin=vmin, vmax=vmax, cmap="jet")

            # ax = plt.gca()
            # for item in (
            #     [ax.xaxis.label, ax.yaxis.label]
            #     + ax.get_xticklabels()
            #     + ax.get_yticklabels()
            # ):
            #     item.set_fontsize(20)

            # if plot_beampattern:
            #     # bp_offset = np.sqrt(
            #     #     (ds.lon_src.isel(src_trajectory_time=i) - x_center_array) ** 2
            #     #     + (ds.lat_src.isel(src_trajectory_time=i) - y_center_array) ** 2
            #     # )
            #     # log_bp = 10 * np.log10(linear_bp)
            #     # bp_alpha = -bp_offset / min(log_bp)
            #     # log_bp = log_bp * bp_alpha.values + bp_offset.values
            #     # x_bp = log_bp * np.cos(theta) + x_center_array
            #     # y_bp = log_bp * np.sin(theta) + y_center_array
            #     # plt.plot(x_bp, y_bp, "k", linestyle="--", label="Beampattern", zorder=1)

            #     # plt.gca().get_images()[0].clim([0, 20])
            #     # ds.ambiguity_surface.isel(idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i).plot(
            #     #     x="x", y="y", zorder=0, clim=[-1, 1]
            #     # )

            if plot_hyperbol:
                xx, yy, delta_rr, delta_rr_ship = get_grid_arrays(ds, grid_info)

                condition = (
                    np.abs(
                        delta_rr
                        - delta_rr_ship.isel(src_trajectory_time=i_src_time).values
                    )
                    < 0.01
                )
                plt.plot(
                    xx[condition],
                    yy[condition],
                    "k",
                    linestyle="--",
                    label=r"$ || \overrightarrow{O_0M} || - || \overrightarrow{O_1M} || =  \Delta_i|| \overrightarrow{O_iX_{ship}} ||$",
                    zorder=1,
                )

            plt.scatter(
                ds.lon_src.isel(src_trajectory_time=i_src_time),
                ds.lat_src.isel(src_trajectory_time=i_src_time),
                # color="magenta",
                facecolors="none",
                edgecolors="magenta",
                # fillstyle="none",
                marker="o",
                s=130,
                linewidths=2.5,
                label=r"$X_{ship}$",
                zorder=2,
            )

            det_pos_lon = ds.detected_pos_lon.isel(
                idx_rcv_pairs=i_rcv_pair,
                src_trajectory_time=i_src_time,
                idx_noise_realisation=-1,
            )  # Ambiguity surface si saved for last noise realisation
            det_pos_lat = ds.detected_pos_lat.isel(
                idx_rcv_pairs=i_rcv_pair,
                src_trajectory_time=i_src_time,
                idx_noise_realisation=-1,
            )
            plt.scatter(
                det_pos_lon,
                det_pos_lat,
                marker="s",
                facecolors="none",
                edgecolors="magenta",
                s=200,
                linewidths=2.5,
                label="Estimated position",
                zorder=3,
            )

            for i_rcv in ds["idx_rcv"].values:
                if i_rcv in rcv_pair:
                    size = 130
                else:
                    size = 50
                plt.scatter(
                    ds.lon_rcv.isel(idx_rcv=i_rcv),
                    ds.lat_rcv.isel(idx_rcv=i_rcv),
                    marker="o",
                    label=ds.rcv_id.isel(idx_rcv=i_rcv).values,
                    s=size,
                )

            # Plot line between rcv and display dist between rcv
            lon_rcv = ds.lon_rcv.sel(idx_rcv=rcv_pair).values
            lat_rcv = ds.lat_rcv.sel(idx_rcv=rcv_pair).values
            dist = np.round(
                Geod(ellps="WGS84").inv(lon_rcv[0], lat_rcv[0], lon_rcv[1], lat_rcv[1])[
                    2
                ],
                0,
            )
            plt.plot(
                lon_rcv,
                lat_rcv,
                "k",
                linestyle="--",
                zorder=1,
                label=f"L = {dist}m",
            )

            # Plot line between the center of the rcv array and the ship
            lon_rcv_center = lon_rcv.mean()
            lat_rcv_center = lat_rcv.mean()
            dist_center = np.round(
                Geod(ellps="WGS84").inv(
                    lon_rcv_center,
                    lat_rcv_center,
                    ds.lon_src.isel(src_trajectory_time=i_src_time),
                    ds.lat_src.isel(src_trajectory_time=i_src_time),
                )[2],
                0,
            )
            plt.plot(
                [lon_rcv_center, ds.lon_src.isel(src_trajectory_time=i_src_time)],
                [lat_rcv_center, ds.lat_src.isel(src_trajectory_time=i_src_time)],
                "k",
                linestyle="-.",
                zorder=1,
                label=r"$r_{ship}$" + f" = {dist_center}m",
            )
            # Add point at the center of the array
            plt.scatter(
                lon_rcv_center,
                lat_rcv_center,
                marker="o",
                color="black",
                s=10,
            )

            if zoom:
                plt.xlim([ds.lon.min(), ds.lon.max()])
                plt.ylim([ds.lat.min(), ds.lat.max()])
                zoom_label = "_zoom"
            else:
                plt.xlim(
                    [
                        min(ds.lon.min(), ds.lon_rcv.min()) - plot_info["lon_offset"],
                        max(ds.lon.max(), ds.lon_rcv.max()) + plot_info["lon_offset"],
                    ]
                )
                plt.ylim(
                    [
                        min(ds.lat.min(), ds.lat_rcv.min()) - plot_info["lat_offset"],
                        max(ds.lat.max(), ds.lat_rcv.max()) + plot_info["lat_offset"],
                    ]
                )
                zoom_label = ""
            plt.title(
                f"Ambiguity surface\n(similarity metric: {sim_metric}, src pos n°{i_src_time}, rcv pair {pair_id})",
            )
            plt.legend(ncol=2)
            img_fpath = os.path.join(
                img_folder,
                f"ambiguity_surface_time_{i_src_time}_pair_{i_rcv_pair}{zoom_label}.png",
            )
            plt.savefig(img_fpath)
            plt.close()

    """Combined ambiguity surface """
    for i_src_time in range(nb_instant_to_plot):
        plt.figure()

        vmin = (
            amb_surf_combined.isel(src_trajectory_time=i_src_time).quantile(0.35).values
        )
        vmax = amb_surf_combined.isel(src_trajectory_time=i_src_time).max()
        amb_surf_combined.isel(src_trajectory_time=i_src_time).plot(
            x="lon", y="lat", zorder=0, vmin=vmin, vmax=vmax, cmap="jet"
        )

        plt.scatter(
            ds.lon_src.isel(src_trajectory_time=i_src_time),
            ds.lat_src.isel(src_trajectory_time=i_src_time),
            # color="magenta",
            facecolors="none",
            edgecolors="magenta",
            # fillstyle="none",
            marker="o",
            s=130,
            linewidths=2.5,
            label=r"$X_{ship}$",
            zorder=2,
        )

        det_pos_lon = ds.detected_pos_lon_combined.isel(
            src_trajectory_time=i_src_time,
            idx_noise_realisation=-1,
        )  # Ambiguity surface si saved for last noise realisation
        det_pos_lat = ds.detected_pos_lat_combined.isel(
            src_trajectory_time=i_src_time,
            idx_noise_realisation=-1,
        )
        plt.scatter(
            det_pos_lon,
            det_pos_lat,
            marker="s",
            facecolors="none",
            edgecolors="magenta",
            s=200,
            linewidths=2.5,
            label="Estimated position",
            zorder=3,
        )

        # Plot line between the center of the rcv array and the ship
        lon_rcv_center = ds.lon_rcv.mean().values
        lat_rcv_center = ds.lat_rcv.mean().values
        dist_center = np.round(
            Geod(ellps="WGS84").inv(
                lon_rcv_center,
                lat_rcv_center,
                ds.lon_src.isel(src_trajectory_time=i_src_time),
                ds.lat_src.isel(src_trajectory_time=i_src_time),
            )[2],
            0,
        )
        plt.plot(
            [lon_rcv_center, ds.lon_src.isel(src_trajectory_time=i_src_time)],
            [lat_rcv_center, ds.lat_src.isel(src_trajectory_time=i_src_time)],
            "k",
            linestyle="-.",
            zorder=1,
            label=r"$r_{ship}$" + f" = {dist_center}m",
        )
        # Add point at the center of the array
        plt.scatter(
            lon_rcv_center,
            lat_rcv_center,
            marker="o",
            color="black",
            s=10,
        )

        for i_rcv in ds["idx_rcv"].values:
            plt.scatter(
                ds.lon_rcv.isel(idx_rcv=i_rcv),
                ds.lat_rcv.isel(idx_rcv=i_rcv),
                marker="o",
                label=ds.rcv_id.isel(idx_rcv=i_rcv).values,
                s=50,
            )
        if zoom:
            plt.xlim([ds.lon.min(), ds.lon.max()])
            plt.ylim([ds.lat.min(), ds.lat.max()])
            zoom_label = "_zoom"
        else:
            plt.xlim(
                [
                    min(ds.lon.min(), ds.lon_rcv.min()) - plot_info["lon_offset"],
                    max(ds.lon.max(), ds.lon_rcv.max()) + plot_info["lon_offset"],
                ]
            )
            plt.ylim(
                [
                    min(ds.lat.min(), ds.lat_rcv.min()) - plot_info["lat_offset"],
                    max(ds.lat.max(), ds.lat_rcv.max()) + plot_info["lat_offset"],
                ]
            )
            zoom_label = ""

        plt.title(
            f"Ambiguity surface\n(similarity metric: {sim_metric}, src pos n°{i_src_time})",
        )
        plt.legend(ncol=2)
        img_fpath = os.path.join(
            img_folder, f"ambiguity_surface_combined_time_{i_src_time}{zoom_label}.png"
        )
        plt.savefig(img_fpath)
        plt.close()


def plot_localisation_moviepy(
    ds,
    nb_frames,
    anim_filename,
    plot_hyperbol=False,
    grid_info={},
    fps_sec=30,
    **kwargs,
):
    """Plot and save localisation animations using MoviePy."""

    figsize = (12, 8)
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    obs_col = ["orange", "magenta", "green"]
    amb_surf, __ = get_ambiguity_surface(ds)

    # Function to update the plot for each frame and return NumPy array
    def animate_func(i_src_time):
        legend = []
        # Plot ambiguity surface
        img = amb_surf.isel(src_trajectory_time=i_src_time).plot(
            ax=ax, add_colorbar=False, **kwargs
        )

        # True source position
        plt.scatter(
            ds.lon_src.isel(src_trajectory_time=i_src_time),
            ds.lat_src.isel(src_trajectory_time=i_src_time),
            color="k",
            marker="+",
            s=90,
            label=r"$X_{ship}$",
        )
        legend.append(r"$X_{ship}$")

        # Estimated source position
        plt.scatter(
            ds.detected_pos_lon.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            ),
            ds.detected_pos_lat.isel(
                idx_rcv_pairs=i_rcv_pair, src_trajectory_time=i_src_time
            ),
            marker="o",
            facecolors="none",
            edgecolors="black",
            s=120,
            linewidths=2.2,
        )
        legend.append(r"$\tilde{X_{ship}}$")

        # Equal delta distance hyperbol
        if plot_hyperbol:
            xx, yy, delta_rr, delta_rr_ship = get_grid_arrays(ds, grid_info)

            condition = (
                np.abs(
                    delta_rr - delta_rr_ship.isel(src_trajectory_time=i_src_time).values
                )
                < 0.1
            )

            plt.plot(
                xx[condition],
                yy[condition],
                "k",
                linestyle="--",
                zorder=1,
            )
            legend.append(
                r"$ || \overrightarrow{O_0M} || - || \overrightarrow{O_1M} || =  \Delta_i|| \overrightarrow{O_iX_{ship}} ||$"
            )

        # Obs location
        for i_rcv in range(ds.dims["idx_rcv"]):
            plt.scatter(
                ds.lon_rcv.isel(idx_rcv=i_rcv),
                ds.lat_rcv.isel(idx_rcv=i_rcv),
                marker="o",
                color=obs_col[i_rcv],
                label=f"$O_{i_rcv}$",
            )

        # plt.xlim(
        #     [
        #         min(ds.lon.min(), ds.lon_rcv.min())
        #         - x_offset,
        #         max(ds.lon.max(), ds.lon_rcv.max())
        #         + x_offset,
        #     ]
        # )
        # plt.ylim(
        #     [
        #         min(ds.lat.min(), ds.lat_rcv.min())
        #         - y_offset,
        #         max(ds.lat.max(), ds.lat_rcv.max())
        #         + y_offset,
        #     ]
        # )

        plt.legend(legend, loc="upper right")

        # Create a PIL Image from the Matplotlib figure
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_pil = Image.open(buf)

        # Convert the PIL Image to a NumPy array
        img_array = np.array(img_pil)

        return img_array

    # Create a list of NumPy arrays for each frame
    frames = [animate_func(i) for i in range(nb_frames)]

    # Create the MoviePy ImageSequenceClip from the list of NumPy arrays
    animation_clip = mpy.ImageSequenceClip(frames, fps=fps_sec)

    # Save the animation as an MP4 video file
    animation_clip.write_videofile(anim_filename)

    # Close the figure to avoid memory leaks
    plt.close(fig)


def plot_ship_trajectory(
    ds, img_root, plot_info={}, noise_realisation_to_plot=0, zoom=False
):
    """Plot ship trajectory."""
    # Init folders
    img_folder = init_plot_folders(
        img_root, "ship_trajectory", ds.similarity_metric.values
    )

    # List of colors for each rcv pairs
    rcv_pair_colors = ["blue", "magenta", "green"]
    """ Plot single detection for a single noise realisation """
    for i_noise in range(noise_realisation_to_plot):

        for i_rcv_pair in ds.idx_rcv_pairs.values:
            pair_id = ds.rcv_pair_id.isel(idx_rcv_pairs=i_rcv_pair).values

            plt.figure()  # figsize=(16, 8)
            plt.plot(
                ds.lon_src,
                ds.lat_src,
                marker="o",
                color="red",
                markersize=6,
                zorder=6,
                label=r"$X_{ref}$",
            )

            # Plot rcv positions
            for i_rcv in range(ds.dims["idx_rcv"]):
                plt.scatter(
                    ds.lon_rcv.isel(idx_rcv=i_rcv),
                    ds.lat_rcv.isel(idx_rcv=i_rcv),
                    marker="o",
                    s=100,
                    zorder=2,
                    label=ds.rcv_id.isel(idx_rcv=i_rcv).values,
                )

            det_pos_lon = ds.detected_pos_lon.isel(
                idx_rcv_pairs=i_rcv_pair,
                idx_noise_realisation=i_noise,
            )
            det_pos_lat = ds.detected_pos_lat.isel(
                idx_rcv_pairs=i_rcv_pair,
                idx_noise_realisation=i_noise,
            )
            plt.scatter(
                det_pos_lon,
                det_pos_lat,
                marker="+",
                color=rcv_pair_colors[i_rcv_pair],
                s=200,
                zorder=1,
                linewidths=2.2,
                label=r"$X_{loc}$ " + f" - {pair_id}",
            )

            if zoom:
                plt.xlim([ds.lon.min(), ds.lon.max()])
                plt.ylim([ds.lat.min(), ds.lat.max()])
                zoom_label = "_zoom"
            else:
                plt.xlim(
                    [
                        min(ds.lon.min(), ds.lon_rcv.min()) - plot_info["lon_offset"],
                        max(ds.lon.max(), ds.lon_rcv.max()) + plot_info["lon_offset"],
                    ]
                )
                plt.ylim(
                    [
                        min(ds.lat.min(), ds.lat_rcv.min()) - plot_info["lat_offset"],
                        max(ds.lat.max(), ds.lat_rcv.max()) + plot_info["lat_offset"],
                    ]
                )
                zoom_label = ""

            plt.xlabel("Longitude [°]")
            plt.ylabel("Latitude [°]")
            plt.grid(True)
            plt.legend()
            img_fpath = os.path.join(
                img_folder, f"ship_trajectory_noise_{i_noise}_{pair_id}{zoom_label}.png"
            )
            plt.savefig(img_fpath)
            plt.close()

    """ Plot detected positions for all noise realisations """
    for i_rcv_pair in ds.idx_rcv_pairs.values:
        pair_id = str(ds.rcv_pair_id.isel(idx_rcv_pairs=i_rcv_pair).values)

        plt.figure()  # figsize=(16, 8)
        plt.plot(
            ds.lon_src,
            ds.lat_src,
            marker="o",
            color="red",
            markersize=6,
            zorder=6,
            label=r"$X_{ref}$",
        )

        # Plot rcv positions
        for i_rcv in range(ds.sizes["idx_rcv"]):
            plt.scatter(
                ds.lon_rcv.isel(idx_rcv=i_rcv),
                ds.lat_rcv.isel(idx_rcv=i_rcv),
                marker="o",
                s=100,
                zorder=4,
                label=ds.rcv_id.isel(idx_rcv=i_rcv).values,
            )

        det_pos_lon = ds.detected_pos_lon.isel(
            idx_rcv_pairs=i_rcv_pair, src_trajectory_time=0
        )
        det_pos_lat = ds.detected_pos_lat.isel(
            idx_rcv_pairs=i_rcv_pair, src_trajectory_time=0
        )
        plt.scatter(
            det_pos_lon,
            det_pos_lat,
            marker=".",
            color=rcv_pair_colors[i_rcv_pair],
            s=35,
            alpha=0.3,
            # linewidths=2.2,
            zorder=5,
            label=r"$X_{" + pair_id + r"}$",
        )

        plt.scatter(
            det_pos_lon.mean(),
            det_pos_lat.mean(),
            marker="+",
            color=rcv_pair_colors[i_rcv_pair],
            s=200,
            linewidths=2.2,
            zorder=5,
            label=r"$\hat{X_{" + pair_id + r"}}$",
        )

        ax = plt.gca()
        confidence_ellipse(
            det_pos_lon,
            det_pos_lat,
            ax,
            n_std=3,
            edgecolor=rcv_pair_colors[i_rcv_pair],
            facecolor=rcv_pair_colors[i_rcv_pair],
            alpha=0.2,
            zorder=2,
            label=r"$3\sigma$" + " confidence ellipse",
        )

        if zoom:
            plt.xlim([ds.lon.min(), ds.lon.max()])
            plt.ylim([ds.lat.min(), ds.lat.max()])
            zoom_label = "_zoom"
        else:
            plt.xlim(
                [
                    min(ds.lon.min(), ds.lon_rcv.min()) - plot_info["lon_offset"],
                    max(ds.lon.max(), ds.lon_rcv.max()) + plot_info["lon_offset"],
                ]
            )
            plt.ylim(
                [
                    min(ds.lat.min(), ds.lat_rcv.min()) - plot_info["lat_offset"],
                    max(ds.lat.max(), ds.lat_rcv.max()) + plot_info["lat_offset"],
                ]
            )
            zoom_label = ""

        plt.xlabel("Longitude [°]")
        plt.ylabel("Latitude [°]")
        plt.grid(True)
        plt.legend()
        img_fpath = os.path.join(
            img_folder, f"ship_trajectory_pos_0_alldet_{pair_id}{zoom_label}.png"
        )
        plt.savefig(img_fpath)
        plt.close()

    """ Plot detected positions for all noise realisations and all pairs """
    plt.figure()  # figsize=(16, 8)
    plt.plot(
        ds.lon_src,
        ds.lat_src,
        marker="o",
        color="red",
        markersize=6,
        zorder=6,
        label=r"$X_{ref}$",
    )
    # Plot rcv positions
    for i_rcv in range(ds.sizes["idx_rcv"]):
        plt.scatter(
            ds.lon_rcv.isel(idx_rcv=i_rcv),
            ds.lat_rcv.isel(idx_rcv=i_rcv),
            marker="o",
            s=100,
            zorder=4,
            label=ds.rcv_id.isel(idx_rcv=i_rcv).values,
        )

    for i_rcv_pair in ds.idx_rcv_pairs.values:
        pair_id = str(ds.rcv_pair_id.isel(idx_rcv_pairs=i_rcv_pair).values)

        det_pos_lon = ds.detected_pos_lon.isel(
            idx_rcv_pairs=i_rcv_pair, src_trajectory_time=0
        )
        det_pos_lat = ds.detected_pos_lat.isel(
            idx_rcv_pairs=i_rcv_pair, src_trajectory_time=0
        )
        plt.scatter(
            det_pos_lon,
            det_pos_lat,
            marker=".",
            color=rcv_pair_colors[i_rcv_pair],
            s=35,
            alpha=0.3,
            # linewidths=2.2,
            zorder=5,
            label=r"$X_{" + pair_id + r"}$",
        )

        plt.scatter(
            det_pos_lon.mean(),
            det_pos_lat.mean(),
            marker="+",
            color=rcv_pair_colors[i_rcv_pair],
            s=200,
            linewidths=2.2,
            zorder=5,
            label=r"$\hat{X_{" + pair_id + r"}}$",
        )

        ax = plt.gca()
        confidence_ellipse(
            det_pos_lon,
            det_pos_lat,
            ax,
            n_std=3,
            edgecolor=rcv_pair_colors[i_rcv_pair],
            facecolor=rcv_pair_colors[i_rcv_pair],
            # linewidth=5,
            alpha=0.2,
            zorder=2,
            label=r"$\Sigma_{" + pair_id + r"}$",
        )

        if zoom:
            plt.xlim([ds.lon.min(), ds.lon.max()])
            plt.ylim([ds.lat.min(), ds.lat.max()])
            zoom_label = "_zoom"
        else:
            plt.xlim(
                [
                    min(ds.lon.min(), ds.lon_rcv.min()) - plot_info["lon_offset"],
                    max(ds.lon.max(), ds.lon_rcv.max()) + plot_info["lon_offset"],
                ]
            )
            plt.ylim(
                [
                    min(ds.lat.min(), ds.lat_rcv.min()) - plot_info["lat_offset"],
                    max(ds.lat.max(), ds.lat_rcv.max()) + plot_info["lat_offset"],
                ]
            )
            zoom_label = ""

    # Add combined detected positions
    det_pos_lon = ds.detected_pos_lon_combined.isel(src_trajectory_time=0)
    det_pos_lat = ds.detected_pos_lat_combined.isel(src_trajectory_time=0)
    plt.scatter(
        det_pos_lon,
        det_pos_lat,
        marker=".",
        color="black",
        s=35,
        alpha=0.3,
        zorder=10,
        label=r"$X_{combined}$",
    )

    plt.scatter(
        det_pos_lon.mean(),
        det_pos_lat.mean(),
        marker="+",
        color="black",
        s=200,
        linewidths=2.2,
        zorder=10,
        label=r"$\hat{X_{combined}}$",
    )

    ax = plt.gca()
    confidence_ellipse(
        det_pos_lon,
        det_pos_lat,
        ax,
        n_std=3,
        edgecolor="black",
        facecolor="black",
        # linewidth=10,
        alpha=0.2,
        zorder=2,
        label=r"$\Sigma_{combined}$",
    )

    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")
    plt.grid(True)
    plt.legend(ncol=ds.sizes["idx_rcv_pairs"] + 2)
    img_fpath = os.path.join(
        img_folder, f"ship_trajectory_pos_0_alldet_allpairs{zoom_label}.png"
    )
    plt.savefig(img_fpath)
    plt.close()


def get_pos_error(ds):
    """Compute position error between true and estimated position."""
    # Define the geodetic object
    geod = Geod(ellps="WGS84")

    # Broadcast real positions to the shape of the estimated positions
    lat_src = ds.lat_src.broadcast_like(ds.detected_pos_lat)
    lon_src = ds.lon_src.broadcast_like(ds.detected_pos_lon)

    _, _, pos_error = geod.inv(
        lats1=lat_src,
        lons1=lon_src,
        lats2=ds.detected_pos_lat,
        lons2=ds.detected_pos_lon,
    )

    # Convert to xr array
    pos_error = xr.DataArray(
        pos_error,
        dims=["idx_rcv_pairs", "src_trajectory_time", "idx_noise_realisation"],
        coords={
            "idx_rcv_pairs": ds.idx_rcv_pairs,
            "src_trajectory_time": ds.src_trajectory_time,
            "idx_noise_realisation": ds.idx_noise_realisation,
        },
    )
    return pos_error


def get_pos_error_metrics(pos_error):
    pos_error_metrics = {
        "median": pos_error.median().round(2).values,
        "mean": pos_error.mean().round(2).values,
        "std": pos_error.std().round(2).values,
        "max": pos_error.max().round(2).values,
        "min": pos_error.min().round(2).values,
        "rmse": np.sqrt(np.mean(pos_error**2)).round(2).values,
        "95_percentile": np.percentile(pos_error, 95).round(2),
        "99_percentile": np.percentile(pos_error, 99).round(2),
    }
    return pos_error_metrics


def plot_pos_error(ds, img_root):
    """Plot position error."""
    # Init folders
    img_folder = init_plot_folders(img_root, "pos_error", ds.similarity_metric.values)

    # Derive position error
    pos_error = get_pos_error(ds)

    rcv_pair_colors = ["blue", "magenta", "green"]
    plt.figure()
    for i_rcv_pair in ds.idx_rcv_pairs.values:
        position_error = pos_error.sel(idx_rcv_pairs=i_rcv_pair)
        mean_label = (
            r"$\mu_{err}$"
            + f" (rcv pair n°{i_rcv_pair}, "
            + r"$n_{noise}$"
            + f"= {ds.dims['idx_noise_realisation']})"
        )
        std_label = (
            r"$\mu_{err} \pm \sigma_{err}$"
            + f" (rcv pair n°{i_rcv_pair}, "
            + r"$n_{noise}$"
            + f"= {ds.dims['idx_noise_realisation']})"
        )
        if ds.src_trajectory_time.size > 1:
            # Plot mean position error over all noise realisation
            position_error.mean(dim="idx_noise_realisation").plot(
                color=rcv_pair_colors[i_rcv_pair],
                label=mean_label,
            )
            # Add shaded area for std
            plt.fill_between(
                position_error.src_trajectory_time,
                position_error.mean(dim="idx_noise_realisation")
                - position_error.std(dim="idx_noise_realisation"),
                position_error.mean(dim="idx_noise_realisation")
                + position_error.std(dim="idx_noise_realisation"),
                alpha=0.3,
                color=rcv_pair_colors[i_rcv_pair],
                label=std_label,
            )
            plt.ylabel("Position error [m]", fontsize=LABEL_FONTSIZE)

        else:
            # Plot error distribution for the single ship position
            position_error = position_error.isel(src_trajectory_time=0)
            position_error.plot.hist(bins="auto")
            # position_error.plot.hist(bins=100)

            plt.axvline(
                position_error.mean(),
                color="red",
                linestyle="--",
                label=mean_label,
            )
            # for sgn in [-1, 1]:
            # One sided (error > 0)
            plt.axvline(
                position_error.mean() + position_error.std(),
                color="green",
                linestyle="--",
                label=std_label,
            )

            plt.xlabel("Position error [m]")
            plt.ylabel("Number of simulations")

        # pos_error.sel(idx_rcv_pairs=i_rcv_pair).plot(
        #     label=f"Position error for the pair of receivers n°{i_rcv_pair}"
        # )

        # plt.axhline(
        #     pos_error.median(),
        #     color="red",
        #     linestyle="--",
        #     label=f"Median error for obs pair {i_rcv_pair.values} = {pos_error.median().round(0).values}m",
        # )

    # plt.gca().xaxis.label.set_fontsize(LABEL_FONTSIZE)

    plt.legend(loc="upper right")
    plt.title("Position error")
    img_fpath = os.path.join(img_folder, f"pos_error.png")
    plt.savefig(img_fpath)
    plt.close()


def plot_correlation(ds, img_root, det_metric="intercorr0", n_instant_to_plot=1):
    """Plot correlation for receiver couple."""
    # Init folders
    img_folder = init_plot_folders(img_root, "signal_corr", ds.similarity_metric.values)

    for i_ship in range(n_instant_to_plot):
        fig, axes = plt.subplots(
            ds.sizes["idx_rcv_pairs"],
            1,
            sharex=True,
        )

        # fig, axes = plt.subplots(
        #     n_instant_to_plot,
        #     ds.sizes["idx_rcv_pairs"],
        #     sharex=True,
        #     sharey=True,
        #     figsize=(16, 8),
        # )
        if ds.sizes["idx_rcv_pairs"] == 1:
            axes = np.reshape(
                axes, (ds.sizes["idx_rcv_pairs"],)
            )  # Ensure 2D axes array in case of single obs pair

        for i_rcv_pair in range(ds.sizes["idx_rcv_pairs"]):
            event_vect = ds.event_corr.isel(
                src_trajectory_time=i_ship, idx_rcv_pairs=i_rcv_pair
            )
            lib_vect = ds.library_corr.sel(
                lon=ds.lon_src.isel(src_trajectory_time=i_ship),
                lat=ds.lat_src.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_rcv_pairs=i_rcv_pair)

            if det_metric == "hilbert_env_intercorr0":
                event_vect.values = np.abs(signal.hilbert(event_vect))
                lib_vect.values = np.abs(signal.hilbert(lib_vect))
                ylabel = r"$|\mathcal{H}[R_{12}(\tau)]|$"
            else:
                ylabel = r"$R_{12}(\tau)$"

            # Plot the signal with the smallest std on top
            if event_vect.std() <= lib_vect.std():
                lib_zorder = 1
                event_zorder = 2
            else:
                event_zorder = 1
                lib_zorder = 2

            event_vect.plot(
                ax=axes[i_rcv_pair],
                label="event",
                color=EVENT_COLOR,
                zorder=event_zorder,
            )
            lib_vect.plot(
                ax=axes[i_rcv_pair],
                label="lib at ship pos",
                color=LIBRARY_COLOR,
                zorder=lib_zorder,
            )
            axes[i_rcv_pair].set_xlabel("")
            axes[i_rcv_pair].set_ylabel("")
            axes[i_rcv_pair].set_title("")
            axes[i_rcv_pair].legend(loc="upper right")

        for ax, lab in zip(axes, ds.rcv_pair_id.values):
            ax.set_title(lab, loc="right")

        fig.supxlabel(r"$\tau$ [s]")
        fig.supylabel(ylabel)

        # plt.tight_layout()
        img_fpath = os.path.join(img_folder, f"signal_corr_pos{i_ship}.png")
        plt.savefig(img_fpath)
        plt.close()


def check_folder_creation(plot_info):
    return (
        plot_info["plot_one_tl_profile"]
        or plot_info["plot_ambiguity_surface_dist"]
        or plot_info["plot_received_signal"]
        or plot_info["plot_ambiguity_surface"]
        or plot_info["plot_video"]
        or plot_info["plot_ship_trajectory"]
        or plot_info["plot_pos_error"]
        or plot_info["plot_correlation"]
    )


def compare_perf_src(src_type_list, simulation_info, testcase_name, snr=0):

    for i_src, src_type in enumerate(src_type_list):
        # Read global report
        global_report_fpath = os.path.join(
            VERLINDEN_ANALYSIS_FOLDER,
            testcase_name,
            src_type,
            simulation_info["src_pos"],
            "global_report.txt",
        )

        data = pd.read_csv(
            global_report_fpath,
            sep=",",
            dtype={
                "SNR": float,
                "MEDIAN": float,
                "MEAN": float,
                "STD": float,
                "RMSE": float,
                "MAX": float,
                "MIN": float,
                "95_percentile": float,
                "99_percentile": float,
                "dynamic_range": float,
            },
        )
        data["src_type"] = [src_type] * data.shape[0]

        if i_src == 0:
            all_src_data = data
        else:
            all_src_data = pd.concat([all_src_data, data])

    all_src_data = all_src_data[all_src_data["SNR"] == snr]
    """ Plot perf for the different detection metrics """
    bar_width = 0.2

    # Define positions of the bars
    positions = np.arange(len(all_src_data["src_type"].unique()))
    m = len(all_src_data["Detection metric"].unique())
    alpha = np.arange(-(m // 2), (m // 2) + m % 2, 1) + ((m + 1) % 2) / 2

    # Create bar plot
    metrics_to_plot = all_src_data.columns[2:-1]
    similarity_metrics = all_src_data["Detection metric"].unique()
    root_img = r"C:\\Users\\baptiste.menetrier\\Desktop\\devPy\\phd\\img\\localisation\\verlinden_process_analysis"
    img_path = os.path.join(root_img, testcase_name)

    for i_m, metric in enumerate(metrics_to_plot):
        bar_values = np.array([])
        all_pos = np.array([])
        colors = []

        # PFIG.set_all_fontsize()
        plt.figure(figsize=PFIG.size)
        for i_dm, similarity_metric in enumerate(similarity_metrics):
            # for i_src, src_type in enumerate(src_type_list):
            bar = all_src_data[all_src_data["Detection metric"] == similarity_metric][
                metric
            ]
            if not bar.empty:
                b = plt.bar(
                    positions + alpha[i_dm] * bar_width,
                    bar.values,
                    width=bar_width,
                    label=similarity_metric,
                )

                # Add value labels to bars
                # +3 / 4 * bar_width * np.sign(alpha[i_dm])
            pos = (
                positions
                + alpha[i_dm] * bar_width
                + 3 / 4 * bar_width * np.sign(alpha[i_dm])
            )
            all_pos = np.concatenate((all_pos, pos))
            bar_values = np.concatenate((bar_values, bar.values))
            colors += [b.patches[0].get_facecolor()] * len(bar)

        val_offset = 0.03 * max(bar_values)
        for i in range(len(bar_values)):
            plt.text(
                all_pos[i],
                bar_values[i] + val_offset,
                bar_values[i],
                ha="center",
                # color=colors[i],
                bbox=dict(facecolor=colors[i], alpha=0.8),
            )

        plt.legend()
        plt.xticks(positions, all_src_data["src_type"].unique())
        plt.legend(ncol=2)

        title = f"{testcase_name} - performance analysis"
        if metric == "95_percentile":
            ylabel = "Position error 95 % percentile [m]"
        elif metric == "99_percentile":
            ylabel = "Position error 99 % percentile [m]"
        elif metric == "MEDIAN":
            ylabel = "Position median error [m]"
        elif metric == "MIN":
            ylabel = "Position minimum error [m]"
        elif metric == "MAX":
            ylabel = "Position maximum error [m]"
        elif metric == "STD":
            ylabel = "Position std error [m]"
        elif metric == "MEAN":
            ylabel = "Position mean error [m]"
        elif metric == "RMSE":
            ylabel = "Position rmse [m]"
        elif metric == "dynamic_range":
            ylabel = "Ambiguity surface dynamic range [dB]"
        else:
            pass

        plt.ylim([0, max(bar_values) + 5 * val_offset])
        img_name = f"localisation_performance_src_type_" + "".join(metric) + ".png"
        plt.xlabel("Source type")
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.tight_layout()
        plt.savefig(os.path.join(img_path, img_name))

    print()


def analysis_main(
    snr_list,
    testcase_name,
    init_time,
    similarity_metrics=None,
    plot_info={},
    simulation_info={},
    grid_info={},
    file_ext="nc",
    img_format="png",
):
    """Main function to analyse the localisation performance."""
    global_header_log = "Detection metric,SNR,MEDIAN,MEAN,STD,RMSE,MAX,MIN,95_percentile,99_percentile,dynamic_range"
    global_log = [global_header_log]

    # PFIG.fmt = img_format

    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    output_f_path = os.path.join(
        VERLINDEN_OUTPUT_FOLDER,
        testcase_name,
        simulation_info["src_type"],
        simulation_info["src_pos"],
        init_time,
        f"output_{testcase_name}_snrs.{file_ext}",
    )
    if file_ext == "nc":
        xr_dataset = xr.open_dataset(output_f_path)
    elif file_ext == "zarr":
        xr_dataset = xr.open_zarr(output_f_path)

    for snr in snr_list:

        ds_snr = xr_dataset.sel(snr=snr).compute()

        if similarity_metrics is None:
            similarity_metrics = ds_snr.similarity_metric.values
        else:
            similarity_metrics = [
                s for s in similarity_metrics if s in ds_snr.similarity_metric.values
            ]

        # Image folder
        root_img = os.path.join(ds_snr.fullpath_analysis, now, f"{snr}_dB")
        img_basepath = os.path.join(root_img, testcase_name + "_")
        img_root = os.path.dirname(img_basepath)

        # Assert at least one plot will be created to avoid creating empty folders
        create_folder = check_folder_creation(plot_info)
        if create_folder and not os.path.exists(img_root):
            os.makedirs(img_root)

        n_instant_to_plot = min(
            simulation_info["n_instant_to_plot"],
            ds_snr.sizes["src_trajectory_time"],
        )
        n_rcv_signals_to_plot = min(
            simulation_info["n_rcv_signals_to_plot"],
            ds_snr.sizes["src_trajectory_time"],
        )

        # Plot emmited signal
        if plot_info["plot_emmited_signal"]:
            plot_emmited_signal(ds_snr, img_root)

        # Figures independent from the similarity metric
        # Plot received signal
        if plot_info["plot_received_signal"]:
            plot_received_signal(ds_snr, img_root, n_instant_to_plot)

        for i_sim_metric, similarity_metric in enumerate(similarity_metrics):

            ds_snr_sim = ds_snr.sel(idx_similarity_metric=i_sim_metric)

            # Plot one TL profile
            if plot_info["plot_one_tl_profile"]:
                shd_fpath = os.path.join(
                    simulation_info["simulation_folder"], testcase_name + ".shd"
                )
                for f in plot_info["tl_freq_to_plot"]:
                    plotshd(shd_fpath, freq=f, units="km")
                plt.savefig(img_basepath + f"tl_profile_{f}Hz.png")
                plt.close()

            # Plot ambiguity surface distribution
            if plot_info["plot_ambiguity_surface_dist"]:
                plot_ambiguity_surface_dist(ds_snr_sim, img_root, n_instant_to_plot)

            # Plot ambiguity surface
            if plot_info["plot_ambiguity_surface"]:
                for zoom in [True, False]:
                    plot_ambiguity_surface(
                        ds_snr_sim,
                        img_root,
                        nb_instant_to_plot=n_instant_to_plot,
                        plot_info=plot_info,
                        zoom=zoom,
                    )

            # Create video
            if plot_info["plot_video"]:
                plot_localisation_moviepy(
                    ds=ds_snr_sim,
                    nb_frames=n_instant_to_plot,
                    anim_filename=img_basepath + "ambiguity_surf.mp4",
                    plot_hyperbol=False,
                    grid_info=grid_info,
                    fps_sec=5,
                    cmap="jet",
                )

            # Plot ship trajectory
            if plot_info["plot_ship_trajectory"]:
                for zoom in [True, False]:
                    plot_ship_trajectory(
                        ds_snr_sim,
                        img_root=img_root,
                        plot_info=plot_info,
                        noise_realisation_to_plot=0,
                        zoom=zoom,
                    )

            # Plot detection error
            if plot_info["plot_pos_error"]:
                plot_pos_error(ds_snr_sim, img_root=img_root)

            # Plot correlation
            if plot_info["plot_correlation"]:
                plot_correlation(
                    ds_snr_sim,
                    img_root,
                    similarity_metric,
                    n_instant_to_plot=n_rcv_signals_to_plot,
                )

            pos_error = get_pos_error(ds_snr_sim)
            pos_error_metrics = get_pos_error_metrics(pos_error)

            amb_surf, __ = get_ambiguity_surface(ds_snr_sim)
            amb_dynamic_range = (amb_surf.max() - amb_surf.min()).round(2).values

            global_line = (
                f"{similarity_metric}, {snr}, {pos_error_metrics['median']:.1f}, {pos_error_metrics['mean']:.1f},"
                f"{pos_error_metrics['std']:.1f}, {pos_error_metrics['rmse']:.1f}, {pos_error_metrics['max']:.1f},"
                f"{pos_error_metrics['min']:.1f}, {pos_error_metrics['95_percentile']:.1f}, {pos_error_metrics['99_percentile']:.1f},"
                f"{amb_dynamic_range:.1f}"
            )

            global_log.append(global_line)

            # Write report in txt file
            local_log = [
                f"Detection metric: {similarity_metric}",
                f"SNR: {ds_snr_sim.snr.values}dB",
                f"Number of sensors: {ds_snr_sim.dims['idx_rcv']}",
                f"Number of sensors pairs: {ds_snr_sim.dims['idx_rcv_pairs']}",
                f"Positions of the source: {ds_snr_sim.attrs['source_positions']}",
                f"Number of source positions analysed: {ds_snr_sim.dims['src_trajectory_time']}",
                f"Ambiguity surface dynamic (max - min): {amb_dynamic_range:.1f}dB",
                f"Position error median: {pos_error_metrics['median']:.1f}m",
                f"Position error mean: {pos_error_metrics['mean']:.1f}m",
                f"Position error std: {pos_error_metrics['std']:.1f}m",
                f"Position rmse: {pos_error_metrics['rmse']:.1f}m",
                f"Position error max: {pos_error_metrics['max']:.1f}m",
                f"Position error min: {pos_error_metrics['min']:.1f}m",
                f"Position error 95 percentile: {pos_error_metrics['95_percentile']:.1f}m",
                f"Position error 99 percentile: {pos_error_metrics['99_percentile']:.1f}m",
            ]

            local_report_fpath = os.path.join(root_img, "loc_report.txt")
            with open(local_report_fpath, "w") as f:
                f.writelines("\n".join(local_log))

    # Write global report in txt file
    global_report_fpath = os.path.join(
        VERLINDEN_ANALYSIS_FOLDER,
        testcase_name,
        simulation_info["src_type"],
        simulation_info["src_pos"],
        "global_report.txt",
    )
    with open(global_report_fpath, "w") as f:
        f.writelines("\n".join(global_log))

    # Analysis global report
    # perf_metrics = ["RMSE", "STD"]
    data = pd.read_csv(
        global_report_fpath,
        sep=",",
        dtype={
            "SNR": str,
            "MEDIAN": float,
            "MEAN": float,
            "STD": float,
            "RMSE": float,
            "MAX": float,
            "MIN": float,
            "95_percentile": float,
            "99_percentile": float,
            "dynamic_range": float,
        },
    )
    # list_perf_metrics = [
    #     ["95_percentile"],
    #     ["99_percentile"],
    #     ["MEDIAN"],
    #     ["MEAN"],
    #     ["STD"],
    #     ["RMSE"],
    #     ["MAX"],
    #     ["MIN"],
    #     ["dynamic_range"],
    # ]

    list_perf_metrics = [
        "95_percentile",
        "99_percentile",
        "MEDIAN",
        "MEAN",
        "STD",
        "RMSE",
        "MAX",
        "MIN",
        "dynamic_range",
    ]
    # for perf_metrics in list_perf_metrics:
    plot_localisation_performance(
        data=data,
        testcase_name=testcase_name,
        similarity_metrics=similarity_metrics,
        metrics_to_plot=list_perf_metrics,
        img_path=os.path.dirname(global_report_fpath),
    )


if __name__ == "__main__":
    # snr = [-30, -20, -10, -5, -1, 1, 5, 10, 20]
    # snr = [-30, -20, -15, -10, -5, -1, 1, 5, 10, 20]
    snr = [-5, 10]
    src_signal_type = ["ship"]
    testcase_name = "testcase1_4"

    similarity_metrics = ["intercorr0", "hilbert_env_intercorr0"]

    grid_info = {
        "Lx": 5 * 1e3,
        "Ly": 5 * 1e3,
        "dx": 10,
        "dy": 10,
    }
    simu_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\verlinden_process_output"
    simulation_info = {
        "simulation_folder": os.path.join(simu_root, testcase_name),
        "src_pos": "not_on_grid",
        "n_instant_to_plot": 20,
        "n_rcv_signals_to_plot": 3,
        "src_type": "ship",
    }

    # plot_info = {
    #     "plot_video": False,
    #     "plot_one_tl_profile": True,
    #     "plot_ambiguity_surface_dist": True,
    #     "plot_received_signal": True,
    #     "plot_ambiguity_surface": True,
    #     "plot_ship_trajectory": True,
    #     "plot_pos_error": True,
    #     "plot_correlation": True,
    #     "tl_freq_to_plot": [20],
    #     "x_offset": 1000,
    #     "y_offset": 1000,
    # }

    plot_info = {
        "plot_video": False,
        "plot_one_tl_profile": False,
        "plot_ambiguity_surface_dist": False,
        "plot_received_signal": False,
        "plot_ambiguity_surface": True,
        "plot_ship_trajectory": False,
        "plot_pos_error": False,
        "plot_correlation": False,
        "tl_freq_to_plot": [20],
        "x_offset": 1000,
        "y_offset": 1000,
    }

    analysis_main(
        snr,
        similarity_metrics=similarity_metrics,
        testcase_name=testcase_name,
        simulation_info=simulation_info,
        grid_info=grid_info,
        plot_info=plot_info,
    )
