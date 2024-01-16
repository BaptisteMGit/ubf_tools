import os
import io
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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
from propa.kraken_toolbox.plot_utils import plotshd
from localisation.verlinden.verlinden_analysis_report import (
    plot_localisation_performance,
)

from localisation.verlinden.utils import plot_localisation_moviepy
from localisation.verlinden.verlinden_path import (
    VERLINDEN_OUTPUT_FOLDER,
    VERLINDEN_ANALYSIS_FOLDER,
    VERLINDEN_POPULATED_FOLDER,
)


def plot_received_signal(xr_dataset, img_basepath, n_instant_to_plot=None):
    """Plot received signal for each receiver pair and each source position."""

    if n_instant_to_plot is None:
        n_instant_to_plot = xr_dataset.dims["src_trajectory_time"]
    else:
        n_instant_to_plot = min(
            n_instant_to_plot, xr_dataset.dims["src_trajectory_time"]
        )

    for i_ship in range(n_instant_to_plot):
        fig, axes = plt.subplots(
            xr_dataset.dims["idx_obs"],
            1,
            figsize=(16, 8),
            sharey=True,
        )

        for i_obs in range(xr_dataset.dims["idx_obs"]):
            lib_sig = xr_dataset.rcv_signal_library.sel(
                x=xr_dataset.x_ship.isel(src_trajectory_time=i_ship),
                y=xr_dataset.y_ship.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_obs=i_obs)
            event_sig = xr_dataset.rcv_signal_event.isel(
                src_trajectory_time=i_ship, idx_obs=i_obs
            )
            # Plot the signal with the smallest std on top
            if event_sig.std() <= lib_sig.std():
                lib_zorder = 1
                event_zorder = 2
            else:
                event_zorder = 1
                lib_zorder = 2

            lib_sig.plot(
                ax=axes[i_obs],
                label=f"library - obs {i_obs}",
                color=LIBRARY_COLOR,
                zorder=lib_zorder,
            )
            event_sig.plot(
                ax=axes[i_obs],
                label=f"event - obs {i_obs}",
                color=EVENT_COLOR,
                zorder=event_zorder,
            )

            axes[i_obs].set_xlabel("")
            axes[i_obs].set_ylabel("")
            axes[i_obs].legend(loc="upper right")
            axes[i_obs].tick_params(labelsize=TICKS_FONTSIZE)

        for ax, col in zip(axes, [f"Receiver {i}" for i in xr_dataset.idx_obs.values]):
            ax.set_title(col, fontsize=TITLE_FONTSIZE)

        fig.supylabel("Received signal", fontsize=SUPLABEL_FONTSIZE)
        fig.supxlabel("Time [s]", fontsize=SUPLABEL_FONTSIZE)
        plt.tight_layout()
        plt.savefig(img_basepath + f"received_signals_{i_ship}.png")
        plt.close()


def get_ambiguity_surface(ds):
    # Avoid singularity for S = 0
    amb_surf_not_0 = ds.ambiguity_surface.values[ds.ambiguity_surface > 0]
    ds.ambiguity_surface.values[ds.ambiguity_surface == 0] = amb_surf_not_0.min()
    amb_surf = 10 * np.log10(ds.ambiguity_surface)  # dB scale

    return amb_surf


def plot_ambiguity_surface_dist(ds, img_basepath):
    """Plot ambiguity surface distribution."""
    amb_surf = get_ambiguity_surface(ds)

    plt.figure(figsize=(16, 10))
    amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=0).plot.hist(bins=10000)
    plt.scatter(amb_surf.max(), 1, marker="o", color="red", label="Max")
    plt.ylabel("Number of points", fontsize=LABEL_FONTSIZE)
    plt.xlabel("Ambiguity surface [dB]", fontsize=LABEL_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.legend(loc="upper right")
    plt.savefig(img_basepath + f"ambiguity_surface_dist.png")
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
                (xx - ds.x_obs.sel(idx_obs=i_obs).values) ** 2
                + (yy - ds.y_obs.sel(idx_obs=i_obs).values) ** 2
            )
            for i_obs in range(ds.dims["idx_obs"])
        ]
    )

    delta_rr = rr_obs[0, ...] - rr_obs[1, ...]
    delta_rr_ship = ds.r_obs_ship.sel(idx_obs=0) - ds.r_obs_ship.sel(idx_obs=1)

    return xx, yy, delta_rr, delta_rr_ship


def plot_ambiguity_surface(
    ds,
    img_basepath,
    nb_instant_to_plot=10,
    plot_beampattern=False,
    plot_hyperbol=False,
    grid_info={},
    plot_info={},
):
    """Plot ambiguity surface for each source position."""

    amb_surf = get_ambiguity_surface(ds)

    for i in range(nb_instant_to_plot):
        plt.figure(figsize=(10, 8))

        vmin = (
            amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).quantile(0.35).values
        )
        vmax = amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).max()
        amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).plot(
            x="x", y="y", zorder=0, vmin=vmin, vmax=vmax, cmap="jet"
        )
        ax = plt.gca()
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(20)

        # if plot_beampattern:
        #     # bp_offset = np.sqrt(
        #     #     (ds.x_ship.isel(src_trajectory_time=i) - x_center_array) ** 2
        #     #     + (ds.y_ship.isel(src_trajectory_time=i) - y_center_array) ** 2
        #     # )
        #     # log_bp = 10 * np.log10(linear_bp)
        #     # bp_alpha = -bp_offset / min(log_bp)
        #     # log_bp = log_bp * bp_alpha.values + bp_offset.values
        #     # x_bp = log_bp * np.cos(theta) + x_center_array
        #     # y_bp = log_bp * np.sin(theta) + y_center_array
        #     # plt.plot(x_bp, y_bp, "k", linestyle="--", label="Beampattern", zorder=1)

        #     # plt.gca().get_images()[0].clim([0, 20])
        #     # ds.ambiguity_surface.isel(idx_obs_pairs=0, src_trajectory_time=i).plot(
        #     #     x="x", y="y", zorder=0, clim=[-1, 1]
        #     # )

        if plot_hyperbol:
            xx, yy, delta_rr, delta_rr_ship = get_grid_arrays(ds, grid_info)

            condition = (
                np.abs(delta_rr - delta_rr_ship.isel(src_trajectory_time=i).values)
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
            ds.x_ship.isel(src_trajectory_time=i),
            ds.y_ship.isel(src_trajectory_time=i),
            color="k",
            marker="+",
            s=90,
            label=r"$X_{ship}$",
            zorder=2,
        )

        plt.scatter(
            ds.detected_pos_x.isel(idx_obs_pairs=0, src_trajectory_time=i),
            ds.detected_pos_y.isel(idx_obs_pairs=0, src_trajectory_time=i),
            marker="o",
            facecolors="none",
            edgecolors="black",
            s=120,
            linewidths=2.2,
            label="Estimated position",
            zorder=3,
        )

        for i_obs in range(ds.dims["idx_obs"]):
            plt.scatter(
                ds.x_obs.isel(idx_obs=i_obs),
                ds.y_obs.isel(idx_obs=i_obs),
                marker="o",
                label=f"$O_{i_obs}$",
            )

            plt.xlim(
                [
                    min(ds.x.min(), ds.x_obs.min()) - plot_info["x_offset"],
                    max(ds.x.max(), ds.x_obs.max()) + plot_info["x_offset"],
                ]
            )
            plt.ylim(
                [
                    min(ds.y.min(), ds.y_obs.min()) - plot_info["y_offset"],
                    max(ds.y.max(), ds.y_obs.max()) + plot_info["y_offset"],
                ]
            )

        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.legend(ncol=2, loc="upper right", fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.savefig(img_basepath + f"ambiguity_surface_{i}.png")
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
    amb_surf = get_ambiguity_surface(ds)

    # Function to update the plot for each frame and return NumPy array
    def animate_func(i):
        legend = []
        # Plot ambiguity surface
        img = amb_surf.isel(src_trajectory_time=i).plot(
            ax=ax, add_colorbar=False, **kwargs
        )

        # True source position
        plt.scatter(
            ds.x_ship.isel(src_trajectory_time=i),
            ds.y_ship.isel(src_trajectory_time=i),
            color="k",
            marker="+",
            s=90,
            label=r"$X_{ship}$",
        )
        legend.append(r"$X_{ship}$")

        # Estimated source position
        plt.scatter(
            ds.detected_pos_x.isel(idx_obs_pairs=0, src_trajectory_time=i),
            ds.detected_pos_y.isel(idx_obs_pairs=0, src_trajectory_time=i),
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
                np.abs(delta_rr - delta_rr_ship.isel(src_trajectory_time=i).values)
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
        for i_obs in range(ds.dims["idx_obs"]):
            plt.scatter(
                ds.x_obs.isel(idx_obs=i_obs),
                ds.y_obs.isel(idx_obs=i_obs),
                marker="o",
                color=obs_col[i_obs],
                label=f"$O_{i_obs}$",
            )

        # plt.xlim(
        #     [
        #         min(ds.x.min(), ds.x_obs.min())
        #         - x_offset,
        #         max(ds.x.max(), ds.x_obs.max())
        #         + x_offset,
        #     ]
        # )
        # plt.ylim(
        #     [
        #         min(ds.y.min(), ds.y_obs.min())
        #         - y_offset,
        #         max(ds.y.max(), ds.y_obs.max())
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


def plot_ship_trajectory(ds, img_basepath):
    """Plot ship trajectory."""

    plt.figure(figsize=(10, 8))
    plt.plot(ds.x_ship, ds.y_ship, marker="+", color="red", label=r"$X_{ship}$")
    for i_obs in range(ds.dims["idx_obs"]):
        plt.scatter(
            ds.x_obs.isel(idx_obs=i_obs),
            ds.y_obs.isel(idx_obs=i_obs),
            marker="o",
            label=f"$O_{i_obs}$",
        )
    for i_pair in ds.idx_obs_pairs:
        plt.scatter(
            ds.detected_pos_x.sel(idx_obs_pairs=i_pair),
            ds.detected_pos_y.sel(idx_obs_pairs=i_pair),
            marker="+",
            color="black",
            s=120,
            linewidths=2.2,
            label="Estimated position",
        )

    plt.xlabel("x [m]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("y [m]", fontsize=LABEL_FONTSIZE)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(img_basepath + f"estimated_pos.png")
    plt.close()


def get_pos_error(ds):
    pos_error = np.sqrt(
        (ds.detected_pos_x - ds.x_ship) ** 2 + (ds.detected_pos_y - ds.y_ship) ** 2
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
    }
    return pos_error_metrics


def plot_pos_error(ds, img_basepath):
    """Plot position error."""

    pos_error = get_pos_error(ds)

    plt.figure(figsize=(10, 8))
    for i_pair in ds.idx_obs_pairs:
        pos_error.sel(idx_obs_pairs=i_pair).plot(label=f"obs pair {i_pair.values}")

        plt.axhline(
            pos_error.median(),
            color="red",
            linestyle="--",
            label=f"Median error for obs pair {i_pair.values} = {pos_error.median().round(0).values}m",
        )

    plt.ylabel("Position error [m]", fontsize=LABEL_FONTSIZE)
    plt.gca().xaxis.label.set_fontsize(LABEL_FONTSIZE)

    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.savefig(img_basepath + f"pos_error.png")
    plt.close()


def plot_correlation(ds, img_basepath, det_metric="intercorr0", nb_instant_to_plot=10):
    """Plot correlation for receiver couple."""

    fig, axes = plt.subplots(
        nb_instant_to_plot,
        ds.dims["idx_obs_pairs"],
        sharex=True,
        sharey=True,
        figsize=(16, 8),
    )
    axes = np.reshape(
        axes, (nb_instant_to_plot, ds.dims["idx_obs_pairs"])
    )  # Ensure 2D axes array in case of single obs pair

    for i_ship in range(nb_instant_to_plot):
        for i_obs_pair in range(ds.dims["idx_obs_pairs"]):
            event_vect = ds.event_corr.isel(
                src_trajectory_time=i_ship, idx_obs_pairs=i_obs_pair
            )
            lib_vect = ds.library_corr.sel(
                x=ds.x_ship.isel(src_trajectory_time=i_ship),
                y=ds.y_ship.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_obs_pairs=i_obs_pair)

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
                ax=axes[i_ship, i_obs_pair],
                label="event",
                color=EVENT_COLOR,
                zorder=event_zorder,
            )
            lib_vect.plot(
                ax=axes[i_ship, i_obs_pair],
                label="lib at ship pos",
                color=LIBRARY_COLOR,
                zorder=lib_zorder,
            )
            axes[i_ship, i_obs_pair].set_xlabel("")
            axes[i_ship, i_obs_pair].set_ylabel("")
            axes[i_ship, i_obs_pair].set_title(
                f"Source pos n°{i_ship}", fontsize=TITLE_FONTSIZE
            )
            axes[i_ship, i_obs_pair].legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
            axes[i_ship, i_obs_pair].tick_params(labelsize=TICKS_FONTSIZE)

    fig.supxlabel(r"$\tau (s)$", fontsize=SUPLABEL_FONTSIZE)
    fig.supylabel(ylabel, fontsize=SUPLABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig(img_basepath + f"signal_corr.png")
    plt.close()


def analysis_main(
    snr_list, detection_metric_list, plot_info={}, simulation_info={}, grid_info={}
):
    global_header_log = "Detection metric,SNR,MEDIAN,MEAN,STD,RMSE,MAX,MIN"
    global_log = [global_header_log]

    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    for snr in snr_list:
        if snr is None:
            snr_tag = "_noiseless"
        else:
            snr_tag = f"_snr{snr}dB"

        for detection_metric in detection_metric_list:
            env_fname = "verlinden_1_test_case"

            output_nc_path = os.path.join(
                VERLINDEN_OUTPUT_FOLDER,
                env_fname,
                simulation_info["src_type"],
                simulation_info["src_pos"],
                detection_metric,
                f"output_{detection_metric}{snr_tag}.nc",
            )
            ds = xr.open_dataset(output_nc_path)

            # Image folder
            root_img = ds.fullpath_analysis
            img_basepath = os.path.join(root_img, now, env_fname + "_")
            if not os.path.exists(os.path.dirname(img_basepath)):
                os.makedirs(os.path.dirname(img_basepath))

            n_instant_to_plot = min(
                simulation_info["n_instant_to_plot"], ds.dims["src_trajectory_time"]
            )

            # Plot one TL profile
            if plot_info["plot_one_tl_profile"]:
                shd_fpath = os.path.join(
                    simulation_info["simulation_folder"], env_fname + ".shd"
                )
                for f in plot_info["tl_freq_to_plot"]:
                    plotshd(shd_fpath, freq=f, units="km")
                plt.savefig(img_basepath + f"tl_profile_{f}Hz.png")
                plt.close()

            # Plot ambiguity surface distribution
            if plot_info["plot_ambiguity_surface_dist"]:
                plot_ambiguity_surface_dist(ds, img_basepath)

            # Plot received signal
            if plot_info["plot_received_signal"]:
                plot_received_signal(
                    ds, img_basepath, simulation_info["n_rcv_signals_to_plot"]
                )

            # Plot ambiguity surface
            if plot_info["plot_ambiguity_surface"]:
                plot_ambiguity_surface(
                    ds,
                    img_basepath,
                    nb_instant_to_plot=n_instant_to_plot,
                    plot_info=plot_info,
                )

            # Create video
            if plot_info["plot_video"]:
                plot_localisation_moviepy(
                    ds=ds,
                    nb_frames=n_instant_to_plot,
                    anim_filename=img_basepath + "ambiguity_surf.mp4",
                    plot_hyperbol=False,
                    grid_info=grid_info,
                    fps_sec=5,
                    cmap="jet",
                )

            # Plot ship trajectory
            if plot_info["plot_ship_trajectory"]:
                plot_ship_trajectory(ds, img_basepath)

            # Plot detection error
            if plot_info["plot_pos_error"]:
                plot_pos_error(ds, img_basepath)

            # Plot correlation
            if plot_info["plot_correlation"]:
                plot_correlation(
                    ds,
                    img_basepath,
                    detection_metric,
                    simulation_info["n_rcv_signals_to_plot"],
                )

            pos_error = get_pos_error(ds)
            pos_error_metrics = get_pos_error_metrics(pos_error)

            amb_surf = get_ambiguity_surface(ds)
            amb_dynamic_range = (amb_surf.max() - amb_surf.min()).round(2).values

            global_line = f"{detection_metric}, {snr}, {pos_error_metrics['median']}, {pos_error_metrics['mean']}, {pos_error_metrics['std']}, {pos_error_metrics['rmse']}, {pos_error_metrics['max']}, {pos_error_metrics['min']}"
            global_log.append(global_line)

            # Write report in txt file
            local_log = [
                f"Detection metric: {detection_metric}",
                f"SNR: {ds.attrs['snr_dB']}dB",
                f"Number of sensors: {ds.dims['idx_obs']}",
                f"Number of sensors pairs: {ds.dims['idx_obs_pairs']}",
                f"Positions of the source: {ds.attrs['source_positions']}",
                f"Number of source positions analysed: {ds.dims['src_trajectory_time']}",
                f"Ambiguity surface dynamic (max - min): {amb_dynamic_range}dB",
                f"Position error median: {pos_error_metrics['median']}m",
                f"Position error mean: {pos_error_metrics['mean']}m",
                f"Position error std: {pos_error_metrics['std']}m",
                f"Position rmse: {pos_error_metrics['rmse']}m",
            ]

            local_report_fpath = os.path.join(root_img, "loc_report.txt")
            with open(local_report_fpath, "w") as f:
                f.writelines("\n".join(local_log))

    # Write global report in txt file
    global_report_fpath = os.path.join(
        VERLINDEN_ANALYSIS_FOLDER,
        env_fname,
        simulation_info["src_type"],
        simulation_info["src_pos"],
        "global_report.txt",
    )
    with open(global_report_fpath, "w") as f:
        f.writelines("\n".join(global_log))

    # Analysis global report
    perf_metrics = ["RMSE", "STD"]
    plot_localisation_performance(
        data=pd.read_csv(global_report_fpath, sep=","),
        detection_metric_list=detection_metric_list,
        metrics_to_plot=perf_metrics,
        img_path=os.path.dirname(global_report_fpath),
    )


if __name__ == "__main__":
    # snr = [-30, -20, -10, -5, -1, 1, 5, 10, 20]
    # snr = [-30, -20, -15, -10, -5, -1, 1, 5, 10, 20]
    # snr = [5, -10, 20, -20, None]
    # detection_metric = ["intercorr0"]
    snr = [None]
    detection_metric = ["intercorr0"]

    grid_info = {
        "Lx": 5 * 1e3,
        "Ly": 5 * 1e3,
        "dx": 10,
        "dy": 10,
    }
    simulation_info = {
        "simulation_folder": r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case",
        "src_pos": "not_on_grid",
        "n_instant_to_plot": 10,
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
        "plot_one_tl_profile": True,
        "plot_ambiguity_surface_dist": True,
        "plot_received_signal": True,
        "plot_ambiguity_surface": True,
        "plot_ship_trajectory": True,
        "plot_pos_error": True,
        "plot_correlation": True,
        "tl_freq_to_plot": [20],
        "x_offset": 1000,
        "y_offset": 1000,
    }

    analysis_main(
        snr,
        detection_metric,
        simulation_info=simulation_info,
        grid_info=grid_info,
        plot_info=plot_info,
    )
