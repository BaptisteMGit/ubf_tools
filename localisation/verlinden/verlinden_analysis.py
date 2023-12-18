import os
import io
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal
import moviepy.editor as mpy

from PIL import Image
from propa.kraken_toolbox.plot_utils import plotshd
from localisation.verlinden.directivity_pattern import (
    linear_beampattern,
    plot_beampattern,
)

from localisation.verlinden.utils import plot_localisation_moviepy

root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"
# env_fname = "verlinden_1_ssp"


def plot_received_signal(xr_dataset, n_instant_to_plot=None):
    """Plot received signal for each receiver pair and each source position."""

    if n_instant_to_plot is None:
        n_instant_to_plot = xr_dataset.dims["src_trajectory_time"]
    else:
        n_instant_to_plot = min(
            n_instant_to_plot, xr_dataset.dims["src_trajectory_time"]
        )

    fig, axes = plt.subplots(
        n_instant_to_plot,
        xr_dataset.dims["idx_obs"],
        sharex=True,
        sharey=True,
    )

    axes = np.reshape(axes, (n_instant_to_plot, xr_dataset.dims["idx_obs"]))

    for i_ship in range(n_instant_to_plot):
        for i_obs in range(xr_dataset.dims["idx_obs"]):
            xr_dataset.rcv_signal_event.isel(
                src_trajectory_time=i_ship, idx_obs=i_obs
            ).plot(ax=axes[i_ship, i_obs], label=f"event - obs {i_obs}")

            xr_dataset.rcv_signal_library.sel(
                x=xr_dataset.x_ship.isel(src_trajectory_time=i_ship),
                y=xr_dataset.y_ship.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_obs=i_obs).plot(
                ax=axes[i_ship, i_obs], label=f"library - obs {i_obs}"
            )

            axes[i_ship, i_obs].set_xlabel("")
            axes[i_ship, i_obs].set_ylabel("")
            axes[i_ship, i_obs].set_title(
                f"x_ship = {xr_dataset.x_ship.isel(src_trajectory_time=i_ship).values.round(0)}m, y_ship = {xr_dataset.y_ship.isel(src_trajectory_time=i_ship).values.round(0)}m",
            )

            # axes[i_ship, i_obs].set_ylim([-0.005, 0.005])

    for ax, col in zip(axes[0], [f"Receiver {i}" for i in xr_dataset.idx_obs.values]):
        ax.set_title(col)

    plt.tight_layout()
    fig.supylabel("Received signal")
    fig.supxlabel("Time (s)")
    axes[-1, 0].legend()
    axes[-1, 1].legend()
    plt.show()


def get_ambiguity_surface(ds):
    # Avoid singularity for S = 0
    amb_surf_not_0 = ds.ambiguity_surface.values[ds.ambiguity_surface > 0]
    ds.ambiguity_surface.values[ds.ambiguity_surface == 0] = amb_surf_not_0.min()
    amb_surf = 10 * np.log10(ds.ambiguity_surface)  # dB scale

    return amb_surf


def plot_ambiguity_surface_dist(ds):
    """Plot ambiguity surface distribution."""
    amb_surf = get_ambiguity_surface(ds)
    plt.figure(figsize=(10, 8))
    amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=0).plot.hist(bins=10000)
    plt.xlabel("Ambiguity surface [dB]")
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
    nb_instant_to_plot=10,
    plot_beampattern=False,
    plot_hyperbol=False,
    grid_info={},
):
    """Plot ambiguity surface for each source position."""

    amb_surf = get_ambiguity_surface(ds)

    for i in range(nb_instant_to_plot):
        plt.figure(figsize=(10, 8))
        vmin = (
            amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).quantile(0.25).values
        )

        amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).plot(
            x="x", y="y", zorder=0, vmin=vmin, vmax=0, cmap="jet"
        )

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
                    min(ds.x.min(), ds.x_obs.min()) - x_offset,
                    max(ds.x.max(), ds.x_obs.max()) + x_offset,
                ]
            )
            plt.ylim(
                [
                    min(ds.y.min(), ds.y_obs.min()) - y_offset,
                    max(ds.y.max(), ds.y_obs.max()) + y_offset,
                ]
            )

        plt.tight_layout()
        plt.legend(ncol=2, loc="upper right")
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


def plot_ship_trajectory(ds):
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

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
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


def plot_pos_error(ds):
    """Plot position error."""

    pos_error = get_pos_error(ds)

    plt.figure(figsize=(10, 8))
    for i_pair in ds.idx_obs_pairs:
        pos_error.sel(idx_obs_pairs=i_pair).plot(label=f"obs pair {i_pair}")

        plt.axhline(
            pos_error.median(),
            color="red",
            linestyle="--",
            label=f"median error for obs pair {i_pair} = {pos_error.median().round(0).values}m",
        )

    plt.ylabel("Position error [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_basepath + f"pos_error.png")
    plt.close()


def plot_correlation(ds, nb_instant_to_plot=10):
    """Plot correlation for receiver couple."""

    fig, axes = plt.subplots(
        nb_instant_to_plot,
        ds.dims["idx_obs_pairs"],
        sharex=True,
        sharey=True,
        figsize=(10, 8),
    )
    axes = np.reshape(
        axes, (nb_instant_to_plot, ds.dims["idx_obs_pairs"])
    )  # Ensure 2D axes array in case of single obs pair

    for i_ship in range(nb_instant_to_plot):
        for i_obs_pair in range(ds.dims["idx_obs_pairs"]):
            ds.event_corr.isel(
                src_trajectory_time=i_ship, idx_obs_pairs=i_obs_pair
            ).plot(ax=axes[i_ship, i_obs_pair], label="event")
            ds.library_corr.sel(
                x=ds.x_ship.isel(src_trajectory_time=i_ship),
                y=ds.y_ship.isel(src_trajectory_time=i_ship),
                method="nearest",
            ).isel(idx_obs_pairs=i_obs_pair).plot(
                ax=axes[i_ship, i_obs_pair], label="lib at ship pos"
            )
            axes[i_ship, i_obs_pair].set_xlabel("")
            axes[i_ship, i_obs_pair].set_ylabel("")
            axes[i_ship, i_obs_pair].set_title("")
            axes[i_ship, i_obs_pair].set_ylim([-1, 1])

    plt.savefig(img_basepath + f"signal_corr.png")
    plt.legend()
    plt.close()


def analysis_main(grid_info={}):
    global_log = []
    snr = 5
    detection_metric = (
        "lstsquares"  # "intercorr0", "lstsquares", "hilbert_env_intercorr0"
    )

    # snr = [-20, -10, -5, 0]
    # detection_metric = ["intercorr0", "hilbert_env_intercorr0"]
    for snr in [-20, -10, -5, 0]:
        if snr is None:
            snr_tag = "_noiseless"
        else:
            snr_tag = f"_snr{snr}dB"

        for detection_metric in ["intercorr0", "hilbert_env_intercorr0"]:
            env_fname = "verlinden_1_test_case"

            nc_path = os.path.join(
                root, env_fname + "_" + detection_metric + snr_tag + ".nc"
            )
            ds = xr.open_dataset(nc_path)

            # Image folder
            root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden\test_case\isotropic\range_independent"
            root_img = os.path.join(
                root_img,
                ds.src_pos,
                f"dx{int(ds.dx)}m_dy{int(ds.dy)}m",
                ds.detection_metric,
                snr_tag[1:],
            )
            if not os.path.exists(root_img):
                os.makedirs(root_img)

            img_basepath = os.path.join(root_img, env_fname + "_")

            # n_instant_to_plot = ds.dims["src_trajectory_time"]
            n_instant_to_plot = 20
            n_instant_to_plot = min(n_instant_to_plot, ds.dims["src_trajectory_time"])

            # Plot one TL profile
            shd_fpath = os.path.join(root, env_fname + ".shd")
            for f in [10, 15, 20, 25, 40, 45]:
                plotshd(shd_fpath, freq=f, units="km")
            plt.show()

            # Plot ambiguity surface distribution
            plot_ambiguity_surface_dist(ds)

            # Plot received signal
            plot_received_signal(ds)

            # Plot ambiguity surface
            plot_ambiguity_surface(ds, nb_instant_to_plot=n_instant_to_plot)

            # Create video
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
            plot_ship_trajectory(ds)

            # Plot detection error
            plot_pos_error(ds)

            # Plot correlation
            plot_correlation(ds)

            pos_error = get_pos_error(ds)
            pos_error_metrics = get_pos_error_metrics(pos_error)

            amb_surf = get_ambiguity_surface(ds)
            amb_dynamic_range = (amb_surf.max() - amb_surf.min()).round(2).values

            # Write report in txt file
            lines = [
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

            global_line = f"{detection_metric}, {snr}, {pos_error_metrics['median']}, {pos_error_metrics['mean']}, {pos_error_metrics['std']}, {pos_error_metrics['rmse']}, {pos_error_metrics['max']}, {pos_error_metrics['min']}"
            global_log.append(global_line)

            report_fpath = os.path.join(root_img, "loc_report.txt")
            with open(report_fpath, "w") as f:
                f.writelines("\n".join(lines))
