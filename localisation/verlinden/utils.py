import io
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt
from PIL import Image


def plot_localisation_moviepy(
    localisation_dataset,
    nb_frames,
    anim_filename,
    fps_sec=30,
    **kwargs,
):
    """
    Plot and save localisation animations using MoviePy.

    :param var_to_plot: Data to be plotted (xarray DataArray).
    :param nb_frames: Number of frames in the animation (integer).
    :param anim_filename: Output filename for the animation (string).
    :param fps_sec: Frames per second (integer).
    :param kwargs: Additional plotting arguments (e.g., cmap, vmin, vmax).
    """

    figsize = (12, 8)
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    obs_col = ["orange", "magenta", "green"]

    # Hyperbol
    dx = 10  # m
    dy = 10  # m
    Lx = 100 * 1e3  # m
    Ly = 80 * 1e3  # m
    grid_x = np.arange(-Lx / 2, Lx / 2, dx, dtype=np.float32)
    grid_y = np.arange(-Ly / 2, Ly / 2, dy, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    x_offset = 5000
    y_offset = 5000

    rr_obs = np.array(
        [
            np.sqrt(
                (xx - localisation_dataset.x_obs.sel(idx_obs=i_obs).values) ** 2
                + (yy - localisation_dataset.y_obs.sel(idx_obs=i_obs).values) ** 2
            )
            for i_obs in range(localisation_dataset.dims["idx_obs"])
        ]
    )

    delta_rr = rr_obs[0, ...] - rr_obs[1, ...]
    delta_rr_ship = localisation_dataset.r_obs_ship.sel(
        idx_obs=0
    ) - localisation_dataset.r_obs_ship.sel(idx_obs=1)

    amb_surf = -10 * np.log10(
        localisation_dataset.ambiguity_surface.isel(idx_obs_pairs=0)
    )
    amb_surf.isel(src_trajectory_time=0).plot(ax=ax, **kwargs)

    # Function to update the plot for each frame and return NumPy array
    def animate_func(i):
        legend = []
        # Plot ambiguity surface
        img = amb_surf.isel(src_trajectory_time=i).plot(
            ax=ax, add_colorbar=False, **kwargs
        )

        # True source position
        plt.scatter(
            localisation_dataset.x_ship.isel(src_trajectory_time=i),
            localisation_dataset.y_ship.isel(src_trajectory_time=i),
            color="k",
            marker="+",
            s=90,
            label=r"$X_{ship}$",
        )
        legend.append(r"$X_{ship}$")

        # Estimated source position
        plt.scatter(
            localisation_dataset.detected_pos_x.isel(
                idx_obs_pairs=0, src_trajectory_time=i
            ),
            localisation_dataset.detected_pos_y.isel(
                idx_obs_pairs=0, src_trajectory_time=i
            ),
            marker="o",
            facecolors="none",
            edgecolors="black",
            s=120,
            linewidths=2.2,
        )
        legend.append(r"$\tilde{X_{ship}}$")

        # Equal delta distance hyperbol
        condition = (
            np.abs(delta_rr - delta_rr_ship.isel(src_trajectory_time=i).values) < 0.1
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
        for i_obs in range(localisation_dataset.dims["idx_obs"]):
            plt.scatter(
                localisation_dataset.x_obs.isel(idx_obs=i_obs),
                localisation_dataset.y_obs.isel(idx_obs=i_obs),
                marker="o",
                color=obs_col[i_obs],
                label=f"$O_{i_obs}$",
            )

        # plt.xlim(
        #     [
        #         min(localisation_dataset.x.min(), localisation_dataset.x_obs.min())
        #         - x_offset,
        #         max(localisation_dataset.x.max(), localisation_dataset.x_obs.max())
        #         + x_offset,
        #     ]
        # )
        # plt.ylim(
        #     [
        #         min(localisation_dataset.y.min(), localisation_dataset.y_obs.min())
        #         - y_offset,
        #         max(localisation_dataset.y.max(), localisation_dataset.y_obs.max())
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


if __name__ == "__main__":
    # Test plot_animation_moviepy
    import os
    import xarray as xr

    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case\verlinden_1_test_case.nc"
    ds = xr.open_dataset(path)

    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden\test_case\isotropic\range_independent"
    root_img = os.path.join(root_img, ds.src_pos, f"dx{ds.dx}m_dy{ds.dy}m")

    # n = ds.dims["src_trajectory_time"]
    n = 15
    var_to_plot = ds.ambiguity_surface.isel(
        idx_obs_pairs=0, src_trajectory_time=slice(0, n)
    )
    var_to_plot = -10 * np.log10(var_to_plot)

    plot_localisation_moviepy(
        localisation_dataset=ds,
        nb_frames=n,
        anim_filename=os.path.join(root_img, "ambiguity_surf.mp4"),
        fps_sec=5,
        cmap="jet",
    )
