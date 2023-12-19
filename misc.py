import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import moviepy.editor as mpy
from PIL import Image


def mult_along_axis(A, B, axis):
    # ensure we're working with Numpy arrays
    A = np.array(A)
    B = np.array(B)

    # shape check
    if axis >= A.ndim:
        raise np.AxisError(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(A, A.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc


def plot_animation(
    var_to_plot,
    nb_frames,
    anim_filename,
    fps_sec=5,
    **kwargs,
):
    """
    Plot and save animations.

    :param var_to_plot:
    :param nb_frames:
    :param anim_filename:
    :param fps_sec:
    :param kwargs:
    :return:
    """
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    var_to_plot.isel(time=0).plot(ax=ax, **kwargs)

    def animate_func(i):
        var_to_plot.isel(time=i).plot(ax=ax, add_colorbar=False, **kwargs)

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=nb_frames,
        interval=1000 / fps_sec,  # in ms
    )

    anim.save(anim_filename)


def plot_animation_moviepy(
    var_to_plot,
    time_label,
    nb_frames,
    anim_filename,
    fps_sec=30,
    **kwargs,
):
    """
    Plot and save animations using MoviePy.

    :param var_to_plot: Data to be plotted (xarray DataArray).
    :param nb_frames: Number of frames in the animation (integer).
    :param anim_filename: Output filename for the animation (string).
    :param fps_sec: Frames per second (integer).
    :param kwargs: Additional plotting arguments (e.g., cmap, vmin, vmax).
    """

    # fig, ax = plt.subplots(1, 1)
    figsize = (12, 8)
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    var_to_plot.isel({time_label: 0}).plot(ax=ax, **kwargs)

    # Function to update the plot for each frame and return NumPy array
    def animate_func(i):
        # Plot the image and get the colorbar object
        img = var_to_plot.isel({time_label: i}).plot(
            ax=ax, add_colorbar=False, **kwargs
        )

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
    n = 30
    var_to_plot = ds.ambiguity_surface.isel(
        idx_obs_pairs=0, src_trajectory_time=slice(0, n)
    )
    var_to_plot = -10 * np.log10(var_to_plot)

    plot_animation_moviepy(
        var_to_plot=var_to_plot,
        time_label="src_trajectory_time",
        nb_frames=n,
        anim_filename=os.path.join(root_img, "ambiguity_surf.mp4"),
        fps_sec=5,
        cmap="jet",
    )
