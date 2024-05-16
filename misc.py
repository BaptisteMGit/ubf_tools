import io
import os
import shutil
import numpy as np
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms

from PIL import Image
from matplotlib.patches import Ellipse


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


def delete_folders(root_dir, folder_name_pattern):
    for current_folder, subfolders, files in os.walk(root_dir, topdown=False):
        for folder in subfolders:
            if folder_name_pattern in folder:
                folder_path = os.path.join(current_folder, folder)
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics

    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    # render plot with "plt.show()".


if __name__ == "__main__":
    # Test delete folders
    root_directory = "/path/to/your/directory"
    folder_pattern = "pattern_to_match"
    delete_folders(root_directory, folder_pattern)

    # # Test plot_animation_moviepy
    # import os
    # import xarray as xr

    # path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case\verlinden_1_test_case.nc"
    # ds = xr.open_dataset(path)

    # root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden\test_case\isotropic\range_independent"
    # root_img = os.path.join(root_img, ds.src_pos, f"dx{ds.dx}m_dy{ds.dy}m")

    # # n = ds.dims["src_trajectory_time"]
    # n = 30
    # var_to_plot = ds.ambiguity_surface.isel(
    #     idx_obs_pairs=0, src_trajectory_time=slice(0, n)
    # )
    # var_to_plot = -10 * np.log10(var_to_plot)

    # plot_animation_moviepy(
    #     var_to_plot=var_to_plot,
    #     time_label="src_trajectory_time",
    #     nb_frames=n,
    #     anim_filename=os.path.join(root_img, "ambiguity_surf.mp4"),
    #     fps_sec=5,
    #     cmap="jet",
    # )
