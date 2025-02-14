#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   misc.py
@Time    :   2024/07/08 09:13:24
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Miscellaneous functions.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import io
import os
import re

# import cv2
import shutil
import psutil
import numpy as np
import pandas as pd
import multiprocessing
import scipy.fft as sp_fft

# import moviepy.editor as mpy
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


def cast_matrix_to_target_shape(matrix, target_shape):

    # Cast matrix to target shape
    # 1) Identify missing dimensions
    missing_dims = [dim for dim in target_shape if dim not in matrix.shape]
    # Link missing dimensions to the corresponding axis
    missing_dims_axis = [
        i for i in range(len(target_shape)) if target_shape[i] in missing_dims
    ]
    # 2) Add missing dimensions
    matrix_target_shape = np.expand_dims(matrix, axis=missing_dims_axis)
    # 3) Repeat matrix along missing dimensions
    tile_shape = tuple(
        [
            target_shape[i] - matrix_target_shape.shape[i] + 1
            for i in range(len(target_shape))
        ]
    )
    # 4) Tile matrix
    matrix_target_shape = np.tile(matrix_target_shape, tile_shape)

    return matrix_target_shape


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


# def plot_animation_moviepy(
#     var_to_plot,
#     time_label,
#     nb_frames,
#     anim_filename,
#     fps_sec=30,
#     **kwargs,
# ):
#     """
#     Plot and save animations using MoviePy.

#     :param var_to_plot: Data to be plotted (xarray DataArray).
#     :param nb_frames: Number of frames in the animation (integer).
#     :param anim_filename: Output filename for the animation (string).
#     :param fps_sec: Frames per second (integer).
#     :param kwargs: Additional plotting arguments (e.g., cmap, vmin, vmax).
#     """

#     # fig, ax = plt.subplots(1, 1)
#     figsize = (12, 8)
#     dpi = 100
#     fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
#     var_to_plot.isel({time_label: 0}).plot(ax=ax, **kwargs)

#     # Function to update the plot for each frame and return NumPy array
#     def animate_func(i):
#         # Plot the image and get the colorbar object
#         img = var_to_plot.isel({time_label: i}).plot(
#             ax=ax, add_colorbar=False, **kwargs
#         )

#         # Create a PIL Image from the Matplotlib figure
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         img_pil = Image.open(buf)

#         # Convert the PIL Image to a NumPy array
#         img_array = np.array(img_pil)

#         return img_array

#     # Create a list of NumPy arrays for each frame
#     frames = [animate_func(i) for i in range(nb_frames)]

#     # Create the MoviePy ImageSequenceClip from the list of NumPy arrays
#     animation_clip = mpy.ImageSequenceClip(frames, fps=fps_sec)

#     # Save the animation as an MP4 video file
#     animation_clip.write_videofile(anim_filename)

#     # Close the figure to avoid memory leaks
#     plt.close(fig)


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

    # Calculating the standard deviation of x from
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


def get_child_pids():
    """
    Get child process pid.

    :return:
    """
    parent_pid = multiprocessing.current_process().pid
    children = psutil.Process(parent_pid).children(recursive=True)
    return [child.pid for child in children]


def fft_convolve_f(a0, a1, axis=-1, workers=8):
    """
    Compute the cross-correlation of two real signals using their Fourier transforms a0 and a1.

    Parameters
    ----------
    a0 : np.ndarray
        Fourier transform of the first signal.
    a1 : np.ndarray
        Fourier transform of the second signal.
    axis : int, optional
        Axis along which to compute the cross-correlation. The default is -1.
    workers : int, optional
        Number of workers. The default is 8.

    Returns
    -------
    np.ndarray
        Cross-correlation of a0 and a1.

    """

    # Compute the cross-correlation of a0 and a1 using the FFT
    corr_01 = sp_fft.irfft(a0 * np.conj(a1), axis=axis, workers=workers)
    # Reorganise so that tau = 0 corresponds to the center of the array
    nmid = corr_01.shape[-1] // 2 + 1
    corr_01 = np.concatenate((corr_01[..., nmid:], corr_01[..., :nmid]), axis=axis)
    return corr_01


def generate_colors(n, colormap_name="Pastel1"):
    """
    Generate a list of n colors using a specified colormap.

    Parameters:
    - n: int, the number of colors to generate.
    - colormap_name: str, the name of the colormap to use (default is 'Pastel1').

    Returns:
    - List of RGBA colors.
    """
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / n) for i in range(n)]
    return colors


def gather_acronyms(manuscript_folder, output_file):
    acronyms = set()

    # Define the regex pattern for the acronym lines
    pattern = re.compile(r"\\newacronym\{(\w+)\}\{(\w+)\}\{([^}]+)\}")

    # Walk through the manuscript folder
    for root, _, files in os.walk(manuscript_folder):
        for file in files:
            if file == "glossary.tex":
                filepath = os.path.join(root, file)
                print(f"Processing file: {filepath}")
                with open(filepath, "r") as f:
                    content = f.readlines()
                    for line in content:
                        match = pattern.match(line.strip())
                        if match:
                            acronyms.add(line.strip())

    # Write the collected acronyms to the output file
    with open(output_file, "w") as f:
        for acronym in sorted(acronyms):
            f.write(acronym + "\n")


def gather_bibliographies(manuscript_folder, output_file):
    bib_entries = {}

    # Define the regex pattern for the bibliography entry
    entry_pattern = re.compile(r"@(\w+)\{([^,]+),")
    note_pattern = re.compile(r"\s*note\s*=\s*\{[^}]+\},?\n?", re.IGNORECASE)

    # Walk through the manuscript folder
    for root, _, files in os.walk(manuscript_folder):
        for file in files:
            if file == "biblio.bib":
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split the content into individual entries
                    entries = re.split(r"(@\w+\{[^,]+,)", content)
                    for i in range(1, len(entries), 2):
                        entry_header = entries[i]
                        entry_body = entries[i + 1]
                        # Remove note field
                        entry_body = note_pattern.sub("", entry_body)
                        # Reconstruct the entry
                        entry = f"{entry_header}{entry_body}"
                        # Extract the reference key
                        ref_key = re.search(entry_pattern, entry_header).group(2)
                        # Add the entry to the dictionary to ensure uniqueness
                        bib_entries[ref_key] = entry

    # Write the collected bibliography entries to the output file, sorted by reference key
    with open(output_file, "w", encoding="utf-8") as f:
        for ref_key in sorted(bib_entries):
            f.write(bib_entries[ref_key])
            f.write("\n")


def robust_normalization(ambiguity_surface):
    median = np.median(ambiguity_surface)
    iqr = np.percentile(ambiguity_surface, 75) - np.percentile(ambiguity_surface, 25)
    robust_surface = (ambiguity_surface - median) / iqr
    robust_surface = np.clip(robust_surface, 0, None)  # Ensure non-negative values
    normalized_surface = robust_surface / np.max(robust_surface)
    return normalized_surface


def ambiguity_enhencement(ambiguity_surface):

    # Get rid of extreme values
    alpha_quantile = 0.75
    th = np.quantile(ambiguity_surface, q=1 - alpha_quantile)
    da_amb_surf = np.where(da_amb_surf < th, th, da_amb_surf)

    # Stretch values in [0, 1]
    da_amb_surf = (da_amb_surf - da_amb_surf.min()) / (
        da_amb_surf.max() - da_amb_surf.min()
    )

    # Replace 0 values by minimum value to avoid log(0)
    min_val = 1e-10  # -100 dB
    da_amb_surf = np.where(da_amb_surf == 0, min_val, da_amb_surf)


def plot_amb(ambiguity_surface):
    import time

    plt.figure()
    plt.hist(ambiguity_surface.flatten(), bins=1000)
    a = np.random.randint(0, int(1e6))
    plt.savefig(f"test_{a}")


def histogram_equalization(ambiguity_surface):
    # Convert to 8-bit image for histogram equalization
    surface_8bit = cv2.normalize(
        ambiguity_surface, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")
    equalized_surface = cv2.equalizeHist(surface_8bit)
    # Convert back to float and normalize to [0, 1]
    normalized_surface = equalized_surface.astype("float") / 255
    return normalized_surface


def count_publications_per_year(
    file_path, output_filename="nombre_publications_par_annee", output_root=None
):
    """
    Compte le nombre de publications par année à partir d'un fichier Excel (extrait de l'outil lens.org) et exporte les résultats dans un nouveau fichier Excel.

    Args:
        file_path (str): Le chemin du fichier csv/Excel contenant les données d'entrée.
        output_file (str): Le nom du fichier Excel de sortie avec les résultats (par défaut: 'nombre_publications_par_annee').

    Returns:
        pd.DataFrame: Un tableau contenant les années et le nombre de publications correspondantes.
    """
    # Lecture de l'extension du fichier
    ext = os.path.splitext(file_path)[1]

    if ext == ".csv":
        # Charger le fichier CSV
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        # Charger le fichier Excel
        df = pd.read_excel(file_path)

    if output_root is None:
        output_root = os.path.dirname(file_path)

    # Extraire la colonne "Publication Year"
    publication_years = df["Publication Year"]

    # Filtrer pour les années après 1950
    publication_years = publication_years[publication_years >= 1950]

    # Compter le nombre de publications par année
    year_counts = publication_years.value_counts().sort_index()

    # Créer le tableau avec deux colonnes : Année et Nombre de publications
    result_df = pd.DataFrame(
        {"Year": year_counts.index, "Number of Publications": year_counts.values}
    )

    # Exporter le tableau dans un nouveau fichier Excel
    output_filename = f"{output_filename}.xlsx"
    output_filepath = os.path.join(output_root, output_filename)
    result_df.to_excel(output_filepath, index=False)

    # Retourner le DataFrame pour un éventuel traitement ultérieur
    return result_df, output_filepath


def export_to_dat(df, file_path):
    """
    Exporte le DataFrame sous forme de fichier .dat compatible avec LaTeX.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à exporter.
        dat_file (str): Le nom du fichier .dat à générer (par défaut: 'publications_data.dat').
    """

    # Add .dat to input filename
    dat_file_path = os.path.splitext(file_path)[0] + ".dat"
    # dat_file_path = os.path.join(os.path.dirname(file_path), dat_file)
    with open(dat_file_path, "w") as f:
        f.write("Year Number_of_Publications\n")
        for _, row in df.iterrows():
            f.write(f"{int(row['Year'])} {int(row['Number of Publications'])}\n")


def compute_hyperbola(receiver1, receiver2, source, num_points=500, tmax=2):
    """
        Computes the hyperbola curve given two receiver positions and a given source position.

        Parameters:
            receiver1 (tuple): Coordinates (x1, y1) of first receiver.
            receiver2 (tuple): Coordinates (x2, y2) of second receiver.
            source (tuple): Coordinates (xs, ys) of source.
            num_points (int): Number of points to plot the hyperbola.

        Returns:
            x_hyper (numpy array), y_hyper (numpy array): Coordinates of the hyperbola.


    # Example usage
    receiver1 = (0, 0)
    receiver2 = (250, 0)
    x_s = 3990
    y_s = 6790
    source = (x_s, y_s)

    # Compute hyperbola branches
    (right_branch, left_branch) = compute_hyperbola(receiver1, receiver2, source)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(right_branch[0], right_branch[1], "r", label="Hyperbola (Right)")
    # plt.plot(left_branch[0], left_branch[1], "r", label="Hyperbola (Left)")

    # Plot receivers
    plt.scatter(
        [receiver1[0], receiver2[0]],
        [receiver1[1], receiver2[1]],
        c="g",
        marker="o",
        label="Receivers",
    )
    # Plot source
    plt.scatter(x_s, y_s, c="b", marker="x", label="Source")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Hyperbola Defined by Two Receivers")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid()
    plt.show()


    """
    x1, y1 = receiver1
    x2, y2 = receiver2
    x_s, y_s = source

    # Compute a
    r_s1 = np.sqrt((x_s - receiver1[0]) ** 2 + (y_s - receiver1[1]) ** 2)
    r_s2 = np.sqrt((x_s - receiver2[0]) ** 2 + (y_s - receiver2[1]) ** 2)
    a = 1 / 2 * np.abs(r_s1 - r_s2)

    # Compute the center and distance between receivers
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Distance between receivers

    if a >= d / 2:
        raise ValueError(
            "a must be smaller than half the distance between the receivers"
        )

    # Compute b using standard hyperbola formula: b^2 = (d/2)^2 - a^2
    b = np.sqrt((d / 2) ** 2 - a**2)

    # Parametric equations of the hyperbola
    t = np.linspace(-tmax, tmax, num_points)  # Parameter values for hyperbola branches
    x_hyper_right = a * np.cosh(t)  # Right branch
    y_hyper = b * np.sinh(t)

    x_hyper_left = -x_hyper_right  # Left branch

    # Rotate and translate hyperbola to align with the receivers
    angle = np.arctan2(y2 - y1, x2 - x1)  # Angle of the hyperbola axis
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)

    # Apply rotation and translation
    x_right = cx + x_hyper_right * cos_theta - y_hyper * sin_theta
    y_right = cy + x_hyper_right * sin_theta + y_hyper * cos_theta

    x_left = cx + x_hyper_left * cos_theta - y_hyper * sin_theta
    y_left = cy + x_hyper_left * sin_theta + y_hyper * cos_theta

    return (x_right, y_right), (x_left, y_left)


if __name__ == "__main__":

    """Gather acronyms and bibliographies from a manuscript folder"""
    # # Usage
    # manuscript_folder = r"C:\Users\baptiste.menetrier\Desktop\rapports\manuscript"
    # output_file = r"C:\Users\baptiste.menetrier\Desktop\rapports\glossary_acoustics.tex"

    # # gather_acronyms(manuscript_folder, output_file)

    # Usage
    manuscript_folder = r"C:\Users\baptiste.menetrier\Desktop\rapports\manuscript"
    output_file = r"C:\Users\baptiste.menetrier\Desktop\rapports\biblio_acoustics.bib"
    gather_bibliographies(manuscript_folder, output_file)

    """ Count publications per year """
    # # Exemple d'utilisation de la fonction
    # file_path = r"C:\Users\baptiste.menetrier\Desktop\doc\biblio\Localisation\lens_mfp_publi_1970_2023_4journals.csv"
    # output_filename = "lens_mfp_publi_1970_2023_4journals_per_year"
    # result_df, result_filepath = count_publications_per_year(file_path, output_filename)
    # print(result_df)

    # # Utilisation
    # export_to_dat(result_df, result_filepath)


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
