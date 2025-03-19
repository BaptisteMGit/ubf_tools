#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_implementation_draft.py
@Time    :   2025/01/27 12:04:42
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from skimage import measure
from sklearn import preprocessing
from sklearn.cluster import KMeans
from misc import cast_matrix_to_target_shape
from propa.kraken_toolbox.run_kraken import readshd
from propa.rtf.rtf_utils import D_hermitian_angle_fast
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import params
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_TMP,
    ROOT_DATA,
    ROOT_IMG,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    plot_ambiguity_surface,
)


# ======================================================================================================================
# Functions
# ======================================================================================================================


def main_lobe_segmentation_study():

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load full array dataset
    fpath = os.path.join(
        ROOT_DATA,
        f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
    )
    ds_fa = xr.open_dataset(fpath)

    dist = "d_gcc"
    amb_surf = ds_fa[dist]

    ### 1) Double k-means approach ###
    # 1.1) K-means clustering on the ambiguity surface level

    # Reshape ambiguity surface to 1D array
    amb_surf_1d = amb_surf.values.flatten()

    # Apply K-means clustering
    n_clusters = 8
    x_coord, y_coord = np.meshgrid(ds_fa.x.values, ds_fa.y.values)
    X = np.vstack(
        [x_coord.flatten(), y_coord.flatten(), amb_surf_1d]
    )  # 3 Columns x, y, S(x, y)
    X_norm = preprocessing.normalize(X).T
    X_norm[:, 0:2] *= 2  # Increase the weight of the spatial coordinates
    # X_norm = X.T
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans.fit(X_norm)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
    #     amb_surf_1d.reshape(-1, 1)
    # )

    # 1.2) Segmentation
    # Reshape labels to 2D array
    labels = kmeans.labels_.reshape(amb_surf.shape)

    # 1.3) Plot
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Define a discrete colormap with n_clusters colors
    cmap = plt.get_cmap("jet", n_clusters)
    im = ax.pcolormesh(
        ds_fa["x"].values * 1e-3,
        ds_fa["y"].values * 1e-3,
        # amb_surf.values.T,
        labels.T,
        cmap=cmap,
        # cmap="jet",
        # vmin=-10,
        # vmax=0,
    )

    # Add colorbar with n_clusters ticks
    cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{Class}$", ticks=range(n_clusters))
    # cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{Class}$")
    ax.set_title(f"Full array")
    ax.set_xlabel(r"$x \textrm{[km]}$")
    ax.set_ylabel(r"$y \, \textrm{[km]}$")
    ax.set_xticks([3.500, 4.000, 4.500])
    ax.set_yticks([6.400, 6.900, 7.400])

    # # Plot segmentation
    # for i in range(n_clusters):
    #     mask = labels == i
    #     ax.contour(mask, levels=[0.5])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_segmentation.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")

    # 1.4) Select the class corresponding to the estimated position defined by the maximum of the ambiguity surface
    x_idx, y_idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
    x_src_hat = amb_surf.x[x_idx]
    y_src_hat = amb_surf.y[y_idx]
    src_hat_class = labels[x_idx, y_idx]

    # Find contours of src_hat_class and select the contour corresponding to the estimated position
    contours = measure.find_contours(labels == src_hat_class, level=0.5)
    for contour in contours:
        # Check if src_hat is within the contour
        idx_x_min = np.min(contour[:, 0].astype(int))
        idx_x_max = np.max(contour[:, 0].astype(int))
        idx_y_min = np.min(contour[:, 1].astype(int))
        idx_y_max = np.max(contour[:, 1].astype(int))
        if (idx_x_min <= x_idx <= idx_x_max) and (idx_y_min <= y_idx <= idx_y_max):
            break

    # 1.5) Plot ambiguity surface and highligh pixels falling into the src_hat_class
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Define a discrete colormap with n_clusters colors
    im = ax.pcolormesh(
        ds_fa["x"].values * 1e-3,
        ds_fa["y"].values * 1e-3,
        amb_surf.values.T,
        cmap="jet",
        vmin=-10,
        vmax=0,
    )

    # # Highligh mainlobe pixels
    ax.plot(
        ds_fa["x"].values[contour[:, 0].astype(int)] * 1e-3,
        ds_fa["y"].values[contour[:, 1].astype(int)] * 1e-3,
        color="k",
        linewidth=2,
    )

    # mask = labels == src_hat_class
    # ax.contour(
    #     ds_fa["x"].values * 1e-3,
    #     ds_fa["y"].values * 1e-3,
    #     mask.T,
    #     levels=[0.5],
    #     colors="k",
    # )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r"$\textrm{[dB]}$")
    ax.scatter(
        x_src_hat * 1e-3,
        y_src_hat * 1e-3,
        facecolors="none",
        edgecolors="k",
        label="Estimated source position",
        s=20,
        linewidths=3,
    )

    ax.set_title(f"Full array")
    ax.set_xlabel(r"$x \textrm{[km]}$")
    ax.set_ylabel(r"$y \, \textrm{[km]}$")
    ax.set_xticks([3.500, 4.000, 4.500])
    ax.set_yticks([6.400, 6.900, 7.400])

    # Save figure
    fpath = os.path.join(ROOT_IMG, "loc_zhang2023_fig5_segmentation_highlight.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")


def localise(sub_array=None):

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load rtf data
    # fpath = os.path.join(ROOT_DATA, "rtf_zhang.nc")
    fpath = os.path.join(ROOT_DATA, f"rtf_zhang_dx{grid['dx']}m_dy{grid['dy']}m.nc")
    ds = xr.open_dataset(fpath)

    # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
    # Match field processing #
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 3,
        "unit": "deg",
        "apply_mean": True,
    }

    # Select a few frequencies
    nf = 10
    df = np.diff(ds.f.values)[0]
    f_loc = np.random.choice(ds.f.values, nf)
    ds = ds.sel(f=f_loc)

    # Select a few receivers
    # idx_rcv = [0, 1, 4, 5]
    if sub_array is not None:
        ds = ds.sel(idx_rcv=sub_array)

    ## RTF ##
    # Compute distance bewteen the estimated RTF and RTF at each grid point
    rtf_src = ds.rtf_src_real.values + 1j * ds.rtf_src_imag.values
    rtf_grid = ds.rtf_real.values + 1j * ds.rtf_imag.values

    theta = dist_func(rtf_src, rtf_grid, **dist_kwargs)

    # Add theta to dataset
    ds["theta"] = (["x", "y"], theta.T)
    amb_surf = ds.theta

    # Plot ambiguity surfaces
    plot_args = {
        "dist": "hermitian_angle",
        "vmax_percentile": 5,
        "root_img": ROOT_IMG,
        "testcase": "zhang_et_al_2023",
        "dist_label": r"$\theta \, \textrm{[°]}$",
        "vmax": 50,
        "vmin": 0,
        "sub_array": sub_array,
    }

    plot_ambiguity_surface(
        amb_surf=amb_surf, source=source, plot_args=plot_args, loc_arg="min"
    )

    # Convert theta to a metric between -1 and 1
    theta_max = 90
    theta_inv = theta_max - amb_surf  # So that the source position is the maximum value
    d = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1
    # q = (np.max(amb_surf) - amb_surf) / (np.max(amb_surf) - np.min(amb_surf))

    # Add d to dataset
    # ds["d"] = (["x", "y"], d)
    # amb_surf = ds.theta

    plot_args["dist"] = "normalized_metric"
    plot_args["dist_label"] = r"$d_{rtf}$"
    plot_args["vmax"] = 1
    plot_args["vmin"] = -0.2
    plot_ambiguity_surface(
        amb_surf=d, source=source, plot_args=plot_args, loc_arg="max"
    )

    ## GCC-SCOT ##
    gcc_src = ds.gcc_src_real.values + 1j * ds.gcc_src_imag.values
    gcc_grid = ds.gcc_real.values + 1j * ds.gcc_imag.values

    # Cast gcc_src to the same shape as gcc_grid
    gcc_src = cast_matrix_to_target_shape(gcc_src, gcc_grid.shape)

    # Build cross corr (Equation (8) in Zhang et al. 2023)
    d = np.abs(np.sum(gcc_grid * np.conj(gcc_src) * df, axis=0))
    d = d / np.max(d)

    plot_args["dist"] = "gcc_scot"
    plot_args["dist_label"] = r"$d$"
    plot_args["vmax"] = 1
    plot_args["vmin"] = -0.2
    for i in range(len(receivers["x"])):
        plot_args["sub_array"] = [0, i]
        amb_surf_i = d[..., i].T
        amb_surf_da = xr.DataArray(
            amb_surf_i,
            coords={"x": ds.x.values, "y": ds.y.values},
            dims=["x", "y"],
        )

        plot_ambiguity_surface(
            amb_surf=amb_surf_da, source=source, plot_args=plot_args, loc_arg="max"
        )


def save_simulation_netcdf():

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Frequency
    f = frequency["freqs"]

    # Read shd from previously run kraken
    # working_dir = os.path.join(ROOT, "tmp")
    working_dir = ROOT_TMP
    os.chdir(working_dir)
    shdfile = r"testcase_zhang2023.shd"

    _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
        filename=shdfile, freq=f
    )

    tf = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)

    # Define xarray dataset to store results
    tf_zhang = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "r"],
                np.real(tf),
            ),
            tf_imag=(["f", "r"], np.imag(tf)),
        ),
        coords={
            "f": f,
            "r": field_pos["r"]["r"],
        },
    )

    # Save waveguide transfert functions as netcdf
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    tf_zhang.to_netcdf(fpath)
    tf_zhang.close()


# Perf vs subarrays (ie nb of rcv in subarrays)
# rmse_gcc = [rmse["dr_gcc"].loc[snr] for rmse in rmse_]
# rmse_rtf = [rmse["dr_rtf"].loc[snr] for rmse in rmse_]
# plt.figure(figsize=(8, 6))
# plt.plot(nr_in_sa, rmse_gcc, "o-", label="DCF")
# plt.plot(nr_in_sa, rmse_rtf, "o-", label="RTF")
# plt.xlabel("Number of receivers")
# plt.ylabel("RMSE [m]")
# plt.title(f"SNR = {snr} dB")
# plt.legend()

# fpath = os.path.join(root_img, f"rmse_subarrays_snr{snr}.png")
# plt.savefig(fpath, dpi=300)
# plt.close("all")

# # Plot DR vs nr_in_sa
# dr_gcc = [dr["dr_gcc"].loc[snr] for dr in dr_mu]
# dr_rtf = [dr["dr_rtf"].loc[snr] for dr in dr_mu]
# plt.figure(figsize=(8, 6))
# plt.plot(nr_in_sa, dr_gcc, "o-", label="DCF")
# plt.plot(nr_in_sa, dr_rtf, "o-", label="RTF")
# plt.xlabel("Number of receivers")
# plt.ylabel("DR [m]")
# plt.title(f"SNR = {snr} dB")
# plt.legend()

# fpath = os.path.join(root_img, f"dr_subarrays_snr{snr}.png")
# plt.savefig(fpath, dpi=300)

# # Plot MSR vs nr_in_sa
# msr_gcc = [msr["d_gcc"].loc[snr] for msr in msr_mu]
# msr_rtf = [msr["d_rtf"].loc[snr] for msr in msr_mu]
# plt.figure(figsize=(8, 6))
# plt.plot(nr_in_sa, msr_gcc, "o-", label="DCF")
# plt.plot(nr_in_sa, msr_rtf, "o-", label="RTF")
# plt.xlabel("Number of receivers")
# plt.ylabel("MSR [dB]")
# plt.title(f"SNR = {snr} dB")
# plt.legend()

# fpath = os.path.join(root_img, f"msr_subarrays_snr{snr}.png")
# plt.savefig(fpath, dpi=300)

# # Plot results
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.errorbar(
#     msr_mean.index,
#     msr_mean["d_gcc"],
#     yerr=msr_std["d_gcc"],
#     fmt="o-",
#     label=r"$\textrm{DCF}$",
# )
# ax.errorbar(
#     msr_mean.index,
#     msr_mean["d_rtf"],
#     yerr=msr_std["d_rtf"],
#     fmt="o-",
#     label=r"$\textrm{RTF}$",
# )
# ax.set_xlabel(r"$\textrm{SNR [dB]}$")
# ax.set_ylabel(r"$\textrm{MSR [dB]}$")
# ax.legend()
# ax.grid()
# # plt.show()
# rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
# plt.suptitle(f"Receivers = ({rcv_str})")

# fpath = os.path.join(root_img, f"msr_snr_{sa_item['array_label']}.png")
# plt.savefig(fpath)
# plt.close("all")

# # Plot results
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.errorbar(
#     dr_mean.index,
#     dr_mean["dr_gcc"],
#     yerr=dr_std["dr_gcc"],
#     fmt="o-",
#     label=r"$\textrm{DCF}$",
# )
# ax.errorbar(
#     dr_mean.index,
#     dr_mean["dr_rtf"],
#     yerr=dr_std["dr_rtf"],
#     fmt="o-",
#     label=r"$\textrm{RTF}$",
# )
# ax.set_xlabel(r"$\textrm{SNR [dB]}$")
# ax.set_ylabel(r"$\Delta_r \textrm{[m]}$")
# ax.legend()
# ax.grid()

# rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
# plt.suptitle(f"Receivers = ({rcv_str})")

# fpath = os.path.join(root_img, f"dr_pos_snr_{sa_item['array_label']}.png")
# plt.savefig(fpath)

# # Plot results
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(rmse.index, rmse["dr_gcc"], "o-", label=r"$\textrm{DCF}$")
# ax.plot(rmse.index, rmse["dr_rtf"], "o-", label=r"$\textrm{RTF}$")
# ax.set_xlabel(r"$\textrm{SNR [dB]}$")
# ax.set_ylabel(r"$\textrm{RMSE [m]}$")
# ax.legend()
# ax.grid()

# rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sa_item["idx_rcv"]]) + "$"
# plt.suptitle(f"Receivers = ({rcv_str})")

# fpath = os.path.join(root_img, f"rmse_snr_{sa_item['array_label']}.png")
# plt.savefig(fpath)
# plt.close("all")

if __name__ == "__main__":
    pass
