#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_variation.py
@Time    :   2024/09/27 16:30:32
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from propa.rtf.ideal_waveguide import *
from propa.rtf.rtf_estimation_const import ROOT_RTF_DATA
from propa.rtf.rtf_utils import (
    D_frobenius,
    D_hermitian_angle,
    D1,
    D2,
    D_hermitian_angle_fast,
)
from publication.PublicationFigure import PubFigure
from mpl_toolkits.axes_grid1 import make_axes_locatable

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=16, ticks_fontsize=20)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def PI_var_r(varin):

    # Read input vars
    depth = varin.get("depth")
    f = varin.get("f")
    r_src = varin.get("r_src")
    z_src = varin.get("z_src")
    x_rcv = varin.get("x_rcv")
    z_rcv = varin.get("z_rcv")
    covered_range = varin.get("covered_range")
    dr = varin.get("dr")
    zmin = varin.get("zmin")
    zmax = varin.get("zmax")
    dz = varin.get("dz")
    dist = varin.get("dist")
    smooth_tf = varin.get("smooth_tf")
    bottom_bc = varin.get("bottom_bc")

    # Create folder to store the images
    root, folder = get_folderpath(
        x_rcv,
        r_src,
        z_src,
        bottom_bc=bottom_bc,
    )
    subfolder = f"dr_{dr}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Create the list of potential source positions
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    r_src_list = np.unique(np.sort(np.append(r_src_list, r_src)))
    z_src_list = np.unique(np.sort(np.append(z_src_list, z_src)))

    ds, loaded = load_rtf_data(r=r_src_list, z=z_src_list)
    if loaded:
        g_r = ds["PI_rz_real"] + 1j * ds["PI_rz_imag"]
        g_r = g_r.sel(z=z_src).values
        g_r = np.expand_dims(g_r, axis=-1)
        g_ref = ds["PI_ref_real"] + 1j * ds["PI_ref_imag"]
        g_ref = np.expand_dims(g_ref.values, axis=(1, 3))

    else:
        raise ValueError("Data not found")

    # Expand g_ref to the same shape as g_r
    tile_shape = tuple([g_r.shape[i] - g_ref.shape[i] + 1 for i in range(g_r.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    #  apply_log, _, title, label, ylabel, _
    dist_func, dist_kwargs, dist_properties = pick_distance_to_apply(dist, axis="r")

    total_dist = dist_func(g_ref, g_r, **dist_kwargs)
    range_displacement = r_src_list - r_src

    # Convert to dB
    if dist_properties["apply_log"]:
        total_dist += 1
        total_dist = 10 * np.log10(total_dist)

    # Plot generalized distance map
    plt.figure()
    plt.plot(
        range_displacement,
        total_dist,
    )
    plt.ylabel(dist_properties["ylabel"])
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.title(dist_properties["title"])
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist}_r_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)

    plt.figure()

    for i_rcv in range(n_rcv):

        g_ref_expanded_single_rcv = np.expand_dims(
            g_ref_expanded[:, :, i_rcv, :], axis=2
        )
        g_r_single_rcv = np.expand_dims(g_r[:, :, i_rcv, :], axis=2)
        # Append a ones array to get shape corresponding to a two receiver case : (nf, nr, 2, nz)
        g_ref_expanded_single_rcv = np.concatenate(
            [np.ones_like(g_ref_expanded_single_rcv), g_ref_expanded_single_rcv], axis=2
        )
        g_r_single_rcv = np.concatenate(
            [np.ones_like(g_r_single_rcv), g_r_single_rcv], axis=2
        )

        single_rcv_pair_dist = dist_func(
            g_ref_expanded_single_rcv, g_r_single_rcv, **dist_kwargs
        )
        plt.plot(
            range_displacement,
            single_rcv_pair_dist,
            label=dist_properties["label"] + r"$\,\, (\Pi_{" + f"{i_rcv},0" + "})$",
        )

    plt.plot(range_displacement, total_dist, label=dist_properties["label"], color="k")
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    # plt.ylim(0, max(d1max, d2max) + 5)
    plt.ylabel(dist_properties["ylabel"])
    plt.title(dist_properties["title"])
    plt.legend()
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist}_r_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_rcvs.png",
    )
    plt.savefig(fpath)
    plt.close("all")


def PI_var_z(varin):

    # Read input vars
    depth = varin.get("depth")
    f = varin.get("f")
    r_src = varin.get("r_src")
    z_src = varin.get("z_src")
    x_rcv = varin.get("x_rcv")
    z_rcv = varin.get("z_rcv")
    covered_range = varin.get("covered_range")
    dr = varin.get("dr")
    zmin = varin.get("zmin")
    zmax = varin.get("zmax")
    dz = varin.get("dz")
    dist = varin.get("dist")
    smooth_tf = varin.get("smooth_tf")
    bottom_bc = varin.get("bottom_bc")

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Create the list of potential source positions
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    r_src_list = np.unique(np.sort(np.append(r_src_list, r_src)))
    z_src_list = np.unique(np.sort(np.append(z_src_list, z_src)))

    ds, loaded = load_rtf_data(r=r_src_list, z=z_src_list)
    if loaded:
        g_z = ds["PI_rz_real"] + 1j * ds["PI_rz_imag"]
        g_z = g_z.sel(r=r_src).values
        g_z = np.expand_dims(g_z, axis=1)
        g_ref = ds["PI_ref_real"].values + 1j * ds["PI_ref_imag"].values
        g_ref = np.expand_dims(g_ref, axis=(1, 3))

    else:
        raise ValueError("Data not found")

    # Expand g_ref to the same shape as g_z
    tile_shape = tuple([g_z.shape[i] - g_ref.shape[i] + 1 for i in range(g_z.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    dist_func, dist_kwargs, dist_properties = pick_distance_to_apply(dist, axis="z")
    total_dist = dist_func(g_ref, g_z, **dist_kwargs)
    depth_displacement = z_src_list - z_src

    # Convert to dB
    if dist_properties["apply_log"]:
        total_dist += 1
        total_dist = 10 * np.log10(total_dist)

    # Plot generalized distance map
    plt.figure()
    plt.plot(
        depth_displacement,
        total_dist,
    )
    plt.ylabel(dist_properties["ylabel"])
    plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
    plt.title(dist_properties["title"])
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist}_z_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath)

    plt.figure()
    # Iterate over receivers
    for i_rcv in range(n_rcv):
        g_ref_expanded_single_rcv = np.expand_dims(
            g_ref_expanded[:, :, i_rcv, :], axis=2
        )
        g_z_single_rcv = np.expand_dims(g_z[:, :, i_rcv, :], axis=2)
        # Append a ones array to get shape corresponding to a two receiver case : (nf, nr, 2, nz)
        g_ref_expanded_single_rcv = np.concatenate(
            [np.ones_like(g_ref_expanded_single_rcv), g_ref_expanded_single_rcv], axis=2
        )
        g_z_single_rcv = np.concatenate(
            [np.ones_like(g_z_single_rcv), g_z_single_rcv], axis=2
        )

        single_rcv_pair_dist = dist_func(
            g_ref_expanded_single_rcv, g_z_single_rcv, **dist_kwargs
        )
        plt.plot(
            depth_displacement,
            single_rcv_pair_dist,
            label=dist_properties["label"] + r"$\,\, (\Pi_{" + f"{i_rcv},0" + "})$",
        )

    plt.plot(depth_displacement, total_dist, label=dist_properties["label"], color="k")
    plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
    plt.ylabel(dist_properties["ylabel"])
    plt.title(dist_properties["title"])
    plt.legend(loc="upper left")
    # plt.ylim(0, max(d1max, d2max) + 5)
    plt.grid()

    # Save
    fpath = os.path.join(
        root,
        f"{dist}_z_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_rcvs.png",
    )
    plt.savefig(fpath)
    plt.close("all")


def Pi_var_rz(varin):
    # Read input vars
    depth = varin.get("depth")
    f = varin.get("f")
    r_src = varin.get("r_src")
    z_src = varin.get("z_src")
    x_rcv = varin.get("x_rcv")
    z_rcv = varin.get("z_rcv")
    covered_range = varin.get("covered_range")
    dr = varin.get("dr")
    zmin = varin.get("zmin")
    zmax = varin.get("zmax")
    dz = varin.get("dz")
    dist = varin.get("dist")
    bottom_bc = varin.get("bottom_bc")

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"dr_{dr}_dz_{dz}"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)

    # Create the list of potential source positions
    r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
    zmin = np.max([1, zmin])  # Avoid negative depths
    z_src_list = np.arange(zmin, zmax, dz)

    # Ensure that the ref position is included in the list
    r_src_list = np.unique(np.sort(np.append(r_src_list, r_src)))
    z_src_list = np.unique(np.sort(np.append(z_src_list, z_src)))

    ds, loaded = load_rtf_data(r=r_src_list, z=z_src_list)
    if loaded:
        g_rz = ds["PI_rz_real"].values + 1j * ds["PI_rz_imag"].values
        g_ref = ds["PI_ref_real"].values + 1j * ds["PI_ref_imag"].values
        g_ref = np.expand_dims(g_ref, axis=(1, 3))

    else:
        # Cast to required shape depending on the number of receivers
        r_src_list_2D = np.tile(r_src_list, (len(x_rcv), 1)).T
        r_src_rcv_ref = r_src_list_2D - x_rcv[0]
        r_src_rcv = r_src_list_2D - x_rcv

        # Derive the reference RTF vector, g_ref is of shape (nf, nr, nrcv, 1)
        r_ref = r_src - x_rcv
        f, g_ref = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_ref[0],
            r=r_ref,
            bottom_bc=bottom_bc,
        )

        print(
            "Computing the RTF for the source varying along the range and depth axes ..."
        )
        t0 = time()
        f, g_rz = g_mat(
            f,
            z_src_list,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )
        print(f"Elapsed time : {time() - t0:.2f} s")

        # Save as netcdf
        save_rtf(
            f=f,
            r=r_src_list,
            z=z_src_list,
            x_rcv=x_rcv,
            PI_rz=g_rz,
            Pi_ref=g_ref,
            r_src=r_src,
            z_src=z_src,
        )

    # Expand g_ref to the same shape as g_rz
    tile_shape = tuple([g_rz.shape[i] - g_ref.shape[i] + 1 for i in range(g_rz.ndim)])
    g_ref_expanded = np.tile(g_ref, tile_shape)

    dist_func, dist_kwargs, dist_properties = pick_distance_to_apply(dist, axis="rz")

    total_dist = dist_func(g_ref, g_rz, **dist_kwargs)

    range_displacement = r_src_list - r_src
    depth_displacement = z_src_list - z_src

    # Convert to dB
    if dist_properties["apply_log"]:
        total_dist += 1
        total_dist = 10 * np.log10(total_dist)

    # Define common vmin and vmax
    vmin = 0
    vmax = np.percentile(total_dist, 50)
    cmap = "jet_r"  # "binary"
    aspect = (
        "equal"
        if np.round(range_displacement[-1] - range_displacement[0], 0)
        == np.round(depth_displacement[-1] - depth_displacement[0], 0)
        else "auto"
    )

    # Plot generalized distance map
    pad = 0.2
    cbar_width = "3%"
    _, ax = plt.subplots(1, 1)
    pc = plt.pcolormesh(
        range_displacement,
        depth_displacement,
        total_dist.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
    plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
    plt.title(dist_properties["title"])
    ax.set_aspect(aspect, adjustable="box")
    divider1 = make_axes_locatable(ax)
    cax = divider1.append_axes("right", size=cbar_width, pad=pad)
    plt.colorbar(pc, cax=cax, label=dist_properties["colorbar_title"])

    # Save
    fpath = os.path.join(
        root,
        f"{dist}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
    )
    plt.savefig(fpath, bbox_inches="tight", dpi=300)

    for i_rcv in range(n_rcv):

        g_ref_expanded_single_rcv = np.expand_dims(
            g_ref_expanded[:, :, i_rcv, :], axis=2
        )
        g_rz_single_rcv = np.expand_dims(g_rz[:, :, i_rcv, :], axis=2)
        # Append a ones array to get shape corresponding to a two receiver case : (nf, nr, 2, nz)
        g_ref_expanded_single_rcv = np.concatenate(
            [np.ones_like(g_ref_expanded_single_rcv), g_ref_expanded_single_rcv], axis=2
        )
        g_rz_single_rcv = np.concatenate(
            [np.ones_like(g_rz_single_rcv), g_rz_single_rcv], axis=2
        )

        single_rcv_pair_dist = dist_func(
            g_ref_expanded_single_rcv, g_rz_single_rcv, **dist_kwargs
        )

        _, ax = plt.subplots(1, 1)
        pc = plt.pcolormesh(
            range_displacement,
            depth_displacement,
            single_rcv_pair_dist.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
        plt.ylabel(r"$z - z_s \, \textrm{[m]}$")
        plt.title(dist_properties["label"] + r"$\,\, (\Pi_{" + f"{i_rcv},0" + "})$")
        ax.set_aspect(aspect, adjustable="box")
        divider1 = make_axes_locatable(ax)
        cax = divider1.append_axes("right", size=cbar_width, pad=pad)
        plt.colorbar(pc, cax=cax, label=dist_properties["colorbar_title"])

        # Save
        fpath = os.path.join(
            root,
            f"{dist}_rcv_{i_rcv}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}.png",
            # f"D2_rcv_{i_rcv+1}_rr_{range_displacement[0]:.0f}_{range_displacement[-1]:.0f}_dr_{dr}_zz_{depth_displacement[0]:.0f}_{depth_displacement[-1]:.0f}_dr_{dz}.png",
        )
        plt.savefig(fpath, bbox_inches="tight", dpi=300)

        plt.close("all")


def save_rtf(f, r, z, x_rcv, PI_rz, Pi_ref, r_src, z_src):
    # PI_rz is of shape (nf, nr, nrcv, nz)
    ds = xr.Dataset(
        data_vars={
            "PI_rz_real": (["f", "r", "idx_rcv", "z"], PI_rz.real),
            "PI_rz_imag": (["f", "r", "idx_rcv", "z"], PI_rz.imag),
            "PI_ref_real": (["f", "idx_rcv"], Pi_ref.squeeze().real),
            "PI_ref_imag": (["f", "idx_rcv"], Pi_ref.squeeze().imag),
            "x_rcv": (["idx_rcv"], x_rcv),
        },
        coords={
            "f": f,
            "r": r,
            "idx_rcv": np.arange(len(x_rcv)),
            "z": z,
        },
        attrs={
            "description": f"Dataset containing the RTF for a source varying along the range and depth axes in ideal waveguide.",
        },
    )

    ds["f"].attrs["units"] = "Hz"
    ds["r"].attrs["units"] = "m"
    ds["z"].attrs["units"] = "m"
    ds["x_rcv"].attrs["units"] = "m"
    ds["PI_rz_real"].attrs[
        "description"
    ] = "RTF for a source varying along the range and depth axes."
    ds["PI_ref_real"].attrs[
        "description"
    ] = f"RTF for a reference source position at r={r_src}m, z={z_src}m."

    dr = np.round(r[1] - r[0], 2)
    dz = np.round(z[1] - z[0], 2)

    fname = (
        f"PI_rz_r_{r[0]:.0f}_{r[-1]:.0f}_z_{z[0]:.0f}_{z[-1]:.0f}_dr_{dr}_dz_{dz}.nc"
    )
    fpath = os.path.join(ROOT_RTF_DATA, fname)

    ds.to_netcdf(fpath)


def load_rtf_data(r, z):
    dr = np.round(r[1] - r[0], 2)
    dz = np.round(z[1] - z[0], 2)

    fname = (
        f"PI_rz_r_{r[0]:.0f}_{r[-1]:.0f}_z_{z[0]:.0f}_{z[-1]:.0f}_dr_{dr}_dz_{dz}.nc"
    )
    fpath = os.path.join(ROOT_RTF_DATA, fname)

    # Check if file exists
    if not os.path.exists(fpath):
        loaded = False  # File does not exist
        ds = None
    else:
        ds = xr.open_dataset(fpath)
        loaded = True  # File exists

    return ds, loaded


def get_folderpath(x_rcv, r_src, z_src, bottom_bc="pressure_release"):
    delta_rcv = x_rcv[1] - x_rcv[0]
    n_rcv = len(x_rcv)
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\ideal_waveguide"
    testname = f"nrcv_{n_rcv}_deltarcv_{delta_rcv:.0f}m_rsrc_{r_src*1e-3:.0f}km_zsrc_{z_src:.0f}m"
    folder = os.path.join(bottom_bc, testname)
    return root, folder


def full_test(
    covered_range,
    dr,
    zmin,
    zmax,
    dz,
    dist="D2",
    bottom_bc="pressure_release",
):

    # Load default parameters
    depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f = default_params()

    # Define the receivers position
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # Input vars
    varin = {
        "depth": depth,
        "f": f,
        "r_src": r_src,
        "z_src": z_src,
        "x_rcv": x_rcv,
        "z_rcv": z_rcv,
        "covered_range": covered_range,
        "dr": dr,
        "zmin": zmin,
        "zmax": zmax,
        "dz": dz,
        "dist": dist,
        "bottom_bc": bottom_bc,
    }

    # Variations along the range and depth axes
    Pi_var_rz(varin)

    # Variations along the range axis
    PI_var_r(varin)

    # Variations along the depth axis
    PI_var_z(varin)

    # Create folder to store the images
    root, folder = get_folderpath(x_rcv, r_src, z_src, bottom_bc=bottom_bc)
    subfolder = f"tl_f"
    root = os.path.join(root, folder, subfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    n_rcv = len(x_rcv)
    delta_rcv = x_rcv[1] - x_rcv[0]
    r_rcv_0 = r_src
    z_rcv_0 = z_rcv
    r_rcv_plot = np.array([r_rcv_0 + delta_rcv * i for i in range(n_rcv)])
    z_rcv_plot = np.array([z_rcv_0] * n_rcv)

    rmax = np.max(r_rcv_plot) + 1e3
    zmax = max(np.max(z_rcv), np.max(z_src), depth)
    r = np.linspace(1, rmax, int(1e3))
    z = np.linspace(0, zmax, 100)

    freq_to_plot = [1, 5, 10, 20, 50]
    f, _, _, p_field = field(freq_to_plot, z_src, r, z, depth)
    for fp in freq_to_plot:
        plot_tl(
            f,
            r,
            z,
            p_field,
            f_plot=fp,
            z_src=z_src,
            r_rcv=r_rcv_plot,
            z_rcv=z_rcv_plot,
            show=False,
        )

        plt.savefig(os.path.join(root, f"tl_f{fp}Hz.png"))

    # Zoom on receivers
    rmin = np.min(r_rcv_plot) - 1000
    rmax = np.max(r_rcv_plot) + 1000
    r = np.linspace(rmin, rmax, int(1e3))
    z = np.linspace(zmax - 20, zmax, 100)

    freq_to_plot = [1, 5, 10, 20, 50]
    f, _, _, p_field = field(freq_to_plot, z_src, r, z, depth)
    for fp in freq_to_plot:
        plot_tl(
            f,
            r,
            z,
            p_field,
            f_plot=fp,
            z_src=None,
            r_rcv=r_rcv_plot,
            z_rcv=z_rcv_plot,
            show=False,
        )

        plt.savefig(os.path.join(root, f"tl_f{fp}Hz_zoomrcv.png"))


def sensibility_ideal_waveguide(
    param="delta_rcv", axis="both", bottom_bc="pressure_release", dist="hermitian_angle"
):
    """Study the distance metric(D_frobenius or hermitian_angle) sensibility.
    The source position is varied along the range axis and the depth axis. The distance metric is computed for each source position.
    The sensibility along each axis is evaluated separately by computing the main lobe aperture of the distance metric.
    Several parameters are studied :
       - The distance between receivers (delta_rcv)
       - The reference source range (r_src)
       - The reference source depth (z_src)
       - The number of receivers (n_rcv)
    """

    # Sensibility to the distance between receivers
    if param == "delta_rcv":
        sensibility_ideal_waveguide_delta_rcv(axis=axis, bottom_bc=bottom_bc, dist=dist)

    # Sensibility to the reference source range
    if param == "r_src":
        sensibility_ideal_waveguide_r_src(axis=axis, bottom_bc=bottom_bc, dist=dist)

    # Sensibility to the reference source depth
    if param == "z_src":
        sensibility_ideal_waveguide_z_src(axis=axis, bottom_bc=bottom_bc, dist=dist)

    # Sensibility to the number of receivers
    if param == "n_rcv":
        sensibility_ideal_waveguide_n_rcv(axis=axis, bottom_bc=bottom_bc, dist=dist)

    # Sensibility to the frequency resolution
    if param == "df":
        sensibility_ideal_waveguide_df(axis=axis, bottom_bc=bottom_bc, dist=dist)


def sensibility_ideal_waveguide_2d(
    param_couple=["n_rcv", "delta_rcv"],
    axis="both",
    bottom_bc="pressure_release",
    dist="hermitian_angle",
):
    """
    Study the distance metric (D_frobenius or hermitian_angle) sensibility as a function of two params.
    """

    # Sensibility to couple of parameters n_rcv and delta_rcv
    if "n_rcv" in param_couple and "delta_rcv" in param_couple:
        sensibility_ideal_waveguide_2d_n_rcv_delta_rcv(
            axis=axis, bottom_bc=bottom_bc, dist=dist
        )


def derive_Pi_ref(r_src, z_src, x_rcv, f, axis="r", bottom_bc="pressure_release"):

    depth, _, _, z_rcv, _, _, _ = default_params()

    if axis == "r":
        # Define the ranges
        dr = 0.1
        covered_range = 30  # Assume the range aperture is smaller than 30 m
        r_src_list = np.arange(r_src - covered_range, r_src + covered_range, dr)
        # Ensure that the ref position is included in the list
        r_src_list = np.sort(np.append(r_src_list, r_src))

        # Cast to required shape depending on the number of receivers
        r_src_list_2D = np.tile(r_src_list, (len(x_rcv) - 1, 1)).T
        nb_pos_r = len(r_src_list)
        r_src_rcv_ref = r_src_list_2D - x_rcv[0]
        r_src_rcv = r_src_list_2D - x_rcv[1:]

        # Derive the RTF vector for the reference source position
        r_ref = r_src - x_rcv
        # Derive the reference RTF vector g_ref is of shape (nf, nrcv, 1)
        f, g_ref = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_ref[0],
            r=r_ref[1:],
            bottom_bc=bottom_bc,
        )

        return r_src_list, r_src_rcv_ref, r_src_rcv, depth, f, g_ref

    if axis == "z":

        # Define detphs
        dz = 0.1
        zmin = z_src - 10
        zmin = np.max([1, zmin])  # Avoid negative depths
        zmax = z_src + 20
        z_src_list = np.arange(zmin, zmax, dz)

        # Ensure that the ref position is included in the list
        z_src_list = np.sort(np.append(z_src_list, z_src))
        nb_pos_z = len(z_src_list)
        r_src_rcv_ref = r_src - x_rcv[0]
        r_src_rcv = r_src - x_rcv[1:]

        # Derive the reference RTF vector, g_ref is of shape (nf, nrcv, 1)
        r_ref = r_src - x_rcv

        f, g_ref = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_ref[0],
            r=r_ref[1:],
            bottom_bc=bottom_bc,
        )

        return z_src_list, r_src_rcv_ref, r_src_rcv, depth, f, g_ref


def sensibility_ideal_waveguide_delta_rcv(
    axis="r", bottom_bc="pressure_release", dist="hermitian_angle"
):
    # Define the range of the study
    # delta_rcv = np.logspace(0, 4, 10)
    # delta_max = 3 * 1e3
    # alpha = 1.2
    # delta_rcv = np.array(
    #     [1 + x**alpha for x in np.linspace(0, delta_max ** (1 / alpha), 100)]
    # )
    # delta_rcv = np.round(delta_rcv, 1)
    # delta_rcv = np.array(
    #     [
    #         0.1,
    #         0.2,
    #         0.3,
    #         0.5,
    #         0.6,
    #         0.7,
    #         0.8,
    #         0.9,
    #         1,
    #         1.5,
    #         2,
    #         3,
    #         5,
    #         7,
    #         10,
    #         15,
    #         20,
    #         30,
    #         50,
    #         70,
    #         100,
    #         150,
    #         200,
    #         300,
    #         500,
    #         700,
    #         1000,
    #     ]
    # )
    delta_rcv = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
            6,
            7,
            8,
            9,
            10,
            15,
            20,
            30,
            50,
        ]
    )

    _, r_src, z_src, _, n_rcv, _, f = default_params()

    root_img = init_sensibility_path("delta_rcv", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["n_rcv"] = n_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "delta_rcv"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["values"] = delta_rcv
    param_var["xlabel"] = r"$\delta_{r_{rcv}} \, \textrm{[m]}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["dist_name"] = dist

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, delta in enumerate(delta_rcv):

        param_var["idx"] = i_d
        param_var["value"] = delta
        input_var["delta_rcv"] = delta
        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis, dist=dist
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_r_src(
    axis="both", bottom_bc="pressure_release", dist="hermitian_angle"
):
    src_range = np.arange(10, 101, 1) * 1e3
    # src_range = np.arange(27, 33, 1) * 1e3

    # Define the source depth
    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("r_src", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["z_src"] = z_src
    input_var["n_rcv"] = n_rcv
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "r_src"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["values"] = src_range
    param_var["xlabel"] = r"$r_{s} \, \textrm{[m]}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["dist_name"] = dist

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, r_src in enumerate(src_range):

        param_var["idx"] = i_d
        param_var["value"] = r_src
        input_var["r_src"] = r_src

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis, dist=dist
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_z_src(
    axis, bottom_bc="pressure_release", dist="hermitian_angle"
):
    src_depth = np.arange(1, 991, 1)

    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("z_src", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["n_rcv"] = n_rcv
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "z_src"
    param_var["unit"] = "m"
    param_var["root_img"] = root_img
    param_var["values"] = src_depth
    param_var["xlabel"] = r"$z_{s} \, \textrm{[m]}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["dist_name"] = dist

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, z_src in enumerate(src_depth):

        param_var["idx"] = i_d
        param_var["value"] = z_src
        input_var["z_src"] = z_src

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis, dist=dist
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_n_rcv(
    axis, bottom_bc="pressure_release", dist="hermitian_angle"
):
    nb_rcv = np.arange(2, 11, 1)

    _, r_src, z_src, _, n_rcv, delta_rcv, f = default_params()

    root_img = init_sensibility_path("n_rcv", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["delta_rcv"] = delta_rcv
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "n_rcv"
    param_var["unit"] = ""
    param_var["root_img"] = root_img
    param_var["values"] = nb_rcv
    param_var["xlabel"] = r"$n_{rcv}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["dist_name"] = dist

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, n_rcv in enumerate(nb_rcv):

        param_var["idx"] = i_d
        param_var["value"] = n_rcv
        input_var["n_rcv"] = n_rcv

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis, dist=dist
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def sensibility_ideal_waveguide_2d_n_rcv_delta_rcv(
    axis, bottom_bc="pressure_release", dist="hermitian_angle"
):
    # nb_rcv = np.arange(2, 11, 1)
    # delta_rcv = np.arange(0.2, 20.2, 0.2)

    nb_rcv = [2, 3, 4, 5, 6]
    delta_rcv = [1, 5, 10, 20]

    _, r_src, z_src, _, n_rcv, _, f = default_params()

    root_img = init_sensibility_path("n_rcv_delta_rcv", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["df"] = f[1] - f[0]

    # Define param vars
    param_var = {}
    param_var["name"] = "n_rcv_delta_rcv"
    param_var["unit"] = ""
    param_var["root_img"] = root_img
    param_var["values_x"] = nb_rcv
    param_var["values_y"] = delta_rcv
    param_var["xlabel"] = r"$n_{rcv}$"
    param_var["ylabel"] = r"$\delta_{rcv}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["dist_name"] = dist
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["colorbar_title"] = dist_properties["colorbar_title"]

    apertures_rr = []
    apertures_zz = []

    # Compute the distance metric for each delta_rcv
    for i_d, n_rcv in enumerate(nb_rcv):
        apertures_r = []
        apertures_z = []
        for i_d2, d_rcv in enumerate(delta_rcv):

            input_var["n_rcv"] = n_rcv
            input_var["delta_rcv"] = d_rcv

            study_param_sensibility_2d(
                input_var, apertures_r, apertures_z, axis=axis, dist=dist
            )

        apertures_rr.append(apertures_r)
        apertures_zz.append(apertures_z)

    apertures_rr = np.array(apertures_rr)
    apertures_zz = np.array(apertures_zz)

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility_2d(apertures_rr, apertures_zz, param_var, axis=axis)


def sensibility_ideal_waveguide_df(
    axis, bottom_bc="pressure_release", dist="hermitian_angle"
):
    freq_res = np.arange(0.1, 15.1, 0.1)

    _, r_src, z_src, _, n_rcv, delta_rcv, _ = default_params()

    root_img = init_sensibility_path("df", bottom_bc=bottom_bc)

    # Define input vars
    input_var = {}
    input_var["n_rcv"] = n_rcv
    input_var["r_src"] = r_src
    input_var["z_src"] = z_src
    input_var["delta_rcv"] = delta_rcv

    # Define param vars
    param_var = {}
    param_var["name"] = "df"
    param_var["unit"] = "Hz"
    param_var["root_img"] = root_img
    param_var["values"] = freq_res
    param_var["xlabel"] = r"$\Delta_f \, \textrm{Hz}$"

    # Get distance properties for plots
    _, _, dist_properties = pick_distance_to_apply(dist)
    param_var["th_r"] = dist_properties["th_main_lobe"]
    param_var["th_z"] = dist_properties["th_main_lobe"]
    param_var["dist_unit"] = dist_properties["dist_unit"]
    param_var["dist_name"] = dist

    apertures_r = []
    apertures_z = []

    # Compute the distance metric for each delta_rcv
    for i_d, df in enumerate(freq_res):

        param_var["idx"] = i_d
        param_var["value"] = df
        input_var["df"] = df

        study_param_sensibility(
            input_var, param_var, apertures_r, apertures_z, axis=axis, dist=dist
        )

    # Plot the main lobe aperture as a function of delta_rcv
    plot_sensibility(apertures_r, apertures_z, param_var, axis=axis)


def init_sensibility_path(param, bottom_bc="pressure_release"):
    # Create folder to store the images
    root = "C:\\Users\\baptiste.menetrier\\Desktop\\devPy\\phd\\img\\illustration\\rtf\\ideal_waveguide"
    folder = bottom_bc
    subfolder = "sensibility"
    subsubfolder = param
    root = os.path.join(root, folder, subfolder, subsubfolder)
    if not os.path.exists(root):
        os.makedirs(root)

    return root


def total_dist_aperture(total_dist, x, th=10):
    """Derive main lobe aperture along the x axis.

    Aperture is defined as the width of the 3dB main lobe.
    """

    # Find the main lobe
    idx_main_lobe = np.where(total_dist <= th)[0]
    i1 = idx_main_lobe[0]
    i2 = idx_main_lobe[-1]
    aperture = np.abs(x[i1] - x[i2])

    return i1, i2, np.round(aperture, 3)


def study_param_sensibility(
    input_var,
    param_var,
    aperture_r,
    aperture_z,
    axis="both",
    bottom_bc="pressure_release",
    dist="hermitian_angle",
):
    # Load default params
    depth, _, _, z_rcv, _, _, f = default_params()

    # Load input vars
    delta_rcv = input_var["delta_rcv"]
    r_src = input_var["r_src"]
    z_src = input_var["z_src"]
    n_rcv = input_var["n_rcv"]
    df = input_var["df"]

    f = np.arange(f[0], f[-1], df)

    # Load param vars
    name = param_var["name"]
    idx = param_var["idx"]
    value = param_var["value"]
    unit = param_var["unit"]
    root_img = param_var["root_img"]
    th_r = param_var["th_r"]
    th_z = param_var["th_z"]

    # Define the receivers position
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # Get distance properties
    dist_func, dist_kwargs, dist_properties = pick_distance_to_apply(dist)

    ### Range axis ###
    if axis == "r" or axis == "both":
        # Derive ref RTF
        r_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="r", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        total_dist_r = dist_func(g_ref, g, **dist_kwargs)
        range_displacement = r_src_list - r_src

        # Apply log if required
        if dist_properties["apply_log"]:
            total_dist_r += 1
            total_dist_r = 10 * np.log10(total_dist_r)

        # Derive aperure of the main lobe
        # th_r = 3  # threshold
        i1_r, i2_r, ap_r = total_dist_aperture(
            total_dist_r, range_displacement, dist_properties["th_main_lobe"]
        )
        aperture_r.append(ap_r)

    ### Depth axis ###
    if axis == "z" or axis == "both":
        # Derive ref RTF
        z_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="z", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src_list,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        total_dist_z = dist_func(g_ref, g, **dist_kwargs)
        depth_displacement = z_src_list - z_src

        # Apply log if required
        if dist_properties["apply_log"]:
            total_dist_z += 1
            total_dist_z = 10 * np.log10(total_dist_z)

        # Derive aperure of the main lobe
        i1_z, i2_z, ap_z = total_dist_aperture(
            total_dist_z, range_displacement, dist_properties["th_main_lobe"]
        )
        aperture_z.append(ap_z)

    # Plot generalized distance map

    if idx % 5 == 0:
        if axis == "r" or axis == "both":

            # Get distance properties for plots
            _, _, dist_properties = pick_distance_to_apply(dist, axis="r")

            plt.figure()
            plt.plot(
                range_displacement,
                total_dist_r,
            )
            # Add the main lobe aperture
            plt.axvline(x=range_displacement[i1_r], color="r", linestyle="--")
            plt.axvline(x=range_displacement[i2_r], color="r", linestyle="--")
            plt.text(
                range_displacement[i2_r] + 0.15,
                0.8 * np.max(total_dist_r),
                r"$2 r_{"
                + f"{th_r}"
                + r"\textrm{"
                + dist_properties["dist_unit"]
                + r"} } = "
                + f"{ap_r}"
                + r"\, \textrm{[m]}$",
                rotation=90,
                color="r",
                fontsize=12,
            )

            plt.ylabel(dist_properties["ylabel"])
            plt.xlabel(r"$r - r_s \, \textrm{[m]}$")
            plt.title(dist_properties["title"])
            # plt.ylim(0, 100)
            plt.grid()

            param_lab = f"{value:.3f}{unit}"
            root_r = os.path.join(root_img, "r")
            if not os.path.exists(root_r):
                os.makedirs(root_r)

            plt.savefig(os.path.join(root_r, f"{dist}_{name}_{param_lab}.png"))
            plt.close("all")

        if axis == "z" or axis == "both":

            # Get distance properties for plots
            _, _, dist_properties = pick_distance_to_apply(dist, axis="z")

            plt.figure()
            plt.plot(
                range_displacement,
                total_dist_r,
            )

            plt.figure()
            plt.plot(
                depth_displacement,
                total_dist_z,
            )
            # Add the main lobe aperture
            plt.axvline(x=depth_displacement[i1_z], color="r", linestyle="--")
            plt.axvline(x=depth_displacement[i2_z], color="r", linestyle="--")

            plt.text(
                depth_displacement[i2_z] + 0.15,
                0.8 * np.max(total_dist_z),
                r"$2 z_{"
                + f"{th_z}"
                + r"\textrm{"
                + dist_properties["dist_unit"]
                + r"} } = "
                + f"{ap_z}"
                + r"\, \textrm{[m]}$",
                rotation=90,
                color="r",
                fontsize=12,
            )

            plt.ylabel(dist_properties["ylabel"])
            plt.xlabel(r"$z - z_s \, \textrm{[m]}$")
            plt.title(dist_properties["title"])
            # plt.ylim(0, 50)
            plt.grid()

            param_lab = f"{value:.3f}{unit}"
            root_z = os.path.join(root_img, "z")
            if not os.path.exists(root_z):
                os.makedirs(root_z)

            plt.savefig(os.path.join(root_z, f"{dist}_{name}_{param_lab}.png"))
            plt.close("all")


def study_param_sensibility_2d(
    input_var,
    aperture_r,
    aperture_z,
    axis="both",
    bottom_bc="pressure_release",
    dist="hermitian_angle",
):
    # Load default params
    depth, _, _, z_rcv, _, _, f = default_params()

    # Load input vars
    delta_rcv = input_var["delta_rcv"]
    r_src = input_var["r_src"]
    z_src = input_var["z_src"]
    n_rcv = input_var["n_rcv"]
    df = input_var["df"]
    f = np.arange(f[0], f[-1], df)

    # Define the receivers position
    x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # Get distance properties
    dist_func, dist_kwargs, dist_properties = pick_distance_to_apply(dist)

    ### Range axis ###
    if axis == "r" or axis == "both":
        # Derive ref RTF
        r_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="r", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        total_dist_r = dist_func(g_ref, g, **dist_kwargs)
        range_displacement = r_src_list - r_src

        # Apply log if required
        if dist_properties["apply_log"]:
            total_dist_r += 1
            total_dist_r = 10 * np.log10(total_dist_r)

        # Derive aperure of the main lobe
        # th_r = 3  # threshold
        i1_r, i2_r, ap_r = total_dist_aperture(
            total_dist_r, range_displacement, dist_properties["th_main_lobe"]
        )
        aperture_r.append(ap_r)

    ### Depth axis ###
    if axis == "z" or axis == "both":
        # Derive ref RTF
        z_src_list, r_src_rcv_ref, r_src_rcv, _, f, g_ref = derive_Pi_ref(
            r_src, z_src, x_rcv, f, axis="z", bottom_bc=bottom_bc
        )

        # Derive RTF for the set of potential source positions (r_src_rcv)
        f, g = g_mat(
            f,
            z_src_list,
            z_rcv_ref=z_rcv,
            z_rcv=z_rcv,
            depth=depth,
            r_rcv_ref=r_src_rcv_ref,
            r=r_src_rcv,
            bottom_bc=bottom_bc,
        )

        # Derive the distance
        total_dist_z = dist_func(g_ref, g, **dist_kwargs)
        depth_displacement = z_src_list - z_src

        # Apply log if required
        if dist_properties["apply_log"]:
            total_dist_z += 1
            total_dist_z = 10 * np.log10(total_dist_z)

        # Derive aperure of the main lobe
        i1_z, i2_z, ap_z = total_dist_aperture(
            total_dist_z, range_displacement, dist_properties["th_main_lobe"]
        )
        aperture_z.append(ap_z)


def plot_sensibility(apertures_r, apertures_z, param_var, axis="both"):

    # Unpack param vars
    root_img = param_var["root_img"]
    values = param_var["values"]
    xlabel = param_var["xlabel"]

    if axis == "r" or axis == "both":
        plt.figure()
        plt.plot(values, apertures_r, marker=".", color="k")
        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 r_{"
            + f"{param_var['th_r']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" } }\, \textrm{[m]}$"
        )
        plt.grid()
        plt.savefig(os.path.join(root_img, f"{param_var['dist_name']}_aperture_r.png"))
        plt.close("all")

    if axis == "z" or axis == "both":
        plt.figure()
        plt.plot(values, apertures_z, marker=".", color="r")
        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 z_{"
            + f"{param_var['th_z']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" } }\, \textrm{[m]}$"
        )
        plt.grid()
        plt.savefig(os.path.join(root_img, f"{param_var['dist_name']}_aperture_z.png"))
        plt.close("all")

    if axis == "both":
        plt.figure()
        plt.plot(
            values,
            apertures_r,
            marker=".",
            color="k",
            label=r"$2 r_{"
            + f"{param_var['th_r']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" }}$",
        )
        plt.plot(
            values,
            apertures_z,
            marker=".",
            color="r",
            label=r"$2 z_{"
            + f"{param_var['th_r']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" }}$",
        )

        plt.xlabel(xlabel)
        plt.ylabel(
            r"$2 x_{"
            + f"{param_var['th_r']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" }} \, \textrm{[m]}$"
        )
        plt.grid()
        plt.legend()
        plt.savefig(
            os.path.join(root_img, f"{param_var['dist_name']}_aperture_r_z.png")
        )
        plt.close("all")


def plot_sensibility_2d(apertures_rr, apertures_zz, param_var, axis="both"):

    # Unpack param vars
    root_img = param_var["root_img"]
    values_x = param_var["values_x"]
    values_y = param_var["values_y"]
    xlabel = param_var["xlabel"]
    ylabel = param_var["ylabel"]

    pad = 0.2
    cmap = "jet_r"  # "binary"
    aspect = "auto"
    cbar_width = "3%"

    if axis == "r" or axis == "both":
        # Define vmin and vmax
        vmin = 0
        vmax = np.percentile(apertures_rr, 50)

        # Plot aperture map
        _, ax = plt.subplots(1, 1)
        pc = plt.pcolormesh(
            values_x,
            values_y,
            apertures_rr.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            r"$2 r_{"
            + f"{param_var['th_r']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" } }\, \textrm{[m]}$"
        )
        ax.set_aspect(aspect, adjustable="box")
        divider1 = make_axes_locatable(ax)
        cax = divider1.append_axes("right", size=cbar_width, pad=pad)
        plt.colorbar(pc, cax=cax, label=param_var["colorbar_title"])

        # Save
        fpath = os.path.join(
            root_img,
            f"{param_var['dist_name']}_aperture_rr.png",
        )
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
        plt.close("all")

    if axis == "z" or axis == "both":
        # Define vmin and vmax
        vmin = 0
        vmax = np.percentile(apertures_zz, 50)

        # Plot aperture map
        _, ax = plt.subplots(1, 1)
        pc = plt.pcolormesh(
            values_x,
            values_y,
            apertures_zz.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            r"$2 z_{"
            + f"{param_var['th_z']}"
            + r"\textrm{"
            + param_var["dist_unit"]
            + r" } }\, \textrm{[m]}$"
        )
        ax.set_aspect(aspect, adjustable="box")
        divider1 = make_axes_locatable(ax)
        cax = divider1.append_axes("right", size=cbar_width, pad=pad)
        plt.colorbar(pc, cax=cax, label=param_var["colorbar_title"])

        # Save
        fpath = os.path.join(
            root_img,
            f"{param_var['dist_name']}_aperture_zz.png",
        )
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
        plt.close("all")


def default_params():
    # Define receivers and source positions
    depth = 1000
    # Source
    z_src = 5
    r_src = 30 * 1e3
    # Receivers
    z_rcv = depth - 1
    n_rcv = 5
    delta_rcv = 10

    # Frequency range
    fmin = 1
    fmax = 50
    df = 0.1
    f = np.arange(fmin, fmax + df, df)
    # nb_freq = 500
    # f = np.linspace(fmin, fmax, nb_freq)

    return depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f


def pick_distance_to_apply(dist, axis="rz"):
    # Select dist function to apply
    if dist == "frobenius":
        dist_func = D_frobenius
        dist_kwargs = {}
        apply_log = True
        colobar_title = r"$10\textrm{log}_{10}(1+\mathcal{D}_F)\, \textrm{[dB]}$"
        th_main_lobe = 3  # 3dB main lobe
        label = r"$\mathcal{D}_F$"
        dist_unit = "dB"

        if axis == "rz":
            title = r"$\mathcal{D}_F(r, z)$"
            ylabel = None

        elif axis == "r":
            title = r"$\mathcal{D}_F(r, z=z_s)$"
            ylabel = r"$10\textrm{log}_{10}(1+\mathcal{D}_F)\, \textrm{[dB]}$"

        elif axis == "z":
            title = r"$\mathcal{D}_F(r=r_s, z)$"
            ylabel = r"$10\textrm{log}_{10}(1+\mathcal{D}_F)\, \textrm{[dB]}$"

    elif dist == "hermitian_angle":
        dist_func = D_hermitian_angle_fast
        dist_kwargs = {
            "unit": "deg",
            "apply_mean": True,
        }
        apply_log = False
        colobar_title = r"$\theta \, \textrm{[°]}$"
        th_main_lobe = 0.5  # 1° main lobe
        label = r"$\theta$"
        dist_unit = "°"

        if axis == "rz":
            title = r"$\theta(r, z)$"
            ylabel = None

        elif axis == "r":
            title = r"$\theta(r, z=z_s)$"
            ylabel = r"$\theta \, \textrm{[°]}$"

        elif axis == "z":
            title = r"$\theta(r=r_s, z)$"
            ylabel = r"$\theta \, \textrm{[°]}$"

    dist_properties = {
        "title": title,
        "label": label,
        "ylabel": ylabel,
        "dist_unit": dist_unit,
        "apply_log": apply_log,
        "th_main_lobe": th_main_lobe,
        "colorbar_title": colobar_title,
    }

    return (dist_func, dist_kwargs, dist_properties)


if __name__ == "__main__":

    # covered_range = 5
    # dr = 0.1
    # zmin = 3
    # zmax = 7
    # dz = 0.1
    # dist = "D2"

    # # Load default parameters
    # depth, r_src, z_src, z_rcv, n_rcv, delta_rcv, f = default_params()

    # # Define the receivers position
    # x_rcv = np.array([i * delta_rcv for i in range(n_rcv)])

    # # # Variations along the range axis
    # PI_var_r(
    #     depth,
    #     f,
    #     r_src,
    #     z_src,
    #     x_rcv,
    #     z_rcv,
    #     covered_range=covered_range,
    #     dr=dr,
    #     dist=dist,
    #     smooth_tf=True,
    # )

    bottom_bc = "pressure_release"
    dists = ["hermitian_angle", "frobenius"]

    # covered_range = 15 * 1e3
    # dr = 100
    # zmin = 1
    # zmax = 999
    # dz = 10
    # for dist in dists:
    #     full_test(covered_range, dr, zmin, zmax, dz, dist=dist, bottom_bc=bottom_bc)

    # covered_range = 5
    # dz = 0.1
    # dr = 0.1
    # zmin = 1
    # zmax = 11

    params = ["n_rcv", "delta_rcv"]
    # params = ["r_src"]
    # dist = "hermitian_angle"
    # for param in params:
    #     for dist in dists:
    #         sensibility_ideal_waveguide(
    #             param=param, axis="both", bottom_bc=bottom_bc, dist=dist
    #         )

    for dist in dists:
        sensibility_ideal_waveguide_2d(
            param_couple=["n_rcv", "delta_rcv"], bottom_bc=bottom_bc, dist=dist
        )

    # covered_range = 10
    # dz = 0.1
    # dr = 0.1
    # zmin = 1
    # zmax = 21
    # for dist in dists:
    #     full_test(covered_range, dr, zmin, zmax, dz, dist=dist, bottom_bc=bottom_bc)
