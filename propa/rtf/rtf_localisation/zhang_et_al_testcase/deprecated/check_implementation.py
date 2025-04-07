#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   check_implementation.py
@Time    :   2025/02/10 18:53:38
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

from propa.rtf.rtf_utils import D_hermitian_angle_fast
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import params

ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data"
root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023\test_implementation"

# ======================================================================================================================
# Functions
# ======================================================================================================================


def check_grid_influence():

    path_test = os.path.join(ROOT_DATA, "save_10022025_test")

    # Check transfert functions
    fname = "tf_zhang_grid_dx20m_dy20m.nc"
    print(f"### Check transfert functions ({fname}) ###")
    ds_tf_test = xr.open_dataset(os.path.join(path_test, fname))
    ds_tf_zhang = xr.open_dataset(os.path.join(ROOT_DATA, fname))

    # Check bounds
    print(
        f"Test dataset with smaller bounds: xmin = {ds_tf_test.x.min().values:.2f},  xmax={ds_tf_test.x.max().values:.2f}, ymin = {ds_tf_test.y.min().values:.2f}, ymax = {ds_tf_test.y.max().values:.2f}"
    )
    print(
        f"Zhang dataset with original bounds: xmin = {ds_tf_zhang.x.min().values:.2f},  xmax={ds_tf_zhang.x.max().values:.2f}, ymin = {ds_tf_zhang.y.min().values:.2f}, ymax = {ds_tf_zhang.y.max().values:.2f}"
    )

    # Check area size length
    lx_test = ds_tf_test.x.max().values - ds_tf_test.x.min().values
    ly_test = ds_tf_test.y.max().values - ds_tf_test.y.min().values
    print(f"Test dataset  lx = {lx_test:.2f}m, ly = {ly_test:.2f}m")
    lx_zhang = ds_tf_zhang.x.max().values - ds_tf_zhang.x.min().values
    ly_zhang = ds_tf_zhang.y.max().values - ds_tf_zhang.y.min().values
    print(f"Zhang dataset  lx = {lx_zhang:.2f}m, ly = {ly_zhang:.2f}m")

    tf_test = ds_tf_test.tf_real + 1j * ds_tf_test.tf_imag
    tf_abs = np.abs(tf_test).isel(idx_rcv=0)
    tf_zhang = ds_tf_zhang.tf_real + 1j * ds_tf_zhang.tf_imag
    tf_abs_zhang = np.abs(tf_zhang).isel(idx_rcv=0)

    test_area_corners_x = [
        tf_abs.x.min().values,
        tf_abs.x.min().values,
        tf_abs.x.max().values,
        tf_abs.x.max().values,
        tf_abs.x.min().values,
    ]
    test_area_corners_y = [
        tf_abs.y.min().values,
        tf_abs.y.max().values,
        tf_abs.y.max().values,
        tf_abs.y.min().values,
        tf_abs.y.min().values,
    ]

    f, axs = plt.subplots(2, 2, figsize=(18, 9))
    tf_abs.sel(f=200).plot(x="x", ax=axs[0, 0], vmin=0, vmax=1e-3, cmap="jet")
    axs[0, 0].set_title("Test dataset")
    tf_abs_zhang.sel(f=200).plot(x="x", ax=axs[0, 1], vmin=0, vmax=1e-3, cmap="jet")
    axs[0, 1].set_title("Zhang dataset")

    tf_abs.sel(f=400).plot(x="x", ax=axs[1, 0], vmin=0, vmax=1e-3, cmap="jet")
    tf_abs_zhang.sel(f=400).plot(x="x", ax=axs[1, 1], vmin=0, vmax=1e-3, cmap="jet")

    # print("axs[0,1] limits:", axs[0, 1].get_xlim(), axs[0, 1].get_ylim())
    # print("axs[1,1] limits:", axs[1, 1].get_xlim(), axs[1, 1].get_ylim())
    # print("Test area x range:", min(test_area_corners_x), max(test_area_corners_x))
    # print("Test area y range:", min(test_area_corners_y), max(test_area_corners_y))

    axs[0, 1].plot(test_area_corners_x, test_area_corners_y, "k", linewidth=2, zorder=5)
    axs[1, 1].plot(test_area_corners_x, test_area_corners_y, "k", linewidth=2, zorder=5)

    plt.savefig(
        os.path.join(root_img, "tf_mod_map_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # Compare transfer functions at given positions
    x_0 = ds_tf_test.x.values[10]
    y_0 = ds_tf_test.y.values[10]
    x_1 = ds_tf_test.x.values[20]
    y_1 = ds_tf_test.y.values[20]

    f, axs = plt.subplots(1, 2, figsize=(18, 9))
    tf_abs.sel(x=x_0, y=y_0).plot(ax=axs[0], label="Test dataset")
    tf_abs_zhang.sel(x=x_0, y=y_0).plot(ax=axs[0], label="Zhang dataset")
    axs[0].set_title(f"Transfer function at x={x_0:.2f}m, y={y_0:.2f}m")
    axs[0].legend()

    tf_abs.sel(x=x_1, y=y_1).plot(ax=axs[1], label="Test dataset")
    tf_abs_zhang.sel(x=x_1, y=y_1).plot(ax=axs[1], label="Zhang dataset")
    axs[1].set_title(f"Transfer function at x={x_1:.2f}m, y={y_1:.2f}m")
    axs[1].legend()

    plt.savefig(
        os.path.join(root_img, "tf_mod_pos_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # Select common coordinates
    idx_x = [i for i, x in enumerate(ds_tf_zhang.x.values) if x in ds_tf_test.x.values]
    idx_y = [i for i, y in enumerate(ds_tf_zhang.y.values) if y in ds_tf_test.y.values]
    ds_tf_zhang = ds_tf_zhang.isel(x=idx_x, y=idx_y)

    # Plot diff at f=200Hz and f=400Hz
    diff = tf_abs - tf_abs_zhang
    f, axs = plt.subplots(1, 2, figsize=(18, 9))
    diff.sel(f=200).plot(ax=axs[0], cmap="bwr")
    axs[0].set_title("Difference at f=200Hz")
    diff.sel(f=400).plot(ax=axs[1], cmap="bwr")
    axs[1].set_title("Difference at f=400Hz")

    plt.savefig(os.path.join(root_img, "tf_mod_diff.png"), dpi=300, bbox_inches="tight")
    # plt.show()

    # Print coordinates to check if they are the same
    print(f"Common x coords : {np.all(ds_tf_test.x.values == ds_tf_zhang.x.values)}")
    print(f"Common y coords : {np.all(ds_tf_test.y.values == ds_tf_zhang.y.values)}")

    # Check if the two datasets are equal along common coordinates
    diff = ds_tf_test - ds_tf_zhang
    diff = diff.isel(idx_rcv=0)
    diff_real = np.abs(diff.tf_real).mean(dim="f")
    diff_imag = np.abs(diff.tf_imag).mean(dim="f")
    plt.figure()
    diff_real.plot()
    plt.title(
        "Absolute difference between test and Zhang transfer functions (real part)"
    )
    plt.savefig(
        os.path.join(root_img, "tf_diff_real.png"), dpi=300, bbox_inches="tight"
    )

    plt.figure()
    diff_imag.plot()
    plt.title(
        "Absolute difference between test and Zhang transfer functions (imaginary part)"
    )
    plt.savefig(
        os.path.join(root_img, "tf_diff_imag.png"), dpi=300, bbox_inches="tight"
    )

    # plt.show()

    print(
        f"Common tf_real : {np.all(ds_tf_test.tf_real.values == ds_tf_zhang.tf_real.values)}"
    )
    print(
        f"Common tf_imag : {np.all(ds_tf_test.tf_imag.values == ds_tf_zhang.tf_imag.values)}"
    )

    # xr.testing.assert_allclose(ds_tf_test.tf_rel, ds_tf_zhang)

    # # Plot the difference between the two datasets
    # diff = ds_tf_test - ds_tf_zhang

    # plt.figure()
    # diff.tf.abs().plot()
    # plt.title("Absolute difference between test and Zhang transfer functions")
    # # plt.show()
    # plt.savefig(os.path.join(root_img, "tf_diff.png"), dpi=300, bbox_inches="tight")

    # # Check library
    # fname = "zhang_library_dx20m_dy20m.nc"
    # print(f"### Check library ({fname}) ###")
    # ds_library_test = xr.open_dataset(os.path.join(path_test, fname))
    # ds_library_zhang = xr.open_dataset(os.path.join(ROOT_DATA, fname))

    # # Check bounds
    # print(
    #     f"Test dataset with smaller bounds: xmin = {ds_library_test.x.min().values:.2f},  xmax={ds_library_test.x.max().values:.2f}, ymin = {ds_library_test.y.min().values:.2f}, ymax = {ds_library_test.y.max().values:.2f}"
    # )
    # print(
    #     f"Zhang dataset with original bounds: xmin = {ds_library_zhang.x.min().values:.2f},  xmax={ds_library_zhang.x.max().values:.2f}, ymin = {ds_library_zhang.y.min().values:.2f}, ymax = {ds_library_zhang.y.max().values:.2f}"
    # )

    # # Check area size length
    # lx_test = ds_library_test.x.max().values - ds_library_test.x.min().values
    # ly_test = ds_library_test.y.max().values - ds_library_test.y.min().values
    # print(f"Test dataset  lx = {lx_test:.2f}m, ly = {ly_test:.2f}m")
    # lx_zhang = ds_library_zhang.x.max().values - ds_library_zhang.x.min().values
    # ly_zhang = ds_library_zhang.y.max().values - ds_library_zhang.y.min().values
    # print(f"Zhang dataset  lx = {lx_zhang:.2f}m, ly = {ly_zhang:.2f}m")

    # # Select common coordinates
    # ds_library_test = ds_library_test.sel(x=ds_library_zhang.x, method="nearest").sel(
    #     y=ds_library_zhang.y, method="nearest"
    # )

    # # Check coordinates
    # print(
    #     f"Common x coords : {np.all(ds_library_test.x.values == ds_library_zhang.x.values)}"
    # )
    # print(
    #     f"Common y coords : {np.all(ds_library_test.y.values == ds_library_zhang.y.values)}"
    # )

    # # Assert that the two datasets are equal along common coordinates
    # xr.testing.assert_allclose(ds_library_test, ds_library_zhang)

    # # Plot the difference between the two datasets
    # diff = ds_library_test - ds_library_zhang

    # plt.figure()
    # diff.rtf.abs().plot()
    # plt.title("Absolute difference between test and Zhang library")
    # # plt.show()
    # plt.savefig(
    #     os.path.join(root_img, "library_diff.png"), dpi=300, bbox_inches="tight"
    # )

    plt.close("all")


def check_grid_construction():
    # Usual grid
    depth, receivers, source, usual_grid, frequency, bott_hs_properties = params(
        debug=False
    )

    # Test grid
    depth, receivers, source, test_grid, frequency, bott_hs_properties = params(
        debug=True
    )

    # Select common x, y coordinates
    mask_x = np.isin(usual_grid["x"][0, :], test_grid["x"][0, :])
    mask_y = np.isin(usual_grid["y"][:, 0], test_grid["y"][:, 0])

    # Range for to each rcv
    for idx_rcv in range(len(receivers["x"])):
        r_test = test_grid["r"][idx_rcv, :, :]
        r_usual = usual_grid["r"][idx_rcv, ...][:, mask_x][mask_y]
        diff_r = r_test - r_usual

        no_diff = np.all(diff_r == 0)
        print(f"No range difference for receiver {idx_rcv} : {no_diff}")
        if not no_diff:
            print(f"Max range difference for receiver {idx_rcv} : {np.max(diff_r)}")
            plt.figure()
            plt.imshow(diff_r, cmap="bwr")
            plt.colorbar()
            plt.title(f"Range difference for receiver {idx_rcv}")
            plt.savefig(os.path.join(root_img, f"range_diff_rcv{idx_rcv}.png"), dpi=300)

    # Check the way gridded tf is computed
    # Load dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    ds = xr.open_dataset(fpath)

    ### Check step 1 ####
    # Usual
    ds_usual = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            x=usual_grid["x"][0, :],
            y=usual_grid["y"][:, 0],
            idx_rcv=range(len(receivers["x"])),
        ),
        attrs=dict(
            df=ds.f.diff("f").values[0],
            dx=usual_grid["dx"],
            dy=usual_grid["dy"],
            testcase="usual",
        ),
    )

    r_grid_all_rcv = np.array(
        [usual_grid["r"][i_rcv].flatten() for i_rcv in range(len(receivers["x"]))]
    )
    r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

    tf_vect = ds.tf_real.sel(
        r=r_grid_all_rcv_unique, method="nearest"
    ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")

    tf_usual = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            r=r_grid_all_rcv_unique,
        ),
        data_vars=dict(tf=(["f", "r"], tf_vect.values)),
    )

    # Test
    ds_test = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            x=test_grid["x"][0, :],
            y=test_grid["y"][:, 0],
            idx_rcv=range(len(receivers["x"])),
        ),
        attrs=dict(
            df=ds.f.diff("f").values[0],
            dx=test_grid["dx"],
            dy=test_grid["dy"],
            testcase="test",
        ),
    )

    r_grid_all_rcv = np.array(
        [test_grid["r"][i_rcv].flatten() for i_rcv in range(len(receivers["x"]))]
    )
    r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

    tf_vect = ds.tf_real.sel(
        r=r_grid_all_rcv_unique, method="nearest"
    ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")

    tf_test = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            r=r_grid_all_rcv_unique,
        ),
        data_vars=dict(tf=(["f", "r"], tf_vect.values)),
    )

    # Check if the two datasets are equal along common coordinates
    diff = tf_usual - tf_test
    diff_abs = np.abs(diff.tf).max(dim="f")
    plt.figure()
    diff_abs.plot()
    plt.title("Absolute difference between usual and test transfer functions")
    plt.savefig(os.path.join(root_img, "tf_diff_usual_test.png"), dpi=300)

    ### Step 2 : check gridding process ###

    # Slower solution

    # Quicker solution
    # Usual
    gridded_tf = []
    grid_shape = (ds_usual.sizes["f"],) + usual_grid["r"].shape[1:]
    for i_rcv in range(len(receivers["x"])):
        r_grid = usual_grid["r"][i_rcv].flatten()
        tf_ircv = tf_usual.tf.sel(r=r_grid, method="nearest")

        tf_grid = tf_ircv.values.reshape(grid_shape)
        gridded_tf.append(tf_grid)

    gridded_tf = np.array(gridded_tf)
    # Add to dataset
    grid_coords = [
        "idx_rcv",
        "f",
        "y",
        "x",
    ]
    ds_usual["tf_real"] = (grid_coords, np.real(gridded_tf))
    ds_usual["tf_imag"] = (grid_coords, np.imag(gridded_tf))

    # Test
    gridded_tf = []
    grid_shape = (ds_test.sizes["f"],) + test_grid["r"].shape[1:]
    for i_rcv in range(len(receivers["x"])):
        r_grid = test_grid["r"][i_rcv].flatten()
        tf_ircv = tf_test.tf.sel(r=r_grid, method="nearest")

        tf_grid = tf_ircv.values.reshape(grid_shape)
        gridded_tf.append(tf_grid)

    gridded_tf = np.array(gridded_tf)
    # Add to dataset
    ds_test["tf_real"] = (grid_coords, np.real(gridded_tf))
    ds_test["tf_imag"] = (grid_coords, np.imag(gridded_tf))

    # Check if the two datasets are equal along common coordinates
    tf_grid_usual = ds_usual.tf_real + 1j * ds_usual.tf_imag
    tf_grid_test = ds_test.tf_real + 1j * ds_test.tf_imag
    diff = tf_grid_usual - tf_grid_test

    # Assert diff == 0 for all receivers
    diff_abs = np.abs(diff).max(dim="f")

    plt.figure()
    diff_abs.plot()
    plt.title("Absolute difference between usual and test gridded transfer functions")
    plt.savefig(os.path.join(root_img, "tf_grid_diff_usual_test.png"), dpi=300)

    print(f"Gridded TF are different : {np.any(diff_abs.values)}")

    plt.close("all")


if __name__ == "__main__":
    # check_grid_influence()
    # check_grid_construction()
    pass
