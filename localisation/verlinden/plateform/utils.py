#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/05/06 15:20:12
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
import pandas as pd
from path import PROJECT_ROOT

if os.name == "nt":
    ROOT_DATASET_PATH = os.path.join(PROJECT_ROOT, "localisation", "verlinden", "localisation_dataset")
else:   
    ROOT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "localisation_dataset")


def set_attrs(xr_dataset, grid_info, testcase):
    """
    Add attributes to the dataset.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    grid_info : dict
        Grid information.
    testcase : Testcase object.
        Tescase.

    Returns
    -------
    xr.Dataset
    """
    # Set attributes
    var_unit_mapping = {
        "°": [
            "lon_rcv",
            "lat_rcv",
            "lon",
            "lat",
        ],
        "m": ["r_from_rcv"],
        "": ["idx_rcv"],
        "s": ["delay_rcv"],
    }
    for unit in var_unit_mapping.keys():
        for var in var_unit_mapping[unit]:
            xr_dataset[var].attrs["units"] = unit

    xr_dataset["lon_rcv"].attrs["long_name"] = "Receiver longitude"
    xr_dataset["lat_rcv"].attrs["long_name"] = "Receiver latitude"
    xr_dataset["r_from_rcv"].attrs["long_name"] = "Range from receiver"
    xr_dataset["lon"].attrs["long_name"] = "Longitude"
    xr_dataset["lat"].attrs["long_name"] = "Latitude"
    xr_dataset["idx_rcv"].attrs["long_name"] = "Receiver index"
    xr_dataset["delay_rcv"].attrs["long_name"] = "Propagation delay from receiver"

    # Initialisation time
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    xr_dataset.attrs["init_time"] = now

    xr_dataset.attrs["dataset_root_dir"] = build_root_dir(testcase.name)
    set_propa_path(xr_dataset, grid_info, testcase)
    set_propa_grid_path(xr_dataset)

    # Boolean to control if the dataset has been populated
    xr_dataset.attrs["propa_done"] = False
    xr_dataset.attrs["propa_grid_done"] = False
    xr_dataset.attrs["propa_grid_src_done"] = False

    return xr_dataset


def build_root_dir(testcase_name):
    root_dir = os.path.join(ROOT_DATASET_PATH, testcase_name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    return root_dir


def build_root_propa(testcase_name):
    root_dir = build_root_dir(testcase_name)
    root_propa = os.path.join(root_dir, "propa")
    if not os.path.exists(root_propa):
        os.makedirs(root_propa)
    return root_propa


def build_boundaries_label(grid_info):
    boundaries_label = "_".join(
        [
            f"{v:.4f}"
            for v in [
                grid_info["min_lon"],
                grid_info["max_lon"],
                grid_info["min_lat"],
                grid_info["max_lat"],
            ]
        ]
    )
    return boundaries_label


def build_propa_path(testcase_name, boundaries_label):
    root_propa = build_root_propa(testcase_name)
    propa_path = os.path.join(
        root_propa,
        f"propa_{boundaries_label}.zarr",
    )
    return propa_path


def set_propa_path(xr_dataset, grid_info, testcase):
    """
    Build dataset propa path and add it as attributes to the dataset.
    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    grid_info : dict
        Grid information.
    testcase : Testcase object.
        Tescase.
    """

    # Propa dataset: folder containing dataset with transfer functions
    xr_dataset.attrs["boundaries_label"] = build_boundaries_label(grid_info)
    xr_dataset.attrs["fullpath_dataset_propa"] = build_propa_path(
        testcase.name, xr_dataset.boundaries_label
    )


def build_root_propa_grid(root_dir):
    root_propa_grid = os.path.join(root_dir, "propa_grid")
    if not os.path.exists(root_propa_grid):
        os.makedirs(root_propa_grid)
    return root_propa_grid


def build_propa_grid_path(root_dir, boundaries_label, grid_label):
    root_propa_grid = build_root_propa_grid(root_dir)
    propa_grid_path = os.path.join(
        root_propa_grid,
        f"propa_grid_{boundaries_label}_{grid_label}.zarr",
    )
    return propa_grid_path


def set_propa_grid_path(xr_dataset):
    """
    Build dataset propa grid path and add it as attribute to the dataset.
    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    """

    # Propa_grid dataset: folder containing dataset with gridded transfer functions
    xr_dataset.attrs["grid_label"] = build_grid_label(xr_dataset.dx, xr_dataset.dy)
    xr_dataset.attrs["fullpath_dataset_propa_grid"] = build_propa_grid_path(
        xr_dataset.dataset_root_dir, xr_dataset.boundaries_label, xr_dataset.grid_label
    )


def set_propa_grid_src_path(xr_dataset):
    """
    Build dataset propa grid path and add it as attributes to the dataset.
    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    grid_info : dict
        Grid information.
    testcase : Testcase object.
        Tescase.
    """

    # Propa_grid dataset: folder containing dataset with gridded transfer functions
    root_propa_grid = os.path.join(xr_dataset.dataset_root_dir, "propa_grid_src")
    if not os.path.exists(root_propa_grid):
        os.makedirs(root_propa_grid)

    propa_grid_src_path = os.path.join(
        root_propa_grid,
        f"propa_grid_src_{xr_dataset.boundaries_label}_{xr_dataset.grid_label}_{xr_dataset.src_label}.zarr",
    )
    xr_dataset.attrs["fullpath_dataset_propa_grid_src"] = propa_grid_src_path


def build_src_label(src_name, f0=None, fs=None, std_fi=None, tau_corr_fi=None):
    # if src_name == "ship":
    #     try:
    #         src_label = f"{src_name}_{f0:.1f}_{fs:.1f}_{std_fi:.1f}_{tau_corr_fi:.1f}"
    #     except:
    #         print("Missing parameters for ship source")
    # else:
    src_label = f"{src_name}"

    return src_label


def build_grid_label(dx, dy):
    return f"{dx}_{dy}"


def get_region_number(nregion_max, var, max_size_bytes=0.1 * 1e9):
    nregion = nregion_max
    # max_size_bytes = 0.1 * 1e9  # 100 Mo
    size = var.nbytes / nregion
    while size <= max_size_bytes and nregion > 1:  # At least 1 region
        nregion -= 1
        size = var.nbytes / nregion
    return nregion


def get_lonlat_sub_regions(ds, nregion):
    lat_slices_lim = np.linspace(0, ds.sizes["lat"], nregion + 1, dtype=int)
    lat_slices = [
        slice(lat_slices_lim[i], lat_slices_lim[i + 1])
        for i in range(len(lat_slices_lim) - 1)
    ]
    lon_slices_lim = np.linspace(0, ds.sizes["lon"], nregion + 1, dtype=int)
    lon_slices = [
        slice(lon_slices_lim[i], lon_slices_lim[i + 1])
        for i in range(len(lon_slices_lim) - 1)
    ]

    return lon_slices, lat_slices
