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
import xarray as xr
import dask.array as da

from cst import C0
from misc import mult_along_axis
from localisation.verlinden.plateform.params import N_WORKERS
from localisation.verlinden.params import ROOT_DATASET_PATH
from localisation.verlinden.verlinden_utils import (
    get_range_src_rcv_range,
    add_noise_to_signal,
)


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


def compute_received_signal(
    ds, propagating_freq, propagating_spectrum, norm_factor, nfft_inv, apply_delay
):

    # Received signal spectrum resulting from the convolution of the src signal and the impulse response
    transmited_field_f = mult_along_axis(
        ds.tf_gridded, propagating_spectrum * norm_factor, axis=-1
    )

    # Apply corresponding delay to the signal
    if apply_delay:
        tau = ds.delay_rcv.min(
            dim="idx_rcv"
        )  # Delay to apply to the signal to take into account the propagation time

        # Expand tau to the signal shape
        tau = tau.expand_dims({"kraken_freq": ds.sizes["kraken_freq"]}, axis=-1)
        # Derive delay factor
        tau_vec = mult_along_axis(tau, propagating_freq, axis=-1)
        delay_f = np.exp(1j * 2 * np.pi * tau_vec)
        # Expand delay factor to the signal shape
        delay_f = tau.copy(deep=True, data=delay_f).expand_dims(
            {"idx_rcv": ds.sizes["idx_rcv"]}, axis=0
        )
        # Apply delay
        transmited_field_f = transmited_field_f * delay_f

    # Fourier synthesis of the received signal -> time domain
    chunk_shape = (
        ds.sizes["idx_rcv"],
        ds.sizes["lat"] // N_WORKERS,
        ds.sizes["lon"] // N_WORKERS,
        ds.sizes["kraken_freq"],
    )
    transmited_field_f = da.from_array(transmited_field_f, chunks=chunk_shape)
    transmited_field_t = np.fft.irfft(transmited_field_f, axis=-1, n=nfft_inv).compute()

    return transmited_field_t


def init_simu_info_dataset():
    n_simu = 100
    n_rcv_max = 10
    n_freq_max = 513

    simu_id = np.arange(0, n_simu, dtype=int)
    simu_unique_id = np.zeros(n_simu, dtype="<U128")

    simu_launched = np.zeros(n_simu, dtype=bool)
    simu_propa_done = np.zeros(n_simu, dtype=bool)
    simu_propa_grid_done = np.zeros(n_simu, dtype=bool)
    simu_propa_grid_src_done = np.zeros(n_simu, dtype=bool)

    simu_min_dist = np.zeros(n_simu, dtype=float)
    simu_nfreq = np.zeros(n_simu, dtype=int)

    simu_freq = np.zeros((n_simu, n_freq_max), dtype=float)
    simu_freq[:] = np.nan
    freq_idx = np.arange(0, n_freq_max, dtype=int)

    boundaries_label = np.zeros(n_simu, dtype="<U64")
    simu_rcv_id = np.zeros((n_simu, n_rcv_max), dtype="<U16")

    rcv_idx = np.arange(0, n_rcv_max, dtype=int)
    nrcv = np.zeros(n_simu, dtype=int)

    ds_info = xr.Dataset(
        data_vars=dict(
            unique_id=("simu_id", simu_unique_id),
            launched=("simu_id", simu_launched),
            min_dist=("simu_id", simu_min_dist),
            nfreq=("simu_id", simu_nfreq),
            freq=(("simu_id", "freq_idx"), simu_freq),
            boundaries_label=("simu_id", boundaries_label),
            rcv_id=(("simu_id", "rcv_idx"), simu_rcv_id),
            nrcv=("simu_id", nrcv),
            propa_done=("simu_id", simu_propa_done),
            propa_grid_done=("simu_id", simu_propa_grid_done),
            propa_grid_src_done=("simu_id", simu_propa_grid_src_done),
        ),
        coords=dict(
            simu_id=("simu_id", simu_id),
            freq_idx=("freq_idx", freq_idx),
            rcv_idx=("rcv_idx", rcv_idx),
        ),
    )

    return ds_info


def set_simu_unique_id(ds, id_to_write_in):
    nf = ds.nfreq.sel(simu_id=id_to_write_in).values
    boundaries_label = ds.boundaries_label.sel(simu_id=id_to_write_in).values
    freq = ds.freq.sel(simu_id=id_to_write_in).values
    rcv_id = "_".join(
        ds.rcv_id.sel(simu_id=id_to_write_in).values[
            : ds.nrcv.sel(simu_id=id_to_write_in).values
        ]
    )
    ds.unique_id.loc[id_to_write_in] = (
        f"{boundaries_label}_{nf}_{np.nanmin(freq):.2f}_{np.nanmax(freq):.2f}_{rcv_id}"
    )


def get_ds_info_path(ds):
    folder = os.path.dirname(os.path.dirname(ds.fullpath_dataset_propa))
    ds_info_path = os.path.join(folder, "simu_index.nc")
    return ds_info_path


def update_info_status(ds, part_done):
    path = get_ds_info_path(ds)
    with xr.open_dataset(path) as ds_info:
        id_to_write_in = (~ds_info.launched).idxmax() - 1
        var_to_update = f"{part_done}_done"
        ds_info[var_to_update].loc[id_to_write_in] = True
    ds_info.to_netcdf(path)


def init_event_dataset(xr_dataset, src_info, rcv_info):
    """
    Initialize the dataset for the event.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Library dataset.
    src_info : dict
        Source information.
    rcv_info : dict
        Receiver information.
    interp_src_pos_on_grid : bool, optional
        Interpolate source positions on grid. The default is False.

    Returns
    -------
    xr.Dataset
        Initialized dataset.

    """

    lon_src, lat_src, t_src = src_info["lons"], src_info["lats"], src_info["time"]
    r_src_rcv = get_range_src_rcv_range(lon_src, lat_src, rcv_info)
    delay_src_rcv = r_src_rcv / C0

    xr_dataset.coords["event_signal_time"] = []
    xr_dataset.coords["src_trajectory_time"] = t_src.astype(np.float32)

    xr_dataset["lon_src"] = (["src_trajectory_time"], lon_src.astype(np.float32))
    xr_dataset["lat_src"] = (["src_trajectory_time"], lat_src.astype(np.float32))
    xr_dataset["r_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(r_src_rcv).astype(np.float32),
    )
    xr_dataset["delay_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(delay_src_rcv).astype(np.float32),
    )

    xr_dataset["event_signal_time"].attrs["units"] = "s"
    xr_dataset["src_trajectory_time"].attrs["units"] = "s"
    xr_dataset["lon_src"].attrs["long_name"] = "lon_src"
    xr_dataset["lat_src"].attrs["long_name"] = "lat_src"
    xr_dataset["r_src_rcv"].attrs["long_name"] = "Range from receiver to source"
    xr_dataset["event_signal_time"].attrs["units"] = "Time"
    xr_dataset["src_trajectory_time"].attrs["long_name"] = "Time"


def add_noise_to_dataset(xr_dataset, target_var, snr_dB):
    """
    Add noise to signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    for i_rcv in xr_dataset.idx_rcv.values:
        rcv_sig = xr_dataset[target_var].sel(idx_rcv=i_rcv).values

        if snr_dB is not None:
            # Add noise to received signal
            xr_dataset[target_var].loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
                np.copy(rcv_sig),
                snr_dB=snr_dB,
            )


def add_noise_to_library(xr_dataset, target_var, snr_dB):
    """
    Add noise to library signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    target_var = "rcv_signal_library"
    add_noise_to_dataset(xr_dataset, target_var, snr_dB)


def add_noise_to_library(xr_dataset, target_var, snr_dB):
    """
    Add noise to event signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    target_var = "event_signal"
    add_noise_to_dataset(xr_dataset, target_var, snr_dB)
