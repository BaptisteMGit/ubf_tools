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
import scipy.fft as sp_fft
import scipy.signal as signal

from cst import C0
from misc import mult_along_axis, fft_convolve_f
from localisation.verlinden.plateform.params import N_WORKERS
from localisation.verlinden.misc.params import ROOT_DATASET, ROOT_PROCESS
from localisation.verlinden.misc.verlinden_utils import (
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
    root_dir = os.path.join(ROOT_DATASET, testcase_name)
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


def build_process_output_path(xr_dataset, src_info, grid_info):
    """
    Build the output path for the process.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset.
    src_info : dict
        Source information.
    grid_info : dict
        Grid information.
    """

    path_components = os.path.normpath(
        xr_dataset.fullpath_dataset_propa_grid_src
    ).split(os.path.sep)
    testcase_name = path_components[-3]
    src_label = src_info["sig"]["sig_type"]
    boundaries_label = build_boundaries_label(grid_info)

    output_folder = f"{boundaries_label}_{src_label}"
    output_dir = os.path.join(ROOT_PROCESS, testcase_name, output_folder)
    output_fname = f"{xr_dataset.attrs['init_time']}.zarr"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xr_dataset.attrs["output_path"] = os.path.join(output_dir, output_fname)


def get_region_number(nregion_max, var, max_size_bytes=0.1 * 1e9):

    max_bytes_allowed = 2.1 * 1e9
    max_size_bytes = min(max_size_bytes, max_bytes_allowed)

    nregion = nregion_max
    # max_size_bytes = 0.1 * 1e9  # 100 Mo
    size = var.nbytes / nregion
    while size < max_size_bytes and nregion > 1:  # At least 1 region
        nregion -= 1
        size = var.nbytes / nregion
    return nregion


def get_lonlat_sub_regions(ds, nregion_lon, nregion_lat):
    step = ds.sizes["lat"] // nregion_lat
    lat_slices_lim = np.arange(0, ds.sizes["lat"], step=step)
    lat_slices_lim = np.append(lat_slices_lim, ds.sizes["lat"])

    # lat_slices_lim = np.linspace(0, ds.sizes["lat"], nregion + 1, dtype=int)
    lat_slices = [
        slice(lat_slices_lim[i], lat_slices_lim[i + 1])
        for i in range(len(lat_slices_lim) - 1)
    ]

    step = ds.sizes["lon"] // nregion_lon
    lon_slices_lim = np.arange(0, ds.sizes["lon"], step=step)
    lon_slices_lim = np.append(lon_slices_lim, ds.sizes["lon"])

    # lon_slices_lim = np.linspace(0, ds.sizes["lon"], nregion + 1, dtype=int)
    lon_slices = [
        slice(lon_slices_lim[i], lon_slices_lim[i + 1])
        for i in range(len(lon_slices_lim) - 1)
    ]

    return lon_slices, lat_slices


def compute_received_signal(
    ds, propagating_freq, propagating_spectrum, norm_factor, nfft_inv, apply_delay
):

    # Received signal spectrum resulting from the convolution of the src signal and the impulse response
    transmitted_field_f = mult_along_axis(
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
        transmitted_field_f = transmitted_field_f * delay_f

    # Fourier synthesis of the received signal -> time domain
    chunk_shape = ds.rcv_signal_library.data.chunksize
    transmitted_field_f = da.from_array(transmitted_field_f, chunks=chunk_shape)
    transmitted_field_t = np.fft.irfft(
        transmitted_field_f, axis=-1, n=nfft_inv
    ).compute()

    return transmitted_field_t


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

    return xr_dataset


def init_ambiguity_surface(xr_dataset):

    ambiguity_surface_dim = [
        "idx_rcv_pairs",
        "idx_similarity_metric",
        "src_trajectory_time",
        "snr",
        "lat",
        "lon",
    ]
    chunksize = dict(xr_dataset.rcv_signal_library.chunksizes)
    chunk_shape = (
        xr_dataset.sizes["idx_rcv_pairs"],
        1,
        xr_dataset.sizes["src_trajectory_time"],
        1,
        chunksize["lat"],
        chunksize["lon"],
    )

    ambiguity_surface = da.empty(
        tuple(xr_dataset.sizes[d] for d in ambiguity_surface_dim),
        dtype=np.float32,
        chunks=chunk_shape,
    )

    # Create ambiguity surface dataarray
    xr_dataset["ambiguity_surface"] = (
        ambiguity_surface_dim,
        ambiguity_surface,
    )
    xr_dataset["ambiguity_surface_combined"] = (
        ambiguity_surface_dim[1:],
        ambiguity_surface[0, ...],
    )
    # Add attributes
    xr_dataset.ambiguity_surface.attrs["long_name"] = "Ambiguity surface"
    xr_dataset.ambiguity_surface.attrs["units"] = "dB"
    xr_dataset.ambiguity_surface_combined.attrs["long_name"] = (
        "Ambiguity surface combined"
    )
    xr_dataset.ambiguity_surface_combined.attrs["units"] = "dB"

    # Create detected position dataarrays
    detected_pos_dim = [
        "idx_rcv_pairs",
        "idx_similarity_metric",
        "src_trajectory_time",
        "snr",
        "idx_noise_realisation",
    ]

    chunk_shape = (
        xr_dataset.sizes["idx_rcv_pairs"],
        1,
        xr_dataset.sizes["src_trajectory_time"],
        1,
        1,
    )

    detected_pos_init_lon = da.empty(
        tuple(xr_dataset.sizes[d] for d in detected_pos_dim),
        dtype=np.float32,
        chunks=chunk_shape,
    )
    detected_pos_init_lat = da.empty(
        tuple(xr_dataset.sizes[d] for d in detected_pos_dim),
        dtype=np.float32,
        chunks=chunk_shape,
    )

    xr_dataset["detected_pos_lon"] = (detected_pos_dim, detected_pos_init_lon)
    xr_dataset["detected_pos_lat"] = (detected_pos_dim, detected_pos_init_lat)
    xr_dataset["detected_pos_lon_combined"] = (
        detected_pos_dim[1:],
        detected_pos_init_lon[0, ...],
    )
    xr_dataset["detected_pos_lat_combined"] = (
        detected_pos_dim[1:],
        detected_pos_init_lat[0, ...],
    )

    return xr_dataset


def init_corr(xr_dataset):
    xr_dataset = init_corr_library(xr_dataset)
    xr_dataset = init_corr_event(xr_dataset)
    return xr_dataset


def init_corr_library(xr_dataset):
    xr_dataset = init_target_corr(xr_dataset, target="library")
    return xr_dataset


def init_corr_event(xr_dataset):
    xr_dataset = init_target_corr(xr_dataset, target="event")
    return xr_dataset


def init_target_corr(xr_dataset, target):
    if target == "library":
        target_time = "library_signal_time"
        corr_lags = "library_corr_lags"
        corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]
        corr_label = "library_corr"
        corr_long_name = r"$R_{ij}^{l}(\tau)$"

    elif target == "event":
        target_time = "event_signal_time"
        corr_lags = "event_corr_lags"
        corr_dim = ["idx_rcv_pairs", "src_trajectory_time", "event_corr_lags"]
        corr_label = "event_corr"
        corr_long_name = r"$R_{ij}^{e}(\tau)$"
    else:
        raise ValueError("Invalid target")

    lags_idx = signal.correlation_lags(
        xr_dataset.sizes[target_time], xr_dataset.sizes[target_time]
    )
    lags = lags_idx * xr_dataset[target_time].diff(target_time).values[0]

    xr_dataset.coords[corr_lags] = lags
    xr_dataset[corr_lags].attrs["units"] = "s"
    xr_dataset[corr_lags].attrs["long_name"] = "Correlation lags"

    if target == "library":
        chunksize = dict(xr_dataset.rcv_signal_library.chunksizes)
        chunk_shape = (
            xr_dataset.sizes["idx_rcv_pairs"],
            chunksize["lat"],
            chunksize["lon"],
            xr_dataset.sizes["library_corr_lags"],
        )
    elif target == "event":
        chunk_shape = (
            xr_dataset.sizes["idx_rcv_pairs"],
            xr_dataset.sizes["src_trajectory_time"],
            xr_dataset.sizes["event_corr_lags"],
        )
    else:
        raise ValueError("Invalid target")

    dummy_array = da.empty(
        tuple(xr_dataset.sizes[d] for d in corr_dim),
        dtype=np.float32,
        chunks=chunk_shape,
    )
    xr_dataset[corr_label] = (corr_dim, dummy_array)
    xr_dataset[corr_label].attrs["long_name"] = corr_long_name

    return xr_dataset


def add_correlation_library(xr_dataset, idx_snr, verbose=True):
    """
    Derive library signals cross-correlation for each grid pixel.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    if verbose:
        print(f"Add library correlation to dataset")

    # nregion_lon = get_region_number(
    #     nregion_max=xr_dataset.sizes["lon"],
    #     var=xr_dataset.library_corr,
    #     max_size_bytes=1 * 1e9,
    # )
    # nregion_lat = get_region_number(
    #     nregion_max=xr_dataset.sizes["lat"],
    #     var=xr_dataset.library_corr,
    #     max_size_bytes=1 * 1e9,
    # )

    # lon_slices, lat_slices = get_lonlat_sub_regions(
    #     xr_dataset, nregion_lon=nregion_lon, nregion_lat=nregion_lat
    # )

    # lat_chunksize = int(xr_dataset.sizes["lat"] // nregion_lat)
    # lon_chunksize = int(xr_dataset.sizes["lon"] / nregion_lon)
    # idx_rcv_chunksize, corr_chunksize = (
    #     xr_dataset.sizes["idx_rcv"],
    #     xr_dataset.sizes["library_corr_lags"],
    # )
    # chunksize = (idx_rcv_chunksize, lat_chunksize, lon_chunksize, corr_chunksize)
    # xr_dataset["library_corr"] = xr_dataset.library_corr.chunk(chunksize)

    lon_slices, lat_slices = get_lonlat_sub_regions(
        xr_dataset, xr_dataset.nregion_lon, xr_dataset.nregion_lat
    )

    for lon_s in lon_slices:
        for lat_s in lat_slices:
            ds_sub = xr_dataset.isel(lat=lat_s, lon=lon_s)
            ds_sub = add_correlation_library_subset(ds_sub)

            xr_dataset.library_corr[dict(lat=lat_s, lon=lon_s)] = ds_sub.library_corr
            sub_region_to_save = xr_dataset.library_corr[dict(lat=lat_s, lon=lon_s)]
            sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)

            sub_region_to_save.to_zarr(
                xr_dataset.output_path,
                mode="r+",
                region={
                    "snr": slice(idx_snr, idx_snr + 1),
                    "idx_rcv_pairs": slice(None),
                    "lat": lat_s,
                    "lon": lon_s,
                    "library_corr_lags": slice(None),
                },
            )

    return xr_dataset


def add_correlation_library_subset(xr_dataset):
    """
    Derive library signals cross-correlation for each grid pixel.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """

    # Faster FFT approach
    ax = 2
    nlag = xr_dataset.sizes["library_corr_lags"]

    for i_pair in range(xr_dataset.sizes["idx_rcv_pairs"]):
        rcv_pair = xr_dataset.rcv_pairs.isel(idx_rcv_pairs=i_pair)
        in1 = xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[0]).values.astype(
            np.float64
        )
        in2 = xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[1]).values.astype(
            np.float64
        )

        nfft = sp_fft.next_fast_len(nlag, True)

        sig_0 = sp_fft.rfft(
            in1,
            n=nfft,
            axis=-1,
        )
        sig_1 = sp_fft.rfft(
            in2,
            n=nfft,
            axis=-1,
        )

        corr_01_fft = fft_convolve_f(sig_0, sig_1, axis=ax, workers=-1)
        corr_01_fft = corr_01_fft[:, :, slice(nlag)]

        autocorr0 = fft_convolve_f(sig_0, sig_0, axis=ax, workers=-1)
        autocorr0 = autocorr0[:, :, slice(nlag)]

        autocorr1 = fft_convolve_f(sig_1, sig_1, axis=ax, workers=-1)
        autocorr1 = autocorr1[:, :, slice(nlag)]

        n0 = corr_01_fft.shape[-1] // 2
        corr_norm = np.sqrt(autocorr0[..., n0] * autocorr1[..., n0])
        corr_norm = np.repeat(np.expand_dims(corr_norm, axis=ax), nlag, axis=ax)
        corr_01_fft /= corr_norm
        xr_dataset.library_corr[dict(idx_rcv_pairs=i_pair)] = corr_01_fft.astype(
            np.float32
        )

    return xr_dataset


def add_correlation_event(xr_dataset, idx_snr, verbose=True):
    """
    Derive cross-correlation for each source position.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    if verbose:
        print(f"Add event correlation to dataset")

    # Derive cross_correlation vector for each src position
    for i_ship in range(xr_dataset.sizes["src_trajectory_time"]):
        for i_pair, rcv_pair in enumerate(xr_dataset.rcv_pairs):
            # Cast to float 64 to avoid 0 values in the cross correlation vector
            s0 = (
                xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[0])
                .isel(src_trajectory_time=i_ship)
                .data
            ).astype(np.float64)
            s1 = (
                xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[1])
                .isel(src_trajectory_time=i_ship)
                .data
            ).astype(np.float64)

            corr_01 = signal.correlate(s0, s1)
            autocorr0 = signal.correlate(s0, s0)
            autocorr1 = signal.correlate(s1, s1)
            n0 = corr_01.shape[0] // 2
            corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

            xr_dataset.event_corr[
                dict(idx_rcv_pairs=i_pair, src_trajectory_time=i_ship)
            ] = corr_01.astype(np.float32)

    sub_region_to_save = xr_dataset.event_corr
    sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)
    sub_region_to_save.to_zarr(
        xr_dataset.output_path,
        mode="r+",
        region={
            "snr": slice(idx_snr, idx_snr + 1),
            "idx_rcv_pairs": slice(None),
            "event_corr_lags": slice(None),
            "src_trajectory_time": slice(None),
        },
    )

    return xr_dataset


def add_noise(xr_dataset, snr_dB):
    add_noise_to_library(xr_dataset, snr_dB)
    add_noise_to_event(xr_dataset, snr_dB)
    return xr_dataset


def add_noise_to_target(xr_dataset, target_var, snr_dB):
    """
    Add noise to the target signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    for i_rcv in xr_dataset.idx_rcv.values:
        rcv_sig = xr_dataset[target_var].sel(idx_rcv=i_rcv).data

        if snr_dB is not None:
            # Add noise to received signal
            xr_dataset[target_var].loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
                rcv_sig,
                snr_dB=snr_dB,
            )

            # xr_dataset[target_var].loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
            #     np.copy(rcv_sig),
            #     snr_dB=snr_dB,
            # )

    return xr_dataset


def add_noise_to_library(xr_dataset, idx_snr, snr_dB, verbose=True):
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
    if verbose:
        print(f"Add noise to library signal")

    target_var = "rcv_signal_library"

    # nregion = get_region_number(
    #     xr_dataset.sizes["lon"], xr_dataset.rcv_signal_library, max_size_bytes=1 * 1e9
    # )
    # lon_slices, lat_slices = get_lonlat_sub_regions(xr_dataset, nregion)

    # lat_chunksize = int(xr_dataset.sizes["lat"] // nregion)
    # lon_chunksize = int(xr_dataset.sizes["lon"] / nregion)
    # idx_rcv_chunksize, time_chunksize = (
    #     xr_dataset.sizes["idx_rcv"],
    #     xr_dataset.sizes["library_signal_time"],
    # )
    # chunksize = (idx_rcv_chunksize, lat_chunksize, lon_chunksize, time_chunksize)

    # var = xr_dataset.rcv_signal_library
    # chunksize =
    # chunksize = {
    #     "idx_rcv":
    # }
    # xr_dataset["rcv_signal_library"] = xr_dataset.rcv_signal_library.chunk(chunksize)
    lon_slices, lat_slices = get_lonlat_sub_regions(
        xr_dataset, xr_dataset.nregion_lon, xr_dataset.nregion_lat
    )

    for lon_s in lon_slices:
        for lat_s in lat_slices:
            ds_sub = xr_dataset.isel(lat=lat_s, lon=lon_s)
            ds_sub = add_noise_to_target(ds_sub, target_var, snr_dB)

            xr_dataset.rcv_signal_library[dict(lat=lat_s, lon=lon_s)] = (
                ds_sub.rcv_signal_library
            )
            sub_region_to_save = xr_dataset.rcv_signal_library[
                dict(lat=lat_s, lon=lon_s)
            ]
            sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)
            sub_region_to_save.to_zarr(
                xr_dataset.output_path,
                mode="r+",
                region={
                    "snr": slice(idx_snr, idx_snr + 1),
                    "idx_rcv": slice(None),
                    "lat": lat_s,
                    "lon": lon_s,
                    "library_signal_time": slice(None),
                },
            )

    return xr_dataset


def add_noise_to_event(xr_dataset, idx_snr, snr_dB, verbose=True):
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

    if verbose:
        print(f"Add noise to event signal")

    target_var = "rcv_signal_event"
    xr_dataset = add_noise_to_target(xr_dataset, target_var, snr_dB)

    sub_region_to_save = xr_dataset.rcv_signal_event
    sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)
    sub_region_to_save.to_zarr(
        xr_dataset.output_path,
        mode="r+",
        region={
            "snr": slice(idx_snr, idx_snr + 1),
            "idx_rcv": slice(None),
            "event_signal_time": slice(None),
            "src_trajectory_time": slice(None),
        },
    )

    return xr_dataset


def add_ambiguity_surf(
    xr_dataset, idx_snr, idx_similarity_metric, i_noise, verbose=True
):
    """
    Derive ambiguity surface for each receiver pair and source position.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    idx_snr : int
        Index of the signal to noise ratio.
    idx_similarity_metric : int
        Index of the similarity metric.
    i_noise : int
        Index of the noise realization.
    verbose : bool, optional
        Verbose. The default is True.

    Returns
    -------
    None

    """

    if verbose:
        print("Derive ambiguity surface")

    xr_subset = xr_dataset.isel(
        dict(
            # snr=slice(idx_snr, idx_snr + 1),
            idx_similarity_metric=slice(
                idx_similarity_metric, idx_similarity_metric + 1
            ),
        )
    )

    lon_slices, lat_slices = get_lonlat_sub_regions(
        xr_dataset, xr_dataset.nregion_lon, xr_dataset.nregion_lat
    )

    for lon_s in lon_slices:
        for lat_s in lat_slices:
            ds_sub = xr_subset.isel(lat=lat_s, lon=lon_s)
            ds_sub = add_ambiguity_surf_subset(ds_sub, verbose)

            # Save ambiguity surface
            xr_subset.ambiguity_surface[dict(lat=lat_s, lon=lon_s)] = (
                ds_sub.ambiguity_surface
            )
            sub_region_to_save = xr_subset.ambiguity_surface[dict(lat=lat_s, lon=lon_s)]
            sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)

            # sub_region_to_save = ds_sub.ambiguity_surface
            sub_region_to_save.to_zarr(
                xr_dataset.output_path,
                mode="r+",
                region={
                    "idx_rcv_pairs": slice(None),
                    "idx_similarity_metric": slice(
                        idx_similarity_metric, idx_similarity_metric + 1
                    ),
                    "lat": lat_s,
                    "lon": lon_s,
                    "snr": slice(idx_snr, idx_snr + 1),
                    "src_trajectory_time": slice(None),
                },
            )

            # Save ambiguity surface combined
            xr_subset.ambiguity_surface_combined[dict(lat=lat_s, lon=lon_s)] = (
                ds_sub.ambiguity_surface_combined
            )
            sub_region_to_save = xr_subset.ambiguity_surface_combined[
                dict(lat=lat_s, lon=lon_s)
            ]
            sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)

            sub_region_to_save.to_zarr(
                xr_dataset.output_path,
                mode="r+",
                region={
                    "idx_similarity_metric": slice(
                        idx_similarity_metric, idx_similarity_metric + 1
                    ),
                    "lat": lat_s,
                    "lon": lon_s,
                    "snr": slice(idx_snr, idx_snr + 1),
                    "src_trajectory_time": slice(None),
                },
            )

    xr_subset = get_detected_pos(xr_subset, idx_similarity_metric, i_noise)
    # Save detected positions
    for det in ["detected_pos_lon", "detected_pos_lat"]:

        sub_region_to_save = xr_subset[det].isel(
            dict(idx_noise_realisation=slice(i_noise, i_noise + 1))
        )
        sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)

        sub_region_to_save.to_zarr(
            xr_dataset.output_path,
            mode="r+",
            region={
                "idx_noise_realisation": slice(i_noise, i_noise + 1),
                "idx_rcv_pairs": slice(None),
                "idx_similarity_metric": slice(
                    idx_similarity_metric, idx_similarity_metric + 1
                ),
                "snr": slice(idx_snr, idx_snr + 1),
                "src_trajectory_time": slice(None),
            },
        )

    # Save combined detected positions
    for det in ["detected_pos_lon_combined", "detected_pos_lat_combined"]:

        sub_region_to_save = xr_subset[det].isel(
            dict(idx_noise_realisation=slice(i_noise, i_noise + 1))
        )
        sub_region_to_save = sub_region_to_save.expand_dims({"snr": 1}, axis=0)

        sub_region_to_save.to_zarr(
            xr_dataset.output_path,
            mode="r+",
            region={
                "idx_noise_realisation": slice(i_noise, i_noise + 1),
                "idx_similarity_metric": slice(
                    idx_similarity_metric, idx_similarity_metric + 1
                ),
                "snr": slice(idx_snr, idx_snr + 1),
                "src_trajectory_time": slice(None),
            },
        )

    # Reload full dataset
    # xr_dataset = xr.open_dataset(xr_dataset.output_path, engine="zarr", chunks={})

    return xr_dataset


def add_ambiguity_surf_subset(xr_subset, verbose=True):
    """
    Derive ambiguity surface for each receiver pair and source position.

    Parameters
    ----------
    xr_subset : xr.Dataset
        Subset to process.
    idx_similarity_metric : int
        Index of the similarity metric.
    i_noise : int
        Index of the noise realization.
    verbose : bool, optional
        Verbose. The default is True.

    Returns
    -------
    None

    """

    similarity_metric = xr_subset.similarity_metrics.values[0]

    if verbose:
        det_msg = (
            f"Building ambiguity surface using {similarity_metric} as similarity metric"
        )
        print("# " + det_msg + " #")

    ambiguity_surfaces = []
    for i_pair in xr_subset.idx_rcv_pairs:
        lib_data = xr_subset.library_corr.sel(idx_rcv_pairs=i_pair)
        event_data = xr_subset.event_corr.sel(idx_rcv_pairs=i_pair)
        amb_surf_da = derive_ambiguity(
            lib_data, event_data, xr_subset.src_trajectory_time, similarity_metric
        )
        ambiguity_surfaces.append(amb_surf_da)

    # Merge dataarrays
    amb_surf_merged = xr.merge(ambiguity_surfaces)
    xr_subset.ambiguity_surface[dict(idx_similarity_metric=0)] = amb_surf_merged[
        "ambiguity_surface"
    ]

    amb_surf_combined = amb_surf_merged.ambiguity_surface.prod(dim="idx_rcv_pairs") ** (
        1 / len(xr_subset.idx_rcv_pairs)
    )
    xr_subset.ambiguity_surface_combined[dict(idx_similarity_metric=0)] = (
        amb_surf_combined
    )

    return xr_subset


def derive_ambiguity(lib_data, event_data, src_traj_times, similarity_metric):
    """
    Derive ambiguity surface for each receiver pair and source position.

    Parameters
    ----------
    lib_data : xr.DataArray
        Library Dataarray.
    event_data : xr.DataArray
        Event DataArray.
    src_traj_times : np.ndarray
        Source trajectory times.
    similarity_metric : str
        Similarity metric.

    Returns
    -------
    xr.DataArray
        Ambiguity surface.

    """

    ambiguity_surface_dim = ["idx_rcv_pairs", "src_trajectory_time", "lat", "lon"]
    ambiguity_surface = np.empty(
        (1, len(src_traj_times)) + tuple(lib_data.sizes[d] for d in ["lat", "lon"]),
        dtype=np.float32,
    )

    da_amb_surf = xr.DataArray(
        data=ambiguity_surface,
        dims=ambiguity_surface_dim,
        coords={
            "idx_rcv_pairs": [lib_data.idx_rcv_pairs.values],
            "src_trajectory_time": src_traj_times,
            "lat": lib_data.lat.values,
            "lon": lib_data.lon.values,
        },
        name="ambiguity_surface",
    )

    lib_data_array = lib_data.values.astype(
        np.float64
    )  # Cast to float64 to avoid devision by 0

    for i_src_time, src_time in enumerate(src_traj_times):

        event_vector = event_data.sel(src_trajectory_time=src_time)
        event_vector_array = event_vector.values.astype(np.float64)

        if similarity_metric == "intercorr0":
            amb_surf = mult_along_axis(
                lib_data_array,
                event_vector_array,
                axis=2,
            )
            autocorr_lib_0 = np.sum(lib_data_array**2, axis=2)
            autocorr_event_0 = np.sum(event_vector_array**2)

            norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
            amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
            amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

        elif similarity_metric == "lstsquares":
            diff = lib_data_array - event_vector_array
            amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
            amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
            amb_surf = (
                1 - amb_surf
            )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

        elif similarity_metric == "hilbert_env_intercorr0":
            lib_env = np.abs(signal.hilbert(lib_data_array))
            event_env = np.abs(signal.hilbert(event_vector_array))

            amb_surf = mult_along_axis(
                lib_env,
                event_env,
                axis=2,
            )

            autocorr_lib_0 = np.sum(lib_env**2, axis=2)
            autocorr_event_0 = np.sum(event_env**2)

            norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
            amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
            amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

        # TODO: remove this part
        # if np.any(np.isnan(amb_surf)):
        #     print(amb_surf)
        #     print(autocorr_lib_0)
        #     print(autocorr_event_0)
        #     print(norm)

    return da_amb_surf


def get_detected_pos(xr_dataset, idx_similarity_metric, i_noise, method="absmax"):
    """
    Derive detected position from ambiguity surface.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    snr_dB : float
        Signal to noise ratio.
    idx_similarity_metric : int
        Index of the similarity metric.
    i_noise : int
        Index of the noise realization.
    method : str, optional
        Method to detect source position. The default is "absmax".

    Returns
    -------
    None

    """
    ambiguity_surface = xr_dataset.ambiguity_surface
    ambiguity_surface_combined = xr_dataset.ambiguity_surface_combined

    if method == "absmax":
        max_pos_idx = ambiguity_surface.argmax(dim=["lon", "lat"])
        ilon_detected = max_pos_idx["lon"]  # Index of detected longitude
        ilat_detected = max_pos_idx["lat"]  # Index of detected longitude

        detected_lon = xr_dataset.lon.isel(lon=ilon_detected.compute()).values
        detected_lat = xr_dataset.lat.isel(lat=ilat_detected.compute()).values

        max_pos_combined_idx = ambiguity_surface_combined.argmax(dim=["lon", "lat"])
        ilon_detected_combined = max_pos_combined_idx[
            "lon"
        ]  # Index of detected longitude
        ilat_detected_combined = max_pos_combined_idx[
            "lat"
        ]  # Index of detected longitude

        detected_lon_combined = xr_dataset.lon.isel(
            lon=ilon_detected_combined.compute()
        ).values
        detected_lat_combined = xr_dataset.lat.isel(
            lat=ilat_detected_combined.compute()
        ).values

    # TODO : add other methods to take a larger number of values into account
    else:
        raise ValueError("Method not supported")

    # Store detected position in dataset
    dict_sel = dict(
        idx_noise_realisation=i_noise,
        # idx_similarity_metric=idx_similarity_metric,
    )
    # ! need to use loc when assigning values to a DataArray to avoid silent failing !
    xr_dataset.detected_pos_lon.loc[dict_sel] = detected_lon
    xr_dataset.detected_pos_lat.loc[dict_sel] = detected_lat
    xr_dataset.detected_pos_lon_combined.loc[dict_sel] = detected_lon_combined
    xr_dataset.detected_pos_lat_combined.loc[dict_sel] = detected_lat_combined

    return xr_dataset
