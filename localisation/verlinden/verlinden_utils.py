#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   verlinden_utils.py
@Time    :   2024/02/15 09:21:29
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import time
import sparse

import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing
import scipy.signal as signal
import scipy.fft as sp_fft
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Geod

# from scipy.sparse import csr_matrix

from cst import BAR_FORMAT, C0, N_CORES
from misc import mult_along_axis, fft_convolve_f
from signals import ship_noise, pulse, pulse_train, generate_ship_signal
from publication.PublicationFigure import PubFigure

# from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import waveguide_cutoff_freq, get_rcv_pos_idx

# from illustration.verlinden_nx2d import plot_angle_repartition
from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal_from_broadband_pressure_field,
)
from propa.kraken_toolbox.run_kraken import runkraken

# from propa.kraken_toolbox.plot_utils import plotshd

from localisation.verlinden.params import VERLINDEN_POPULATED_FOLDER, PROJECT_ROOT

def populate_isotropic_env(xr_dataset, library_src, signal_library_dim, testcase):
    """
    Populate the dataset with received signals in isotropic environment.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset to populate.
    library_src : AcousticSource
        Acoustic source.
    signal_library_dim : tuple
        Dimensions of the signal library.
    testcase : TestCase
        Test case.

    Returns
    -------
    xr.Dataset
        Populated dataset.
    np.ndarray
        Received signal library.
    np.ndarray
        Grid pressure field.

    """

    # Run KRAKEN
    grid_pressure_field, _ = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=library_src.kraken_freq,
        parallel=False,
        verbose=False,
    )

    delay_to_apply = xr_dataset.delay_rcv.min(dim="idx_rcv").values.flatten()

    # Loop over receivers
    for i_rcv in tqdm(
        xr_dataset.idx_rcv,
        bar_format=BAR_FORMAT,
        desc="Populate grid with received signal",
    ):

        rr_from_rcv_flat = xr_dataset.r_from_rcv.sel(idx_rcv=i_rcv).values.flatten()

        (
            t_rcv,
            s_rcv,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=testcase.env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=library_src.kraken_freq,
            source=library_src,
            rcv_range=rr_from_rcv_flat,
            rcv_depth=[library_src.z_src],
            apply_delay=True,
            delay=delay_to_apply,
            minimum_waveguide_depth=testcase.min_depth,
        )

        if i_rcv == 0:
            xr_dataset["library_signal_time"] = t_rcv.astype(np.float32)
            xr_dataset["library_signal_time"].attrs["units"] = "s"
            xr_dataset["library_signal_time"].attrs["long_name"] = "Time"

            rcv_signal_library = np.empty(
                tuple(xr_dataset.sizes[d] for d in signal_library_dim),
                dtype=np.float32,
            )

        # Time domain signal
        s_rcv = s_rcv[:, 0, :].T
        s_rcv = s_rcv.reshape(
            xr_dataset.sizes["lat"],
            xr_dataset.sizes["lon"],
            xr_dataset.sizes["library_signal_time"],
        )

        # Add 1 dimension for snrs
        rcv_signal_library[i_rcv, ...] = s_rcv

        # # Add 1 dimension for snrs
        # s_rcv = np.repeat(s_rcv[np.newaxis, :, :], xr_dataset.sizes["snr"], axis=0)
        # rcv_signal_library[:, i_rcv, ...] = s_rcv

    return xr_dataset, rcv_signal_library, grid_pressure_field


def populate_anistropic_env(
    xr_dataset, library_src, signal_library_dim, testcase, rcv_info, src_info
):
    """
    Populate the dataset with received signals in anisotropic environment.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset to populate.
    library_src : AcousticSource
        Acoustic source.
    signal_library_dim : tuple
        Dimensions of the signal library.
    testcase : TestCase
        Test case.
    rcv_info : dict
        Receiver information.
    src_info : dict
        Source information.

    Returns
    -------
    xr.Dataset
        Populated dataset.
    np.ndarray
        Received signal library.
    np.ndarray
        Grid pressure field.

    """

    # Array of receiver indexes
    idx_rcv = xr_dataset.idx_rcv.values

    # Grid pressure field for all azimuths : list of nested lists required due to inhomogeneous sizes (depending on the azimuths)
    grid_pressure_field = []
    kraken_grid = {}
    kraken_range_rcv = []
    kraken_depth_rcv = []
    kraken_min_wg_depth_rcv = []
    rcv_depth = [library_src.z_src]

    # Loop over receivers
    for i_rcv in tqdm(
        idx_rcv,
        bar_format=BAR_FORMAT,
        desc="Populate grid with received signal",
    ):

        # Loop over possible azimuths
        azimuths_rcv = get_unique_azimuths(xr_dataset, i_rcv)
        az_pressure_field = []
        az_kraken_min_wg_depth = []

        for i_az in tqdm(
            range(azimuths_rcv.size),
            bar_format=BAR_FORMAT,
            desc="Scanning azimuths",
        ):

            az = azimuths_rcv[i_az]

            # Get environement for selected angle
            testcase_varin = dict(
                freq=library_src.kraken_freq,
                max_range_m=rcv_info["max_kraken_range_m"][i_rcv],
                azimuth=az,
                rcv_lon=rcv_info["lons"][i_rcv],
                rcv_lat=rcv_info["lats"][i_rcv],
            )
            # kraken_env, kraken_flp = testcase(testcase_varin)
            testcase.update(testcase_varin)

            # Assert kraken freq is set with correct min_depth (otherwise postprocess will fail)
            # kraken_env, kraken_flp, library_src = check_waveguide_cutoff(
            #     testcase=testcase,
            #     testcase_varin=testcase_varin,
            #     kraken_env=kraken_env,
            #     kraken_flp=kraken_flp,
            #     library_src=library_src,
            #     dt=src_info["dt"],
            #     max_range_m=rcv_info["max_kraken_range_m"][i_rcv],
            #     sig_type=src_info["signal_type"],
            # )
            library_src = check_waveguide_cutoff(
                testcase=testcase,
                src=library_src,
                dt=src_info["dt"],
                src_info=src_info["library"],
            )

            # Store minimum waveguide depth to be used when populating env with events
            # min_waveguide_depth = kraken_env.bathy.bathy_depth.min()
            min_waveguide_depth = testcase.min_depth
            az_kraken_min_wg_depth.append(min_waveguide_depth)

            # Get receiver ranges for selected angle
            idx_az = xr_dataset.az_propa.sel(idx_rcv=i_rcv).values == az
            rr_from_rcv_az = (
                xr_dataset.r_from_rcv.sel(idx_rcv=i_rcv).values[idx_az].flatten()
            )
            # Get received signal for selected angle
            delay_to_apply = (
                xr_dataset.delay_rcv.min(dim="idx_rcv").values[idx_az].flatten()
            )

            # Run kraken for selected angle
            pf, field_pos = runkraken(
                env=testcase.env,
                flp=testcase.flp,
                frequencies=library_src.kraken_freq,
                parallel=True,
                verbose=False,
            )
            if i_az == 0:
                # Store grid pos in dataset (receiver grid depends on rcv)
                r_rcv = field_pos["r"]["r"]
                z_rcv = field_pos["r"]["z"]
                kraken_range_rcv.append(r_rcv)
                kraken_depth_rcv.append(z_rcv)

            (
                t_rcv,
                s_rcv,
                _,
            ) = postprocess_received_signal_from_broadband_pressure_field(
                kraken_range=r_rcv,
                kraken_depth=z_rcv,
                broadband_pressure_field=pf,
                frequencies=library_src.kraken_freq,
                source=library_src,
                rcv_range=rr_from_rcv_az,
                rcv_depth=rcv_depth,
                apply_delay=True,
                delay=delay_to_apply,
                minimum_waveguide_depth=min_waveguide_depth,
            )

            pf = np.squeeze(pf, axis=(1, 2))

            # Values outside of the grid range are set to 0j and the sparse pressure field is stored to save memory
            r_offset = 5 * xr_dataset.dx
            r_min_usefull = np.min(rr_from_rcv_az) - r_offset
            r_max_usefull = np.max(rr_from_rcv_az) + r_offset
            idx_r_min_usefull = np.argmin(np.abs(r_rcv - r_min_usefull))
            idx_r_max_usefull = np.argmin(np.abs(r_rcv - r_max_usefull))
            all_idx = np.arange(r_rcv.size)
            idx_not_usefull = (all_idx <= idx_r_min_usefull) | (
                all_idx >= idx_r_max_usefull
            )
            pf[:, :, idx_not_usefull] = 0j

            sparse_pf = sparse.COO(pf, fill_value=0j)
            az_pressure_field.append(sparse_pf)
            # az_pressure_field[i_az] = sparse_pf

            # Init dataset variable
            if i_rcv == 0 and i_az == 0:
                xr_dataset["library_signal_time"] = t_rcv.astype(np.float32)
                xr_dataset["library_signal_time"].attrs["units"] = "s"
                xr_dataset["library_signal_time"].attrs["long_name"] = "Time"
                rcv_signal_library = np.empty(
                    tuple(xr_dataset.sizes[d] for d in signal_library_dim),
                    dtype=np.float32,
                )

            # Time domain signal
            s_rcv = s_rcv[:, 0, :].T
            # 1 dataset / snr version
            rcv_signal_library[i_rcv, idx_az, ...] = s_rcv

            # 1 dataset for all snrs version
            # # Add 1 dimension for snrs
            # s_rcv = np.repeat(s_rcv[np.newaxis, :, :], xr_dataset.sizes["snr"], axis=0)
            # rcv_signal_library[:, i_rcv, idx_az, ...] = s_rcv

            # print(f"Contains nans ? : {np.any(np.isnan(s_rcv.astype(np.float32)))}")
            # if np.any(np.abs(s_rcv) > 1e20):
            #     greater = True
            #     print(f"Is greater than 1e20")
            # if np.any(np.logical_and((0 < np.abs(s_rcv)), (np.abs(s_rcv) < 1e-20))):
            #     smaller = True
            #     print(f"Is smaller than 1e-20")

        # Store pressure field for all azimuths relative to current rcv
        grid_pressure_field.append(az_pressure_field)
        # Store minimum waveguide depth for all azimuths relative to current rcv
        kraken_min_wg_depth_rcv.append(az_kraken_min_wg_depth)

    # Cast kraken grid range/depth arrays to the same size
    nr_max = np.max(
        [kraken_range_rcv[i_rcv].size for i_rcv in xr_dataset.idx_rcv.values]
    )
    nz_max = np.max(
        [kraken_depth_rcv[i_rcv].size for i_rcv in xr_dataset.idx_rcv.values]
    )

    # Store kraken range and depth for rcv
    for i_rcv in xr_dataset.idx_rcv.values:
        if kraken_range_rcv[i_rcv].size < nr_max:
            kraken_range_rcv[i_rcv] = np.pad(
                kraken_range_rcv[i_rcv],
                (0, nr_max - kraken_range_rcv[i_rcv].size),
                constant_values=np.nan,
            )

        if kraken_depth_rcv[i_rcv].size < nz_max:
            kraken_depth_rcv[i_rcv] = np.pad(
                kraken_depth_rcv[i_rcv],
                (0, nz_max - kraken_depth_rcv[i_rcv].size),
                constant_values=np.nan,
            )

        # Size calculation.
        # print("Receiver %s" % i_rcv)
        # print("Size of kraken_range_rcv in bytes: %s" % kraken_range_rcv[i_rcv].nbytes)
        # print(
        #     "Size of sparse kraken_range_rcv in bytes: %s" % sp_kraken_range_rcv.nbytes
        # )
        # print("Size of kraken_depth_rcv in bytes: %s" % kraken_depth_rcv[i_rcv].nbytes)
        # print(
        #     "Size of sparse kraken_depth_rcv in bytes: %s" % sp_kraken_depth_rcv.nbytes
        # )

    # Convert to float array
    kraken_range_rcv = np.array(kraken_range_rcv, dtype=np.float32)
    kraken_depth_rcv = np.array(kraken_depth_rcv, dtype=np.float32)

    # Sparse representation to save memory
    kraken_range_rcv = sparse.COO(kraken_range_rcv, fill_value=np.nan)
    kraken_depth_rcv = sparse.COO(kraken_depth_rcv, fill_value=np.nan)

    for i_rcv in xr_dataset.idx_rcv.values:
        kraken_grid[i_rcv] = {
            "range": kraken_range_rcv[i_rcv, :],
            "depth": kraken_depth_rcv[i_rcv, :],
            "min_waveguide_depth": {
                i_az: kraken_min_wg_depth_rcv[i_rcv][i_az]
                for i_az in range(len(kraken_min_wg_depth_rcv[i_rcv]))
            },
        }

    return xr_dataset, rcv_signal_library, grid_pressure_field, kraken_grid


def add_event_isotropic_env(
    xr_dataset,
    # snr_dB,
    event_src,
    kraken_env,
    signal_event_dim,
    grid_pressure_field,
    # init_corr=False,
):
    """
    Add event in isotropic environment.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset to populate.
    snr_dB : float
        Signal to noise ratio.
    event_src : AcousticSource
        Acoustic source.
    kraken_env : KrakenEnv
        Kraken environment.
    signal_event_dim : tuple
        Dimensions of the signal event.
    grid_pressure_field : np.ndarray
        Grid pressure field.

    Returns
    -------
    None

    """

    rcv_depth = [event_src.z_src]

    # Derive received signal for successive positions of the ship
    for i_rcv in tqdm(
        range(xr_dataset.sizes["idx_rcv"]),
        bar_format=BAR_FORMAT,
        desc="Derive received signal for successive positions of the ship",
    ):

        delay_to_apply_ship = (
            xr_dataset.delay_rcv.min(dim="idx_rcv")
            .sel(lon=xr_dataset.lon_src, lat=xr_dataset.lat_src, method="nearest")
            .values.flatten()
        )

        (
            t_rcv,
            s_rcv,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=kraken_env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=event_src.kraken_freq,
            source=event_src,
            rcv_range=xr_dataset.r_src_rcv.sel(idx_rcv=i_rcv).values,
            rcv_depth=rcv_depth,
            apply_delay=True,
            delay=delay_to_apply_ship,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
        )

        if i_rcv == 0:
            xr_dataset["event_signal_time"] = t_rcv.astype(np.float32)
            rcv_signal_event = np.empty(
                tuple(xr_dataset.sizes[d] for d in signal_event_dim), dtype=np.float32
            )

        s_rcv = s_rcv[:, 0, :].T
        # 1 dataset / snr version
        rcv_signal_event[i_rcv, ...] = s_rcv

        # # Add 1 dimension for snrs
        # s_rcv = np.repeat(s_rcv[np.newaxis, ...], xr_dataset.sizes["snr"], axis=0)
        # rcv_signal_event[:, i_rcv, ...] = s_rcv

    xr_dataset["rcv_signal_event"] = (
        signal_event_dim,
        rcv_signal_event.astype(np.float32),
    )

    noise_free = rcv_signal_event.copy()
    xr_dataset["rcv_signal_event_noise_free"] = (
        signal_event_dim,
        noise_free.astype(np.float32),
    )

    # add_noise_to_event(xr_dataset, snr_dB=snr_dB)
    # add_event_correlation(xr_dataset, snr_dB=snr_dB, init_corr=init_corr)


def add_event_anisotropic_env(
    xr_dataset,
    event_src,
    kraken_grid,
    signal_event_dim,
    grid_pressure_field,
):
    """
    Add event in anisotropic environment.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset to populate.
    snr_dB : float
        Signal to noise ratio.
    event_src : AcousticSource
        Acoustic source.
    kraken_grid : dict
        Kraken grid.
    signal_event_dim : tuple
        Dimensions of the signal event.
    grid_pressure_field : np.ndarray
        Grid pressure field.
    init_corr : bool, optional
        Initialize correlation. The default is True.

    Returns
    -------
    None

    """

    # Array of receiver indexes
    idx_rcv = xr_dataset.idx_rcv.values
    # Array of ship positions idx
    idx_src_pos = np.arange(xr_dataset.sizes["src_trajectory_time"])
    # Array of receiver depths
    rcv_depth = [event_src.z_src]

    # Loop over receivers
    for i_rcv in tqdm(
        idx_rcv,
        bar_format=BAR_FORMAT,
        desc="Derive received signal for successive positions of the ship",
    ):

        # Get kraken grid for the given rcv
        kraken_range = kraken_grid[i_rcv]["range"].todense()
        kraken_depth = kraken_grid[i_rcv]["depth"].todense()

        # Loop over src positions
        for i_src in tqdm(
            idx_src_pos,
            bar_format=BAR_FORMAT,
            desc="Scanning src positions",
        ):

            delay_to_apply_ship = (
                xr_dataset.delay_rcv.min(dim="idx_rcv")
                .sel(lon=xr_dataset.lon_src, lat=xr_dataset.lat_src, method="nearest")
                .values.flatten()
            )

            # Get azimuths for current ship position
            i_az_kraken, az_kraken, rcv_range, transfert_function = (
                get_src_transfert_function(
                    xr_dataset, i_rcv, i_src, grid_pressure_field
                )
            )
            rcv_range = rcv_range.reshape(1)
            transfert_function = transfert_function.todense()
            min_wg_depth = kraken_grid[i_rcv]["min_waveguide_depth"][i_az_kraken]

            (
                t_rcv,
                s_rcv,
                Pos,
            ) = postprocess_received_signal_from_broadband_pressure_field(
                kraken_range=kraken_range,
                kraken_depth=kraken_depth,
                broadband_pressure_field=transfert_function,
                frequencies=event_src.kraken_freq,
                source=event_src,
                rcv_range=rcv_range,
                rcv_depth=rcv_depth,
                apply_delay=True,
                delay=delay_to_apply_ship,
                minimum_waveguide_depth=min_wg_depth,
                squeeze=False,
            )

            if i_rcv == 0 and i_src == 0:
                xr_dataset["event_signal_time"] = t_rcv.astype(np.float32)
                rcv_signal_event = np.empty(
                    tuple(xr_dataset.sizes[d] for d in signal_event_dim), dtype=np.float32
                )

            s_rcv = s_rcv[:, 0, :].T

            # 1 dataset / snr version
            rcv_signal_event[i_rcv, i_src, ...] = s_rcv

            # # Add 1 dimension for snrs
            # s_rcv = np.repeat(s_rcv[np.newaxis, ...], xr_dataset.sizes["snr"], axis=0)
            # rcv_signal_event[:, i_rcv, i_src, ...] = np.squeeze(
            #     s_rcv
            # )  # Squeeze to remove the 1 dimension for range

    # if not "rcv_signal_event" in xr_dataset:
    xr_dataset["rcv_signal_event"] = (
        signal_event_dim,
        rcv_signal_event.astype(np.float32),
    )

    xr_dataset["rcv_signal_event_noise_free"] = (
        signal_event_dim,
        rcv_signal_event.astype(np.float32),
    )


def get_src_transfert_function(xr_dataset, i_rcv, i_src, grid_pressure_field):
    """
    Extract waveguide transfert function for the given source position index and receiver index.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Library dataset.
    i_rcv : int
        Receiver index.
    i_src : int
        Source position index.
    grid_pressure_field : np.ndarray
        Grid pressure field.

    Returns
    -------
    int
        Index of the azimuth for which we have the transfert function.
    float
        Azimuth for which we have the transfert function.
    float
        Range along the given azimuth.
    np.ndarray
        Transfert function for the given azimuth.

    """
    az_propa_unique = get_unique_azimuths(xr_dataset, i_rcv)
    az_src = (
        xr_dataset.az_src_rcv.sel(idx_rcv=i_rcv).isel(src_trajectory_time=i_src).values
    )
    i_az_src = np.argmin(np.abs(az_propa_unique - az_src))

    az = az_propa_unique[i_az_src]  # Azimuth for which we have the transfert function
    transfert_function = grid_pressure_field[i_rcv][
        i_az_src
    ]  # Transfert function for the given azimuth
    rcv_range = (
        xr_dataset.r_src_rcv.sel(idx_rcv=i_rcv).isel(src_trajectory_time=i_src).values
    )  # Range along the given azimuth

    return i_az_src, az, rcv_range, transfert_function


def get_unique_azimuths(xr_dataset, i_rcv):
    """
    Get unique azimuths for the given receiver index.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Library dataset.
    i_rcv : int
        Receiver index.

    Returns
    -------
    np.ndarray
        Unique azimuths.

    """
    return np.unique(xr_dataset.az_propa.sel(idx_rcv=i_rcv).values)


def get_range_from_rcv(grid_info, rcv_info):
    """
    Compute range between grid points and receivers.
    This function is the equivalent of get_azimuth_rcv for ranges.

    Parameters
    ----------
    grid_info : dict
        Grid information.
    rcv_info : dict
        Receiver information.

    Returns
    -------
    np.ndarray
        Range between grid points and receivers.

    """
    llon, llat = np.meshgrid(grid_info["lons"], grid_info["lats"])
    s = llon.shape
    geod = Geod(ellps="WGS84")

    rr_from_rcv = np.empty((len(rcv_info["id"]), s[0], s[1]))
    for i, id in enumerate(rcv_info["id"]):
        # Derive distance from rcv n°i to all grid points
        _, _, ranges = geod.inv(
            lons1=np.ones(s) * rcv_info["lons"][i],
            lats1=np.ones(s) * rcv_info["lats"][i],
            lons2=llon,
            lats2=llat,
            return_back_azimuth=True,
        )
        rr_from_rcv[i, :, :] = ranges

    return rr_from_rcv


def get_range_src_rcv_range(lon_src, lat_src, rcv_info):
    """
    Compute range between source and receivers positions.
    This function is the equivalent of get_range_from_rcv for source positions.

    Parameters
    ----------
    lon_src : np.ndarray
        Source longitudes.
    lat_src : np.ndarray
        Source latitudes.
    rcv_info : dict
        Receiver information.

    Returns
    -------
    np.ndarray
        Range between source and receivers positions.

    """
    geod = Geod(ellps="WGS84")

    s = lon_src.size
    rr_src_rcv = np.empty((len(rcv_info["id"]), s))
    for i in range(len(rcv_info["id"])):
        # Derive distance from rcv n°i to src positions
        _, _, ranges = geod.inv(
            lons1=np.ones(s) * rcv_info["lons"][i],
            lats1=np.ones(s) * rcv_info["lats"][i],
            lons2=lon_src,
            lats2=lat_src,
            return_back_azimuth=True,
        )
        rr_src_rcv[i, :] = ranges

    return rr_src_rcv


def get_azimuth_rcv(grid_info, rcv_info):
    """
    Compute azimuth between grid points and receivers positions.
    This function is the equivalent of get_range_from_rcv for azimuths.

    Parameters
    ----------
    grid_info : dict
        Grid information.
    rcv_info : dict
        Receiver information.

    Returns
    -------
    np.ndarray
        Azimuth between grid points and receivers positions.

    """

    llon, llat = np.meshgrid(grid_info["lons"], grid_info["lats"])
    s = llon.shape

    geod = Geod(ellps="WGS84")
    az_rcv = np.empty((len(rcv_info["id"]), s[0], s[1]))

    for i, id in enumerate(rcv_info["id"]):
        # Derive azimuth from rcv n°i to all grid points
        fwd_az, _, _ = geod.inv(
            lons1=np.ones(s) * rcv_info["lons"][i],
            lats1=np.ones(s) * rcv_info["lats"][i],
            lons2=llon,
            lats2=llat,
            return_back_azimuth=True,
        )
        az_rcv[i, :, :] = fwd_az

    return az_rcv


def get_azimuth_src_rcv(lon_src, lat_src, rcv_info):
    """Compute azimuth between source and receivers positions.
    This function is the equivalent of get_azimuth_rcv for source positions.

    Parameters
    ----------
    lon_src : np.ndarray
        Source longitudes.
    lat_src : np.ndarray
        Source latitudes.
    rcv_info : dict
        Receiver information.

    Returns
    -------
    np.ndarray
        Azimuth between source and receivers positions.

    """

    s = lon_src.size

    geod = Geod(ellps="WGS84")
    az_rcv = np.empty((len(rcv_info["id"]), s))

    for i, id in enumerate(rcv_info["id"]):
        # Derive azimuth from rcv n°i to src positions
        fwd_az, _, _ = geod.inv(
            lons1=np.ones(s) * rcv_info["lons"][i],
            lats1=np.ones(s) * rcv_info["lats"][i],
            lons2=lon_src,
            lats2=lat_src,
            return_back_azimuth=True,
        )
        az_rcv[i, :] = fwd_az

    return az_rcv


def init_library_dataset(
    grid_info,
    rcv_info,
    src_info,
    snrs_dB,
    n_noise_realisations,
    similarity_metrics,
    testcase,
    file_ext,
):
    """
    Initialize the dataset for the library.

    Parameters
    ----------
    grid_info : dict
        Grid information.
    rcv_info : dict
        Receiver information.
    n_noise_realisations : int
        Number of noise realisations.
    similarity_metrics : list
        List of similarity metrics.
    isotropic_env : bool, optional
        Isotropic environment. The default is True.

    Returns
    -------
    xr.Dataset
        Initialized dataset.

    """

    # Init Dataset
    n_rcv = len(rcv_info["id"])
    n_similarity_metrics = len(similarity_metrics)

    # Compute range from each receiver
    rr_rcv = get_range_from_rcv(grid_info, rcv_info)

    xr_dataset = xr.Dataset(
        data_vars=dict(
            lon_rcv=(["idx_rcv"], rcv_info["lons"]),
            lat_rcv=(["idx_rcv"], rcv_info["lats"]),
            rcv_id=(["idx_rcv"], rcv_info["id"]),
            r_from_rcv=(["idx_rcv", "lat", "lon"], rr_rcv),
            similarity_metric=(
                ["idx_similarity_metric"],
                similarity_metrics,
            ),
        ),
        coords=dict(
            lon=grid_info["lons"],
            lat=grid_info["lats"],
            snr=snrs_dB,
            idx_rcv=np.arange(n_rcv),
            idx_similarity_metric=np.arange(n_similarity_metrics),
            idx_noise_realisation=np.arange(n_noise_realisations),
        ),
        attrs=dict(
            title=testcase.title,
            description=testcase.desc,
            dx=grid_info["dx"],
            dy=grid_info["dy"],
        ),
    )

    xr_dataset["delay_rcv"] = xr_dataset.r_from_rcv / C0

    if not testcase.isotropic:
        # Associate azimuths to grid cells for each receiver
        xr_dataset = set_azimuths(xr_dataset, grid_info, rcv_info, n_rcv)

    # Set attrs 
    set_dataset_attrs(xr_dataset, grid_info, testcase, src_info, file_ext)

    # Build rcv pairs
    build_rcv_pairs(xr_dataset)

    return xr_dataset

def build_rcv_pairs(xr_dataset):
    rcv_pairs = []
    for i in xr_dataset.idx_rcv.values:
        for j in range(i + 1, xr_dataset.idx_rcv.values[-1] + 1):
            rcv_pairs.append((i, j))
    xr_dataset.coords["idx_rcv_pairs"] = np.arange(len(rcv_pairs))
    xr_dataset.coords["idx_rcv_in_pair"] = np.arange(2)
    xr_dataset["rcv_pairs"] = (["idx_rcv_pairs", "idx_rcv_in_pair"], rcv_pairs)

    # Build pair ids
    pair_ids = []
    for i_rcv_pair in xr_dataset["idx_rcv_pairs"]:
        r0_id = xr_dataset.rcv_id.isel(
            idx_rcv=xr_dataset.rcv_pairs.isel(
                idx_rcv_pairs=i_rcv_pair, idx_rcv_in_pair=0
            )
        ).values
        r1_id = xr_dataset.rcv_id.isel(
            idx_rcv=xr_dataset.rcv_pairs.isel(
                idx_rcv_pairs=i_rcv_pair, idx_rcv_in_pair=1
            )
        ).values
        pair_ids.append(f"{r0_id}{r1_id}")
    xr_dataset["rcv_pair_id"] = (["idx_rcv_pairs"], pair_ids)


def set_dataset_attrs(xr_dataset, grid_info, testcase, src_info, file_ext):
    # Set attributes
    var_unit_mapping = {
        "°": [
            "lon_rcv",
            "lat_rcv",
            "lon",
            "lat",
        ],
        "m": ["r_from_rcv"],
        "dB": ["snr"],
        "": ["idx_rcv", "idx_similarity_metric", "idx_noise_realisation"],
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
    xr_dataset["snr"].attrs["long_name"] = "Signal to noise ratio"
    xr_dataset["idx_rcv"].attrs["long_name"] = "Receiver index"
    xr_dataset["idx_similarity_metric"].attrs["long_name"] = "Similarity metric index"
    xr_dataset["idx_noise_realisation"].attrs["long_name"] = "Noise realisation index"
    xr_dataset["delay_rcv"].attrs["long_name"] = "Propagation delay from receiver"

    # Initialisation time
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    xr_dataset.attrs["init_time"] = now

    xr_dataset.attrs["fullpath_populated"] = get_populated_path(
        grid_info,
        kraken_env=testcase.env,
        src_signal_type=src_info["signal_type"],
        init_time=xr_dataset.attrs["init_time"],
        ext=file_ext,
    )


def set_azimuths(xr_dataset, grid_info, rcv_info, n_rcv):
    az_rcv = get_azimuth_rcv(grid_info, rcv_info)
    xr_dataset["az_rcv"] = (["idx_rcv", "lat", "lon"], az_rcv)

    # Build list of angles to be used in kraken
    dmax = xr_dataset.r_from_rcv.max(dim=["lat", "lon", "idx_rcv"]).round(0).values
    delta = np.sqrt(grid_info["dlat_bathy"] ** 2 + grid_info["dlon_bathy"] ** 2)
    d_az = np.arctan(delta / dmax) * 180 / np.pi
    # list_az_th = np.arange(xr_dataset.az_rcv.min(), xr_dataset.az_rcv.max(), d_az)
    list_az_th = np.arange(-180, 180, d_az)

    az_propa = np.empty(xr_dataset.az_rcv.shape)
    # t0 = time.time()
    # visited_sites = az_propa.copy()
    # visited_sites[:] = False
    for i_rcv in range(n_rcv):
        for i_az, az in enumerate(list_az_th):
            if i_az == len(list_az_th) - 1:
                closest_points_idx = (
                    xr_dataset.az_rcv.sel(idx_rcv=i_rcv) >= az - d_az / 2
                )
            else:
                closest_points_idx = np.logical_and(
                    xr_dataset.az_rcv.sel(idx_rcv=i_rcv) >= az - d_az / 2,
                    xr_dataset.az_rcv.sel(idx_rcv=i_rcv) < az + d_az / 2,
                )

            az_propa[i_rcv, closest_points_idx] = az
            # visited_sites[i_rcv, closest_points_idx] = True
    # print(f"Elapsed time : {time.time() - t0}")

    # Add az_propa to dataset
    xr_dataset["az_propa"] = (["idx_rcv", "lat", "lon"], az_propa)
    xr_dataset["az_propa"].attrs["units"] = "°"
    xr_dataset["az_propa"].attrs["long_name"] = "Propagation azimuth"

    list_az = [get_unique_azimuths(xr_dataset, i_rcv) for i_rcv in range(n_rcv)]
    n_az_max = max([az.size for az in list_az])
    azimuth_rcv = np.array(
        [np.pad(az, (0, n_az_max - len(az)), constant_values=np.nan) for az in list_az]
    )  # Pad to get homegenous dimensions
    xr_dataset["azimuths_rcv"] = (["idx_rcv", "azimuth"], azimuth_rcv)

    # Remove az_rcv
    ds = xr_dataset.drop_vars("az_rcv")

    return ds 


def init_event_dataset(xr_dataset, src_info, rcv_info, interp_src_pos_on_grid=False):
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
    az_src_rcv = get_azimuth_src_rcv(lon_src, lat_src, rcv_info)

    xr_dataset.coords["event_signal_time"] = []
    xr_dataset.coords["src_trajectory_time"] = t_src.astype(np.float32)

    xr_dataset["lon_src"] = (["src_trajectory_time"], lon_src.astype(np.float32))
    xr_dataset["lat_src"] = (["src_trajectory_time"], lat_src.astype(np.float32))
    xr_dataset["r_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(r_src_rcv).astype(np.float32),
    )
    xr_dataset["az_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(az_src_rcv).astype(np.float32),
    )

    xr_dataset["event_signal_time"].attrs["units"] = "s"
    xr_dataset["src_trajectory_time"].attrs["units"] = "s"

    xr_dataset["lon_src"].attrs["long_name"] = "lon_src"
    xr_dataset["lat_src"].attrs["long_name"] = "lat_src"
    xr_dataset["r_src_rcv"].attrs["long_name"] = "Range from receiver to source"
    xr_dataset["event_signal_time"].attrs["units"] = "Time"
    xr_dataset["src_trajectory_time"].attrs["long_name"] = "Time"

    if interp_src_pos_on_grid:
        xr_dataset["lon_src"].values = xr_dataset.lon.sel(
            lon=xr_dataset.lon_src, method="nearest"
        )
        xr_dataset["lat_src"].values = xr_dataset.lat.sel(
            lat=xr_dataset.lat_src, method="nearest"
        )
        xr_dataset["r_src_rcv"].values = get_range_src_rcv_range(
            xr_dataset["lon_src"], xr_dataset["lat_src"], rcv_info
        )
        xr_dataset.attrs["source_positions"] = "Interpolated on grid"
        xr_dataset.attrs["src_pos"] = "on_grid"
    else:
        xr_dataset.attrs["source_positions"] = "Not interpolated on grid"
        xr_dataset.attrs["src_pos"] = "not_on_grid"

    return xr_dataset


def check_waveguide_cutoff(
    testcase,
    src,
    dt,
    src_info,
):
    """
    Check if the waveguide cutoff frequency is set correctly in the source library.

    Parameters
    ----------
    testcase : Testcase
        Testcase.
    src : AcouticSource
        Acoustic source.
    dt : float
        Time step.
    src_info : dict
        Source information.

    Returns
    -------
    AcousticSource
        Updated acoustic source library.

    """

    fc = waveguide_cutoff_freq(waveguide_depth=testcase.min_depth)
    propagating_freq = src.positive_freq[src.positive_freq > fc]
    if propagating_freq.size != src.kraken_freq.size:
        src = init_src(dt, testcase.min_depth, src_info)

        # Update testcase with new frequency vector
        varin = {"freq": src.kraken_freq}
        testcase.update(varin)

    return src


def add_noise_to_dataset(xr_dataset, snr_dB):
    """
    Add noise to received signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """
    for i_rcv in tqdm(
        xr_dataset.idx_rcv.values,
        bar_format=BAR_FORMAT,
        desc="Add noise to received signal",
    ):
        rcv_sig = xr_dataset.rcv_signal_library_noise_free.sel(idx_rcv=i_rcv).values

        # for snr_dB in xr_dataset.snr.values:

        if snr_dB is not None:
            # Add noise to received signal
            xr_dataset.rcv_signal_library.loc[dict(idx_rcv=i_rcv)] = (
                add_noise_to_signal(
                    np.copy(rcv_sig),
                    snr_dB=snr_dB,
                )
            )

            # xr_dataset.attrs["snr_dB"] = snr_dB
        else:
            # xr_dataset.attrs["snr_dB"] = "Noiseless"
            pass  # TODO: need to be updated to fit with snr integration in a single dataset

    # for i_rcv in tqdm(
    #     xr_dataset.idx_rcv.values,
    #     bar_format=BAR_FORMAT,
    #     desc="Add noise to received signal",
    # ):
    #     rcv_sig = (
    #         xr_dataset.rcv_signal_library_noise_free.sel(idx_rcv=i_rcv).load().values
    #     )

    #     for snr_dB in xr_dataset.snr.values:

    #         if snr_dB is not None:
    #             # Add noise to received signal
    #             xr_dataset.rcv_signal_library.loc[dict(idx_rcv=i_rcv, snr=snr_dB)] = (
    #                 add_noise_to_signal(
    #                     np.copy(rcv_sig),
    #                     snr_dB=snr_dB,
    #                 )
    #             )

    #             # xr_dataset.attrs["snr_dB"] = snr_dB
    #         else:
    #             # xr_dataset.attrs["snr_dB"] = "Noiseless"
    #             pass  # TODO: need to be updated to fit with snr integration in a single dataset


def init_corr_library(xr_dataset):
    # Derive correlation lags
    lags_idx = signal.correlation_lags(
        xr_dataset.sizes["library_signal_time"], xr_dataset.sizes["library_signal_time"]
    )
    lags = (
        lags_idx * xr_dataset.library_signal_time.diff("library_signal_time").values[0]
    )
    xr_dataset.coords["library_corr_lags"] = lags
    xr_dataset["library_corr_lags"].attrs["units"] = "s"
    xr_dataset["library_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each grid pixel
    library_corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]
    library_corr = np.empty(
        tuple(xr_dataset.sizes[d] for d in library_corr_dim), dtype=np.float32
    )

    # Init variable
    xr_dataset["library_corr"] = (library_corr_dim, library_corr.astype(np.float32))
    xr_dataset["library_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"
    # Expand dims to add snr dimension to the library_corr dataarray
    # Using expand_dims to avoid loading the entire dataset in memory -> overflow
    # xr_dataset["library_corr"] = xr_dataset.library_corr.expand_dims(
    #     dim={"snr": xr_dataset.sizes["snr"]}
    # ).assign_coords({"snr": xr_dataset.snr})


def add_correlation_to_dataset(xr_dataset):
    """
    Derive correlation for each grid pixel.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """

    # # Derive correlation lags
    # lags_idx = signal.correlation_lags(
    #     xr_dataset.sizes["library_signal_time"], xr_dataset.sizes["library_signal_time"]
    # )
    # lags = (
    #     lags_idx * xr_dataset.library_signal_time.diff("library_signal_time").values[0]
    # )
    # xr_dataset.coords["library_corr_lags"] = lags
    # xr_dataset["library_corr_lags"].attrs["units"] = "s"
    # xr_dataset["library_corr_lags"].attrs["long_name"] = "Correlation lags"

    # # Derive cross_correlation vector for each grid pixel
    # library_corr = np.empty(
    #     tuple(xr_dataset.sizes[d] for d in library_corr_dim), dtype=np.float32
    # )

    # # Init variable
    # xr_dataset["library_corr"] = (library_corr_dim, library_corr.astype(np.float32))
    # xr_dataset["library_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"
    # # Expand dims to add snr dimension to the library_corr dataarray
    # # Using expand_dims to avoid loading the entire dataset in memory -> overflow
    # xr_dataset["library_corr"] = xr_dataset.library_corr.expand_dims(
    #     dim={"snr": xr_dataset.sizes["snr"]}
    # ).assign_coords({"snr": xr_dataset.snr})

    # # Close an reopen to ensure dataarray is mutable
    # xr_dataset.to_zarr(xr_dataset.fullpath_populated, mode="a")
    # xr_dataset.close()
    # # Open mutable zarr dataset
    # xr_dataset = xr.open_zarr(xr_dataset.fullpath_populated)

    # Faster FFT approach
    ax = 2
    nlag = xr_dataset.sizes["library_corr_lags"]

    library_corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]
    library_corr = np.empty(
        tuple(xr_dataset.sizes[d] for d in library_corr_dim), dtype=np.float32
    )

    for i_pair in range(xr_dataset.sizes["idx_rcv_pairs"]):

        rcv_pair = xr_dataset.rcv_pairs.isel(idx_rcv_pairs=i_pair)
        in1 = xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[0]).values
        in2 = xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[1]).values

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
        # library_corr[idx_snr, i_pair, ...] = corr_01_fft
        library_corr[i_pair, ...] = corr_01_fft

    # Assign to dataset
    xr_dataset["library_corr"].values = library_corr.astype(np.float32)

    # library_corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]

    # # Faster FFT approach
    # ax = 2
    # nlag = xr_dataset.sizes["library_corr_lags"]
    # for idx_snr in tqdm(
    #     range(xr_dataset.sizes["snr"]),
    #     bar_format=BAR_FORMAT,
    #     desc="Receiver pair cross-correlation computation",
    # ):
    #     library_corr = np.empty(
    #         tuple(xr_dataset.sizes[d] for d in library_corr_dim), dtype=np.float32
    #     )

    #     for i_pair in range(xr_dataset.sizes["idx_rcv_pairs"]):

    #         rcv_pair = xr_dataset.rcv_pairs.isel(idx_rcv_pairs=i_pair)
    #         in1 = (
    #             xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[0])
    #             .isel(snr=idx_snr)
    #             .load()
    #             .values
    #         )
    #         in2 = (
    #             xr_dataset.rcv_signal_library.sel(idx_rcv=rcv_pair[1])
    #             .isel(snr=idx_snr)
    #             .load()
    #             .values
    #         )

    #         nfft = sp_fft.next_fast_len(nlag, True)

    #         sig_0 = sp_fft.rfft(
    #             in1,
    #             n=nfft,
    #             axis=-1,
    #         )
    #         sig_1 = sp_fft.rfft(
    #             in2,
    #             n=nfft,
    #             axis=-1,
    #         )

    #         corr_01_fft = fft_convolve_f(sig_0, sig_1, axis=ax, workers=-1)
    #         corr_01_fft = corr_01_fft[:, :, slice(nlag)]

    #         autocorr0 = fft_convolve_f(sig_0, sig_0, axis=ax, workers=-1)
    #         autocorr0 = autocorr0[:, :, slice(nlag)]

    #         autocorr1 = fft_convolve_f(sig_1, sig_1, axis=ax, workers=-1)
    #         autocorr1 = autocorr1[:, :, slice(nlag)]

    #         n0 = corr_01_fft.shape[-1] // 2
    #         corr_norm = np.sqrt(autocorr0[..., n0] * autocorr1[..., n0])
    #         corr_norm = np.repeat(np.expand_dims(corr_norm, axis=ax), nlag, axis=ax)
    #         corr_01_fft /= corr_norm
    #         # library_corr[idx_snr, i_pair, ...] = corr_01_fft
    #         library_corr[i_pair, ...] = corr_01_fft

    #     # Assign to dataset for each snr
    #     xr_dataset["library_corr"][dict(snr=idx_snr)] = library_corr.astype(np.float32)

    # if init_corr:
    # library_corr_dim += ["snr"]
    # dummy_array = np.empty(tuple(xr_dataset.sizes[d] for d in library_corr_dim))
    # xr_dataset["library_corr"] = (library_corr_dim, library_corr.astype(np.float32))
    # xr_dataset["library_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"

    # xr_dataset.library_corr.loc[dict(snr=snr_dB)] = library_corr.astype(np.float32)
    # return xr_dataset


def add_noise_to_event(xr_dataset, snr_dB):
    """
    Add noise to event signal.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    snr_dB : float
        Signal to noise ratio.

    Returns
    -------
    None

    """

    # ds = library_dataset
    for i_rcv in tqdm(
        xr_dataset.idx_rcv, bar_format=BAR_FORMAT, desc="Add noise to event signal"
    ):
        rcv_sig = xr_dataset.rcv_signal_event_noise_free.sel(idx_rcv=i_rcv).values
        if snr_dB is not None:
            # Add noise to received signal
            xr_dataset.rcv_signal_event.loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
                np.copy(rcv_sig),
                snr_dB,
            )
            # xr_dataset.attrs["snr_dB"] = snr_dB
        else:
            # xr_dataset.attrs["snr_dB"] = "Noiseless"
            pass  # TODO: need to be updated to fit with snr integration in a single dataset

    # # ds = library_dataset
    # for i_rcv in tqdm(
    #     xr_dataset.idx_rcv, bar_format=BAR_FORMAT, desc="Add noise to event signal"
    # ):
    #     rcv_sig = (
    #         xr_dataset.rcv_signal_event_noise_free.sel(idx_rcv=i_rcv).load().values
    #     )
    #     if snr_dB is not None:
    #         # Add noise to received signal
    #         xr_dataset.rcv_signal_event.loc[dict(idx_rcv=i_rcv, snr=snr_dB)] = (
    #             add_noise_to_signal(
    #                 np.copy(rcv_sig),
    #                 snr_dB,
    #             )
    #         )
    #         # xr_dataset.attrs["snr_dB"] = snr_dB
    #     else:
    #         # xr_dataset.attrs["snr_dB"] = "Noiseless"
    #         pass  # TODO: need to be updated to fit with snr integration in a single dataset


def add_event_correlation(xr_dataset):
    """
    Derive correlation for each source position.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.

    Returns
    -------
    None

    """

    # ds = library_dataset
    # lags_idx = signal.correlation_lags(
    #     xr_dataset.sizes["event_signal_time"], xr_dataset.sizes["event_signal_time"]
    # )
    # lags = lags_idx * xr_dataset.event_signal_time.diff("event_signal_time").values[0]

    # xr_dataset.coords["event_corr_lags"] = lags
    # xr_dataset["event_corr_lags"].attrs["units"] = "s"
    # xr_dataset["event_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each ship position
    event_corr_dim = ["idx_rcv_pairs", "src_trajectory_time", "event_corr_lags"]
    event_corr = np.empty(tuple(xr_dataset.sizes[d] for d in event_corr_dim))

    for i_ship in tqdm(
        range(xr_dataset.sizes["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Derive correlation vector for each ship position",
    ):
        for i_pair, rcv_pair in enumerate(xr_dataset.rcv_pairs):
            s0 = (
                xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[0])
                .isel(src_trajectory_time=i_ship)
                .values
            )
            s1 = (
                xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[1])
                .isel(src_trajectory_time=i_ship)
                .values
            )

            corr_01 = signal.correlate(s0, s1)
            autocorr0 = signal.correlate(s0, s0)
            autocorr1 = signal.correlate(s1, s1)
            n0 = corr_01.shape[0] // 2
            corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

            event_corr[i_pair, i_ship, :] = corr_01

    xr_dataset["event_corr"].values = event_corr.astype(np.float32)

    # # Derive cross_correlation vector for each ship position
    # event_corr_dim = ["idx_rcv_pairs", "src_trajectory_time", "event_corr_lags"]
    # event_corr = np.empty(tuple(xr_dataset.sizes[d] for d in event_corr_dim))

    # for i_ship in tqdm(
    #     range(xr_dataset.sizes["src_trajectory_time"]),
    #     bar_format=BAR_FORMAT,
    #     desc="Derive correlation vector for each ship position",
    # ):
    #     for i_pair, rcv_pair in enumerate(xr_dataset.rcv_pairs):
    #         s0 = (
    #             xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[0], snr=snr_dB)
    #             .isel(src_trajectory_time=i_ship)
    #             .load()
    #             .values
    #         )
    #         s1 = (
    #             xr_dataset.rcv_signal_event.sel(idx_rcv=rcv_pair[1], snr=snr_dB)
    #             .isel(src_trajectory_time=i_ship)
    #             .load()
    #             .values
    #         )

    #         corr_01 = signal.correlate(s0, s1)
    #         autocorr0 = signal.correlate(s0, s0)
    #         autocorr1 = signal.correlate(s1, s1)
    #         n0 = corr_01.shape[0] // 2
    #         corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

    #         event_corr[i_pair, i_ship, :] = corr_01

    #         # del s0, s1, corr_01

    # # if init_corr:
    # #     event_corr_dim += ["snr"]
    # #     dummy_array = np.zeros(tuple(xr_dataset.sizes[d] for d in event_corr_dim))
    # #     xr_dataset["event_corr"] = (event_corr_dim, dummy_array)
    # #     xr_dataset["event_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"

    # xr_dataset.event_corr.loc[dict(snr=snr_dB)] = event_corr.astype(np.float32)

    # xr_dataset["event_corr"] = (event_corr_dim, event_corr.astype(np.float32))


def add_noise_to_signal(sig, snr_dB, noise_type="gaussian"):
    """
    Add noise to signal assuming sig is either a 2D (event signal (pos_idx, t)) or a 3D (library signal (x, y, t)) array.
    The noise level is adjusted to garantee the desired SNR at all positions.

    Parameters
    ----------
    sig : np.ndarray
        Signal.
    snr_dB : float
        Signal to noise ratio.
    noise_type : str, optional
        Noise type. The default is "gaussian".

    Returns
    -------
    np.ndarray
        Signal with added noise.

    """

    if snr_dB is not None:
        # The further the grid point is from the source, the lower the signal power is and the lower the noise level should be
        P_sig = (
            1 / sig.shape[-1] * np.sum(sig**2, axis=-1)
        )  # Signal power for each position (last dimension is assumed to be time)
        sigma_noise = np.sqrt(
            P_sig * 10 ** (-snr_dB / 10)
        )  # Noise level for each position

        # print(f"Desired SNR: {snr_dB} dB")
        # print("Signal power at first pos : ", P_sig[0])
        # print("Noise level at first pos : ", sigma_noise[0])
        # print(P_sig, sigma_noise)

        if noise_type == "gaussian":
            # Generate 3D Gaussian noise matrix with corresponding sigma in each grid position
            # Broadcasting sigma_noise to match the shape of sig for element-wise multiplication
            noise = np.random.normal(0, 1, sig.shape) * sigma_noise[..., np.newaxis]

        else:
            raise ValueError("Noise type not supported")

        # Add the noise to the original signal
        noisy_sig = sig + noise

    return noisy_sig


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
        (1, len(src_traj_times)) + tuple(lib_data.sizes[d] for d in ["lat", "lon"])
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

    lib_data_array = lib_data.values

    for i_src_time, src_time in enumerate(src_traj_times):

        event_vector = event_data.sel(src_trajectory_time=src_time)
        event_vector_array = event_vector.values

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

    return da_amb_surf


def build_ambiguity_surf(xr_dataset, idx_similarity_metric, i_noise, verbose=True):
    """
    Build ambiguity surface for each receiver pair and source position.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    snr_dB : float
        Signal to noise ratio (dB).
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

    similarity_metric = xr_dataset.similarity_metric.sel(
        idx_similarity_metric=idx_similarity_metric
    ).values
    if verbose:
        det_msg = (
            f"Building ambiguity surface using {similarity_metric} as similarity metric"
        )
        print("# " + det_msg + " #")

    t0 = time.time()

    # Init ambiguity surface
    # Ambiguity surface is not store for each noise realization to avoid memory issues (only detected positions are stored)
    ambiguity_surface_dim = [
        "idx_similarity_metric",
        "idx_rcv_pairs",
        "src_trajectory_time",
        "lat",
        "lon",
    ]
    ambiguity_surface = np.empty(
        tuple(xr_dataset.sizes[d] for d in ambiguity_surface_dim)
    )
    ambiguity_surface_sim = ambiguity_surface[0, ...]

    parallel = False

    ambiguity_surfaces = []
    for i_pair in xr_dataset.idx_rcv_pairs:
        lib_data = xr_dataset.library_corr.sel(idx_rcv_pairs=i_pair)
        event_data = xr_dataset.event_corr.sel(idx_rcv_pairs=i_pair)

        if parallel:
            """ Parallel processing"""
            # Store all ambiguity surfaces in a list (for parrallel processing)
            # TODO: swicht between parallel and sequential processing depending on the number of ship positions
            # Init pool
            pool = multiprocessing.Pool(processes=N_CORES)
            # Build the parameter pool
            idx_ship_intervalls = np.array_split(
                np.arange(xr_dataset.sizes["src_trajectory_time"]), N_CORES
            )
            src_traj_time_intervalls = np.array_split(
                xr_dataset["src_trajectory_time"], N_CORES
            )
            param_pool = [
                (lib_data, event_data, src_traj_time_intervalls[i], similarity_metric)
                for i in range(len(idx_ship_intervalls))
            ]
            # Run parallel processes
            results = pool.starmap(derive_ambiguity, param_pool)
            # Close pool
            pool.close()
            # Wait for all processes to finish
            pool.join()

            ambiguity_surfaces += results
            print(f"Elapsed time (parallel) : {time.time() - t0}")

        else:
            """ Sequential processing"""
            # t0 = time.time()
            amb_surf_da = derive_ambiguity(lib_data, event_data, xr_dataset.src_trajectory_time, similarity_metric)
            ambiguity_surfaces.append(amb_surf_da)

        # lib_data_array = lib_data.values

        # for i_ship in tqdm(
        #     range(xr_dataset.sizes["src_trajectory_time"]),
        #     bar_format=BAR_FORMAT,
        #     desc="Build ambiguity surface",
        # ):
        #     event_vector = event_data.isel(src_trajectory_time=i_ship)
        #     event_vector_array = event_vector.values

        #     if similarity_metric == "intercorr0":
        #         amb_surf = mult_along_axis(
        #             lib_data_array,
        #             event_vector_array,
        #             axis=2,
        #         )
        #         autocorr_lib_0 = np.sum(lib_data_array**2, axis=2)
        #         autocorr_event_0 = np.sum(event_vector_array**2)

        #         norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
        #         amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
        #         amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
        #         ambiguity_surface_sim[i_pair, i_ship, ...] = amb_surf

        #     elif similarity_metric == "lstsquares":

        #         diff = lib_data_array - event_vector_array
        #         amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
        #         amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
        #         amb_surf = (
        #             1 - amb_surf
        #         )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
        #         ambiguity_surface[i_pair, i_ship, ...] = amb_surf

        #     elif similarity_metric == "hilbert_env_intercorr0":
        #         lib_env = np.abs(signal.hilbert(lib_data_array))
        #         event_env = np.abs(signal.hilbert(event_vector_array))

        #         amb_surf = mult_along_axis(
        #             lib_env,
        #             event_env,
        #             axis=2,
        #         )

        #         autocorr_lib_0 = np.sum(lib_env**2, axis=2)
        #         autocorr_event_0 = np.sum(event_env**2)

        #         norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
        #         amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
        #         amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
        #         ambiguity_surface_sim[i_pair, i_ship, ...] = amb_surf

    # # Store ambiguity surface in dataset
    # xr_dataset.ambiguity_surface.loc[
    #     dict(idx_similarity_metric=idx_similarity_metric)
    # ] = ambiguity_surface_sim

    # xr_dataset["ambiguity_surface"] = (
    #     ambiguity_surface_dim,
    #     ambiguity_surface,
    # )

    # Merge dataarrays
    amb_surf_merged = xr.merge(ambiguity_surfaces)
    xr_dataset.ambiguity_surface.loc[
        dict(idx_similarity_metric=idx_similarity_metric)
    ] = amb_surf_merged["ambiguity_surface"]

    # amb_surf_combined = amb_surf_merged.ambiguity_surface.mean(dim=["idx_rcv_pairs"])
    amb_surf_combined = amb_surf_merged.ambiguity_surface.prod(dim="idx_rcv_pairs") ** (1 / len(xr_dataset.idx_rcv_pairs))
    xr_dataset.ambiguity_surface_combined.loc[
        dict(idx_similarity_metric=idx_similarity_metric)
    ] = amb_surf_combined
    # Analyse ambiguity surface to detect source position
    get_detected_pos(xr_dataset, idx_similarity_metric, i_noise)


def init_ambiguity_surface(xr_dataset):

    ambiguity_surface_dim = [
        "idx_rcv_pairs",
        "idx_similarity_metric",
        "src_trajectory_time",
        "lat",
        "lon",
    ]
    ambiguity_surface = np.empty(
        tuple(xr_dataset.sizes[d] for d in ambiguity_surface_dim)
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
    xr_dataset.ambiguity_surface_combined.attrs["long_name"] = "Ambiguity surface combined"
    xr_dataset.ambiguity_surface_combined.attrs["units"] = "dB"


    # # Expand dims to add snr dimension to the rcv_signal_library dataarray
    # # Using expand_dims to avoid loading the entire dataset in memory -> overflow
    # xr_dataset["ambiguity_surface"] = (
    #     xr_dataset.ambiguity_surface.expand_dims(dim={"snr": xr_dataset.sizes["snr"]})
    #     .assign_coords({"snr": xr_dataset.snr})
    #     .copy()
    # )

    # Create detected position dataarrays
    detected_pos_dim = [
        "idx_rcv_pairs",
        "src_trajectory_time",
        "idx_noise_realisation",
        "idx_similarity_metric",
    ]

    # Weird behavior of xarray : if detected_pos_lon and detected_pos_lat are initialized with the same np array (e.g. np.empty),
    # the two dataarrays are linked and the values of detected_pos_lon are updated when detected_pos_lat is updated
    detected_pos_init_lon = np.empty(
        tuple(xr_dataset.sizes[d] for d in detected_pos_dim)
    )
    detected_pos_init_lat = np.empty(
        tuple(xr_dataset.sizes[d] for d in detected_pos_dim)
    )

    xr_dataset["detected_pos_lon"] = (detected_pos_dim, detected_pos_init_lon)
    xr_dataset["detected_pos_lat"] = (detected_pos_dim, detected_pos_init_lat)
    xr_dataset["detected_pos_lon_combined"] = (detected_pos_dim[1:], detected_pos_init_lon[0, ...])
    xr_dataset["detected_pos_lat_combined"] = (detected_pos_dim[1:], detected_pos_init_lat[0, ...])

    # # Expand dims to add snr dimension to the rcv_signal_library dataarray
    # # Using expand_dims to avoid loading the entire dataset in memory -> overflow
    # xr_dataset["detected_pos_lon"] = (
    #     xr_dataset.detected_pos_lon.expand_dims(dim={"snr": xr_dataset.sizes["snr"]})
    #     .assign_coords({"snr": xr_dataset.snr})
    #     .copy()
    # )
    # xr_dataset["detected_pos_lat"] = (
    #     xr_dataset.detected_pos_lat.expand_dims(dim={"snr": xr_dataset.sizes["snr"]})
    #     .assign_coords({"snr": xr_dataset.snr})
    #     .copy()
    # )

    # # Save and reload to avoid read-only issues with zarr
    # xr_dataset.to_zarr(xr_dataset.fullpath_output, mode="w")
    # xr_dataset.close()
    # # Open mutable zarr dataset
    # xr_dataset = xr.open_zarr(xr_dataset.fullpath_output)

    # return xr_dataset


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
    ambiguity_surface = xr_dataset.ambiguity_surface.isel(
        idx_similarity_metric=idx_similarity_metric
    )
    ambiguity_surface_combined = xr_dataset.ambiguity_surface_combined.isel(
        idx_similarity_metric=idx_similarity_metric
    )

    if method == "absmax":
        max_pos_idx = ambiguity_surface.argmax(dim=["lon", "lat"])
        ilon_detected = max_pos_idx["lon"]  # Index of detected longitude
        ilat_detected = max_pos_idx["lat"]  # Index of detected longitude

        detected_lon = xr_dataset.lon.isel(lon=ilon_detected).values
        detected_lat = xr_dataset.lat.isel(lat=ilat_detected).values

        max_pos_combined_idx = ambiguity_surface_combined.argmax(dim=["lon", "lat"])
        ilon_detected_combined = max_pos_combined_idx["lon"]  # Index of detected longitude
        ilat_detected_combined = max_pos_combined_idx["lat"]  # Index of detected longitude

        detected_lon_combined = xr_dataset.lon.isel(lon=ilon_detected_combined).values
        detected_lat_combined = xr_dataset.lat.isel(lat=ilat_detected_combined).values

    # TODO : add other methods to take a larger number of values into account
    else:
        raise ValueError("Method not supported")

    # Store detected position in dataset
    dict_sel = dict(
        idx_noise_realisation=i_noise,
        idx_similarity_metric=idx_similarity_metric,
    )
    # ! need to use loc when assigning values to a DataArray to avoid silent failing !
    xr_dataset.detected_pos_lon.loc[dict_sel] = detected_lon
    xr_dataset.detected_pos_lat.loc[dict_sel] = detected_lat
    xr_dataset.detected_pos_lon_combined.loc[dict_sel] = detected_lon_combined
    xr_dataset.detected_pos_lat_combined.loc[dict_sel] = detected_lat_combined


def init_src(dt, min_waveguide_depth, src_info):
    """
    Initialize source signal.

    Parameters
    ----------
    dt : float
        Time step.
    min_waveguide_depth : float
        Minimum waveguide depth.
    sig_type : str, optional
        Signal type. The default is "pulse".

    Returns
    -------
    AcousticSource
        Acoustic source.

    """
    sig_type = src_info["sig_type"]

    nfft = None
    if sig_type == "ship":
        # library_src_sig, t_library_src_sig = ship_noise(T=dt)
        src_sig, t_src_sig = generate_ship_signal(
            Ttot=dt,
            f0=src_info["f0"],
            std_fi=src_info["std_fi"],
            tau_corr_fi=src_info["tau_corr_fi"],
            fs=src_info["fs"],
        )

    elif sig_type == "pulse":
        src_sig, t_src_sig = pulse(T=dt, f=src_info["f0"], fs=src_info["fs"])

    elif sig_type == "pulse_train":
        src_sig, t_src_sig = pulse_train(T=dt, f=src_info["f0"], fs=src_info["fs"])

    elif sig_type == "debug_pulse":
        src_info["f0"], src_info["fs"] = 10, 40
        src_sig, t_src_sig = pulse(T=dt, f=src_info["f0"], fs=src_info["fs"])
        nfft = int(src_info["fs"] * dt)

    if sig_type in ["ship", "pulse_train"]:
        # Apply hanning window
        src_sig *= np.hanning(len(src_sig))

    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name=sig_type,
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    return src


def plot_src(src, testcase, usage="library"):
    """
    Plot source signal.

    Parameters
    ----------
    library_src : AcousticSource
        Acoustic source.
    testcase : Testcase
        Testcase.

    Returns
    -------
    None

    """
    pfig = PubFigure()
    src.display_source(plot_spectrum=False)
    fig = plt.gcf()
    fig.set_size_inches(pfig.size)
    axs = fig.axes
    axs[0].set_title("")
    axs[1].set_title("")
    fig.suptitle("Source signal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(testcase.env_dir, "env_desc_img", f"{usage}_src_{testcase.name}.png")
    )
    plt.close()


def init_event_src_traj(src_info, dt):
    """
    Initialize the source trajectory.

    Parameters
    ----------
    src_info : dict
        Source information.
    dt : float
        Time step.

    Returns
    -------
    None

    """
    # Save time info
    src_info["time"] = np.arange(0, src_info["max_nb_of_pos"] * dt, dt)
    src_info["dt"] = dt

    Dtot = src_info["speed"] * src_info["duration"]

    # Define the geodetic object
    geod = Geod(ellps="WGS84")
    # Determine longitude and latitude of terminus point
    lat_i, lon_i = src_info["initial_pos"]["lat"], src_info["initial_pos"]["lon"]
    lon_f, lat_f, back_az = geod.fwd(
        lons=lon_i,
        lats=lat_i,
        az=src_info["route_azimuth"],
        dist=Dtot,
        return_back_azimuth=True,
    )

    # Determine coordinates along trajectory
    traj = geod.inv_intermediate(
        lat1=lat_i, lon1=lon_i, lat2=lat_f, lon2=lon_f, npts=src_info["max_nb_of_pos"]
    )

    src_info["lons"] = np.array(traj.lons)
    src_info["lats"] = np.array(traj.lats)
    src_info["n_pos"] = len(src_info["lons"])


def init_grid_around_event_src_traj(src_info, grid_info):
    """
    Initialize the grid around the source trajectory.

    Parameters
    ----------
    src_info : dict
        Source information.
    grid_info : dict
        Grid information.

    Returns
    -------
    None

    """
    min_lon, max_lon = np.min(src_info["lons"]), np.max(src_info["lons"])
    min_lat, max_lat = np.min(src_info["lats"]), np.max(src_info["lats"])
    mean_lon, mean_lat = np.mean(src_info["lons"]), np.mean(src_info["lats"])

    offset_lon = max(2, grid_info["offset_cells_lon"]) * grid_info["dx"]
    offset_lat = max(2, grid_info["offset_cells_lat"]) * grid_info["dy"]

    geod = Geod(ellps="WGS84")
    min_lon_grid, _, _ = geod.fwd(
        lons=min_lon,
        lats=mean_lat,
        az=270,
        dist=offset_lon,
        return_back_azimuth=True,
    )
    max_lon_grid, _, _ = geod.fwd(
        lons=max_lon,
        lats=mean_lat,
        az=90,
        dist=offset_lon,
        return_back_azimuth=True,
    )

    _, min_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=min_lat,
        az=180,
        dist=offset_lat,
        return_back_azimuth=True,
    )
    _, max_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=max_lat,
        az=0,
        dist=offset_lat,
        return_back_azimuth=True,
    )

    grid_lons = np.array(
        geod.inv_intermediate(
            lat1=mean_lat,
            lon1=min_lon_grid,
            lat2=mean_lat,
            lon2=max_lon_grid,
            del_s=grid_info["dx"],
            return_back_azimuth=True,
        ).lons
    )
    grid_lats = np.array(
        geod.inv_intermediate(
            lat1=min_lat_grid,
            lon1=mean_lon,
            lat2=max_lat_grid,
            lon2=mean_lon,
            del_s=grid_info["dy"],
            return_back_azimuth=True,
        ).lats
    )
    grid_info["lons"] = grid_lons
    grid_info["lats"] = grid_lats
    grid_info["min_lat"] = np.min(grid_lats)
    grid_info["max_lat"] = np.max(grid_lats)
    grid_info["min_lon"] = np.min(grid_lons)
    grid_info["max_lon"] = np.max(grid_lons)


def get_max_kraken_range(rcv_info, grid_info):
    """
    Derive maximum range to be covered by KRAKEN for each receiver.

    Parameters
    ----------
    rcv_info : dict
        Receiver information.
    grid_info : dict
        Grid information.

    Returns
    -------
    None

    """
    geod = Geod(ellps="WGS84")
    max_r = []

    for i, id in enumerate(rcv_info["id"]):
        # Derive distance to the 4 corners of the grid
        _, _, ranges = geod.inv(
            lons1=[rcv_info["lons"][i]] * 4,
            lats1=[rcv_info["lats"][i]] * 4,
            lons2=[
                grid_info["min_lon"],
                grid_info["min_lon"],
                grid_info["max_lon"],
                grid_info["max_lon"],
            ],
            lats2=[
                grid_info["min_lat"],
                grid_info["max_lat"],
                grid_info["max_lat"],
                grid_info["min_lat"],
            ],
            return_back_azimuth=True,
        )

        max_r.append(np.max(ranges))

    rcv_info["max_kraken_range_m"] = np.round(max_r, -2)


def get_bathy_grid_size(lon, lat):
    lat_rad = np.radians(lat)  # Latitude en radians
    lon_rad = np.radians(lon)  # Longitude en radians

    grid_size = 15 / 3600 * np.pi / 180  # 15" (secondes d'arc)
    lat_0 = lat_rad - grid_size
    lat_1 = lat_rad + grid_size
    lon_0 = lon_rad - grid_size
    lon_1 = lon_rad + grid_size

    geod = Geod(ellps="WGS84")
    _, _, dlat = geod.inv(
        lons1=lon,
        lats1=np.degrees(lat_0),
        lons2=lon,
        lats2=np.degrees(lat_1),
        return_back_azimuth=True,
    )
    _, _, dlon = geod.inv(
        lons1=np.degrees(lon_0),
        lats1=lat,
        lons2=np.degrees(lon_1),
        lats2=lat,
        return_back_azimuth=True,
    )

    return dlon, dlat

def get_dist_between_rcv(rcv_info):
    """
    Derive distance between receivers.

    Parameters
    ----------
    rcv_info : dict
        Receiver information.

    Returns
    -------
    None

    """
    geod = Geod(ellps="WGS84")

    dist_inter_rcv = []

    rcv_pairs = []
    for i in range((len(rcv_info["id"]))):
        for j in range(i + 1, len(rcv_info["id"])):
            rcv_pairs.append((i, j))

    for i_pair, pair in enumerate(rcv_pairs):
        _, _, dist = geod.inv(
            lons1=[rcv_info["lons"][pair[0]]],
            lats1=[rcv_info["lats"][pair[0]]],
            lons2=[rcv_info["lons"][pair[1]]],
            lats2=[rcv_info["lats"][pair[1]]],
            return_back_azimuth=True,
        )
        dist_inter_rcv.append(np.round(dist, 0))

    rcv_info["dist_inter_rcv"] = dist_inter_rcv


def load_rhumrum_obs_pos(obs_id):
    """
    Load RHUM-RUM OBS position.

    Parameters
    ----------
    obs_id : int
        OBS ID.

    Returns
    -------
    pd.Series
        OBS position.

    """
    pos_path = os.path.join(PROJECT_ROOT, "data", "rhum_rum_obs_pos.csv")

    pos = pd.read_csv(
        pos_path,
        index_col="id",
        delimiter=",",
    )
    return pos.loc[obs_id]


def print_simulation_info(testcase, src_info, rcv_info, grid_info):
    """
    Print simulation information.

    Parameters
    ----------
    testcase : Testcase
        Testcase.
    src_info : dict
        Source information.
    rcv_info : dict
        Receiver information.
    grid_info : dict
        Grid information.

    Returns
    -------
    None

    """
    balises = "".join(["#"] * 80)
    balises_inter = "".join(["#"] * 40)
    header_msg = f"Start simulation - {testcase.name}: \n{testcase.desc}"
    src_pos = "\n\t\t".join(
        [""]
        + [
            f"{pos_s}: (lon, lat) = ({lon:.4f}°, {lat:.4f}°)"
            for pos_s, lon, lat in zip(
                ["First", "Last"],
                [src_info["lons"][0], src_info["lons"][-1]],
                [src_info["lats"][0], src_info["lats"][-1]],
            )
        ]
    )
    src_msg = [
        "Source properties:",
        f"Positions: {src_pos}",
        f"Depth: {src_info['depth']} m",
        f"Speed: {src_info['speed']:.2f} m/s",
        f"Azimuth: {src_info['route_azimuth']}°",
        f"Route duration: {src_info['duration'] / 60} min",
        f"Number of positions: {len(src_info['lons'])}",
    ]
    src_msg = "\n\t".join(src_msg)

    # Rcv info
    rcv_pos = "\n\t\t".join(
        [""]
        + [
            f"Receiver {i}: (lon, lat) = ({lon}°, {lat}°)"
            for i, lon, lat in zip(rcv_info["id"], rcv_info["lons"], rcv_info["lats"])
        ]
        + [
            f"Distance between receivers of pair n°{i_pair}: {rcv_info['dist_inter_rcv'][i_pair]} m"
            for i_pair in range(len(rcv_info["dist_inter_rcv"]))
        ]
    )
    max_range = "\n\t\t".join(
        [""]
        + [
            f"Receiver {i}: {r} m"
            for i, r in zip(rcv_info["id"], rcv_info["max_kraken_range_m"])
        ]
    )
    rcv_msg = [
        "Receivers properties:",
        f"Number of receivers: {len(rcv_info['id'])}",
        f"Receivers IDs: {rcv_info['id']}",
        f"Receivers positions: {rcv_pos}",
        f"Maximum range to be covered by KRAKEN: {max_range}",
    ]
    rcv_msg = "\n\t".join(rcv_msg)

    # Grid info
    grid_res = "\n\t\t".join(
        [""] + [f"dx = {grid_info['dx']} m", f"dy = {grid_info['dy']} m"]
    )
    grid_msg = [
        "Grid properties:",
        f"Grid resolution: {grid_res}",
        f"Grid cells offset around src trajectory: {grid_info['offset_cells_lon']} in lon and {grid_info['offset_cells_lat']} in lat",
        f"Number of grid points: {len(grid_info['lons']) * len(grid_info['lats'])}",
    ]
    grid_msg = "\n\t".join(grid_msg)

    msg = "\n".join(
        [
            balises,
            header_msg,
            balises_inter,
            src_msg,
            balises_inter,
            rcv_msg,
            balises_inter,
            grid_msg,
            balises,
        ]
    )
    print(msg)


def get_populated_path(grid_info, kraken_env, src_signal_type, init_time, ext="nc"):
    """
    Get fullpath to populated dataset.

    Parameters
    ----------
    grid_info : dict
        Grid information.
    kraken_env : KrakenEnv
        Kraken environment.
    src_signal_type : str
        Source signal type.
    ext : str, optional
        File extension. The default is "nc".

    Returns
    -------
    str
        Fullpath to populated dataset.

    """
    boundaries = "_".join(
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
    populated_path = os.path.join(
        VERLINDEN_POPULATED_FOLDER,
        kraken_env.filename,
        init_time,
        f"populated_{src_signal_type}.{ext}",
    )
    return populated_path


def build_output_save_path(
    xr_dataset,
    output_folder,
    analysis_folder,
    testcase_name,
    src_name,
    snr_tag,
    ext="nc",
):
    """
    Build path to save dataset and related figures

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    output_folder : str
        Output folder.
    analysis_folder : str
        Analysis folder.
    testcase_name : str
        Testcase name.
    src_name : str
        Source name.

    Returns
    -------
    None

    """
    # Build path to save dataset and corresponding path to save analysis results produced later on
    # Check if fullpath_output and fullpath_analysis are already in xr_dataset.attrs
    if not "fullpath_output" in xr_dataset.attrs:
        xr_dataset.attrs["fullpath_output"] = os.path.join(
            output_folder,
            testcase_name,
            src_name,
            xr_dataset.src_pos,
            xr_dataset.attrs["init_time"],
            f"output_{testcase_name}_{snr_tag}.{ext}",
        )
    if not "fullpath_analysis" in xr_dataset.attrs:
        xr_dataset.attrs["fullpath_analysis"] = os.path.join(
            analysis_folder,
            testcase_name,
            src_name,
            xr_dataset.src_pos,
            xr_dataset.attrs["init_time"],
            # snr_tag,
        )

    # Ensure that the output folder exists
    if not os.path.exists(os.path.dirname(xr_dataset.fullpath_output)):
        os.makedirs(os.path.dirname(xr_dataset.fullpath_output))

    if not os.path.exists(xr_dataset.fullpath_analysis):
        os.makedirs(xr_dataset.fullpath_analysis)

    # # Save dataset to netcdf
    # # xr_dataset.to_netcdf(xr_dataset.fullpath_output)
    # xr_dataset.to_zarr(xr_dataset.fullpath_output, mode="r+")


def get_snr_tag(snr_dB, verbose=True):
    """
    Get SNR tag.

    Parameters
    ----------
    snr_dB : float
        Signal to noise ratio.
    verbose : bool, optional
        Verbose. The default is True.

    Returns
    -------
    str
        SNR tag.

    """
    if snr_dB is None:
        snr_tag = "noiseless"
        snr_msg = "Performing localisation process without noise"
    else:
        snr_tag = f"snr{snr_dB}dB"
        snr_msg = f"Performing localisation process with additive gaussian white noise SNR = {snr_dB}dB"

    if verbose:
        print("## " + snr_msg + " ##")

    return snr_tag


def add_src_to_dataset(xr_dataset, library_src, event_src, src_info):
    """
    Add source information to dataset.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Populated dataset.
    library_src : AcousticSource
        Library source.
    event_src : AcousticSource
        Event source.
    src_info : dict
        Source information.

    Returns
    -------
    None

    """

    xr_dataset.coords["library_src_time"] = library_src.time
    xr_dataset["library_src_time"].attrs["units"] = "s"
    xr_dataset["library_src_time"].attrs["long_name"] = "Time"

    xr_dataset.coords["event_src_time"] = event_src.time
    xr_dataset["event_src_time"].attrs["units"] = "s"
    xr_dataset["event_src_time"].attrs["long_name"] = "Time"

    xr_dataset["library_src"] = (["library_src_time"], library_src.signal)
    xr_dataset["library_src"].attrs["long_name"] = "Library source signal"
    xr_dataset["library_src"].attrs["short_name"] = "Library"
    xr_dataset["event_src"] = (["event_src_time"], event_src.signal)
    xr_dataset["event_src"].attrs["long_name"] = "Event source signal"
    xr_dataset["event_src"].attrs["short_name"] = "Event"

    param = src_info["library"].keys()
    for p in param:
        xr_dataset["library_src"].attrs[p] = src_info["library"][p]

    param = src_info["event"].keys()
    for p in param:
        xr_dataset["event_src"].attrs[p] = src_info["event"][p]

def merge_file_path(output_dir, testcase_name, ext="nc"):
    merge_fpath = os.path.join(output_dir, f"output_{testcase_name}_snrs.{ext}")
    return merge_fpath


def merge_results(output_dir, testcase_name, snr, ext="nc"):
    if ext == "zarr":
        engine="zarr"
    else:
        engine="netcdf4"

    merged_fpath = merge_file_path(output_dir, testcase_name, ext=ext)
    if not os.path.exists(merged_fpath):
        snr_filepaths_to_merge = [
            os.path.join(
                output_dir,
                f"output_{testcase_name}_{get_snr_tag(snri, verbose=False)}.{ext}",
            )
            for snri in snr
        ]
        # Open snr file and save it as snrs file
        with xr.open_mfdataset(
            snr_filepaths_to_merge, combine="nested", concat_dim="snr", engine=engine
        ) as merged_ds:

            needed_vars = [
                "rcv_signal_event",
                "rcv_signal_library",
                "event_corr",
                "library_corr",
                "ambiguity_surface",
                "ambiguity_surface_combined"
                "detected_pos_lon",
                "detected_pos_lat",
                "detected_pos_lon_combined",
                "detected_pos_lat_combined",
            ]  # Var that depend on snr
            non_needed_vars = [
                var for var in list(merged_ds.keys()) if var not in needed_vars
            ]  # Var that do not depend on snr
            for var in non_needed_vars:
                merged_ds[var] = merged_ds[var].isel(snr=0, drop=True)

        if ext == "zarr":
            merged_ds.to_zarr(merged_fpath, mode="w")
        else:
            merged_ds.to_netcdf(merged_fpath)

    else:
        with xr.open_dataset(merged_fpath) as merged_ds:
            # Add new snr to the existing snrs file
            snr_filepaths_to_merge = [
                os.path.join(
                    output_dir,
                    f"output_{testcase_name}_{get_snr_tag(snri, verbose=False)}.{ext}",
                )
                for snri in snr
                if not snri in merged_ds.snr.values
            ]

            # If list is not empty
            if snr_filepaths_to_merge:
                with xr.open_mfdataset(
                    snr_filepaths_to_merge, combine="nested", concat_dim="snr", engine=engine
                ) as ds:

                    merged_ds = xr.concat([merged_ds, ds], dim="snr")
                    needed_vars = [
                        "rcv_signal_event",
                        "rcv_signal_library",
                        "event_corr",
                        "library_corr",
                        "ambiguity_surface",
                        "detected_pos_lon",
                        "detected_pos_lat",
                    ]  # Var that depend on snr
                    non_needed_vars = [
                        var for var in list(ds.keys()) if var not in needed_vars
                    ]  # Var that do not depend on snr
                    for var in non_needed_vars:
                        merged_ds[var] = merged_ds[var].isel(snr=0, drop=True)

        if snr_filepaths_to_merge:
            if ext == "zarr":
                merged_ds.to_zarr(merged_fpath, mode="w")
            else:
                merged_ds.to_netcdf(merged_fpath)

    # List files to remove
    snr_filepaths_to_delete = []
    for path in os.listdir(output_dir):
        if path.startswith(f"output_{testcase_name}_snr") and path.endswith(f".{ext}") and path != f"output_{testcase_name}_snrs.{ext}":
            snr = float(path.split("snr")[-1].split("dB")[0])
            if snr in merged_ds.snr.values:
                fullpath = os.path.join(output_dir, path)
                snr_filepaths_to_delete.append(fullpath)

    merged_ds.close()

    return snr_filepaths_to_delete
# # Remove snr file
#     os.remove(snr_filepaths)

# list_fpath = [os.path.join(output_dir, f"output_{testcase_name}_{get_snr_tag(snr_i, verbose=False)}.nc") for snr_i in snr]
# if os.path.exists(merged_fpath):
#     list_fpath.append(merged_fpath)

# with xr.open_mfdataset(
#     list_fpath,
#     combine="nested",
#     concat_dim="snr",
# ) as merged_ds:
# # merged_ds = xr.open_mfdataset(
# #     list_fpath,
# #     combine="nested",
# #     concat_dim="snr",
# # )

#     needed_vars = [
#         "rcv_signal_event",
#         "rcv_signal_library",
#         "event_corr",
#         "library_corr",
#         "ambiguity_surface",
#         "detected_pos_lon",
#         "detected_pos_lat",
#     ]  # Var that depend on snr
#     non_needed_vars = [
#         var for var in list(merged_ds.keys()) if var not in needed_vars
#     ]  # Var that do not depend on snr
#     for var in non_needed_vars:
#         merged_ds[var] = merged_ds[var].isel(snr=0, drop=True)

# # Save merged dataset
# merged_ds.to_netcdf(merged_fpath)
# merged_ds.close()

# os.remove(
#         os.path.join(
#             output_dir,
#             f"output_{testcase_name}_{get_snr_tag(snr_i, verbose=False)}.nc",
#         )
#     )

# Remove individual snr files
# for snr_i in snr:
#     os.remove(
#         os.path.join(
#             output_dir,
#             f"output_{testcase_name}_{get_snr_tag(snr_i, verbose=False)}.nc",
#         )
#     )


if __name__ == "__main__":
    pass
