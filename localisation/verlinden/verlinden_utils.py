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
import multiprocessing

import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as signal
import scipy.fft as sp_fft
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Geod

# from scipy.sparse import csr_matrix

from cst import BAR_FORMAT, C0, N_CORES
from misc import mult_along_axis
from signals import ship_noise, pulse, pulse_train
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

from localisation.verlinden.verlinden_path import (
    # VERLINDEN_OUTPUT_FOLDER,
    # VERLINDEN_ANALYSIS_FOLDER,
    VERLINDEN_POPULATED_FOLDER,
)


def populate_isotropic_env(ds, library_src, signal_library_dim, testcase):
    # ds, library_src, kraken_env, kraken_flp, signal_library_dim

    delay_to_apply = ds.delay_rcv.min(dim="idx_rcv").values.flatten()

    # Run KRAKEN
    grid_pressure_field, _ = runkraken(
        env=testcase.env,
        flp=testcase.flp,
        frequencies=library_src.kraken_freq,
        parallel=False,
        verbose=False,
    )

    # Loop over receivers
    for i_rcv in tqdm(
        ds.idx_rcv, bar_format=BAR_FORMAT, desc="Populate grid with received signal"
    ):

        rr_from_rcv_flat = ds.r_from_rcv.sel(idx_rcv=i_rcv).values.flatten()

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
            ds["library_signal_time"] = t_rcv.astype(np.float32)
            ds["library_signal_time"].attrs["units"] = "s"
            ds["library_signal_time"].attrs["long_name"] = "Time"

            rcv_signal_library = np.empty(tuple(ds.dims[d] for d in signal_library_dim))

        # Time domain signal
        s_rcv = s_rcv[:, 0, :].T
        s_rcv = s_rcv.reshape(
            ds.dims["lat"], ds.dims["lon"], ds.dims["library_signal_time"]
        )

        rcv_signal_library[i_rcv, :] = s_rcv

        # Free memory
        del s_rcv, rr_from_rcv_flat

    return ds, rcv_signal_library, grid_pressure_field


def populate_anistropic_env(
    ds, library_src, signal_library_dim, testcase, rcv_info, src_info
):

    # Array of receiver indexes
    idx_rcv = ds.idx_rcv.values

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
        azimuths_rcv = get_unique_azimuths(ds, i_rcv)
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
                library_src=library_src,
                dt=src_info["dt"],
                sig_type=src_info["signal_type"],
            )

            # Store minimum waveguide depth to be used when populating env with events
            # min_waveguide_depth = kraken_env.bathy.bathy_depth.min()
            min_waveguide_depth = testcase.min_depth
            az_kraken_min_wg_depth.append(min_waveguide_depth)

            # Get receiver ranges for selected angle
            idx_az = ds.az_propa.sel(idx_rcv=i_rcv).values == az
            rr_from_rcv_az = ds.r_from_rcv.sel(idx_rcv=i_rcv).values[idx_az].flatten()
            # Get received signal for selected angle
            delay_to_apply = ds.delay_rcv.min(dim="idx_rcv").values[idx_az].flatten()

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
            r_offset = 5 * ds.dx
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

            # Store received signal in dataset
            if i_rcv == 0 and i_az == 0:
                ds["library_signal_time"] = t_rcv.astype(np.float32)
                ds["library_signal_time"].attrs["units"] = "s"
                ds["library_signal_time"].attrs["long_name"] = "Time"
                rcv_signal_library = np.zeros(
                    tuple(ds.dims[d] for d in signal_library_dim)
                )

            # Time domain signal
            s_rcv = s_rcv[:, 0, :].T
            rcv_signal_library[i_rcv, idx_az, :] = s_rcv

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
    nr_max = np.max([kraken_range_rcv[i_rcv].size for i_rcv in ds.idx_rcv.values])
    nz_max = np.max([kraken_depth_rcv[i_rcv].size for i_rcv in ds.idx_rcv.values])

    # Store kraken range and depth for rcv
    for i_rcv in ds.idx_rcv.values:
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

    for i_rcv in ds.idx_rcv.values:
        kraken_grid[i_rcv] = {
            "range": kraken_range_rcv[i_rcv, :],
            "depth": kraken_depth_rcv[i_rcv, :],
            "min_waveguide_depth": {
                i_az: kraken_min_wg_depth_rcv[i_rcv][i_az]
                for i_az in range(len(kraken_min_wg_depth_rcv[i_rcv]))
            },
        }

    return ds, rcv_signal_library, grid_pressure_field, kraken_grid


def add_event_isotropic_env(
    ds,
    snr_dB,
    event_src,
    kraken_env,
    signal_event_dim,
    grid_pressure_field,
):
    rcv_depth = [event_src.z_src]

    # Derive received signal for successive positions of the ship
    for i_rcv in tqdm(
        range(ds.dims["idx_rcv"]),
        bar_format=BAR_FORMAT,
        desc="Derive received signal for successive positions of the ship",
    ):

        delay_to_apply_ship = (
            ds.delay_rcv.min(dim="idx_rcv")
            .sel(lon=ds.lon_src, lat=ds.lat_src, method="nearest")
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
            rcv_range=ds.r_src_rcv.sel(idx_rcv=i_rcv).values,
            rcv_depth=rcv_depth,
            apply_delay=True,
            delay=delay_to_apply_ship,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
        )

        if i_rcv == 0:
            ds["event_signal_time"] = t_rcv.astype(np.float32)
            rcv_signal_event = np.empty(tuple(ds.dims[d] for d in signal_event_dim))

        rcv_signal_event[i_rcv, :] = s_rcv[:, 0, :].T

    ds["rcv_signal_event"] = (
        ["idx_rcv", "src_trajectory_time", "event_signal_time"],
        rcv_signal_event.astype(np.float32),
    )

    ds = add_noise_to_event(ds, snr_dB=snr_dB)
    ds = add_event_correlation(ds)

    return ds


def add_event_anisotropic_env(
    ds,
    snr_dB,
    event_src,
    kraken_grid,
    signal_event_dim,
    grid_pressure_field,
):

    # TODO remove
    greater = False
    smaller = False

    # Array of receiver indexes
    idx_rcv = ds.idx_rcv.values
    # Array of ship positions idx
    idx_src_pos = np.arange(ds.dims["src_trajectory_time"])
    # Array of receiver depths
    rcv_depth = [event_src.z_src]

    # Loop over receivers
    for i_rcv in tqdm(
        idx_rcv,
        bar_format=BAR_FORMAT,
        desc="Derive received signal for successive positions of the ship",
    ):
        # TODO remove
        # i_rcv = 1
        # Get kraken grid for the given rcv
        kraken_range = kraken_grid[i_rcv]["range"].todense()
        kraken_depth = kraken_grid[i_rcv]["depth"].todense()

        # Loop over src positions
        for i_src in tqdm(
            idx_src_pos,
            bar_format=BAR_FORMAT,
            desc="Scanning src positions",
        ):

            # TODO remove
            # i_src = 2

            delay_to_apply_ship = (
                ds.delay_rcv.min(dim="idx_rcv")
                .sel(lon=ds.lon_src, lat=ds.lat_src, method="nearest")
                .values.flatten()
            )

            # Get azimuths for current ship position
            i_az_kraken, az_kraken, rcv_range, transfert_function = (
                get_src_transfert_function(ds, i_rcv, i_src, grid_pressure_field)
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
                ds["event_signal_time"] = t_rcv.astype(np.float32)
                rcv_signal_event = np.zeros(tuple(ds.dims[d] for d in signal_event_dim))
                # rcv_signal_event = np.empty(tuple(ds.dims[d] for d in signal_event_dim))
                rcv_signal_event_visits = np.zeros(
                    tuple(ds.dims[d] for d in signal_event_dim)
                )
                # print(s_rcv)

            s_rcv = s_rcv[:, 0, :].T
            rcv_signal_event[i_rcv, i_src, :] = s_rcv
            rcv_signal_event_visits[i_rcv, i_src, :] += 1

            # print(f"Contains nans ? : {np.any(np.isnan(s_rcv.astype(np.float32)))}")
            # if np.any(np.abs(s_rcv) > 1e20):
            #     greater = True
            #     print(f"Is greater than 1e20")
            # if np.any(np.logical_and((0 < np.abs(s_rcv)), (np.abs(s_rcv) < 1e-20))):
            #     smaller = True
            #     print(f"Is smaller than 1e-20")

    ds["rcv_signal_event"] = (
        ["idx_rcv", "src_trajectory_time", "event_signal_time"],
        rcv_signal_event.astype(np.float32),
    )

    # print(f"Greater : {greater}; smaller : {smaller}")
    # print(f"greater :{np.any(np.abs(rcv_signal_event) > 1e20)}")
    # print(
    #     f"smaller : {np.any(np.logical_and((0 < np.abs(rcv_signal_event)), (np.abs(rcv_signal_event) < 1e-20)))}"
    # )

    ds = add_noise_to_event(ds, snr_dB=snr_dB)
    ds = add_event_correlation(ds)

    return ds


def get_src_transfert_function(ds, i_rcv, i_src, grid_pressure_field):
    """Extract waveguide transfert function for the given source position index and receiver index."""
    az_propa_unique = get_unique_azimuths(ds, i_rcv)
    az_src = ds.az_src_rcv.sel(idx_rcv=i_rcv).isel(src_trajectory_time=i_src).values
    i_az_src = np.argmin(np.abs(az_propa_unique - az_src))

    az = az_propa_unique[i_az_src]  # Azimuth for which we have the transfert function
    transfert_function = grid_pressure_field[i_rcv][
        i_az_src
    ]  # Transfert function for the given azimuth
    rcv_range = (
        ds.r_src_rcv.sel(idx_rcv=i_rcv).isel(src_trajectory_time=i_src).values
    )  # Range along the given azimuth

    return i_az_src, az, rcv_range, transfert_function


def get_unique_azimuths(ds, i_rcv):
    return np.unique(ds.az_propa.sel(idx_rcv=i_rcv).values)


def get_range_from_rcv(grid_info, rcv_info):
    """Compute range between grid points and rcvs. Equivalent to get_azimuth_rcv but for ranges."""
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
        )
        rr_from_rcv[i, :, :] = ranges

    return rr_from_rcv


def get_range_src_rcv_range(lon_src, lat_src, rcv_info):
    """Compute range between src positions and rcvs. Equivalent to get_range_from_rcv but for src positions."""
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
        )
        rr_src_rcv[i, :] = ranges

    return rr_src_rcv


def get_azimuth_rcv(grid_info, rcv_info):
    """Compute azimuth between grid points and rcvs. Equivalent to get_range_from_rcv but for azimuths."""

    llon, llat = np.meshgrid(grid_info["lons"], grid_info["lats"])
    s = llon.shape

    geod = Geod(ellps="WGS84")
    az_rcv = np.empty((len(rcv_info["id"]), s[0], s[1]))

    for i, id in enumerate(rcv_info["id"]):
        # Derive distance from rcv n°i to all grid points
        fwd_az, _, _ = geod.inv(
            lons1=np.ones(s) * rcv_info["lons"][i],
            lats1=np.ones(s) * rcv_info["lats"][i],
            lons2=llon,
            lats2=llat,
        )
        az_rcv[i, :, :] = fwd_az

    return az_rcv


def get_azimuth_src_rcv(lon_src, lat_src, rcv_info):
    """Compute azimuth between src positions and rcvs. Equivalent to get_azimuth_rcv but for src positions."""

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
        )
        az_rcv[i, :] = fwd_az

    return az_rcv


def init_library_dataset(
    grid_info, rcv_info, n_noise_realisations, similarity_metrics, isotropic_env=True
):

    # Init Dataset
    n_rcv = len(rcv_info["id"])
    n_similarity_metrics = len(similarity_metrics)

    # Compute range from each receiver
    rr_rcv = get_range_from_rcv(grid_info, rcv_info)

    ds = xr.Dataset(
        data_vars=dict(
            lon_rcv=(["idx_rcv"], rcv_info["lons"]),
            lat_rcv=(["idx_rcv"], rcv_info["lats"]),
            r_from_rcv=(["idx_rcv", "lat", "lon"], rr_rcv),
            similarity_metric=(
                ["idx_similarity_metric"],
                similarity_metrics,
            ),
        ),
        coords=dict(
            lon=grid_info["lons"],
            lat=grid_info["lats"],
            idx_rcv=np.arange(n_rcv),
            idx_similarity_metric=np.arange(n_similarity_metrics),
            idx_noise_realisation=np.arange(n_noise_realisations),
        ),
        attrs=dict(
            title="Verlinden simulation with simple environment",
            dx=grid_info["dx"],
            dy=grid_info["dy"],
        ),
    )

    if not isotropic_env:
        # Compute azimuths
        az_rcv = get_azimuth_rcv(grid_info, rcv_info)
        ds["az_rcv"] = (["idx_rcv", "lat", "lon"], az_rcv)
        ds["az_rcv"].attrs["units"] = "°"
        ds["az_rcv"].attrs["long_name"] = "Azimuth"

        # Build list of angles to be used in kraken
        dmax = ds.r_from_rcv.min(dim=["lat", "lon", "idx_rcv"]).round(0).values
        delta = min(ds.dx, ds.dy)
        d_az = np.arctan(delta / dmax) * 180 / np.pi
        list_az_th = np.arange(ds.az_rcv.min(), ds.az_rcv.max(), d_az)

        az_propa = np.empty(ds.az_rcv.shape)
        # t0 = time.time()
        for i_rcv in range(n_rcv):
            for az in list_az_th:
                closest_points_idx = (
                    np.abs(ds.az_rcv.sel(idx_rcv=i_rcv) - az) <= d_az / 2
                )
                az_propa[i_rcv, closest_points_idx] = az
        # print(f"Elapsed time : {time.time() - t0}")

        # Add az_propa to dataset
        ds["az_propa"] = (["idx_rcv", "lat", "lon"], az_propa)
        ds["az_propa"].attrs["units"] = "°"
        ds["az_propa"].attrs["long_name"] = "Propagation azimuth"

        # plot_angle_repartition(ds, grid_info)

        # Remove az_rcv
        ds = ds.drop_vars("az_rcv")

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
    }
    for unit in var_unit_mapping.keys():
        for var in var_unit_mapping[unit]:
            ds[var].attrs["units"] = unit

    ds["lon_rcv"].attrs["long_name"] = "Receiver longitude"
    ds["lat_rcv"].attrs["long_name"] = "Receiver latitude"
    ds["r_from_rcv"].attrs["long_name"] = "Range from receiver"
    ds["lon"].attrs["long_name"] = "Longitude"
    ds["lat"].attrs["long_name"] = "Latitude"
    ds["idx_rcv"].attrs["long_name"] = "Receiver index"

    # TODO : need to be changed in case of multiple receivers couples
    ds["delay_rcv"] = ds.r_from_rcv / C0

    # Build OBS pairs
    rcv_pairs = []
    for i in ds.idx_rcv.values:
        for j in range(i + 1, ds.idx_rcv.values[-1] + 1):
            rcv_pairs.append((i, j))
    ds.coords["idx_rcv_pairs"] = np.arange(len(rcv_pairs))
    ds.coords["idx_rcv_in_pair"] = np.arange(2)
    ds["rcv_pairs"] = (["idx_rcv_pairs", "idx_rcv_in_pair"], rcv_pairs)

    return ds


def init_event_dataset(ds, src_info, rcv_info, interp_src_pos_on_grid=False):

    lon_src, lat_src, t_src = src_info["lons"], src_info["lats"], src_info["time"]
    r_src_rcv = get_range_src_rcv_range(lon_src, lat_src, rcv_info)
    az_src_rcv = get_azimuth_src_rcv(lon_src, lat_src, rcv_info)

    ds.coords["event_signal_time"] = []
    ds.coords["src_trajectory_time"] = t_src.astype(np.float32)

    ds["lon_src"] = (["src_trajectory_time"], lon_src.astype(np.float32))
    ds["lat_src"] = (["src_trajectory_time"], lat_src.astype(np.float32))
    ds["r_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(r_src_rcv).astype(np.float32),
    )
    ds["az_src_rcv"] = (
        ["idx_rcv", "src_trajectory_time"],
        np.array(az_src_rcv).astype(np.float32),
    )

    ds["event_signal_time"].attrs["units"] = "s"
    ds["src_trajectory_time"].attrs["units"] = "s"

    ds["lon_src"].attrs["long_name"] = "lon_src"
    ds["lat_src"].attrs["long_name"] = "lat_src"
    ds["r_src_rcv"].attrs["long_name"] = "Range from receiver to source"
    ds["event_signal_time"].attrs["units"] = "Time"
    ds["src_trajectory_time"].attrs["long_name"] = "Time"

    if interp_src_pos_on_grid:
        ds["lon_src"].values = ds.lon.sel(lon=ds.lon_src, method="nearest")
        ds["lat_src"].values = ds.lat.sel(lat=ds.lat_src, method="nearest")
        ds["r_src_rcv"].values = get_range_src_rcv_range(
            ds["lon_src"], ds["lat_src"], rcv_info
        )
        ds.attrs["source_positions"] = "Interpolated on grid"
        ds.attrs["src_pos"] = "on_grid"
    else:
        ds.attrs["source_positions"] = "Not interpolated on grid"
        ds.attrs["src_pos"] = "not_on_grid"

    return ds


def check_waveguide_cutoff(
    testcase,
    library_src,
    dt,
    sig_type,
):

    varin = {}

    fc = waveguide_cutoff_freq(waveguide_depth=testcase.min_depth)
    propagating_freq = library_src.positive_freq[library_src.positive_freq > fc]
    if propagating_freq.size != library_src.kraken_freq.size:
        library_src = init_library_src(dt, testcase.min_depth, sig_type=sig_type)

        # Update testcase with new frequency vector
        varin["freq"] = library_src.kraken_freq
        testcase.update(varin)

    return library_src


def add_noise_to_dataset(xr_dataset, snr_dB):
    for i_rcv in tqdm(
        xr_dataset.idx_rcv, bar_format=BAR_FORMAT, desc="Add noise to received signal"
    ):
        if snr_dB is not None:
            # Add noise to received signal
            xr_dataset.rcv_signal_library.loc[dict(idx_rcv=i_rcv)] = (
                add_noise_to_signal(
                    xr_dataset.rcv_signal_library.sel(idx_rcv=i_rcv).values,
                    snr_dB=snr_dB,
                )
            )
            xr_dataset.attrs["snr_dB"] = snr_dB
        else:
            xr_dataset.attrs["snr_dB"] = "Noiseless"


def fft_convolve_f(a0, a1, axis=-1, workers=8):

    # Compute the cross-correlation of a0 and a1 using the FFT
    corr_01 = sp_fft.irfft(a0 * np.conj(a1), axis=axis, workers=workers)
    # Reorganise so that tau = 0 corresponds to the center of the array
    nmid = corr_01.shape[-1] // 2 + 1
    corr_01 = np.concatenate((corr_01[..., nmid:], corr_01[..., :nmid]), axis=axis)
    return corr_01


def add_correlation_to_dataset(xr_dataset):
    """Derive correlation for each grid pixel."""

    # Derive correlation lags
    xr_dataset.coords["library_corr_lags"] = signal.correlation_lags(
        xr_dataset.sizes["library_signal_time"], xr_dataset.sizes["library_signal_time"]
    )
    xr_dataset["library_corr_lags"].attrs["units"] = "s"
    xr_dataset["library_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each grid pixel
    library_corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]
    library_corr = np.empty(tuple(xr_dataset.sizes[d] for d in library_corr_dim))

    # Faster FFT approach
    ax = 2
    nlag = xr_dataset.sizes["library_corr_lags"]
    for i_pair in tqdm(
        range(xr_dataset.dims["idx_rcv_pairs"]),
        bar_format=BAR_FORMAT,
        desc="Receiver pair cross-correlation computation",
    ):
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
        library_corr[i_pair, ...] = corr_01_fft

    xr_dataset["library_corr"] = (library_corr_dim, library_corr.astype(np.float32))
    xr_dataset["library_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"


def add_noise_to_event(library_dataset, snr_dB):
    ds = library_dataset
    for i_rcv in tqdm(
        ds.idx_rcv, bar_format=BAR_FORMAT, desc="Add noise to event signal"
    ):
        if snr_dB is not None:
            # Add noise to received signal
            ds.rcv_signal_event.loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
                ds.rcv_signal_event.sel(idx_rcv=i_rcv).values, snr_dB
            )
            ds.attrs["snr_dB"] = snr_dB
        else:
            ds.attrs["snr_dB"] = "Noiseless"

    return ds


def add_event_correlation(library_dataset):
    ds = library_dataset
    ds.coords["event_corr_lags"] = signal.correlation_lags(
        ds.dims["event_signal_time"], ds.dims["event_signal_time"]
    )
    ds["event_corr_lags"].attrs["units"] = "s"
    ds["event_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each ship position
    event_corr_dim = ["idx_rcv_pairs", "src_trajectory_time", "event_corr_lags"]
    event_corr = np.empty(tuple(ds.dims[d] for d in event_corr_dim))

    for i_ship in tqdm(
        range(ds.dims["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Derive correlation vector for each ship position",
    ):
        for i_pair, rcv_pair in enumerate(ds.rcv_pairs):
            s0 = ds.rcv_signal_event.sel(idx_rcv=rcv_pair[0]).isel(
                src_trajectory_time=i_ship
            )
            s1 = ds.rcv_signal_event.sel(idx_rcv=rcv_pair[1]).isel(
                src_trajectory_time=i_ship
            )

            corr_01 = signal.correlate(s0, s1)
            autocorr0 = signal.correlate(s0, s0)
            autocorr1 = signal.correlate(s1, s1)
            n0 = corr_01.shape[0] // 2
            corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

            event_corr[i_pair, i_ship, :] = corr_01

            del s0, s1, corr_01

    ds["event_corr"] = (event_corr_dim, event_corr.astype(np.float32))

    return ds


def add_noise_to_signal(sig, snr_dB, noise_type="gaussian"):
    """Add noise to signal assuming sig is either a 3D array (event signal (pos_idx, t)) or a 3D (library signal (x, y, t)) array.
    The noise level is adjusted to garantee the desired SNR at all positions.
    sig : np.array (2D or 3D)
    snr_dB : float
    noise_type : str
    """

    if snr_dB is not None:
        # The further the grid point is from the source, the lower the signal power is and the lower the noise level should be
        P_sig = (
            1 / sig.shape[-1] * np.sum(sig**2, axis=-1)
        )  # Signal power for each position (last dimension is assumed to be time)
        sigma_noise = np.sqrt(
            P_sig * 10 ** (-snr_dB / 10)
        )  # Noise level for each position

        if sig.ndim == 2:  # 2D array (event signal) (pos, time)

            if noise_type == "gaussian":
                # Generate gaussian noise
                for i_ship in range(sig.shape[0]):
                    noise = np.random.normal(0, sigma_noise[i_ship], sig.shape[-1])
                    sig[i_ship, :] += noise
            else:
                raise ValueError("Noise type not supported")

        elif sig.ndim == 3:  # 3D array (library signal) -> (x, y, time)
            if noise_type == "gaussian":
                # Generate gaussian noise
                for i_lon in range(sig.shape[0]):
                    for i_lat in range(sig.shape[1]):
                        noise = np.random.normal(
                            0, sigma_noise[i_lon, i_lat], sig.shape[-1]
                        )
                        sig[i_lon, i_lat, :] += noise
            else:
                raise ValueError("Noise type not supported")

    return sig


def derive_ambiguity(lib_data, event_data, src_traj_times, detection_metric):

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

        if detection_metric == "intercorr0":
            amb_surf = mult_along_axis(
                lib_data_array,
                event_vector_array,
                axis=2,
            )
            autocorr_lib_0 = np.sum(lib_data_array**2, axis=2)
            autocorr_event_0 = np.sum(event_vector_array**2)
            # del lib_data, event_vector

            norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
            amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
            amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

        elif detection_metric == "lstsquares":
            # lib_data = lib_data.values
            # event = event_vector.values

            diff = lib_data_array - event_vector_array
            amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
            amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
            amb_surf = (
                1 - amb_surf
            )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

        elif detection_metric == "hilbert_env_intercorr0":
            lib_env = np.abs(signal.hilbert(lib_data_array))
            event_env = np.abs(signal.hilbert(event_vector_array))

            amb_surf = mult_along_axis(
                lib_env,
                event_env,
                axis=2,
            )

            autocorr_lib_0 = np.sum(lib_env**2, axis=2)
            autocorr_event_0 = np.sum(event_env**2)
            # del lib_env, event_env

            norm = np.sqrt(autocorr_lib_0 * autocorr_event_0)
            amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
            amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
            da_amb_surf[dict(src_trajectory_time=i_src_time)] = amb_surf

    return da_amb_surf


def build_ambiguity_surf(xr_dataset, idx_similarity_metric, i_noise, verbose=True):
    """Build ambiguity surface for each receiver pair and ship position."""

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

    if i_noise == 0 and idx_similarity_metric == 0:
        # Create ambiguity surface dataarray
        xr_dataset["ambiguity_surface"] = (
            ambiguity_surface_dim,
            ambiguity_surface,
        )
        # Add attributes
        xr_dataset.ambiguity_surface.attrs["long_name"] = "Ambiguity surface"
        xr_dataset.ambiguity_surface.attrs["units"] = "dB"

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

    # Store all ambiguity surfaces in a list (for parrallel processing)
    ambiguity_surfaces = []

    for i_pair in xr_dataset.idx_rcv_pairs:
        lib_data = xr_dataset.library_corr.sel(idx_rcv_pairs=i_pair)
        event_data = xr_dataset.event_corr.sel(idx_rcv_pairs=i_pair)

        """ Parallel processing"""
        # # TODO: swicht between parallel and sequential processing depending on the number of ship positions
        # # Init pool
        # pool = multiprocessing.Pool(processes=N_CORES)
        # # Build the parameter pool
        # idx_ship_intervalls = np.array_split(
        #     np.arange(xr_dataset.dims["src_trajectory_time"]), N_CORES
        # )
        # src_traj_time_intervalls = np.array_split(xr_dataset["src_trajectory_time"], N_CORES)
        # param_pool = [
        #     (lib_data, event_data, src_traj_time_intervalls[i], detection_metric)
        #     for i in range(len(idx_ship_intervalls))
        # ]
        # # Run parallel processes
        # results = pool.starmap(derive_ambiguity, param_pool)
        # # Close pool
        # pool.close()
        # # Wait for all processes to finish
        # pool.join()

        # ambiguity_surfaces += results
        # print(f"Elapsed time (parallel) : {time.time() - t0}")

        """ Sequential processing"""
        t0 = time.time()

        lib_data_array = lib_data.values

        for i_ship in tqdm(
            range(xr_dataset.dims["src_trajectory_time"]),
            bar_format=BAR_FORMAT,
            desc="Build ambiguity surface",
        ):
            event_vector = event_data.isel(src_trajectory_time=i_ship)
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
                ambiguity_surface_sim[i_pair, i_ship, ...] = amb_surf

            elif similarity_metric == "lstsquares":

                diff = lib_data_array - event_vector_array
                amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
                amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
                amb_surf = (
                    1 - amb_surf
                )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

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
                ambiguity_surface_sim[i_pair, i_ship, ...] = amb_surf

            del amb_surf

        print(f"Elapsed time (for loop) : {time.time() - t0}")

    # Store ambiguity surface in dataset
    xr_dataset.ambiguity_surface[dict(idx_similarity_metric=idx_similarity_metric)] = (
        ambiguity_surface_sim
    )

    # xr_dataset["ambiguity_surface"] = (
    #     ambiguity_surface_dim,
    #     ambiguity_surface,
    # )

    # # Merge dataarrays # TODO : uncomment in case of parallel processing
    # amb_surf_merged = xr.merge(ambiguity_surfaces)
    # xr_dataset["ambiguity_surface"] = amb_surf_merged["ambiguity_surface"]
    # xr_dataset = xr.merge([xr_dataset, amb_surf_merged])

    # Analyse ambiguity surface to detect source position
    get_detected_pos(xr_dataset, idx_similarity_metric, i_noise)


def get_detected_pos(xr_dataset, idx_similarity_metric, i_noise, method="absmax"):
    """Get detected position from ambiguity surface."""
    ambiguity_surface = xr_dataset.ambiguity_surface.isel(
        idx_similarity_metric=idx_similarity_metric
    )

    if method == "absmax":
        max_pos_idx = ambiguity_surface.argmax(dim=["lon", "lat"])
        ilon_detected = max_pos_idx["lon"]  # Index of detected longitude
        ilat_detected = max_pos_idx["lat"]  # Index of detected longitude

        detected_lon = xr_dataset.lon.isel(lon=ilon_detected).values
        detected_lat = xr_dataset.lat.isel(lat=ilat_detected).values

    # TODO : add other methods to take a larger number of values into account
    else:
        raise ValueError("Method not supported")

    # Store detected position in dataset
    dict_sel = dict(
        idx_noise_realisation=i_noise, idx_similarity_metric=idx_similarity_metric
    )
    # ! need to use loc when assigning values to a DataArray to avoid silent failing !
    xr_dataset.detected_pos_lon[dict_sel] = detected_lon
    xr_dataset.detected_pos_lat[dict_sel] = detected_lat


def init_library_src(dt, min_waveguide_depth, sig_type="pulse"):
    nfft = None
    if sig_type == "ship":
        library_src_sig, t_library_src_sig = ship_noise(T=dt)
        # nfft = fs * dt

    elif sig_type == "pulse":
        library_src_sig, t_library_src_sig = pulse(T=dt, f=25, fs=100)

    elif sig_type == "pulse_train":
        library_src_sig, t_library_src_sig = pulse_train(T=dt, f=25, fs=100)

    elif sig_type == "debug_pulse":
        fs = 40
        library_src_sig, t_library_src_sig = pulse(T=0.1, f=10, fs=fs)
        nfft = int(fs * dt)

    if sig_type in ["ship", "pulse_train"]:
        # Apply hanning window
        library_src_sig *= np.hanning(len(library_src_sig))

    library_src = AcousticSource(
        signal=library_src_sig,
        time=t_library_src_sig,
        name=sig_type,
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    return library_src


def plot_src(library_src, testcase):
    pfig = PubFigure()
    library_src.display_source(plot_spectrum=False)
    fig = plt.gcf()
    fig.set_size_inches(pfig.size)
    axs = fig.axes
    axs[0].set_title("")
    axs[1].set_title("")
    fig.suptitle("Source signal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(testcase.env_dir, "env_desc_img", f"src_{testcase.name}.png")
    )


def init_event_src_traj(src_info, dt):
    """Init the source trajectory given initial position, speed and duration"""
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
    )

    # Determine coordinates along trajectory
    traj = geod.inv_intermediate(
        lat1=lat_i, lon1=lon_i, lat2=lat_f, lon2=lon_f, npts=src_info["max_nb_of_pos"]
    )

    src_info["lons"] = np.array(traj.lons)
    src_info["lats"] = np.array(traj.lats)


def init_grid_around_event_src_traj(src_info, grid_info):
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
    )
    max_lon_grid, _, _ = geod.fwd(
        lons=max_lon,
        lats=mean_lat,
        az=90,
        dist=offset_lon,
    )

    # ):  # Case where the trajectory is in the southern hemisphere
    _, min_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=min_lat,
        az=180,
        dist=offset_lat,
    )
    _, max_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=max_lat,
        az=0,
        dist=offset_lat,
    )

    grid_lons = np.array(
        geod.inv_intermediate(
            lat1=mean_lat,
            lon1=min_lon_grid,
            lat2=mean_lat,
            lon2=max_lon_grid,
            del_s=grid_info["dx"],
        ).lons
    )
    grid_lats = np.array(
        geod.inv_intermediate(
            lat1=min_lat_grid,
            lon1=mean_lon,
            lat2=max_lat_grid,
            lon2=mean_lon,
            del_s=grid_info["dy"],
        ).lats
    )
    grid_info["lons"] = grid_lons
    grid_info["lats"] = grid_lats
    grid_info["min_lat"] = np.min(grid_lats)
    grid_info["max_lat"] = np.max(grid_lats)
    grid_info["min_lon"] = np.min(grid_lons)
    grid_info["max_lon"] = np.max(grid_lons)


def get_max_kraken_range(rcv_info, grid_info):
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
        )

        max_r.append(np.max(ranges))

    rcv_info["max_kraken_range_m"] = np.round(max_r, -2)


def get_dist_between_rcv(rcv_info):
    geod = Geod(ellps="WGS84")

    dist_inter_rcv = []
    for i_rcv in range(len(rcv_info["id"]) - 1):
        # Derive distance to the 4 corners of the grid
        _, _, dist_i = geod.inv(
            lons1=[rcv_info["lons"][i_rcv]],
            lats1=[rcv_info["lats"][i_rcv]],
            lons2=[rcv_info["lons"][i_rcv + 1]],
            lats2=[rcv_info["lats"][i_rcv + 1]],
        )

        dist_inter_rcv.append(np.round(dist_i, 0))

    rcv_info["dist_inter_rcv"] = dist_inter_rcv


def load_rhumrum_obs_pos(obs_id):
    pos = pd.read_csv(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\rhum_rum_obs_pos.csv",
        index_col="id",
        delimiter=",",
    )
    return pos.loc[obs_id]


def print_simulation_info(testcase, src_info, rcv_info, grid_info):
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


def get_populated_path(grid_info, kraken_env, src_signal_type):
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
        f"populated_{boundaries}_{src_signal_type}.nc",
    )
    return populated_path


def save_dataset(
    xr_dataset,
    output_folder,
    analysis_folder,
    env_filename,
    src_name,
    snr_tag,
):
    # Build path to save dataset and corresponding path to save analysis results produced later on
    xr_dataset.attrs["fullpath_output"] = os.path.join(
        output_folder,
        env_filename,
        src_name,
        xr_dataset.src_pos,
        f"output_{snr_tag}.nc",
    )
    xr_dataset.attrs["fullpath_analysis"] = os.path.join(
        analysis_folder,
        env_filename,
        src_name,
        xr_dataset.src_pos,
        snr_tag,
    )

    # Ensure that the output folder exists
    if not os.path.exists(os.path.dirname(xr_dataset.fullpath_output)):
        os.makedirs(os.path.dirname(xr_dataset.fullpath_output))

    if not os.path.exists(xr_dataset.fullpath_analysis):
        os.makedirs(xr_dataset.fullpath_analysis)

    # Save dataset to netcdf
    xr_dataset.to_netcdf(xr_dataset.fullpath_output)


def get_snr_tag(snr_dB, verbose=True):
    if snr_dB is None:
        snr_tag = "noiseless"
        snr_msg = "Performing localisation process without noise"
    else:
        snr_tag = f"snr{snr_dB}dB"
        snr_msg = f"Performing localisation process with additive gaussian white noise SNR = {snr_dB}dB"

    if verbose:
        print("## " + snr_msg + " ##")

    return snr_tag


if __name__ == "__main__":
    pass
