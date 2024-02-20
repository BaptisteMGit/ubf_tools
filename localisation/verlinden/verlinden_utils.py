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
import scipy.signal as signal
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Geod
from scipy.sparse import csr_matrix

from cst import BAR_FORMAT, C0
from misc import mult_along_axis
from signals import ship_noise, pulse, pulse_train
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import waveguide_cutoff_freq, get_rcv_pos_idx
from illustration.verlinden_nx2d import plot_angle_repartition
from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal_from_broadband_pressure_field,
)
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.plot_utils import plotshd

from localisation.verlinden.verlinden_path import (
    VERLINDEN_OUTPUT_FOLDER,
    VERLINDEN_ANALYSIS_FOLDER,
    VERLINDEN_POPULATED_FOLDER,
)


def populate_istropic_env(ds, library_src, kraken_env, kraken_flp, signal_library_dim):

    delay_to_apply = ds.delay_rcv.min(dim="idx_rcv").values.flatten()

    # Run KRAKEN
    grid_pressure_field = runkraken(kraken_env, kraken_flp, library_src.kraken_freq)

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
            shd_fpath=kraken_env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=library_src.kraken_freq,
            source=library_src,
            rcv_range=rr_from_rcv_flat,
            rcv_depth=[library_src.z_src],
            apply_delay=True,
            delay=delay_to_apply,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
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
    grid_pressure_field = np.empty((idx_rcv.size), dtype=object)
    kraken_range_rcv = np.empty((idx_rcv.size), dtype=object)
    kraken_depth_rcv = np.empty((idx_rcv.size), dtype=object)
    rcv_depth = [library_src.z_src]

    # Loop over receivers
    for i_rcv in tqdm(
        idx_rcv,
        bar_format=BAR_FORMAT,
        desc="Populate grid with received signal",
    ):
        # Loop over possible azimuths
        # azimuths_rcv = np.unique(ds.az_propa.sel(idx_rcv=i_rcv).values)
        azimuths_rcv = get_unique_azimuths(ds, i_rcv)
        az_pressure_field = np.empty((azimuths_rcv.size), dtype=object)

        for i_az in tqdm(
            range(azimuths_rcv.size),
            bar_format=BAR_FORMAT,
            desc="Scanning azimuths",
        ):
            az = azimuths_rcv[i_az]

            # Get environement for selected angle
            kraken_env, kraken_flp = testcase(
                freq=library_src.kraken_freq,
                max_range_m=rcv_info["max_kraken_range_m"][i_rcv],
                azimuth=az,
                rcv_lon=rcv_info["lons"][i_rcv],
                rcv_lat=rcv_info["lats"][i_rcv],
            )

            # Assert kraken freq set with correct min_depth (otherwise postprocess will fail)
            kraken_env, kraken_flp, library_src = check_waveguide_cutoff(
                testcase,
                kraken_env,
                kraken_flp,
                library_src,
                max_range_m=rcv_info["max_kraken_range_m"][i_rcv],
                dt=src_info["dt"],
                sig_type=src_info["signal_type"],
            )
            # Get receiver ranges for selected angle
            idx_az = ds.az_propa.sel(idx_rcv=i_rcv).values == az
            rr_from_rcv_az = ds.r_from_rcv.sel(idx_rcv=i_rcv).values[idx_az].flatten()
            # Get received signal for selected angle
            delay_to_apply = ds.delay_rcv.min(dim="idx_rcv").values[idx_az].flatten()

            # Run kraken for selected angle
            pf = runkraken(kraken_env, kraken_flp, library_src.kraken_freq)

            (
                t_rcv,
                s_rcv,
                Pos,
            ) = postprocess_received_signal_from_broadband_pressure_field(
                shd_fpath=kraken_env.shd_fpath,
                broadband_pressure_field=pf,
                frequencies=library_src.kraken_freq,
                source=library_src,
                rcv_range=rr_from_rcv_az,
                rcv_depth=rcv_depth,
                apply_delay=True,
                delay=delay_to_apply,
                minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
            )

            pf = np.squeeze(pf, axis=(1, 2))
            # rr, zz, _ = get_rcv_pos_idx(
            #     shd_fpath=kraken_env.shd_fpath,
            #     rcv_depth=rcv_depth,
            #     rcv_range=rr_from_rcv_az,
            # )
            # az_pressure_field[i_az] = pf[:, zz, rr]
            # Store azimuth specific pressure field
            # az_pressure_field[i_az] = pf

            if i_az == 0:
                # Get pos field and store it in dataset (receiver grid depends on rcv)
                _, _, _, _, _, _, field_pos, _ = readshd(
                    filename=kraken_env.shd_fpath, freq=0
                )
                r_rcv = field_pos["r"]["r"]
                z_rcv = field_pos["r"]["z"]
                kraken_range_rcv[i_rcv] = r_rcv
                kraken_depth_rcv[i_rcv] = z_rcv

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

            sparse_pf = sparse.COO(pf)
            az_pressure_field[i_az] = sparse_pf

            # Store received signal in dataset
            if i_rcv == 0:
                ds["library_signal_time"] = t_rcv.astype(np.float32)
                ds["library_signal_time"].attrs["units"] = "s"
                ds["library_signal_time"].attrs["long_name"] = "Time"
                rcv_signal_library = np.empty(
                    tuple(ds.dims[d] for d in signal_library_dim)
                )

            # Time domain signal
            s_rcv = s_rcv[:, 0, :].T
            rcv_signal_library[i_rcv, idx_az, :] = s_rcv

        # Store pressure field for all azimuths relative to current rcv
        grid_pressure_field[i_rcv] = az_pressure_field

    print("1")
    # Store kraken range and depth for rcv
    ds["kraken_range"] = (["idx_rcv"], kraken_range_rcv)
    ds["kraken_depth"] = (["idx_rcv"], kraken_depth_rcv)

    return ds, rcv_signal_library, grid_pressure_field


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
    kraken_env,
    signal_event_dim,
    grid_pressure_field,
):

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
        # Get kraken grid for the given rcv
        kraken_range = ds.kraken_range.sel(idx_rcv=i_rcv).values
        kraken_depth = ds.kraken_depth.sel(idx_rcv=i_rcv).values
        min_wg_depth = kraken_depth.min()

        # Loop over src positions
        for i_src in tqdm(
            idx_src_pos,
            bar_format=BAR_FORMAT,
            desc="Scanning ship positions",
        ):

            delay_to_apply_ship = (
                ds.delay_rcv.min(dim="idx_rcv")
                .sel(lon=ds.lon_src, lat=ds.lat_src, method="nearest")
                .values.flatten()
            )

            # Get azimuths for current ship position
            az_kraken, rcv_range, transfert_function = get_src_transfert_function(
                ds, i_rcv, i_src, grid_pressure_field
            )

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
    return az, rcv_range, transfert_function


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


def init_library_dataset(grid_info, rcv_info, isotropic_env=True):

    # Init Dataset
    n_rcv = len(rcv_info["id"])

    # Compute range from each receiver
    rr_rcv = get_range_from_rcv(grid_info, rcv_info)

    ds = xr.Dataset(
        data_vars=dict(
            lon_rcv=(["idx_rcv"], rcv_info["lons"]),
            lat_rcv=(["idx_rcv"], rcv_info["lats"]),
            r_from_rcv=(["idx_rcv", "lat", "lon"], rr_rcv),
        ),
        coords=dict(
            lon=grid_info["lons"],
            lat=grid_info["lats"],
            idx_rcv=np.arange(n_rcv),
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
    testcase, kraken_env, kraken_flp, library_src, max_range_m, dt, sig_type
):
    fc = waveguide_cutoff_freq(max_depth=kraken_env.bathy.bathy_depth.min())
    propagating_freq = library_src.positive_freq[library_src.positive_freq > fc]
    if propagating_freq.size != library_src.kraken_freq.size:
        min_waveguide_depth = kraken_env.bathy.bathy_depth.min()
        library_src = init_library_src(dt, min_waveguide_depth, sig_type=sig_type)

        # Update env and flp
        kraken_env, kraken_flp = testcase(
            freq=library_src.kraken_freq,
            min_waveguide_depth=min_waveguide_depth,
            max_range_m=max_range_m,
        )

    return kraken_env, kraken_flp, library_src


def add_noise_to_dataset(library_dataset, snr_dB):
    ds = library_dataset
    for i_rcv in tqdm(
        ds.idx_rcv, bar_format=BAR_FORMAT, desc="Add noise to received signal"
    ):
        if snr_dB is not None:
            # Add noise to received signal
            ds.rcv_signal_library.loc[dict(idx_rcv=i_rcv)] = add_noise_to_signal(
                ds.rcv_signal_library.sel(idx_rcv=i_rcv).values, snr_dB=snr_dB
            )
            ds.attrs["snr_dB"] = snr_dB
        else:
            ds.attrs["snr_dB"] = "Noiseless"

    return ds


def add_correlation_to_dataset(library_dataset):
    ds = library_dataset
    ds.coords["library_corr_lags"] = signal.correlation_lags(
        ds.dims["library_signal_time"], ds.dims["library_signal_time"]
    )
    ds["library_corr_lags"].attrs["units"] = "s"
    ds["library_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each grid pixel
    library_corr_dim = ["idx_rcv_pairs", "lat", "lon", "library_corr_lags"]
    library_corr = np.empty(tuple(ds.dims[d] for d in library_corr_dim))

    # May be way faster with a FFT based approach
    ns = ds.dims["library_signal_time"]
    for i_pair in tqdm(
        range(ds.dims["idx_rcv_pairs"]),
        bar_format=BAR_FORMAT,
        desc="Derive correlation vector for each grid pixel",
    ):
        rcv_pair = ds.rcv_pairs.isel(idx_rcv_pairs=i_pair)
        for i_lon in tqdm(
            range(ds.dims["lon"]),
            bar_format=BAR_FORMAT,
            desc="Scanning longitude axis",
            leave=False,
        ):
            for i_lat in tqdm(
                range(ds.dims["lat"]),
                bar_format=BAR_FORMAT,
                desc="Scanning latitude axis",
                leave=False,
            ):
                s0 = ds.rcv_signal_library.sel(
                    idx_rcv=rcv_pair[0],
                    lon=ds.lon.isel(lon=i_lon),
                    lat=ds.lat.isel(lat=i_lat),
                )
                s1 = ds.rcv_signal_library.sel(
                    idx_rcv=rcv_pair[1],
                    lon=ds.lon.isel(lon=i_lon),
                    lat=ds.lat.isel(lat=i_lat),
                )
                corr_01 = signal.correlate(s0, s1)
                n0 = corr_01.shape[0] // 2
                autocorr0 = signal.correlate(s0, s0)
                autocorr1 = signal.correlate(s1, s1)
                corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

                library_corr[i_pair, i_lat, i_lon, :] = corr_01

                del s0, s1, corr_01

    ds["library_corr"] = (library_corr_dim, library_corr.astype(np.float32))
    # if snr_dB is None:
    #     ds.attrs["snr_dB"] = "noiseless"
    #     snr_tag = "noiseless"
    # else:
    #     ds.attrs["snr_dB"] = snr_dB
    #     snr_tag = f"snr{snr_dB}dB"

    # Build path to save populated dataset

    # ds.attrs["fullpath_populated"] = os.path.join(
    #     VERLINDEN_POPULATED_FOLDER,
    #     kraken_env.filename,
    #     library_src.name,
    #     f"populated_{snr_tag}.nc",
    # )

    # ds.attrs["fullpath_populated"] = os.path.join(
    #     VERLINDEN_POPULATED_FOLDER,
    #     kraken_env.filename,
    #     f"populated_{library_src.name}.nc",
    # )
    # if not os.path.exists(os.path.dirname(ds.fullpath_populated)):
    #     os.makedirs(os.path.dirname(ds.fullpath_populated))

    # ds.to_netcdf(ds.fullpath_populated)

    return ds


# def add_event_to_dataset(
#     library_dataset,
#     grid_pressure_field,
#     kraken_env,
#     event_src,
#     event_t,
#     x_event_t,
#     y_event_t,
#     z_event,
#     interp_src_pos_on_grid=False,
#     snr_dB=None,
# ):
#     ds = library_dataset

#     r_event_t = [
#         np.sqrt(
#             (x_event_t - ds.x_rcv.sel(idx_rcv=i_rcv).values) ** 2
#             + (y_event_t - ds.y_rcv.sel(idx_rcv=i_rcv).values) ** 2
#         )
#         for i_rcv in range(ds.dims["idx_rcv"])
#     ]

#     ds.coords["event_signal_time"] = []
#     ds.coords["src_trajectory_time"] = event_t.astype(np.float32)

#     ds["x_ship"] = (["src_trajectory_time"], x_event_t.astype(np.float32))
#     ds["y_ship"] = (["src_trajectory_time"], y_event_t.astype(np.float32))
#     ds["r_src_rcv"] = (
#         ["idx_rcv", "src_trajectory_time"],
#         np.array(r_event_t).astype(np.float32),
#     )

#     ds["event_signal_time"].attrs["units"] = "s"
#     ds["src_trajectory_time"].attrs["units"] = "s"

#     ds["x_ship"].attrs["long_name"] = "x_ship"
#     ds["y_ship"].attrs["long_name"] = "y_ship"
#     ds["r_src_rcv"].attrs["long_name"] = "Range from receiver to source"
#     ds["event_signal_time"].attrs["units"] = "Time"
#     ds["src_trajectory_time"].attrs["long_name"] = "Time"

#     if interp_src_pos_on_grid:
#         ds["x_ship"] = ds.x.sel(x=ds.x_ship, method="nearest")
#         ds["y_ship"] = ds.y.sel(y=ds.y_ship, method="nearest")
#         ds["r_src_rcv"].values = [
#             np.sqrt(
#                 (ds.x_ship - ds.x_rcv.sel(idx_rcv=i_rcv)) ** 2
#                 + (ds.y_ship - ds.y_rcv.sel(idx_rcv=i_rcv)) ** 2
#             )
#             for i_rcv in range(ds.dims["idx_rcv"])
#         ]
#         ds.attrs["source_positions"] = "Interpolated on grid"
#         ds.attrs["src_pos"] = "on_grid"
#     else:
#         ds.attrs["source_positions"] = "Not interpolated on grid"
#         ds.attrs["src_pos"] = "not_on_grid"

#     signal_event_dim = ["idx_rcv", "src_trajectory_time", "event_signal_time"]

#     # Derive received signal for successive positions of the ship
#     for i_rcv in tqdm(
#         range(ds.dims["idx_rcv"]),
#         bar_format=BAR_FORMAT,
#         desc="Derive received signal for successive positions of the ship",
#     ):
#         delay_to_apply_ship = (
#             ds.delay_rcv.min(dim="idx_rcv")
#             .sel(x=ds.x_ship, y=ds.y_ship, method="nearest")
#             .values.flatten()
#         )

#         (
#             t_rcv,
#             s_rcv,
#             Pos,
#         ) = postprocess_received_signal_from_broadband_pressure_field(
#             shd_fpath=kraken_env.shd_fpath,
#             broadband_pressure_field=grid_pressure_field,
#             frequencies=event_src.kraken_freq,
#             source=event_src,
#             rcv_range=ds.r_src_rcv.sel(idx_rcv=i_rcv).values,
#             rcv_depth=[z_event],
#             apply_delay=True,
#             delay=delay_to_apply_ship,
#             minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
#         )

#         if i_rcv == 0:
#             ds["event_signal_time"] = t_rcv.astype(np.float32)
#             rcv_signal_event = np.empty(tuple(ds.dims[d] for d in signal_event_dim))

#         rcv_signal_event[i_rcv, :] = s_rcv[:, 0, :].T

#         # Free memory
#         del t_rcv, s_rcv, Pos

#     ds["rcv_signal_event"] = (
#         ["idx_rcv", "src_trajectory_time", "event_signal_time"],
#         rcv_signal_event.astype(np.float32),
#     )

#     ds = add_noise_to_event(ds, snr_dB=snr_dB)
#     ds = add_event_correlation(ds)

#     return ds


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
            n0 = corr_01.shape[0] // 2
            autocorr0 = signal.correlate(s0, s0)
            autocorr1 = signal.correlate(s1, s1)
            corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

            event_corr[i_pair, i_ship, :] = corr_01

            del s0, s1, corr_01

    ds["event_corr"] = (event_corr_dim, event_corr.astype(np.float32))

    return ds


def add_noise_to_signal(sig, snr_dB):
    # Add noise to signal assuming sig is either a 1D (like event signal (t)) or a 3D (like library signal (x, y, t)) array
    if snr_dB is not None:
        # First simple implementation : same noise level for all positions
        # TODO : This need to be improved to take into account the propagation loss

        P_sig = (
            1 / sig.shape[-1] * np.sum(sig**2, axis=-1)
        )  # Signal power for each position
        sigma_noise = np.sqrt(
            P_sig * 10 ** (-snr_dB / 10)
        )  # Noise level for each position

        if sig.ndim == 2:  # 2D array (event signal) (pos, time)
            # Generate gaussian noise
            for i_ship in range(sig.shape[0]):
                noise = np.random.normal(0, sigma_noise[i_ship], sig.shape[-1])
                sig[i_ship, :] += noise

        elif sig.ndim == 3:  # 3D array (library signal) -> (x, y, time)
            # Generate gaussian noise
            for i_lon in range(sig.shape[0]):
                for i_lat in range(sig.shape[1]):
                    noise = np.random.normal(
                        0, sigma_noise[i_lon, i_lat], sig.shape[-1]
                    )
                    sig[i_lon, i_lat, :] += noise

    return sig


def build_ambiguity_surf(ds, detection_metric):
    ambiguity_surface_dim = ["idx_rcv_pairs", "src_trajectory_time", "lat", "lon"]
    ambiguity_surface = np.empty(tuple(ds.dims[d] for d in ambiguity_surface_dim))

    for i_ship in tqdm(
        range(ds.dims["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Build ambiguity surface",
    ):
        for i_pair in ds.idx_rcv_pairs:
            lib_data = ds.library_corr.sel(idx_rcv_pairs=i_pair)
            event_vector = ds.event_corr.sel(idx_rcv_pairs=i_pair).isel(
                src_trajectory_time=i_ship
            )

            if detection_metric == "intercorr0":
                amb_surf = mult_along_axis(
                    lib_data,
                    event_vector,
                    axis=2,
                )
                autocorr_lib = np.sum(lib_data.values**2, axis=2)
                autocorr_event = np.sum(event_vector.values**2)
                del lib_data, event_vector

                norm = np.sqrt(autocorr_lib * autocorr_event)
                amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
                amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

            elif detection_metric == "lstsquares":
                lib = lib_data.values
                event = event_vector.values
                del lib_data, event_vector

                diff = lib - event
                amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
                amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
                amb_surf = (
                    1 - amb_surf
                )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

            elif detection_metric == "hilbert_env_intercorr0":
                lib_env = np.abs(signal.hilbert(lib_data))
                event_env = np.abs(signal.hilbert(event_vector))
                del lib_data, event_vector

                amb_surf = mult_along_axis(
                    lib_env,
                    event_env,
                    axis=2,
                )

                autocorr_lib = np.sum(lib_env**2, axis=2)
                autocorr_event = np.sum(event_env**2)
                del lib_env, event_env

                norm = np.sqrt(autocorr_lib * autocorr_event)
                amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
                amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

            del amb_surf

    ds["ambiguity_surface"] = (
        ambiguity_surface_dim,
        ambiguity_surface,
    )

    # Derive src position
    ds["detected_pos_x"] = ds.x.isel(
        x=ds.ambiguity_surface.argmax(dim=["lon", "lat"])["lon"]
    )
    ds["detected_pos_y"] = ds.y.isel(
        y=ds.ambiguity_surface.argmax(dim=["lon", "lat"])["lat"]
    )

    ds.attrs["detection_metric"] = detection_metric

    return ds


def init_library_src(dt, min_waveguide_depth, sig_type="pulse"):
    if sig_type == "ship":
        library_src_sig, t_library_src_sig = ship_noise(T=dt)

    elif sig_type == "pulse":
        library_src_sig, t_library_src_sig = pulse(T=dt, f=25, fs=100)

    elif sig_type == "pulse_train":
        library_src_sig, t_library_src_sig = pulse_train(T=dt, f=25, fs=100)

    elif sig_type == "debug_pulse":
        library_src_sig, t_library_src_sig = pulse(T=dt, f=5, fs=20)

    if sig_type in ["ship", "pulse_train"]:
        # Apply hanning window
        library_src_sig *= np.hanning(len(library_src_sig))

    library_src = AcousticSource(
        signal=library_src_sig,
        time=t_library_src_sig,
        name=sig_type,
        waveguide_depth=min_waveguide_depth,
    )

    return library_src


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

    # Subsample the trajectory to match the desired number of positions
    # TODO


def init_grid_around_event_src_traj(src_info, grid_info):
    min_lon, max_lon = np.min(src_info["lons"]), np.max(src_info["lons"])
    min_lat, max_lat = np.min(src_info["lats"]), np.max(src_info["lats"])
    mean_lon, mean_lat = np.mean(src_info["lons"]), np.mean(src_info["lats"])

    geod = Geod(ellps="WGS84")
    min_lon_grid, _, _ = geod.fwd(
        lons=min_lon,
        lats=mean_lat,
        az=270,
        dist=grid_info["Lx"] / 2,
    )
    max_lon_grid, _, _ = geod.fwd(
        lons=max_lon,
        lats=mean_lat,
        az=90,
        dist=grid_info["Lx"] / 2,
    )
    _, min_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=min_lat,
        az=0,
        dist=grid_info["Ly"] / 2,
    )
    _, max_lat_grid, _ = geod.fwd(
        lons=mean_lon,
        lats=max_lat,
        az=180,
        dist=grid_info["Ly"] / 2,
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


def load_rhumrum_obs_pos(obs_id):
    pos = pd.read_csv(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\rhum_rum_obs_pos.csv",
        index_col="id",
        delimiter=",",
    )
    return pos.loc[obs_id]


def print_simulation_info(src_info, rcv_info, grid_info):
    balises = "".join(["#"] * 80)
    balises_inter = "".join(["#"] * 40)
    header_msg = "Start simulation with the following parameters:"
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


if __name__ == "__main__":
    pass
