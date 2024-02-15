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
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Geod

from cst import BAR_FORMAT, C0
from misc import mult_along_axis
from signals import ship_noise, pulse, pulse_train
from propa.kraken_toolbox.utils import waveguide_cutoff_freq
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

    delay_to_apply = ds.delay_obs.min(dim="idx_obs").values.flatten()

    # Run KRAKEN
    grid_pressure_field = runkraken(kraken_env, kraken_flp, library_src.kraken_freq)

    # Loop over receivers
    for i_obs in tqdm(
        ds.idx_obs, bar_format=BAR_FORMAT, desc="Populate grid with received signal"
    ):

        rr_from_obs_flat = ds.r_from_obs.sel(idx_obs=i_obs).values.flatten()

        (
            t_obs,
            s_obs,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=kraken_env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=library_src.kraken_freq,
            source=library_src,
            rcv_range=rr_from_obs_flat,
            rcv_depth=[library_src.z_src],
            apply_delay=True,
            delay=delay_to_apply,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
        )

        if i_obs == 0:
            ds["library_signal_time"] = t_obs.astype(np.float32)
            ds["library_signal_time"].attrs["units"] = "s"
            ds["library_signal_time"].attrs["long_name"] = "Time"

            rcv_signal_library = np.empty(tuple(ds.dims[d] for d in signal_library_dim))

        # Time domain signal
        s_obs = s_obs[:, 0, :].T
        s_obs = s_obs.reshape(
            ds.dims["y"], ds.dims["x"], ds.dims["library_signal_time"]
        )

        rcv_signal_library[i_obs, :] = s_obs

        # Free memory
        del s_obs, rr_from_obs_flat

    return ds, rcv_signal_library, grid_pressure_field


def populate_anistropic_env(ds, library_src, signal_library_dim, testcase):

    # delay_to_apply = ds.delay_obs.min(dim="idx_obs").values.flatten()

    # # Run KRAKEN
    # grid_pressure_field = runkraken(kraken_env, kraken_flp, library_src.kraken_freq)

    # Loop over receivers
    for i_obs in tqdm(
        ds.idx_obs, bar_format=BAR_FORMAT, desc="Populate grid with received signal"
    ):
        # Loop over possible azimuths
        for i_theta, theta in tqdm(
            enumerate(np.unique(ds.theta_propa.values)),
            bar_format=BAR_FORMAT,
            desc="Scanning azimuths",
        ):

            # TODO add params to load testcase :
            kraken_env, kraken_flp = testcase(
                freq=[20],
                min_waveguide_depth=100,
                max_range_m=50 * 1e3,
                azimuth=theta,
                obs_lon=65.94,
                obs_lat=-27.58,
            )

            kraken_env, kraken_flp = testcase(
                freq=library_src.kraken_freq,
                max_range_m=max_range_m,
                azimuth=theta,
                obs_lon=ds.lon_obs.sel(
                    idx_obs=i_obs
                ).values,  # TODO : those variables does not exist yet !!
                obs_lat=ds.lat_obs.sel(idx_obs=i_obs).values,
            )

            # Assert kraken freq set with correct min_depth (otherwise postprocess will fail)
            kraken_env, kraken_flp, library_src = check_waveguide_cutoff(
                testcase,
                kraken_env,
                library_src,
                max_range_m,
                dt,
                sig_type=src_info["src_signal_type"],
            )

            # TODO
            # Get environement for selected angle
            # Run kraken for selected angle
            # Get receiver ranges for selected angle
            # Get received signal for selected angle
            # Store received signal in dataset
            pass

        rr_from_obs_flat = ds.r_from_obs.sel(idx_obs=i_obs).values.flatten()

        (
            t_obs,
            s_obs,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=kraken_env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=library_src.kraken_freq,
            source=library_src,
            rcv_range=rr_from_obs_flat,
            rcv_depth=[library_src.z_src],
            apply_delay=True,
            delay=delay_to_apply,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
        )

        # t_obs, s_obs, Pos = postprocess_received_signal(
        #     shd_fpath=kraken_env.shd_fpath,
        #     source=library_src,
        #     rcv_range=rr_from_obs_flat,
        #     rcv_depth=[z_src],
        #     apply_delay=True,
        #     delay=delay_to_apply,
        # )
        if i_obs == 0:
            ds["library_signal_time"] = t_obs.astype(np.float32)
            ds["library_signal_time"].attrs["units"] = "s"
            ds["library_signal_time"].attrs["long_name"] = "Time"

            rcv_signal_library = np.empty(tuple(ds.dims[d] for d in signal_library_dim))

        # Time domain signal
        s_obs = s_obs[:, 0, :].T
        s_obs = s_obs.reshape(
            ds.dims["y"], ds.dims["x"], ds.dims["library_signal_time"]
        )

        rcv_signal_library[i_obs, :] = s_obs

        # Free memory
        del s_obs, rr_from_obs_flat

    return ds, rcv_signal_library, grid_pressure_field


def init_library_dataset(grid_x, grid_y, x_obs, y_obs, isotropic_env=True):

    # Init Dataset
    n_obs = len(x_obs)
    xx, yy = np.meshgrid(grid_x, grid_y, sparse=True)

    # Compute range from each receiver
    rr_obs = np.array(
        [
            np.sqrt((xx - x_obs[i_obs]) ** 2 + (yy - y_obs[i_obs]) ** 2)
            for i_obs in range(n_obs)
        ]
    )

    ds = xr.Dataset(
        data_vars=dict(
            x_obs=(["idx_obs"], x_obs),
            y_obs=(["idx_obs"], y_obs),
            r_from_obs=(["idx_obs", "y", "x"], rr_obs),
        ),
        coords=dict(
            x=grid_x,
            y=grid_y,
            idx_obs=np.arange(n_obs),
        ),
        attrs=dict(
            title="Verlinden simulation with simple environment",
            dx=np.diff(grid_x)[0],
            dy=np.diff(grid_y)[0],
        ),
    )

    if not isotropic_env:
        # Compute angles relatives to each receiver
        theta_obs = np.array(
            [np.arctan2(yy - y_obs[i_obs], xx - x_obs[i_obs]) for i_obs in range(n_obs)]
        )
        ds["theta_obs"] = (["idx_obs", "y", "x"], theta_obs)

        # Build list of angles to be used in kraken
        dmax = ds.r_from_obs.max(dim=["y", "x", "idx_obs"]).round(0).values
        delta = min(ds.dx, ds.dy)
        dtheta_th = np.arctan(delta / dmax)
        list_theta_th = np.arange(ds.theta_obs.min(), ds.theta_obs.max(), dtheta_th)

        theta_propa = np.empty(ds.theta_obs.shape)
        for i_obs in range(n_obs):
            for theta in list_theta_th:
                closest_points_idx = (
                    np.abs(ds.theta_obs.sel(idx_obs=i_obs) - theta) < dtheta_th / 2
                )
                theta_propa[i_obs, closest_points_idx] = theta

        # Add theta_propa to dataset
        ds["theta_propa"] = (["idx_obs", "y", "x"], theta_propa)
        # Remove theta_obs
        ds = ds.drop_vars("theta_obs")

    # Set attributes
    var_unit_mapping = {
        "m": [
            "x_obs",
            "y_obs",
            "x",
            "y",
            "r_from_obs",
        ],
        "": ["idx_obs"],
    }
    for unit in var_unit_mapping.keys():
        for var in var_unit_mapping[unit]:
            ds[var].attrs["units"] = unit

    ds["x_obs"].attrs["long_name"] = "x_obs"
    ds["y_obs"].attrs["long_name"] = "y_obs"
    ds["r_from_obs"].attrs["long_name"] = "Range from receiver"
    ds["x"].attrs["long_name"] = "x"
    ds["y"].attrs["long_name"] = "y"
    ds["idx_obs"].attrs["long_name"] = "Receiver index"

    # TODO : need to be changed in case of multiple receivers couples
    ds["delay_obs"] = ds.r_from_obs / C0

    # Build OBS pairs
    obs_pairs = []
    for i in ds.idx_obs.values:
        for j in range(i + 1, ds.idx_obs.values[-1] + 1):
            obs_pairs.append((i, j))
    ds.coords["idx_obs_pairs"] = np.arange(len(obs_pairs))
    ds.coords["idx_obs_in_pair"] = np.arange(2)
    ds["obs_pairs"] = (["idx_obs_pairs", "idx_obs_in_pair"], obs_pairs)

    return ds


def check_waveguide_cutoff(
    testcase, kraken_env, library_src, max_range_m, dt, sig_type
):
    fc = waveguide_cutoff_freq(max_depth=kraken_env.bathy.bathy_depth.min())
    propagating_freq = library_src.positive_freq[library_src.positive_freq > fc]
    if propagating_freq.size != library_src.kraken_freq.size:
        min_waveguide_depth = kraken_env.bathy.bathy_depth.min()
        library_src = init_library_src(dt, min_waveguide_depth, sig_type=sig_type)
        kraken_env, kraken_flp = testcase(
            freq=library_src.kraken_freq,
            min_waveguide_depth=min_waveguide_depth,
            max_range_m=max_range_m,
        )

    return kraken_env, kraken_flp, library_src


def add_noise_to_dataset(library_dataset, snr_dB):
    ds = library_dataset
    for i_obs in tqdm(
        ds.idx_obs, bar_format=BAR_FORMAT, desc="Add noise to received signal"
    ):
        if snr_dB is not None:
            # Add noise to received signal
            ds.rcv_signal_library.loc[dict(idx_obs=i_obs)] = add_noise_to_signal(
                ds.rcv_signal_library.sel(idx_obs=i_obs).values, snr_dB=snr_dB
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
    library_corr_dim = ["idx_obs_pairs", "y", "x", "library_corr_lags"]
    library_corr = np.empty(tuple(ds.dims[d] for d in library_corr_dim))

    # May be way faster with a FFT based approach
    ns = ds.dims["library_signal_time"]
    for i_pair in tqdm(
        range(ds.dims["idx_obs_pairs"]),
        bar_format=BAR_FORMAT,
        desc="Derive correlation vector for each grid pixel",
    ):
        rcv_pair = ds.obs_pairs.isel(idx_obs_pairs=i_pair)
        for i_x in tqdm(
            range(ds.dims["x"]),
            bar_format=BAR_FORMAT,
            desc="Scanning x axis",
            leave=False,
        ):
            for i_y in tqdm(
                range(ds.dims["y"]),
                bar_format=BAR_FORMAT,
                desc="Scanning y axis",
                leave=False,
            ):
                s0 = ds.rcv_signal_library.sel(
                    idx_obs=rcv_pair[0], x=ds.x.isel(x=i_x), y=ds.y.isel(y=i_y)
                )
                s1 = ds.rcv_signal_library.sel(
                    idx_obs=rcv_pair[1], x=ds.x.isel(x=i_x), y=ds.y.isel(y=i_y)
                )
                corr_01 = signal.correlate(s0, s1)
                n0 = corr_01.shape[0] // 2
                autocorr0 = signal.correlate(s0, s0)
                autocorr1 = signal.correlate(s1, s1)
                corr_01 /= np.sqrt(autocorr0[n0] * autocorr1[n0])

                library_corr[i_pair, i_y, i_x, :] = corr_01

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


def add_event_to_dataset(
    library_dataset,
    grid_pressure_field,
    kraken_env,
    event_src,
    event_t,
    x_event_t,
    y_event_t,
    z_event,
    interp_src_pos_on_grid=False,
    snr_dB=None,
):
    ds = library_dataset

    r_event_t = [
        np.sqrt(
            (x_event_t - ds.x_obs.sel(idx_obs=i_obs).values) ** 2
            + (y_event_t - ds.y_obs.sel(idx_obs=i_obs).values) ** 2
        )
        for i_obs in range(ds.dims["idx_obs"])
    ]

    ds.coords["event_signal_time"] = []
    ds.coords["src_trajectory_time"] = event_t.astype(np.float32)

    ds["x_ship"] = (["src_trajectory_time"], x_event_t.astype(np.float32))
    ds["y_ship"] = (["src_trajectory_time"], y_event_t.astype(np.float32))
    ds["r_obs_ship"] = (
        ["idx_obs", "src_trajectory_time"],
        np.array(r_event_t).astype(np.float32),
    )

    ds["event_signal_time"].attrs["units"] = "s"
    ds["src_trajectory_time"].attrs["units"] = "s"

    ds["x_ship"].attrs["long_name"] = "x_ship"
    ds["y_ship"].attrs["long_name"] = "y_ship"
    ds["r_obs_ship"].attrs["long_name"] = "Range from receiver to source"
    ds["event_signal_time"].attrs["units"] = "Time"
    ds["src_trajectory_time"].attrs["long_name"] = "Time"

    if interp_src_pos_on_grid:
        ds["x_ship"] = ds.x.sel(x=ds.x_ship, method="nearest")
        ds["y_ship"] = ds.y.sel(y=ds.y_ship, method="nearest")
        ds["r_obs_ship"].values = [
            np.sqrt(
                (ds.x_ship - ds.x_obs.sel(idx_obs=i_obs)) ** 2
                + (ds.y_ship - ds.y_obs.sel(idx_obs=i_obs)) ** 2
            )
            for i_obs in range(ds.dims["idx_obs"])
        ]
        ds.attrs["source_positions"] = "Interpolated on grid"
        ds.attrs["src_pos"] = "on_grid"
    else:
        ds.attrs["source_positions"] = "Not interpolated on grid"
        ds.attrs["src_pos"] = "not_on_grid"

    signal_event_dim = ["idx_obs", "src_trajectory_time", "event_signal_time"]

    # Derive received signal for successive positions of the ship
    for i_obs in tqdm(
        range(ds.dims["idx_obs"]),
        bar_format=BAR_FORMAT,
        desc="Derive received signal for successive positions of the ship",
    ):
        delay_to_apply_ship = (
            ds.delay_obs.min(dim="idx_obs")
            .sel(x=ds.x_ship, y=ds.y_ship, method="nearest")
            .values.flatten()
        )

        (
            t_obs,
            s_obs,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=kraken_env.shd_fpath,
            broadband_pressure_field=grid_pressure_field,
            frequencies=event_src.kraken_freq,
            source=event_src,
            rcv_range=ds.r_obs_ship.sel(idx_obs=i_obs).values,
            rcv_depth=[z_event],
            apply_delay=True,
            delay=delay_to_apply_ship,
            minimum_waveguide_depth=kraken_env.bathy.bathy_depth.min(),
        )

        if i_obs == 0:
            ds["event_signal_time"] = t_obs.astype(np.float32)
            rcv_signal_event = np.empty(tuple(ds.dims[d] for d in signal_event_dim))

        rcv_signal_event[i_obs, :] = s_obs[:, 0, :].T

        # Free memory
        del t_obs, s_obs, Pos

    ds["rcv_signal_event"] = (
        ["idx_obs", "src_trajectory_time", "event_signal_time"],
        rcv_signal_event.astype(np.float32),
    )

    ds = add_noise_to_event(ds, snr_dB=snr_dB)
    ds = add_event_correlation(ds)

    return ds


def add_noise_to_event(library_dataset, snr_dB):
    ds = library_dataset
    for i_obs in tqdm(
        ds.idx_obs, bar_format=BAR_FORMAT, desc="Add noise to event signal"
    ):
        if snr_dB is not None:
            # Add noise to received signal
            ds.rcv_signal_event.loc[dict(idx_obs=i_obs)] = add_noise_to_signal(
                ds.rcv_signal_event.sel(idx_obs=i_obs).values, snr_dB
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
    event_corr_dim = ["idx_obs_pairs", "src_trajectory_time", "event_corr_lags"]
    event_corr = np.empty(tuple(ds.dims[d] for d in event_corr_dim))

    for i_ship in tqdm(
        range(ds.dims["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Derive correlation vector for each ship position",
    ):
        for i_pair, rcv_pair in enumerate(ds.obs_pairs):
            s0 = ds.rcv_signal_event.sel(idx_obs=rcv_pair[0]).isel(
                src_trajectory_time=i_ship
            )
            s1 = ds.rcv_signal_event.sel(idx_obs=rcv_pair[1]).isel(
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
            for i_x in range(sig.shape[0]):
                for i_y in range(sig.shape[1]):
                    noise = np.random.normal(0, sigma_noise[i_x, i_y], sig.shape[-1])
                    sig[i_x, i_y, :] += noise

    return sig


def build_ambiguity_surf(ds, detection_metric):
    ambiguity_surface_dim = ["idx_obs_pairs", "src_trajectory_time", "y", "x"]
    ambiguity_surface = np.empty(tuple(ds.dims[d] for d in ambiguity_surface_dim))

    for i_ship in tqdm(
        range(ds.dims["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Build ambiguity surface",
    ):
        for i_pair in ds.idx_obs_pairs:
            lib_data = ds.library_corr.sel(idx_obs_pairs=i_pair)
            event_vector = ds.event_corr.sel(idx_obs_pairs=i_pair).isel(
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
    ds["detected_pos_x"] = ds.x.isel(x=ds.ambiguity_surface.argmax(dim=["x", "y"])["x"])
    ds["detected_pos_y"] = ds.y.isel(y=ds.ambiguity_surface.argmax(dim=["x", "y"])["y"])

    ds.attrs["detection_metric"] = detection_metric

    return ds


def init_library_src(dt, min_waveguide_depth, sig_type="pulse"):
    if sig_type == "ship":
        library_src_sig, t_library_src_sig = ship_noise(T=dt)

    elif sig_type == "pulse":
        library_src_sig, t_library_src_sig = pulse(T=dt, f=25, fs=100)

    elif sig_type == "pulse_train":
        library_src_sig, t_library_src_sig = pulse_train(T=dt, f=25, fs=100)

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


def init_event_src_traj(src_info):
    """Init the source trajectory given initial position, speed and duration"""
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
    grid_info["grid_lons"] = grid_lons
    grid_info["grid_lats"] = grid_lats
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
        f"Number of grid points: {len(grid_info['grid_lons']) * len(grid_info['grid_lats'])}",
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
