#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   verlinden_process.py
@Time    :   2024/03/12 13:22:32
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

from localisation.verlinden.verlinden_path import (
    VERLINDEN_OUTPUT_FOLDER,
    VERLINDEN_ANALYSIS_FOLDER,
)

from localisation.verlinden.verlinden_utils import *


def populate_grid(
    library_src,
    grid_info,
    rcv_info,
    src_info,
    testcase,
    snrs_dB,
    n_noise_realisations,
    similarity_metrics,
):
    """
    Populate grid with received signal.
    This function is used to generate the library of received signals.
    This library is meant to be used as a reference to compare the received signal of an event.

    Parameters
    ----------
    library_src : LibrarySrc
        LibrarySrc object containing the source signal
    grid_info : dict
        Dictionary containing the grid information
    rcv_info : dict
        Dictionary containing the receiver information
    src_info : dict
        Dictionary containing the source information
    testcase : TestCase
        TestCase object containing the environment information
    snrs_dB : list
        List of signal to noise ratio in dB
    n_noise_realisations : int
        Number of noise realisations
    similarity_metrics : list
        List of similarity metrics to be computed

    Returns
    -------
    xr_dataset : xarray.Dataset
        Dataset containing the populated grid
    grid_pressure_field : np.ndarray
        Pressure field of the grid
    kraken_grid : np.ndarray
        Kraken grid


    """

    # Init Dataset
    xr_dataset = init_library_dataset(
        grid_info=grid_info,
        rcv_info=rcv_info,
        snrs_dB=snrs_dB,
        n_noise_realisations=n_noise_realisations,
        similarity_metrics=similarity_metrics,
        isotropic_env=testcase.isotropic,
    )

    signal_library_dim = ["idx_rcv", "lat", "lon", "library_signal_time"]

    # Switch between isotropic and anisotropic environment
    if testcase.isotropic:
        # TODO update with testcase object properties
        xr_dataset, rcv_signal_library, grid_pressure_field = populate_isotropic_env(
            xr_dataset, library_src, signal_library_dim, testcase
        )
        kraken_grid = None  # TODO : update this ?
    else:
        xr_dataset, rcv_signal_library, grid_pressure_field, kraken_grid = (
            populate_anistropic_env(
                xr_dataset=xr_dataset,
                library_src=library_src,
                signal_library_dim=signal_library_dim,
                testcase=testcase,
                rcv_info=rcv_info,
                src_info=src_info,
            )
        )

    xr_dataset["rcv_signal_library"] = (
        signal_library_dim,
        rcv_signal_library.astype(np.float32),
    )

    # Save noise free rcv signal
    xr_dataset["rcv_signal_library_noise_free"] = (
        signal_library_dim,
        rcv_signal_library.astype(np.float32),
    )

    # Expand dims to add snr dimension to the rcv_signal_library dataarray
    # Using expand_dims to avoid loading the entire dataset in memory -> overflow
    xr_dataset["rcv_signal_library"] = xr_dataset.rcv_signal_library.expand_dims(
        dim={"snr": xr_dataset.sizes["snr"]}
    ).assign_coords({"snr": xr_dataset.snr})

    xr_dataset["rcv_signal_library"].attrs["long_name"] = r"$s_{i}$"

    xr_dataset.attrs["fullpath_populated"] = get_populated_path(
        grid_info,
        kraken_env=testcase.env,
        src_signal_type=src_info["signal_type"],
        ext="zarr",
    )

    # Initialize correlation array
    init_corr_library(xr_dataset)

    if not os.path.exists(os.path.dirname(xr_dataset.fullpath_populated)):
        os.makedirs(os.path.dirname(xr_dataset.fullpath_populated))

    # Save populated dataset to zarr format (for very large arrays and parallel I/O)
    print(f"Saving populated dataset to {xr_dataset.fullpath_populated} ...")
    xr_dataset.to_zarr(xr_dataset.fullpath_populated, mode="w")
    xr_dataset.close()
    print("Dataset saved")

    # Open mutable zarr dataset
    xr_dataset = xr.open_zarr(xr_dataset.fullpath_populated)

    # populated_dataarray = xr_dataset.rcv_signal_library
    # populated_dataarray.to_netcdf(xr_dataset.fullpath_populated)

    return xr_dataset, grid_pressure_field, kraken_grid


def add_event_to_dataset(
    xr_dataset,
    grid_pressure_field,
    kraken_grid,
    kraken_env,
    event_src,
    src_info,
    rcv_info,
    # init_event=True,
    # snr_dB=None,
    isotropic_env=True,
    interp_src_pos_on_grid=False,
):
    """
    Add event to dataset.

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        Dataset containing the populated grid
    grid_pressure_field : np.ndarray
        Pressure field of the grid
    kraken_grid : np.ndarray
        Kraken grid
    kraken_env : KrakenEnv
        KrakenEnv object containing the environment information
    event_src : LibrarySrc
        LibrarySrc object containing the source signal
    src_info : dict
        Dictionary containing the source information
    rcv_info : dict
        Dictionary containing the receiver information
    init_event : bool
        Boolean to initialize the event
    snr_dB : float
        Signal to noise ratio in dB
    isotropic_env : bool
        Boolean to switch between isotropic and anisotropic environment
    interp_src_pos_on_grid : bool
        Boolean to interpolate the source position on the grid

    Returns
    -------
    None

    """

    # if init_event:
    init_event_dataset(
        xr_dataset,
        src_info,
        rcv_info,
        interp_src_pos_on_grid=interp_src_pos_on_grid,
    )

    signal_event_dim = ["snr", "idx_rcv", "src_trajectory_time", "event_signal_time"]

    if isotropic_env:
        add_event_isotropic_env(
            xr_dataset=xr_dataset,
            # snr_dB=snr_dB,
            event_src=event_src,
            kraken_env=kraken_env,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
            # init_corr=init_event,
        )
    else:

        add_event_anisotropic_env(
            xr_dataset=xr_dataset,
            # snr_dB=snr_dB,
            event_src=event_src,
            kraken_grid=kraken_grid,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
        )

    lags_idx = signal.correlation_lags(
        xr_dataset.sizes["event_signal_time"], xr_dataset.sizes["event_signal_time"]
    )
    lags = lags_idx * xr_dataset.event_signal_time.diff("event_signal_time").values[0]

    xr_dataset.coords["event_corr_lags"] = lags
    xr_dataset["event_corr_lags"].attrs["units"] = "s"
    xr_dataset["event_corr_lags"].attrs["long_name"] = "Correlation lags"

    event_corr_dim = ["snr", "idx_rcv_pairs", "src_trajectory_time", "event_corr_lags"]
    dummy_array = np.empty(tuple(xr_dataset.sizes[d] for d in event_corr_dim))
    xr_dataset["event_corr"] = (event_corr_dim, dummy_array)
    xr_dataset["event_corr"].attrs["long_name"] = r"$R_{ij}^{l}(\tau)$"


def load_noiseless_data(xr_dataset, populated_path):
    """
    Load noiseless data.

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        Dataset containing the populated grid
    populated_path : str
        Path to the populated grid

    Returns
    -------
    None

    """
    noiseless_dataarray = xr.open_dataarray(populated_path)
    xr_dataset["rcv_signal_library"] = noiseless_dataarray


def verlinden_main(
    testcase,
    src_info,
    grid_info,
    rcv_info,
    snr,
    similarity_metrics,
    nb_noise_realisations_per_snr=10,
    dt=None,
):
    """
    Main function to run the Verlinden simulation process.

    Parameters
    ----------
    testcase : TestCase
        TestCase object containing the environment information
    src_info : dict
        Dictionary containing the source information
    grid_info : dict
        Dictionary containing the grid information
    rcv_info : dict
        Dictionary containing the receiver information
    snr : list
        List of signal to noise ratio in dB
    similarity_metrics : list
        List of similarity metrics to be computed
    nb_noise_realisations_per_snr : int
        Number of noise realisations per SNR
    dt : float
        Time step

    Returns
    -------
    simu_folder : str
        Simulation folder
    env_filename : str
        Environment filename

    """
    if dt is None:
        dt = (
            min(grid_info["dx"], grid_info["dy"]) / src_info["speed"]
        )  # Minimum time spent by the source in a single grid cell (s)

    # Initialize source
    min_waveguide_depth = 150  # Dummy value updated once bathy is loaded
    library_src = init_src(
        dt, min_waveguide_depth=min_waveguide_depth, src_info=src_info["library"]
    )

    event_src = init_src(
        dt, min_waveguide_depth=min_waveguide_depth, src_info=src_info["event"]
    )

    # Plot source signal and spectrum
    plot_src(library_src, testcase)

    # Define ship trajectory
    init_event_src_traj(src_info, dt)

    # Define grid around the src positions
    init_grid_around_event_src_traj(src_info, grid_info)

    # Derive max distance to be used in kraken for each receiver
    get_max_kraken_range(rcv_info, grid_info)

    # Derive dist between rcvs
    get_dist_between_rcv(rcv_info)

    # Display usefull information
    print_simulation_info(testcase, src_info, rcv_info, grid_info)

    # Define environment
    max_range_m = np.max(rcv_info["max_kraken_range_m"])
    testcase_varin = dict(
        freq=library_src.kraken_freq,
        max_range_m=max_range_m,
        min_waveguide_depth=min_waveguide_depth,
    )
    testcase.update(testcase_varin)

    # Assert kraken freq set with correct min_depth (otherwise postprocess will fail)
    library_src = check_waveguide_cutoff(
        testcase=testcase,
        src=library_src,
        dt=dt,
        src_info=src_info["library"],
    )

    event_src = check_waveguide_cutoff(
        testcase=testcase,
        src=event_src,
        dt=dt,
        src_info=src_info["event"],
    )

    # Populate grid with received signal
    verlinden_dataset, grid_pressure_field, kraken_grid = populate_grid(
        library_src,
        grid_info,
        rcv_info,
        src_info,
        snrs_dB=snr,
        testcase=testcase,
        n_noise_realisations=nb_noise_realisations_per_snr,
        similarity_metrics=similarity_metrics,
    )
    # Add source library/event to dataset
    add_src_to_dataset(verlinden_dataset, library_src, event_src, src_info)

    # Add noise to dataset
    add_noise_to_dataset(verlinden_dataset)
    # Derive correlation vector for the entire grid
    verlinden_dataset = add_correlation_to_dataset(verlinden_dataset)

    event_src.z_src = src_info["depth"]
    add_event_to_dataset(
        xr_dataset=verlinden_dataset,
        grid_pressure_field=grid_pressure_field,
        kraken_grid=kraken_grid,
        kraken_env=testcase.env,
        event_src=event_src,
        src_info=src_info,
        rcv_info=rcv_info,
        isotropic_env=testcase.isotropic,
    )

    for idx_snr, snr_i in enumerate(snr):

        # Loop over different realisation of noise for a given SNR
        for i in range(nb_noise_realisations_per_snr):
            print(f"## Monte Carlo iteration {i+1}/{nb_noise_realisations_per_snr} ##")

            # Add event to dataset
            add_noise_to_event(xr_dataset=verlinden_dataset, snr_dB=snr_i)
            add_event_correlation(xr_dataset=verlinden_dataset, snr_dB=snr_i)

            for i_sim_metric in range(len(similarity_metrics)):
                init_amb_surf = i == 0 and idx_snr == 0 and i_sim_metric == 0

                if init_amb_surf:
                    build_output_save_path(
                        verlinden_dataset,
                        output_folder=VERLINDEN_OUTPUT_FOLDER,
                        analysis_folder=VERLINDEN_ANALYSIS_FOLDER,
                        env_filename=testcase.env.filename,
                        src_name=library_src.name,
                    )

                    verlinden_dataset = init_ambiguity_surface(verlinden_dataset)

                build_ambiguity_surf(
                    verlinden_dataset,
                    snr_dB=snr_i,
                    idx_similarity_metric=i_sim_metric,
                    i_noise=i,
                    init_amb_surf=init_amb_surf,
                )

        # Save dataset
        # xr_dataset.to_netcdf(xr_dataset.fullpath_output)
        verlinden_dataset.to_zarr(verlinden_dataset.fullpath_output, mode="a")

    print(f"### Verlinden simulation process done ###")

    simu_folder = os.path.dirname(testcase.env.env_fpath)

    return simu_folder, testcase.env.filename


if __name__ == "__main__":
    pass
