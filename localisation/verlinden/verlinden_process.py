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
    VERLINDEN_POPULATED_FOLDER,
)

from localisation.verlinden.verlinden_utils import *


def populate_grid(
    library_src,
    grid_info,
    rcv_info,
    src_info,
    testcase,
    n_noise_realisations,
    similarity_metrics,
):
    """
    Populate grid with received signal for isotropic environment. This function is used to generate the library of received signals for the Verlinden method.
    In this specific case, the environment is isotropic and the received signal only depends on the range from the source to the receiver.
    """

    # Init Dataset
    xr_dataset = init_library_dataset(
        grid_info,
        rcv_info,
        n_noise_realisations,
        similarity_metrics,
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
                xr_dataset,
                library_src,
                signal_library_dim,
                testcase,
                rcv_info,
                src_info,
            )
        )

    xr_dataset["rcv_signal_library"] = (
        signal_library_dim,
        rcv_signal_library.astype(np.float32),
    )
    xr_dataset["rcv_signal_library"].attrs["long_name"] = r"$s_{i}$"

    xr_dataset.attrs["fullpath_populated"] = get_populated_path(
        grid_info,
        kraken_env=testcase.env,
        src_signal_type=src_info["signal_type"],
    )

    if not os.path.exists(os.path.dirname(xr_dataset.fullpath_populated)):
        os.makedirs(os.path.dirname(xr_dataset.fullpath_populated))

    populated_dataset = xr_dataset.copy(deep=True)
    populated_dataarray = populated_dataset.rcv_signal_library
    populated_dataarray.to_netcdf(xr_dataset.fullpath_populated)

    return xr_dataset, grid_pressure_field, kraken_grid


def add_event_to_dataset(
    xr_dataset,
    grid_pressure_field,
    kraken_grid,
    kraken_env,
    event_src,
    src_info,
    rcv_info,
    init_event=True,
    snr_dB=None,
    isotropic_env=True,
    interp_src_pos_on_grid=False,
):

    if init_event:
        init_event_dataset(
            xr_dataset,
            src_info,
            rcv_info,
            interp_src_pos_on_grid=interp_src_pos_on_grid,
        )

    signal_event_dim = ["idx_rcv", "src_trajectory_time", "event_signal_time"]

    if isotropic_env:
        add_event_isotropic_env(
            ds=xr_dataset,
            snr_dB=snr_dB,
            event_src=event_src,
            kraken_env=kraken_env,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
        )
    else:

        add_event_anisotropic_env(
            ds=xr_dataset,
            snr_dB=snr_dB,
            event_src=event_src,
            kraken_grid=kraken_grid,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
        )


def load_noiseless_data(xr_dataset, populated_path):
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
    if dt is None:
        dt = (
            min(grid_info["dx"], grid_info["dy"]) / src_info["speed"]
        )  # Minimum time spent by the source in a single grid cell (s)

    # Initialize source
    min_waveguide_depth = 150  # Dummy value updated once bathy is loaded
    library_src = init_library_src(
        dt, min_waveguide_depth=min_waveguide_depth, sig_type=src_info["signal_type"]
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
        library_src=library_src,
        dt=dt,
        sig_type=src_info["signal_type"],
    )

    # grid_pressure_field = None  # Init to None to avoid redundancy
    for idx_snr, snr_i in enumerate(snr):
        snr_tag = get_snr_tag(snr_dB=snr_i)

        populated_path = get_populated_path(
            grid_info, kraken_env=testcase.env, src_signal_type=src_info["signal_type"]
        )

        # Loop over different realisation of noise for a given SNR
        n_noise_realisations = nb_noise_realisations_per_snr  # TODO pass as param
        for i in range(n_noise_realisations):
            print(f"## Monte Carlo iteration {i+1}/{n_noise_realisations} ##")

            if i == 0:
                if idx_snr == 0:
                    # Populate grid with received signal at the very first iteration
                    verlinden_dataset, grid_pressure_field, kraken_grid = populate_grid(
                        library_src,
                        grid_info,
                        rcv_info,
                        src_info,
                        testcase=testcase,
                        n_noise_realisations=n_noise_realisations,
                        similarity_metrics=similarity_metrics,
                    )
                else:
                    # Load noiseless data
                    load_noiseless_data(verlinden_dataset, populated_path)

                # Add noise to dataset
                add_noise_to_dataset(verlinden_dataset, snr_dB=snr_i)

                # Derive correlation vector for the entire grid
                add_correlation_to_dataset(verlinden_dataset)

            # Add event to dataset
            init_event = (i == 0) and (
                idx_snr == 0
            )  # Init event only at the first iteration
            event_src = library_src
            event_src.z_src = src_info["depth"]
            add_event_to_dataset(
                xr_dataset=verlinden_dataset,
                grid_pressure_field=grid_pressure_field,
                kraken_grid=kraken_grid,
                kraken_env=testcase.env,
                event_src=event_src,
                src_info=src_info,
                rcv_info=rcv_info,
                snr_dB=snr_i,
                isotropic_env=testcase.isotropic,
                init_event=init_event,
            )

            for i_sim_metric in range(len(similarity_metrics)):

                build_ambiguity_surf(
                    verlinden_dataset, idx_similarity_metric=i_sim_metric, i_noise=i
                )

        # Save dataset
        save_dataset(
            verlinden_dataset,
            output_folder=VERLINDEN_OUTPUT_FOLDER,
            analysis_folder=VERLINDEN_ANALYSIS_FOLDER,
            env_filename=testcase.env.filename,
            src_name=library_src.name,
            snr_tag=snr_tag,
        )

    # verlinden_dataset = verlinden_dataset.drop_vars("event_signal_time")

    print(f"### Verlinden simulation process done ###")

    simu_folder = os.path.dirname(testcase.env.env_fpath)

    return simu_folder, testcase.env.filename


if __name__ == "__main__":
    pass
