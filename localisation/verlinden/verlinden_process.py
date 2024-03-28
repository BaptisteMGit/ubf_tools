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


from localisation.verlinden.verlinden_utils import (
    add_correlation_to_dataset,
    add_noise_to_dataset,
    build_ambiguity_surf,
    get_populated_path,
    init_event_src_traj,
    init_grid_around_event_src_traj,
    init_library_src,
    init_library_dataset,
    init_event_dataset,
    populate_isotropic_env,
    populate_anistropic_env,
    check_waveguide_cutoff,
    get_max_kraken_range,
    get_dist_between_rcv,
    print_simulation_info,
    add_event_isotropic_env,
    add_event_anisotropic_env,
    plot_src,
)


def populate_grid(
    library_src,
    grid_info,
    rcv_info,
    src_info,
    testcase,
):
    """
    Populate grid with received signal for isotropic environment. This function is used to generate the library of received signals for the Verlinden method.
    In this specific case, the environment is isotropic and the received signal only depends on the range from the source to the receiver.
    """

    # Init Dataset
    ds = init_library_dataset(
        grid_info,
        rcv_info,
        isotropic_env=testcase.isotropic,
    )

    signal_library_dim = ["idx_rcv", "lat", "lon", "library_signal_time"]

    # Switch between isotropic and anisotropic environment
    if testcase.isotropic:
        # TODO update with testcase object properties
        ds, rcv_signal_library, grid_pressure_field = populate_isotropic_env(
            ds, library_src, signal_library_dim, testcase
        )
        kraken_grid = None  # TODO : update this ?
    else:
        ds, rcv_signal_library, grid_pressure_field, kraken_grid = (
            populate_anistropic_env(
                ds, library_src, signal_library_dim, testcase, rcv_info, src_info
            )
        )

    ds["rcv_signal_library"] = (
        signal_library_dim,
        rcv_signal_library.astype(np.float32),
    )
    ds["rcv_signal_library"].attrs["long_name"] = r"$s_{i}$"

    ds.attrs["fullpath_populated"] = get_populated_path(
        grid_info,
        kraken_env=testcase.env,
        src_signal_type=src_info["signal_type"],
    )
    # ds.x.values, ds.y.values, kraken_env, library_src.name

    if not os.path.exists(os.path.dirname(ds.fullpath_populated)):
        os.makedirs(os.path.dirname(ds.fullpath_populated))

    ds.to_netcdf(ds.fullpath_populated)

    return ds, grid_pressure_field, kraken_grid


def add_event_to_dataset(
    library_dataset,
    grid_pressure_field,
    kraken_grid,
    kraken_env,
    event_src,
    src_info,
    rcv_info,
    snr_dB=None,
    isotropic_env=True,
    interp_src_pos_on_grid=False,
):
    ds = library_dataset

    ds = init_event_dataset(
        ds, src_info, rcv_info, interp_src_pos_on_grid=interp_src_pos_on_grid
    )

    signal_event_dim = ["idx_rcv", "src_trajectory_time", "event_signal_time"]

    if isotropic_env:
        ds = add_event_isotropic_env(
            ds=ds,
            snr_dB=snr_dB,
            event_src=event_src,
            kraken_env=kraken_env,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
        )
    else:

        ds = add_event_anisotropic_env(
            ds=ds,
            snr_dB=snr_dB,
            event_src=event_src,
            kraken_grid=kraken_grid,
            signal_event_dim=signal_event_dim,
            grid_pressure_field=grid_pressure_field,
        )

    return ds


def verlinden_main(
    testcase,
    src_info,
    grid_info,
    rcv_info,
    snr,
    detection_metric,
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

    grid_pressure_field = None  # Init to None to avoid redundancy
    for snr_i in snr:
        if snr_i is None:
            snr_tag = "noiseless"
            snr_msg = "Performing localisation process without noise"
        else:
            snr_tag = f"snr{snr_i}dB"
            snr_msg = f"Performing localisation process with additive gaussian white noise SNR = {snr_i}dB"
        print("## " + snr_msg + " ##")

        populated_path = get_populated_path(
            grid_info, kraken_env=testcase.env, src_signal_type=src_info["signal_type"]
        )

        event_in_dataset = False
        complete_dataset_loaded = False
        for det_metric in detection_metric:
            det_msg = f"Detection metric: {det_metric}"
            print("# " + det_msg + " #")

            if (
                os.path.exists(populated_path)
                and not complete_dataset_loaded
                and grid_pressure_field is not None
            ):
                ds_library = xr.open_dataset(populated_path)
            elif not os.path.exists(populated_path) or grid_pressure_field is None:
                # Populate grid with received signal
                ds_library, grid_pressure_field, kraken_grid = populate_grid(
                    library_src,
                    grid_info,
                    rcv_info,
                    src_info,
                    testcase=testcase,
                )

            # 10/01/2024 No more 1 save/snr to save memory
            if not complete_dataset_loaded:
                # Add noise to dataset
                ds_library = add_noise_to_dataset(ds_library, snr_dB=snr_i)

                # Derive correlation vector for the entire grid
                ds_library = add_correlation_to_dataset(ds_library)

                # Switch flag to avoid redundancy
                complete_dataset_loaded = True

            event_src = library_src
            event_src.z_src = src_info["depth"]
            if not event_in_dataset:
                ds = add_event_to_dataset(
                    library_dataset=ds_library,
                    grid_pressure_field=grid_pressure_field,
                    kraken_grid=kraken_grid,
                    kraken_env=testcase.env,
                    event_src=event_src,
                    src_info=src_info,
                    rcv_info=rcv_info,
                    snr_dB=snr_i,
                    isotropic_env=testcase.isotropic,
                )
                event_in_dataset = True

            ds = build_ambiguity_surf(ds, det_metric)

            # Build path to save dataset and corresponding path to save analysis results produced later on
            ds.attrs["fullpath_output"] = os.path.join(
                VERLINDEN_OUTPUT_FOLDER,
                testcase.env.filename,
                library_src.name,
                ds.src_pos,
                det_metric,
                f"output_{det_metric}_{snr_tag}.nc",
            )
            ds.attrs["fullpath_analysis"] = os.path.join(
                VERLINDEN_ANALYSIS_FOLDER,
                testcase.env.filename,
                library_src.name,
                ds.src_pos,
                det_metric,
                snr_tag,
            )

            if not os.path.exists(os.path.dirname(ds.fullpath_output)):
                os.makedirs(os.path.dirname(ds.fullpath_output))

            if not os.path.exists(ds.fullpath_analysis):
                os.makedirs(ds.fullpath_analysis)

            ds.to_netcdf(ds.fullpath_output)

        ds = ds.drop_vars("event_signal_time")
    print(f"### Verlinden simulation process done ###")

    simu_folder = os.path.dirname(testcase.env.env_fpath)

    return simu_folder, testcase.env.filename


if __name__ == "__main__":
    v_ship = 50 / 3.6  # m/s
    src_info = dict(
        x_pos=[-1000, 2500],
        y_pos=[3000, 2000],
        v_src=v_ship,
        nmax_ship=100,
        src_signal_type="pulse_train",
        z_src=5,
        on_grid=False,
    )

    grid_info = dict(
        Lx=5 * 1e3,
        Ly=5 * 1e3,
        dx=100,
        dy=100,
    )

    obs_info = dict(
        x_obs=[0, 1500],
        y_obs=[0, 0],
    )

    snr = [-30, 0]
    detection_metric = ["intercorr0"]
    # detection_metric = ["intercorr0", "lstsquares", "hilbert_env_intercorr0"]

    depth = 150  # Depth m
    env_fname = "verlinden_1_test_case"
    env_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"
    verlinden_main(
        env_fname=env_fname,
        env_root=env_root,
        src_info=src_info,
        grid_info=grid_info,
        rcv_info=obs_info,
        snr=snr,
        detection_metric=detection_metric,
        depth_max=depth,
    )
