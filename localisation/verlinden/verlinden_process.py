import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal

from tqdm import tqdm

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


from localisation.verlinden.verlinden_utils import (
    add_correlation_to_dataset,
    add_event_correlation,
    add_noise_to_dataset,
    add_noise_to_event,
    build_ambiguity_surf,
    get_populated_path,
    init_event_src_traj,
    init_grid_around_event_src_traj,
    init_library_src,
    init_library_dataset,
    populate_istropic_env,
    populate_anistropic_env,
    check_waveguide_cutoff,
    get_max_kraken_range,
    print_simulation_info,
)


def populate_grid(
    library_src,
    kraken_env,
    kraken_flp,
    grid_x,
    grid_y,
    x_obs,
    y_obs,
    isotropic_env=True,
    testcase=None,
):
    """
    Populate grid with received signal for isotropic environment. This function is used to generate the library of received signals for the Verlinden method.
    In this specific case, the environment is isotropic and the received signal only depends on the range from the source to the receiver.
    """

    # Init Dataset
    ds = init_library_dataset(
        grid_x=grid_x,
        grid_y=grid_y,
        x_obs=x_obs,
        y_obs=y_obs,
        isotropic_env=isotropic_env,
    )
    # Free memory
    del x_obs, y_obs, grid_x, grid_y

    signal_library_dim = ["idx_obs", "y", "x", "library_signal_time"]

    # Switch between isotropic and anisotropic environment
    if isotropic_env:
        ds, rcv_signal_library, grid_pressure_field = populate_istropic_env(
            ds, library_src, kraken_env, kraken_flp, signal_library_dim
        )
    else:
        ds, rcv_signal_library, grid_pressure_field = populate_anistropic_env(
            ds, library_src, signal_library_dim, testcase
        )

    ds["rcv_signal_library"] = (
        signal_library_dim,
        rcv_signal_library.astype(np.float32),
    )

    ds.attrs["fullpath_populated"] = get_populated_path(
        ds.x.values, ds.y.values, kraken_env, library_src.name
    )

    if not os.path.exists(os.path.dirname(ds.fullpath_populated)):
        os.makedirs(os.path.dirname(ds.fullpath_populated))

    ds.to_netcdf(ds.fullpath_populated)

    return ds, grid_pressure_field


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

        # t_obs, s_obs, Pos = postprocess_received_signal(
        #     shd_fpath=kraken_env.shd_fpath,
        #     source=event_src,
        #     rcv_range=ds.r_obs_ship.sel(idx_obs=i_obs).values,
        #     rcv_depth=[z_event],
        #     apply_delay=True,
        #     delay=delay_to_apply_ship,
        # )

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


def verlinden_main(
    testcase,
    min_waveguide_depth,
    src_info,
    grid_info,
    obs_info,
    snr,
    detection_metric,
):
    dt = (
        min(grid_info["dx"], grid_info["dy"]) / src_info["speed"]
    )  # Minimum time spent by the source in a single grid box (s)

    library_src = init_library_src(
        dt, min_waveguide_depth, sig_type=src_info["signal_type"]
    )

    t_ship = np.arange(0, src_info["max_nb_of_pos"] * dt, dt)
    init_event_src_traj(src_info)
    # x_ship_t, y_ship_t, t_ship = init_event_src_traj(
    #     src_info["x_pos"][0],
    #     src_info["y_pos"][0],
    #     src_info["x_pos"][1],
    #     src_info["y_pos"][1],
    #     src_info["v_src"],
    #     dt,
    # )

    # # Might be usefull to reduce the number of src positions to consider -> downsample
    # nmax_ship = min(src_info["nmax_ship"], len(x_ship_t))
    # ship_step = len(x_ship_t) // nmax_ship
    # x_ship_t = x_ship_t[0::ship_step]
    # y_ship_t = y_ship_t[0::ship_step]
    # t_ship = t_ship[0::ship_step]

    # Define grid around the src positions
    init_grid_around_event_src_traj(src_info, grid_info)

    # grid_x, grid_y = init_grid_around_event_src_traj(
    #     x_ship_t,
    #     y_ship_t,
    #     grid_info["Lx"],
    #     grid_info["Ly"],
    #     grid_info["dx"],
    #     grid_info["dy"],
    # )

    # Derive max distance to be used in kraken for each receiver
    get_max_kraken_range(obs_info, grid_info)

    # Display usefull information
    print_simulation_info(src_info, obs_info, grid_info)

    # print(
    #     f"    -> Source (event) properties:\n "
    #     f"\tFirst position = {src_info['x_pos'][0], src_info['y_pos'][0]}\n "
    #     f"\tLast position = {src_info['x_pos'][1], src_info['y_pos'][1]}\n "
    #     f"\tNumber of positions = {nmax_ship}\n "
    #     f"\tSource speed = {src_info['v_src']:.2f}m.s-1\n "
    #     f"\tSignal type = {src_info['src_signal_type']}"
    # )

    # Define environment
    max_range_m = np.max(obs_info["max_kraken_range_m"])
    kraken_env, kraken_flp = testcase(
        freq=library_src.kraken_freq,
        min_waveguide_depth=min_waveguide_depth,
        max_range_m=max_range_m,
    )

    # Assert kraken freq set with correct min_depth (otherwise postprocess will fail)
    kraken_env, kraken_flp, library_src = check_waveguide_cutoff(
        testcase,
        kraken_env,
        library_src,
        max_range_m,
        dt,
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
            grid_x, grid_y, kraken_env, src_info["src_signal_type"]
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
                ds_library, grid_pressure_field = populate_grid(
                    library_src,
                    kraken_env,
                    kraken_flp,
                    grid_x,
                    grid_y,
                    obs_info["x_obs"],
                    obs_info["y_obs"],
                    isotropic_env=False,
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
            if not event_in_dataset:
                ds = add_event_to_dataset(
                    library_dataset=ds_library,
                    grid_pressure_field=grid_pressure_field,
                    kraken_env=kraken_env,
                    event_src=event_src,
                    event_t=t_ship,
                    x_event_t=x_ship_t,
                    y_event_t=y_ship_t,
                    z_event=src_info["z_src"],
                    interp_src_pos_on_grid=src_info["on_grid"],
                    snr_dB=snr_i,
                )
                event_in_dataset = True

            ds = build_ambiguity_surf(ds, det_metric)

            # Build path to save dataset and corresponding path to save analysis results produced later on
            ds.attrs["fullpath_output"] = os.path.join(
                VERLINDEN_OUTPUT_FOLDER,
                kraken_env.filename,
                library_src.name,
                ds.src_pos,
                det_metric,
                f"output_{det_metric}_{snr_tag}.nc",
            )
            ds.attrs["fullpath_analysis"] = os.path.join(
                VERLINDEN_ANALYSIS_FOLDER,
                kraken_env.filename,
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

    simu_folder = os.path.dirname(kraken_env.env_fpath)

    return simu_folder, kraken_env.filename


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
        obs_info=obs_info,
        snr=snr,
        detection_metric=detection_metric,
        depth_max=depth,
    )
