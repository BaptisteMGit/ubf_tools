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


def populate_grid(
    library_src,
    z_src,
    kraken_env,
    kraken_flp,
    grid_x,
    grid_y,
    x_obs,
    y_obs,
):
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

    # Compute angles relatives to each receiver
    theta_obs = np.array(
        [np.arctan2(yy - y_obs[i_obs], xx - x_obs[i_obs]) for i_obs in range(n_obs)]
    )

    ds = xr.Dataset(
        data_vars=dict(
            x_obs=(["idx_obs"], x_obs),
            y_obs=(["idx_obs"], y_obs),
            r_from_obs=(["idx_obs", "y", "x"], rr_obs),
            theta_obs=(["idx_obs", "y", "x"], theta_obs),
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

    # Test : TODO : remove
    dtheta = np.abs(ds.theta_obs.isel(idx_obs=1).diff(dim="x")).min()
    print(f"Minimum angle step : {dtheta.values * 180 / np.pi}°")
    min_theta = ds.theta_obs.isel(idx_obs=1).min()
    max_theta = ds.theta_obs.isel(idx_obs=1).max()
    list_theta = np.arange(min_theta, max_theta, dtheta)
    print(f"Number of angles : {len(list_theta)}")

    # Plot angles
    # plt.figure()
    # ds.theta_obs.isel(idx_obs=0).plot()
    # plt.scatter(ds.x_obs.isel(idx_obs=0), ds.y_obs.isel(idx_obs=0))
    # plt.ylim([ds.y_obs.isel(idx_obs=0), ds.y.max() + 1e3])
    # plt.xlim([ds.x.min() - 3e3, ds.x.max() + 3e3])
    # plt.show()

    # Free memory
    del x_obs, y_obs, rr_obs, grid_x, grid_y

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
    delay_to_apply = ds.delay_obs.min(dim="idx_obs").values.flatten()

    # Build OBS pairs
    obs_pairs = []
    for i in ds.idx_obs.values:
        for j in range(i + 1, ds.idx_obs.values[-1] + 1):
            obs_pairs.append((i, j))
    ds.coords["idx_obs_pairs"] = np.arange(len(obs_pairs))
    ds.coords["idx_obs_in_pair"] = np.arange(2)
    ds["obs_pairs"] = (["idx_obs_pairs", "idx_obs_in_pair"], obs_pairs)

    signal_library_dim = ["idx_obs", "y", "x", "library_signal_time"]

    # Loop over angles to populate grid -> TODO

    # Run KRAKEN
    grid_pressure_field = runkraken(kraken_env, kraken_flp, library_src.kraken_freq)

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
            rcv_depth=[z_src],
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


def init_event_src_traj(x_begin, y_begin, x_end, y_end, v, dt):
    Dtot = np.sqrt((x_begin - x_end) ** 2 + (y_begin - y_end) ** 2)

    vx = v * (x_end - x_begin) / Dtot
    vy = v * (y_end - y_begin) / Dtot

    Ttot = Dtot / v + 3
    t = np.arange(0, Ttot - dt, dt)

    x_t = x_begin + vx * t
    y_t = y_begin + vy * t

    return x_t, y_t, t


def init_grid_around_event_src_traj(x_event_t, y_event_t, Lx, Ly, dx, dy):
    grid_x = np.arange(
        -Lx / 2 + min(x_event_t), Lx / 2 + max(x_event_t), dx, dtype=np.float32
    )
    grid_y = np.arange(
        -Ly / 2 + min(y_event_t), Ly / 2 + max(y_event_t), dy, dtype=np.float32
    )
    return grid_x, grid_y


def get_populated_path(grid_x, grid_y, kraken_env, src_signal_type):
    area_label = "_".join(
        [str(v) for v in [min(grid_x), max(grid_x), min(grid_y), max(grid_y)]]
    )
    populated_path = os.path.join(
        VERLINDEN_POPULATED_FOLDER,
        kraken_env.filename,
        f"populated_{area_label}_{src_signal_type}.nc",
    )
    return populated_path


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
        min(grid_info["dx"], grid_info["dy"]) / src_info["v_src"]
    )  # Minimum time spent by the source in a single grid box (s)

    print(f"### Starting Verlinden simulation process ... ###")
    print(
        f"    -> Grid properties: dx = {grid_info['dx']} m, dy = {grid_info['dy']} m, dt = {dt} s"
    )

    library_src = init_library_src(
        dt, min_waveguide_depth, sig_type=src_info["src_signal_type"]
    )

    x_ship_t, y_ship_t, t_ship = init_event_src_traj(
        src_info["x_pos"][0],
        src_info["y_pos"][0],
        src_info["x_pos"][1],
        src_info["y_pos"][1],
        src_info["v_src"],
        dt,
    )

    # Might be usefull to reduce the number of src positions to consider -> downsample
    nmax_ship = min(src_info["nmax_ship"], len(x_ship_t))
    ship_step = len(x_ship_t) // nmax_ship
    x_ship_t = x_ship_t[0::ship_step]
    y_ship_t = y_ship_t[0::ship_step]
    t_ship = t_ship[0::ship_step]

    # Define grid around the src positions
    grid_x, grid_y = init_grid_around_event_src_traj(
        x_ship_t,
        y_ship_t,
        grid_info["Lx"],
        grid_info["Ly"],
        grid_info["dx"],
        grid_info["dy"],
    )

    print(
        f"    -> Source (event) properties:\n \tFirst position = {src_info['x_pos'][0], src_info['y_pos'][0]}\n "
        f"\tLast position = {src_info['x_pos'][1], src_info['y_pos'][1]}\n \tNumber of positions = {nmax_ship}\n \tSource speed = {src_info['v_src']}m.s-1\n \tSignal type = {src_info['src_signal_type']}"
    )

    # Derive max distance to be used in kraken = grid diagonal
    max_range = np.round(
        np.sqrt((grid_x[-1] - grid_x[0]) ** 2 + (grid_y[-1] - grid_y[0]) ** 2), -2
    )

    # Define environment
    kraken_env, kraken_flp = testcase(
        freq=library_src.kraken_freq,
        min_waveguide_depth=min_waveguide_depth,
        max_range_m=max_range,
    )

    # Assert kraken freq set with correct min_depth (otherwise postprocess will fail)
    fc = waveguide_cutoff_freq(max_depth=kraken_env.bathy.bathy_depth.min())
    propagating_freq = library_src.positive_freq[library_src.positive_freq > fc]
    if propagating_freq.size != library_src.kraken_freq.size:
        min_waveguide_depth = kraken_env.bathy.bathy_depth.min()
        library_src = init_library_src(
            dt, min_waveguide_depth, sig_type=src_info["src_signal_type"]
        )
        kraken_env, kraken_flp = testcase(
            freq=library_src.kraken_freq,
            min_waveguide_depth=min_waveguide_depth,
            max_range_m=max_range,
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
                    src_info["z_src"],
                    kraken_env,
                    kraken_flp,
                    grid_x,
                    grid_y,
                    obs_info["x_obs"],
                    obs_info["y_obs"],
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
