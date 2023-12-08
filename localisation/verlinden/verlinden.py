import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal

from tqdm import tqdm

from cst import BAR_FORMAT
from misc import mult_along_axis
from signals import ship_noise, ship_spectrum
from localisation.verlinden.AcousticComponent import AcousticSource
from localisation.verlinden.RHUM_RHUM_env import (
    isotropic_ideal_env,
    rhum_rum_isotropic_env,
    verlinden_test_case_env,
)
from propa.kraken_toolbox.post_process import postprocess
from propa.kraken_toolbox.utils import runkraken, runfield, waveguide_cutoff_freq
from propa.kraken_toolbox.plot_utils import plotshd


def populate_grid(
    library_src, z_src, kraken_env, kraken_flp, grid_x, grid_y, x_obs, y_obs
):
    # Write env and flp files
    kraken_env.write_env()
    kraken_flp.write_flp()
    # Run KRAKEN
    os.chdir(kraken_env.root)
    runkraken(kraken_env.filename)

    # Init Dataset
    n_obs = len(x_obs)
    xx, yy = np.meshgrid(grid_x, grid_y, sparse=True)
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

    # Build OBS pairs
    obs_pairs = []
    for i in ds.idx_obs.values:
        for j in range(i + 1, ds.idx_obs.values[-1] + 1):
            obs_pairs.append((i, j))
    ds.coords["idx_obs_pairs"] = np.arange(len(obs_pairs))
    ds.coords["idx_obs_in_pair"] = np.arange(2)
    ds["obs_pairs"] = (["idx_obs_pairs", "idx_obs_in_pair"], obs_pairs)

    signal_library_dim = ["idx_obs", "y", "x", "library_signal_time"]

    for i_obs in tqdm(
        ds.idx_obs, bar_format=BAR_FORMAT, desc="Populate grid with received signal"
    ):
        rr_from_obs_flat = ds.r_from_obs.sel(idx_obs=i_obs).values.flatten()

        t_obs, s_obs, Pos = postprocess(
            shd_fpath=kraken_env.shd_fpath,
            source=library_src,
            # rcv_range=ds.r_obs_ship.sel(idx_obs=i_obs).values,
            rcv_range=rr_from_obs_flat,
            rcv_depth=[z_src],
        )
        if i_obs == 0:
            ds["library_signal_time"] = t_obs
            rcv_signal_library = np.empty(tuple(ds.dims[d] for d in signal_library_dim))

        s_obs = s_obs[:, 0, :].T
        s_obs = s_obs.reshape(
            ds.dims["y"], ds.dims["x"], ds.dims["library_signal_time"]
        )
        rcv_signal_library[i_obs, :] = s_obs

        # Free memory
        del s_obs, rr_from_obs_flat

    ds["rcv_signal_library"] = (
        signal_library_dim,
        rcv_signal_library,
    )
    ds.coords["library_corr_lags"] = signal.correlation_lags(
        ds.dims["library_signal_time"], ds.dims["library_signal_time"]
    )
    ds["library_corr_lags"].attrs["units"] = "s"
    ds["library_corr_lags"].attrs["long_name"] = "Correlation lags"

    # Derive cross_correlation vector for each grid pixel
    library_corr_dim = ["idx_obs_pairs", "y", "x", "library_corr_lags"]
    library_corr = np.empty(tuple(ds.dims[d] for d in library_corr_dim))

    # Could be way faster with a FFT based approach
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

                # # Plot autocorrelation
                # plt.figure()
                # plt.plot(
                #     signal.correlation_lags(len(s0), len(s0)),
                #     autocorr0,
                #     label="autocorr",
                # )
                # # plt.plot(t_obs, s0, label="s0")
                # plt.legend()

                # plt.figure()
                # plt.plot(
                #     signal.correlation_lags(len(s0), len(s0)),
                #     autocorr0,
                #     label="autocorr",
                # )
                # # plt.plot(t_obs, s1, label="s1")
                # plt.legend()
                # plt.show()

                # corr_01 /= np.max(corr_01)

                library_corr[i_pair, i_y, i_x, :] = corr_01

                del s0, s1, corr_01

    ds["library_corr"] = (library_corr_dim, library_corr)

    ds.to_netcdf(
        os.path.join(kraken_env.root, kraken_env.filename + "_populated" ".nc")
    )

    return ds


def add_event_to_dataset(
    library_dataset,
    event_src,
    event_t,
    x_event_t,
    y_event_t,
    z_event,
    interp_src_pos_on_grid=False,
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
    ds.coords["src_trajectory_time"] = event_t

    ds["x_ship"] = (["src_trajectory_time"], x_event_t)
    ds["y_ship"] = (["src_trajectory_time"], y_event_t)
    ds["r_obs_ship"] = (["idx_obs", "src_trajectory_time"], np.array(r_event_t))

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
        t_obs, s_obs, Pos = postprocess(
            shd_fpath=kraken_env.shd_fpath,
            source=event_src,
            rcv_range=ds.r_obs_ship.sel(idx_obs=i_obs).values,
            rcv_depth=[z_event],
        )
        if i_obs == 0:
            ds["event_signal_time"] = t_obs
            rcv_signal_event = np.empty(tuple(ds.dims[d] for d in signal_event_dim))

        rcv_signal_event[i_obs, :] = s_obs[:, 0, :].T

        # Free memory
        del t_obs, s_obs, Pos

    ds["rcv_signal_event"] = (
        ["idx_obs", "src_trajectory_time", "event_signal_time"],
        rcv_signal_event,
    )
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

    ds["event_corr"] = (event_corr_dim, event_corr)

    return ds


def build_ambiguity_surf(ds, detection_metric):
    ambiguity_surface_dim = ["idx_obs_pairs", "src_trajectory_time", "y", "x"]
    ambiguity_surface = np.empty(tuple(ds.dims[d] for d in ambiguity_surface_dim))

    for i_ship in tqdm(
        range(ds.dims["src_trajectory_time"]),
        bar_format=BAR_FORMAT,
        desc="Build ambiguity surface",
    ):
        for i_pair in ds.idx_obs_pairs:
            if detection_metric == "intercorr0":
                amb_surf = mult_along_axis(
                    ds.library_corr.sel(idx_obs_pairs=i_pair),
                    ds.event_corr.sel(idx_obs_pairs=i_pair).isel(
                        src_trajectory_time=i_ship
                    ),
                    axis=2,
                )
                autocorr_lib = np.sum(
                    ds.library_corr.sel(idx_obs_pairs=i_pair).values ** 2, axis=2
                )
                autocorr_event = np.sum(
                    ds.event_corr.sel(idx_obs_pairs=i_pair)
                    .isel(src_trajectory_time=i_ship)
                    .values
                    ** 2
                )
                norm = np.sqrt(autocorr_lib * autocorr_event)
                amb_surf = np.sum(amb_surf, axis=2) / norm  # Values in [-1, 1]
                amb_surf = (amb_surf + 1) / 2  # Values in [0, 1]
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

            elif detection_metric == "lstsquares":
                lib = ds.library_corr.sel(idx_obs_pairs=i_pair).values
                event = (
                    ds.event_corr.sel(idx_obs_pairs=i_pair)
                    .isel(src_trajectory_time=i_ship)
                    .values
                )
                diff = lib - event
                amb_surf = np.sum(diff**2, axis=2)  # Values in [0, max_diff**2]
                amb_surf = amb_surf / np.max(amb_surf)  # Values in [0, 1]
                amb_surf = (
                    1 - amb_surf
                )  # Revert order so that diff = 0 correspond to maximum of ambiguity surface
                ambiguity_surface[i_pair, i_ship, ...] = amb_surf

            elif detection_metric == "hilbert_env_intercorr0":
                lib_env = np.abs(
                    signal.hilbert(ds.library_corr.sel(idx_obs_pairs=i_pair))
                )
                event_env = np.abs(
                    signal.hilbert(
                        ds.event_corr.sel(idx_obs_pairs=i_pair).isel(
                            src_trajectory_time=i_ship
                        )
                    )
                )
                amb_surf = mult_along_axis(
                    lib_env,
                    event_env,
                    axis=2,
                )

                autocorr_lib = np.sum(lib_env**2, axis=2)
                autocorr_event = np.sum(event_env**2)
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


def init_library_src(dt, f0, fmax, depth):
    library_src_sig, t_library_src_sig = ship_noise()
    fs = 1 / (t_library_src_sig[1] - t_library_src_sig[0])
    nmax = int(fs * dt)
    library_src_sig = library_src_sig[0:nmax]
    t_library_src_sig = t_library_src_sig[0:nmax]

    library_src = AcousticSource(signal=library_src_sig, time=t_library_src_sig)
    library_src.set_kraken_freq(
        fmin=max(f0, waveguide_cutoff_freq(max_depth=depth) + 1), fmax=fmax, df=1
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


if __name__ == "__main__":
    # detection_metric = (
    #     "lstsquares"  # "intercorr0", "lstsquares", "hilbert_env_intercorr0"
    # )

    for detection_metric in ["intercorr0", "lstsquares", "hilbert_env_intercorr0"]:
        env_fname = "verlinden_1_test_case"
        env_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"

        dx = 100  # m
        dy = 100  # m
        v_ship = 50 / 3.6  # m/s
        dt = (
            min(dx, dy) / v_ship
        )  # Minimum time spent by the source in a single grid box (s)
        depth = 150  # Depth m

        print(f"dx = {dx} m, dy = {dy} m, dt = {dt} s")

        f0 = 0.5
        fmax = 50

        library_src = init_library_src(dt, f0, fmax, depth)

        # Spectrogram of the library source
        # library_src_sig, t_library_src_sig = ship_noise()
        # fs = 1 / (t_library_src_sig[1] - t_library_src_sig[0])
        # library_src_disp = AcousticSource(
        #     signal=library_src_sig, time=t_library_src_sig
        # )

        # Nt_w = 512 * 16  # window size
        # overlap_window = 3 / 4

        # f_, t_ref, Sxx = signal.spectrogram(
        #     library_src_disp.signal,
        #     fs=library_src_disp.fs,
        #     window="hamming",
        #     nperseg=Nt_w,
        #     noverlap=int(Nt_w * overlap_window),
        #     nfft=Nt_w,
        #     mode="complex",
        # )

        # # Plot the spectrogram
        # plt.figure()
        # h = plt.pcolormesh(
        #     t_ref, f_, 20 * np.log10(np.abs(Sxx)), shading="gouraud", cmap="jet"
        # )
        # plt.colorbar(label="PSD [dB/Hz]")
        # plt.ylim([0, library_src_disp.fs / 2])
        # plt.clim(
        #     vmin=np.max(20 * np.log10(np.abs(Sxx))) - 80,
        #     vmax=np.max(20 * np.log10(np.abs(Sxx))),
        # )
        # plt.xlabel("Time (s)")
        # plt.ylabel("Frequency (Hz)")
        # plt.title("Spectrogram")

        # plt.show()

        z_src = 5

        # Define environment
        kraken_env, kraken_flp = verlinden_test_case_env(
            env_root=env_root,
            env_filename=env_fname,
            title=env_fname,
            freq=library_src.kraken_freq,
        )
        # Define ship trajecory
        x_ship_begin = -20000
        y_ship_begin = 15000
        x_ship_end = 9000
        y_ship_end = 5000

        x_ship_t, y_ship_t, t_ship = init_event_src_traj(
            x_ship_begin, y_ship_begin, x_ship_end, y_ship_end, v_ship, dt
        )

        # # TODO : remove
        nmax_ship = 50
        x_ship_t = x_ship_t[0:nmax_ship]
        y_ship_t = y_ship_t[0:nmax_ship]
        t_ship = t_ship[0:nmax_ship]

        # Grid around the ship trajectory
        Lx = 15 * 1e3  # m
        Ly = 15 * 1e3  # m
        grid_x, grid_y = init_grid_around_event_src_traj(
            x_ship_t, y_ship_t, Lx, Ly, dx, dy
        )

        # OBS positions
        x_obs = [0, 500]
        y_obs = [0, 0]

        # ds_library = populate_grid(
        #     library_src, z_src, kraken_env, kraken_flp, grid_x, grid_y, x_obs, y_obs
        # )
        ds_library = xr.open_dataset(
            os.path.join(kraken_env.root, kraken_env.filename + "_populated" ".nc")
        )

        event_src = library_src
        ds = add_event_to_dataset(
            library_dataset=ds_library,
            event_src=event_src,
            event_t=t_ship,
            x_event_t=x_ship_t,
            y_event_t=y_ship_t,
            z_event=z_src,
            interp_src_pos_on_grid=True,
        )

        ds = build_ambiguity_surf(ds, detection_metric)
        ds.to_netcdf(
            os.path.join(
                kraken_env.root, kraken_env.filename + "_" + detection_metric + ".nc"
            )
        )
