import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal

from tqdm import tqdm

from cst import BAR_FORMAT
from utils import mult_along_axis
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

""" Simple example to test the Verlinden method 
The ocean environment is assumed to be fully isotropic ; only one profile is considered for the whole domain. """


dx = 1000  # m
dy = 1000  # m
v_ship = 50 / 3.6  # m/s
dt = min(dx, dy) / v_ship  # Minimum time spent by the source in a single grid box (s)
depth = 150  # Depth m

print(f"dx = {dx} m, dy = {dy} m, dt = {dt} s")


# Source signal
# y, t = ship_noise()
# fs = 1 / (t[1] - t[0])
fs_kraken = 8 * 50
# L = int(fs * dt)
f0 = 0.5
fmax = 50
# f = np.arange(f0, fmax, fs / L)
# ship_spec = ship_spectrum(f)

# plt.figure()
# plt.plot(f, np.abs(ship_spec))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("|FFT(s(t))|")
# plt.title("Ship spectrum")


# Associated time serie of the source
# y = np.fft.irfft(ship_spec, n=L)
# event_src_sig = np.real(y)
# t_event_src_sig = np.arange(0, event_src_sig.shape[0] / fs, 1 / fs)

event_src_sig, t_event_src_sig = ship_noise()
fs = 1 / (t_event_src_sig[1] - t_event_src_sig[0])
nmax = int(fs * dt)
event_src_sig = event_src_sig[0:nmax]
t_event_src_sig = t_event_src_sig[0:nmax]


# Plot spectrogram of the src signal
# nperseg = 512 * 2
# overlap_window = 2 / 4
# noverlap = int(nperseg * overlap_window)
# f, t, Sxx = signal.spectrogram(
#     event_src_sig, fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
# )

# plt.figure()
# plt.pcolormesh(t, f, 20 * np.log10(np.abs(Sxx)))
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")

# plt.show()


# plt.figure()
# plt.plot(t_event_src_sig, event_src_sig)
# plt.xlabel("Time (s)")
# plt.ylabel("s(t)")
# plt.title("Ship signal")
# plt.show()

# Dummy initialisation of the source
# event_src.display_source()
event_src = AcousticSource(signal=event_src_sig, time=t_event_src_sig)
f_cutoff = waveguide_cutoff_freq(max_depth=depth)

# f_interp = np.arange(f0, fmax, f0)
# spec_interp = np.interp(f, event_src.freq, event_src.spectrum)
# event_src.freq = f
# event_src.spectrum = spec_interp

# Filter frequencies below the cutoff frequency
# f = f[f >= f_cutoff]
# event_src.freq = f
# event_src.spectrum = ship_spec
event_src.kraken_freq = np.arange(max(f0, f_cutoff + 10), fmax, 1)

# KRAKEN
run_kraken = False
run_field = False
# env_fname = "verlinden_1_ssp"
env_fname = "verlinden_1_test_case"
env_root = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\simple_case"
)

# kraken_env, kraken_flp = isotropic_ideal_env(
#     env_root=env_root,
#     env_filename=env_fname,
#     title="verlinden_1",
#     max_depth=depth,
#     freq=event_src.kraken_freq,
# )

# kraken_env, kraken_flp = rhum_rum_isotropic_env(
#     env_root=env_root,
#     env_filename=env_fname,
#     title=env_fname,
#     freq=event_src.kraken_freq,
# )

kraken_env, kraken_flp = verlinden_test_case_env(
    env_root=env_root,
    env_filename=env_fname,
    title=env_fname,
    freq=event_src.kraken_freq,
)

# kraken_env.medium.plot_medium()
# plt.show()

if run_kraken:
    # Write env and flp files
    kraken_env.write_env()
    kraken_flp.write_flp()
    # Run KRAKEN
    os.chdir(env_root)
    runkraken(env_fname)

if run_field:
    # Run FIELD
    os.chdir(env_root)
    kraken_flp.write_flp()
    runfield(env_fname)


# for f in [10, 15, 20, 25, 40, 45]:
#     plotshd(kraken_env.shd_fpath, freq=f, units="km")
# plt.show()

# Source position
# Define ship trajecory
x_ship_begin = -20000
y_ship_begin = 15000
x_ship_end = 9000
y_ship_end = 5000
z_ship = 5

Dtot = np.sqrt((x_ship_begin - x_ship_end) ** 2 + (y_ship_begin - y_ship_end) ** 2)

vx = v_ship * (x_ship_end - x_ship_begin) / Dtot
vy = v_ship * (y_ship_end - y_ship_begin) / Dtot

Ttot_ship = Dtot / v_ship + 3
t_ship = np.arange(0, Ttot_ship - dt, dt)
npos_ship = t_ship.size

x_ship_t = x_ship_begin + vx * t_ship
y_ship_t = y_ship_begin + vy * t_ship

# TODO : remove
x_ship_t = x_ship_t[0:10]
y_ship_t = y_ship_t[0:10]
t_ship = t_ship[0:10]

# Grid around the ship trajectory
Lx = 15 * 1e3  # m
Ly = 15 * 1e3  # m
grid_x = np.arange(
    -Lx / 2 + min(x_ship_t), Lx / 2 + max(x_ship_t), dx, dtype=np.float32
)
grid_y = np.arange(
    -Ly / 2 + min(y_ship_t), Ly / 2 + max(y_ship_t), dy, dtype=np.float32
)
xx, yy = np.meshgrid(grid_x, grid_y, sparse=True)

# OBS positions
x_obs = [-2000, 18000]
y_obs = [-14000, -10000]
n_obs = len(x_obs)
rr_obs = np.array(
    [
        np.sqrt((xx - x_obs[i_obs]) ** 2 + (yy - y_obs[i_obs]) ** 2)
        for i_obs in range(n_obs)
    ]
)

# Distance to receiver
r_ship_t = [
    np.sqrt((x_ship_t - x_obs[i_obs]) ** 2 + (y_ship_t - y_obs[i_obs]) ** 2)
    for i_obs in range(n_obs)
]
r_ship = np.array(r_ship_t)

ds = xr.Dataset(
    data_vars=dict(
        x_obs=(["idx_obs"], x_obs),
        y_obs=(["idx_obs"], y_obs),
        r_from_obs=(["idx_obs", "y", "x"], rr_obs),
        x_ship=(["src_trajectory_time"], x_ship_t),
        y_ship=(["src_trajectory_time"], y_ship_t),
        r_obs_ship=(["idx_obs", "src_trajectory_time"], r_ship),
    ),
    coords=dict(
        x=grid_x,
        y=grid_y,
        event_signal_time=[],
        src_trajectory_time=t_ship,
        idx_obs=np.arange(n_obs),
    ),
    attrs=dict(
        title="Verlinden simulation with simple environment",
        dx=dx,
        dy=dy,
    ),
)

# Free memory
del x_obs, y_obs, rr_obs, x_ship_t, y_ship_t, r_ship_t, r_ship, grid_x, grid_y, t_ship

# Interpolate ship position on grid
interp_ship_on_grid = True

if interp_ship_on_grid:
    ds["x_ship"] = ds.x.sel(x=ds.x_ship, method="nearest")
    ds["y_ship"] = ds.y.sel(y=ds.y_ship, method="nearest")
    ds["r_obs_ship"].values = [
        np.sqrt(
            (ds.x_ship - ds.x_obs.sel(idx_obs=i_obs)) ** 2
            + (ds.y_ship - ds.y_obs.sel(idx_obs=i_obs)) ** 2
        )
        for i_obs in range(n_obs)
    ]
    ds.attrs["source_positions"] = "Interpolated on grid"
    ds.attrs["src_pos"] = "on_grid"
else:
    ds.attrs["source_positions"] = "Not interpolated on grid"
    ds.attrs["src_pos"] = "not_on_grid"


obs_pairs = []
for i in ds.idx_obs.values:
    for j in range(i + 1, ds.idx_obs.values[-1] + 1):
        obs_pairs.append((i, j))
ds.coords["idx_obs_pairs"] = np.arange(len(obs_pairs))
ds.coords["idx_obs_in_pair"] = np.arange(2)
ds["obs_pairs"] = (["idx_obs_pairs", "idx_obs_in_pair"], obs_pairs)


# Set attributes
var_unit_mapping = {
    "m": ["x_obs", "y_obs", "x", "y", "r_from_obs", "x_ship", "y_ship", "r_obs_ship"],
    "s": ["event_signal_time", "src_trajectory_time"],
    "": ["idx_obs"],
}
for unit in var_unit_mapping.keys():
    for var in var_unit_mapping[unit]:
        ds[var].attrs["units"] = unit

ds["x_obs"].attrs["long_name"] = "x_obs"
ds["y_obs"].attrs["long_name"] = "y_obs"
ds["r_from_obs"].attrs["long_name"] = "Range from receiver"
ds["x_ship"].attrs["long_name"] = "x_ship"
ds["y_ship"].attrs["long_name"] = "y_ship"
ds["r_obs_ship"].attrs["long_name"] = "Range from receiver to source"
ds["x"].attrs["long_name"] = "x"
ds["y"].attrs["long_name"] = "y"
ds["event_signal_time"].attrs["units"] = "Time"
ds["src_trajectory_time"].attrs["long_name"] = "Time"
ds["idx_obs"].attrs["long_name"] = "Receiver index"


# plt.figure()
# for i in ds.idx_obs.values:
#     ds.r_obs_ship.sel(idx_obs=i).plot(label=f"Receiver {i}")
# plt.legend()
# plt.tight_layout()

# plt.figure()
# plt.scatter(ds["x_obs"], ds["y_obs"], marker="o", color="red")
# plt.plot(ds["x_ship"], ds["y_ship"], marker="x", color="k")
# plt.xlim(ds["x"].min(), ds["x"].max())
# plt.ylim(ds["y"].min(), ds["y"].max())
# plt.grid(True)
# plt.tight_layout()
# plt.show()

""" Populate the entire grid with the library vectors"""
# TODO this will need to be uptaded for a non isotropic environment
# Here we assume that the environment is isotropic and therefore we only consider one profile
# Therefore, the received signal and thus the correlation vector only depends on the range difference between the receivers but not on the azimuth


# Check if populated ds already exists
populated_ds_path = os.path.join(env_root, env_fname + "_populated.nc")
# if os.path.exists(populated_ds_path):
#     ds_populated = xr.open_dataset(populated_ds_path)
#     ds = ds_populated.copy(deep=True)
#     ds_populated.close()

if False:
    pass
# Populate library
else:
    library_src = event_src
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
            rcv_depth=[z_ship],
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
                corr_01 /= np.max(corr_01)

                library_corr[i_pair, i_y, i_x, :] = corr_01

                del s0, s1, corr_01

    ds["library_corr"] = (library_corr_dim, library_corr)

    ds.to_netcdf(populated_ds_path)

# plt.figure()
# ds.library_corr.sel(idx_obs_pairs=0, x=x_obs[0], y=y_obs[0]).plot()

"""Create the event vector for each pair of receivers and for each position of the ship """
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
        rcv_depth=[z_ship],
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
        # corr_0 = signal.correlate(s0, s0)
        # corr_1 = signal.correlate(s1, s1)
        # corr_01 /= np.sqrt(corr_0 * corr_1)
        corr_01 /= np.max(corr_01)

        event_corr[i_pair, i_ship, :] = corr_01

        del s0, s1, corr_01

ds["event_corr"] = (event_corr_dim, event_corr)


""" Build ambiguity surface """
detection_metric = (
    "hibert_env_intercorr0"  # "intercorr0", "lstsquares", "hibert_env_intercorr0"
)
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
            amb_surf = np.sum(amb_surf, axis=2)
            ambiguity_surface[i_pair, i_ship, ...] = amb_surf / np.max(amb_surf)

        elif detection_metric == "lstsquares":
            lib = ds.library_corr.sel(idx_obs_pairs=i_pair).values
            event = (
                ds.event_corr.sel(idx_obs_pairs=i_pair)
                .isel(src_trajectory_time=i_ship)
                .values
            )
            diff = np.abs(lib) - np.abs(event)
            amb_surf = np.sum(diff**2, axis=2)
            # amb_surf = np.sum(amb_surf, axis=2)
            ambiguity_surface[i_pair, i_ship, ...] = amb_surf / np.max(amb_surf)

        elif detection_metric == "hibert_env_intercorr0":
            lib_env = np.abs(signal.hilbert(ds.library_corr.sel(idx_obs_pairs=i_pair)))
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
            amb_surf = np.sum(amb_surf, axis=2)
            ambiguity_surface[i_pair, i_ship, ...] = amb_surf / np.max(amb_surf)

        del amb_surf

ds["ambiguity_surface"] = (
    ambiguity_surface_dim,
    ambiguity_surface,
)

# Derive src position
detected_pos_dim = ["idx_obs_pairs", "src_trajectory_time"]
if detection_metric in ["intercorr0", "hibert_env_intercorr0"]:
    ds["detected_pos_x"] = ds.x.isel(x=ds.ambiguity_surface.argmax(dim=["x", "y"])["x"])
    ds["detected_pos_y"] = ds.y.isel(y=ds.ambiguity_surface.argmax(dim=["x", "y"])["y"])

elif detection_metric == "lstsquares":
    ds["detected_pos_x"] = ds.x.isel(x=ds.ambiguity_surface.argmin(dim=["x", "y"])["x"])
    ds["detected_pos_y"] = ds.y.isel(y=ds.ambiguity_surface.argmin(dim=["x", "y"])["y"])

root = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\simple_case"
)

nc_path = os.path.join(root, env_fname + ".nc")

# # Remove or rename the existing file
# if os.path.exists(nc_path):
#     os.remove(nc_path)

ds.to_netcdf(nc_path)
