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

fs_kraken = 8 * 50
f0 = 0.5
fmax = 50


def populate_grid(kraken_env, kraken_flp, grid_x, grid_y, x_obs, y_obs):
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
            dx=grid_x.diff[0],
            dy=grid_y.diff[0],
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
            "x_ship",
            "y_ship",
            "r_obs_ship",
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
