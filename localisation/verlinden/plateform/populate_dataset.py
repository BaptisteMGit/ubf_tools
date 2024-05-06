#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   grid_tf.py
@Time    :   2024/05/02 08:55:24
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
import dask.array as da

from misc import mult_along_axis
from dask.distributed import Client

from localisation.verlinden.plateform.init_dataset import (
    init_grid,
    get_range_from_rcv,
)

from localisation.verlinden.verlinden_utils import get_azimuth_rcv
from cst import C0


def grid_tf(ds, dx=100, dy=100, rcv_info=None):

    # Check if we need to update the grid
    update_grid = not ((ds.dx == dx) & (ds.dy == dy))
    if update_grid:
        assert (
            rcv_info is not None
        ), "rcv_info must be provided if grid size is different from default (dx, dy) = (100, 100)"
        # Update lon, lat and variables depending on these coords: az_propa, delay_rcv, r_from_rcv
        grid_info = init_grid(
            rcv_info, minimum_distance_around_rcv=20 * 1e3, dx=dx, dy=dy
        )
        # Compute range from each receiver
        rr_rcv = get_range_from_rcv(grid_info, rcv_info)

        # Remove arrays to update
        var_to_drop = [
            v
            for v in [
                "lon",
                "lat",
                "az_propa",
                "r_from_rcv",
                "delay_rcv",
                "tf_gridded",
                "az_rcv",
            ]
            if v in ds.variables
        ]
        ds = ds.drop_vars(var_to_drop)
        # Update dataset
        ds = ds.assign_coords(
            lon=grid_info["lons"], lat=grid_info["lats"]
        )  # Update lon, lat
        ds["r_from_rcv"] = xr.DataArray(
            rr_rcv, dims=("idx_rcv", "lat", "lon")
        )  # Update r_from_rcv

        # Reset azimuths
        az_rcv = get_azimuth_rcv(grid_info, rcv_info)
        az_propa = np.ones(az_rcv.shape)
        ds["az_rcv"] = (["idx_rcv", "lat", "lon"], az_rcv)
        d_az = ds.all_az.values[1] - ds.all_az.values[0]
        for i_rcv in range(len(rcv_info["id"])):
            for i_az, az in enumerate(ds.all_az.values):

                if i_az == len(ds.all_az.values) - 1:
                    closest_points_idx = ds.az_rcv.sel(idx_rcv=i_rcv) >= az - d_az / 2
                else:
                    closest_points_idx = np.logical_and(
                        ds.az_rcv.sel(idx_rcv=i_rcv) >= az - d_az / 2,
                        ds.az_rcv.sel(idx_rcv=i_rcv) < az + d_az / 2,
                    )

                az_propa[i_rcv, closest_points_idx] = az

        ds = ds.drop_vars("az_rcv")
        # Add az_propa to dataset
        ds["az_propa"] = (["idx_rcv", "lat", "lon"], az_propa)
        ds["az_propa"].attrs["units"] = "°"
        ds["az_propa"].attrs["long_name"] = "Propagation azimuth"

        ds["delay_rcv"] = ds.r_from_rcv / C0  # Update delay_rcv

        # Update grid size attributes
        ds.attrs["dx"] = dx
        ds.attrs["dy"] = dy

    tf_gridded_array = np.empty(
        (ds.sizes["idx_rcv"], ds.sizes["lat"], ds.sizes["lon"], ds.sizes["kraken_freq"])
    )
    for i_rcv in ds.idx_rcv.values:
        ds_to_pop = ds.sel(idx_rcv=i_rcv)

        tf = ds_to_pop.tf.isel(kraken_depth=0)  # Only one depth

        for az in ds_to_pop.all_az.values:
            az_mask_2d = ds_to_pop.az_propa == az
            r_az = ds_to_pop.r_from_rcv.values[az_mask_2d]
            tf_az = tf.sel(all_az=az).sel(kraken_range=r_az, method="nearest")
            tf_gridded_array[i_rcv, az_mask_2d, :] = tf_az.values.T

    ds["tf_gridded"] = (["idx_rcv", "lat", "lon", "kraken_freq"], tf_gridded_array)
    ds["tf_gridded"].attrs["units"] = ""
    ds["tf_gridded"].attrs["long_name"] = "Transfer function gridded"

    # Update dataset
    if update_grid:
        ds.to_zarr(ds.fullpath_dataset, mode="w")
    else:
        ds.to_zarr(ds.fullpath_dataset, mode="a")

    return ds


def grid_synthesis(
    ds,
    src,
    apply_delay=True,
):
    propagating_freq = src.positive_freq
    propagating_spectrum = src.positive_spectrum

    # # TODO remove
    # propagating_freq = propagating_freq[200 : ds.sizes["kraken_freq"] + 200]
    # propagating_spectrum = propagating_spectrum[200 : ds.sizes["kraken_freq"] + 200]

    k0 = 2 * np.pi * propagating_freq / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    nfft_inv = (
        4 * src.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / src.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt)

    # Loop over sub_regions of the grid
    nregion = ds.sizes["lon"]
    max_size_bytes = 0.5 * 1e9  # 500 Mo
    size = ds.tf.nbytes / nregion
    while size <= max_size_bytes and nregion > 1:  # At least 1 region
        nregion -= 1
        size = ds.tf.nbytes / nregion

    lat_slices_lim = np.linspace(0, ds.sizes["lat"], nregion + 1, dtype=int)
    lat_slices = [
        slice(lat_slices_lim[i], lat_slices_lim[i + 1])
        for i in range(len(lat_slices_lim) - 1)
    ]
    lon_slices_lim = np.linspace(0, ds.sizes["lon"], nregion + 1, dtype=int)
    lon_slices = [
        slice(lon_slices_lim[i], lon_slices_lim[i + 1])
        for i in range(len(lon_slices_lim) - 1)
    ]

    ds.coords["library_signal_time"] = time_vector
    ts_dim = ["idx_rcv", "lat", "lon", "library_signal_time"]
    ts_shape = [ds.sizes[dim] for dim in ts_dim]
    transmited_field_t = np.empty(ts_shape)
    ds["rcv_signal_library"] = (
        ts_dim,
        transmited_field_t,
    )
    ds.rcv_signal_library.attrs["long_name"] = r"$s_{i}$"
    # Save to zarr without computing
    ds.to_zarr(ds.fullpath_dataset, mode="a", compute=False)

    for lon_s in lon_slices:
        for lat_s in lat_slices:
            ds_sub = ds.isel(lat=lat_s, lon=lon_s)

            # Compute received signal in sub_region
            transmited_field_t = compute_received_signal(
                ds_sub,
                propagating_freq,
                propagating_spectrum,
                norm_factor,
                nfft_inv,
                apply_delay,
            )

            ds.rcv_signal_library[dict(lat=lat_s, lon=lon_s)] = transmited_field_t
            sub_region_to_save = ds.rcv_signal_library[dict(lat=lat_s, lon=lon_s)]

            # Save to zarr
            sub_region_to_save.to_zarr(
                ds.fullpath_dataset,
                mode="r+",
                region={
                    "idx_rcv": slice(0, ds.sizes["idx_rcv"]),
                    "lat": lat_s,
                    "lon": lon_s,
                    "library_signal_time": slice(0, ds.sizes["library_signal_time"]),
                },
            )

    return ds


def compute_received_signal(
    ds, propagating_freq, propagating_spectrum, norm_factor, nfft_inv, apply_delay
):

    # Received signal spectrum resulting from the convolution of the src signal and the impulse response
    transmited_field_f = mult_along_axis(
        ds.tf_gridded, propagating_spectrum * norm_factor, axis=-1
    )

    # Apply corresponding delay to the signal
    if apply_delay:
        tau = ds.delay_rcv.min(
            dim="idx_rcv"
        )  # Delay to apply to the signal to take into account the propagation time

        # Expand tau to the signal shape
        tau = tau.expand_dims({"kraken_freq": ds.sizes["kraken_freq"]}, axis=-1)
        # Derive delay factor
        tau_vec = mult_along_axis(tau, propagating_freq, axis=-1)
        delay_f = np.exp(1j * 2 * np.pi * tau_vec)
        # Expand delay factor to the signal shape
        delay_f = tau.copy(deep=True, data=delay_f).expand_dims(
            {"idx_rcv": ds.sizes["idx_rcv"]}, axis=0
        )
        # Apply delay
        transmited_field_f = transmited_field_f * delay_f

    # Fourier synthesis of the received signal -> time domain
    n_workers = 6
    chunk_shape = (
        ds.sizes["idx_rcv"],
        ds.sizes["lat"] // n_workers,
        ds.sizes["lon"] // n_workers,
        ds.sizes["kraken_freq"],
    )
    transmited_field_f = da.from_array(transmited_field_f, chunks=chunk_shape)
    transmited_field_t = np.fft.irfft(transmited_field_f, axis=-1, n=nfft_inv).compute()

    return transmited_field_t


def populate_dataset(ds, src, **kwargs):
    # Grid transfer functions
    grid_tf_kw = {key: kwargs[key] for key in kwargs if key in ["dx", "dy", "rcv_info"]}
    ds = grid_tf(ds, **grid_tf_kw)

    client = Client(n_workers=6, threads_per_worker=1)
    print(client.dashboard_link)

    # Grid synthesis
    grid_synthesis_kw = {key: kwargs[key] for key in kwargs if key in ["apply_delay"]}
    ds = grid_synthesis(ds, src, **grid_synthesis_kw)

    return ds


if __name__ == "__main__":
    # fname = "propa_dataset_65.4003_66.1446_-27.8377_-27.3528.zarr"
    # fname = "propa_dataset_65.4003_66.1446_-27.8377_-27.3528_10000m.zarr"
    fname = "propa_dataset_65.5928_65.9521_-27.6662_-27.5711_backup.zarr"
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1"
    fpath = os.path.join(root, fname)

    from dask.distributed import Client

    ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos

    fname = "propa_dataset_65.5928_65.9521_-27.6662_-27.5711.zarr"
    fpath = os.path.join(root, fname)
    ds_full = xr.open_dataset(fpath, engine="zarr", chunks={})

    rcv_info_dw = {
        # "id": ["RR45", "RR48", "RR44"],
        "id": ["RR45", "RR48"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)

    # ds = grid_tf(
    #     ds,
    #     dx=100,
    #     dy=100,
    #     rcv_info=rcv_info_dw,
    # )

    fs = 100  # Sampling frequency
    # Library
    f0_lib = 1  # Fundamental frequency of the ship signal
    src_info = {
        "sig_type": "ship",
        "f0": f0_lib,
        "std_fi": f0_lib * 1 / 100,
        "tau_corr_fi": 1 / f0_lib,
        "fs": fs,
    }

    dt = 5
    min_waveguide_depth = 5000
    from localisation.verlinden.AcousticComponent import AcousticSource
    from signals import generate_ship_signal

    src_sig, t_src_sig = generate_ship_signal(
        Ttot=dt,
        f0=src_info["f0"],
        std_fi=src_info["std_fi"],
        tau_corr_fi=src_info["tau_corr_fi"],
        fs=src_info["fs"],
    )

    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="ship",
        waveguide_depth=min_waveguide_depth,
    )

    # ds = grid_synthesis(ds, src)
    ds = populate_dataset(ds, src, rcv_info=rcv_info_dw)

    print()
