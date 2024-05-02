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

from time import sleep
from dask.distributed import Client
import dask

from propa.kraken_toolbox.run_kraken import runkraken, clear_kraken_parallel_working_dir
from localisation.verlinden.plateform.init_dataset import (
    init_grid,
    get_range_from_rcv,
    set_azimuths,
)

from localisation.verlinden.verlinden_utils import get_azimuth_rcv
from cst import C0


def grid_tf(ds, dx=100, dy=100, rcv_info=None):

    # Check if we need to update the grid
    update_grid = ~((ds.dx == dx) & (ds.dy == dy))
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

        #   Dummy lon lat loop version
        # for lon in ds.lon.values:
        #     for lat in ds.lat.values:
        #         az = ds_to_pop.az_propa.sel(lon=lon, lat=lat)
        #         r = ds_to_pop.r_from_rcv.sel(lon=lon, lat=lat)
        #         tf_lon_lat = tf.sel(all_az=az, kraken_range=r, method="nearest")
        #         tf_gridded.loc[dict(idx_rcv=i_rcv, lon=lon, lat=lat)] = tf_lon_lat

        for az in ds_to_pop.all_az.values:
            az_mask_2d = ds_to_pop.az_propa == az
            r_az = ds_to_pop.r_from_rcv.values[az_mask_2d]
            tf_az = tf.sel(all_az=az, kraken_range=r_az, method="nearest")
            tf_gridded_array[i_rcv, az_mask_2d, :] = tf_az.values.T

    ds["tf_gridded"] = (["idx_rcv", "lat", "lon", "kraken_freq"], tf_gridded_array)
    ds["tf_gridded"].attrs["units"] = ""
    ds["tf_gridded"].attrs["long_name"] = "Transfer function gridded"

    # Update dataset
    if update_grid:
        ds.to_zarr(ds.fullpath_dataset, mode="w")
    else:
        ds.tf_gridded.to_zarr(ds.fullpath_dataset, mode="a")

    return ds


def grid_synthesis(ds, src):
    # Extract propagating spectrum from entire spectrum
    fc = waveguide_cutoff_freq(waveguide_depth=minimum_waveguide_depth)
    propagating_freq = src.positive_freq[src.positive_freq > fc]
    propagating_spectrum = src.positive_spectrum[src.positive_freq > fc]

    k0 = 2 * np.pi * propagating_freq / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    # Received signal spectrum resulting from the convolution of the src signal and the impulse response
    transmited_field_f = mult_along_axis(
        pressure_field, propagating_spectrum * norm_factor, axis=0
    )

    nfft_inv = (
        4 * src.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / src.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt)

    # Apply corresponding delay to the signal
    if apply_delay:
        for ir, rcv_r in enumerate(rcv_range):  # TODO: remove loop for efficiency
            if delay is None:
                tau = rcv_r / C0
            else:
                tau = delay[ir]

            delay_f = np.exp(1j * 2 * np.pi * tau * propagating_freq)

            transmited_field_f[..., ir] = mult_along_axis(
                transmited_field_f[..., ir], delay_f, axis=0
            )

    # Fourier synthesis of the received signal -> time domain
    received_signal_t = np.fft.irfft(transmited_field_f, axis=0, n=nfft_inv)
    transmited_field_t = np.real(received_signal_t)


if __name__ == "__main__":
    # fname = "propa_dataset_65.4003_66.1446_-27.8377_-27.3528_10000m.zarr"
    fname = "propa_dataset_65.4003_66.1446_-27.8377_-27.3996.zarr"
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\localisation_dataset\testcase3_1"
    fpath = os.path.join(root, fname)
    ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos

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

    ds = grid_tf(
        ds,
        dx=100,
        dy=100,
        rcv_info=rcv_info_dw,
    )

    print()
