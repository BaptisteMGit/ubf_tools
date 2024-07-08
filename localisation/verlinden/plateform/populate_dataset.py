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
import zarr
import numpy as np
import xarray as xr
import dask.array as da

from misc import mult_along_axis
from dask.distributed import Client

from localisation.verlinden.plateform.utils import *
from localisation.verlinden.plateform.init_dataset import (
    init_grid,
    get_range_from_rcv,
)

from localisation.verlinden.verlinden_utils import get_azimuth_rcv
from cst import C0
from localisation.verlinden.plateform.params import N_WORKERS


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
        set_propa_grid_path(ds)

    # Save existing vars (no need to save tf)
    ds.drop_vars("tf").to_zarr(ds.fullpath_dataset_propa_grid, mode="w")
    # ds = ds.drop_vars("tf")
    tf_gridded_dim = ["idx_rcv", "lat", "lon", "kraken_freq"]
    tf_gridded_shape = [ds.sizes[dim] for dim in tf_gridded_dim]
    # chunk_lon = max(1, ds.sizes["lon"] // 10)
    # chunk_lat = max(1, ds.sizes["lat"] // 10)
    # , chunks=(1, chunk_lat, chunk_lon, 1)
    tf_gridded_array = da.empty(tf_gridded_shape, dtype=np.complex64)
    ds["tf_gridded"] = (
        tf_gridded_dim,
        tf_gridded_array,
    )
    ds["tf_gridded"].attrs["units"] = ""
    ds["tf_gridded"].attrs["long_name"] = "Transfer function gridded"

    nregion_lon = get_region_number(
        ds.sizes["lon"], ds.tf_gridded, max_size_bytes=1 * 1e9
    )
    nregion_lat = get_region_number(
        ds.sizes["lat"], ds.tf_gridded, max_size_bytes=1 * 1e9
    )
    lon_slices, lat_slices = get_lonlat_sub_regions(ds, nregion_lon, nregion_lat)

    lat_chunksize = int(ds.sizes["lat"] // nregion_lat)
    lon_chunksize = int(ds.sizes["lon"] / nregion_lon)

    # lon_chunksize = ds.sizes['lon']
    idx_rcv_chunksize, freq_chunksize = ds.sizes["idx_rcv"], ds.sizes["kraken_freq"]

    chunksize = (idx_rcv_chunksize, lat_chunksize, lon_chunksize, freq_chunksize)
    # chunksize = tuple([ds.sizes[v] for v in  tf_gridded_dim])
    ds["tf_gridded"] = ds.tf_gridded.chunk(chunksize)

    # Save to zarr without computing
    ds.to_zarr(ds.fullpath_dataset_propa_grid, mode="a", compute=False)

    for lat_s in lat_slices:
        for lon_s in lon_slices:
            ds_sub = ds.isel(lat=lat_s, lon=lon_s)

            tf_sub = np.empty_like(ds_sub.tf_gridded, dtype=np.complex64)

            for i_rcv in ds.idx_rcv.values:
                ds_to_pop = ds_sub.sel(idx_rcv=i_rcv)

                tf = ds_to_pop.tf.isel(kraken_depth=0)  # Only one depth

                for az in ds_to_pop.all_az.values:
                    az_mask_2d = ds_to_pop.az_propa == az
                    r_az = ds_to_pop.r_from_rcv.values[az_mask_2d]
                    tf_az = tf.sel(all_az=az).sel(kraken_range=r_az, method="nearest")
                    tf_sub[i_rcv, az_mask_2d, :] = tf_az.values.T

            # ds_sub.tf_gridded[dict(lat=lat_s, lon=lon_s)] = tf_sub
            # print(tf_sub)
            ds.tf_gridded[dict(lat=lat_s, lon=lon_s)] = tf_sub
            sub_region_to_save = ds.tf_gridded[dict(lat=lat_s, lon=lon_s)]
            # print(sub_region_to_save)
            sub_region_to_save.to_zarr(
                ds.fullpath_dataset_propa_grid,
                mode="r+",
                region={
                    "idx_rcv": slice(None),
                    "lat": lat_s,
                    "lon": lon_s,
                    "kraken_freq": slice(None),
                },
            )

            # print(f"region processes in {time() - t0}s")

    # Open the Zarr store and update attrs
    zarr_store = zarr.open(ds.fullpath_dataset_propa)
    zarr_store.attrs.update({"propa_grid_done": True})
    zarr.consolidate_metadata(ds.fullpath_dataset_propa)

    ds = ds.drop_vars("tf")

    return ds


def grid_synthesis(
    ds,
    src,
    apply_delay=True,
):
    # with Client(n_workers=N_WORKERS, threads_per_worker=1) as client:
    #     print(client.dashboard_link)

    # Set path to save the dataset and save existing vars
    ds.attrs["src_label"] = build_src_label(src_name=src.name)
    set_propa_grid_src_path(ds)
    # ds.attrs["fullpath_dataset_propa_grid_src"] = (
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\testzarr"
    # )
    ds.to_zarr(ds.fullpath_dataset_propa_grid_src, mode="w")

    propagating_freq = src.positive_freq
    propagating_spectrum = src.positive_spectrum

    k0 = 2 * np.pi * propagating_freq / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    nfft_inv = (
        4 * src.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / src.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt)

    ds.coords["library_signal_time"] = time_vector
    ts_dim = ["idx_rcv", "lat", "lon", "library_signal_time"]
    ts_shape = [ds.sizes[dim] for dim in ts_dim]
    transmited_field_t = da.empty(ts_shape, dtype=np.float32)

    ds["rcv_signal_library"] = (
        ts_dim,
        transmited_field_t,
    )
    ds.rcv_signal_library.attrs["long_name"] = r"$s_{i}$"

    # Loop over sub_regions of the grid
    nregion_lon = get_region_number(
        ds.sizes["lon"], ds.rcv_signal_library, max_size_bytes=1 * 1e9
    )
    nregion_lat = get_region_number(
        ds.sizes["lat"], ds.rcv_signal_library, max_size_bytes=1 * 1e9
    )

    lon_slices, lat_slices = get_lonlat_sub_regions(ds, nregion_lon, nregion_lat)

    lat_chunksize = int(ds.sizes["lat"] // nregion_lat)
    lon_chunksize = int(ds.sizes["lon"] / nregion_lon)
    idx_rcv_chunksize, time_chunksize = (
        ds.sizes["idx_rcv"],
        ds.sizes["library_signal_time"],
    )
    chunksize = (idx_rcv_chunksize, lat_chunksize, lon_chunksize, time_chunksize)
    ds["rcv_signal_library"] = ds.rcv_signal_library.chunk(chunksize)

    # Fix the values at the receiver positions
    rcv_grid_lon = ds.sel(lon=ds.lon_rcv.values, method="nearest").lon.values
    rcv_grid_lat = ds.sel(lat=ds.lat_rcv.values, method="nearest").lat.values

    for i in range(len(rcv_grid_lon)):
        ds["tf_gridded"].loc[
            dict(lon=rcv_grid_lon[i], lat=rcv_grid_lat[i], idx_rcv=i)
        ] = 1

    # Save to zarr without computing
    ds.to_zarr(ds.fullpath_dataset_propa_grid_src, mode="a", compute=False)

    # print(ds.rcv_signal_library)
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
            # print(sub_region_to_save)
            # Save to zarr
            sub_region_to_save.to_zarr(
                ds.fullpath_dataset_propa_grid_src,
                mode="r+",
                region={
                    "idx_rcv": slice(None),
                    "lat": lat_s,
                    "lon": lon_s,
                    "library_signal_time": slice(None),
                },
            )

    # Open the Zarr store and update attrs
    zarr_store = zarr.open(ds.fullpath_dataset_propa_grid_src)
    zarr_store.attrs.update({"propa_grid_src_done": True})
    zarr.consolidate_metadata(ds.fullpath_dataset_propa_grid_src)

    return ds


def populate_dataset(ds, src, **kwargs):
    # Grid transfer functions
    grid_tf_kw = {key: kwargs[key] for key in kwargs if key in ["dx", "dy", "rcv_info"]}
    ds = grid_tf(ds, **grid_tf_kw)

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
    from localisation.verlinden.misc.AcousticComponent import AcousticSource
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
