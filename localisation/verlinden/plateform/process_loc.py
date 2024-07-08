#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   add_event.py
@Time    :   2024/05/16 16:19:10
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
import pandas as pd
import dask.array as da

from cst import C0
from misc import mult_along_axis
from dask.distributed import Client
from localisation.verlinden.plateform.utils import (
    init_event_dataset,
    init_ambiguity_surface,
    init_corr_library,
    init_corr_event,
    add_noise_to_library,
    add_correlation_library,
    add_noise_to_event,
    add_correlation_event,
    build_process_output_path,
    add_ambiguity_surf,
    get_region_number,
    get_lonlat_sub_regions,
)
from localisation.verlinden.misc.verlinden_utils import (
    init_event_src_traj,
    init_grid_around_event_src_traj,
    load_rhumrum_obs_pos,
    get_bathy_grid_size,
)

from localisation.verlinden.plateform.analysis_loc import analysis


# ======================================================================================================================
# Functions
# ======================================================================================================================


def add_event(ds, src_info, rcv_info, apply_delay, verbose=True):

    if verbose:
        print(f"Add event received signal to dataset: \n\t{src_info}")

    pos_src_info = src_info["pos"]
    src = src_info["sig"]["src"]

    propagating_freq = src.positive_freq
    propagating_spectrum = src.positive_spectrum

    k0 = 2 * np.pi * propagating_freq / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    nfft_inv = (
        4 * src.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / src.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt, dtype=np.float32)

    ds = init_event_dataset(ds, pos_src_info, rcv_info)

    ds["event_signal_time"] = time_vector
    signal_event_dim = ["idx_rcv", "src_trajectory_time", "event_signal_time"]

    # TODO change into dask array
    chunks = (1, 1, len(time_vector))
    rcv_signal_event = da.empty(
        tuple(ds.sizes[d] for d in signal_event_dim), chunks=chunks, dtype=np.float32
    )
    ds["rcv_signal_event"] = (signal_event_dim, rcv_signal_event)

    # Apply corresponding delay to the signal
    for i_pos in range(pos_src_info["n_pos"]):
        tf = ds.tf_gridded.sel(
            lon=pos_src_info["lons"][i_pos],
            lat=pos_src_info["lats"][i_pos],
            method="nearest",
        )
        transmitted_sig_f = mult_along_axis(
            tf, propagating_spectrum * norm_factor, axis=-1
        )
        if apply_delay:
            # Delay to apply to the signal to take into account the propagation time
            tau = (
                ds.delay_src_rcv.min(dim="idx_rcv")
                .isel(src_trajectory_time=i_pos)
                .values
            )
            # tau = ds.delay_rcv.isel(idx_rcv=0).sel(lat=pos_src_info["lats"][i_pos], lon=pos_src_info["lons"][i_pos], method="nearest").values

            # Derive delay factor
            tau_vec = tau * propagating_freq
            delay_f = np.exp(1j * 2 * np.pi * tau_vec)
            # Apply delay
            transmitted_sig_f *= delay_f

        transmitted_sig_t = np.fft.irfft(transmitted_sig_f, n=nfft_inv, axis=-1)
        ds.rcv_signal_event[dict(src_trajectory_time=i_pos)] = transmitted_sig_t

    # Init corr for event signal
    ds = init_corr_event(ds)

    # Expand rcv_signal_library to snr dims
    ds["rcv_signal_event"] = ds["rcv_signal_event"].expand_dims(
        {"snr": ds.sizes["snr"]}, axis=0
    )

    ds["event_corr"] = ds["event_corr"].expand_dims({"snr": ds.sizes["snr"]}, axis=0)

    return ds


def load_subset(fpath, pos_src_info, grid_info, dt, verbose=True):
    """
    Load a subset of the dataset around the source to be localized.
    """

    # Load the dataset
    ds = xr.open_dataset(fpath, engine="zarr", chunks={})

    # Define limits of the subset area
    init_event_src_traj(pos_src_info, dt)
    init_grid_around_event_src_traj(pos_src_info, grid_info)

    if verbose:
        print(
            f"Load dataset subset: \n\tlon ({grid_info['min_lon']}, {grid_info['max_lon']}) \n\tlat ({grid_info['min_lat']}, {grid_info['max_lat']})"
        )

    # Extract area around the source
    ds_subset = ds.sel(
        lon=slice(grid_info["min_lon"], grid_info["max_lon"]),
        lat=slice(grid_info["min_lat"], grid_info["max_lat"]),
    )

    return ds_subset


def init_dataset(
    main_ds_path,
    src_info,
    grid_info,
    dt,
    similarity_metrics,
    snrs_dB,
    n_noise_realisations=100,
    verbose=True,
):
    # Load subset of the main dataset
    ds = load_subset(
        main_ds_path, pos_src_info=src_info["pos"], grid_info=grid_info, dt=dt
    )

    if verbose:
        print(f"Initialise dataset")

    # Initialisation time
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ds.attrs["init_time"] = now

    # Add similarity metrics info to the dataset
    n_similarity_metrics = len(similarity_metrics)
    ds.coords["idx_similarity_metric"] = np.arange(n_similarity_metrics)
    ds["similarity_metrics"] = (["idx_similarity_metric"], similarity_metrics)
    ds["idx_similarity_metric"].attrs["long_name"] = "Similarity metric index"
    ds["idx_similarity_metric"].attrs["unit"] = ""

    # Noise info
    ds.coords["snr"] = snrs_dB
    ds.coords["idx_noise_realisation"] = np.arange(n_noise_realisations)
    ds["snr"].attrs["unit"] = "dB"
    ds["idx_noise_realisation"].attrs["long_name"] = "Noise realisation index"
    ds["idx_noise_realisation"].attrs["unit"] = ""

    # Output path
    build_process_output_path(ds, src_info, grid_info)

    # Expand rcv_signal_library to snr dims
    ds["rcv_signal_library"] = ds["rcv_signal_library"].expand_dims(
        {"snr": ds.sizes["snr"]}, axis=0
    )

    # Chunk rcv_signal_library according to lon/lat regions
    max_size = 0.5 * 1e9
    var = ds.rcv_signal_library
    nregion_lon = get_region_number(
        nregion_max=ds.sizes["lon"],
        var=var,
        max_size_bytes=max_size,
    )
    nregion_lat = get_region_number(
        nregion_max=ds.sizes["lat"],
        var=var,
        max_size_bytes=max_size,
    )
    # Save nregions
    ds.attrs["nregion_lon"] = nregion_lon
    ds.attrs["nregion_lat"] = nregion_lat

    chunksize = dict(ds.rcv_signal_library.chunksizes)
    lat_chunksize = int(ds.sizes["lat"] // nregion_lat)
    lon_chunksize = int(ds.sizes["lon"] / nregion_lon)
    chunksize = {
        "snr": ds.sizes["snr"],
        "idx_rcv": ds.sizes["idx_rcv"],
        "lat": lat_chunksize,
        "lon": lon_chunksize,
        "library_signal_time": ds.sizes["library_signal_time"],
    }
    ds["rcv_signal_library"] = ds.rcv_signal_library.chunk(chunksize)
    ds.rcv_signal_library.encoding["chunks"] = tuple(chunksize.values())
    ds.rcv_signal_library.encoding["preferred_chunks"] = tuple(chunksize.values())

    # Rechunk to ensure dask chunks are aligned with the encoding chunks
    for v in list(ds):
        if ds[v].chunks is not None:  # if var is a dask array
            ds[v] = ds[v].chunk(ds[v].encoding["preferred_chunks"])

    # Save previously computed data
    ds.to_zarr(ds.output_path, mode="w", compute=True)

    # Init corr for library signal
    ds = init_corr_library(ds)
    # Expand library_corr to snr dims
    ds["library_corr"] = ds["library_corr"].expand_dims(
        {"snr": ds.sizes["snr"]}, axis=0
    )

    return ds


def process(
    main_ds_path,
    src_info,
    rcv_info,
    grid_info,
    dt,
    similarity_metrics,
    snrs_dB,
    n_noise_realisations=100,
    verbose=False,
):

    # Load subset and init usefull vars
    ds = init_dataset(
        main_ds_path=main_ds_path,
        src_info=src_info,
        grid_info=grid_info,
        dt=dt,
        similarity_metrics=similarity_metrics,
        snrs_dB=snrs_dB,
        n_noise_realisations=n_noise_realisations,
        verbose=True,
    )

    # Quick fix for received signal at rcv position : for r = 0 -> sig = 0
    rcv_grid_lon = ds.sel(lon=ds.lon_rcv.values, method="nearest").lon.values
    rcv_grid_lat = ds.sel(lat=ds.lat_rcv.values, method="nearest").lat.values

    for i in range(len(rcv_grid_lon)):
        ds["rcv_signal_library"].loc[dict(lon=rcv_grid_lon[i], lat=rcv_grid_lat[i])] = (
            np.nan
        )

    # from time import time
    # t0 = time()
    # n_workers = 8
    # with Client(n_workers=n_workers, threads_per_worker=1) as client:
    #     print(client.dashboard_link)

    # Add event to the dataset
    ds = add_event(ds, src_info, rcv_info, apply_delay=True, verbose=verbose)

    # Init ambiguity surface
    ds = init_ambiguity_surface(ds)
    ds_no_noise = ds.copy(deep=True)

    # Save to zarr without computing
    ds_no_noise.to_zarr(ds_no_noise.output_path, mode="a", compute=False)
    # no_noise_lib = np.copy(ds_no_noise.rcv_signal_library.isel(snr=0).values)
    # no_noise_event = np.copy(ds_no_noise.rcv_signal_event.isel(snr=0).values)

    no_noise_lib = ds_no_noise.rcv_signal_library.isel(snr=0).data
    no_noise_event = ds_no_noise.rcv_signal_event.isel(snr=0).data

    # Loop over the snr values
    for idx_snr, snr_dB_i in enumerate(snrs_dB):

        # Add noise to library signal
        ds = ds_no_noise.isel(snr=idx_snr)

        # ds["rcv_signal_library"].values = np.copy(
        #     no_noise_lib
        # )  # Reset to the original signal

        ds["rcv_signal_library"].values = (
            no_noise_lib.copy()
        )  # Reset to the original signal
        ds = add_noise_to_library(ds, idx_snr=idx_snr, snr_dB=snr_dB_i, verbose=verbose)
        # Derive correlation vector for the entire grid
        ds = add_correlation_library(ds, idx_snr=idx_snr, verbose=verbose)

        # Loop over different realisation of noise for a given SNR
        for i in range(n_noise_realisations):
            print(f"## Monte Carlo iteration {i+1}/{n_noise_realisations} ##")

            # Reset to the original signal
            ds["rcv_signal_event"].values = no_noise_event.copy()
            # ds["rcv_signal_event"].values = np.copy(no_noise_event)
            ds = add_noise_to_event(
                ds, idx_snr=idx_snr, snr_dB=snr_dB_i, verbose=verbose
            )
            # Derive cross-correlation vector for each source position
            ds = add_correlation_event(ds, idx_snr=idx_snr, verbose=verbose)

            for i_sim_metric in range(len(similarity_metrics)):
                # Compute ambiguity surface
                ds = add_ambiguity_surf(
                    ds,
                    idx_snr=idx_snr,
                    idx_similarity_metric=i_sim_metric,
                    i_noise=i,
                    verbose=verbose,
                )

    # Reload full dataset
    ds = xr.open_dataset(ds.output_path, engine="zarr", chunks={})

    # print(f"Process duration: {time()-t0} s")

    return ds


if __name__ == "__main__":

    from signals.signals import pulse, generate_ship_signal
    from signals.AcousticComponent import AcousticSource
    from localisation.verlinden.misc.params import ROOT_DATASET

    testcase = "testcase3_1"
    root_dir = os.path.join(
        ROOT_DATASET,
        testcase,
    )
    root_propa = os.path.join(root_dir, "propa")
    root_propa_grid = os.path.join(root_dir, "propa_grid")
    root_propa_grid_src = os.path.join(root_dir, "propa_grid_src")

    fname = "propa_grid_src_65.5523_65.9926_-27.7023_-27.4882_100_100_ship.zarr"
    fpath = os.path.join(root_propa_grid_src, fname)

    dt = 7
    v_knots = 20  # 20 knots
    v_ship = v_knots * 1852 / 3600  # m/s

    z_src = 5
    route_azimuth = 45  # North-East route

    fs = 100
    duration = 200  # 1000 s
    nmax_ship = 1
    src_stype = "ship"

    rcv_info = {
        "id": ["RR45", "RR48", "RR44"],
        # "id": ["RRpftim0", "RRpftim1", "RRpftim2"],
        # "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)

    initial_ship_pos = {
        "lon": rcv_info["lons"][0],
        "lat": rcv_info["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    event_pos_info = {
        "speed": v_ship,
        "depth": z_src,
        "duration": duration,
        "signal_type": src_stype,
        "max_nb_of_pos": nmax_ship,
        "route_azimuth": route_azimuth,
        "initial_pos": initial_ship_pos,
    }
    # Event
    f0_event = 5  # Fundamental frequency of the ship signal
    event_sig_info = {
        "sig_type": "ship",
        "f0": f0_event,
        "std_fi": f0_event * 10 / 100,
        "tau_corr_fi": 1 / f0_event,
        "fs": fs,
    }

    src_sig, t_src_sig = generate_ship_signal(
        Ttot=dt,
        f0=event_sig_info["f0"],
        std_fi=event_sig_info["std_fi"],
        tau_corr_fi=event_sig_info["tau_corr_fi"],
        fs=event_sig_info["fs"],
    )

    src_sig *= np.hanning(len(src_sig))
    nfft = None
    min_waveguide_depth = 5000
    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="ship",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )
    event_sig_info["src"] = src

    src_info = {}
    src_info["pos"] = event_pos_info
    src_info["sig"] = event_sig_info

    lon, lat = rcv_info["lons"][0], rcv_info["lats"][0]
    dlon, dlat = get_bathy_grid_size(lon, lat)

    grid_offset_cells = 40

    grid_info = dict(
        offset_cells_lon=grid_offset_cells,
        offset_cells_lat=grid_offset_cells,
        dx=100,
        dy=100,
        dlat_bathy=dlat,
        dlon_bathy=dlon,
    )

    n_noise_realisations = 1
    # snr = np.arange(-10, 5, 1)
    # n_noise_realisations = 1
    snr = [200]
    ds = process(
        main_ds_path=fpath,
        src_info=src_info,
        rcv_info=rcv_info,
        grid_info=grid_info,
        dt=dt,
        similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
        snrs_dB=snr,
        n_noise_realisations=n_noise_realisations,
    )

    snrs = ds.snr.values
    similarity_metrics = ds.similarity_metrics.values

    plot_info = {
        "plot_video": False,
        "plot_one_tl_profile": False,
        "plot_ambiguity_surface_dist": False,
        "plot_received_signal": True,
        "plot_emmited_signal": True,
        "plot_ambiguity_surface": True,
        "plot_ship_trajectory": True,
        "plot_pos_error": False,
        "plot_correlation": True,
        "tl_freq_to_plot": [20],
        "lon_offset": 0.005,
        "lat_offset": 0.005,
        "n_instant_to_plot": 10,
        "n_rcv_signals_to_plot": 2,
    }

    analysis(
        fpath=ds.output_path,
        snrs=snrs,
        similarity_metrics=similarity_metrics,
        grid_info=grid_info,
        plot_info=plot_info,
    )
