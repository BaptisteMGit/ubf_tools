#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   to_run.py
@Time    :   2024/05/07 11:31:36
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
from signals import pulse, generate_ship_signal
from localisation.verlinden.misc.AcousticComponent import AcousticSource
from localisation.verlinden.testcases.testcase_envs import (
    TestCase1_0,
    TestCase1_1,
    TestCase1_3,
    TestCase1_4,
    TestCase2_1,
    TestCase2_2,
    TestCase3_1,
)
from localisation.verlinden.plateform.run_plateform import run_on_plateform
from signals import pulse, generate_ship_signal
from localisation.verlinden.misc.AcousticComponent import AcousticSource
from localisation.verlinden.misc.params import ROOT_DATASET

from localisation.verlinden.plateform.process_loc import process
from localisation.verlinden.plateform.analysis_loc import analysis
from localisation.verlinden.misc.verlinden_utils import (
    load_rhumrum_obs_pos,
    get_bathy_grid_size,
)


def test():
    rcv_info_dw = {
        "id": ["RRdebug0", "RRdebug1"],
        "lons": [],
        "lats": [],
    }
    tc = TestCase3_1()
    min_dist = 15 * 1e3
    dx, dy = 100, 100

    # Define source signal
    dt = 1
    min_waveguide_depth = 5000
    f0, fs = 5, 10
    nfft = int(fs * dt)
    src_sig, t_src_sig = pulse(T=dt, f=f0, fs=fs)

    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="debug_pulse",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    (
        fullpath_dataset_propa,
        fullpath_dataset_propa_grid,
        fullpath_dataset_propa_grid_sr,
    ) = run_on_plateform(
        rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
    )


def run_swir(rcv_id):
    rcv_info_dw = {
        # "id": ["RR41", "RR42", "RR43", "RR44", "RR45", "RR46", "RR47", "RR48"],
        # "id": ["RR44", "RR45", "RR48"],
        # "id": ["R1", "R2", "R3"],
        "id": rcv_id,
        "lons": [],
        "lats": [],
    }
    tc = TestCase3_1()
    # tc.name = "testcase_3_10"
    min_dist = 5 * 1e3
    dx, dy = 100, 100

    # Define source signal
    min_waveguide_depth = 5000
    dt = 10
    fs = 100  # Sampling frequency
    f0_lib = 1  # Fundamental frequency of the ship signal
    src_info = {
        "sig_type": "ship",
        "f0": f0_lib,
        "std_fi": f0_lib * 1 / 100,
        "tau_corr_fi": 1 / f0_lib,
        "fs": fs,
    }
    src_sig, t_src_sig = generate_ship_signal(
        Ttot=dt,
        f0=src_info["f0"],
        std_fi=src_info["std_fi"],
        tau_corr_fi=src_info["tau_corr_fi"],
        fs=src_info["fs"],
    )

    src_sig *= np.hanning(len(src_sig))
    nfft = None
    # nfft = 2**3
    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="ship",
        waveguide_depth=min_waveguide_depth,
        nfft=nfft,
    )

    print(f"nfft = {src.nfft}")
    (
        ds,
        fullpath_dataset_propa,
        fullpath_dataset_propa_grid,
        fullpath_dataset_propa_grid_sr,
    ) = run_on_plateform(
        rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
    )

    return ds


def common_process_loc(ds, rcv_id):

    # Set path to gridded dataset
    # testcase = "testcase3_1"
    # root_dir = os.path.join(
    #     ROOT_DATASET,
    #     testcase,
    # )
    # root_propa = os.path.join(root_dir, "propa")
    # root_propa_grid = os.path.join(root_dir, "propa_grid")
    # root_propa_grid_src = os.path.join(root_dir, "propa_grid_src")

    # # fname = "propa_grid_src_65.5523_65.9926_-27.7023_-27.4882_100_100_ship.zarr"
    # fname = "propa_grid_src_65.5624_65.9825_-27.6933_-27.4972_100_100_ship.zarr"
    # fpath = os.path.join(root_propa_grid_src, fname)
    fpath = ds.fullpath_dataset_propa_grid_src

    # Source infos
    z_src = 5
    v_knots = 20  # 20 knots
    v_ship = v_knots * 1852 / 3600  # m/s
    route_azimuth = 45  # North-East route

    duration = 200  # 1000 s
    nmax_ship = 1
    src_stype = "ship"

    # Receiver infos
    rcv_info = {
        "id": rcv_id,
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)

    # Set initial position of the source
    initial_ship_pos = {
        # "lon": rcv_info["lons"][0],
        # "lat": rcv_info["lats"][0] + 0.001,
        "lon": rcv_info["lons"][0] + 0.1,
        "lat": rcv_info["lats"][0] - 0.03,
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

    lon, lat = rcv_info["lons"][0], rcv_info["lats"][0]
    dlon, dlat = get_bathy_grid_size(lon, lat)

    grid_offset_cells = 200
    # grid_offset_cells = 10

    grid_info = dict(
        offset_cells_lon=grid_offset_cells,
        offset_cells_lat=grid_offset_cells,
        dx=100,
        dy=100,
        dlat_bathy=dlat,
        dlon_bathy=dlon,
    )

    return fpath, event_pos_info, grid_info, rcv_info


def set_event_sig_info(f0):
    # Event
    dt = 7
    fs = 100
    # f0 = 1.5  # Fundamental frequency of the ship signal
    event_sig_info = {
        "sig_type": "ship",
        "f0": f0,
        "std_fi": f0 * 1 / 100,
        "tau_corr_fi": 1 / f0,
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
    # src_sig *= np.hamming(len(src_sig))

    min_waveguide_depth = 5000
    src = AcousticSource(
        signal=src_sig,
        time=t_src_sig,
        name="ship",
        waveguide_depth=min_waveguide_depth,
        nfft=2**10,
    )
    event_sig_info["src"] = src

    return dt, fs, event_sig_info


def process_analysis(ds, grid_info):
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
        mode="publication",
    )


def run_process_loc(ds, rcv_id):

    # f0_library = 1
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc()
    # f0_library = 1
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc()

    # """ Single test to ensure everything is ok """
    # n_noise = 1
    # snr = [-10, 0, 5]
    # f0 = f0_library
    # dt, fs, event_sig_info = set_event_sig_info(f0)
    # src_info = {}
    # src_info["pos"] = event_pos_info
    # src_info["sig"] = event_sig_info

    # ds = process(
    #     main_ds_path=fpath,
    #     src_info=src_info,
    #     rcv_info=rcv_info,
    #     grid_info=grid_info,
    #     dt=dt,
    #     similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
    #     snrs_dB=snr,
    #     n_noise_realisations=n_noise,
    #     verbose=True,
    # )

    # process_analysis(ds, grid_info)

    # n_noise = 20
    # f0_library = 1
    # snr = [-10]
    # # snr = np.arange(-15, 10, 0.5)
    # # snr = np.arange(-10, 5, 1)
    # # snr = [-10, 5]
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)

    # """ Test with same spectral content """
    # f0 = f0_library
    # dt, fs, event_sig_info = set_event_sig_info(f0)
    # src_info = {}
    # src_info["pos"] = event_pos_info
    # src_info["sig"] = event_sig_info

    # ds = process(
    #     main_ds_path=fpath,
    #     src_info=src_info,
    #     rcv_info=rcv_info,
    #     grid_info=grid_info,
    #     dt=dt,
    #     similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
    #     snrs_dB=snr,
    #     n_noise_realisations=n_noise,
    #     verbose=True,
    # )

    # process_analysis(ds, grid_info)

    n_noise = 20
    f0_library = 1
    snr = [-5]
    # snr = np.arange(-15, 10, 0.5)
    # snr = np.arange(-20, 0, 1)
    # snr = [-10, 5]
    fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)

    """ Test with same spectral content """
    f0 = f0_library
    dt, fs, event_sig_info = set_event_sig_info(f0)
    src_info = {}
    src_info["pos"] = event_pos_info
    src_info["sig"] = event_sig_info

    ds = process(
        main_ds_path=fpath,
        src_info=src_info,
        rcv_info=rcv_info,
        grid_info=grid_info,
        dt=dt,
        similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
        snrs_dB=snr,
        n_noise_realisations=n_noise,
        verbose=True,
    )

    process_analysis(ds, grid_info)

    # n_noise = 200
    # f0_library = 1
    # # snr = [0]
    # # snr = np.arange(-15, 10, 0.5)
    # snr = np.arange(-10, 5, 0.5)
    # # snr = [-10, 5]
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc(rcv_id)

    # """ Test with same spectral content """
    # f0 = f0_library
    # dt, fs, event_sig_info = set_event_sig_info(f0)
    # src_info = {}
    # src_info["pos"] = event_pos_info
    # src_info["sig"] = event_sig_info

    # ds = process(
    #     main_ds_path=fpath,
    #     src_info=src_info,
    #     rcv_info=rcv_info,
    #     grid_info=grid_info,
    #     dt=dt,
    #     similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
    #     snrs_dB=snr,
    #     n_noise_realisations=n_noise,
    #     verbose=True,
    # )

    # process_analysis(ds, grid_info)

    # """ Test with different spectral content """

    # n_noise = 100
    # snr = [15]
    # f0 = 1.5*f0_library
    # dt, fs, event_sig_info = set_event_sig_info(f0)
    # src_info = {}
    # src_info["pos"] = event_pos_info
    # src_info["sig"] = event_sig_info

    # ds = process(
    #     main_ds_path=fpath,
    #     src_info=src_info,
    #     rcv_info=rcv_info,
    #     grid_info=grid_info,
    #     dt=dt,
    #     similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
    #     snrs_dB=snr,
    #     n_noise_realisations=n_noise,
    #     verbose=True,
    # )

    # process_analysis(ds, grid_info)


def run_all_testcases():
    rcv_id = ["R1", "R2", "R3"]

    dt = 10
    fs = 100  # Sampling frequency
    f0_lib = 1.5  # Fundamental frequency of the ship signal
    src_info = {
        "sig_type": "ship",
        "f0": f0_lib,
        "std_fi": f0_lib * 1 / 100,
        "tau_corr_fi": 1 / f0_lib,
        "fs": fs,
    }
    src_sig, t_src_sig = generate_ship_signal(
        Ttot=dt,
        f0=src_info["f0"],
        std_fi=src_info["std_fi"],
        tau_corr_fi=src_info["tau_corr_fi"],
        fs=src_info["fs"],
    )
    src_sig *= np.hanning(len(src_sig))

    # snr = np.arange(-10, 0, 1)  # [10]
    snr = [15]
    n_noise = 1
    dt, fs, event_sig_info = set_event_sig_info(f0_lib)

    for tc in [
        # TestCase1_0,
        # TestCase1_1,
        # TestCase1_3,
        TestCase1_4,
        # TestCase2_1,
        # TestCase2_2,
        # TestCase3_1,
    ]:

        rcv_info_dw = {
            "id": rcv_id,
            "lons": [],
            "lats": [],
        }

        tc_var_in = {"max_range_m": 30 * 1e3}
        tc = tc()
        tc.update(tc_var_in)

        min_dist = 5 * 1e3
        dx, dy = 100, 100

        # Define source signal
        src = AcousticSource(
            signal=src_sig,
            time=t_src_sig,
            name="ship",
            waveguide_depth=tc.min_depth,
            # nfft=5,
        )

        print(f"nfft = {src.nfft}")
        (
            ds,
            fullpath_dataset_propa,
            fullpath_dataset_propa_grid,
            fullpath_dataset_propa_grid_sr,
        ) = run_on_plateform(
            rcv_info=rcv_info_dw, testcase=tc, min_dist=min_dist, dx=dx, dy=dy, src=src
        )

        # Process loc
        fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)

        src_info = {}
        src_info["pos"] = event_pos_info
        src_info["sig"] = event_sig_info

        ds = process(
            main_ds_path=fpath,
            src_info=src_info,
            rcv_info=rcv_info,
            grid_info=grid_info,
            dt=dt,
            similarity_metrics=["intercorr0", "hilbert_env_intercorr0"],
            snrs_dB=snr,
            n_noise_realisations=n_noise,
            verbose=True,
        )
        # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process\testcase1_0_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240628_063914.zarr"
        # ds = xr.open_dataset(fpath, engine="zarr", chunks={})
        # fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)

        process_analysis(ds, grid_info)


if __name__ == "__main__":
    import xarray as xr

    rcv_id = ["R1", "R2", "R3"]

    # # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process\testcase1_3_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240628_075448.zarr"
    # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process\testcase1_4_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240630_125938.zarr"
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)
    # process_analysis(ds, grid_info)

    run_all_testcases()
    # Build dataset
    # rcv_id = ["R1", "R2", "R3", "R4"]
    # rcv_id = ["RR41", "RR44", "RR45", "RR47"]

    # rcv_id = ["RRdebug0", "RRdebug1"]
    # ds = run_swir(rcv_id)

    # # Exploit dataset for localisation
    # run_process_loc(ds, rcv_id)

    # Process
    # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_dataset\testcase3_1_AC198EBFF716\propa_grid_src\propa_grid_src_65.4903_65.6797_-27.7342_-27.5758_100_100_ship.zarr"
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})
    # run_process_loc(ds, rcv_id)

    # # Reanalyse
    # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_dataset\testcase3_1_AC198EBFF716\propa_grid_src\propa_grid_src_65.4903_65.6797_-27.7342_-27.5758_100_100_ship.zarr"
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})
    # fpath, event_pos_info, grid_info, rcv_info = common_process_loc(ds, rcv_id)

    # # Processed dataset
    # fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\loc\localisation_process\testcase3_1_AC198EBFF716\65.4656_65.8692_-27.8930_-27.5339_ship\20240609_171854.zarr"
    # ds = xr.open_dataset(fpath, engine="zarr", chunks={})
    # process_analysis(ds, grid_info)
