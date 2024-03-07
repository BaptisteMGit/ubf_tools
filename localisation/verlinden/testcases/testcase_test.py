#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   testcase_test.py
@Time    :   2024/03/07 09:15:39
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from localisation.verlinden.verlinden_process import verlinden_main
from localisation.verlinden.verlinden_analysis import analysis_main
from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos
from testcase_envs import (
    testcase1_0,
    testcase1_1,
    testcase1_2,
    testcase1_3,
    testcase1_4,
    testcase2_0,
    testcase2_1,
    testcase2_2,
    testcase3_1,
)


def run_tc(testcase, rcv_info, initial_ship_pos, debug=False):

    detection_metric = ["intercorr0"]

    # depth = 150  # Depth m
    v_knots = 20  # 20 knots
    v_ship = v_knots * 1852 / 3600  # m/s

    z_src = 5
    route_azimuth = 45  # North-East route

    if debug:
        src_signal_type = ["debug_pulse"]
        snr = [0]
        duration = 200  # 1000 s
        nmax_ship = 10
        grid_info = dict(
            offset_cells_lon=1,
            offset_cells_lat=1,
            dx=100,
            dy=100,
        )

    else:
        src_signal_type = ["ship"]
        snr = [0]
        duration = 200  # 1000 s
        nmax_ship = 10

        grid_info = dict(
            offset_cells_lon=1,
            offset_cells_lat=1,
            dx=100,
            dy=100,
        )

    for src_stype in src_signal_type:
        # Define the parameters
        src_info = {
            "speed": v_ship,
            "depth": z_src,
            "duration": duration,
            "signal_type": src_stype,
            "max_nb_of_pos": nmax_ship,
            "route_azimuth": route_azimuth,
            "initial_pos": initial_ship_pos,
        }

        # Run all the process
        simu_folder, testcase_name = verlinden_main(
            testcase=testcase,
            src_info=src_info,
            grid_info=grid_info,
            rcv_info=rcv_info,
            snr=snr,
            detection_metric=detection_metric,
        )

        simulation_info = {
            "simulation_folder": simu_folder,
            "src_pos": "not_on_grid",
            "n_instant_to_plot": 10,
            "n_rcv_signals_to_plot": 2,
            "src_type": src_stype,
        }

        plot_info = {
            "plot_video": False,
            "plot_one_tl_profile": False,
            "plot_ambiguity_surface_dist": True,
            "plot_received_signal": True,
            "plot_ambiguity_surface": True,
            "plot_ship_trajectory": True,
            "plot_pos_error": True,
            "plot_correlation": True,
            "tl_freq_to_plot": [20],
            "lon_offset": 0.05,
            "lat_offset": 0.05,
        }

        # Analyse the results
        analysis_main(
            snr,
            detection_metric,
            testcase_name=testcase_name,
            simulation_info=simulation_info,
            grid_info=grid_info,
            plot_info=plot_info,
        )


def run_tests(run_mode):

    if run_mode == "normal":
        debug = False
    elif run_mode == "debug":
        debug = True

    # Receiver info for shallow water test cases
    rcv_info_sw = {
        "id": ["R0", "R1"],
        "lons": [-4.87, -4.85],
        "lats": [52.52, 52.52],
    }
    # Initial ship position for shallow water test cases
    initial_ship_pos_sw = {
        "lon": rcv_info_sw["lons"][0] - 0.05,
        "lat": rcv_info_sw["lats"][0] + 0.05,
        "crs": "WGS84",
    }

    # Receiver info for deep water test cases
    rcv_info_dw = {
        "id": ["RR45", "RR48"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)

    initial_ship_pos_dw = {
        "lon": rcv_info_dw["lons"][0],
        "lat": rcv_info_dw["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    # Test case 1.0
    run_tc(
        testcase=testcase1_0,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 1.1
    run_tc(
        testcase=testcase1_1,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 1.2
    run_tc(
        testcase=testcase1_2,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 1.3
    run_tc(
        testcase=testcase1_3,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 1.4
    run_tc(
        testcase=testcase1_4,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 2.0
    run_tc(
        testcase=testcase2_0,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 2.1
    run_tc(
        testcase=testcase2_1,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 2.2
    run_tc(
        testcase=testcase2_2,
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        debug=debug,
    )

    # Test case 3.1
    run_tc(
        testcase=testcase3_1,
        rcv_info=rcv_info_dw,
        initial_ship_pos=initial_ship_pos_dw,
        debug=debug,
    )


if __name__ == "__main__":

    # run_mode = "normal"
    run_mode = "debug"
    run_tests(run_mode)
