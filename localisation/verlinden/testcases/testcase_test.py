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
import os
from localisation.verlinden.verlinden_process import verlinden_main
from localisation.verlinden.verlinden_analysis import analysis_main, compare_perf_src
from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos
from testcase_envs import (
    TestCase1_0,
    TestCase1_1,
    TestCase1_2,
    TestCase1_3,
    TestCase1_4,
    TestCase2_0,
    TestCase2_1,
    TestCase2_2,
    TestCase3_1,
)


def get_simu_info(
    testcase,
):
    simu_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\verlinden_process_output"
    testcase_name = testcase.name
    simu_folder = os.path.join(simu_root, testcase_name)
    return simu_folder, testcase_name


def run_tc(
    testcase,
    rcv_info,
    initial_ship_pos,
    snr=[],
    src_signal_type=[],
    detection_metric=[],
    grid_offset_cells=35,
    debug=False,
    re_analysis=False,
):

    # detection_metric = ["intercorr0"]
    if not detection_metric:
        detection_metric = ["hilbert_env_intercorr0"]
    if not snr:
        snr = [None]
    if not src_signal_type:
        src_signal_type = ["ship"]

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
        # src_signal_type = ["ship"]
        # snr = [0]
        # snr = [None]
        # duration = 200  # 1000 s
        # nmax_ship = 10
        duration = 200  # 1000 s
        nmax_ship = 1000

        grid_info = dict(
            offset_cells_lon=grid_offset_cells,
            offset_cells_lat=grid_offset_cells,
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

        if not re_analysis:
            # Run all the process
            simu_folder, testcase_name = verlinden_main(
                testcase=testcase,
                src_info=src_info,
                grid_info=grid_info,
                rcv_info=rcv_info,
                snr=snr,
                detection_metric=detection_metric,
            )
        else:
            simu_folder, testcase_name = get_simu_info(testcase)

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
            "plot_ambiguity_surface_dist": False,
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
    # compare_perf_src(src_type_list, simulation_info, testcase_name, snr=0)


def run_tests(run_mode, re_analysis=False):

    if run_mode == "normal":
        debug = False
    elif run_mode == "debug":
        debug = True

    # Receiver info for shallow water test cases
    rcv_info_sw = {
        "id": ["R0", "R1"],
        "lons": [-4.87, -4.8626],
        "lats": [52.52, 52.52],
    }
    # Initial ship position for shallow water test cases
    # initial_ship_pos_sw = {
    #     "lon": rcv_info_sw["lons"][0] - 0.2,
    #     "lat": rcv_info_sw["lats"][0] + 0.2,
    #     "crs": "WGS84",
    #     "route_azimuth": 45 + 90,
    # }

    initial_ship_pos_sw = {
        "lon": rcv_info_sw["lons"][0] - 0.04,
        "lat": rcv_info_sw["lats"][0] + 0.04,
        "crs": "WGS84",
        "route_azimuth": 45 + 90,
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
    src_signal_type = ["ship"]
    detection_metric = ["intercorr0"]
    # detection_metric = ["hilbert_env_intercorr0"]

    # snr = [-10, -5, 0, 5, 10, None]
    # grid_offset_cells = 80
    snr = [0]
    grid_offset_cells = 80

    run_tc(
        testcase=TestCase1_0(),
        rcv_info=rcv_info_sw,
        initial_ship_pos=initial_ship_pos_sw,
        snr=snr,
        src_signal_type=src_signal_type,
        detection_metric=detection_metric,
        grid_offset_cells=grid_offset_cells,
        debug=debug,
        re_analysis=re_analysis,
    )

    # Test case 1.1
    # run_tc(
    #     testcase=TestCase1_1(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     snr=snr,
    #     src_signal_type=src_signal_type,
    #     detection_metric=detection_metric,
    #     grid_offset_cells=grid_offset_cells,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 1.2
    # run_tc(
    #     testcase=TestCase1_2(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 1.3
    # run_tc(
    #     testcase=TestCase1_3(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 1.4
    # run_tc(
    #     testcase=TestCase1_4(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     snr=snr,
    #     src_signal_type=src_signal_type,
    #     detection_metric=detection_metric,
    #     grid_offset_cells=grid_offset_cells,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 2.0
    # run_tc(
    #     testcase=TestCase2_0(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 2.1
    # run_tc(
    #     testcase=TestCase2_1(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # # Test case 2.2
    # run_tc(
    #     testcase=TestCase2_2(),
    #     rcv_info=rcv_info_sw,
    #     initial_ship_pos=initial_ship_pos_sw,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )

    # Test case 3.1
    src_signal_type = ["ship"]
    detection_metric = ["hilbert_env_intercorr0"]
    snr = [-10, -5, 0, 5, 10, None]
    # snr = [None]
    # run_tc(
    #     testcase=TestCase3_1(),
    #     rcv_info=rcv_info_dw,
    #     initial_ship_pos=initial_ship_pos_dw,
    #     snr=snr,
    #     src_signal_type=src_signal_type,
    #     detection_metric=detection_metric,
    #     grid_offset_cells=grid_offset_cells,
    #     debug=debug,
    #     re_analysis=re_analysis,
    # )


if __name__ == "__main__":

    # from dask.distributed import LocalCluster

    # cluster = LocalCluster()  # Fully-featured local Dask cluster
    # client = cluster.get_client()

    # run_mode = "normal"
    re_analysis = False

    run_mode = "debug"
    import sys

    if hasattr(sys, "gettrace") and (sys.gettrace() is not None):
        run_mode = "debug"
    else:
        run_mode = "normal"

    run_tests(run_mode, re_analysis)
