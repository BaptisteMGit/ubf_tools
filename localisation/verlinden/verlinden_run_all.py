from verlinden_process import verlinden_main
from verlinden_analysis import analysis_main
from verlinden_utils import load_rhumrum_obs_pos
from testcases.testcase_envs import (
    testcase1_0,
    testcase1_1,
    testcase1_2,
    testcase1_3,
    testcase1_4,
    testcase2_1,
)

if __name__ == "__main__":
    # snr = [None]
    # detection_metric = ["intercorr0"]
    testcase = testcase2_1
    # snr = [-10, -5, 5, 10, None]
    snr = [-5, 10]
    # src_signal_type = ["pulse"]
    src_signal_type = ["ship"]
    detection_metric = ["intercorr0", "hilbert_env_intercorr0"]

    obs_info = {
        "id": ["RR48", "RR45"],
        "lons": [],
        "lats": [],
    }

    for obs_id in obs_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        obs_info["lons"].append(pos_obs.lon)
        obs_info["lats"].append(pos_obs.lat)

    depth = 150  # Depth m
    v_knots = 20  # 20 knots
    v_ship = v_knots * 1852 / 3600  # m/s
    # v_ship = 50 / 3.6  # m/s

    # x_pos_ship = [7000, 10000]
    # y_pos_ship = [50000, 35000]

    initial_ship_pos = {
        "lon": 65.44,
        "lat": -27.08,
        "crs": "WGS84",
    }  # CRS might not be usefull directly but it is better to keep track of the coordinates system used
    z_src = 5
    nmax_ship = 100
    duration = 3600  # 1 hour
    route_azimuth = 120  # East

    grid_info = dict(
        Lx=1 * 1e3,
        Ly=1 * 1e3,
        dx=100,
        dy=100,
    )

    # x_pos = ([-1800, 3000],)
    # y_pos = ([8000, 2500],)
    for src_stype in src_signal_type:
        # Define the parameters
        # src_info = dict(
        #     x_pos=x_pos_ship,
        #     y_pos=y_pos_ship,
        #     v_src=v_ship,
        #     nmax_ship=nmax_ship,
        #     src_signal_type=src_stype,
        #     z_src=5,
        #     on_grid=False,
        # )
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
            obs_info=obs_info,
            snr=snr,
            detection_metric=detection_metric,
            min_waveguide_depth=depth,
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
            "x_offset": 1000,
            "y_offset": 1000,
        }

        # Analyse the results
        # snr = [-20, -10, -5, 0, 5, 10, 20, None]
        analysis_main(
            snr,
            detection_metric,
            testcase_name=testcase_name,
            simulation_info=simulation_info,
            grid_info=grid_info,
            plot_info=plot_info,
        )
