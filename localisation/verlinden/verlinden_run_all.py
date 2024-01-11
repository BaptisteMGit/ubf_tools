from localisation.verlinden.verlinden_process import verlinden_main
from verlinden_analysis import analysis_main

if __name__ == "__main__":
    # snr = [-5, -1, 1, 5, 10, 20]
    snr = [-30, -20, -15, -10, -5, -1, 1, 5, 10, 20]
    detection_metric = ["intercorr0", "lstsquares", "hilbert_env_intercorr0"]
    src_signal_type = ["ship"]

    grid_info = dict(
        Lx=10 * 1e3,
        Ly=10 * 1e3,
        dx=100,
        dy=100,
    )

    obs_info = dict(
        x_obs=[0, 1500],
        y_obs=[0, 0],
    )

    depth = 150  # Depth m
    v_ship = 50 / 3.6  # m/s
    env_fname = "verlinden_1_test_case"
    env_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"

    for src_stype in src_signal_type:
        # Define the parameters
        src_info = dict(
            x_pos=[-15000, -13000],
            y_pos=[40000, 38000],
            v_src=v_ship,
            nmax_ship=30,
            src_signal_type=src_stype,
            z_src=5,
            on_grid=False,
        )

        # Run all the process
        verlinden_main(
            env_fname=env_fname,
            env_root=env_root,
            src_info=src_info,
            grid_info=grid_info,
            obs_info=obs_info,
            snr=snr,
            detection_metric=detection_metric,
            depth_max=depth,
        )

        simulation_info = {
            "simulation_folder": r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case",
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
        snr = [-30, -20, -10, -5, -1, 1, 5, 10, 20]
        analysis_main(
            snr,
            detection_metric,
            simulation_info=simulation_info,
            grid_info=grid_info,
            plot_info=plot_info,
        )
