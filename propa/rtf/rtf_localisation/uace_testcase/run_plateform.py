import os
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import (
    LocalizationProcessor,
)
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
    DeepWaterPekerisMunk,
    DeepWaterPekerisRhumrumSSP,
    DeepWaterRealEnv,
)
from publication.publication_figure import PubFigure

if __name__ == "__main__":

    ### Common properties ###
    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    # antenna.plot_antenna()
    # plt.savefig("antenna")

    n_mc = 100
    search_area_length = 1 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Flags
    check = False
    debug = False
    verbose = True
    use_weighted_rtf = True

    # Ship signal plot properties
    nperseg_plot = 2**9
    noverlap_plot = 2**8

    """ Simulation 1 : single library ship 
    Library ships = 1
    Sediment = coarse sediment 
    dr_bathy = 100 m 

    """

    name = "dw_real_env_single_ship_fs_100"
    fs = 100

    # Use single library ship
    library_ship = [p.unique_library_ship]

    # Build simulation object
    simu = Simulation(
        name=name,
        fs=fs,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg=nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
        verbose=verbose,
    )

    # Plot library ships
    root_img_library_ship = os.path.join(simu.img_folder, "library_sources")
    if not os.path.exists(root_img_library_ship):
        os.makedirs(root_img_library_ship)

    for library_ship_i in library_ship:
        library_ship_i.root_img = root_img_library_ship
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)
        plt.close("all")

    # Set testcase environment
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    # Build dataset
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # Process localization
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    """ Simulation 2 : multi-ship library 
    Library ships = 1
    Sediment = coarse sediment 
    dr_bathy = 100 m
    fs = 100 Hz
    """

    name = "dw_real_env_multi_ships_fs_100"
    fs = 100

    # Library ships used
    library_ship = p.library_ship

    # Build simulation object
    simu = Simulation(
        name=name,
        fs=fs,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg=nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
        verbose=verbose,
    )

    # Plot library ships
    root_img_library_ship = os.path.join(simu.img_folder, "library_sources")
    if not os.path.exists(root_img_library_ship):
        os.makedirs(root_img_library_ship)

    for library_ship_i in library_ship:
        library_ship_i.root_img = root_img_library_ship
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)
        plt.close("all")

    # Set testcase environment
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    # Build dataset
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # Process localization
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # Same as above but with fs=200
    """ Simulation 1 : single library ship 
    Library ships = 1
    Sediment = coarse sediment 
    dr_bathy = 100 m 
    fs = 200 Hz

    """

    name = "dw_real_env_single_ship_fs_200"
    fs = 200

    # Use single library ship
    library_ship = [p.unique_library_ship]

    # Build simulation object
    simu = Simulation(
        name=name,
        fs=fs,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg=nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
        verbose=verbose,
    )

    # Plot library ships
    root_img_library_ship = os.path.join(simu.img_folder, "library_sources")
    if not os.path.exists(root_img_library_ship):
        os.makedirs(root_img_library_ship)

    for library_ship_i in library_ship:
        library_ship_i.root_img = root_img_library_ship
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)
        plt.close("all")

    # Set testcase environment
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    # Build dataset
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # Process localization
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    """ Simulation 2 : multi-ship library 
    Library ships = 5
    Sediment = coarse sediment 
    dr_bathy = 100 m
    fs = 200 Hz
    """

    name = "dw_real_env_multi_ships_fs_200"
    fs = 200

    # Library ships used
    library_ship = p.library_ship

    # Build simulation object
    simu = Simulation(
        name=name,
        fs=fs,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg=nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
        verbose=verbose,
    )

    # Plot library ships
    root_img_library_ship = os.path.join(simu.img_folder, "library_sources")
    if not os.path.exists(root_img_library_ship):
        os.makedirs(root_img_library_ship)

    for library_ship_i in library_ship:
        library_ship_i.root_img = root_img_library_ship
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)
        plt.close("all")

    # Set testcase environment
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    # Build dataset
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # Process localization
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # # # TODO : remove this
    # import xarray as xr
    # ship_dist = xr.open_dataset(os.path.join(simu.data_folder, "library_ship_distribution.nc"))
    # amb_surf_fa = xr.open_dataset(os.path.join(simu.data_folder, "from_signal_dx20m_dy20m",
    #                                            "snr_50.0dB", "localization_dx20m_dy20m_fullarray_s1_s2_s3_s4_s5_s6.nc"))

    # fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True)
    # axs = axs.flatten()
    # ships_idx = np.unique(ship_dist.library_ship_id.values)
    # for iship in ships_idx:
    #     mask_i = np.where(ship_dist.library_ship_id.values == iship, 1, np.nan)
    #     # mask_i = (ship_dist.library_ship_id.values == iship).astype(int)
    #     amb_surf_fa_i = amb_surf_fa.d_rtf * mask_i
    #     amb_surf_fa_i.plot(x="x", y="y", cmap="jet", ax=axs[iship], vmin=-6, vmax=0)
    #     axs[iship].set_title(f"ship n°{iship}")

    # amb_surf_fa.d_rtf.plot(x="x", y="y", cmap="jet", ax=axs[-1],  vmin=-6, vmax=0)
    # axs[-1].set_title("Ambiguity surface")
    # fpath = os.path.join(simu.img_folder, "q_rtf_masked.png")
    # plt.savefig(fpath)
