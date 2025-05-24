import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.acoustic_source import ZcallInterferer
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import (
    LocalizationProcessor,
)
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
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

    n_mc = 1
    search_area_length = 0.5 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Flags
    check = True
    debug = False
    verbose = True
    use_weighted_rtf = False

    # Ship signal plot properties
    nperseg_plot = 2**9
    noverlap_plot = 2**8

    """ Interferer simulation : 
    Library ships = 1
    Sediment = coarse sediment 
    dr_bathy = 100 m 
    """

    name = "dw_real_demo_interferer"
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
        library_ship=library_ship,
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

    # Define interferer ABW
    rng = np.random.default_rng(seed=36)
    x_abw = simu.grid_x[rng.integers(low=0, high=len(simu.grid_x), size=1)[0]]
    y_abw = simu.grid_y[rng.integers(low=0, high=len(simu.grid_y), size=1)[0]]
    z_abw = 5

    print("Interferer position : x={}m, y={}m, z={}m".format(x_abw, y_abw, z_abw))
    interferer = ZcallInterferer(
        name="ABW_zcall",
        fs=p.fs,
        duration=p.duration,
        root_img=p.root_img_interference,
        x=x_abw,
        y=y_abw,
        z=z_abw,
        start_offset_seconds=0,
        stop_offset_seconds=0,
        sl=5,
        M=15,
    )
    interferer.plot_signal()
    interferer.plot_spectrum()
    interferer.plot_psd()
    interferer.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)

    simu.interferer = interferer
    simu.sir = 30

    # Set testcase environment
    test_case = DeepWaterRealEnv(
        simulation=simu, mode="run", name=name, n_bathy_subsample=20
    )

    # Build dataset
    t0 = time()
    db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # print("Grid dataset")
    # db.grid_dataset()
    db.build_signal()
    print(f"Time to build dataset : {time() - t0:.2f} s")

    # # Process localization
    # snrs = np.arange(-10, 16, 1)[::-1]
    snrs = [10]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # Evaluate rtf estimation mean error for each frequency bin
    # 1) Load processed data
