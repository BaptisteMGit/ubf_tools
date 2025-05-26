import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.acoustic_source import (
    ZcallInterferer,
    Ship,
)
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


def run_wgn_simulation(mode="demo"):
    """Run first simulation test case for UACE 2025 paper.
    Test case name = "dw_real_wgn_testcase"

    Test case properties :
        - Sampling frequency fs = 100 Hz
        - Number of frequency bins used for RTF estimation = all
        - Number of library ships = 1
        - Type of sediment used = coarse sediment
        - Bathymetric resolution = 100 m
        - Number of Monte Carlo iteration = 100
        - Interferer : None
        - Signal duration = 20 s

    """

    print("Start running the interferer test case !")

    ### Common properties ###
    name = f"dw_real_interferer_testcase_{mode}"
    fs = 100
    duration = 20
    n_bathy_subsample = (
        1  # 1 for no subsampling (original resolution), 20 for original resolution / 20
    )

    # Antenna
    antenna = SparseAntenna(
        name="Random sparse antenna (R = 5km)",
        n_elements=6,
        random_radius=5e3,
        rng_seed=42,
    )

    n_mc = 100
    search_area_length = 1 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Flags
    check = False
    debug = False
    verbose = True
    use_weighted_rtf = False

    # Ship signal plot properties
    nperseg_plot = 2**9
    noverlap_plot = 2**8

    # Use single library ship
    f0_l = 4.889
    std_fi_l = 0.058 * f0_l
    tau_corr_fi_l = 0.067 * 1 / f0_l

    library_ship = Ship(
        name="library_ship",
        f0=f0_l,
        fs=fs,
        duration=duration,
        std_fi=std_fi_l,
        tau_corr_fi=tau_corr_fi_l,
        root_img=p.root_img_ship_sigs,
    )
    library_ship = [library_ship]

    # Event ship signal -> signal 19 de la base de signaux
    f0_e = 4.629
    std_fi_e = 0.072 * f0_e
    tau_corr_fi_e = 0.304 * 1 / f0_e
    event_ship = Ship(
        name="event_ship",
        f0=f0_e,
        fs=fs,
        duration=duration,
        std_fi=std_fi_e,
        tau_corr_fi=tau_corr_fi_e,
        root_img=p.root_img_ship_sigs,
    )
    event_ship_x = 25000
    event_ship_y = 12000
    event_ship_z = 5

    # If demo mode is selected - use a quicker configuration
    if mode == "demo":
        search_area_length = 0.5 * 1e3
        check = True
        n_mc = 3
        n_bathy_subsample = 20  # Use a lower resolution for demo mode

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
        event_ship=event_ship,
        event_ship_x=event_ship_x,
        event_ship_y=event_ship_y,
        event_ship_z=event_ship_z,
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

    # Plot antenna
    antenna.plot_antenna()
    fpath = os.path.join(simu.img_folder, "rcv_array.png")
    plt.savefig(fpath)

    # Set testcase environment
    test_case = DeepWaterRealEnv(
        simulation=simu, mode="run", name=name, n_bathy_subsample=n_bathy_subsample
    )

    # Build dataset
    t0 = time()
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()
    print(f"Time to build dataset : {time() - t0:.2f} s")

    # # Process localization
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")


def run_interferer_simulation(mode="demo"):
    """Run second simulation test case for UACE 2025 paper.
    Test case name = "dw_real_interferer_testcase"

    Test case properties :
        - Sampling frequency fs = 100 Hz
        - Number of frequency bins used for RTF estimation = all
        - Number of library ships = 1
        - Type of sediment used = coarse sediment
        - Bathymetric resolution = 100 m
        - Number of Monte Carlo iteration = 1
        - Signal to Interference Ratio (SIR) = 0 dB
        - SNR (WGN) = 5 dB

    """

    print("Start running the interferer test case !")

    ### Common properties ###
    name = f"dw_real_interferer_testcase_{mode}"
    fs = 100
    duration = 20
    n_bathy_subsample = (
        1  # 1 for no subsampling (original resolution), 20 for original resolution / 20
    )

    # Antenna
    antenna = SparseAntenna(
        name="Random sparse antenna (R = 5km)",
        n_elements=6,
        random_radius=5e3,
        rng_seed=42,
    )

    n_mc = 100
    search_area_length = 1 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Flags
    check = False
    debug = False
    verbose = True
    use_weighted_rtf = False

    # Ship signal plot properties
    nperseg_plot = 2**9
    noverlap_plot = 2**8

    # Use single library ship
    f0_l = 4.889
    std_fi_l = 0.058 * f0_l
    tau_corr_fi_l = 0.067 * 1 / f0_l

    library_ship = Ship(
        name="library_ship",
        f0=f0_l,
        fs=fs,
        duration=duration,
        std_fi=std_fi_l,
        tau_corr_fi=tau_corr_fi_l,
        root_img=p.root_img_ship_sigs,
    )
    library_ship = [library_ship]

    # Event ship signal -> signal 19 de la base de signaux
    f0_e = 4.629
    std_fi_e = 0.072 * f0_e
    tau_corr_fi_e = 0.304 * 1 / f0_e
    event_ship = Ship(
        name="event_ship",
        f0=f0_e,
        fs=fs,
        duration=duration,
        std_fi=std_fi_e,
        tau_corr_fi=tau_corr_fi_e,
        root_img=p.root_img_ship_sigs,
    )
    event_ship_x = 25000
    event_ship_y = 12000
    event_ship_z = 5

    # If demo mode is selected - use a quicker configuration
    if mode == "demo":
        search_area_length = 0.5 * 1e3
        check = True
        n_mc = 3
        n_bathy_subsample = 20  # Use a lower resolution for demo mode

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
        event_ship=event_ship,
        event_ship_x=event_ship_x,
        event_ship_y=event_ship_y,
        event_ship_z=event_ship_z,
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

    # Plot antenna
    antenna.plot_antenna()
    fpath = os.path.join(simu.img_folder, "rcv_array.png")
    plt.savefig(fpath)

    # Define interferer ABW
    rng = np.random.default_rng(seed=22)
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
        M=15,
    )
    interferer.plot_signal()
    interferer.plot_spectrum()
    interferer.plot_psd()
    interferer.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)

    simu.interferer = interferer
    simu.sir = 0

    # Set testcase environment
    test_case = DeepWaterRealEnv(
        simulation=simu, mode="run", name=name, n_bathy_subsample=n_bathy_subsample
    )

    # Build dataset
    t0 = time()
    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()
    print(f"Time to build dataset : {time() - t0:.2f} s")

    # # Process localization
    snrs = [5]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")


if __name__ == "__main__":

    # Select run mode
    mode = "demo"  # "demo" / "publi"
    # mode = "publi"

    # Select which simulation to run
    # testcase = "wgn"  # "wgn" / "interferer"
    testcase = "interferer"

    if testcase == "wgn":
        run_wgn_simulation()

    elif testcase == "interferer":
        run_interferer_simulation(mode=mode)
