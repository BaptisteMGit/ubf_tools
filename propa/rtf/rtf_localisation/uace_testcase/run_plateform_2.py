import os
import numpy as np
import matplotlib.pyplot as plt

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.acoustic_source import Ship
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

if __name__ == "__main__":

    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    # antenna.plot_antenna()
    # plt.savefig("antenna")

    debug = False
    check = True
    n_mc = 1
    use_weighted_rtf = True
    name = "dw_real_env"

    # Test multiple library ship
    nl_ship = 10
    library_ship = []
    for iship in range(nl_ship):
        f0_l = np.random.uniform(low=p.f0_min, high=p.f0_max + 5)
        std_fi_l = np.random.uniform(low=p.std_fi_min, high=p.std_fi_max) * f0_l
        tau_corr_fi_l = (
            np.random.uniform(low=p.tau_corr_fi_min, high=p.tau_corr_fi_max) * 1 / f0_l
        )
        library_ship_i = Ship(
            name=f"library_ship_{iship}",
            f0=f0_l,
            fs=p.fs,
            duration=p.duration,
            std_fi=std_fi_l,
            tau_corr_fi=tau_corr_fi_l,
            root_img=p.root_img_ship_sigs,
        )
        # library_ship_i.plot_signal(tmax=2)
        # library_ship_i.plot_spectrum()
        # library_ship_i.plot_psd()
        # library_ship_i.plot_stft()
        library_ship.append(library_ship_i)

    search_area_length = 0.5 * 1e3
    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        library_ship=library_ship,
        monte_carlo_iterations=n_mc,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name="dw_real_env")
    db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # db.grid_dataset()
    db.build_signal()

    snrs = [50]
    simu.monte_carlo_iterations = 1
    simu.check_features = True
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    ### Impulse response study ###
    # debug = False
    # check = True
    # n_mc = 1
    # use_weighted_rtf = True
    # # name = "dw_pekerismunk_impulsive_response"
    # name = "dw_real_env_impulsive_response"
    # # name = "dw_real_env"

    # # Modify library ship to perfect dirac by setting all spectrum components to 1
    # library_ship = p.library_ship
    # library_ship.spectrum = np.ones_like(library_ship.spectrum)
    # library_ship.plot_spectrum()

    # # Get corresponding signal
    # library_ship.signal = np.fft.irfft(library_ship.spectrum)
    # library_ship.plot_signal()

    # # Do the same for the event ship signal
    # event_ship = p.event_ship
    # event_ship.spectrum = np.ones_like(event_ship.spectrum)
    # event_ship.signal = np.fft.irfft(event_ship.spectrum)
    # event_ship.plot_signal()
    # event_ship.plot_spectrum()

    # simu = Simulation(
    #     name=name,
    #     debug=debug,
    #     antenna=antenna,
    #     check_features=check,
    #     event_ship=event_ship,
    #     library_ship=library_ship,
    #     monte_carlo_iterations=n_mc,
    #     use_weighted_rtf=use_weighted_rtf,
    # )
    # test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name, depth_offset=0)
    # # test_case = DeepWaterPekerisMunk(simulation=simu, mode="run", name=name)

    # db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # # print("Grid dataset")
    # db.grid_dataset()
    # db.build_signal()

    # # First run to make sure everything is ok
    # snrs = [50]
    # simu.monte_carlo_iterations = 1
    # simu.check_features = True
    # lp = LocalizationProcessor(simulation=simu, use_dask=False)
    # lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # snrs = np.arange(-10, 17, 2)[::-1]
    # simu.monte_carlo_iterations = 100
    # simu.check_features = False
    # print(f"Processing snrs : {snrs}")
    # lp = LocalizationProcessor(simulation=simu, use_dask=False)
    # lp.process_multiple_snrs(snrs=snrs, run_mode="w")
