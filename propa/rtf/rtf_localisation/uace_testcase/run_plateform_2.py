import os
import numpy as np
import matplotlib.pyplot as plt

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.ship_signal import ShipSignal
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
    # name = "dw_pekerismunk_impulsive_response"
    name = "dw_real_env_impulsive_response"
    # name = "dw_real_env"

    # Modify library ship to perfect dirac by setting all spectrum components to 1
    library_ship = p.library_ship
    library_ship.spectrum = np.ones_like(library_ship.spectrum)
    library_ship.plot_spectrum()

    # Get corresponding signal
    library_ship.signal = np.fft.irfft(library_ship.spectrum)
    library_ship.plot_signal()

    # Do the same for the event ship signal
    event_ship = p.event_ship
    event_ship.spectrum = np.ones_like(event_ship.spectrum)
    event_ship.signal = np.fft.irfft(event_ship.spectrum)
    event_ship.plot_signal()
    event_ship.plot_spectrum()

    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        event_ship=event_ship,
        library_ship=library_ship,
        monte_carlo_iterations=n_mc,
        use_weighted_rtf=use_weighted_rtf,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name, depth_offset=0)
    # test_case = DeepWaterPekerisMunk(simulation=simu, mode="run", name=name)

    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    # print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # First run to make sure everything is ok
    snrs = [50]
    simu.monte_carlo_iterations = 1
    simu.check_features = True
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # snrs = np.arange(-10, 17, 2)[::-1]
    # simu.monte_carlo_iterations = 100
    # simu.check_features = False
    # print(f"Processing snrs : {snrs}")
    # lp = LocalizationProcessor(simulation=simu, use_dask=False)
    # lp.process_multiple_snrs(snrs=snrs, run_mode="w")
