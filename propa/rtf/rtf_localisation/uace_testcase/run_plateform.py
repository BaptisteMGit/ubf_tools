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

    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    # antenna.plot_antenna()
    # plt.savefig("antenna")

    name = "dw_real_env"
    n_mc = 100
    search_area_length = 1 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Library ships used
    library_ship = p.library_ship
    # Plot library ships 
    for library_ship_i in library_ship:
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft()

    # Flags
    check = True
    debug = False
    use_weighted_rtf = True

    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg= nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    db = DataBuilder(simulation=simu)
    db.build_tf_dataset()
    print("Grid dataset")
    db.grid_dataset()
    db.build_signal()

    # First run to make sure everything is ok
    snrs = [0]
    simu.monte_carlo_iterations = 1
    simu.check_features = True
    lp = LocalizationProcessor(simulation=simu, use_dask=True)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    snrs = [-6, -4, -2, 0]
    simu.monte_carlo_iterations = 10
    simu.check_features = False
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")
