import os
import numpy as np
import matplotlib.pyplot as plt

from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import LocalizationProcessor
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
    DeepWaterPekerisMunk,
    DeepWaterPekerisRhumrumSSP,
    DeepWaterRealEnv,
)

antenna = SparseAntenna(
    name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
)
antenna.plot_antenna()
plt.savefig("antenna")

debug = False
check = False
n_mc = 1
use_weighted_rtf = True
name = "dw_real_env"

search_area_length = 6 * 1e3
simu = Simulation(
    name=name,
    debug=debug,
    antenna=antenna,
    check_features=check,
    monte_carlo_iterations=n_mc,
    use_weighted_rtf=use_weighted_rtf,
    search_area_length=search_area_length,
)
test_case = DeepWaterRealEnv(simulation=simu, mode="run")

db = DataBuilder(simulation=simu)
db.build_tf_dataset()
db.grid_dataset()
db.build_signal()