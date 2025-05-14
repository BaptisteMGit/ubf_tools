import os
import numpy as np
import matplotlib.pyplot as plt

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
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
# antenna.plot_antenna()
# plt.savefig("antenna")

debug = False
check = True
n_mc = 1
use_weighted_rtf = True
name = "dw_real_env_3km"

search_area_length = 3 * 1e3
simu = Simulation(
    name=name,
    debug=debug,
    antenna=antenna,
    check_features=check,
    monte_carlo_iterations=n_mc,
    use_weighted_rtf=use_weighted_rtf,
    search_area_length=search_area_length,
)
test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

# db = DataBuilder(simulation=simu)
# db.build_tf_dataset()
# print("Grid dataset")
# db.grid_dataset()
# db.build_signal()


# First run to make sure everything is ok 
snrs = [0]
simu.monte_carlo_iterations = 1
simu.check_features = True
lp = LocalizationProcessor(simulation=simu, use_dask=False)
lp.process_multiple_snrs(snrs=snrs, run_mode="w")

# with Client(
#     n_workers=p.n_workers,
#     threads_per_worker=1,
#     memory_limit=f"{p.max_ram_per_worker_gb}GB",
# ) as client:
# Second run to derive perf vs snrs 
snrs = np.arange(-10, 17, 2)[::-1]
simu.monte_carlo_iterations = 100
simu.check_features = False
print(f"Processing snrs : {snrs}")
lp = LocalizationProcessor(simulation=simu, use_dask=False)
lp.process_multiple_snrs(snrs=snrs, run_mode="w")


# with Client(
#     n_workers=p.n_workers,
#     threads_per_worker=1,
#     memory_limit=f"{p.max_ram_per_worker_gb}GB",
# ) as client:
#     # Print dashboard link
#     print("Dask Dashboard:", client.dashboard_link)
#     import matplotlib.pyplot as plt
#     plt.pause(60)

#     # First run to make sure everything is ok 
#     snrs = [0]
#     simu.monte_carlo_iterations = 1
#     simu.check_features = True
#     lp = LocalizationProcessor(simulation=simu)
#     lp.process_multiple_snrs(snrs=snrs, run_mode="w")

#     # Second run to derive perf vs snrs 
#     snrs = np.arange(-10, 17, 2)[::-1]
#     simu.monte_carlo_iterations = 100
#     simu.check_features = False
#     print(f"Processing snrs : {snrs}")
#     lp = LocalizationProcessor(simulation=simu)
#     lp.process_multiple_snrs(snrs=snrs, run_mode="a")