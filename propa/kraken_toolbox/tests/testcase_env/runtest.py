import matplotlib.pyplot as plt
from localisation.verlinden.testcases.testcase_envs import TestCase1_0, TestCase3_1
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.plot_utils import plotshd, plotmode
import os

root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\testcase_env"
# Testcase 1

tc_varin = {
    "freq": [20],
    "max_range_m": 15 * 1e3,
    "azimuth": 0,
    "rcv_lon": 65.943,
    "rcv_lat": -27.5792,
    # "mode_theory": "adiabatic",
}
testcase = TestCase1_0(testcase_varin=tc_varin)

pressure_field, field_pos = runkraken(
    env=testcase.env,
    flp=testcase.flp,
    frequencies=testcase.env.freq,
    parallel=True,
    verbose=False,
)

plotshd(testcase.env.shd_fpath)
plt.ylim([120, 0])


# Testcase 1

tc_varin = {
    "freq": [20],
    "max_range_m": 15 * 1e3,
    "azimuth": 0,
    "rcv_lon": 65.943,
    "rcv_lat": -27.5792,
    "mode_theory": "adiabatic",
    "src_depth": 100,
}
testcase = TestCase3_1(testcase_varin=tc_varin)

pressure_field, field_pos = runkraken(
    env=testcase.env,
    flp=testcase.flp,
    frequencies=testcase.env.freq,
    parallel=True,
    verbose=False,
)

plotshd(testcase.env.shd_fpath, bathy=testcase.bathy)
plt.savefig(os.path.join(root, "tc31_adiabatic"))


tc_varin = {
    "freq": [20],
    "max_range_m": 15 * 1e3,
    "azimuth": 0,
    "rcv_lon": 65.943,
    "rcv_lat": -27.5792,
    "mode_theory": "coupled",
    "src_depth": 100,
}
testcase = TestCase3_1(testcase_varin=tc_varin)

pressure_field, field_pos = runkraken(
    env=testcase.env,
    flp=testcase.flp,
    frequencies=testcase.env.freq,
    parallel=True,
    verbose=False,
)

plotshd(testcase.env.shd_fpath, bathy=testcase.bathy)
plt.savefig(os.path.join(root, "tc31_coupled"))
# plt.ylim([120, 0])
# plt.show()
