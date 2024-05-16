import os
import time
import numpy as np
import pandas as pd

from propa.kraken_toolbox.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenEnv,
    KrakenFlp,
    Bathymetry,
)
from propa.kraken_toolbox.run_kraken import runkraken_broadband_range_dependent
from propa.kraken_toolbox.utils import default_nb_rcv_z

# from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE


frequencies = range(50, 151)

# Kraken
t0 = time.time()
filename = r"calib_ray_mode_kraken"
working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\compare_ray_modes\env_kraken"
os.chdir(working_dir)

os.system(f"kraken {filename}")
os.system(f"field {filename}")

t_kraken = time.time() - t0
print(f"Kraken time: {t_kraken:.2f} s")

# # Bellhop
t0 = time.time()
filename = r"calib_ray_mode_bellhop"
os.system(f"bellhop {filename}")  # Only one run required to derive the impulse response
t_bellhop = time.time() - t0
print(f"Bellhop time: {t_bellhop:.2f} s")


# Kraken rd and broadband
t0 = time.time()
filename = r"calib_ray_mode_kraken_rd_broadband"
filename_template = r"calib_ray_mode_kraken_rd_broadband_template.env"
working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\compare_ray_modes\env_kraken_rd"
os.chdir(working_dir)

# Top halfspace
top_hs = KrakenTopHalfspace()
# SSP
ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
z_ssp = ssp_data[0].values
cp_ssp = ssp_data[1].values

medium = KrakenMedium(
    ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=5000
)

# Attenuation
att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

# Field
n_rcv_z = default_nb_rcv_z(max(frequencies), 1000, n_per_l=15)
field = KrakenField(
    src_depth=20,
    phase_speed_limits=[0, 20000],
    n_rcv_z=n_rcv_z,
    rcv_z_max=1000,
)

# Range dependent bathymetry
# Define bathymetry
r = np.linspace(0, 100, 1001)
h = np.ones(len(r)) * 1000

pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
    os.path.join(working_dir, "bathy.csv"), index=False, header=False
)

bathy = Bathymetry(
    data_file=os.path.join(working_dir, "bathy.csv"),
    interpolation_method="linear",
    units="km",
)


bott_hs_properties = {
    "rho": 1.5,
    "c_p": 1600.0,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s)
    "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}
bott_hs_properties["z"] = z_ssp.max()
bott_hs = KrakenBottomHalfspace(
    halfspace_properties=bott_hs_properties,
)

env_filename = "calib_ray_mode_kraken_rd_broadband"
env = KrakenEnv(
    title="STURM test case",
    env_root=working_dir,
    env_filename=env_filename,
    freq=frequencies,
    kraken_top_hs=top_hs,
    kraken_medium=medium,
    kraken_attenuation=att,
    kraken_bottom_hs=bott_hs,
    kraken_field=field,
    kraken_bathy=bathy,
    rModes=r,
)

env.write_env()
flp = KrakenFlp(
    env=env,
    src_depth=20,
    mode_theory="coupled",
    rcv_r_max=100,
    rcv_z_max=1000,
    nb_modes=50,
)
flp.write_flp()

runkraken_broadband_range_dependent(env, flp, frequencies)

t_kraken = time.time() - t0
print(f"Kraken rd time: {t_kraken:.2f} s")
