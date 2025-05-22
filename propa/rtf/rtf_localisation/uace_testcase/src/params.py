#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_params.py
@Time    :   2025/03/16 18:50:51
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Global params for Zhang testcase
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import source.global_constants as g
from propa.rtf.rtf_localisation.uace_testcase.src.ship_signal import ShipSignal
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna

# ======================================================================================================================
# Global params
# ======================================================================================================================

# Use tex with matplotlib
use_tex = True

# Minimum value to replace 0 before converting metrics to dB scale
min_val_log = 1e-5


# Usefull paths
if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    root = os.path.join(project_root, r"propa\rtf\rtf_localisation\uace_testcase")
    root_tmp = os.path.join(root, "tmp_kraken")
    root_data = os.path.join(root, "data")
    root_img = os.path.join(
        project_root, r"img\illustration\rtf\rtf_localisation\uace_testcase"
    )
    root_img_publi = os.path.join(root_img, "publication")

else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"
    root = os.path.join(project_root, "uace_testcase")
    data_folder = os.path.join(data_root, "uace_testcase")
    root_tmp = os.path.join(data_folder, "tmp_kraken")
    root_data = os.path.join(data_folder, "data")
    root_img = os.path.join(data_folder, "img")
    root_img_publi = os.path.join(data_folder, "img", "publication")

# Ensure folders exist
for folder in [root_tmp, root_data, root_img]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Hardware limitations
if os.name == "nt":
    # Windows
    n_workers = 6
    max_ram_gb = 8
    max_ram_per_worker_gb = np.ceil(max_ram_gb / n_workers)
    block_sizes = {
        "t": -1,
        "idx_rcv": -1,
        "x": 2,
        "y": 2,
    }
else:
    # Linux
    n_workers = 35
    max_ram_gb = 80
    max_ram_per_worker_gb = np.ceil(max_ram_gb / n_workers)
    block_sizes = {
        "t": -1,
        "idx_rcv": -1,
        "x": 10,
        "y": 5,
    }

# ======================================================================================================================
# Simulation params
# ======================================================================================================================

# name = "uace_testcase"
name = "testcase_zhang2023"

# Default antenna
antenna = SparseAntenna(
    name="default_antenna", n_elements=6, random_radius=5e3, rng_seed=42
)

# Waveguide properties
bott_hs_properties = {
    "rho": 1.5 * g.rho_w * 1e-3,  # Density (g/cm^3)
    "c_p": 1550,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.2,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    "z": None,
}


# Path to real bathymetry file
bathy_fpath = r"data\bathy\mmdpm\PVA_RR48\mmdpm_test_PVA_RR48_360.csv"
# Path to real ssp file
ssp_fpath = ""

# Grid properties
dx = 20
dy = 20
search_area_length = 1e3

# Env properties
cmin = 1488  # Min speed in SwellEx96 mean profile

# Localization params
monte_carlo_iterations = 100
frequency_drawing_method = "equally_spaced"  # "equally_spaced" / "random"
number_of_drawn_frequencies = 100

# Plot results from loc params
plot_args = {
    "plot_array": False,
    "plot_single_cpl_surf": False,
    "plot_fullarray_surf": True,
    "plot_cpl_surf_comparison": True,
    "plot_fullarray_surf_comparison": True,
    "plot_surf_dist_comparison": False,
    "plot_mainlobe_contour": False,
    "plot_msr_estimation": False,
}

# Signal general properties
fmin = 5
fmax = 50
fs = 200
# fs = 100
# duration = 10
duration = 20

# nfft = 1024

# Ship signal params

# Plot params
nperseg = 2**8
noverlap = 2**7
tmin, tmax = 1, 2
root_img_ship_sigs = os.path.join(root_img, "ship_signals")

# Library ship signal
# # Parametres Samuel
# f0_l = 10
# std_fi_l = 0.15 * f0_l
# tau_corr_fi_l = 1 / np.sqrt(2 * np.pi) * 1 / f0_l

# My params
f0_l = 4.889
std_fi_l = 0.058 * f0_l
tau_corr_fi_l = 0.067 * 1 / f0_l

library_ship = ShipSignal(
    name="library_ship",
    f0=f0_l,
    fs=fs,
    duration=duration,
    std_fi=std_fi_l,
    tau_corr_fi=tau_corr_fi_l,
    root_img=root_img_ship_sigs,
)

# library_ship.plot_signal(tmin=tmin, tmax=tmax)
# library_ship.plot_spectrum(fmin=0, fmax=fmax)
# library_ship.plot_psd(
#     window="hann", nperseg=nperseg, noverlap=noverlap, fmin=0, fmax=fmax
# )
# library_ship.plot_stft(
#     window="hann", nperseg=nperseg, noverlap=noverlap, fmin=0, fmax=fmax
# )
# import matplotlib.pyplot as plt
# plt.show()

unique_library_ship = library_ship


# Event ship signal -> signal 19 de la base de signaux
f0_e = 4.629
std_fi_e = 0.072 * f0_e
tau_corr_fi_e = 0.304 * 1 / f0_e
event_ship = ShipSignal(
    name="event_ship",
    f0=f0_e,
    fs=fs,
    duration=duration,
    std_fi=std_fi_e,
    tau_corr_fi=tau_corr_fi_e,
    root_img=root_img_ship_sigs,
)

# event_ship.plot_signal(tmin=tmin, tmax=tmax)
# event_ship.plot_spectrum(fmin=0, fmax=fmax)
# event_ship.plot_psd(
#     window="hann", nperseg=nperseg, noverlap=noverlap, fmin=0, fmax=fmax
# )
# event_ship.plot_stft(
#     window="hann", nperseg=nperseg, noverlap=noverlap, fmin=0, fmax=fmax
# )

# Position of the ship to localize
# event_ship_x = 7500
# event_ship_y = 8000
event_ship_x = 25000
event_ship_y = 12000
event_ship_z = 5

# Window parameters to derive rtfs
nperseg = 2**10
alpha_overlap = 0.5

### Mutliple ship library ###
# Default set
# f0_min = 4
# f0_max = 5
# std_fi_min = 1e-3
# std_fi_max = 1e-1
# tau_corr_fi_min = 1e-3
# tau_corr_fi_max = 0.5

# Small variations allowed
f0_min = 4
f0_max = 12
std_fi_min = 1e-2  # 1 %
std_fi_max = 20 * 1e-2  # 20 %
# std_fi_max = 1e-2
tau_corr_fi_min = 5 * 1e-3
tau_corr_fi_max = 0.5

# f0_min = 4
# f0_max = 5.5
# std_fi_min = 1e-3
# std_fi_max = 1 * 1e-1
# tau_corr_fi_min = 1e-3
# tau_corr_fi_max = 3 * 1e-1

# library_ship = [library_ship]
rng_seed = 65
rng = np.random.default_rng(seed=rng_seed)
nl_ship = 5
library_ship = []
for iship in range(nl_ship):
    f0_l = rng.uniform(low=f0_min, high=f0_max + 5)
    std_fi_l = rng.uniform(low=std_fi_min, high=std_fi_max) * f0_l
    tau_corr_fi_l = rng.uniform(low=tau_corr_fi_min, high=tau_corr_fi_max) * 1 / f0_l
    library_ship_i = ShipSignal(
        name=f"library_ship_{iship}",
        f0=f0_l,
        fs=fs,
        duration=duration,
        std_fi=std_fi_l,
        tau_corr_fi=tau_corr_fi_l,
        root_img=root_img_ship_sigs,
    )
    library_ship.append(library_ship_i)

use_weighted_rtf = True
antenna_type = "random"
rtf_method = "cs_eigve"
gcc_method = "scot"
