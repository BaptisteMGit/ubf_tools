#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   pekeris_short_ri_waveguide_params.py
@Time    :   2025/09/09 11:49:40
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Common params to use for the Pekeris Short Impulse Response Waveguide Simulations
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd")

# Load usefull functions
import source.global_constants as g
from publication.publication_figure import PubFigure, LargeFigure
from propa.ideal_waveguide import (
    print_arrivals,
)

from propa.kraken_toolbox.src.kraken_testcase import (
    KrakenTestCase,
    DomainProperties,
    SourceProperties,
    ReceiverProperties,
    KrakenProperties,
)
from propa.kraken_toolbox.src.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox.plot_utils import plotmode, plotmode_several_freqs
from propa.kraken_toolbox.utils import default_nb_rcv_z
from signals.AcousticComponent import AcousticSource
from source.signal_generator import SignalGenerator
from source.ssp_profiles import SSPProfile
from misc import mult_along_axis

pfig = PubFigure()


# ======================================================================================================================
# Paramètres du guide d'onde de Pekeris
# ======================================================================================================================

# Waveguide geometry
waveguide_depth = 1000.0  # m
max_range_km = 50  # km

# Water
c_water = g.c0
rho_water = g.rho_w  # kg/m^3

# # Sediment
# c_sediment = 1550.0  # m/s
# rho_sediment = 1500.0  # kg/m^3
# alpha_sediment = 0.2  # dB/lambda

# Sediment (Pekeris pulse)
c_sediment = 1600.0  # m/s
rho_sediment = 1500.0  # kg/m^3
alpha_sediment = 0.2  # dB/lambda

# Impulse response duration (derived in the pekeris_short_ir_waveguide_definition notebook)
tau_th = 1.8

# ======================================================================================================================
# Paramètres de la source
# ======================================================================================================================
src_depth = 5  # m
src_min_freq = 0  # Hz
src_max_freq = 50  # Hz
src_fs = 100  # Hz
src_signal_duration = 5  # s


# ======================================================================================================================
# Paramètres de la simulation Kraken
# ======================================================================================================================

kraken_simu_name = "perekis_short_ir_waveguide"
kraken_title = "Pekeris waveguide with short impulse response"


# ======================================================================================================================
# Chemins d'accès
# ======================================================================================================================

folder_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\pekeris_short_ir_waveguide"
tc_root_dir = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases"
)
img_folder_path = os.path.join(folder_root, "img")
data_folder_path = os.path.join(folder_root, "data")

# Dataset path
tf_demo_fpath = os.path.join(data_folder_path, "tf_demo.nc")
tf_perf_fpath = os.path.join(data_folder_path, "tf_perf.nc")

sig_fpath = os.path.join(data_folder_path, "received_signals.nc")
