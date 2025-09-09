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

# Load usefull functions
import source.global_constants as g

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

kraken_simu_name = "pekeris_short_ir_waveguide"
kraken_title = "Pekeris waveguide with short impulse response"

if os.name == "nt":  # Windows
    n_workers = 4
else:  # Linux
    n_workers = 30

# ======================================================================================================================
# Paramètres antenne linéaire horizontale HLA
# ======================================================================================================================
idx_rcv_ref = 0
r_rcv_hla = 30 * 1e3      # Range from source to receiver 1
z_rcv_hla = 5
hla_delta_r_rcv = 700  # Receiver range step in meters
hla_nrcv = 5  # Number of receivers in the HLA

# Subsample transfer function dataset to reduce memory usage
# rcv_rmin = ds_tf.r.min().values
rcv_rmin = r_rcv_hla - (hla_nrcv + 1) * hla_delta_r_rcv
rcv_rmax = ds_tf.r.max().values
rcv_dr = 100
rcv_range = np.arange(rcv_rmin, rcv_rmax + rcv_dr, rcv_dr)

# rcv_zmin = ds_tf.z.min().values
rcv_zmin = 5
rcv_zmax = np.max([50, z_hat, z_plot])
rcv_dz = 10
rcv_depth = np.arange(rcv_zmin, rcv_zmax + rcv_dz, rcv_dz)

# ======================================================================================================================
# Chemins d'accès
# ======================================================================================================================

folder_root = os.path.join(
    g.project_root, "propa", "rtf", "rtf_estimation", kraken_simu_name
)
tc_root_dir = os.path.join(g.project_root, "propa", "kraken_toolbox", "testcases")
img_folder_path = os.path.join(folder_root, "img")
data_folder_path = os.path.join(folder_root, "data")

# Ensure all folders exist 
for fpath in [tc_root_dir, img_folder_path, data_folder_path]:
    if not os.path.exists(fpath):
        os.makedirs(fpath)

# Dataset path
tf_demo_fpath = os.path.join(data_folder_path, "tf_demo.nc")
tf_perf_fpath = os.path.join(data_folder_path, "tf_perf.nc")

sig_fpath = os.path.join(data_folder_path, "received_signals.nc")
