#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   params.py
@Time    :   2026/01/22 14:04:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Global params for Fiberscope Groix analysis
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

# OBS voie hydro
obs_hydro_sensitivity = -160  # in dB re 1V/uPa
obs_hydro_gain = 0  # 20*np.log10(1/2.56) #signal en volt full scale (cf ELOBSBin2Wav)

# Convention Gen_Axes_D_V4 (Cf ELOBSBin2Wav.py)
channels_order = {
    "Z": 0,
    "X": 1,
    "Y": 2,
    "H": 3,
}
hydro_channel = "H"

pre_reception_time = 10
post_reception_time = 15.0

tau_ir = 0.1
h_index_ref = 2
alpha_overlap = 0.5

bandfilter_order = 4
bandfilter_lowcut = 200
bandfilter_highcut = 900
# ======================================================================================================================
# Paths
# ======================================================================================================================

if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

    root_groix_data = os.path.join(project_root, "data", "fiberscope_groix_oct_2025")
    sbe39_obs_folder = os.path.join(root_groix_data, "SBE39_OBS")
    root_groix_wav = os.path.join(root_groix_data, "wav")
    root_folder = os.path.join(project_root, "real_data_analysis", "fiberscope_groix")
    root_data = os.path.join(root_folder, "data")
    root_img = os.path.join(root_folder, "img")


else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"

    # TODO : update this if deployement on TIM

    # root = os.path.join(project_root, "rtf_zhang_et_al_testcase")
    # data_folder = os.path.join(data_root, "rtf_zhang_et_al_testcase")
    # root_tmp = os.path.join(data_folder, "tmp")
    # root_data = os.path.join(data_folder, "data")
    # root_img = os.path.join(data_folder, "img")
    # root_img_publi = os.path.join(root_img, "publication_rtf")
