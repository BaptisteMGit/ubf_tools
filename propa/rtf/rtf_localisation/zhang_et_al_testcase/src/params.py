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

# ======================================================================================================================
# Global params
# ======================================================================================================================
# Simulation params
fmin = 100
fmax = 500
fs = 1200
duration = 10

antenna_type = "zhang"
library_stype = "lfm"
event_stype = "wn"
rtf_method = "cs_eigve"
gcc_method = "scot"


# Use tex with matplotlib
use_tex = True

# Minimum value to replace 0 before converting metrics to dB scale
min_val_log = 1e-5

# Usefull paths
if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    root = os.path.join(
        project_root, r"propa\rtf\rtf_localisation\zhang_et_al_testcase"
    )
    root_tmp = os.path.join(root, "tmp")
    root_data = os.path.join(root, "data")
    root_img = os.path.join(
        project_root, r"img\illustration\rtf\rtf_localisation\zhang_et_al_2023"
    )
    root_img_publi = os.path.join(root_img, "publication_rtf")

else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"
    root = os.path.join(project_root, "rtf_zhang_et_al_testcase")
    data_folder = os.path.join(data_root, "rtf_zhang_et_al_testcase")
    root_tmp = os.path.join(data_folder, "tmp")
    root_data = os.path.join(data_folder, "data")
    root_img = os.path.join(data_folder, "img")

# Ensure folders exist
for folder in [root_tmp, root_data, root_img, root_img_publi]:
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
    n_workers = 20
    max_ram_gb = 80
    max_ram_per_worker_gb = np.ceil(max_ram_gb / n_workers)
    block_sizes = {
        "t": -1,
        "idx_rcv": -1,
        "x": 10,
        "y": 5,
    }
