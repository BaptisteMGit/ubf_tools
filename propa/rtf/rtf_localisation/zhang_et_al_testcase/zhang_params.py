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

# Minimum value to replace 0 before converting metrics to dB scale
MIN_VAL_LOG = 1e-5

# Usefull paths
if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    root = os.path.join(
        project_root, r"propa\rtf\rtf_localisation\zhang_et_al_testcase"
    )
    ROOT_TMP = os.path.join(root, "tmp")
    ROOT_DATA = os.path.join(root, "data")
    ROOT_IMG = os.path.join(
        project_root, r"img\illustration\rtf\rtf_localisation\zhang_et_al_2023"
    )
else:  # Linux
    project_root = "/home/program/ubf_tools"
    data_root = "/home/data"
    root = os.path.join(project_root, "rtf_zhang_et_al_testcase")
    data_folder = os.path.join(data_root, "rtf_zhang_et_al_testcase")
    ROOT_TMP = os.path.join(data_folder, "tmp")
    ROOT_DATA = os.path.join(data_folder, "data")
    ROOT_IMG = os.path.join(data_folder, "img")

# Ensure folders exist
for folder in [ROOT_TMP, ROOT_DATA, ROOT_IMG]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Hardware limitations
if os.name == "nt":
    # Windows
    N_WORKERS = 8
    max_ram_gb = 14
    MAX_RAM_PER_WORKER_GB = np.ceil(max_ram_gb / N_WORKERS)
    DASK_SIZES = {
        "t": -1,
        "idx_rcv": -1,
        "x": 10,
        "y": 10,
    }
else:
    # Linux
    N_WORKERS = 80
    max_ram_gb = 90
    MAX_RAM_PER_WORKER_GB = np.ceil(max_ram_gb / N_WORKERS)
    DASK_SIZES = {
        "t": -1,
        "idx_rcv": -1,
        "x": 10,
        "y": 10,
    }
