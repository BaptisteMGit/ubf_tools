#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   params.py
@Time    :   2024/05/17 10:22:59
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os


# Hardware limitations
N_WORKERS = 80
MAX_RAM_GB = 125

# Usefull paths

if os.name == "nt":
    PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "localisation", "verlinden")
else:
    PROJECT_ROOT = "/home/program/ubf_tools-main/"
    DATA_ROOT = "/home/data"

ROOT_DATASET_PATH = os.path.join(DATA_ROOT, "localisation_dataset")

BATHY_FILENAME = "GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
