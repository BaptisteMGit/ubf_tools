#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   params.py
@Time    :   2024/05/17 10:26:01
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

# Usefull paths
# Usefull paths

if os.name == "nt":
    PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "loc")
else:
    PROJECT_ROOT = "/home/program/ubf_tools"
    DATA_ROOT = "/home/data"

ROOT_DATASET = os.path.join(DATA_ROOT, "localisation_dataset")
ROOT_ANALYSIS = os.path.join(DATA_ROOT, "localisation_analysis")
ROOT_PROCESS = os.path.join(DATA_ROOT, "localisation_process")

VERLINDEN_ANALYSIS_FOLDER = os.path.join(DATA_ROOT, "localisation_analysis")
TC_WORKING_DIR = os.path.join(DATA_ROOT, "testcase_working_directory")

VERLINDEN_POPULATED_FOLDER = os.path.join(
    DATA_ROOT, "verlinden_process_populated_library"
)
BATHY_FILENAME = "GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
