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

if os.name == "nt":
    PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "localisation", "verlinden")
else:
    PROJECT_ROOT = "/home/program/ubf_tools-main/"
    DATA_ROOT = "/home/data"

ROOT_DATASET_PATH = os.path.join(DATA_ROOT, "localisation_dataset")

ROOT = "/home/program/ubf_tools-main/localisation/verlinden"
# ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation"

VERLINDEN_POPULATED_FOLDER = os.path.join(ROOT, "verlinden_process_populated_library")
VERLINDEN_OUTPUT_FOLDER = os.path.join(ROOT, "verlinden_process_output")
TC_WORKING_DIR = os.path.join(ROOT, "testcase_working_directory")
VERLINDEN_ANALYSIS_FOLDER = os.path.join(ROOT, "verlinden_process_analysis")


BATHY_FILENAME = "GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
