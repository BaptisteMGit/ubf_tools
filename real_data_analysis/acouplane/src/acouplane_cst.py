#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   path.py
@Time    :   2026/10/02 15:57:19
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Define useful constants for the acouplane project
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

from publication.publication_figure import PubFigure
from source.global_constants import ROOT_PROGRAM, ROOT_DATA

# ======================================================================================================================
# Useful paths
# ======================================================================================================================

### Directories ###

# Raw data
ROOT_ACOUPLANE_DATA_DIR = os.path.join(ROOT_DATA, "ACOUPLANE")
ACOUPLANE_METADATA_DIR = os.path.join(ROOT_ACOUPLANE_DATA_DIR, "METADATA")
ACOUPLANE_DATA_DIR = os.path.join(ROOT_ACOUPLANE_DATA_DIR, "DATA")
ACOUPLANE_DATA_POSITION_DIR = os.path.join(ACOUPLANE_DATA_DIR, "POSITION")
ACOUPLANE_DATA_POSITION_TRAME_GPS_ATL_DIR = os.path.join(
    ACOUPLANE_DATA_POSITION_DIR, "Trame_GPS_ATL"
)
ACOUPLANE_DATA_POSITION_SHOT_DIR = os.path.join(ACOUPLANE_DATA_POSITION_DIR, "SHOT")
ACOUPLANE_DATA_PRESSURE_DIR = os.path.join(ACOUPLANE_DATA_DIR, "PRESSURE")
ACOUPLANE_DATA_PRESSURE_BIN_DIR = os.path.join(ACOUPLANE_DATA_PRESSURE_DIR, "BIN")
ACOUPLANE_DATA_PRESSURE_WAV_DIR = os.path.join(ACOUPLANE_DATA_PRESSURE_DIR, "WAV")

# Processed data
ROOT_ACOUPLANE_PROGRAM_DIR = os.path.join(
    ROOT_PROGRAM, "real_data_analysis", "acouplane"
)
ROOT_ACOUPLANE_PROCESSED_DATA_DIR = os.path.join(ROOT_ACOUPLANE_PROGRAM_DIR, "data")
ROOT_ACOUPLANE_PROCESSED_IMG_DIR = os.path.join(ROOT_ACOUPLANE_PROGRAM_DIR, "img")


### Files paths ###

# Raw data
OBS_TLQ_POS_FPATH = os.path.join(ACOUPLANE_DATA_POSITION_DIR, "OBS_TLQ_POS.csv")
AIS_FPATH = os.path.join(ACOUPLANE_DATA_POSITION_DIR, "AIS_LPLNE.csv")

# Processed data
ACOUPLANE_POSITIONS_NC_FPATH = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_DATA_DIR, "acouplane_pos.nc"
)
ACOUPLANE_PRESSURE_NC_FPATH = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_DATA_DIR, "acouplane_raw_pressure.nc"
)

# Ensure direcotries exist 
def ensure_dirs_exist():
    dirs = [
        ROOT_ACOUPLANE_PROGRAM_DIR, 
        ROOT_ACOUPLANE_PROCESSED_DATA_DIR,
        ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
    ]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

# ======================================================================================================================
# Constants
# ======================================================================================================================
PubFigure(label_fontsize=24, ticks_fontsize=20)


if __name__ == "__main__":

    print(ROOT_ACOUPLANE_PROGRAM_DIR, ROOT_ACOUPLANE_PROCESSED_DATA_DIR)
    ensure_dirs_exist()