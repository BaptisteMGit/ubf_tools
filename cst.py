#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   cst.py
@Time    :   2024/07/08 09:12:59
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Define useful constants.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
from colorama import Fore

""" Usefull constants """
# Physical properties
C0 = 1500  # Sound celerity in water (m/s)
RHO_W = 1000  # Water density (kg/m3)
SAND_PROPERTIES = {
    "rho": 1.9 * RHO_W,
    "c_p": 1650,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
    "a_s": 2.5,  # Shear wave attenuation (dB/wavelength)
}  # Sand properties from Jensen et al. (2000) p.39


# Verlinden parameters
MAX_LOC_DISTANCE = 50 * 1e3  # Maximum localisation distance (m)
AVERAGE_LOC_DISTANCE = 10 * 1e3  # Average localisation distance (m)
LIBRARY_COLOR = "red"
EVENT_COLOR = "black"

# TDQM bar format
BAR_FORMAT = "%s{l_bar}%s{bar}%s{r_bar}%s" % (
    Fore.YELLOW,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.RESET,
)

# Graph constants
LABEL_FONTSIZE = 20
TICKS_FONTSIZE = 20
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 18
SUPLABEL_FONTSIZE = 22

# Parrallel processing
if os.name == "nt":
    N_CORES = 10  # Windows PC
else:
    N_CORES = 100  # Linux plateforme TIM
