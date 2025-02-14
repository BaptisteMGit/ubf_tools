#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_short_ri_consts.py
@Time    :   2024/11/04 14:02:25
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
from cst import RHO_W, C0

ROOT_FOLDER = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_estimation\short_ri_waveguide"
ROOT_DATA = os.path.join(ROOT_FOLDER, "data")

N_RCV = 5  # Number of receivers
TAU_IR = 5  # Impulse response duration in seconds
