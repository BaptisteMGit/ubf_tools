#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   global_constants.py
@Time    :   2025/03/31 13:59:01
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Computer related constants
# ======================================================================================================================
eps = np.finfo(float).eps
diagonal_loading = (
    1e-8  # amount of diagonal loading when adding identity matrix to covariance matrix
)

# ======================================================================================================================
# Physical constants
# ======================================================================================================================
p0 = 1e-6  # reference pressure in Pa
c0 = 1500  # sound celerity in water (m/s)
rho_w = 1000  # water density (kg/m3)
sand_properties = {
    "rho": 1.9 * rho_w * 1e-3,  # Sand density (g/cm3)
    "c_p": 1650,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
    "a_s": 2.5,
}  # Shear wave attenuation (dB/wavelength) # Sand properties from Jensen et al. (2000) p.39
