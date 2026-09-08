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
import os
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

### Sediment properties ###
# Sand properties from Jensen et al. (2000) p.39
sand_properties = {
    "rho": 1.9 * rho_w * 1e-3,  # Sand density (g/cm3)
    "c_p": 1650,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # 2.5
}  # Shear wave attenuation (dB/wavelength)

"""
Regarding sediment properties, the classification is based on the Folk 7-class system from the European Marine Observation and Data Network (EMODnet) :
https://www.emodnet-geology.eu/map-viewer/?p=seabed_substrate

The associated values were derived from a table provided in the appendix of the attached document (p.57). This appendix is from Quiet-Oceans, the company where I interned a few months before the project. According to the document, the values are sourced from:
[19] EMODnet, 2012: Seabed substrate data (version 28.6.2012) made available by the EMODnet (European Marine Observation and Data Network) Geology project, funded by the European Commission Directorate General for Maritime Affairs and Fisheries. EMODnet Geology.
"""
boulders_bedrock_properties = {
    "rho": 2.5,  # Sand density (g/cm3)
    "c_p": 3820,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.75,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}

coarse_sediment_properties = {
    "rho": 2.37,  # Sand density (g/cm3)
    "c_p": 2122,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.88,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}

mixed_sediment_properties = {
    "rho": 2.03,  # Sand density (g/cm3)
    "c_p": 1855,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.89,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}
muddy_sand_sand_properties = {
    "rho": 1.53,  # Sand density (g/cm3)
    "c_p": 1708,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.91,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}

mud_sand_sandy_mud_properties = {
    "rho": 1.16,  # Sand density (g/cm3)
    "c_p": 1517,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.37,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}

# All from EMODnet
sediments_EMODnet = {
    "sand_properties": sand_properties,
    "boulders_bedrock_properties": boulders_bedrock_properties,
    "coarse_sediment_properties": coarse_sediment_properties,
    "mixed_sediment_properties": mixed_sediment_properties,
    "muddy_sand_sand_properties": muddy_sand_sand_properties,
    "mud_sand_sandy_mud_properties": mud_sand_sandy_mud_properties,
}

# Document Thierry Garlan, 2010
# Paramètres géoacoustiques extraits du tableau "Types de fond"
#
# rho  : densité (g/cm3)
# c_p  : célérité des ondes P (m/s)
# a_p  : atténuation des ondes P (dB/longueur d'onde)
# c_s  : célérité des ondes S (m/s)
# a_s  : atténuation des ondes S (dB/longueur d'onde)
#
# NOTE : les densités du document sont exprimées en kg/m3 dans
#        l'en-tête mais les valeurs correspondent à des g/cm3.
# NOTE : la dernière colonne semble correspondre à alpha_s.


basalte_TG = {
    "rho": 2.7,
    "c_p": 5250,
    "a_p": 0.1,
    "c_s": 2500,
    "a_s": 0.2,
}

calcaire_TG = {
    "rho": 2.4,
    "c_p": 3000,
    "a_p": 0.1,
    "c_s": 1500,
    "a_s": 0.2,
}

craie_TG = {
    "rho": 2.2,
    "c_p": 2400,
    "a_p": 0.2,
    "c_s": 1000,
    "a_s": 0.2,
}

cailloutis_TG = {
    "rho": 2.2,
    "c_p": 2500,
    "a_p": 0.2,
    "c_s": 1500,
    "a_s": 0.2,
}

graviers_TG = {
    "rho": 2.1,
    "c_p": 2000,
    "a_p": 0.4,
    "c_s": 1000,
    "a_s": 0.5,
}

sables_TG = {
    "rho": 2.0,
    "c_p": 1800,
    "a_p": 0.6,
    "c_s": 180,
    "a_s": 1.5,
}

sables_fins_TG = {
    "rho": 1.9,
    "c_p": 1700,
    "a_p": 0.7,
    "c_s": 110,
    "a_s": 2.5,
}

vase_TG = {
    "rho": 1.8,
    "c_p": 1600,
    "a_p": 0.8,
    "c_s": 95,
    "a_s": 2.0,
}

silts_argileux_TG = {
    "rho": 1.7,
    "c_p": 1550,
    "a_p": 1.0,
    "c_s": 80,
    "a_s": 1.5,
}

argiles_TG = {
    "rho": 1.5,
    "c_p": 1500,
    "a_p": 0.2,
    "c_s": 80,
    "a_s": 1.0,
}

# Tous les sédiments d'après Thierry Garlan 2010
sediments_TG = {
    "basalte_TG": basalte_TG,
    "calcaire_TG": calcaire_TG,
    "craie_TG": craie_TG,
    "cailloutis_TG": cailloutis_TG,
    "graviers_TG": graviers_TG,
    "sables_TG": sables_TG,
    "sables_fins_TG": sables_fins_TG,
    "vase_TG": vase_TG,
    "silts_argileux_TG": silts_argileux_TG,
    "argiles_TG": argiles_TG,
}

# ======================================================================================================================
# Set paths depending on the os
# ======================================================================================================================

# Usefull paths
if os.name == "nt":  # Windows
    project_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"

else:  # Linux
    project_root = "/home/program/ubf_tools"
