#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ship_ssl_randi3.py
@Time    :   2024/08/23 14:52:17
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Plot Spectral Source Level (SSL) of a ship based on the Randi3 model (Breeding et al. 1996) 
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# Parameters
# ======================================================================================================================

# Ship parameters
L = 100  # Length of the ship (m)
B = 20  # Beam of the ship (m)
T = 10  # Draft of the ship (m)
V = 10  # Speed of the ship (m/s)
D = 10  # Depth of the water (m)
rho = 1000  # Density of the water (kg/m^3)
c = 1500  # Speed of sound in water (m/s)
f = np.linspace(10, 10000, 1000)  # Frequency range (Hz)

# ======================================================================================================================
# Functions
# ======================================================================================================================


def randi3(f, l, v):
    """
    Compute the SSL of a ship based on the Randi3 model (Breeding et al. 1996)

    Parameters
    ----------
    f : float or array
        Frequency range (Hz)
    l : float
        Length of the ship (m)
    v : float
        Speed of the ship (m/s)
    """

    # Convert l to feet
    l_ft = l * 3.28084
    # Convert v to knots
    v_kt = v * 1.94384

    # Convert f to array
    f = np.array(f)
    Ls0 = np.empty_like(f)
    df = np.empty_like(f)

    # Compute Ls0
    # f < 500 Hz - > -10 * np.log10(10**(-1.06 * log(f - 14.34)) + 10**(3.32 * log(f - 21.425)))
    if1_ls0 = f < 500
    a = -1.06 * np.log10(f[if1_ls0]) - 14.34
    b = 3.32 * np.log10(f[if1_ls0]) - 21.425
    Ls0[if1_ls0] = -10 * np.log10(10**a + 10**b)

    # f > 500 Hz -> Ls0 = 173.2 -18.0 log(f)
    if2_ls0 = f >= 500
    Ls0[if2_ls0] = 173.2 - 18.0 * np.log10(f[if2_ls0])

    # Compute df
    if1_df = f <= 28.4
    df[if1_df] = 8.1
    if2_df = (f > 28.4) & (f <= 191.6)  # What about f > 191.6 Hz ?
    df[if2_df] = 22.3 - 9.77 * np.log10(f[if2_df])

    # Compute dl
    dl = l_ft**1.15 / 3643.0

    # Compute SSL
    SSL = Ls0 + 60 * np.log10(v_kt / 12) + 20 * np.log10(l_ft / 300) + df * dl + 3.0

    return SSL


# ======================================================================================================================
# Main
# =================================================================================================================
# f = np.linspace(1, 150, 300)
f = np.arange(0, 150, 0.5)

# Params of f M/V OVERSEAS HARRIETTE from Arveson, P. T., & Vendittis, D. J. (2000). Radiated noise characteristics of a modern cargo ship
l = 172.9
v_kt = 16
v_ms = v_kt / 1.94384

SSL = randi3(f, l, v_ms) - 10

sl_harmonics = np.array(
    [
        174,
        174,
        175,
        179,
        185,
        176,
        175,
        175,
        172,
        163,
        172,
        173,
        170,
        # 163,
    ]
)

rpm = 140
FR = rpm / 10  # Firing rate of the engine (Hz)
BR = 2 / 3 * FR  # Blade rate of the propeller (Hz)
f_harmonics = np.array(
    [
        BR,
        FR,
        2 * BR,
        3 * BR,
        4 * BR,
        3 * FR,
        5 * BR,
        6 * BR,
        7 * BR,
        5 * FR,
        8 * BR,
        9 * BR,
        10 * BR,
    ]
)
f_harmonics = np.round(2 * f_harmonics, 0) / 2

print(f"Harmonics: {f_harmonics}")
print(f"SSL of harmonics: {sl_harmonics}")

# Add harmonics to the SSL
for i_h, f_h in enumerate(f_harmonics):
    SSL[f == f_h] = sl_harmonics[i_h]

# f_harm
plt.figure()
plt.plot(f, SSL)
plt.xlabel("Frequency (Hz)")
plt.ylabel("SSL (dB)")
plt.title("SSL of a ship based on the Randi3 model")
plt.ylim([140, 190])
plt.xlim([0, 150])
plt.grid()
plt.show()

# Save data to .dat file
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssl_randi3"
fpath = root + r"\ssl_randi3_overseas_harriette.dat"
np.savetxt(fpath, np.vstack((f, SSL)).T, delimiter=",")
