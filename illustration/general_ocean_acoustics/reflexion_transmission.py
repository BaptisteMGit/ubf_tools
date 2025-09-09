#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   reflexion_transmission.py
@Time    :   2025/07/08 11:21:18
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Script to illustrate reflexion and transmission properties of a simple fluid-fluid interface
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

from publication.publication_figure import PubFigure

pfig = PubFigure()

## Define properties


# Frequency
f = 10  # Hz
# Angular frequency
omega = 2 * np.pi * f  # rad/s

# Fluid 1 properties
rho1 = 1000  # kg/m^3
c1r = 1500  # m/s
alpha1 = 0
# Wave number in fluid 1
k1 = omega / c1r  # rad/m
c1 = c1r

# Fluid 2 properties
rho2 = 2.0 * rho1  # kg/m^3
c2r = 1800  # m/s
# Wave number in fluid 2
k2 = omega / c2r  # rad/m
lambda2 = 2 * np.pi / k2  # wavelength in fluid 2 (m)

alpha2 = 0.6  # dB/wavelength
alpha2 = alpha2 / lambda2  # dB/wavelength -> dB/m
# Convert to nepers/m
alpha2 = (
    alpha2 / 8.686
)  # dB/m -> nepers/m (1 neper = 8.686 dB) Cf Eq 1.41 Jensen 2000 (p.35)
c2i = alpha2 / omega * c2r**2  # Cf Eq 1.43 Jensen 2000 (p.35)

c2 = c2r + 1j * c2i  # Complex sound speed in fluid 2 (m/s)


# Air properties
rho_air = 1.204  # kg/m^3 (p0 = 1 013,25 hPa, T = 20 °C)
c_air = 20.05 * np.sqrt(20 + 273.15)  # m/s (T = 20 °C)
# rho_air = 0.1  # kg/m^3 (p0 = 1 013,25 hPa, T = 20 °C)
# c_air = 0.01  # m/s (T = 20 °C)
kair = omega / c_air  # Wave number in air (rad/m)


## Derive reflection and transmission coefficients
def calc_coef(rho1, c1, rho2, c2, theta1, theta2):
    """
    Calculate the reflection coefficient for a fluid-fluid interface.

    Parameters:
    rho1 : float
        Density of fluid 1 (kg/m^3)
    c1 : float
        Sound speed in fluid 1 (m/s)
    rho2 : float
        Density of fluid 2 (kg/m^3)
    c2 : float
        Sound speed in fluid 2 (m/s)

    Returns:
    float
        Reflection coefficient
    """

    z1 = rho1 * c1 / np.sin(theta1)  # Effective acoustic impedance of fluid 1
    z2 = rho2 * c2 / np.sin(theta2)  # Effective acoustic impedance of fluid 2

    r12 = (z2 - z1) / (z1 + z2)  # Reflection coefficient
    t12 = 2 * z2 / (z1 + z2)  # Transmission coefficient

    return r12, t12


def snell_law(c1, c2, theta1):
    """
    Calculate the angle of transmission using Snell's law.

    Parameters:
    c1 : float
        Sound speed in fluid 1 (m/s)
    c2 : float
        Sound speed in fluid 2 (m/s)
    theta1 : float
        Angle of incidence in fluid 1 (radians)

    Returns:
    float
        Angle of transmission in fluid 2 (radians)
    """
    theta2 = np.arccos(c2 / c1 * np.cos(theta1))

    return theta2


# Plot reflection coef as function of angle of incidence between fluid 1 and fluid 2
theta1 = np.linspace(0.01, np.pi / 2, 1000)  # Angle of incidence in fluid 1 (radians)
theta2 = snell_law(c1, c2, theta1)  # Angle of transmission in fluid 2 (radians)
r12, t12 = calc_coef(
    rho1, c1, rho2, c2, theta1, theta2
)  # Reflection and transmission coefficients
bottom_loss = -20 * np.log10(np.abs(r12))

theta2r = snell_law(
    c1, c2r + 0 * 1j, theta1
)  # Angle of transmission in fluid 2 (radians)
r12_lossless, t12_lossless = calc_coef(rho1, c1, rho2, c2r + 0 * 1j, theta1, theta2r)
bottom_loss_lossless = -20 * np.log10(np.abs(r12_lossless))

# # Find kneedle to determine apparent critical angle
# kneedle = KneeLocator(
#     np.degrees(theta1)[30:40],
#     bottom_loss[30:40],
#     S=1.0,
#     curve="convex",
#     direction="increasing",
# )

# print(f"Apparent critical angle for water-sediment interface: {kneedle.knee:.2f}°")

theta_cr = np.degrees(np.arccos(c1 / c2r))

# Reflection coefficient for the air interface
r1air, t1air = calc_coef(
    rho1, c1, rho_air, c_air, theta1, theta2
)  # Reflection and transmission coefficients
bottom_loss_air = -20 * np.log10(np.abs(r1air))

# Plotting
plt.figure(figsize=(10, 6))

# kneedle.plot_knee()
plt.plot(
    np.degrees(theta1),
    bottom_loss,
    label="Eau - sédiment fluide"
    + r" ($\alpha_p = 0.6 \, \textrm{dB} \, \lambda^{-1}$)",
    color="black",
)
plt.plot(
    np.degrees(theta1),
    bottom_loss_lossless,
    label="Eau - sédiment fluide" r" ($\alpha_p = 0 \, \textrm{dB} \, \lambda^{-1}$)",
    color="black",
    ls="--",
)
plt.plot(
    np.degrees(theta1),
    bottom_loss_air,
    label="Eau - air",
    color="blue",
    ls="-",
)
plt.legend(loc="upper left")
plt.xlabel("Angle de rasance [°]")
plt.ylabel("Bottom loss [dB]")

folder_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration\general_ocean_acoustics"
fpath = os.path.join(folder_root, f"bottom_loss_f{f:.1f}Hz.pdf")
plt.savefig(fpath, dpi=300)


# Plot reflection coefficient module and phase using two y-axes
pfig = PubFigure(legend_fontsize=16)

plt.figure(figsize=(10, 6))
plt.title(f"f = {f} Hz")
ax1 = plt.gca()
l0 = ax1.axvline(
    theta_cr,
    label=r"$\theta_c = " + f"{theta_cr:.1f}" + "^\circ$",
    color="blue",
    ls="--",
)

l1 = ax1.plot(
    np.degrees(theta1),
    np.abs(r12),
    label=r"$\alpha_p = 0.6 \, \textrm{dB} \, \lambda^{-1}$",
    color="black",
)
l2 = ax1.plot(
    np.degrees(theta1),
    np.abs(r12_lossless),
    label=r"$\alpha_p = 0 \, \textrm{dB} \, \lambda^{-1}$",
    color="black",
    ls="--",
)


ax1.set_xlabel("Angle de rasance [°]")
ax1.set_ylabel(r"$|\mathcal{R}|$")
ax1.legend(loc="upper right")

ax2 = ax1.twinx()  # Create a second y-axis
# Color the second y-axis red
ax2.spines["right"].set_color("red")  # Set color for second y-axis
ax2.tick_params(axis="y", colors="red")  # Set tick color for second y-axis

ax2.set_ylabel(r"$\arg(\mathcal{R})$ [rad]", color="red")  # Set label for second y-axis
ax2.set_ylim(-np.pi * 1.1, np.pi * 1.1)  # Set limits for second y-axis
ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax2.set_yticklabels(["$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", "$\pi$"])
l3 = ax2.plot(
    np.degrees(theta1),
    np.angle(r12),
    label=r"$\alpha_p = 0.6 \, \textrm{dB} \, \lambda^{-1}$",
    color="red",
    ls="-",
)
l4 = ax2.plot(
    np.degrees(theta1),
    np.angle(r12_lossless),
    label=r"$\alpha_p = 0 \, \textrm{dB} \, \lambda^{-1}$",
    color="red",
    ls="--",
)


# l5 = ax1.plot(
#     np.degrees(theta1),
#     np.abs(r1air),
#     label="Eau - air",
#     color="black",
#     ls="-.",
# )
# l6 = ax2.plot(
#     np.degrees(theta1),
#     np.angle(r1air),
#     label="Eau - air",
#     color="red",
#     ls="-.",
# )
# lns = l5 + l6

# Handle legend from both axis
lns = [l0] + l1 + l2 + l3 + l4  # Combine the lines from both axes
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

fpath = os.path.join(folder_root, "ref_amp_phi.pdf")
plt.savefig(fpath, dpi=300)

plt.show()
