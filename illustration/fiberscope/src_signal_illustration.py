#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   src_signal_illustration.py
@Time    :   2024/11/19 14:04:23
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
import scipy.signal as sp
import matplotlib.pyplot as plt

# Create a typical signal
n_pulse = 3
# f0 = 8 * 1e3
# f1 = 15 * 1e3
# f0 = 10
# f1 = 50
# t1 = 500 * 1e-3
# t_inter_pulse = 1
# ts = 1 / (f1 * 20)

# tmax = t_inter_pulse * n_pulse
# t = np.arange(0, tmax + ts, ts)
# x = sp.chirp(t, f0=f0, f1=f1, t1=t1, method="linear", phi=90)
# x[t > t1] = 0

# t_start_offset = 0.1
# s = np.zeros(t.size)
# for i in range(n_pulse):
#     roll_index = int((i * t_inter_pulse + t_start_offset) / ts)
#     s += np.roll(x, roll_index)


# plt.figure()
# plt.plot(t, s)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("Pulse signal")

# # Save data to .dat file
# root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration\fiberscope"
# fpath = os.path.join(root_data, "chirp_signal_model.dat")
# data = np.vstack((t, s)).T
# np.savetxt(fpath, data, delimiter=" ", header="t s")

f0 = 2 * 1e3
f1 = 20 * 1e3
t1 = 800 * 1e-3
t_inter_pulse = 1
fs = 200 * 1e3
ts = 1 / fs

t = np.arange(0, t1 + ts, ts)
x = sp.chirp(t, f0=f0, f1=f1, t1=t1, method="linear", phi=0)

# Derive autocorrelation
r_xx_p = np.correlate(x, x, mode="full")
# Derive lags
lags = np.arange(-len(x) + 1, len(x)) * ts


def r_xx(tau, T, B, f0):
    """
    Compute the expression for r_xx(τ).

    Parameters:
    - tau : float or ndarray
        Time delay (τ).
    - T : float
        Parameter T.
    - B : float
        Bandwidth (B).
    - f0 : float
        Center frequency (f0).

    Returns:
    - r_xx : float or ndarray
        The value of r_xx(τ).
    """
    # Compute the wedge_T(tau) term
    wedge_T = np.clip(T - np.abs(tau), 0, None) / T

    # Compute the sinc function
    sinc_term = np.sinc(B * tau * wedge_T)

    # Compute the exponential term
    exp_term = np.exp(1j * 2 * np.pi * B * f0 * tau)

    # Combine all terms
    r_xx = np.sqrt(T) * wedge_T * sinc_term * exp_term

    return np.real(r_xx)


r_xx_th = r_xx(lags, T=t1, B=(f1 - f0), f0=f0)

# # Plot
plt.figure()
plt.plot(lags, r_xx_p / np.max(r_xx_p))
plt.plot(lags, r_xx_th / np.max(r_xx_th), "--")
plt.xlabel("Lags [s]")
plt.ylabel("Amplitude")
plt.title("Autocorrelation of chirp signal")

# plt.figure()
# plt.plot(t, x)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("Pulse signal")

# # Save data to .dat file
# root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration\fiberscope"
# fpath = os.path.join(root_data, "chirp_signal_model.dat")
# data = np.vstack((t, s)).T
# np.savetxt(fpath, data, delimiter=" ", header="t s")


plt.show()
