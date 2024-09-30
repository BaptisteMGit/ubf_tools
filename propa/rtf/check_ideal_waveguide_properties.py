#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   check_ideal_waveguide_properties.py
@Time    :   2024/09/27 15:25:23
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from propa.rtf.ideal_waveguide import *
from signals.signals import ricker_pulse


def nb_propagating_modes_freq(fmin, fmax, D):
    # Number of propagating modes
    f = np.linspace(fmin, fmax, 1000)
    n = nb_propagating_modes(f, c0, D)
    fc = cutoff_frequency(c0, D)

    plt.figure()
    plt.plot(f, n)
    plt.axvline(fc, color="red", linestyle="--", label=f"Cutoff frequency: {fc:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Number of propagating modes")
    plt.title("Number of propagating modes in an ideal waveguide")
    plt.legend()
    plt.grid()
    plt.show()


def plot_modes(f, D):

    # Modes
    n = nb_propagating_modes(f, c0, D)
    z = np.linspace(0, D, 1000)

    m = np.arange(1, n + 1)
    psi_m = [psi_normalised(l, z, D, rho_0) for l in m]
    # psi_m = [psi(l, z) for l in m]

    nb_mode_to_plot = 4
    fig, ax = plt.subplots(1, nb_mode_to_plot, figsize=(16, 10), sharey=True)
    ax[0].invert_yaxis()

    for i in range(nb_mode_to_plot):
        ax[i].plot(psi_m[i], z)
        ax[i].set_title(f"Mode {i + 1}")
        ax[i].set_xlim([-1, 1])
        ax[i].axvline(0, color="k", linestyle="--")

    plt.show()


def plot_tf(fmin, fmax, z_src, z, r, D):
    # Transfer function

    f = np.arange(fmin, fmax, 0.01)
    f, h_f = h(f, z_src, z, r, D)

    plt.figure()
    plt.plot(f, np.abs(h_f))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|H(f)|")
    plt.title("Fonction de transfert")
    plt.grid()
    plt.show()


def get_irs(D):
    # Image source
    Ts = 0.001
    fs = 1 / Ts
    # print(f"Ts = {Ts:.2e}s")
    t = np.arange(0, 1000, Ts)
    t_img, ir_img = image_source_ri(z_src, z, r, D, n=100, t=t)

    # Modes
    f = fft.rfftfreq(len(t), Ts) + 5
    f_mode, h_f_mode = h(f, z_src, z, r, D)
    ir_mode = fft.irfft(h_f_mode).real
    t_mode = np.arange(0, len(ir_mode)) / fs

    return t_img, ir_img, t_mode, ir_mode, f_mode, h_f_mode


def plot_ir(D):
    # Impulse response
    t_img, ir_img, t_mode, ir_mode, f_mode, h_f_mode = get_irs(D)

    plt.figure()
    plt.plot(t_img, ir_img, label="Image source")
    plt.plot(t_mode, ir_mode, label="Modes")
    plt.xlabel("Time (s)")
    plt.ylabel("h(t)")
    plt.title("Impulse response")
    plt.legend()
    plt.grid()
    plt.show()


def compare_tf(D):
    t_img, ir_img, t_mode, ir_mode, f_mode, h_f_mode = get_irs(D)
    fs = 1 / (t_img[1] - t_img[0])
    # Transfert function from ir_img
    h_img = fft.rfft(ir_img)
    f_img = np.fft.rfftfreq(len(ir_img), 1 / fs)

    plt.figure()
    plt.plot(f_mode, np.abs(h_f_mode), label="Modes")
    plt.plot(f_img, np.abs(h_img), label="Image source")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|H(f)|")
    plt.title("Transfer function")
    plt.grid()
    plt.legend()
    plt.show()

    # Create ricker pulse
    fc = 50
    T = 0.1
    s_pulse, t_pulse = ricker_pulse(fc, fs, T, t0=0, center=True)

    plt.figure()
    plt.plot(t_pulse, s_pulse)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Ricker pulse")
    plt.grid()

    # Convolution
    s_conv_img = np.convolve(s_pulse, ir_img, mode="full")
    s_conv_modes = np.convolve(s_pulse, ir_mode, mode="full")
    t_conv = np.arange(0, len(s_conv_img)) / fs
    plt.figure()
    plt.plot(t_conv, s_conv_img)
    plt.plot(t_conv, s_conv_modes)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Convolution")
    plt.grid()
    plt.show()


# def plot_rtf(D):

#     # # RTF
#     D = 100
#     z_src = 25
#     z_rcv_ref = 90
#     z_rcv = z_rcv_ref

#     r_rcv_ref = 1e5
#     rl = r_rcv_ref + 10

#     f = np.arange(fmin, fmax, 0.1)
#     t0 = time()
#     f_h, h_l = h(f, z_src, z_rcv, rl, D)
#     f_h, h_ref = h(f, z_src, z_rcv, r_rcv_ref, D)
#     rtf_mano = h_l / h_ref
#     print(f"Elapsed time: {time() - t0:.2f}s")
#     t0 = time()
#     f, g_l = g(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, rl)
#     print(f"Elapsed time: {time() - t0:.2f}s")

#     # print(np.abs(g_l / rtf_mano))
#     plt.figure()
#     plt.plot(f, np.abs(g_l), label="g")
#     plt.plot(f_h, np.abs(rtf_mano), label="mano")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("RTF")
#     plt.title("RTF")
#     plt.grid()
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    D = 1000
    fmin = 1
    fmax = 1e3
    z_src = 100
    z = 100
    r = 1e5

    f0 = 25
    # plot_modes(f0, D)
    # plot_tf(fmin, fmax, z_src, z, r, D)
    # plot_ir(D)
    # compare_tf(D)

    # # RTF
    D = 100
    z_src = 25
    z_rcv_ref = 90
    z_rcv = z_rcv_ref

    r_rcv_ref = 1e5
    rl = r_rcv_ref + 10

    f = np.arange(fmin, fmax, 0.1)
    t0 = time()
    f, g_l = g(f, z_src, z_rcv_ref, z_rcv, D, r_rcv_ref, rl)
