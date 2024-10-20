#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_utils.py
@Time    :   2024/10/17 10:11:34
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
# ======================================================================================================================
import scipy
import numpy as np
import matplotlib.pyplot as plt


from misc import *
from propa.rtf.ideal_waveguide import *
from signals.signals import generate_ship_signal
from propa.rtf.rtf_estimation.rtf_estimation_const import *
from real_data_analysis.real_data_utils import compute_csd_matrix_fast


def rtf_covariance_whitening(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the RTF using covariance whitening method described in  Markovich-Golan, S., & Gannot, S. (2015).
    """
    # Derive usefull params
    x = rcv_sig + rcv_noise
    n_rcv = x.shape[1]
    ts = t[1] - t[0]
    fs = 1 / ts

    f, Rx, Rs, Rv = get_csdm(t, rcv_sig, rcv_noise, nperseg, noverlap)

    ff, tt, stft_list_x = get_stft_list(x, fs, nperseg, noverlap)
    stft_x = np.array(stft_list_x)

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        Rv_f = Rv[i]
        # Rs_f = Rs[i]
        # Rx_f = Rx[i]
        stft_x_f = stft_x[:, i, :]

        # Cholesky decomposition of the noise csdm and its inverse : Equation (25a) and (25b)
        Rv_half = scipy.linalg.cholesky(Rv_f, lower=False)
        Rv_inv_f = np.linalg.inv(Rv_f)
        Rv_half_inv = scipy.linalg.cholesky(Rv_inv_f, lower=False)

        # Compute the whitened signal csdm : Equation (26)
        stft_y_f = Rv_half_inv @ stft_x_f

        # Compute the whitened signal csdm : Equation (31)
        # Reshape to the required shape for the computation
        stft_y_f = [
            stft_y_f[i, np.newaxis, :] for i in range(n_rcv)
        ]  # List of stft at frequency f : n_rcv element of shape (n_freq=1, n_seg)
        Ry_f = compute_csd_matrix_fast(
            stft_y_f, n_seg_cov="all"
        )  # Covariance matrix at frequency f
        Ry_f = (
            Ry_f.squeeze()
        )  # Remove useless frequency dimension to get shape (n_rcv, n_rcv)

        # Eigenvalue decomposition of Ry_f to get q (major eingenvector) : Equation (32)
        eig_val, eig_vect = np.linalg.eig(Ry_f)
        # We can check that the Ry_f can be diagonalized np.round(np.abs(np.linalg.inv(eig_vect) @ Ry_f @ eig_vect), 5)

        i_max_eig = np.argmax(np.abs(eig_val))
        q = eig_vect[:, i_max_eig]

        rtf_f = (Rv_half @ q) / (e1.T @ Rv_half @ q)  # Equation (32)
        rtf[i, :] = rtf_f

    return f, rtf, Rx, Rs, Rv


def rtf_covariance_substraction(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the RTF using covariance substraction method described in Markovich-Golan, S., & Gannot, S. (2015).
    """
    # Derive usefull params
    x = rcv_sig + rcv_noise
    n_rcv = x.shape[1]
    ts = t[1] - t[0]
    fs = 1 / ts

    # x = rcv_sig + rcv_noise
    f, Rx, Rs, Rv = get_csdm(t, rcv_sig, rcv_noise, nperseg, noverlap)

    # Check that Rs is of rank 1
    # for i in range(len(f)):
    #     rank = np.linalg.matrix_rank(Rs[i])
    #     print(f"Rank of Rs at f = {f[i]} Hz : {rank}")

    # Rx : CSDM of signal + noise
    # Rs : CSDM of signal
    # Rv : CSDM of noise
    R_delta = Rx - Rv  # Equation (9)

    # Loop over frequencies
    rtf = np.zeros((len(f), n_rcv), dtype=complex)
    # First receiver is considered as the reference
    e1 = np.eye(n_rcv)[:, 0]

    for i, f_i in enumerate(f):
        R_delta_f = R_delta[i]
        rtf_f = (R_delta_f @ e1) / (e1.T @ R_delta_f @ e1)
        rtf[i, :] = rtf_f

    return f, rtf, Rx, Rs, Rv


def get_csdm(t, rcv_sig, rcv_noise, nperseg=2**12, noverlap=2**11):
    """
    Derive the CSDM of the received signal and noise.
    """
    x = rcv_sig + rcv_noise
    ff, csdm_x = get_csdm_from_signal(t, x, nperseg, noverlap)
    ff, csdm_sig = get_csdm_from_signal(t, rcv_sig, nperseg, noverlap)
    ff, csdm_noise = get_csdm_from_signal(t, rcv_noise, nperseg, noverlap)

    return ff, csdm_x, csdm_sig, csdm_noise


def get_csdm_from_signal(t, y, nperseg=2**12, noverlap=2**11):
    """
    Derive the CSDM of y.
    """
    fs = 1 / (t[1] - t[0])
    ff, _, stft_list = get_stft_list(y, fs, nperseg, noverlap)
    csdm_y = compute_csd_matrix_fast(stft_list, n_seg_cov="all")

    return ff, csdm_y


def get_stft_list(y, fs, nperseg, noverlap):
    """
    Derive the STFT of each component of y.
    """

    stft_list = []
    n_rcv = y.shape[1]

    for i in range(n_rcv):
        ff, tt, stft = sp.stft(
            y[:, i],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        stft_list.append(stft)

    return ff, tt, stft_list


def derive_received_signal(tau_ir):
    """
    Derive the received signal at the receivers.
    The signal is modeled as a ship signal propagating in the ideal waveguide.
    """

    # Load params
    # depth, r_src, z_src, z_rcv, _ = waveguide_params()
    duration = 50 * tau_ir

    # Load kraken data
    kraken_data = load_data()

    # Define useful params
    n_rcv = kraken_data["n_rcv"]
    ts = kraken_data["t"][1] - kraken_data["t"][0]
    fs = 1 / ts

    # Create source signal
    f0 = 4.5
    # std_fi = 1e-2 * f0
    std_fi = 0.1 * f0
    tau_corr_fi = 0.1 * 1 / f0

    s, t = generate_ship_signal(
        Ttot=duration,
        f0=f0,
        std_fi=std_fi,
        tau_corr_fi=tau_corr_fi,
        fs=fs,
        normalize="max",
    )
    s *= np.hanning(len(s))
    # s /= np.std(s)

    src_spectrum = np.fft.rfft(s)

    # Derive psd
    psd = scipy.signal.welch(s, fs=fs, nperseg=2**12, noverlap=2**11)

    received_signal = {
        "t": t,
        "src": s,
        "f": kraken_data["f"],
        "spect": src_spectrum,
        "psd": psd,
        "std_fi": std_fi,
        "tau_corr_fi": tau_corr_fi,
        "f0": f0,
        "fs": fs,
        "tau_ir": tau_ir,
    }

    for i in range(n_rcv):
        # Get transfert function
        h_kraken = kraken_data[f"rcv{i}"]["h_f"]

        # Received signal spectrum resulting from the convolution of the source signal and the impulse response
        transmited_sig_field_f = h_kraken * src_spectrum
        rcv_sig = np.fft.irfft(transmited_sig_field_f)

        # psd
        psd_rcv = scipy.signal.welch(rcv_sig, fs=fs, nperseg=2**12, noverlap=2**11)

        received_signal[f"rcv{i}"] = {
            "sig": rcv_sig,
            "spect": transmited_sig_field_f,
            "psd": psd_rcv,
        }

    return received_signal


def load_data():

    kraken_data = {}
    for i in range(N_RCV):
        # Load tf
        fpath = os.path.join(ROOT_DATA, f"kraken_tf_rcv{i}.csv")
        f_kraken, h_kraken_real, h_kraken_imag = np.loadtxt(
            fpath, delimiter=",", unpack=True
        )
        h_kraken = h_kraken_real + 1j * h_kraken_imag

        # Load ir kraken
        fpath = os.path.join(ROOT_DATA, f"kraken_ir_rcv{i}.csv")
        t_kraken, ir_kraken = np.loadtxt(fpath, delimiter=",", unpack=True)

        kraken_data[f"rcv{i}"] = {"ir": ir_kraken, "h_f": h_kraken}

    kraken_data.update({"t": t_kraken, "f": f_kraken, "n_rcv": N_RCV})

    return kraken_data


if __name__ == "__main__":
    pass
