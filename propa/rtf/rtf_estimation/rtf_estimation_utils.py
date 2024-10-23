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
import xarray as xr
import matplotlib.pyplot as plt


from misc import *
from propa.rtf.ideal_waveguide import *
from signals.signals import generate_ship_signal, z_call, ricker_pulse, dirac
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


def derive_received_signal(tau_ir, delay_correction=0, sl=200):
    """
    Derive the received signal at the receivers.
    The signal is modeled as a ship signal propagating in the ideal waveguide.

    sl : 200 dB re 1 uPa at 1 m (default value) according to Gassmann, M., Wiggins, S. M., & Hildebrand, J. A. (2017).
    Deep-water measurements of container ship radiated noise signatures and directionality.
    The Journal of the Acoustical Society of America, 142(3), 1563–1574. https://doi.org/10.1121/1.5001063

    Parameters
    ----------
    tau_ir : float
        Impulse response duration.
    delay_correction : float
        Delay correction to apply to the signal.
    sl : float
        Source level in dB re 1 uPa at 1 m.

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
    f = kraken_data["f"]

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
        normalize="sl",
        sl=sl,
    )

    p0 = 1e-6  # 1 uPa
    print(
        f"Effective source level before windowing : {20 * np.log10(np.std(s) / p0)} dB re 1 uPa at 1 m"
    )
    # Apply windowing to avoid side-effects
    # s *= sp.windows.tukey(len(s), alpha=0.5)
    s *= np.hanning(len(s))
    src_spectrum = np.fft.rfft(s)
    # Apply delay correction so that the signal is centered within the time window (otherwise the signal is shifted with wrap around effect in the time domain)
    src_spectrum *= np.exp(1j * 2 * np.pi * f * delay_correction)

    # Derive psd
    psd = scipy.signal.welch(s, fs=fs, nperseg=2**12, noverlap=2**11)

    received_signal = {
        "t": t,
        "src": s,
        "f": f,
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


def derive_received_noise(
    ns, fs, propagated=False, noise_model="gaussian", snr_dB=10, propagated_args={}
):

    received_noise = {}

    # Compute the received noise signal
    if propagated:

        # Load noise dataset
        ds_tf = xr.open_dataset(os.path.join(ROOT_DATA, "kraken_tf_surface_noise.nc"))

        delta_rcv = 500
        if "rmin" in propagated_args.keys() and propagated_args["rmin"] is not None:
            rmin_noise = propagated_args["rmin"]
        else:
            rmin_noise = 5 * 1e3  # Default minimal range for noise source

        if "rmax" in propagated_args.keys() and propagated_args["rmax"] is not None:
            rmax_noise = propagated_args["rmax"]
        else:
            rmax_noise = ds_tf.r.max().values  # Default maximal range for noise source

        for i in range(N_RCV):
            r_src_noise_start = rmin_noise - i * delta_rcv
            r_src_noise_end = rmax_noise - i * delta_rcv
            idx_r_min = np.argmin(np.abs(ds_tf.r.values - r_src_noise_start))
            idx_r_max = np.argmin(np.abs(ds_tf.r.values - r_src_noise_end))

            tf_noise_rcv_i = (
                ds_tf.tf_real[:, idx_r_min:idx_r_max]
                + 1j * ds_tf.tf_imag[:, idx_r_min:idx_r_max]
            )

            # Noise spectrum
            if noise_model == "gaussian":
                v = np.random.normal(loc=0, scale=1, size=ns)
                noise_spectrum = np.fft.rfft(v)

            # Multiply the transfert function by the noise source spectrum
            noise_field_f = mult_along_axis(tf_noise_rcv_i, noise_spectrum, axis=0)
            noise_field = np.fft.irfft(noise_field_f, axis=0)
            noise_sig = np.sum(noise_field, axis=1)  # Sum over all noise sources

            # Normalise to required lvl at receiver 0
            if i == 0:
                sigma_v2 = 10 ** (-snr_dB / 10)
                sigma_noise = np.std(noise_sig)
                alpha = np.sqrt(sigma_v2) / sigma_noise

            noise_sig *= alpha

            # Psd
            psd_noise = scipy.signal.welch(
                noise_sig, fs=fs, nperseg=2**12, noverlap=2**11
            )

            # Save noise signal
            received_noise[f"rcv{i}"] = {
                "psd": psd_noise,
                "sig": noise_sig,
                "spect": noise_field_f,
            }

    else:
        if noise_model == "gaussian":
            for i in range(N_RCV):
                sigma_v2 = 10 ** (-snr_dB / 10)
                v = np.random.normal(loc=0, scale=np.sqrt(sigma_v2), size=ns)
                noise_sig = v
                noise_spectrum = np.fft.rfft(noise_sig)

                # Psd
                psd_noise = scipy.signal.welch(
                    noise_sig, fs=fs, nperseg=2**12, noverlap=2**11
                )

                # Save noise signal
                received_noise[f"rcv{i}"] = {
                    "psd": psd_noise,
                    "sig": noise_sig,
                    "spect": noise_spectrum,
                }

    return received_noise


def derive_received_interference(ns, fs, interference_arg={}):

    # Load values and set default values
    if "signal_type" in interference_arg.keys():
        signal_type = interference_arg["signal_type"]
    else:
        signal_type = "z_call"

    if "src_position" in interference_arg.keys():
        src_position = interference_arg["src_position"]
        r_src = np.atleast_1d(src_position["r"])
        z_src = np.atleast_1d(src_position["z"])
        n_src = len(r_src)
    else:
        r_src = 20 * 1e3
        z_src = 20
        n_src = 1

    received_interference = {}

    # Load transfert function dataset
    delta_rcv = 500
    ds_tf = xr.open_dataset(os.path.join(ROOT_DATA, "kraken_tf_loose_grid.nc"))

    for i in range(N_RCV):
        r_src_rcv = r_src - i * delta_rcv
        idx_r = [np.argmin(np.abs(ds_tf.r.values - r_src_rcv[k])) for k in range(n_src)]
        idx_z = [np.argmin(np.abs(ds_tf.z.values - z_src[k])) for k in range(n_src)]

        tf_interference_rcv_i = np.array(
            [
                ds_tf.tf_real[:, idx_z[k], idx_r[k]]
                + 1j * ds_tf.tf_imag[:, idx_z[k], idx_r[k]]
                for k in range(n_src)
            ]
        ).T
        # tf_interference_rcv_i = (
        #     ds_tf.tf_real[:, idx_z, idx_r] + 1j * ds_tf.tf_imag[:, idx_z, idx_r]
        # )
        # tf_interference_rcv_i = np.atleast_2d(tf_interference_rcv_i)

        # Interference signal
        if signal_type == "z_call":

            signal_args = {
                "fs": fs,
                "nz": 1,  # Let the function derived the maximum number of z-calls in the signal duration
                "signal_duration": ns / fs,
                "sl": 188.5,
                "start_offset_seconds": 100,
            }
            interference_sig, _ = z_call(signal_args)
            interference_spectrum = np.fft.rfft(interference_sig)

        if signal_type == "ricker_pulse":
            interference_sig, _ = ricker_pulse(
                fc=10, fs=fs, T=ns / fs, center=True, normalize="sl", sl=188.5
            )
            interference_spectrum = np.fft.rfft(interference_sig)

        if signal_type == "dirac":
            interference_sig, _ = dirac(
                fs=fs, T=ns / fs, center=True, normalize="sl", sl=188.5
            )
            interference_spectrum = np.fft.rfft(interference_sig)

        # Multiply the transfert function by the interference source spectrum
        interference_field_f = mult_along_axis(
            tf_interference_rcv_i, interference_spectrum, axis=0
        )
        interference_field = np.fft.irfft(interference_field_f, axis=0)
        interference_sig = np.sum(
            interference_field, axis=1
        )  # Sum over all interference sources

        # Psd
        psd_interference = scipy.signal.welch(
            interference_sig, fs=fs, nperseg=2**12, noverlap=2**11
        )

        # Save interference signal
        received_interference[f"rcv{i}"] = {
            "psd": psd_interference,
            "sig": interference_sig,
            "spect": interference_field_f,
        }

    return received_interference


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
