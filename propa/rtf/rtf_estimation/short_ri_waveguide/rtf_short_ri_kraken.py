#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_estimation_kraken.py
@Time    :   2024/10/17 09:15:19
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.interpolate as sp_int

from misc import *
from propa.rtf.ideal_waveguide import *
from propa.kraken_toolbox.run_kraken import runkraken
from propa.rtf.ideal_waveguide import waveguide_params
from localisation.verlinden.testcases.testcase_envs import TestCase1_0
from real_data_analysis.real_data_utils import get_csdm_snapshot_number
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_consts import *

from propa.rtf.rtf_estimation_const import *
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_plot_tools import *
from signals.signals import generate_ship_signal, dirac, z_call, ricker_pulse


def derive_kraken_tf():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    t0 = time()
    f, ts, h_kraken_dict = run_kraken_simulation(r_src, z_src, z_rcv, depth)
    print("Broadband kraken: ", time() - t0)

    # Some nan values can appear in the transfert function
    # h_kraken = np.nan_to_num(h_kraken)
    for i in range(N_RCV):
        h_kraken_dict[f"rcv{i}"] = np.nan_to_num(h_kraken_dict[f"rcv{i}"])
        h_kraken = h_kraken_dict[f"rcv{i}"]

        # Save transfert function as a csv
        fpath = os.path.join(ROOT_DATA, f"kraken_tf_rcv{i}.csv")
        np.savetxt(fpath, np.array([f, h_kraken.real, h_kraken.imag]).T, delimiter=",")

        ir_kraken = fft.irfft(h_kraken)
        t_kraken = np.arange(0, len(ir_kraken)) * ts

        # Save kraken ir
        fpath = os.path.join(ROOT_DATA, f"kraken_ir_rcv{i}.csv")
        np.savetxt(fpath, np.array([t_kraken, ir_kraken]).T, delimiter=",")


def derive_kraken_tf_surface_noise():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    t0 = time()
    f, ts, r, h_kraken_surface_noise = run_kraken_simulation_surface_noise(
        r_src, z_src, z_rcv, depth
    )
    print("Broadband kraken: ", time() - t0)

    # Define xarray dataset for the transfert function
    h_kraken_surface_noise_xr = xr.Dataset(
        data_vars=dict(
            tf_real=(["f", "r"], np.real(h_kraken_surface_noise)),
            tf_imag=(["f", "r"], np.imag(h_kraken_surface_noise)),
        ),
        coords={"f": f, "r": r},
    )
    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf_surface_noise.nc")
    h_kraken_surface_noise_xr.to_netcdf(fpath)


def derive_kraken_tf_loose_grid():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    f, ts, r, z, h_kraken_loose_grid = run_kraken_simulation_loose_grid(
        r_src, z_src, z_rcv, depth
    )

    # Define xarray dataset for the transfert function
    h_kraken_surface_noise_xr = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "z", "r"],
                np.real(h_kraken_loose_grid),
            ),
            tf_imag=(["f", "z", "r"], np.imag(h_kraken_loose_grid)),
        ),
        coords={"f": f, "r": r, "z": z},
    )
    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf_loose_grid.nc")
    h_kraken_surface_noise_xr.to_netcdf(fpath)


def derive_kraken_tf_mfp_grid():

    # Load params
    depth, r_src, z_src, z_rcv, _ = waveguide_params()

    # Run kraken
    f, r, z, h_kraken_loose_grid = run_kraken_simulation_mfp_grid(
        r_src, z_src, z_rcv, depth
    )

    # Define xarray dataset for the transfert function
    h_kraken_surface_noise_xr = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "z", "r"],
                np.real(h_kraken_loose_grid),
            ),
            tf_imag=(["f", "z", "r"], np.imag(h_kraken_loose_grid)),
        ),
        coords={"f": f, "r": r, "z": z},
    )
    # Save transfert function as a csv
    fpath = os.path.join(ROOT_DATA, "kraken_tf_mfp_grid.nc")
    h_kraken_surface_noise_xr.to_netcdf(fpath)


def run_kraken_simulation(r_src, z_src, z_rcv, depth):

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    r_src_rcv = r_src - x_rcv

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": r_src,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": 1,
        "flp_rcv_z_min": z_rcv,
        "flp_rcv_z_max": z_rcv,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": delta_rcv,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    h_kraken_dict = {f"rcv{i}": np.zeros_like(f, dtype=complex) for i in range(N_RCV)}

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=False,
        )

        idx_r = [
            np.argmin(np.abs(field_pos["r"]["r"] - r_src_rcv[i])) for i in range(N_RCV)
        ]
        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2, 3))[:, idx_r]
        # print(pad_before, pad_after)
        for i in range(N_RCV):
            # Zero padding of the transfert function to match the length of the global transfert function
            h_kraken_dict[f"rcv{i}"] += np.pad(
                h_kraken_subband[:, i], (pad_before, pad_after)
            )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, h_kraken_dict


def run_kraken_simulation_surface_noise(r_src, z_src, z_rcv, depth):

    # Noise source spacing (m): the minimal wavelength is 30m for f = 50Hz and c = 1500m/s, dr = 1m ensure at least 30 points per wavelength
    dr_noise = 10
    # z_src = 0.5  # Noise source just below surface
    rmax_noise = r_src + 10 * 1e3  # Maximal range for noise source

    # Reciprocity
    z_src = z_rcv
    z_rcv = 0.5  # Noise source just below surface

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    # r_src_rcv = r_src - x_rcv

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": rmax_noise,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": 1,
        "flp_rcv_z_min": z_rcv,
        "flp_rcv_z_max": z_rcv,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": dr_noise,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    nr = int(rmax_noise / dr_noise + 1)
    nf = len(f)
    # h_kraken_dict = {f"rcv{i}": np.zeros((nf, nr), dtype=complex) for i in range(N_RCV)}
    h_kraken_surface_noise = np.zeros((nf, nr), dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2, 3))
        r = field_pos["r"]["r"]

        h_kraken_surface_noise += np.pad(
            h_kraken_subband, ((pad_before, pad_after), (0, 0))
        )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, r, h_kraken_surface_noise


def run_kraken_simulation_loose_grid(r_src, z_src, z_rcv, depth):

    # Noise source spacing (m): the minimal wavelength is 30m for f = 50Hz and c = 1500m/s, dr = 1m ensure at least 30 points per wavelength
    dr_loose_grid = 100
    rmax = r_src + 20 * 1e3  # Maximal range

    delta_rcv = 500
    x_rcv = np.array([i * delta_rcv for i in range(N_RCV)])
    r_src_rcv = r_src - x_rcv

    # Reciprocity
    z_src = z_rcv
    # Potential interferers located near the surface
    dz_loose_grid = 1
    z_min = 1
    z_max = 50
    nz = int((z_max - z_min) / dz_loose_grid) + 1

    # Create the frequency vector
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": rmax,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": nz,
        "flp_rcv_z_min": z_min,
        "flp_rcv_z_max": z_max,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": dr_loose_grid,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 500
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    nr = int(rmax / dr_loose_grid + 1)
    nf = len(f)
    h_kraken_loose_grid = np.zeros((nf, nz, nr), dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2))
        r = field_pos["r"]["r"]
        z = field_pos["r"]["z"]

        h_kraken_loose_grid += np.pad(
            h_kraken_subband, ((pad_before, pad_after), (0, 0), (0, 0))
        )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, ts, r, z, h_kraken_loose_grid


def run_kraken_simulation_mfp_grid(r_src, z_src, z_rcv, depth):

    dr_mfp_grid = 50
    rmax = r_src + 20 * 1e3  # Maximal range

    # Reciprocity
    z_src = z_rcv
    # Potential interferers located near the surface
    dz_mfp_grid = 5
    z_min = 0
    z_max = 200
    nz = int((z_max - z_min) / dz_mfp_grid) + 1

    # Create the frequency vector
    # nf = 1025
    # f = np.linspace(0, 50, nf)
    duration = 50 * TAU_IR
    ts = 1e-2
    nt = int(duration / ts)
    f = fft.rfftfreq(nt, ts)

    # Init env
    bott_hs_properties = testcase_bottom_properties()

    tc_varin = {
        "freq": f,
        "src_depth": z_src,
        "max_range_m": rmax,
        "mode_theory": "adiabatic",
        "flp_n_rcv_z": nz,
        "flp_rcv_z_min": z_min,
        "flp_rcv_z_max": z_max,
        "min_depth": depth,
        "max_depth": depth,
        "dr_flp": dr_mfp_grid,
        "nb_modes": 200,
        "bottom_boundary_condition": "acousto_elastic",
        "nmedia": 2,
        "phase_speed_limits": [200, 20000],
        "bott_hs_properties": bott_hs_properties,
    }
    tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
    title = "Simple waveguide with short impulse response"
    tc.title = title
    tc.env_dir = os.path.join(ROOT_FOLDER, "tmp")
    tc.update(tc_varin)

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    fmax = 50
    fmin = cutoff_frequency(C0, depth, bottom_bc="pressure_release")
    n_subband = 600
    i_subband = 1
    f0 = fmin
    f1 = f[n_subband]
    # h_kraken = np.zeros_like(f, dtype=complex)
    nr = int(rmax / dr_mfp_grid + 1)
    nf = len(f)
    h_kraken_loose_grid = np.zeros((nf, nz, nr), dtype=complex)

    while f0 < fmax:
        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Update env
        varin_update = {"freq": f_kraken}
        tc.update(varin_update)

        pressure_field, field_pos = runkraken(
            env=tc.env,
            flp=tc.flp,
            frequencies=tc.env.freq,
            parallel=True,
            verbose=True,
        )

        h_kraken_subband = np.squeeze(pressure_field, axis=(1, 2))
        r = field_pos["r"]["r"]
        z = field_pos["r"]["z"]

        h_kraken_loose_grid += np.pad(
            h_kraken_subband, ((pad_before, pad_after), (0, 0), (0, 0))
        )

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    return f, r, z, h_kraken_loose_grid


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

    # p0 = 1e-6  # 1 uPa
    # print(
    #     f"Effective source level before windowing : {20 * np.log10(np.std(s) / p0)} dB re 1 uPa at 1 m"
    # )

    # Apply windowing to avoid side-effects
    # s *= sp.windows.tukey(len(s), alpha=0.5)
    s *= np.hanning(len(s))
    src_spectrum = np.fft.rfft(s)
    # Apply delay correction so that the signal is centered within the time window (otherwise the signal is shifted with wrap around effect in the time domain)
    src_spectrum *= np.exp(1j * 2 * np.pi * f * delay_correction)

    # tau = f * delay_correction
    # print(f"Tau corr at src pos (2000, 2050): {tau[2000:2050]}")
    # Derive psd
    psd = scipy.signal.welch(s, fs=fs, nperseg=2**12, noverlap=2**11)

    received_signal = {
        "t": t,
        "src": s,
        "f": f,
        "n_rcv": n_rcv,
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
            "psd": psd_rcv,
            "spect": transmited_sig_field_f,
        }

    return received_signal


def derive_received_noise(
    ns,
    fs,
    propagated=False,
    noise_model="gaussian",
    snr_dB=10,
    sigma_ref=1,
    propagated_args={},
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

            # Normalize to account for the signal power to reach required snr at receiver 0
            noise_sig *= sigma_ref

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
                # Normalize to account for the signal power to reach required snr at receiver 0
                noise_sig *= sigma_ref
                # Derive noise spectrum
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

        # Interference signal
        if signal_type == "z_call":

            signal_args = {
                "fs": fs,
                "nz": 0,  # Let the function derived the maximum number of z-calls in the signal duration
                "signal_duration": ns / fs,
                "sl": 188.5,
                # "sl": 170,
                # "start_offset_seconds": 0,
                # "end_offset_seconds": 0,
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


def testcase_bottom_properties():
    # Environment with properties to minimize the impulse response duration
    bott_hs_properties = {
        "rho": 1.5 * RHO_W * 1e-3,  # Density (g/cm^3)
        # "c_p": 1500,  # P-wave celerity (m/s)
        "c_p": 1550,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.2,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
        "z": None,
    }

    return bott_hs_properties


if __name__ == "__main__":
    # derive_kraken_tf()
    # derive_kraken_tf_surface_noise()
    # derive_kraken_tf_loose_grid()
    # kraken_data = load_data()
    # plot_ir(kraken_data, shift_ir=True)
    # plot_tf(kraken_data)

    # xr_surfnoise = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_surface_noise.nc"
    # )
    # xr_surfnoise_rcv = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_noise_rcv0.nc"
    # )

    # rcv_sig = derive_received_signal()
    # rcv_sig = derive_received_signal(tau_ir=TAU_IR)
    # plot_signal(rcv_sig=rcv_sig, root_img=ROOT_IMG)
    # plt.show()

    derive_kraken_tf_mfp_grid()
    # pass
