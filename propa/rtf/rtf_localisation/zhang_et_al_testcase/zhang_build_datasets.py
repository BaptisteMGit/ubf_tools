#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_build_datasets.py
@Time    :   2025/01/27 11:56:49
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Functions to build useful dataset for the Zhang et al 2023 testcase 
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import dask.array as da

from time import time
from dask import delayed
from dask.diagnostics import ProgressBar
from misc import cast_matrix_to_target_shape, mult_along_axis
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import *
from propa.kraken_toolbox.run_kraken import readshd, run_kraken_exec, run_field_exec
from propa.rtf.rtf_estimation.rtf_estimation_utils import rtf_covariance_substraction

# ======================================================================================================================
# Functions
# ======================================================================================================================


def build_tf_dataset():
    """Step 1 : use Kraken propagation model to derive the broadband transfert function of the testcase waveguide"""

    library_props, S_f_library, freq, idx_in_band = library_src_spectrum()

    f = freq[idx_in_band]

    # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
    n_subband = 900
    i_subband = 1
    f0 = f[0]
    f1 = f[n_subband]

    # Load tf to get range vector information
    fpath = os.path.join(ROOT_DATA, "tf_zhang.nc")
    ds = xr.open_dataset(fpath)
    nr = ds.sizes["r"]

    nf = len(f)
    h_grid = np.zeros((nf, nr), dtype=complex)

    fname = "testcase_zhang2023"
    working_dir = os.path.join(ROOT, "tmp")
    env_file = os.path.join(working_dir, f"{fname}.env")

    # Read env file
    with open(env_file, "r") as file:
        lines = file.readlines()

    while f0 < f[-1]:

        # Frequency subband
        f_kraken = f[(f < f1) & (f >= f0)]
        # print(i_subband, f0, f1, len(f_kraken))
        pad_before = np.sum(f < f0)
        pad_after = np.sum(f >= f1)

        # Modify number of frequencies
        nb_freq = f"{len(f_kraken)}                                                     ! Number of frequencies\n"
        lines[-2] = nb_freq
        # Replace frequencies in the env file
        new_freq_line = " ".join([f"{fi:.2f}" for fi in f_kraken])
        new_freq_line += "    ! Frequencies (Hz)"
        lines[-1] = new_freq_line

        # Write new env file
        with open(env_file, "w") as file:
            file.writelines(lines)

        # Run kraken and field
        os.chdir(working_dir)
        run_kraken_exec(fname)
        run_field_exec(fname)

        # Read shd from previously run kraken
        shdfile = f"{fname}.shd"

        _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
            filename=shdfile, freq=f_kraken
        )
        tf_subband = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)

        h_grid += np.pad(tf_subband, ((pad_before, pad_after), (0, 0)))

        # Update frequency subband
        i_subband += 1
        f0 = f1
        f1 = f[min(n_subband * i_subband, len(f) - 1)]

    # return f, r, z, h_grid

    # Pad h_grid with 0 for frequencies outside the 50 - 550 Hz band
    pad_before = np.sum(freq < 50)
    pad_after = np.sum(freq > 550)
    h_grid = np.pad(h_grid, ((pad_before, pad_after), (0, 0)))

    # Build xarray dataset
    library_zhang = xr.Dataset(
        data_vars=dict(
            tf_real=(
                ["f", "r"],
                np.real(h_grid),
            ),
            tf_imag=(["f", "r"], np.imag(h_grid)),
        ),
        coords={
            "f": freq,
            "r": field_pos["r"]["r"],
        },
    )

    # Save as netcdf
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    library_zhang.to_netcdf(fpath)


def grid_dataset():
    """Step 2 : Associate each grid pixel to the corresponding broadband transfert function caracterized by the range to the receiver."""

    # Load dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    ds = xr.open_dataset(fpath)

    # Load param
    depth, receivers, source, grid, frequency, _ = params()

    # Create new dataset
    ds_grid = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            x=grid["x"][0, :],
            y=grid["y"][:, 0],
            idx_rcv=range(len(receivers["x"])),
        ),
        attrs=dict(
            df=ds.f.diff("f").values[0],
            dx=grid["dx"],
            dy=grid["dy"],
            testcase="zhang_et_al_2023",
        ),
    )

    # Grid tf to the desired resolution
    # Preprocess tf to decrease the number of point for further interpolation
    r_grid_all_rcv = np.array(
        [grid["r"][i_rcv].flatten() for i_rcv in range(len(receivers["x"]))]
    )
    r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

    tf_vect = ds.tf_real.sel(
        r=r_grid_all_rcv_unique, method="nearest"
    ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")

    tf_grid_ds = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            r=r_grid_all_rcv_unique,
        ),
        data_vars=dict(tf=(["f", "r"], tf_vect.values)),
    )

    gridded_tf = []
    grid_shape = (ds_grid.sizes["f"], ds_grid.sizes["x"], ds_grid.sizes["y"])
    for i_rcv in range(len(receivers["x"])):
        r_grid = grid["r"][i_rcv].flatten()
        tf_ircv = tf_grid_ds.tf.sel(r=r_grid, method="nearest")

        tf_grid = tf_ircv.values.reshape(grid_shape)
        gridded_tf.append(tf_grid)

    gridded_tf = np.array(gridded_tf)
    # Add to dataset
    grid_coords = ["idx_rcv", "f", "x", "y"]
    ds_grid["tf_real"] = (grid_coords, np.real(gridded_tf))
    ds_grid["tf_imag"] = (grid_coords, np.imag(gridded_tf))

    # Save dataset
    fname = f"tf_zhang_grid_dx{ds_grid.dx}m_dy{ds_grid.dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_grid.to_netcdf(fpath)
    ds_grid.close()


def build_signal():
    """Step 3 : derive signal received from each grid pixel using the library source spectrum."""

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    # Load gridded dataset
    fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds = xr.open_dataset(fpath)

    # Limit max frequency to speed up
    fs_target = 1200
    fmax = fs_target / 2
    ds = ds.sel(f=slice(0, fmax))

    # Load library spectrum
    # f = ds.f.values
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum(
        fs=fs_target
    )

    # Load event spectrum
    _, S_f_event, _ = event_src_spectrum(
        # target_var=library_props["sig_var"],
        T=library_props["T"],
        fs=library_props["fs"],
    )

    # Derive delay for each receiver
    delay_rcv = []
    for i_rcv in range(len(receivers["x"])):
        r_grid = grid["r"][i_rcv].flatten()
        tau_rcv = r_grid / C0
        tau_rcv = tau_rcv.reshape((ds.sizes["x"], ds.sizes["y"]))
        delay_rcv.append(tau_rcv)

    delay_rcv = np.array(delay_rcv)

    # Add delay to dataset
    ds["delay_rcv"] = (["idx_rcv", "x", "y"], delay_rcv)

    # Same delay is applied to each receiver : the receiver with the minimum delay is taken as the time reference
    # (we are only interested on relative time difference)
    tau = ds.delay_rcv.min(dim="idx_rcv")
    # Cast tau to grid shape
    tau_lib = cast_matrix_to_target_shape(tau, ds.tf_real.shape[1:])

    y_t_event = []
    y_t_library = []
    for i_rcv in range(len(receivers["x"])):

        tf_library = ds.tf_real.sel(idx_rcv=i_rcv) + 1j * ds.tf_imag.sel(idx_rcv=i_rcv)
        tf_event = tf_library.sel(x=source["x"], y=source["y"], method="nearest")

        # Derive received spectrum (Y = SH)
        k0 = 2 * np.pi * ds.f / C0
        norm_factor = np.exp(1j * k0) / (4 * np.pi)

        y_f_library = mult_along_axis(tf_library, S_f_library * norm_factor, axis=0)
        y_f_event = tf_event * S_f_event * norm_factor

        # Derive delay factor to take into account the propagation time
        tau_vec = mult_along_axis(tau_lib, ds.f, axis=0)
        delay_library = np.exp(1j * 2 * np.pi * tau_vec)

        tau_event = tau.sel(x=source["x"], y=source["y"], method="nearest")
        delay_event = np.exp(1j * 2 * np.pi * tau_event * ds.f)

        # Apply delay
        y_f_library *= delay_library
        y_f_event *= delay_event

        # FFT inv to get signal
        y_t_l = np.fft.irfft(y_f_library, axis=0)
        y_t_e = np.fft.irfft(y_f_event)

        # Store for current receiver
        y_t_library.append(y_t_l)
        y_t_event.append(y_t_e)

        # plt.figure()
        # plt.plot(y_t_e)
        # plt.savefig(f"test_{i_rcv}_e.png")
        # plt.figure()
        # plt.plot(y_t_l)
        # plt.savefig(f"test_{i_rcv}_l.png")
        # plt.show()

    y_t_library = np.array(y_t_library)
    y_t_event = np.array(y_t_event)

    # Add to dataset
    t = np.arange(0, library_props["T"], 1 / library_props["fs"])
    ds.coords["t"] = t
    ds["s_l"] = (["idx_rcv", "t", "x", "y"], y_t_library)
    ds["s_e"] = (["idx_rcv", "t"], y_t_event)

    # Drop vars to reduce size
    ds = ds.drop_vars(["tf_real", "tf_imag", "f"])

    # Save dataset
    fname = f"zhang_library_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds.to_netcdf(fpath)


def derive_received_noise(
    s_library,
    s_event_rcv0,
    event_source,
    snr_dB=10,
    noise_model="gaussian",
):
    """Derive the noise grid field.
    The noise power is defined to match required snr relative to the signal level for the
    """

    # Create noise dataarray
    noise_da = xr.zeros_like(s_library)

    # Build noise signal
    sigma_noise = np.sqrt(10 ** (-snr_dB / 10))
    # Correct definition should use multivariate gaussian noise to ensure the std is the same at each grid pixel
    # Yet, for simplicity we use a simple gaussian distibution over the entire grid to increase perf (this is valid for long time series)

    # nb_noise_vectors = (
    #     rcv_sig_xr.sizes["r"] * rcv_sig_xr.sizes["z"] * rcv_sig_xr.sizes["idx_rcv"]
    # )
    # mean_noise = np.zeros(nb_noise_vectors)
    # cov_noise = np.eye(nb_noise_vectors) * sigma_noise**2
    # noise_sig = np.random.multivariate_normal(
    #     mean_noise, cov_noise, size=rcv_sig_xr.sizes["t"]
    # )
    # # Reshape to xarray shape
    # noise_sig = noise_sig.reshape(rcv_sig_xr.shape)

    # Derive noise signal
    noise_sig = np.random.normal(loc=0, scale=sigma_noise, size=s_library.shape)
    # Normalize to account for the signal power to reach required snr at receiver n°0
    # We assume that the noise is due to ambiant noise and is the same at each receiver position and does not depend on the source position within the search grid
    sigma_rcv_ref = np.std(s_event_rcv0.values)
    noise_sig *= sigma_rcv_ref

    # Store in xarray
    noise_da.values = noise_sig
    noise_da.attrs["sigma_ref"] = sigma_rcv_ref
    noise_da.attrs["sigma_noise"] = sigma_noise

    # Check snr
    n_e = noise_da.sel(x=event_source["x"], y=event_source["y"], method="nearest")
    snr = 10 * np.log10(np.var(s_event_rcv0.values) / np.var(n_e.values))
    print(f"SNR at event source position : {np.round(snr, 2)} dB (required {snr_dB}dB)")

    return noise_da


def estimate_rtf(
    ds_sig, noise_da, i_ref, source, library_props, nperseg=2**11, noverlap=2**10
):
    # By default rtf estimation method assumed the first receiver as the reference -> need to roll along the receiver axis
    ds_sig_rtf = ds_sig.copy(deep=True)
    idx_pos_ref = np.argmin(np.abs(ds_sig_rtf.idx_rcv.values - i_ref))
    npos_to_roll = ds_sig_rtf.sizes["idx_rcv"] - idx_pos_ref
    ds_sig_rtf = ds_sig_rtf.roll(
        idx_rcv=npos_to_roll,
        roll_coords=True,
    )
    t = ds_sig_rtf.t.values

    # Noise
    rcv_noise = noise_da.sel(x=source["x"], y=source["y"], method="nearest").values
    rcv_noise = rcv_noise.T

    ## Event ##
    rcv_sig = ds_sig_rtf.s_e.values

    # Transpose to fit rtf estimation required input shape (ns, nrcv)
    rcv_sig = rcv_sig.T
    f_rtf, rtf_cs_e, _, _, _ = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    ## Library ##
    # Dummy and dirty way to derive the RTF from the received signal -> loop over each grid pixel

    # Signal
    rcv_sig = ds_sig_rtf.s_l.sel(x=source["x"], y=source["y"], method="nearest").values

    # Transpose to fit rtf estimation required input shape (ns, nrcv)
    rcv_sig = rcv_sig.T

    # Use the first signal slice to set f_cs by running rtf_covariance_substraction once
    f_rtf, rtf, _, _, _ = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # # Use Dask delayed and progress bar for tracking
    with ProgressBar():
        results_cs = []
        # results_cw = []
        for x_i in ds_sig_rtf.x.values:
            for y_i in ds_sig_rtf.y.values:
                rcv_sig = ds_sig_rtf.s_l.sel(x=x_i, y=y_i).values
                rcv_noise = noise_da.sel(x=x_i, y=y_i).values

                # Transpose to fit rtf estimation required input shape (ns, nrcv)
                rcv_sig = rcv_sig.T
                rcv_noise = rcv_noise.T

                # Wrap function call in dask delayed
                delayed_rtf_cs = da.from_delayed(
                    delayed(
                        lambda t, sig, noise: rtf_covariance_substraction(
                            t, sig, noise, nperseg, noverlap
                        )[1]
                    )(t, rcv_sig, rcv_noise),
                    shape=(len(f_rtf), rcv_sig.shape[1]),
                    dtype=complex,
                )
                results_cs.append(delayed_rtf_cs)

        # Compute all RTFs
        rtf_cs_l = da.stack(results_cs).compute()

    # Reshape to the required shape
    # First step : reshape to (nx, ny, nf, n_rcv)
    shape = (len(ds_sig.x), len(ds_sig.y)) + rtf_cs_l.shape[1:]
    rtf_cs_l = rtf_cs_l.reshape(shape)

    # Step 2 : permute to (nf, nx, ny, n_rcv)
    axis_permutation = (2, 0, 1, 3)
    rtf_cs_l = np.transpose(rtf_cs_l, axis_permutation)

    # Restict to the frequency band of interest
    idx_band = (f_rtf >= library_props["f0"]) & (f_rtf <= library_props["f1"])
    f_rtf = f_rtf[idx_band]
    rtf_cs_l = rtf_cs_l[idx_band]
    rtf_cs_e = rtf_cs_e[idx_band]

    return rtf_cs_l, rtf_cs_e


def build_features_from_time_signal(snr_dB=0, all_rcv_ref=True):
    """Step 4.2 : build localisation features for DCF GCC and RTF MFP from syntethic time signal."""

    t_start = time()

    # Load params
    depth, receivers, source, grid, frequency, _ = params()
    dx = grid["dx"]
    dy = grid["dy"]

    # Dataset with time signal (for realistic approach)
    fname = f"zhang_library_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_sig = xr.open_dataset(fpath)

    # Load library spectrum
    fs = 1200
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum(fs=fs)

    # Load event spectrum
    # _, S_f_event = event_src_spectrum(f_library)

    rtf_event = []  # RFT vector at the source position
    rtf_library = []  # RTF vector evaluated at each grid pixel
    gcc_event = []  # GCC vector evaluated at the source position
    gcc_library = []  # GCC-SCOT vector evaluated at each grid pixel

    noise_da = derive_received_noise(
        s_library=ds_sig.s_l,
        s_event_rcv0=ds_sig.s_e.sel(idx_rcv=0),
        event_source=source,
        snr_dB=snr_dB,
    )

    # Add noise
    ds_sig_gcc = ds_sig.copy(deep=True)
    ds_sig_gcc["s_l"] += noise_da
    ds_sig_gcc["s_e"] += noise_da.sel(x=source["x"], y=source["y"], method="nearest")

    # Plot signal and noise
    f, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    # Only signal at source position
    ds_sig.s_l.sel(idx_rcv=0, x=source["x"], y=source["y"], method="nearest").plot(
        ax=axs[0]
    )
    axs[0].set_title("Library signal at source position")
    # Only noise at source position
    noise_da.sel(idx_rcv=0, x=source["x"], y=source["y"], method="nearest").plot(
        ax=axs[1]
    )
    axs[1].set_title("Noise at source position")
    # Signal + noise at source position
    ds_sig_gcc.s_l.sel(idx_rcv=0, x=source["x"], y=source["y"], method="nearest").plot(
        ax=axs[2]
    )
    axs[2].set_title("Library signal + noise at source position")

    fpath = os.path.join(
        ROOT_IMG,
        f"signal_noise_snr{snr_dB}dB.png",
    )
    plt.savefig(fpath)

    # List of potential reference receivers to test
    rcv_refs = range(len(receivers["x"]))  # General case

    # if all_rcv_ref:
    #     rcv_refs = range(len(receivers["x"]))  # General case
    #     rcv_ref_tag = "all_rcv_ref"
    # else:
    #     rcv_refs = [0]  # Quicker case for snr study
    #     rcv_ref_tag = "rcv_ref_0"

    for i_ref in rcv_refs:

        y_library_ref = ds_sig.s_l.sel(idx_rcv=i_ref)
        y_event_ref = ds_sig.s_e.sel(idx_rcv=i_ref)

        ## RTF ##
        nperseg = 2**11
        # nperseg = 256
        noverlap = nperseg // 2

        rtf_cs_l, rtf_cs_e = estimate_rtf(
            ds_sig=ds_sig,
            noise_da=noise_da,
            i_ref=i_ref,
            source=source,
            library_props=library_props,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        rtf_library.append(rtf_cs_l)
        rtf_event.append(rtf_cs_e)

        ### DCF GCC ###
        # Power spectral density at each grid pixel associated to the reference receiver -> library
        fxx, Sxx_library_ref = sp.welch(
            y_library_ref,
            fs=library_props["fs"],
            nperseg=nperseg,
            noverlap=noverlap,
            axis=0,
        )

        # Power spectral density at the source position associated to the reference receiver -> event
        _, Sxx_event_ref = sp.welch(
            y_event_ref,
            fs=library_props["fs"],
            nperseg=nperseg,
            noverlap=noverlap,
            axis=0,
        )

        # Restict to the frequency band of interest
        idx_band = (fxx >= library_props["f0"]) & (fxx <= library_props["f1"])
        fxx = fxx[idx_band]
        Sxx_library_ref = Sxx_library_ref[idx_band]
        Sxx_event_ref = Sxx_event_ref[idx_band]

        for i_rcv in range(len(receivers["x"])):

            y_library_i = ds_sig_gcc.s_l.sel(idx_rcv=i_rcv)
            y_event_i = ds_sig_gcc.s_e.sel(idx_rcv=i_rcv)

            ## GCC SCOT ##

            ## library ##
            # Power spectral density at each grid point associated to the receiver i
            _, Syy_library_i = sp.welch(
                y_library_i,
                fs=library_props["fs"],
                nperseg=nperseg,
                noverlap=noverlap,
                axis=0,
            )
            Syy_library_i = Syy_library_i[idx_band]

            # Cross power spectral density between the reference receiver and receiver i
            _, Sxy_library = sp.csd(
                y_library_ref,
                y_library_i,
                fs=library_props["fs"],
                nperseg=nperseg,
                noverlap=noverlap,
                axis=0,
            )
            Sxy_library = Sxy_library[idx_band]

            # Compute weights for GCC-SCOT
            w = 1 / np.abs(np.sqrt(Sxx_library_ref * Syy_library_i))
            # Apply GCC-SCOT
            gcc_library_i = w * Sxy_library
            gcc_library_i = gcc_library_i.reshape(
                (fxx.size, ds_sig_gcc.sizes["x"], ds_sig_gcc.sizes["y"])
            )
            gcc_library.append(gcc_library_i)

            ## Event ##
            # Power spectral density at the source position associated to the receiver i
            _, Syy_event_i = sp.welch(
                y_event_i,
                fs=library_props["fs"],
                nperseg=nperseg,
                noverlap=noverlap,
            )
            Syy_event_i = Syy_event_i[idx_band]

            # Cross power spectral density between reference receiver and receiver i at source position$
            _, Sxy_event = sp.csd(
                y_event_ref,
                y_event_i,
                fs=library_props["fs"],
                nperseg=nperseg,
                noverlap=noverlap,
                axis=0,
            )
            Sxy_event = Sxy_event[idx_band]

            # Compute weights for GCC-SCOT
            w_src = 1 / np.abs(np.sqrt(Sxx_event_ref * Syy_event_i))
            # Apply GCC-SCOT
            gcc_event_i = w_src * Sxy_event
            gcc_event.append(gcc_event_i)

    # Create dataset to store full simulation result
    ds_res_from_sig = ds_sig.copy()

    # Add coords
    ds_res_from_sig.coords["idx_rcv_ref"] = range(len(receivers["x"]))
    ds_res_from_sig.coords["f"] = fxx

    shape_event = (
        ds_res_from_sig.sizes["idx_rcv_ref"],
        ds_res_from_sig.sizes["idx_rcv"],
        ds_res_from_sig.sizes["f"],
    )
    shape_library = (
        ds_res_from_sig.sizes["idx_rcv_ref"],
        ds_res_from_sig.sizes["idx_rcv"],
        ds_res_from_sig.sizes["f"],
        ds_res_from_sig.sizes["x"],
        ds_res_from_sig.sizes["y"],
    )

    # GCC SCOT (idx_rcv_ref, f, x, y, idx_rcv)
    gcc_event = np.array(gcc_event).reshape(shape_event)  # (idx_rcv_ref, f, idx_rcv)
    gcc_event = np.moveaxis(gcc_event, 1, -1)
    gcc_library = np.array(gcc_library).reshape(shape_library)
    gcc_library = np.moveaxis(gcc_library, 1, -1)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # Add gcc-scot to dataset
    ds_res_from_sig["gcc_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_library.real,
    )
    ds_res_from_sig["gcc_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_library.imag,
    )
    ds_res_from_sig["gcc_event_real"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        gcc_event.real,
    )
    ds_res_from_sig["gcc_event_imag"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        gcc_event.imag,
    )

    # RTF
    rtf_event = np.array(rtf_event)
    rtf_library = np.array(rtf_library)

    # Add rft to dataset
    ds_res_from_sig["rtf_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_library.real,
    )
    ds_res_from_sig["rtf_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_library.imag,
    )
    ds_res_from_sig["rtf_event_real"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        rtf_event.real,
    )
    ds_res_from_sig["rtf_event_imag"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        rtf_event.imag,
    )

    # Subsample frequency to save memory
    subsample_idx = np.arange(0, ds_res_from_sig.sizes["f"])[::5]
    ds_res_from_sig = ds_res_from_sig.isel(f=subsample_idx)

    # Save updated dataset
    fpath = os.path.join(
        ROOT_DATA,
        f"zhang_output_from_signal_dx{grid['dx']}m_dy{grid['dy']}m_snr{snr_dB}dB.nc",
    )
    ds_res_from_sig.to_netcdf(fpath)
    ds_res_from_sig.close()

    print(f"Features derived from time signal in {time() - t_start:.2f} s")


def build_features_fullsimu():
    """Step 4.1 : build localisation features for DCF GCC and RTF methods.
    Full simulation approach : DCF and RTF are build directly from transfer functions"""

    # Load params
    depth, receivers, source, grid, frequency, _ = params()
    dx = grid["dx"]
    dy = grid["dy"]

    # Dataset with gridded tf (for full simulation)
    fname = f"tf_zhang_grid_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_tf = xr.open_dataset(fpath)

    # Limit max frequency to speed up
    fs = 1200
    fmax = fs / 2
    ds_tf = ds_tf.sel(f=slice(0, fmax))

    # Load library spectrum
    # f = ds_tf.f.values
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum(fs=fs)

    # Load event spectrum
    _, S_f_event, _ = event_src_spectrum(
        # target_var=library_props["sig_var"],
        T=library_props["T"],
        fs=library_props["fs"],
    )

    # Restrict ds_tf, S_flibrary and S_f_event to the signal band [100, 500]
    idx_band = (f_library >= library_props["f0"]) & (f_library <= library_props["f1"])
    ds_tf = ds_tf.sel(f=slice(library_props["f0"], library_props["f1"]))
    S_f_library = S_f_library[idx_band]
    S_f_event = S_f_event[idx_band]

    # Subsample frequency to save memory
    subsample_idx = np.arange(0, ds_tf.sizes["f"])[::5]
    S_f_library = S_f_library[subsample_idx]
    S_f_event = S_f_event[subsample_idx]
    ds_tf = ds_tf.isel(f=subsample_idx)

    ### 1) Full simulation approach : rtf and gcc are derived directly from tfs ###

    rtf_event = []  # RFT vector at the source position
    rtf_library = []  # RTF vector evaluated at each grid pixel
    gcc_event = []  # GCC vector evaluated at the source position
    gcc_library = []  # GCC-SCOT vector evaluated at each grid pixel

    for i_ref in range(len(receivers["x"])):

        tf_ref = ds_tf.tf_real.sel(idx_rcv=i_ref) + 1j * ds_tf.tf_imag.sel(
            idx_rcv=i_ref
        )
        tf_src_ref = tf_ref.sel(x=source["x"], y=source["y"], method="nearest")

        # Received spectrum -> reference receiver
        y_ref = mult_along_axis(tf_ref.values, S_f_library, axis=0)
        y_ref_src = mult_along_axis(tf_src_ref.values, S_f_event, axis=0)

        # Power spectral density at each grid pixel associated to the reference receiver -> library
        Sxx_library_ref = y_ref * np.conj(y_ref)
        # Power spectral density at the source position associated to the reference receiver -> event
        Sxx_event_ref = y_ref_src * np.conj(y_ref_src)

        for i_rcv in range(len(receivers["x"])):

            ## Kraken RTF ##
            tf_i = ds_tf.tf_real.sel(idx_rcv=i_rcv) + 1j * ds_tf.tf_imag.sel(
                idx_rcv=i_rcv
            )
            rtf_i = tf_i.values / tf_ref.values
            rtf_i = rtf_i.reshape(
                (ds_tf.sizes["f"], ds_tf.sizes["x"], ds_tf.sizes["y"])
            )
            rtf_library.append(rtf_i)

            # Source
            tf_src_i = tf_i.sel(x=source["x"], y=source["y"], method="nearest")
            rtf_event_i = tf_src_i.values / tf_src_ref.values
            rtf_event.append(rtf_event_i)

            ## GCC SCOT ##

            ## Grid -> library ##
            # Add the signal spectrum information
            y_i = mult_along_axis(tf_i.values, S_f_library, axis=0)

            # Power spectral density at each grid point associated to the receiver i
            Syy = y_i * np.conj(y_i)

            # Cross power spectral density between the reference receiver and receiver i
            Sxy = y_ref * np.conj(y_i)

            # Compute weights for GCC-SCOT
            w = 1 / np.abs(np.sqrt(Sxx_library_ref * Syy))
            # Apply GCC-SCOT
            gcc_library_i = w * Sxy
            gcc_library_i = gcc_library_i.reshape(
                (ds_tf.sizes["f"], ds_tf.sizes["x"], ds_tf.sizes["y"])
            )
            gcc_library.append(gcc_library_i)

            ## Event source -> event ##
            y_src_i = mult_along_axis(tf_src_i.values, S_f_event, axis=0)

            # Power spectral density at the source position associated to the receiver i
            Syy_src = y_src_i * np.conj(y_src_i)

            # Cross power spectral density between reference receiver and receiver i at source position$
            Sxy_src = y_ref_src * np.conj(y_src_i)

            # Compute weights for GCC-SCOT
            w_src = 1 / np.abs(np.sqrt(Sxx_event_ref * Syy_src))
            # Apply GCC-SCOT
            gcc_event_i = w_src * Sxy_src
            gcc_event.append(gcc_event_i)

    # Create dataset to store full simulation result
    ds_res_full_simu = ds_tf.copy()

    # Add coords
    ds_res_full_simu.coords["idx_rcv_ref"] = range(len(receivers["x"]))

    shape_event = (
        ds_res_full_simu.sizes["idx_rcv_ref"],
        ds_res_full_simu.sizes["idx_rcv"],
        ds_res_full_simu.sizes["f"],
    )
    shape_library = (
        ds_res_full_simu.sizes["idx_rcv_ref"],
        ds_res_full_simu.sizes["idx_rcv"],
        ds_res_full_simu.sizes["f"],
        ds_res_full_simu.sizes["x"],
        ds_res_full_simu.sizes["y"],
    )

    # RTF
    rtf_event = np.array(rtf_event).reshape(shape_event)
    rtf_event = np.moveaxis(rtf_event, 1, -1)  # (idx_rcv_ref, f, idx_rcv)
    rtf_library = np.array(rtf_library).reshape(shape_library)
    rtf_library = np.moveaxis(rtf_library, 1, -1)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # GCC SCOT (idx_rcv_ref, f, x, y, idx_rcv)
    gcc_event = np.array(gcc_event).reshape(shape_event)  # (idx_rcv_ref, f, idx_rcv)
    gcc_event = np.moveaxis(gcc_event, 1, -1)
    gcc_library = np.array(gcc_library).reshape(shape_library)
    gcc_library = np.moveaxis(gcc_library, 1, -1)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # Add rft to dataset
    ds_res_full_simu["rtf_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_library.real,
    )
    ds_res_full_simu["rtf_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        rtf_library.imag,
    )
    ds_res_full_simu["rtf_event_real"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        rtf_event.real,
    )
    ds_res_full_simu["rtf_event_imag"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        rtf_event.imag,
    )

    # Add gcc-scot to dataset
    ds_res_full_simu["gcc_real"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_library.real,
    )
    ds_res_full_simu["gcc_imag"] = (
        ["idx_rcv_ref", "f", "x", "y", "idx_rcv"],
        gcc_library.imag,
    )
    ds_res_full_simu["gcc_event_real"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        gcc_event.real,
    )
    ds_res_full_simu["gcc_event_imag"] = (
        ["idx_rcv_ref", "f", "idx_rcv"],
        gcc_event.imag,
    )

    # Save updated dataset
    fpath = os.path.join(
        ROOT_DATA, f"zhang_output_fullsimu_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    )
    ds_res_full_simu.to_netcdf(fpath)
    ds_res_full_simu.close()

    # ## Covariance substraction RTF ##
    # # Grid
    # r_i = grid["r"][i_rcv].flatten()
    # tf_i = ds.tf_real.sel(r=r_i, method="nearest") + 1j * ds.tf_imag.sel(
    #     r=r_i, method="nearest"
    # )
    # rtf_i = tf_i.values / tf_ref.values
    # rtf_i = rtf_i.reshape((ds.sizes["f"], ds.sizes["x"], ds.sizes["y"]))
    # rtf_library.append(rtf_i)

    # # Source
    # r_src_i = r_src_ref = source["r"][i_rcv]
    # tf_src_i = ds.tf_real.sel(
    #     r=r_src_i, method="nearest"
    # ) + 1j * ds.tf_imag.sel(r=r_src_i, method="nearest")
    # rtf_event_i = tf_src_i.values / tf_src_ref.values
    # rtf_event.append(rtf_event_i)


# def build_features_snr(snrs):
#     for snr in snrs:
#         print(f"Start building loc features for snr = {snr}dB ...")
#         t0 = time()
#         build_features_from_time_signal(snr)
#         elasped_time = time() - t0
#         print(f"Features built (elapsed time = {np.round(elasped_time,0)}s)")


if __name__ == "__main__":
    ## Step 1
    # build_tf_dataset()
    ## Step 2
    # grid_dataset()
    ## Step 3
    # build_signal()
    ## Step 4
    # build_features_fullsimu()
    build_features_from_time_signal(snr_dB=-40)
