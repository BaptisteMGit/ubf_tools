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
import gc
import dask
import numpy as np
import scipy.signal as sp
import xarray as xr
import dask.array as da

from cst import C0
from time import time
from multiprocessing import Pool
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from misc import cast_matrix_to_target_shape, mult_along_axis
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_TMP,
    ROOT_IMG,
    ROOT_DATA,
    BLOCK_SIZES,
    N_WORKERS,
    MAX_RAM_PER_WORKER_GB,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    library_src_spectrum,
    event_src_spectrum,
    params,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    check_signal_noise,
    check_signal_noise_stft,
    check_gcc_features,
    check_rtf_features,
)
from propa.kraken_toolbox.run_kraken import readshd, run_kraken_exec, run_field_exec


from source.rtf_estimator import RTFEstimator
from source.cov_manager import CovManager


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

    fname = "testcase_zhang2023"
    # working_dir = os.path.join(ROOT, "tmp")
    working_dir = ROOT_TMP
    env_file = os.path.join(working_dir, f"{fname}.env")

    # Read env file
    with open(env_file, "r") as file:
        lines = file.readlines()

    first_iter = True
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

        if first_iter:
            nf = len(f)
            nr = tf_subband.shape[1]
            h_grid = np.zeros((nf, nr), dtype=complex)
            first_iter = False
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


def grid_dataset(debug=False, antenna_type="zhang"):
    """Step 2 : Associate each grid pixel to the corresponding broadband transfert function caracterized by the range to the receiver.

    -   Kraken tf : H(f, r)
    -   Grid : r(x, y)
    -   Gridded tf : H(f, x, y) = H(f, r(x, y))
    """

    # Load dataset
    fpath = os.path.join(ROOT_DATA, "tf_zhang_dataset.nc")
    ds = xr.open_dataset(fpath)

    # Load param
    depth, receivers, source, grid, frequency, _ = params(
        debug=debug, antenna_type=antenna_type
    )

    # Create new dataset
    ds_grid = xr.Dataset(
        coords=dict(
            f=ds.f.values,
            y=grid["y"][:, 0],
            x=grid["x"][0, :],
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

    # tf_grid_ds = xr.Dataset(
    #     coords=dict(
    #         f=ds.f.values,
    #         r=r_grid_all_rcv_unique,
    #     ),
    #     data_vars=dict(tf=(["f", "r"], tf_vect.values)),
    # )  # (nf, n_ranges)

    gridded_tf = []
    # grid_shape = (ds_grid.sizes["f"], ds_grid.sizes["x"], ds_grid.sizes["y"])
    grid_shape = (ds_grid.sizes["f"],) + grid["r"].shape[
        1:
    ]  # Try to fix search grid issues 11/02/202    (nf, ny, nx)
    for i_rcv in range(len(receivers["x"])):
        r_grid = grid["r"][i_rcv].flatten()
        # tf_ircv = tf_grid_ds.tf.sel(r=r_grid, method="nearest")
        tf_ircv = tf_vect.sel(r=r_grid, method="nearest")

        tf_grid = tf_ircv.values.reshape(grid_shape)  # (nf, ny, nx)
        gridded_tf.append(tf_grid)

    gridded_tf = np.array(gridded_tf)  # (nr, nf, ny, nx)
    # Add to dataset
    grid_coords = ["idx_rcv", "f", "y", "x"]  # Fix 11/02/2025
    ds_grid["tf_real"] = (grid_coords, np.real(gridded_tf))
    ds_grid["tf_imag"] = (grid_coords, np.imag(gridded_tf))

    # Save dataset
    fname = f"tf_zhang_grid_dx{ds_grid.dx}m_dy{ds_grid.dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_grid.to_netcdf(fpath)
    ds_grid.close()


def build_signal(debug=False, antenna_type="zhang", event_stype="wn"):
    """Step 3 : derive signal received from each grid pixel using library source spectrum and gridded transfert functions.

    -   Gridded tf : H(x, y, f)
    -   Source spectrum : S(f)
    -   Gridded spectrum : Y(x, y, f) = S(f) H(x, y, f)
    -   Gridded signal : y(x, y, t) = FFT_inv(Y(x, y, f))

    """

    # Load params
    depth, receivers, source, grid, frequency, _ = params(
        debug=debug, antenna_type=antenna_type
    )

    # Load gridded dataset
    fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_gridded_tf = xr.open_dataset(fpath)  # (nf, ny, nx, nrcv)

    # Limit max frequency to speed up
    fs_target = 1200
    fmax = fs_target / 2
    ds_gridded_tf = ds_gridded_tf.sel(f=slice(0, fmax))

    # Load library spectrum
    # f = ds_gridded_tf.f.values
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum(
        fs=fs_target
    )

    # Load event spectrum
    _, S_f_event, _ = event_src_spectrum(
        T=library_props["T"],
        fs=library_props["fs"],
        stype=event_stype,
    )

    # Derive delay for each receiver
    cmin = 1488  # Min speed in SwellEx96 mean profile
    delay_rcv = grid["r"] / cmin  # (nrcv, ny, nx)

    # Same delay is applied to each receiver : the receiver with the minimum delay is taken as the time reference
    # (we are only interested in relative time difference)
    # tau = ds_gridded_tf.delay_rcv.min(dim="idx_rcv")
    f = ds_gridded_tf.f
    tau = np.min(delay_rcv, axis=0)  # (ny, nx)
    da_tau = xr.DataArray(data=tau, coords=dict(y=ds_gridded_tf.y, x=ds_gridded_tf.x))
    tau_event = da_tau.sel(y=source["y"], x=source["x"], method="nearest")
    delay_event = np.exp(1j * 2 * np.pi * tau_event * f)

    # Cast tau gridded tf shape
    tau_lib = cast_matrix_to_target_shape(
        tau, ds_gridded_tf.tf_real.shape[1:]
    )  # (nf, ny, nx)

    y_t_event = []
    y_t_library = []
    for i_rcv in range(len(receivers["x"])):

        tf_library = ds_gridded_tf.tf_real.sel(
            idx_rcv=i_rcv
        ) + 1j * ds_gridded_tf.tf_imag.sel(
            idx_rcv=i_rcv
        )  # (nf, ny, nx)
        tf_event = tf_library.sel(
            y=source["y"], x=source["x"], method="nearest"
        )  # (nf,)

        # Derive received spectrum (Y = SH)
        k0 = 2 * np.pi * f / C0
        norm_factor = np.exp(1j * k0) / (4 * np.pi)

        y_f_library = mult_along_axis(tf_library, S_f_library * norm_factor, axis=0)
        y_f_event = tf_event * S_f_event * norm_factor

        # Derive delay factor to take into account the propagation time
        tau_vec = mult_along_axis(tau_lib, f, axis=0)
        delay_library = np.exp(1j * 2 * np.pi * tau_vec)

        # Apply delay
        y_f_library *= delay_library  # (nf, ny, nx)
        y_f_event *= delay_event  # (nf,)

        # FFT inv to get signal
        y_t_l = np.fft.irfft(y_f_library, axis=0)  # (nt, ny, nx)
        y_t_e = np.fft.irfft(y_f_event)  # (nt,)

        # Store for current receiver
        y_t_library.append(y_t_l)
        y_t_event.append(y_t_e)

    y_t_library = np.array(y_t_library)  # (nrcv, nt, ny, nx)
    y_t_event = np.array(y_t_event)  # (nrcv, nt)

    # Build dataset to save
    t = np.arange(0, library_props["T"], 1 / library_props["fs"])

    ds_sig = xr.Dataset(
        coords=dict(
            idx_rcv=ds_gridded_tf.idx_rcv,
            t=t,
            y=ds_gridded_tf.y,
            x=ds_gridded_tf.x,
        ),
        data_vars=dict(
            s_l=(["idx_rcv", "t", "y", "x"], y_t_library),
            s_e=(["idx_rcv", "t"], y_t_event),
        ),
        attrs=ds_gridded_tf.attrs,
    )

    # Save dataset
    fname = f"zhang_library_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_sig.to_netcdf(fpath)


# @dask.delayed
def derive_received_noise(
    s_library,
    s_event,
    event_source,
    snr_dB=10,
    noise_model="gaussian",
    verbose=False,
):
    """
    Function to derive noise signals according to target SNR.

    Event signal and library signal at event source position do not have the exact same power due to the different nature of the source signal
    (even if the source signal are both normalized to unit variance). To account for that we need to use different noise power for library and
    event signals to ensure both reach the target SNR.

    """

    ## Library
    s_library_src_pos_rcv0 = s_library.sel(idx_rcv=0).sel(
        x=event_source["x"], y=event_source["y"], method="nearest"
    )
    # Library signal power at receiver n°0 and source position used as reference
    sigma_rcv_ref_library = np.std(s_library_src_pos_rcv0.values)
    # Normalize to account for the reference signal power to reach required snr at receiver n°0
    sigma_v_library = sigma_rcv_ref_library * np.sqrt(10 ** (-snr_dB / 10))
    # We assume that the noise is due to ambiant noise (hence it does not depend on the source position within the search grid) and is the same at each receiver position (receiver electronic noise )
    noise_library = np.random.normal(loc=0, scale=sigma_v_library, size=s_library.shape)

    ## Event ##
    s_event_rcv0 = s_event.sel(idx_rcv=0)
    # Event signal power at receiver n°0
    sigma_rcv_ref_event = np.std(s_event_rcv0.values)
    # Normalize to account for the reference signal power to reach required snr at receiver n°0
    sigma_v_event = sigma_rcv_ref_event * np.sqrt(10 ** (-snr_dB / 10))
    # We assume that the noise is due to ambiant noise (hence it does not depend on the source position within the search grid)
    # and is the same at each receiver position (receiver electronic noise negligible)
    noise_event = np.random.normal(loc=0, scale=sigma_v_event, size=s_event.shape)

    # Create dataset to store noise signals
    ds_noise = xr.Dataset(
        data_vars=dict(
            n_l=(["idx_rcv", "t", "y", "x"], noise_library),
            n_e=(["idx_rcv", "t"], noise_event),
        ),
        coords=dict(
            t=s_library.t,
            x=s_library.x,
            y=s_library.y,
            idx_rcv=s_library.idx_rcv,
        ),
        attrs=dict(
            std_ref_event=sigma_rcv_ref_event,
            std_ref_library=sigma_rcv_ref_library,
            snr=snr_dB,
        ),
    )

    if verbose:
        # Check SNR
        snr_rcv0_event = 10 * np.log10(
            np.var(s_event_rcv0.values) / np.var(ds_noise.n_e.sel(idx_rcv=0).values)
        )
        snr_rcv0_library = 10 * np.log10(
            np.var(s_library_src_pos_rcv0.values)
            / np.var(ds_noise.n_l.sel(idx_rcv=0).values)
        )
        print(
            f"SNR event signal at receiver n°0 : {np.round(snr_rcv0_event, 2)} dB (required {snr_dB}dB)"
        )
        print(
            f"SNR library signal at receiver n°0 : {np.round(snr_rcv0_library, 2)} dB (required {snr_dB}dB)"
        )

    return ds_noise


def dask_rtf_4D(x_4D, v_4D, fs, idx_rcv_refs, nperseg, noverlap, window="hann"):
    """Wrapper function for rtf_4D with dask."""

    # x_4D is a (nrcv, nt, ny, nx) array
    dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
    re = RTFEstimator()

    _, rtf = re.covariance_subtraction_major_eigen_vector_4D(
        x_4D,
        v_4D,
        dims_order,
        fs,
        idx_rcv_refs,
        nperseg,
        noverlap,
        window,
        return_csdm=False,
    )
    return rtf


def dask_gcc_4D(x_4D, fs, idx_rcv_refs, nperseg, noverlap, window="hann"):
    """Wrapper function for gcc_4D with dask."""
    ff, gcc = gcc_4D(x_4D, fs, idx_rcv_refs, nperseg, noverlap, window)
    return gcc


def gcc_4D(x_4D, fs, idx_rcv_refs, nperseg, noverlap, window="hann"):
    # Power spectral density of signals
    ff, sxx = sp.welch(
        x_4D,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        axis=1,
    )  # (nrcv, nf, ny, nx)

    iref = 0

    gcc = np.empty((sxx.shape[0],) + sxx.shape, dtype=complex)
    for iref in idx_rcv_refs:
        # Cross power spectral density of signals between the reference receiver and other receivers
        x_4D_ref = x_4D[
            iref : iref + 1, ...
        ]  # Keep first dimension to allow broadcasting
        ff, Sxy_library = sp.csd(
            x_4D_ref,
            x_4D,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            axis=1,
        )  # (nrcv, nf, ny, nx)

        # Compute GCC-SCOT weights
        sxx_ref = sxx[
            iref : iref + 1, ...
        ]  # Keep first dimension to allow broadcasting
        w = 1 / np.abs(np.sqrt(sxx_ref * sxx))  # (nrcv, nf, ny, nx)

        # Apply GCC-SCOT
        gcc[iref, ...] = w * Sxy_library

    return ff, gcc


def compute_chunks(shape, n_workers):
    """
    Compute chunk sizes for axes 2 and 3 to distribute workload efficiently among workers.

    Parameters:
    - shape: Tuple, full shape of the array (dim0, dim1, dim2, dim3, dim4)
    - n_workers: Int, number of Dask workers available

    Returns:
    - Tuple of chunk sizes for each dimension
    """
    # Extract shape dimensions
    dim0, dim1, dim2, dim3 = shape

    # Compute approximate number of chunks along axes 2 and 3
    total_elements_2_3 = dim2 * dim3

    # Ideal number of chunks (as close to n_workers as possible)
    target_chunks = min(n_workers, total_elements_2_3)

    # Split axis 2 and 3 proportionally
    aspect_ratio = dim2 / dim3
    n_chunks_2 = int(np.sqrt(target_chunks * aspect_ratio))
    n_chunks_3 = max(1, target_chunks // n_chunks_2)  # Ensure at least 1 chunk

    # Compute chunk sizes
    chunk_size_2 = np.ceil(dim2 / n_chunks_2)
    chunk_size_3 = np.ceil(dim3 / n_chunks_3)

    # Ensure chunk sizes are valid (don't exceed original dimensions)
    chunk_size_2 = min(chunk_size_2, dim2).astype(int)
    chunk_size_3 = min(chunk_size_3, dim3).astype(int)

    # Keep other dimensions unchanged
    chunk_sizes = (dim0, dim1, chunk_size_2, chunk_size_3)

    return chunk_sizes


def build_features_from_time_signal(
    snr_dB=0,
    debug=False,
    check=False,
    use_welch_estimator=False,
    antenna_type="zhang",
    rtf_estimator="cs",
    root_name="zhang_output_from_signal",
    verbose=False,
):
    """
    Step 4.2 : build localisation features
        -> GCC for the DCF method
        -> RTF for the RTF-MFP method

    """

    # if debug:
    #     verbose = True

    t_start = time()

    # Load params
    _, receivers, source, grid, _, _ = params(debug=debug, antenna_type=antenna_type)
    dx = grid["dx"]
    dy = grid["dy"]

    # Dataset with time signal (for realistic approach)
    fname = f"zhang_library_dx{dx}m_dy{dy}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_sig = xr.open_dataset(fpath)
    attrs = ds_sig.attrs

    # Load library spectrum
    fs = 1200
    library_props, _, _, _ = library_src_spectrum(fs=fs)

    # Derive noise dataset
    ds_noise = derive_received_noise(
        s_library=ds_sig.s_l,
        s_event=ds_sig.s_e,
        event_source=source,
        snr_dB=snr_dB,
        verbose=verbose,
    )  # (nr, nt, ny, nx)
    # NOTE : different noise dataset to simulate the fact that in real life the noise CSDM is
    # estimated from a different segment of the signal than the signal + noise CSDM (assuming noise is stationnary)
    ds_noise_bis = derive_received_noise(
        s_library=ds_sig.s_l,
        s_event=ds_sig.s_e,
        event_source=source,
        snr_dB=snr_dB,
        verbose=verbose,
    )  # (nr, nt, ny, nx)

    # Noisy signals
    noisy_signal_library = ds_noise.n_l + ds_sig.s_l  # (nrcv, nt, ny, nx)
    noisy_signal_event = ds_noise.n_e + ds_sig.s_e  # (nrcv, nt)

    # We don't need all datasets anymore
    # ds_noise.close()
    # ds_sig.close()

    # Plot signal and noise at source position -> library
    if check:
        ds_sig_noise = xr.Dataset(
            data_vars=dict(
                x_l=(["idx_rcv", "t", "y", "x"], noisy_signal_library.values),
                n_l=(["idx_rcv", "t", "y", "x"], ds_noise.n_l.values),
                s_l=(["idx_rcv", "t", "y", "x"], ds_sig.s_l.values),
                x_e=(["idx_rcv", "t"], noisy_signal_event.values),
                n_e=(["idx_rcv", "t"], ds_noise.n_e.values),
                s_e=(["idx_rcv", "t"], ds_sig.s_e.values),
            ),
            coords=dict(
                t=ds_sig.t,
                x=ds_sig.x,
                y=ds_sig.y,
                idx_rcv=ds_sig.idx_rcv,
            ),
            attrs=dict(
                std_ref_event=ds_noise.std_ref_event,
                std_ref_library=ds_noise.std_ref_library,
                snr=snr_dB,
                xs=source["x"],
                ys=source["y"],
                dx=dx,
                dy=dy,
                root_img=os.path.join(
                    ROOT_IMG, f"from_signal_dx{dx}m_dy{dy}m", f"snr_{snr_dB:.1f}dB"
                ),
            ),
        )

        check_signal_noise(ds_sig_noise)
        check_signal_noise_stft(ds_sig_noise)

    # List of potential reference receivers to test
    idx_rcv_refs = range(
        len(receivers["x"])
    )  # General case -> all receivers are used as reference to build the ambiguity surface for all couples in array (required for DCF method)

    nperseg = 2**11
    noverlap = nperseg // 2

    ds_feature_estimate = xr.Dataset(
        data_vars=dict(
            x_l=(["idx_rcv", "t", "y", "x"], noisy_signal_library.data),
            n_l_bis=(["idx_rcv", "t", "y", "x"], ds_noise_bis.n_l.data),
            x_e=(["idx_rcv", "t"], noisy_signal_event.data),
            n_e_bis=(["idx_rcv", "t"], ds_noise_bis.n_e.data),
        ),
        coords=dict(
            t=ds_sig.t,
            x=ds_sig.x,
            y=ds_sig.y,
            idx_rcv=ds_sig.idx_rcv,
        ),
    )

    # ====================================================================================================
    # Optimized version: RTF and GCC are computed directly on the whole 4D array (nrcv, nt, ny, nx)
    # ====================================================================================================
    use_dask = True
    window = "hann"
    fs = library_props["fs"]

    xl_4D = ds_feature_estimate.x_l.data
    vl_4D = ds_feature_estimate.n_l_bis.data

    if use_dask:

        ### RTF ###
        # Define optimal chunk size to fit with the number of workers
        # shape = xl_4D.shape
        # optimal_chunks = compute_chunks(shape, N_WORKERS)
        optimal_chunks = (
            -1,
            -1,
            4,
            4,
        )  # Seems to be one one the fastest configuration with 20 workers

        # Convert to dask arrays
        xl_4D = da.from_array(xl_4D, chunks=optimal_chunks)
        xl_4D = xl_4D.persist()
        vl_4D = da.from_array(vl_4D, chunks=optimal_chunks)
        vl_4D = vl_4D.persist()

        # print("Total number of chunks : ", xl_4D.numblocks[-2]*xl_4D.numblocks[-1])
        # t0 = time()
        output_chunks = (xl_4D.shape[0],) + xl_4D.shape
        rtf_dask = da.map_blocks(
            dask_rtf_4D,
            xl_4D,
            vl_4D,
            dtype=complex,
            chunks=output_chunks,
            fs=fs,
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        rtf_library = rtf_dask.compute()

        # We will use the same 4D function for the event signal so we need to add dimensions (single positions x, y)
        xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
        ve_4D = ds_feature_estimate.n_e_bis.values[..., np.newaxis, np.newaxis]

        re = RTFEstimator()
        dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
        ff, rtf_event = re.covariance_subtraction_major_eigen_vector_4D(
            xe_4D,
            ve_4D,
            dims_order,
            fs,
            idx_rcv_refs,
            nperseg,
            noverlap,
            window,
            return_csdm=False,
        )

        # Squeeze to remove dummy dimensions
        rtf_event = np.squeeze(rtf_event)

        # Restict to the frequency band of interest
        idx_band = (ff >= library_props["f0"]) & (ff <= library_props["f1"])
        rtf_library = rtf_library[:, :, idx_band, ...]
        rtf_event = rtf_event[..., idx_band]
        f_rtf = ff[idx_band]

        ### DCF ###
        gcc_dask = da.map_blocks(
            dask_gcc_4D,
            xl_4D,
            dtype=complex,
            chunks=output_chunks,
            fs=fs,
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        gcc_library = gcc_dask.compute()

        xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
        ff, gcc_event = gcc_4D(
            x_4D=xe_4D,
            fs=fs,
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        gcc_event = np.squeeze(gcc_event)

        # Restict to the frequency band of interest
        idx_band = (ff >= library_props["f0"]) & (ff <= library_props["f1"])
        gcc_library = gcc_library[:, :, idx_band, ...]
        gcc_event = gcc_event[:, :, idx_band]
        f_gcc = ff[idx_band]

        # Clean up memory
        del xl_4D, vl_4D, xe_4D, ve_4D, gcc_dask, rtf_dask
        gc.collect()

    else:
        ### RTF ###
        # t0 = time()
        xl_4D = ds_feature_estimate.x_l.values
        vl_4D = ds_feature_estimate.n_l_bis.values

        ff, rtf_library = rtf_4D(
            x_4D=xl_4D,
            v_4D=vl_4D,
            fs=library_props["fs"],
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        # We will use the same 4D function for the event signal so we need to add dimensions (single positions x, y)
        xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
        ve_4D = ds_feature_estimate.n_e_bis.values[..., np.newaxis, np.newaxis]
        ff, rtf_event = rtf_4D(
            x_4D=xe_4D,
            v_4D=ve_4D,
            fs=library_props["fs"],
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        # Squeeze to remove dummy dimensions
        rtf_event = np.squeeze(rtf_event)

        # Restict to the frequency band of interest
        idx_band = (ff >= library_props["f0"]) & (ff <= library_props["f1"])
        rtf_library = rtf_library[:, :, idx_band, ...]
        rtf_event = rtf_event[..., idx_band]
        f_rtf = ff[idx_band]

        # print(f"RTFs computed in {time()-t0} s")
        # print("Dask RTF matches RTF", np.allclose(rtf_library, rtf_library_dask))

        ### GCC ###
        # t0 = time()
        xl_4D = ds_feature_estimate.x_l.values
        xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]

        ff, gcc_library = gcc_4D(
            x_4D=xl_4D,
            fs=library_props["fs"],
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        _, gcc_event = gcc_4D(
            x_4D=xe_4D,
            fs=library_props["fs"],
            idx_rcv_refs=idx_rcv_refs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        gcc_event = np.squeeze(gcc_event)

        # Restict to the frequency band of interest
        idx_band = (ff >= library_props["f0"]) & (ff <= library_props["f1"])
        gcc_library = gcc_library[:, :, idx_band, ...]
        gcc_event = gcc_event[:, :, idx_band]
        f_gcc = ff[idx_band]

        # print("GCC dask matches GCC ", np.allclose(gcc_library, gcc_library_dask))

    # print(f"GCCs computed in {time()-t0} s")

    # Create dataset to store results
    attrs.update(
        dict(
            # std_ref_event=ds_noise.std_ref_event,
            # std_ref_library=ds_noise.std_ref_library,
            snr=snr_dB,
            xs=source["x"],
            ys=source["y"],
            root_img=os.path.join(
                ROOT_IMG, f"from_signal_dx{dx}m_dy{dy}m", f"snr_{snr_dB:.1f}dB"
            ),
        )
    )

    ds_res_from_sig = xr.Dataset(
        data_vars=dict(
            rtf_event_real=(["idx_rcv_ref", "idx_rcv", "f_rtf"], rtf_event.real),
            rtf_event_imag=(["idx_rcv_ref", "idx_rcv", "f_rtf"], rtf_event.imag),
            gcc_event_real=(["idx_rcv_ref", "idx_rcv", "f_gcc"], gcc_event.real),
            gcc_event_imag=(["idx_rcv_ref", "idx_rcv", "f_gcc"], gcc_event.imag),
            rtf_real=(["idx_rcv_ref", "idx_rcv", "f_rtf", "y", "x"], rtf_library.real),
            rtf_imag=(["idx_rcv_ref", "idx_rcv", "f_rtf", "y", "x"], rtf_library.imag),
            gcc_real=(["idx_rcv_ref", "idx_rcv", "f_gcc", "y", "x"], gcc_library.real),
            gcc_imag=(["idx_rcv_ref", "idx_rcv", "f_gcc", "y", "x"], gcc_library.imag),
        ),
        coords=dict(
            x=ds_feature_estimate.x.values,
            y=ds_feature_estimate.y.values,
            idx_rcv=ds_feature_estimate.idx_rcv.values,
            idx_rcv_ref=ds_feature_estimate.idx_rcv.values,
            f_gcc=f_gcc,
            f_rtf=f_rtf,
        ),
        attrs=attrs,
    )

    # Subsample frequency to save memory
    # subsample_idx = np.arange(0, ds_res_from_sig.sizes["f"])[::5]
    # ds_res_from_sig = ds_res_from_sig.isel(f=subsample_idx)

    if check:
        check_rtf_features(ds_res_from_sig, folder=ds_sig_noise.attrs["root_img"])
        check_gcc_features(ds_res_from_sig, folder=ds_sig_noise.attrs["root_img"])

    # Save updated dataset
    fpath = os.path.join(
        ROOT_DATA,
        f"{root_name}_dx{grid['dx']}m_dy{grid['dy']}m_snr{snr_dB:.1f}dB.nc",
    )
    ds_res_from_sig.to_netcdf(fpath)
    ds_res_from_sig.close()

    print(f"Features derived from time signal in {time() - t_start:.2f} s")


def build_features_fullsimu(debug=False, antenna_type="zhang", event_stype="wn"):
    """
    Step 4.1 : build localisation features for DCF GCC and RTF methods.
    Full simulation approach : DCF and RTF are build directly from transfer functions"""

    # Load params
    depth, receivers, source, grid, frequency, _ = params(
        debug=debug, antenna_type=antenna_type
    )
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
    library_props, S_f_library, f_library, idx_in_band = library_src_spectrum(fs=fs)

    # Load event spectrum
    _, S_f_event, _ = event_src_spectrum(
        T=library_props["T"],
        fs=library_props["fs"],
        stype=event_stype,
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

    # Init lists to save results
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

    # Read arrays sizes
    nf = ds_tf.sizes["f"]
    nx = ds_tf.sizes["x"]
    ny = ds_tf.sizes["y"]
    n_rcv_ref = ds_tf.sizes["idx_rcv"]
    n_rcv = ds_tf.sizes["idx_rcv"]

    # Set target shapes
    shape_event = (n_rcv_ref, n_rcv, nf)
    shape_library = (n_rcv_ref, n_rcv, nf, ny, nx)

    # RTF
    rtf_event = np.array(rtf_event).reshape(shape_event)
    rtf_event = np.moveaxis(rtf_event, 1, -1)  # (idx_rcv_ref, f, idx_rcv)
    rtf_library = np.array(rtf_library).reshape(shape_library)
    rtf_library = np.moveaxis(rtf_library, 1, -1)  # (idx_rcv_ref, f, y, x, idx_rcv)
    # Reshape to order x, y
    rtf_library = np.moveaxis(rtf_library, 2, 3)  # (idx_rcv_ref, f, x, y, idx_rcv)

    # GCC SCOT (idx_rcv_ref, f, x, y, idx_rcv)
    gcc_event = np.array(gcc_event).reshape(shape_event)  # (idx_rcv_ref, f, idx_rcv)
    gcc_event = np.moveaxis(gcc_event, 1, -1)
    gcc_library = np.array(gcc_library).reshape(shape_library)
    gcc_library = np.moveaxis(gcc_library, 1, -1)  # (idx_rcv_ref, f, y, x, idx_rcv)
    # Reshape to order x, y
    gcc_library = np.moveaxis(gcc_library, 2, 3)  # (idx_rcv_ref, f, y, x, idx_rcv)

    # Create dataset to store results
    ds_res_full_simu = xr.Dataset(
        data_vars=dict(
            rtf_event_real=(["idx_rcv_ref", "f_rtf", "idx_rcv"], rtf_event.real),
            rtf_event_imag=(["idx_rcv_ref", "f_rtf", "idx_rcv"], rtf_event.imag),
            gcc_event_real=(["idx_rcv_ref", "f_gcc", "idx_rcv"], gcc_event.real),
            gcc_event_imag=(["idx_rcv_ref", "f_gcc", "idx_rcv"], gcc_event.imag),
            rtf_real=(["idx_rcv_ref", "f_rtf", "x", "y", "idx_rcv"], rtf_library.real),
            rtf_imag=(["idx_rcv_ref", "f_rtf", "x", "y", "idx_rcv"], rtf_library.imag),
            gcc_real=(["idx_rcv_ref", "f_gcc", "x", "y", "idx_rcv"], gcc_library.real),
            gcc_imag=(["idx_rcv_ref", "f_gcc", "x", "y", "idx_rcv"], gcc_library.imag),
        ),
        coords=dict(
            x=ds_tf.x.values,
            y=ds_tf.y.values,
            idx_rcv=ds_tf.idx_rcv.values,
            idx_rcv_ref=ds_tf.idx_rcv.values,
            f_gcc=ds_tf.f.values,
            f_rtf=ds_tf.f.values,
        ),
        attrs=dict(
            xs=source["x"],
            ys=source["y"],
            snr=np.nan,
            dx=dx,
            dy=dy,
            root_img=os.path.join(ROOT_IMG, f"fullsimu_dx{dx}m_dy{dy}m"),
        ),
    )

    # Save updated dataset
    fpath = os.path.join(
        ROOT_DATA, f"zhang_output_fullsimu_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    )
    ds_res_full_simu.to_netcdf(fpath)
    ds_res_full_simu.close()


if __name__ == "__main__":
    nf = 100
    dx = dy = 20
    from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_process_testcase import (
        process_localisation_zhang2023,
        plot_study_zhang2023,
    )

    # debug = True
    check = False
    use_welch_estimator = True
    # snr_dB = 0
    # rcv_in_fullarray = [0, 1, 4]
    # event_stype = "wn"
    # antenna_type = "random"

    antenna_type = "zhang"
    debug = False
    event_stype = "wn"
    freq_draw_method = "equally_spaced"

    plot_args = {
        "plot_array": False,
        "plot_single_cpl_surf": False,
        "plot_fullarray_surf": True,
        "plot_cpl_surf_comparison": False,
        "plot_fullarray_surf_comparison": False,
        "plot_surf_dist_comparison": False,
        "plot_mainlobe_contour": False,
        "plot_msr_estimation": False,
    }

    # # # Step 2
    # grid_dataset(debug=debug, antenna_type=antenna_type)
    # build_signal(debug=debug, event_stype=event_stype, antenna_type=antenna_type)
    # # # Step 4
    # build_features_fullsimu(
    #     debug=debug, event_stype=event_stype, antenna_type=antenna_type
    # )

    # # Full simu
    # fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
    # ds = xr.open_dataset(fpath)

    # folder = f"fullsimu_dx{dx}m_dy{dy}m"
    # process_localisation_zhang2023(ds, folder, nf=nf, freq_draw_method=freq_draw_method)
    # plot_study_zhang2023(folder, plot_args=plot_args)

    # # snrs = [-20, -10, 0, 10, 20]
    # snrs = [-30]

    # # snrs = np.arange(-30, 32, 2)
    # for snr_dB in snrs:

    #     build_features_from_time_signal(
    #         snr_dB=snr_dB,
    #         debug=debug,
    #         check=check,
    #         use_welch_estimator=use_welch_estimator,
    #         antenna_type=antenna_type,
    #     )

    #     # Step 5 : analysis
    #     fpath = os.path.join(
    #         ROOT_DATA, f"zhang_output_from_signal_dx20m_dy20m_snr{snr_dB:.0f}dB.nc"
    #     )
    #     ds = xr.open_dataset(fpath)

    #     folder = os.path.join(f"from_signal_dx{dx}m_dy{dy}m", f"snr_{snr_dB:.0f}dB")
    #     process_localisation_zhang2023(
    #         ds,
    #         folder,
    #         nf=nf,
    #         freq_draw_method="equally_spaced",
    #         rcv_in_fullarray=rcv_in_fullarray,
    #         antenna_type=antenna_type,
    #         debug=debug,
    #     )
    #     plot_study_zhang2023(
    #         folder, debug=debug, antenna_type=antenna_type, plot_args=plot_args
    #     )

    # from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_diag import (
    #     diag_hermitian_angle_vs_snr,
    # )

    # diag_hermitian_angle_vs_snr(
    #     ref_to_use="kraken", debug=debug, antenna_type=antenna_type
    # )
    # diag_hermitian_angle_vs_snr(
    #     ref_to_use="event", debug=debug, antenna_type=antenna_type
    # )

    # # ## Step 1
    # build_tf_dataset()
    # # # # Step 2
    # grid_dataset(debug=False)
    # # # # Step 3
    # build_signal(debug=False)
    # # # Step 4
    # build_features_fullsimu(debug=debug)
# build_features_from_time_signal(snr_dB=snr_dB, debug=debug)

# # Step 5 : analysis
# nf = 100
# dx = dy = 20
# from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_process_testcase import (
#     process_localisation_zhang2023,
#     plot_study_zhang2023,
# )

# fpath = os.path.join(
#     ROOT_DATA, f"zhang_output_from_signal_dx20m_dy20m_snr{snr_dB:.0f}dB.nc"
# )
# ds = xr.open_dataset(fpath)

# folder = os.path.join(f"from_signal_dx{dx}m_dy{dy}m", f"snr_{snr_dB:.0f}dB")
# process_localisation_zhang2023(ds, folder, nf=nf, freq_draw_method="equally_spaced")
# plot_study_zhang2023(folder)

# # Full simu
# fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
# ds = xr.open_dataset(fpath)

# folder = f"fullsimu_dx{dx}m_dy{dy}m"
# process_localisation_zhang2023(ds, folder, nf=nf)
# plot_study_zhang2023(folder)
