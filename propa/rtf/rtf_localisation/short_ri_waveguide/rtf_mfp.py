#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   classical_mfp.py
@Time    :   2024/11/05 15:44:06
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import sys

sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd")

import dask.array as da
import scipy.interpolate as sp_int

from dask import delayed
from dask.diagnostics import ProgressBar

from propa.rtf.rtf_localisation.rtf_localisation_utils import plot_ambiguity_surface
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_kraken import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_consts import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_testcases import *


ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\rtf\short_ri_waveguide"
ROOT_IMG_TEST = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide\test_implementation"

# ======================================================================================================================
# MFP with RTF built from Kraken
# ======================================================================================================================


def common_params():
    n_rcv = 5
    z_src = 5
    r_src = 30 * 1e3
    delta_rcv = 500

    return n_rcv, z_src, r_src, delta_rcv


def get_target_signal(testcase, snr_dB):
    plot = False
    if testcase == 1:
        result = testcase_1_unpropagated_whitenoise(snr_dB=snr_dB, plot=plot)
    elif testcase == 2:
        result = testcase_2_propagated_whitenoise(snr_dB=snr_dB, plot=plot)
    elif testcase == 3:
        result = testcase_3_propagated_interference(
            snr_dB=snr_dB, plot=plot, interference_type="z_call"
        )

    return result


def mfp_simulated_replicas(testcase, snr_dB):

    n_rcv, z_src, r_src, delta_rcv = common_params()

    ## Step 1 : Derive the received signal for the testcase of interest ##
    result = get_target_signal(testcase, snr_dB)

    # Extract the estimated RTF for both estimation methods
    f_cs = result["cs"]["f"]
    f_cw = result["cw"]["f"]
    rtf_cs = result["cs"]["rtf"]
    rtf_cw = result["cw"]["rtf"]
    # Restrict to a given frequency band
    fmin = 5
    fmax = 45
    idx_band = (f_cs >= fmin) & (f_cs <= fmax)
    f_cs = f_cs[idx_band]
    f_cw = f_cw[idx_band]
    rtf_cs = rtf_cs[idx_band]
    rtf_cw = rtf_cw[idx_band]

    ## Step 2 : derive rtf at each grid point ##
    # Load data
    data_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_mfp_grid.nc"
    ds_tf = xr.open_dataset(data_path)

    # Subsample the dataset along range axis to increase performance
    dr_subsample = 50
    r_subsample = np.arange(0, ds_tf.r.max(), dr_subsample)
    ds_tf = ds_tf.sel(r=r_subsample, method="nearest")

    # Define grid limits
    rmin = 5 * 1e3
    rmax = 45 * 1e3
    rmin_load = rmin - n_rcv * delta_rcv * 1.5
    rmax_load = rmax + n_rcv * delta_rcv * 1.5
    zmin = 1
    zmax = 950
    ds_tf = ds_tf.sel(r=slice(rmin_load, rmax_load), z=slice(zmin, zmax))

    # Interp tf dataset to the same frequency grid as the estimated RTF
    ds_tf = ds_tf.sel(f=f_cs, method="nearest")

    # Reference transfert function
    tf_rcv_ref = (
        ds_tf.sel(r=slice(rmin, rmax)).tf_real
        + 1j * ds_tf.sel(r=slice(rmin, rmax)).tf_imag
    )

    # Initialize the rtf grid
    grid_shape = tf_rcv_ref.shape + (n_rcv,)
    rtf_grid = np.zeros(grid_shape, dtype=np.complex128)
    for i in range(0, n_rcv):
        # Receiver range displacement from the reference position
        delta_r = i * delta_rcv
        r_min_i = rmin - delta_r
        r_max_i = rmax - delta_r
        # Slice the dataset corresponding to the current receiver position
        ds_tf_rcv_i = ds_tf.sel(r=slice(r_min_i, r_max_i))
        tf_rcv_i = ds_tf_rcv_i.tf_real + 1j * ds_tf_rcv_i.tf_imag
        # Compute the RTF
        rtf_i = tf_rcv_i.values / tf_rcv_ref
        # Save the RTF
        rtf_grid[..., i] = rtf_i.values

    # Convert to xarray for easy manipulation
    r = tf_rcv_ref.r.values
    z = tf_rcv_ref.z.values
    f = tf_rcv_ref.f.values
    rtf_grid = xr.DataArray(
        rtf_grid,
        dims=["f", "z", "r", "idx_rcv"],
        coords={"f": f, "z": z, "r": r, "idx_rcv": np.arange(n_rcv)},
    )

    # Convert rtf_cs and rtf_cw to xarray
    rtf_cs = xr.DataArray(
        rtf_cs,
        dims=["f", "idx_rcv"],
        coords={"f": f_cs, "idx_rcv": np.arange(n_rcv)},
    )
    rtf_cw = xr.DataArray(
        rtf_cw,
        dims=["f", "idx_rcv"],
        coords={"f": f_cw, "idx_rcv": np.arange(n_rcv)},
    )

    ## Step 3 : Match field processing ##
    # Select dist function to apply
    dist = "hermitian_angle"
    if dist == "frobenius":
        dist_func = D_frobenius
        dist_kwargs = {}

    elif dist == "hermitian_angle":
        dist_func = D_hermitian_angle_fast
        dist_kwargs = {
            "ax_rcv": 3,
            "unit": "deg",
            "apply_mean": True,
        }

    # Compute distance bewteen the estimated RTF and RTF at each grid point
    D_cs = dist_func(rtf_cs, rtf_grid, **dist_kwargs)
    D_cw = dist_func(rtf_cw, rtf_grid, **dist_kwargs)

    # Add the distance to the xarray
    D_cs = xr.DataArray(
        D_cs,
        dims=["z", "r"],
        coords={"z": z, "r": r},
    )
    D_cw = xr.DataArray(
        D_cw,
        dims=["z", "r"],
        coords={"z": z, "r": r},
    )

    # Plot ambiguity surfaces
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide"

    plot_args = {
        "dist": dist,
        "snr": snr_dB,
        "rtf_method": None,
        "vmax_percentile": 5,
        "root_img": root_img,
        "testcase": result["tc_label"],
        "mfp_method": "simulated_replicas",
        "dist_label": r"$\theta \, \textrm{[°]}$",
    }

    plot_args["rtf_method"] = "cs"
    plot_ambiguity_surface(amb_surf=D_cs, r_src=r_src, z_src=z_src, plot_args=plot_args)

    plot_args["rtf_method"] = "cw"
    plot_ambiguity_surface(amb_surf=D_cw, r_src=r_src, z_src=z_src, plot_args=plot_args)

    # Estimate source position
    z_min_cs, r_min_cs = np.unravel_index(np.argmin(D_cs.values), D_cs.shape)
    r_src_hat_cs = D_cs.r[r_min_cs]
    z_src_hat_cs = D_cs.z[z_min_cs]

    z_min_cw, r_min_cw = np.unravel_index(np.argmin(D_cw.values), D_cw.shape)
    r_src_hat_cw = D_cw.r[r_min_cw]
    z_src_hat_cw = D_cw.z[z_min_cw]

    pos_hat_cs = (r_src_hat_cs, z_src_hat_cs)
    pos_hat_cw = (r_src_hat_cw, z_src_hat_cw)

    return pos_hat_cs, pos_hat_cw


def mfp_measured_replicas(testcase, snr_dB):

    # Load params
    n_rcv, z_src, r_src, delta_rcv = common_params()

    ## Step 1 : Derive the received signal for the testcase of interest ##
    result = get_target_signal(testcase, snr_dB)

    # Extract the estimated RTF for both estimation methods
    f_cs = result["cs"]["f"]
    f_cw = result["cw"]["f"]
    rtf_cs = result["cs"]["rtf"]
    rtf_cw = result["cw"]["rtf"]
    # Restrict to a given frequency band
    fmin = 10
    fmax = 50
    idx_band = (f_cs >= fmin) & (f_cs <= fmax)
    f_cs = f_cs[idx_band]
    f_cw = f_cw[idx_band]
    rtf_cs = rtf_cs[idx_band]
    rtf_cw = rtf_cw[idx_band]

    ## Step 2 : derive rtf at each grid point ##
    # Load data
    data_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_mfp_grid.nc"
    ds_tf = xr.open_dataset(data_path)

    # Subsample the dataset along range axis to increase performance
    dr_subsample = 100
    r_subsample = np.arange(0, ds_tf.r.max(), dr_subsample)
    ds_tf = ds_tf.sel(r=r_subsample, method="nearest")

    # Subsample the dataset along depth axis to increase performance
    dz_subsample = 5
    z_subsample = np.arange(0, ds_tf.z.max(), dz_subsample)
    ds_tf = ds_tf.sel(z=z_subsample, method="nearest")

    # Define grid limits
    rmin = 15 * 1e3
    rmax = 45 * 1e3
    rmin_load = rmin - n_rcv * delta_rcv * 1.5
    rmax_load = rmax + n_rcv * delta_rcv * 1.5
    rmin_load = min(rmin_load, ds_tf.r.min())
    rmax_load = min(rmax_load, ds_tf.r.max())

    zmin = 1
    zmax = 100
    ds_tf = ds_tf.sel(r=slice(rmin_load, rmax_load), z=slice(zmin, zmax))

    grid_lims = {
        "rmin": rmin,
        "rmax": rmax,
        "zmin": zmin,
        "zmax": zmax,
    }
    resolution = {
        "dr": dr_subsample,
        "dz": dz_subsample,
    }
    # Derive received signal at each receiver position
    ds_rcv_sig = get_received_signal(
        tf_xr=ds_tf,
        tau_ir=TAU_IR,
        rmin=rmin,
        rmax=rmax,
        grid_lims=grid_lims,
        grid_resolution=resolution,
    )

    # Derive received noise
    ds_noise = derive_received_noise(rcv_sig_xr=ds_rcv_sig, snr_dB=snr_dB)

    # Dummy and dirty way to derive the RTF from the received signal -> loop over each grid pixel

    t = ds_rcv_sig.t.values

    alpha_tau_ir = 3
    seg_length = alpha_tau_ir * TAU_IR
    nperseg = int(seg_length / (t[1] - t[0]))
    # Find the nearest power of 2
    nperseg = 2 ** int(np.log2(nperseg) + 1)
    alpha_overlap = 1 / 2
    noverlap = int(nperseg * alpha_overlap)

    # Use the first signal slice to set f_cs by running rtf_covariance_substraction once
    rcv_sig = ds_rcv_sig.sel(r=r_src, z=z_src).values
    rcv_noise = ds_noise.sel(r=r_src, z=z_src).values
    f_rtf, rtf, _, _, _ = rtf_covariance_substraction(
        t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
    )

    # Use Dask delayed and progress bar for tracking
    with ProgressBar():
        results_cs = []
        results_cw = []
        for z_i in ds_rcv_sig.z.values:
            for r_i in ds_rcv_sig.r.values:
                rcv_sig = ds_rcv_sig.sel(r=r_i, z=z_i).values
                rcv_noise = ds_noise.sel(r=r_i, z=z_i).values

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

                delayed_rtf_cw = da.from_delayed(
                    delayed(
                        lambda t, sig, noise: rtf_covariance_whitening(
                            t, sig, noise, nperseg, noverlap
                        )[1]
                    )(t, rcv_sig, rcv_noise),
                    shape=(len(f_rtf), rcv_sig.shape[1]),
                    dtype=complex,
                )
                results_cw.append(delayed_rtf_cw)

        # Compute all RTFs
        rtf_cs_all = da.stack(results_cs).compute()
        rtf_cw_all = da.stack(results_cw).compute()

    # Reshape to the required shape
    # First step : reshape tp (nz, nr, nf, n_rcv)
    shape = (len(ds_rcv_sig.z), len(ds_rcv_sig.r)) + rtf_cs_all.shape[1:]
    rtf_cs_all = rtf_cs_all.reshape(shape)
    rtf_cw_all = rtf_cw_all.reshape(shape)

    # Step 2 : permute to (nf, nz, nr, n_rcv)
    axis_permutation = (2, 0, 1, 3)
    rtf_cs_all = np.transpose(rtf_cs_all, axis_permutation)
    rtf_cw_all = np.transpose(rtf_cw_all, axis_permutation)

    # Restict to the frequency band of interest
    idx_band = (f_rtf >= fmin) & (f_rtf <= fmax)
    f_rtf = f_rtf[idx_band]
    rtf_cs_all = rtf_cs_all[idx_band]
    rtf_cw_all = rtf_cw_all[idx_band]

    # Convert to xarray
    rtf_cs_xr = xr.DataArray(
        rtf_cs_all,
        dims=["f", "z", "r", "idx_rcv"],
        coords={
            "f": f_rtf,
            "z": ds_rcv_sig.z,
            "r": ds_rcv_sig.r,
            "idx_rcv": np.arange(n_rcv),
        },
    )

    rtf_cw_xr = xr.DataArray(
        rtf_cw_all,
        dims=["f", "z", "r", "idx_rcv"],
        coords={
            "f": f_rtf,
            "z": ds_rcv_sig.z,
            "r": ds_rcv_sig.r,
            "idx_rcv": np.arange(n_rcv),
        },
    )

    # Compare to ref rtf
    kraken_data = load_data()
    f_true, rtf_true = true_rtf(kraken_data)
    plt.figure()
    np.abs(rtf_cs_xr).sel(r=r_src, z=z_src, idx_rcv=4).plot(
        label="mfp (cs)", marker="o"
    )
    # plt.plot(f_rtf, np.abs(rtf[:, 4]), label="mfp (cs) - single run")
    # np.abs(rtf_cw_xr).sel(r=r_src, z=z_src, idx_rcv=4).plot(label="mfp (cw)")
    plt.plot(f_cs, np.abs(rtf_cs[:, 4]), label="testcase")
    plt.plot(f_true, np.abs(rtf_true[:, 4]), label="kraken")
    plt.yscale("log")
    plt.legend()
    fpath = os.path.join(ROOT_IMG_TEST, "rtf_mfp_vs_testcase.png")
    plt.savefig(fpath)

    ## Step 3 : Match field processing ##
    # Select dist function to apply
    dist = "hermitian_angle"
    if dist == "frobenius":
        dist_func = D_frobenius
        dist_kwargs = {}

    elif dist == "hermitian_angle":
        dist_func = D_hermitian_angle_fast
        dist_kwargs = {
            "ax_rcv": 3,
            "unit": "deg",
            "apply_mean": True,
        }

    # Compute distance bewteen the estimated RTF and RTF at each grid point
    D_cs = dist_func(rtf_cs, rtf_cs_xr, **dist_kwargs)
    D_cw = dist_func(rtf_cw, rtf_cw_xr, **dist_kwargs)

    # Add the distance to the xarray
    r = rtf_cs_xr.r.values
    z = rtf_cs_xr.z.values
    D_cs = xr.DataArray(
        D_cs,
        dims=["z", "r"],
        coords={"z": z, "r": r},
    )
    D_cw = xr.DataArray(
        D_cw,
        dims=["z", "r"],
        coords={"z": z, "r": r},
    )

    # Plot ambiguity surfaces
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide"

    plot_args = {
        "dist": dist,
        "snr": snr_dB,
        "rtf_method": None,
        "vmax_percentile": 5,
        "root_img": root_img,
        "testcase": result["tc_label"],
        "mfp_method": "measured_replicas",
        "dist_label": r"$\theta \, \textrm{[°]}$",
    }

    plot_args["rtf_method"] = "cs"
    plot_ambiguity_surface(amb_surf=D_cs, r_src=r_src, z_src=z_src, plot_args=plot_args)

    plot_args["rtf_method"] = "cw"
    plot_ambiguity_surface(amb_surf=D_cw, r_src=r_src, z_src=z_src, plot_args=plot_args)

    # Estimate source position
    z_min_cs, r_min_cs = np.unravel_index(np.argmin(D_cs.values), D_cs.shape)
    r_src_hat_cs = D_cs.r[r_min_cs]
    z_src_hat_cs = D_cs.z[z_min_cs]

    z_min_cw, r_min_cw = np.unravel_index(np.argmin(D_cw.values), D_cw.shape)
    r_src_hat_cw = D_cw.r[r_min_cw]
    z_src_hat_cw = D_cw.z[z_min_cw]

    pos_hat_cs = (r_src_hat_cs, z_src_hat_cs)
    pos_hat_cw = (r_src_hat_cw, z_src_hat_cw)

    return pos_hat_cs, pos_hat_cw


def get_received_signal(
    tf_xr, tau_ir, rmin, rmax, grid_lims, grid_resolution, sl=200, z_rcv=999, r_rcv=0
):
    fpath = os.path.join(
        ROOT_DATA,
        f"received_sig_mfp_grid_rmin{np.round(grid_lims['rmin'], 0)}_rmax{np.round(grid_lims['rmax'], 0)}.nc",
    )
    if os.path.exists(fpath):
        ds_rcv_sig = xr.open_dataarray(fpath)
    else:
        ds_rcv_sig = derive_rcv_signal(tf_xr, tau_ir, rmin, rmax, sl, z_rcv, r_rcv)
        ds_rcv_sig.to_netcdf(fpath)

    # Restrict to the grid limits
    ds_rcv_sig = ds_rcv_sig.sel(
        r=slice(grid_lims["rmin"], grid_lims["rmax"]),
        z=slice(grid_lims["zmin"], grid_lims["zmax"]),
    )
    # Subsample to desired resolution
    r_subsample = np.arange(
        ds_rcv_sig.r.min(), ds_rcv_sig.r.max(), grid_resolution["dr"]
    )
    ds_rcv_sig = ds_rcv_sig.sel(r=r_subsample, method="nearest")
    z_subsample = np.arange(
        ds_rcv_sig.z.min(), ds_rcv_sig.z.max(), grid_resolution["dz"]
    )
    ds_rcv_sig = ds_rcv_sig.sel(z=z_subsample, method="nearest")

    return ds_rcv_sig


def derive_rcv_signal(tf_xr, tau_ir, rmin, rmax, sl=200, z_rcv=999, r_rcv=0):

    n_rcv, z_src, r_src, delta_rcv = common_params()

    duration = 50 * tau_ir
    f = tf_xr.f
    # nf = tf_xr.sizes["f"]
    fs = 100

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

    # Apply windowing to avoid side-effects
    s *= np.hanning(len(s))
    src_spectrum = np.fft.rfft(s)
    # Spectrum frequency vector
    nt = len(t)
    ts = t[1] - t[0]
    f_src = fft.rfftfreq(nt, ts)

    tf = tf_xr.tf_real + 1j * tf_xr.tf_imag

    # Broadcast src_spectrum to the same shape as tf
    src_spectrum = np.expand_dims(src_spectrum, axis=(1, 2))
    tile_shape = tuple(
        [tf.shape[i] - src_spectrum.shape[i] + 1 for i in range(src_spectrum.ndim)]
    )
    src_spectrum_expanded = np.tile(src_spectrum, tile_shape)

    # Apply delay correction so that the signal is centered within the time window (otherwise the signal is shifted with wrap around effect in the time domain)
    delay_correction = np.sqrt(((tf.z - z_rcv) ** 2) + (tf.r - r_rcv) ** 2) / C0
    delay_correction = np.expand_dims(delay_correction, axis=0)
    tile_shape = tuple(
        [
            src_spectrum_expanded.shape[i] - delay_correction.shape[i] + 1
            for i in range(delay_correction.ndim)
        ]
    )
    delay_correction = np.tile(delay_correction, tile_shape)
    tau = mult_along_axis(delay_correction, f_src, axis=0)
    src_spec = src_spectrum_expanded * np.exp(1j * 2 * np.pi * tau)

    # Derive received spectrum by multiplying with the transfert function array
    rcv_spec = src_spec * tf

    # grid_shape = (nt,) + tf.sel(r=slice(rmin, rmax)).shape[1:] + (n_rcv,)
    # rcv_sig_grid = np.zeros(grid_shape, dtype=np.float64)
    # for i in range(0, n_rcv):
    #     # Receiver range displacement from the reference position
    #     delta_r = i * delta_rcv
    #     r_min_i = rmin - delta_r
    #     r_max_i = rmax - delta_r
    #     # Slice the dataset corresponding to the current receiver position
    #     rcv_spec_rcv_i = rcv_spec.sel(r=slice(r_min_i, r_max_i))
    #     # Compute the received signal associated to each grid pixel
    #     sig_rcv_i = np.fft.irfft(rcv_spec_rcv_i, axis=0)
    #     # Save the received signal array
    #     rcv_sig_grid[..., i] = sig_rcv_i

    # Create Dask delayed version for receiver grid

    def compute_rcv_signal(i, rmin, rmax, rcv_spec, delta_rcv):
        r_min_i = rmin - i * delta_rcv
        r_max_i = rmax - i * delta_rcv
        rcv_spec_rcv_i = rcv_spec.sel(r=slice(r_min_i, r_max_i))
        sig_rcv_i = np.fft.irfft(rcv_spec_rcv_i, axis=0)
        return sig_rcv_i

    # Use Dask to process each receiver's signal in parallel
    with ProgressBar():
        results = []
        for i in range(n_rcv):
            delayed_rcv_sig = delayed(compute_rcv_signal)(
                i, rmin, rmax, rcv_spec, delta_rcv
            )
            results.append(delayed_rcv_sig)

        # Compute all receiver signals in parallel
        rcv_sig_grid = da.compute(*results)

    # Convert to numpy array
    rcv_sig_grid = np.array(rcv_sig_grid)
    # Transpose from (n_rcv, nt, nz, nr) to (nt, nz, nr, n_rcv)
    rcv_sig_grid = np.transpose(rcv_sig_grid, (1, 2, 3, 0))

    # Convert back to xarray
    z = tf.z
    r = tf.sel(r=slice(rmin, rmax)).r
    rcv_sig_xr = xr.DataArray(
        rcv_sig_grid,
        dims=["t", "z", "r", "idx_rcv"],
        coords={"t": t, "z": z, "r": r, "idx_rcv": np.arange(n_rcv)},
        attrs={
            "n_rcv": n_rcv,
            "r_src": r_src,
            "z_src": z_src,
            "f0": f0,
            "std_fi": std_fi,
            "tau_corr_fi": tau_corr_fi,
        },
    )

    return rcv_sig_xr


def derive_received_noise(
    rcv_sig_xr,
    snr_dB=10,
    noise_model="gaussian",
):

    # Extract the received signal at the source position
    r_src = rcv_sig_xr.r_src
    z_src = rcv_sig_xr.z_src
    rcv_sig_src = rcv_sig_xr.sel(r=r_src, z=z_src, method="nearest")

    # Derive noise signal
    # We assume that the noise is due to ambiant noise and is the same at each receiver position and does not depend on the source position within the search grid
    sigma_rcv_ref = np.std(rcv_sig_src.sel(idx_rcv=0).values)

    # Create noise dataarray
    rcv_noise_xr = xr.zeros_like(rcv_sig_xr)

    # Build noise signal
    sigma_noise = np.sqrt(10 ** (-snr_dB / 10))
    # Correct definition shoul use multivariate gaussian noise to ensure the std is the same at each grid pixel
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

    noise_sig = np.random.normal(loc=0, scale=sigma_noise, size=rcv_sig_xr.shape)
    # Normalize to account for the signal power to reach required snr at receiver 0
    noise_sig *= sigma_rcv_ref

    # Store in xarray
    rcv_noise_xr.values = noise_sig
    rcv_noise_xr.attrs["sigma_ref"] = sigma_rcv_ref
    rcv_noise_xr.attrs["sigma_noise"] = sigma_noise

    return rcv_noise_xr


# ======================================================================================================================
# Test functions
# ======================================================================================================================


def test(
    rtf_cs, rtf_cw, f_cs, f_cw, rtf_grid, z_src, r_src, n_rcv, dist_func, dist_kwargs
):
    kraken_data = load_data()
    f_true, rtf_true = true_rtf(kraken_data)

    # Check distance
    dist_kwargs["ax_rcv"] = 1
    rtf_kraken_rcv_pos = rtf_grid.sel(r=r_src, z=z_src)
    d_cs = dist_func(rtf_cs.values, rtf_kraken_rcv_pos.values, **dist_kwargs)
    d_cw = dist_func(rtf_cw.values, rtf_kraken_rcv_pos.values, **dist_kwargs)

    print("Distance between estimated RTF and RTF at source position:")
    print(f"CS: {d_cs:.2f}°")
    print(f"CW: {d_cw:.2f}°")

    f_interp, rtf_true_interp = interp_true_rtf(kraken_data, f_cs)
    d_cs = dist_func(rtf_cs.values, rtf_true_interp, **dist_kwargs)
    d_cw = dist_func(rtf_cw.values, rtf_true_interp, **dist_kwargs)

    print("Distance between estimated RTF and true RTF at source position:")
    print(f"CS: {d_cs:.2f}°")
    print(f"CW: {d_cw:.2f}°")

    d = dist_func(rtf_kraken_rcv_pos.values, rtf_true_interp, **dist_kwargs)

    print(f"Distance between true and grid rtf at src pos: {d:.2f}°")

    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide\test_implementation"
    for i in range(0, n_rcv):
        # delta_r = i * delta_rcv
        # hf_ref = kraken_data[f"rcv{i}"]["h_f"]
        # hf_loose_grid = tf_rcv_ref.sel(r=r_src - delta_r, z=z_src).values
        # plt.figure()
        # plt.plot(ds_tf.f, np.abs(hf_loose_grid), label=f"loose grid - {i}")
        # plt.plot(f_true, np.abs(hf_ref), label=f"ref - {i}")
        # # plt.yscale("log")
        # plt.legend()
        # fpath = os.path.join(root_img, f"tf_ref_vs_loose_grid_{i}.png")
        # plt.savefig(fpath)

        # rtf_ref = rtf_true[:, i]
        # rtf_loose_grid = (
        #     tf_rcv_ref.sel(r=r_src - delta_r, z=z_src).values
        #     / tf_rcv_ref.sel(r=r_src, z=z_src).values
        # )
        # plt.figure()
        # plt.plot(ds_tf.f, np.abs(rtf_loose_grid), label=f"loose grid - {i}")
        # plt.plot(f_true, np.abs(rtf_ref), label=f"ref - {i}")
        # # plt.yscale("log")
        # plt.legend()
        # fpath = os.path.join(root_img, f"rtf_ref_vs_loose_grid_{i}.png")
        # plt.savefig(fpath)

        # rtf_grid_i = rtf_grid[..., i]
        # rtf_i = rtf_grid_i[:, idx_z_src, idx_r_src]
        rtf_i = np.abs(rtf_grid.sel(idx_rcv=i, z=z_src, r=r_src))

        plt.figure()
        plt.plot(f_cs, np.abs(rtf_cs.sel(idx_rcv=i)), label="cs")
        plt.plot(f_cw, np.abs(rtf_cw.sel(idx_rcv=i)), label="cw")
        plt.plot(f_interp, np.abs(rtf_true_interp[:, i]), label="ref", marker="o")
        plt.plot(f_cs, rtf_i, label="loose grid", color="k", linestyle="--", marker="x")
        plt.legend()
        plt.yscale("log")
        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$|\Pi(f)|$")

        fpath = os.path.join(root_img, f"rtf_estim_ref_vs_loose_grid_{i}.png")
        plt.savefig(fpath)

        # Same plot but for the phase
        rtf_i = np.angle(rtf_grid.sel(idx_rcv=i, z=z_src, r=r_src))
        plt.figure()
        plt.plot(f_cs, np.angle(rtf_cs.sel(idx_rcv=i)), label="cs")
        plt.plot(f_cw, np.angle(rtf_cw.sel(idx_rcv=i)), label="cw")
        plt.plot(f_cs, np.angle(rtf_true_interp[:, i]), label="ref")
        plt.plot(f_cs, rtf_i, label="loose grid", color="k", linestyle="--")
        plt.legend()
        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$\angle \Pi(f)$")

        fpath = os.path.join(root_img, f"rtf_estim_ref_vs_loose_grid_phase_{i}.png")
        plt.savefig(fpath)


def plot_waveguide_mean_tl():

    n_rcv, z_src, r_src, delta_rcv = common_params()

    # Load data
    data_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_mfp_grid.nc"
    ds_tf = xr.open_dataset(data_path)

    # Subsample the dataset along range axis to increase performance
    dr_subsample = 500
    r_subsample = np.arange(0, ds_tf.r.max(), dr_subsample)
    ds_tf = ds_tf.sel(r=r_subsample, method="nearest")

    # Define grid limits
    rmin = 5 * 1e3
    rmax = 45 * 1e3
    rmin_load = rmin - n_rcv * delta_rcv * 1.5
    rmax_load = rmax + n_rcv * delta_rcv * 1.5
    zmin = 1
    zmax = 950
    ds_tf = ds_tf.sel(r=slice(rmin_load, rmax_load), z=slice(zmin, zmax))

    tf = ds_tf.tf_real + 1j * ds_tf.tf_imag
    # Replace 0 by nan
    tf = np.abs(tf).where(tf != 0, np.nan)
    tl_mean = 20 * np.log10(tf).mean(dim="f")

    plt.figure()
    tl_mean.plot(
        yincrease=False,
        cmap="jet",
        cbar_kwargs={"label": "Mean TL [dB]"},
    )
    # Save
    fpath = os.path.join(ROOT_IMG_TEST, "mean_tl_waveguide.png")
    plt.savefig(fpath)

    # # Plot tl at diffent freqs
    # f_to_plot = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    # for f in f_to_plot:
    #     plt.figure()
    #     tl_f = 20 * np.log10(tf.sel(f=f))
    #     tl_f.plot(
    #         yincrease=False,
    #         cmap="jet",
    #         cbar_kwargs={"label": "Mean TL [dB]"},
    #     )
    #     plt.title(f"TL at {f} Hz")
    #     # Save
    #     fpath = os.path.join(ROOT_IMG_TEST, f"tl_waveguide_{f}Hz.png")
    #     plt.savefig(fpath)

    # Plot mean tl over a given frequency band
    fmin = 11
    fmax = 16
    tl_band = 20 * np.log10(tf.sel(f=slice(fmin, fmax)))
    tl_mean_band = tl_band.mean(dim="f")

    plt.figure()
    tl_mean_band.plot(
        yincrease=False,
        cmap="jet",
        cbar_kwargs={"label": f"Mean TL [dB] ({fmin}-{fmax} Hz)"},
    )
    # Save
    fpath = os.path.join(ROOT_IMG_TEST, f"mean_tl_waveguide_{fmin}-{fmax}Hz.png")
    plt.savefig(fpath)


if __name__ == "__main__":

    # plot_waveguide_mean_tl()

    tc = 1
    snr = 0

    mfp_simulated_replicas(tc, snr)
    mfp_measured_replicas(tc, snr)

    snrs = [-30, -20, -10, 0, 10, 20, 30]
    # snrs = np.arange(-50, 55, 5)

    _, z_src, r_src, _ = common_params()

    # snrs = [0, 1]
    n_monte_carlo = 10

    # pos_cs_hat_simulated_replicas = []
    # pos_cw_hat_simulated_replicas = []
    # for snr in snrs:
    #     pos_cs = []
    #     pos_cw = []
    #     for i in range(n_monte_carlo):
    #         pos_cs_hat, pos_cw_hat = mfp_simulated_replicas(tc, snr)
    #         pos_cs.append(pos_cs_hat)
    #         pos_cw.append(pos_cw_hat)

    #     pos_cs_hat_simulated_replicas.append(pos_cs)
    #     pos_cw_hat_simulated_replicas.append(pos_cw)

    # pos_cs_hat_simulated_replicas = np.array(pos_cs_hat_simulated_replicas)
    # pos_cw_hat_simulated_replicas = np.array(pos_cw_hat_simulated_replicas)

    # rmse_pos_cs = np.sqrt(
    #     np.mean((pos_cs_hat_simulated_replicas - np.array([r_src, z_src])) ** 2, axis=1)
    # )
    # rmse_pos_cw = np.sqrt(
    #     np.mean((pos_cw_hat_simulated_replicas - np.array([r_src, z_src])) ** 2, axis=1)
    # )

    # # Plot rmse vs snr
    # root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide\simulated_replicas\testcase_1_unpropagated_whitenoise"
    # # Range
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     rmse_pos_cs[:, 0],
    #     label="CS",
    #     linestyle="-",
    #     color="b",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.plot(
    #     snrs,
    #     rmse_pos_cw[:, 0],
    #     label="CW",
    #     linestyle="-",
    #     color="r",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.xlabel(r"$\textrm{SNR [dB]}$")
    # plt.ylabel(r"$\textrm{RMSE [m]}$")
    # plt.legend()

    # plt.savefig(os.path.join(root_img, "rmse_vs_snr_range.png"))

    # # Depth
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     rmse_pos_cs[:, 1],
    #     label="CS",
    #     linestyle="-",
    #     color="b",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.plot(
    #     snrs,
    #     rmse_pos_cw[:, 1],
    #     label="CW",
    #     linestyle="-",
    #     color="r",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.xlabel(r"$\textrm{SNR [dB]}$")
    # plt.ylabel(r"$\textrm{RMSE [m]}$")
    # plt.legend()

    # plt.savefig(os.path.join(root_img, "rmse_vs_snr_depth.png"))

    # pos_cs_hat_measured_replicas = []
    # pos_cw_hat_measured_replicas = []
    # for snr in snrs:
    #     pos_cs = []
    #     pos_cw = []
    #     for i in range(n_monte_carlo):
    #         pos_cs_hat, pos_cw_hat = mfp_measured_replicas(tc, snr)
    #         pos_cs.append(pos_cs_hat)
    #         pos_cw.append(pos_cw_hat)

    #     pos_cs_hat_measured_replicas.append(pos_cs)
    #     pos_cw_hat_measured_replicas.append(pos_cw)

    # pos_cs_hat_measured_replicas = np.array(pos_cs_hat_measured_replicas)
    # pos_cw_hat_measured_replicas = np.array(pos_cw_hat_measured_replicas)

    # rmse_pos_cs = np.sqrt(
    #     np.mean((pos_cs_hat_measured_replicas - np.array([r_src, z_src])) ** 2, axis=1)
    # )
    # rmse_pos_cw = np.sqrt(
    #     np.mean((pos_cw_hat_measured_replicas - np.array([r_src, z_src])) ** 2, axis=1)
    # )

    # # Plot rmse vs snr
    # root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\short_ri_waveguide\measured_replicas\testcase_1_unpropagated_whitenoise"
    # # Range
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     rmse_pos_cs[:, 0],
    #     label="CS",
    #     linestyle="-",
    #     color="b",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.plot(
    #     snrs,
    #     rmse_pos_cw[:, 0],
    #     label="CW",
    #     linestyle="-",
    #     color="r",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.xlabel(r"$\textrm{SNR [dB]}$")
    # plt.ylabel(r"$\textrm{RMSE [m]}$")
    # plt.legend()

    # plt.savefig(os.path.join(root_img, "rmse_vs_snr_range.png"))

    # # Depth
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     rmse_pos_cs[:, 1],
    #     label="CS",
    #     linestyle="-",
    #     color="b",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.plot(
    #     snrs,
    #     rmse_pos_cw[:, 1],
    #     label="CW",
    #     linestyle="-",
    #     color="r",
    #     marker="o",
    #     linewidth=0.5,
    #     markersize=2,
    # )
    # plt.xlabel(r"$\textrm{SNR [dB]}$")
    # plt.ylabel(r"$\textrm{RMSE [m]}$")
    # plt.legend()

    # plt.savefig(os.path.join(root_img, "rmse_vs_snr_depth.png"))

    # if noise_model == "gaussian":
    #     for i in range(n_rcv):
    #         sigma_v2 = 10 ** (-snr_dB / 10)
    #         v = np.random.normal(loc=0, scale=np.sqrt(sigma_v2), size=ns)
    #         noise_sig = v
    #         # Normalize to account for the signal power to reach required snr at receiver 0
    #         noise_sig *= sigma_ref
    #         # Derive noise spectrum
    #         noise_spectrum = np.fft.rfft(noise_sig)

    #         # Psd
    #         psd_noise = scipy.signal.welch(
    #             noise_sig, fs=fs, nperseg=2**12, noverlap=2**11
    #         )

    #         # Save noise signal
    #         received_noise[f"rcv{i}"] = {
    #             "psd": psd_noise,
    #             "sig": noise_sig,
    #             "spect": noise_spectrum,
    #         }

    # # Compute the received noise signal
    # if propagated:

    #     # Load noise dataset
    #     ds_tf = xr.open_dataset(os.path.join(ROOT_DATA, "kraken_tf_surface_noise.nc"))

    #     delta_rcv = 500
    #     if "rmin" in propagated_args.keys() and propagated_args["rmin"] is not None:
    #         rmin_noise = propagated_args["rmin"]
    #     else:
    #         rmin_noise = 5 * 1e3  # Default minimal range for noise source

    #     if "rmax" in propagated_args.keys() and propagated_args["rmax"] is not None:
    #         rmax_noise = propagated_args["rmax"]
    #     else:
    #         rmax_noise = ds_tf.r.max().values  # Default maximal range for noise source

    #     for i in range(N_RCV):
    #         r_src_noise_start = rmin_noise - i * delta_rcv
    #         r_src_noise_end = rmax_noise - i * delta_rcv
    #         idx_r_min = np.argmin(np.abs(ds_tf.r.values - r_src_noise_start))
    #         idx_r_max = np.argmin(np.abs(ds_tf.r.values - r_src_noise_end))

    #         tf_noise_rcv_i = (
    #             ds_tf.tf_real[:, idx_r_min:idx_r_max]
    #             + 1j * ds_tf.tf_imag[:, idx_r_min:idx_r_max]
    #         )

    #         # Noise spectrum
    #         if noise_model == "gaussian":
    #             v = np.random.normal(loc=0, scale=1, size=ns)
    #             noise_spectrum = np.fft.rfft(v)

    #         # Multiply the transfert function by the noise source spectrum
    #         noise_field_f = mult_along_axis(tf_noise_rcv_i, noise_spectrum, axis=0)
    #         noise_field = np.fft.irfft(noise_field_f, axis=0)
    #         noise_sig = np.sum(noise_field, axis=1)  # Sum over all noise sources

    #         # Normalise to required lvl at receiver 0
    #         if i == 0:
    #             sigma_v2 = 10 ** (-snr_dB / 10)
    #             sigma_noise = np.std(noise_sig)
    #             alpha = np.sqrt(sigma_v2) / sigma_noise

    #         noise_sig *= alpha

    #         # Normalize to account for the signal power to reach required snr at receiver 0
    #         noise_sig *= sigma_ref

    #         # Psd
    #         psd_noise = scipy.signal.welch(
    #             noise_sig, fs=fs, nperseg=2**12, noverlap=2**11
    #         )

    #         # Save noise signal
    #         received_noise[f"rcv{i}"] = {
    #             "psd": psd_noise,
    #             "sig": noise_sig,
    #             "spect": noise_field_f,
    #         }

    # return received_noise


## Left overs ##
# # Check rcv_sig
# plt.figure()
# rcv_sig_xr.sel(r=r_src, z=z_src, idx_rcv=0, method="nearest").plot()
# plt.savefig("test.png")

# src_spectrum *= np.exp(1j * 2 * np.pi * f * delay_correction)

# Check the effect of interpolation on a single tf
# plt.figure()
# plt.plot(
#     tf.f,
#     np.abs(tf.sel(r=r_rcv, z=z_rcv, method="nearest")).values,
#     label="ref",
#     marker="o",
# )
# plt.plot(
#     tf_interp.f,
#     np.abs(tf_interp.sel(r=r_rcv, z=z_rcv, method="nearest")).values,
#     label="interp",
#     marker="+",
# )
# plt.legend()

# plt.figure()
# plt.plot(
#     tf.f,
#     np.angle(tf.sel(r=r_rcv, z=z_rcv, method="nearest")),
#     label="ref",
#     marker="o",
# )
# plt.plot(
#     tf_interp.f,
#     np.angle(tf_interp.sel(r=r_rcv, z=z_rcv, method="nearest")),
#     label="interp",
#     marker="+",
# )
# plt.legend()

# plt.show()
# plt.savefig("test.png")

# # Derive psd
# psd = scipy.signal.welch(s, fs=fs, nperseg=2**12, noverlap=2**11)

# received_signal = {
#     "t": t,
#     "src": s,
#     "f": f,
#     "n_rcv": n_rcv,
#     "spect": src_spectrum,
#     "psd": psd,
#     "std_fi": std_fi,
#     "tau_corr_fi": tau_corr_fi,
#     "f0": f0,
#     "fs": fs,
#     "tau_ir": tau_ir,
# }

# for i in range(n_rcv):
#     pass
# # Get transfert function
# h_kraken = kraken_data[f"rcv{i}"]["h_f"]

# # Received signal spectrum resulting from the convolution of the source signal and the impulse response
# transmited_sig_field_f = h_kraken * src_spectrum
# rcv_sig = np.fft.irfft(transmited_sig_field_f)

# # psd
# psd_rcv = scipy.signal.welch(rcv_sig, fs=fs, nperseg=2**12, noverlap=2**11)

# received_signal[f"rcv{i}"] = {
#     "sig": rcv_sig,
#     "psd": psd_rcv,
#     "spect": transmited_sig_field_f,
# }

# return received_signal

# # Compare to the received signal from the testcase
# plt.figure()
# rcv_sig_src.sel(idx_rcv=0).plot(label="Mfp")
# plt.plot(result["signal"]["t"], result["signal"]["rcv0"]["sig"], label="Testcase")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "rcv_sig_src_vs_testcase.png")
# plt.savefig(fpath)

# plt.figure()
# plt.plot(result["signal"]["t"], result["signal"]["src"], label="mfp")
# plt.plot(t, s, label="testcase")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "src_sig_vs_testcase.png")
# plt.savefig(fpath)

# plt.figure()
# plt.plot(result["signal"]["f"], np.abs(result["signal"]["spect"]), label="testcase")
# plt.plot(result["signal"]["f"], np.abs(src_spectrum), label="mfp")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "src_spect_vs_testcase.png")
# plt.savefig(fpath)

# kraken_data = load_data()
# h_kraken = kraken_data[f"rcv{0}"]["h_f"]

# plt.figure()
# np.abs(tf_interp).sel(r=r_src, z=z_src).plot(label="mfp")
# plt.plot(
#     kraken_data["f"],
#     np.abs(
#         h_kraken,
#     ),
#     label="testcase",
# )
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "tf_interp_vs_testcase.png")
# plt.savefig(fpath)

# plt.figure()
# plt.plot(kraken_data["f"], np.angle(h_kraken), label="testcase")
# plt.plot(tf_interp.f, np.angle(tf_interp.sel(r=r_src, z=z_src)), label="mfp")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "tf_interp_vs_testcase_phase.png")
# plt.savefig(fpath)

# delay = 20.010975166420828
# spect_hand = (
#     src_spectrum * h_kraken * np.exp(1j * 2 * np.pi * kraken_data["f"] * delay)
# )
# plt.figure()
# plt.plot(
#     result["signal"]["f"], np.abs(rcv_spec_rcv_0.sel(r=r_src, z=z_src)), label="mfp"
# )
# plt.plot(result["signal"]["f"], np.abs(spect_hand), label="hand", linestyle="--")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "spect_hand_vs_mfp.png")
# plt.savefig(fpath)

# plt.figure()
# plt.plot(
#     result["signal"]["f"],
#     np.angle(rcv_spec_rcv_0.sel(r=r_src, z=z_src)),
#     label="mfp",
# )
# plt.plot(result["signal"]["f"], np.angle(spect_hand), label="hand", linestyle="--")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "spect_hand_vs_mfp_phase.png")
# plt.savefig(fpath)
# plt.show()

# sig_mano = np.fft.irfft(rcv_spec_rcv_0.sel(r=r_src, z=z_src).values)
# sig_hand = np.fft.irfft(spect_hand)
# plt.figure()
# plt.plot(t, rcv_sig_src.sel(idx_rcv=0), label="mfp")
# plt.plot(result["signal"]["t"], result["signal"]["rcv0"]["sig"], label="testcase")
# plt.plot(t, sig_hand, label="hand")
# plt.plot(t, sig_mano, label="mano")
# plt.legend()
# fpath = os.path.join(ROOT_IMG_TEST, "sig_hand_vs_mfp_vs_testcase.png")
# plt.savefig(fpath)


# # Compare sig and noise to the one used for the testcase
# fig, axs = plt.subplots(3, 2, sharex=True)
# ds_rcv_sig.sel(r=r_src, z=z_src, idx_rcv=4).plot(label="sig", ax=axs[0, 0])
# ds_noise.sel(r=r_src, z=z_src, idx_rcv=4).plot(label="noise", ax=axs[1, 0])
# ds_sig_noise.sel(r=r_src, z=z_src, idx_rcv=4).plot(label="sig+noise", ax=axs[2, 0])
# axs[0, 1].plot(
#     result["signal"]["t"],
#     result["signal"]["rcv4"]["sig"],
#     "r--",
#     label="sig (testcase)",
# )
# axs[1, 1].plot(
#     result["signal"]["t"],
#     result["noise"]["rcv4"]["sig"],
#     "r--",
#     label="noise (testcase)",
# )
# axs[2, 1].plot(
#     result["signal"]["t"],
#     result["signal"]["rcv4"]["sig"] + result["noise"]["rcv4"]["sig"],
#     "r--",
#     label="sig+noise (testcase)",
# )
# for ax in axs.flatten():
#     ax.legend()

# fpath = os.path.join(ROOT_IMG_TEST, "sig_noise_comparison.png")
# plt.savefig(fpath)

# Plot stft for comparison
# fig_props = {"folder_path": ROOT_IMG_TEST}
# plot_rcv_stfts(
#     fig_props,
#     result["signal"]["t"],
#     np.expand_dims(result["signal"]["rcv4"]["sig"], axis=1),
#     np.expand_dims(result["noise"]["rcv4"]["sig"], axis=1),
#     rcv_idx_to_plot=0,
# )

# plot_rcv_stfts(
#     fig_props,
#     ds_rcv_sig.t.values,
#     ds_rcv_sig.sel(r=r_src, z=z_src).values,
#     ds_noise.sel(r=r_src, z=z_src).values,
#     rcv_idx_to_plot=4,
# )

# plt.show()

# for r_i in ds_tf.r.values:
#     for z_i in ds_tf.z.values:
#         rcv_sig = ds_sig_noise.sel(r=r_i, z=z_i)
#         rcv_noise = ds_noise.sel(r=r_i, z=z_i)

#         # Estimate RTF using covariance substraction method
#         f_cs, rtf_cs, Rx, Rs, Rv = rtf_covariance_substraction(
#             t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
#         )

# # Estimate RTF using covariance whitening method
# f_cw, rtf_cw, _, _, _ = rtf_covariance_whitening(
#     t, rcv_sig, rcv_noise, nperseg=nperseg, noverlap=noverlap
# )

# # Check snr at src pos
# snr = 10 * np.log10(ds_rcv_sig.var(dim=["t"]) / ds_noise.var(dim=["t"]))
# print(f"SNR at source position: {snr.sel(r=r_src, z=z_src, idx_rcv=0).values}")

# plt.figure()
# snr.sel(idx_rcv=0).plot(
#     yincrease=False,
#     vmin=-15,
#     vmax=15,
#     cmap="bwr",
#     cbar_kwargs={"label": "SNR [dB]"},
# )
# fpath = os.path.join(ROOT_IMG_TEST, "snr_rcv0.png")
# plt.savefig(fpath)

# plt.figure()
# ds_noise.sel(idx_rcv=0).var(dim=["t"]).plot(yincrease=False)
# fpath = os.path.join(ROOT_IMG_TEST, "noise_var_rcv0.png")
# plt.savefig(fpath)
