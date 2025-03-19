#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_waveguide.py
@Time    :   2025/01/15 13:55:32
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Test case following Zhang et al. 2023
Zhang, T., Zhou, D., Cheng, L., & Xu, W. (2023). Correlation-based passive localization: Linear system modeling and sparsity-aware optimization.
The Journal of the Acoustical Society of America, 154(1), 295–306. https://doi.org/10.1121/10.0020154
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr

# import pandas as pd
import matplotlib.pyplot as plt

# from time import time
from misc import cast_matrix_to_target_shape
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_DATA,
    ROOT_IMG,
    MIN_VAL_LOG,
    USE_TEX,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    estimate_msr,
    init_dr_file,
    get_subarrays,
    init_msr_file,
    get_rcv_couples,
    get_array_label,
    build_subarrays_args,
    load_msr_rmse_res_subarrays,
)

from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    study_perf_vs_snr,
    check_rtf_features,
    check_gcc_features,
    plot_study_zhang2023,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    build_features_from_time_signal,
)

from publication.PublicationFigure import PubFigure

PubFigure(ticks_fontsize=22, use_tex=USE_TEX)


def process_localisation_zhang2023(
    ds,
    folder,
    nf=10,
    freq_draw_method="random",
    data_fname=None,
    rcv_in_fullarray=None,
    antenna_type="zhang",
    debug=False,
):
    # Load params
    _, _, _, grid, _, _ = params(debug=debug, antenna_type=antenna_type)

    # Define folder to store data
    root_data = os.path.join(ROOT_DATA, folder)
    if not os.path.exists(root_data):
        os.makedirs(root_data)

    # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
    # Match field processing #
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 3,
        "unit": "deg",
        "apply_mean": True,
    }

    # Select a few frequencies
    if (
        freq_draw_method == "random"
    ):  # Same option as used by Zhang et al 2023 yet results (especially the msr are not reproductible from one run to another with the same input dataset)
        f_loc_rtf = np.random.choice(ds.f_rtf.values, nf)
        f_loc_gcc = np.random.choice(ds.f_gcc.values, nf)
    elif (
        freq_draw_method == "equally_spaced"
    ):  # Reproductible option used for msr study
        idx_f_loc = np.linspace(0, ds.sizes["f_rtf"] - 1, nf, dtype=int)
        f_loc_rtf = ds.f_rtf.values[idx_f_loc]
        idx_f_loc = np.linspace(0, ds.sizes["f_gcc"] - 1, nf, dtype=int)
        f_loc_gcc = ds.f_gcc.values[idx_f_loc]

    ds = ds.sel(f_rtf=f_loc_rtf)
    ds = ds.sel(f_gcc=f_loc_gcc)
    df_gcc = np.diff(ds.f_gcc.values)[0]

    # d_gcc_fullarray = []

    # Restrict the dataset to the receivers of interest
    if rcv_in_fullarray is None:
        rcv_in_fullarray = ds.idx_rcv.values

    # Select receivers to build the full array
    ds_fa = ds.sel(idx_rcv=rcv_in_fullarray).sel(idx_rcv_ref=rcv_in_fullarray)

    # Build full array gcc with all required couples
    rcv_couples_fa = get_rcv_couples(idx_receivers=ds_fa.idx_rcv.values)

    ###### Two sensor pairs ######
    # # Select receivers to build the sub-array
    # if (
    #     len(rcv_in_fullarray) < ds.sizes["idx_rcv"]
    # ):  # Not all receivers used in the full array
    #     rcv_couples_sa = rcv_couples_fa[0:3]  # First three couples of the full array

    # else:
    #     # Use couples defined in Zhang et al. 2023
    #     rcv_couples_sa = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    # rcv_couples_sa = rcv_couples_fa[0:3]
    rcv_couples_sa = rcv_couples_fa

    for rcv_cpl in rcv_couples_sa:
        i_ref = rcv_cpl[0]

        ## RTF ##
        # Extract data corresponding to the two-sensor pair rcv_cpl
        ds_cpl_rtf = ds.sel(idx_rcv_ref=i_ref, idx_rcv=rcv_cpl)

        rtf_grid = ds_cpl_rtf.rtf_real.values + 1j * ds_cpl_rtf.rtf_imag.values
        rtf_event = (
            ds_cpl_rtf.rtf_event_real.values + 1j * ds_cpl_rtf.rtf_event_imag.values
        )

        theta = dist_func(rtf_event, rtf_grid, **dist_kwargs)

        # Add theta to dataset
        ds_cpl_rtf["theta"] = (["x", "y"], theta)

        # Normalize
        d_rtf = normalize_metric_contrast(-ds_cpl_rtf.theta)

        # Convert to dB
        d_rtf = d_rtf.values
        d_rtf[d_rtf == 0] = MIN_VAL_LOG
        d_rtf = 10 * np.log10(d_rtf)
        ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf)

        ## GCC ##
        ds_cpl_gcc = ds.sel(idx_rcv_ref=rcv_cpl[0], idx_rcv=rcv_cpl[1])

        gcc_grid = ds_cpl_gcc.gcc_real.values + 1j * ds_cpl_gcc.gcc_imag.values
        gcc_event = (
            ds_cpl_gcc.gcc_event_real.values + 1j * ds_cpl_gcc.gcc_event_imag.values
        )

        # Cast gcc_event to the same shape as gcc_grid
        gcc_event = cast_matrix_to_target_shape(
            gcc_event, gcc_grid.shape
        )  # TODO might need to fix a bug for nf=50

        # Build cross corr (Equation (8) in Zhang et al. 2023)
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_event) * df_gcc, axis=0)
        d_gcc = np.abs(
            np.sum(gcc_grid * np.conj(gcc_event) * df_gcc, axis=0)
        )  # TODO : check if ok to use module
        # d_gcc = d_gcc / np.max(d_gcc)

        # Normalize
        d_gcc = normalize_metric_contrast(d_gcc)

        # Convert to dB
        d_gcc = d_gcc
        d_gcc[d_gcc == 0] = MIN_VAL_LOG
        d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

        # Add d to dataset
        ds_cpl_gcc["d_gcc"] = (["x", "y"], d_gcc)

        # Store d_gcc for full array incoherent processing
        # d_gcc_fullarray.append(d_gcc)

        # Build dataset to be saved as netcdf
        ds_cpl = xr.Dataset(
            data_vars=dict(
                theta_rtf=(["x", "y"], ds_cpl_rtf.theta.values),
                d_rtf=(["x", "y"], ds_cpl_rtf.d_rtf.values),
                d_gcc=(["x", "y"], ds_cpl_gcc.d_gcc.values),
            ),
            coords={
                "x": ds.x.values,
                "y": ds.y.values,
            },
            attrs={
                "idx_rcv": rcv_cpl,
                "snr": ds.attrs["snr"],
            },
        )

        # Add attrs to dataarrays
        for key in ["theta_rtf", "d_rtf", "d_gcc"]:
            ds_cpl[key].attrs["snr"] = ds_cpl.attrs["snr"]
            ds_cpl[key].attrs["idx_rcv"] = ds_cpl.attrs["idx_rcv"]

        # Save dataset
        if data_fname is None:
            data_fname_cpl = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
        else:
            data_fname_cpl = f"{data_fname}_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"

        fpath = os.path.join(
            root_data,
            # f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
            data_fname_cpl,
        )
        ds_cpl.to_netcdf(fpath)
        ds_cpl.close()

    ###### Full array ######
    d_gcc_fullarray = []

    for rcv_cpl in rcv_couples_fa:
        i_ref = rcv_cpl[0]

        ## GCC ##
        ds_cpl_gcc = ds.sel(idx_rcv_ref=rcv_cpl[0], idx_rcv=rcv_cpl[1])

        gcc_grid = ds_cpl_gcc.gcc_real.values + 1j * ds_cpl_gcc.gcc_imag.values
        gcc_event = (
            ds_cpl_gcc.gcc_event_real.values + 1j * ds_cpl_gcc.gcc_event_imag.values
        )

        # Cast gcc_event to the same shape as gcc_grid
        gcc_event = cast_matrix_to_target_shape(
            gcc_event, gcc_grid.shape
        )  # TODO might need to fix a bug for nf=50

        # Build cross corr (Equation (8) in Zhang et al. 2023)
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_event) * df_gcc, axis=0)
        d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_event) * df_gcc, axis=0))
        # d_gcc = d_gcc / np.max(d_gcc)

        # # Normalize
        d_gcc = normalize_metric_contrast(d_gcc)

        # # Convert to dB
        # d_gcc = d_gcc
        # d_gcc[d_gcc == 0] = MIN_VAL_LOG
        # d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

        # Store d_gcc for full array incoherent processing
        d_gcc_fullarray.append(d_gcc)

    ## RTF ##
    # Select reference receiver (by default the first receiver of the array is selected)
    i_ref = rcv_in_fullarray[0]
    ds_fa_rtf = ds_fa.sel(idx_rcv_ref=i_ref)

    rtf_grid = ds_fa_rtf.rtf_real.values + 1j * ds_fa_rtf.rtf_imag.values
    rtf_event = ds_fa_rtf.rtf_event_real.values + 1j * ds_fa_rtf.rtf_event_imag.values

    theta = dist_func(rtf_event, rtf_grid, **dist_kwargs)

    # Add theta to dataset
    ds_fa_rtf["theta"] = (["x", "y"], theta)

    # # Convert theta to a metric between -1 and 1
    # theta_inv = (
    #     theta_max - ds_fa_rtf.theta
    # )  # So that the source position is the maximum value
    # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

    d_rtf = normalize_metric_contrast(-ds_fa_rtf.theta)  # q in [0, 1]

    #  Replace 0 by 1e-5 to avoid log(0) in dB conversion
    d_rtf = d_rtf.values
    d_rtf[d_rtf == 0] = MIN_VAL_LOG
    d_rtf = 10 * np.log10(d_rtf)  # Convert to dB
    ds_fa_rtf["d_rtf"] = (["x", "y"], d_rtf)

    ## GCC ##
    d_gcc_fullarray = np.array(d_gcc_fullarray)
    d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

    # Normalize
    # d_gcc_fullarray = normalize_metric_contrast(d_gcc_fullarray)

    # # Convert to dB
    # d_gcc = d_gcc
    # d_gcc[d_gcc == 0] = MIN_VAL_LOG
    # d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

    # # Convert back to linear scale before computing the mean
    # d_gcc_fullarray = 10 ** (d_gcc_fullarray / 10)
    # d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

    # Convert to dB
    d_gcc_fullarray[d_gcc_fullarray == 0] = MIN_VAL_LOG
    d_gcc_fullarray = 10 * np.log10(d_gcc_fullarray)
    # d_gcc_fullarray = d_gcc_fullarray

    # Build dataset to be saved as netcdf
    ds_fullarray = xr.Dataset(
        data_vars=dict(
            theta_rtf=(["x", "y"], ds_fa_rtf.theta.values),
            d_rtf=(["x", "y"], ds_fa_rtf.d_rtf.values),
            d_gcc=(["x", "y"], d_gcc_fullarray),
        ),
        coords={
            "x": ds.x.values,
            "y": ds.y.values,
        },
        attrs={
            "idx_rcv": ds_fa.idx_rcv.values,
            "snr": ds.attrs["snr"],
        },
    )

    # Add attrs to dataarrays
    for key in ["theta_rtf", "d_rtf", "d_gcc"]:
        ds_fullarray[key].attrs["snr"] = ds_fullarray.attrs["snr"]
        ds_fullarray[key].attrs["idx_rcv"] = ds_fullarray.attrs["idx_rcv"]

    # Save dataset
    array_label = get_array_label(rcv_in_fullarray)
    if data_fname is None:
        data_fname_fa = (
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray_{array_label}.nc"
        )
    else:
        data_fname_fa = f"{data_fname}_fullarray_{array_label}.nc"

    fpath = os.path.join(
        root_data,
        # f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
        data_fname_fa,
    )
    ds_fullarray.to_netcdf(fpath)
    ds_fullarray.close()


def process_all_snr(
    snrs,
    n_monte_carlo,
    dx=20,
    dy=20,
    nf=100,
    run_mode="a",
    subarrays_list=None,
    freq_draw_method="equally_spaced",
    antenna_type="zhang",
    debug=False,
    verbose=False,
    check=False,
    plot_args={},
):

    # Load params
    _, receivers, source, _, _, _ = params(debug=debug, antenna_type=antenna_type)
    if subarrays_list is None:
        subarrays_list = np.arange(len(receivers["x"]))  # Fullarray

    folder = f"from_signal_dx{dx}m_dy{dy}m"

    # subarrays_args = {index: {} for index in range(len(subarrays_list))}
    subarrays_args = build_subarrays_args(subarrays_list)

    init_dr_file(folder, run_mode, subarrays_args)
    init_msr_file(folder, run_mode, subarrays_args)

    for snr in snrs:
        subfolder = os.path.join(folder, f"snr_{snr:.1f}dB")
        subfolder_fullpath = os.path.join(ROOT_DATA, subfolder)
        if not os.path.exists(subfolder_fullpath):
            os.makedirs(subfolder_fullpath)

        # List existing files in subfolder
        snr_files = os.listdir(subfolder_fullpath)
        snr_files = [sfile for sfile in snr_files if "mc" in sfile]

        if run_mode == "a":  # Append mode -> do not overwrite existing values
            # Keep only fullarray files
            fa_snr_files = [sfile for sfile in snr_files if "fullarray" in sfile]
            # Parse i_mc and get max
            i_mcs = [int(sfile.split("_")[5].split("mc")[1]) for sfile in fa_snr_files]
            # Set offset
            i_mc_offset = max(i_mcs) + 1 if i_mcs else 0

        elif run_mode == "w":  # Write mode -> overwrite existing files
            # Remove file in subfolder
            for sfile in snr_files:
                os.remove(os.path.join(subfolder_fullpath, sfile))
            # Set i_mc_offset to 0
            i_mc_offset = 0

        if verbose:
            print(
                f"Start processing snr = {snr} dB (i_mc from {i_mc_offset} to {n_monte_carlo + i_mc_offset-1})"
            )

        plot_study = True
        # Run simulation n_monte_carlo times at the same snr to derive the mean MSR
        for i_mc in range(i_mc_offset, n_monte_carlo + i_mc_offset):

            if verbose:
                print(f"i_mc = {i_mc}")

            # Run simulation (one simulation = 1 generation of noise)
            # t0 = time()
            # build_features_from_time_signal(snr)
            build_features_from_time_signal(
                snr_dB=snr,
                debug=debug,
                check=check,
                use_welch_estimator=True,
                antenna_type=antenna_type,
                verbose=verbose,
            )

            # elasped_time = time() - t0
            # print(f"Features built (elapsed time = {np.round(elasped_time,0)}s)")

            # Load results
            fpath = os.path.join(
                ROOT_DATA, f"zhang_output_from_signal_dx{dx}m_dy{dy}m_snr{snr:.1f}dB.nc"
            )
            ds = xr.open_dataset(fpath)

            # Process results
            data_rootname = f"loc_zhang_dx{dx}m_dy{dy}m_snr{snr}dB_mc{i_mc}"

            # Loop over subarrays of interest
            for sa_idx, sa_item in subarrays_args.items():

                rcv_in_fullarray = sa_item["idx_rcv"]
                process_localisation_zhang2023(
                    ds,
                    subfolder,
                    nf,
                    freq_draw_method,
                    data_fname=data_rootname,
                    rcv_in_fullarray=rcv_in_fullarray,
                    antenna_type=antenna_type,
                    debug=debug,
                )

                # Plot results
                if plot_study:
                    plot_study_zhang2023(
                        subfolder,
                        data_fname=data_rootname,
                        debug=debug,
                        antenna_type=antenna_type,
                        rcv_in_fullarray=rcv_in_fullarray,
                        plot_args=plot_args,
                    )
                    plt.close("all")

                # Load processed surface and derive msr
                # array_label = get_array_label(rcv_in_fullarray)
                fpath = os.path.join(
                    subfolder_fullpath,
                    f"{data_rootname}_fullarray_{sa_item['array_label']}.nc",
                )
                ds_fa = xr.open_dataset(fpath)

                msr, pos_hat = estimate_msr(ds_fa, plot=False)
                ds_fa.close()

                # Store MSR and DR
                msr_txt_filepath = sa_item["msr_filepath"]
                dr_txt_filepath = sa_item["dr_pos_filepath"]
                # MSR
                # msr_gcc.append(msr["d_gcc"])
                # msr_rtf.append(msr["d_rtf"])

                # Save to text file for further analysis
                newline = f"{snr} {i_mc} {msr['d_gcc']:.2f} {msr['d_rtf']:.2f}\n"
                with open(msr_txt_filepath, "a") as f:
                    f.write(newline)

                # Position error
                delta_r_gcc = np.sqrt(
                    (pos_hat["d_gcc"]["x"] - source["x"]) ** 2
                    + (pos_hat["d_gcc"]["y"] - source["y"]) ** 2
                )
                delta_r_rtf = np.sqrt(
                    (pos_hat["d_rtf"]["x"] - source["x"]) ** 2
                    + (pos_hat["d_rtf"]["y"] - source["y"]) ** 2
                )
                # dr_pos_gcc.append(delta_r_gcc)
                # dr_pos_rtf.append(delta_r_rtf)

                # Save to text file for further analysis
                newline = f"{snr} {i_mc} {delta_r_gcc:.2f} {delta_r_rtf:.2f}\n"
                with open(dr_txt_filepath, "a") as f:
                    f.write(newline)

            plot_study = False

            # Check RTF estimation at a few grid points
            # check_rtf_features(ds_rtf_cs=ds, folder=subfolder)
            ds.close()

        study_perf_vs_snr(subarrays_list=subarrays_list)


def replay_all_snr(
    snrs,
    dx=20,
    dy=20,
):

    # Load params
    _, _, source, _, _, _ = params()

    folder = f"from_signal_dx{dx}m_dy{dy}m"

    dr_pos_gcc = []
    dr_pos_rtf = []
    dr_txt_filepath = os.path.join(ROOT_DATA, folder, "dr_pos_snr.txt")
    header_line = "snr i_mc dr_gcc dr_rtf\n"
    with open(dr_txt_filepath, "w") as f:
        f.write(header_line)

    msr_gcc = []
    msr_rtf = []

    msr_txt_filepath = os.path.join(ROOT_DATA, folder, "msr_snr.txt")
    header_line = "snr i_mc d_gcc d_rtf\n"
    with open(msr_txt_filepath, "w") as f:
        f.write(header_line)

    for snr in snrs:
        subfolder = os.path.join(folder, f"snr_{snr}dB")
        subfolder_fullpath = os.path.join(ROOT_DATA, subfolder)
        # List available files in subfolder
        snr_files = [
            file for file in os.listdir(subfolder_fullpath) if "fullarray" in file
        ]
        # for i_mc in range(i_mc_offset, n_monte_carlo + i_mc_offset):
        for filename in snr_files:
            file_fullpath = os.path.join(subfolder_fullpath, filename)
            imc_str = filename.split("_")[-2]
            i_mc = int(imc_str[2:])

            # Load processed surface and derive msr
            ds_fa = xr.open_dataset(file_fullpath)

            msr, pos_hat = estimate_msr(ds_fa, plot=False)
            ds_fa.close()

            # MSR
            msr_gcc.append(msr["d_gcc"])
            msr_rtf.append(msr["d_rtf"])

            # Save to text file for further analysis
            newline = f"{snr} {i_mc} {msr['d_gcc']:.2f} {msr['d_rtf']:.2f}\n"
            with open(msr_txt_filepath, "a") as f:
                f.write(newline)

            # Position error
            delta_r_gcc = np.sqrt(
                (pos_hat["d_gcc"]["x"] - source["x"]) ** 2
                + (pos_hat["d_gcc"]["y"] - source["y"]) ** 2
            ).values
            delta_r_rtf = np.sqrt(
                (pos_hat["d_rtf"]["x"] - source["x"]) ** 2
                + (pos_hat["d_rtf"]["y"] - source["y"]) ** 2
            ).values
            dr_pos_gcc.append(delta_r_gcc)
            dr_pos_rtf.append(delta_r_rtf)

            # Save to text file for further analysis
            newline = f"{snr} {i_mc} {delta_r_gcc:.2f} {delta_r_rtf:.2f}\n"
            with open(dr_txt_filepath, "a") as f:
                f.write(newline)

        study_perf_vs_snr()


def study_perf_vs_subarrays(subarrays_list, snrs, var="std", dx=20, dy=20):

    folder = "from_signal_dx20m_dy20m"
    root_img = os.path.join(ROOT_IMG, folder, "perf_vs_subarrays")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    # Build sub arrays labels
    subarray_sizes = [len(sa) for sa in subarrays_list]
    subarray_sizes_unique = np.unique(subarray_sizes)

    msr, dr, rmse = load_msr_rmse_res_subarrays(subarrays_list)

    for snr in snrs:
        # Plot RMSE vs nr

        rmse_rtf = []
        rmse_gcc = []
        dr_rtf = []
        dr_gcc = []
        msr_rtf = []
        msr_gcc = []

        # Group by number of receivers in subarray
        for sa_size in subarray_sizes_unique:
            idx_required_size_sa = np.where(subarray_sizes == sa_size)[0]
            key_required_size_sa = [
                list(msr.keys())[idx] for idx in idx_required_size_sa
            ]
            # Get rmse, dr and msr for subarrays of size sa_size
            rmse_for_required_size_sa = [
                rmse[key].loc[snr] for key in key_required_size_sa
            ]
            dr_mu_for_required_size_sa = [
                dr[key].loc[snr][["rtf_mean", "dcf_mean"]]
                for key in key_required_size_sa
            ]
            msr_mu_for_required_size_sa = [
                msr[key].loc[snr][["rtf_mean", "dcf_mean"]]
                for key in key_required_size_sa
            ]

            # RMSE
            rmse_gcc_for_required_size_sa = [
                rmse_for_required_size_sa[i]["dcf"]
                for i in range(len(rmse_for_required_size_sa))
            ]
            rmse_rtf_for_required_size_sa = [
                rmse_for_required_size_sa[i]["rtf"]
                for i in range(len(rmse_for_required_size_sa))
            ]
            rmse_gcc.append(rmse_gcc_for_required_size_sa)
            rmse_rtf.append(rmse_rtf_for_required_size_sa)

            # DR
            dr_gcc_for_required_size_sa = [
                # dr["dr_gcc"].loc[snr] for dr in dr_mu_for_required_size_sa
                dr_mu_for_required_size_sa[i]["dcf_mean"]
                for i in range(len(dr_mu_for_required_size_sa))
            ]
            dr_rtf_for_required_size_sa = [
                dr_mu_for_required_size_sa[i]["rtf_mean"]
                for i in range(len(dr_mu_for_required_size_sa))
            ]
            dr_gcc.append(dr_gcc_for_required_size_sa)
            dr_rtf.append(dr_rtf_for_required_size_sa)

            # MSR
            msr_gcc_for_required_size_sa = [
                msr_mu_for_required_size_sa[i]["dcf_mean"]
                for i in range(len(msr_mu_for_required_size_sa))
            ]
            msr_rtf_for_required_size_sa = [
                msr_mu_for_required_size_sa[i]["rtf_mean"]
                for i in range(len(msr_mu_for_required_size_sa))
            ]
            msr_gcc.append(msr_gcc_for_required_size_sa)
            msr_rtf.append(msr_rtf_for_required_size_sa)

        # Derive mean of each metric per subarray size
        rmse_gcc_mean = np.array([np.mean(rmse) for rmse in rmse_gcc])
        rmse_rtf_mean = np.array([np.mean(rmse) for rmse in rmse_rtf])
        dr_gcc_mean = np.array([np.mean(dr) for dr in dr_gcc])
        dr_rtf_mean = np.array([np.mean(dr) for dr in dr_rtf])
        msr_gcc_mean = np.array([np.mean(msr) for msr in msr_gcc])
        msr_rtf_mean = np.array([np.mean(msr) for msr in msr_rtf])

        # Derive min and max of each metric per subarray size
        rmse_gcc_min = np.array([np.min(rmse) for rmse in rmse_gcc])
        rmse_rtf_min = np.array([np.min(rmse) for rmse in rmse_rtf])
        dr_gcc_min = np.array([np.min(dr) for dr in dr_gcc])
        dr_rtf_min = np.array([np.min(dr) for dr in dr_rtf])
        msr_gcc_min = np.array([np.min(msr) for msr in msr_gcc])
        msr_rtf_min = np.array([np.min(msr) for msr in msr_rtf])

        rmse_gcc_max = np.array([np.max(rmse) for rmse in rmse_gcc])
        rmse_rtf_max = np.array([np.max(rmse) for rmse in rmse_rtf])
        dr_gcc_max = np.array([np.max(dr) for dr in dr_gcc])
        dr_rtf_max = np.array([np.max(dr) for dr in dr_rtf])
        msr_gcc_max = np.array([np.max(msr) for msr in msr_gcc])
        msr_rtf_max = np.array([np.max(msr) for msr in msr_rtf])

        # Derive std for each metric per subarray size
        rmse_gcc_std = np.array([np.std(rmse) for rmse in rmse_gcc])
        rmse_rtf_std = np.array([np.std(rmse) for rmse in rmse_rtf])
        dr_gcc_std = np.array([np.std(dr) for dr in dr_gcc])
        dr_rtf_std = np.array([np.std(dr) for dr in dr_rtf])
        msr_gcc_std = np.array([np.std(msr) for msr in msr_gcc])
        msr_rtf_std = np.array([np.std(msr) for msr in msr_rtf])

        # Plot RMSE vs subarray size
        plt.figure(figsize=(8, 6))
        plt.plot(subarray_sizes_unique, rmse_gcc_mean, "o-", label="DCF")
        if var == "minmax":
            plt.fill_between(
                subarray_sizes_unique, rmse_gcc_min, rmse_gcc_max, alpha=0.2
            )
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                rmse_gcc_mean - rmse_gcc_std,
                rmse_gcc_mean + rmse_gcc_std,
                alpha=0.2,
            )

        plt.plot(subarray_sizes_unique, rmse_rtf_mean, "o-", label="RTF")
        if var == "minmax":
            plt.fill_between(
                subarray_sizes_unique, rmse_rtf_min, rmse_rtf_max, alpha=0.2
            )
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                rmse_rtf_mean - rmse_rtf_std,
                rmse_rtf_mean + rmse_rtf_std,
                alpha=0.2,
            )
        plt.xlabel("Number of receivers in subarray")
        plt.ylabel("RMSE [m]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"rmse_subarrays_snr{snr}_{var}.png")
        plt.savefig(fpath, dpi=300)

        # Plot DR vs subarray size
        plt.figure(figsize=(8, 6))
        plt.plot(subarray_sizes_unique, dr_gcc_mean, "o-", label="DCF")
        if var == "minmax":
            plt.fill_between(subarray_sizes_unique, dr_gcc_min, dr_gcc_max, alpha=0.2)
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                dr_gcc_mean - dr_gcc_std,
                dr_gcc_mean + dr_gcc_std,
                alpha=0.2,
            )

        plt.plot(subarray_sizes_unique, dr_rtf_mean, "o-", label="RTF")
        if var == "minmax":
            plt.fill_between(subarray_sizes_unique, dr_rtf_min, dr_rtf_max, alpha=0.2)
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                dr_rtf_mean - dr_rtf_std,
                dr_rtf_mean + dr_rtf_std,
                alpha=0.2,
            )

        plt.xlabel("Number of receivers in subarray")
        plt.ylabel("DR [m]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"dr_subarrays_snr{snr}_{var}.png")
        plt.savefig(fpath, dpi=300)

        # Plot MSR vs subarray size
        plt.figure(figsize=(8, 6))
        plt.plot(subarray_sizes_unique, msr_gcc_mean, "o-", label="DCF")
        if var == "minmax":
            plt.fill_between(subarray_sizes_unique, msr_gcc_min, msr_gcc_max, alpha=0.2)
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                msr_gcc_mean - msr_gcc_std,
                msr_gcc_mean + msr_gcc_std,
                alpha=0.2,
            )

        plt.plot(subarray_sizes_unique, msr_rtf_mean, "o-", label="RTF")
        if var == "minmax":
            plt.fill_between(subarray_sizes_unique, msr_rtf_min, msr_rtf_max, alpha=0.2)
        elif var == "std":
            plt.fill_between(
                subarray_sizes_unique,
                msr_rtf_mean - msr_rtf_std,
                msr_rtf_mean + msr_rtf_std,
                alpha=0.2,
            )

        plt.xlabel("Number of receivers in subarray")
        plt.ylabel("MSR [dB]")
        plt.title(f"SNR = {snr} dB")
        plt.legend()

        fpath = os.path.join(root_img, f"msr_subarrays_snr{snr}_{var}.png")
        plt.savefig(fpath, dpi=300)


if __name__ == "__main__":

    nf = 100
    dx, dy = 20, 20
    # antenna_type = "random"
    # debug = True
    antenna_type = "zhang"
    debug = False
    event_stype = "wn"

    # # To run if debug or antenna type changes
    # from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    #     grid_dataset,
    #     build_signal,
    # )

    # grid_dataset(debug=debug, antenna_type=antenna_type)
    # build_signal(debug=debug, event_stype=event_stype, antenna_type=antenna_type)

    # subarrays_list = None   # Only consider full array
    subarrays_list = [[0, 1, 4]]  # s1 s2 s5
    # rcv_in_fullarray = [0, 1, 4]      # s1 s2 s5

    # # All possible subarrays with 3, 4, 5 or 6 receivers
    # subarrays_list = []
    # for i in [3, 4, 5, 6]:
    #     subarrays_list += list(get_subarrays(nr_fullarray=6, nr_subarray=i))
    # subarrays_list = [
    #     [0, 1],
    #     [0, 1, 2],
    #     [0, 1, 2, 3],
    #     [0, 1, 2, 3, 4],
    #     [0, 1, 2, 3, 4, 5],
    # ]

    plot_args = {
        "plot_array": True,
        "plot_single_cpl_surf": False,
        "plot_fullarray_surf": False,
        "plot_cpl_surf_comparison": True,
        "plot_fullarray_surf_comparison": True,
        "plot_surf_dist_comparison": False,
        "plot_mainlobe_contour": False,
        "plot_msr_estimation": False,
    }

    # snrs = np.arange(-40, 25, 2.5)
    # # snrs = np.arange(-20, 15, 5)
    # snrs = [15, 20]

    # n_monte_carlo = 1
    # i_mc_offset = 0
    # process_all_snr(
    #     snrs,
    #     n_monte_carlo,
    #     dx=20,
    #     dy=20,
    #     nf=100,
    #     freq_draw_method="equally_spaced",
    #     run_mode="a",
    #     # i_mc_offset=i_mc_offset,
    #     rcv_in_fullarray=rcv_in_fullarray,
    #     antenna_type=antenna_type,
    #     debug=debug,
    #     verbose=True,
    # )

    # snrs = np.arange(-40, 25, 2.5)
    # # Update figures after appearence modifications
    # subarrays_args = build_subarrays_args(subarrays_list)
    # study_perf_vs_snr(subarrays_args=subarrays_args)

    # # snrs = np.arange(-20, 15, 5)
    # from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    #     grid_dataset,
    #     build_signal,
    # )

    # grid_dataset(debug=True)
    # build_signal(debug=True)
    # from dask.distributed import Client, LocalCluster

    # # Launch Dask cluster and client
    # cluster = LocalCluster(
    #     n_workers=4, threads_per_worker=1, memory_limit="3GB")
    # client = Client(cluster)

    # # Print dashboard link
    # print("Dask Dashboard:", client.dashboard_link)

    """
    TEST 1a :
    Jeux de simulations pour la publication JASA EL
    -> 10 simulations (dans un premier temps)
    -> Test de toutes les configurations de sous-arrays possibles
    -> Analyse des résultats :
        -> MSR, RMSE en fonction du SNR avec les deux méthodes
        -> Ajout des maximums et minimum pour chaque nombre de récepteurs afin d'étudier la variabilité en fonction de la configuration des capteurs.

    """
    print("TEST 1a : Study of the performance vs array configuration")

    nf = 100  # Nombre de points fréquentiel pour le calcul des grandeurs signantes (DCF, RTF)
    dx, dy = 20, 20  # Taille des mailles de la grille de recherche
    antenna_type = "zhang"  # Type d'antenne utilisée pour la simulation : antenne hexagonale (a = 250 m)
    debug = True  # Calcul sur une zone réduite pour plus de rapidité (TODO changer pour la publication si la figure est ok)
    event_stype = "wn"  # Signal source à localiser : bruit blanc gaussien

    # Paramètres graphiques pour la génération des figures
    plot_args = {
        "plot_array": True,
        "plot_single_cpl_surf": False,
        "plot_fullarray_surf": False,
        "plot_cpl_surf_comparison": False,
        "plot_fullarray_surf_comparison": True,
        "plot_surf_dist_comparison": False,
        "plot_mainlobe_contour": False,
        "plot_msr_estimation": False,
    }

    # Liste des sous antennes considérées : toutes les sous antennes possibles pour 2, 3, 4, 5 et 6 récepteurs
    subarrays_list = []
    n_rcv = [2, 3, 4, 5, 6]
    for i in n_rcv:
        subarrays_list += list(get_subarrays(nr_fullarray=6, nr_subarray=i))
    print(f"Number of subarrays = {len(subarrays_list)}")
    print("Subarrays list : ", subarrays_list)

    # Liste des SNR considérés
    snrs = [0.5]
    print(f"Number of SNRs = {len(snrs)}")
    print("SNRs : ", snrs)

    # Nombre de simulations à réaliser pour chaque SNR
    n_monte_carlo = 10
    print(f"Number of Monte Carlo simulations / snr / array config = {n_monte_carlo}")

    process_all_snr(
        snrs,
        n_monte_carlo,
        dx=dx,
        dy=dy,
        nf=nf,
        freq_draw_method="equally_spaced",
        run_mode="w",
        subarrays_list=subarrays_list,
        antenna_type=antenna_type,
        debug=debug,
        verbose=True,
        check=True,
        plot_args=plot_args,
    )

    study_perf_vs_subarrays(subarrays_list, snrs)

    print("End of TEST 1a")

    # # Shutdown the client and cluster after computation is done
    # client.close()
    # cluster.close()

    """ Fin du TEST 1a """

    ### STUDY PERF VS ARRAY CONFIG ###
    # snrs = [-15]

    # n_monte_carlo = 10
    # process_all_snr(
    #     snrs,
    #     n_monte_carlo,
    #     dx=20,
    #     dy=20,
    #     nf=100,
    #     freq_draw_method="equally_spaced",
    #     run_mode="w",
    #     # i_mc_offset=i_mc_offset,
    #     subarrays_list=subarrays_list,
    #     antenna_type=antenna_type,
    #     debug=debug,
    #     verbose=True,
    #     plot_args=plot_args,
    # )

    # if subarrays_list is not None:
    #     study_perf_vs_subarrays(subarrays_list, snrs)

    # snrs = [-2.5]

    # n_monte_carlo = 20
    # process_all_snr(
    #     snrs,
    #     n_monte_carlo,
    #     dx=20,
    #     dy=20,
    #     nf=100,
    #     freq_draw_method="equally_spaced",
    #     run_mode="a",
    #     # i_mc_offset=i_mc_offset,
    #     subarrays_list=subarrays_list,
    #     antenna_type=antenna_type,
    #     debug=debug,
    #     verbose=True,
    #     plot_args=plot_args,
    # )
    # study_perf_vs_subarrays(subarrays_list, snrs)

    ### END STUDY ###

    # n_monte_carlo = 10
    # i_mc_offset = 10  # TO start numbering simulations results at 10
    # process_all_snr(
    #     snrs,
    #     n_monte_carlo,
    #     dx=20,
    #     dy=20,
    #     nf=100,
    #     freq_draw_method="equally_spaced",
    #     i_mc_offset=i_mc_offset,
    # )

    # replay_all_snr(snrs=snrs, dx=dx, dy=dy)
    # study_perf_vs_snr()


## Left overs ##


# tc_varin = {
#     "freq": f,
#     "src_depth": z_src,
#     "max_range_m": rmax,
#     "mode_theory": "adiabatic",
#     "flp_n_rcv_z": nz,
#     "flp_rcv_z_min": z_min,
#     "flp_rcv_z_max": z_max,
#     "min_depth": depth,
#     "max_depth": depth,
#     "dr_flp": dr,
#     "nb_modes": 200,
#     "bottom_boundary_condition": "acousto_elastic",
#     "nmedia": 4,
#     "phase_speed_limits": [200, 20000],
#     "bott_hs_properties": bott_hs_properties,
# }
# tc = TestCase1_0(mode="prod", testcase_varin=tc_varin)
# title = "Zhang et al. 2023 test case"
# tc.title = title
# tc.env_dir = os.path.join(ROOT, "tmp")
# tc.update(tc_varin)

# rho = np.array([1.0, 1.75, 2.01, 2.5]) * RHO_W * 1e-3
# c_p = np.array([1500, 1600, 1900, 4650])  # P-wave celerity (m/s)
# a_p = np.array([0, 0.2, 0.06, 0.01])  # Compression wave attenuation (dB/wavelength)
# z = np.array([0, 150, 175, 275])

# medium = KrakenMedium(
#     ssp_interpolation_method="C_linear", z_ssp=z, c_p=c_p, a_p=a_p, rho=rho
# )
# tc.medium = medium
# # Write flp and env files
# tc.write_kraken_files()


# # Load rtf data
# fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
# ds = xr.open_dataset(fpath)

# folder = f"fullsimu_dx{dx}m_dy{dy}m"
# process_localisation_zhang2023(ds, folder, nf=nf)
# plot_study_zhang2023(folder)

# # Check noise component
# fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\zhang_output_from_signal_dx20m_dy20m_snr0dB.nc"
# ds = xr.open_dataset(fpath)

# folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023\from_signal_dx20m_dy20m\snr_0dB"
# # check_rtf_features(ds_rtf_cs=ds, folder=folder)
# check_gcc_features(ds_gcc=ds, folder=folder)

# fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\zhang_output_from_signal_dx20m_dy20m_snr-10dB.nc"
# ds10 = xr.open_dataset(fpath)

# fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\zhang_output_from_signal_dx20m_dy20m_snr-20dB.nc"
# ds20 = xr.open_dataset(fpath)

# fname = f"zhang_library_dx{dx}m_dy{dy}m.nc"
# fpath = os.path.join(ROOT_DATA, fname)
# ds_sig = xr.open_dataset(fpath)

# fs = 1 / ds.t.diff("t").values[0]
# # s = ds.s_e.sel(idx_rcv=0)
# xs = 3900
# ys = 6800
# s = ds.s_l.sel(x=xs, y=ys, method="nearest").sel(idx_rcv=0)
# s10 = ds10.s_l.sel(x=xs, y=ys, method="nearest").sel(idx_rcv=0)
# s20 = ds20.s_l.sel(x=xs, y=ys, method="nearest").sel(idx_rcv=0)
# s_original = ds_sig.s_l.sel(x=xs, y=ys, method="nearest").sel(idx_rcv=0)

# se = ds.s_e.sel(idx_rcv=0)
# se10 = ds10.s_e.sel(idx_rcv=0)
# se20 = ds20.s_e.sel(idx_rcv=0)
# se_original = ds_sig.s_e.sel(idx_rcv=0)

# plt.figure()
# se.plot(x="t", label="snr=0dB")
# se10.plot(x="t", label="snr=-10dB")
# se20.plot(x="t", label="snr=-20dB")
# se_original.plot(x="t", label="original")
# plt.legend()
# plt.savefig("test_noise_se")

# ff, tt, stft = sp.stft(s.values, fs=fs, nperseg=2**8, noverlap=2**7)
# plt.figure()
# # ds.s_e.sel(idx_rcv=0).plot(x="t")
# s.plot(x="t", label="snr=0dB")
# s10.plot(x="t", label="snr=-10dB")
# s_original.plot(x="t", label="original")
# plt.legend()
# plt.savefig("test_noise")

# plt.figure()
# plt.pcolormesh(np.abs(stft))
# plt.savefig("test")
# plt.show()

# # fpath = os.path.join(ROOT_DATA, f"zhang_output_from_signal_dx{dx}m_dy{dy}m.nc")
# fpath = os.path.join(ROOT_DATA, "zhang_output_from_signal_dx20m_dy20m_snr0dB.nc")
# ds = xr.open_dataset(fpath)

# folder = os.path.join(f"from_signal_dx{dx}m_dy{dy}m", "snr_0dB")
# process_localisation_zhang2023(ds, folder, nf=nf, freq_draw_method="equally_spaced")
# plot_study_zhang2023(folder)

# snrs = np.arange(-40, 15, 5)
# print(snrs)
# snrs = [5]
# n_monte_carlo = 2
# i_mc_offset = 8  # TO start numbering simulations results at 10
# process_all_snr(
#     snrs,
#     n_monte_carlo,
#     dx=20,
#     dy=20,
#     nf=100,
#     freq_draw_method="equally_spaced",
#     i_mc_offset=i_mc_offset,
# )

# snrs = [10]
# n_monte_carlo = 10
# i_mc_offset = 0  # TO start numbering simulations results at 10
# process_all_snr(
#     snrs,
#     n_monte_carlo,
#     dx=20,
#     dy=20,
#     nf=100,
#     freq_draw_method="equally_spaced",
#     i_mc_offset=i_mc_offset,
# )
