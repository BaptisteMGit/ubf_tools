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
import pandas as pd

from time import time
from misc import cast_matrix_to_target_shape
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import *
from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    plot_study_zhang2023,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    build_features_from_time_signal,
)

PubFigure(ticks_fontsize=22)


def process_localisation_zhang2023(
    ds, folder, nf=10, freq_draw_method="random", data_fname=None
):
    # Load params
    depth, receivers, source, grid, frequency, _ = params()

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
    df = np.diff(ds.f.values)[0]
    if (
        freq_draw_method == "random"
    ):  # Same option as used by Zhang et al 2023 yet results (especially the msr are not reproductible from one run to another with the same input dataset)
        f_loc = np.random.choice(ds.f.values, nf)
    elif (
        freq_draw_method == "equally_spaced"
    ):  # Reproductible option used for msr study
        idx_f_loc = np.linspace(0, ds.sizes["f"] - 1, nf, dtype=int)
        f_loc = ds.f.values[idx_f_loc]

    ds = ds.sel(f=f_loc)

    # d_gcc_fullarray = []
    ###### Two sensor pairs ######
    # Select receivers to build the sub-array
    rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    for rcv_cpl in rcv_couples:
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
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0)
        d_gcc = np.abs(
            np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0)
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
        )

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

    # Build full array gcc with all couples
    rcv_couples = []
    for i in ds.idx_rcv.values:
        for j in ds.idx_rcv.values:
            if i > j:
                rcv_couples.append([i, j])
    rcv_couples = np.array(rcv_couples)

    # rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6
    for rcv_cpl in rcv_couples:
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
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0)
        d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0))
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
    i_ref = 0
    # Extract data corresponding to the two-sensor pair rcv_cpl
    ds_cpl_rtf = ds.sel(idx_rcv_ref=i_ref)

    rtf_grid = ds_cpl_rtf.rtf_real.values + 1j * ds_cpl_rtf.rtf_imag.values
    rtf_event = ds_cpl_rtf.rtf_event_real.values + 1j * ds_cpl_rtf.rtf_event_imag.values

    theta = dist_func(rtf_event, rtf_grid, **dist_kwargs)

    # Add theta to dataset
    ds_cpl_rtf["theta"] = (["x", "y"], theta)
    # # Convert theta to a metric between -1 and 1
    # theta_inv = (
    #     theta_max - ds_cpl_rtf.theta
    # )  # So that the source position is the maximum value
    # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

    d_rtf = normalize_metric_contrast(-ds_cpl_rtf.theta)  # q in [0, 1]

    #  Replace 0 by 1e-5 to avoid log(0) in dB conversion
    d_rtf = d_rtf.values
    d_rtf[d_rtf == 0] = MIN_VAL_LOG
    d_rtf = 10 * np.log10(d_rtf)  # Convert to dB
    ds_cpl_rtf["d_rtf"] = (["x", "y"], d_rtf)

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
            theta_rtf=(["x", "y"], ds_cpl_rtf.theta.values),
            d_rtf=(["x", "y"], ds_cpl_rtf.d_rtf.values),
            d_gcc=(["x", "y"], d_gcc_fullarray),
        ),
        coords={
            "x": ds.x.values,
            "y": ds.y.values,
        },
    )

    # Save dataset
    if data_fname is None:
        data_fname_fa = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc"
    else:
        data_fname_fa = f"{data_fname}_fullarray.nc"

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
    freq_draw_method="equally_spaced",
    i_mc_offset=0,
):

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

    folder = f"from_signal_dx{dx}m_dy{dy}m"

    dr_pos_gcc = []
    dr_pos_rtf = []
    dr_txt_filepath = os.path.join(ROOT_DATA, folder, "dr_pos_snr.txt")

    if not os.path.exists(dr_txt_filepath):  # To avoid writting over existing file
        header_line = "snr i_mc dr_gcc dr_rtf\n"
        with open(dr_txt_filepath, "w") as f:
            f.write(header_line)

    msr_gcc = []
    msr_rtf = []

    msr_txt_filepath = os.path.join(ROOT_DATA, folder, "msr_snr.txt")

    if not os.path.exists(msr_txt_filepath):  # To avoid writting over existing file
        header_line = "snr i_mc d_gcc d_rtf\n"
        with open(msr_txt_filepath, "w") as f:
            f.write(header_line)

    for snr in snrs:
        subfolder = os.path.join(folder, f"snr_{snr}dB")
        if not os.path.exists(os.path.join(ROOT_DATA, subfolder)):
            os.makedirs(os.path.join(ROOT_DATA, subfolder))

        # Run simulation n_monte_carlo times at the same snr to derive the mean MSR
        for i_mc in range(i_mc_offset, n_monte_carlo + i_mc_offset):
            # Run simulation (one simulation = 1 generation of noise)
            print(f"Start building loc features for snr = {snr}dB ...")
            t0 = time()
            build_features_from_time_signal(snr)
            elasped_time = time() - t0
            print(f"Features built (elapsed time = {np.round(elasped_time,0)}s)")

            # Load results
            fpath = os.path.join(
                ROOT_DATA, f"zhang_output_from_signal_dx{dx}m_dy{dy}m_snr{snr}dB.nc"
            )
            ds = xr.open_dataset(fpath)

            # Process results
            data_rootname = f"loc_zhang_dx{dx}m_dy{dy}m_snr{snr}dB_mc{i_mc}"
            process_localisation_zhang2023(
                ds, subfolder, nf, freq_draw_method, data_fname=data_rootname
            )
            ds.close()

            # Plot results
            if i_mc == 0:
                plot_study_zhang2023(subfolder, data_fname=data_rootname)
                plt.close("all")

            # Load processed surface and derive msr
            fpath = os.path.join(
                os.path.join(ROOT_DATA, subfolder),
                f"{data_rootname}_fullarray.nc",
            )
            ds_fa = xr.open_dataset(fpath)

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

        study_msr_vs_snr()


def replay_all_snr(
    snrs,
    dx=20,
    dy=20,
):

    # Load params
    depth, receivers, source, grid, frequency, _ = params()

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

        study_msr_vs_snr()


def study_msr_vs_snr():
    """Plot metrics (MSR, RMSE) vs SNR for both GCC and RTF"""

    folder = "from_signal_dx20m_dy20m"
    # Load msr results
    msr_txt_filepath = os.path.join(ROOT_DATA, folder, "msr_snr.txt")
    msr = pd.read_csv(msr_txt_filepath, sep=" ")

    # Compute mean and std of msr for each snr
    msr_mean = msr.groupby("snr").mean()
    msr_std = msr.groupby("snr").std()

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        msr_mean.index,
        msr_mean["d_gcc"],
        yerr=msr_std["d_gcc"],
        fmt="o-",
        label=r"$\textrm{DCF (GCC SCOT)}$",
    )
    ax.errorbar(
        msr_mean.index,
        msr_mean["d_rtf"],
        yerr=msr_std["d_rtf"],
        fmt="o-",
        label=r"$\textrm{RTF}$",
    )
    ax.set_xlabel(r"$\textrm{SNR [dB]}$")
    ax.set_ylabel(r"$\textrm{MSR [dB]}$")
    ax.legend()
    ax.grid()
    # plt.show()
    fpath = os.path.join(ROOT_IMG, folder, "msr_snr.png")
    plt.savefig(fpath)

    # Load position error results
    dr_txt_filepath = os.path.join(ROOT_DATA, folder, "dr_pos_snr.txt")
    dr = pd.read_csv(dr_txt_filepath, sep=" ")

    # Compute mean and std of position error for each snr
    dr_mean = dr.groupby("snr").mean()
    dr_std = dr.groupby("snr").std()

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        dr_mean.index,
        dr_mean["dr_gcc"],
        yerr=dr_std["dr_gcc"],
        fmt="o-",
        label=r"$\textrm{DCF (GCC SCOT)}$",
    )
    ax.errorbar(
        dr_mean.index,
        dr_mean["dr_rtf"],
        yerr=dr_std["dr_rtf"],
        fmt="o-",
        label=r"$\textrm{RTF}$",
    )
    ax.set_xlabel(r"$\textrm{SNR [dB]}$")
    ax.set_ylabel(r"$\Delta_r \textrm{[m]}$")
    ax.legend()
    ax.grid()

    fpath = os.path.join(ROOT_IMG, folder, "dr_pos_snr.png")
    plt.savefig(fpath)

    dr["dr_gcc"] = dr["dr_gcc"] ** 2
    dr["dr_rtf"] = dr["dr_rtf"] ** 2
    mse = dr.groupby("snr").mean()
    rmse = np.sqrt(mse)

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(rmse.index, rmse["dr_gcc"], "o-", label=r"$\textrm{DCF (GCC SCOT)}$")
    ax.plot(rmse.index, rmse["dr_rtf"], "o-", label=r"$\textrm{RTF}$")
    ax.set_xlabel(r"$\textrm{SNR [dB]}$")
    ax.set_ylabel(r"$\textrm{RMSE [m]}$")
    ax.legend()
    ax.grid()

    fpath = os.path.join(ROOT_IMG, folder, "rmse_snr.png")
    plt.savefig(fpath)

    # plt.show()


if __name__ == "__main__":

    nf = 20
    dx, dy = 20, 20
    # # Load rtf data
    # fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
    # ds = xr.open_dataset(fpath)

    # folder = f"fullsimu_dx{dx}m_dy{dy}m"
    # process_localisation_zhang2023(ds, folder, nf=nf)
    # plot_study_zhang2023(folder)

    # fpath = os.path.join(ROOT_DATA, f"zhang_output_from_signal_dx{dx}m_dy{dy}m.nc")
    # fpath = os.path.join(ROOT_DATA, "zhang_output_from_signal_dx20m_dy20m_snr-40dB.nc")
    # ds = xr.open_dataset(fpath)

    # folder = f"from_signal_dx{dx}m_dy{dy}m"
    # process_localisation_zhang2023(ds, folder, nf=nf, freq_draw_method="equally_spaced")
    # plot_study_zhang2023(folder)

    # # snrs = np.arange(-40, 15, 5)
    # # print(snrs)
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

    snrs = np.arange(-40, 15, 5)
    # snrs = np.arange(-20, 15, 5)

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

    replay_all_snr(snrs=snrs, dx=dx, dy=dy)

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
