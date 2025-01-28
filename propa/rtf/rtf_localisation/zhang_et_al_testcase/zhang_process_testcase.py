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

from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import *
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    plot_study_zhang2023,
)
from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast


def process_localisation_zhang2023(ds, folder, nf=10):
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
    # nf = 10
    df = np.diff(ds.f.values)[0]
    f_loc = np.random.choice(ds.f.values, nf)
    ds = ds.sel(f=f_loc)

    d_gcc_fullarray = []
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
        # # Convert theta to a metric between -1 and 1
        # theta_inv = (
        #     theta_max - ds_cpl_rtf.theta
        # )  # So that the source position is the maximum value
        # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

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
        gcc_event = cast_matrix_to_target_shape(gcc_event, gcc_grid.shape)

        # Build cross corr (Equation (8) in Zhang et al. 2023)
        # d_gcc = np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0)
        d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_event) * df, axis=0))
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
        d_gcc_fullarray.append(d_gcc)

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
        fpath = os.path.join(
            root_data,
            f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc",
        )
        ds_cpl.to_netcdf(fpath)

    ###### Full array ######

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
    # Convert back to linear scale before computing the mean
    d_gcc_fullarray = 10 ** (d_gcc_fullarray / 10)
    d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

    # Convert to dB
    d_gcc_fullarray[d_gcc_fullarray == 0] = MIN_VAL_LOG
    d_gcc_fullarray = 10 * np.log10(d_gcc_fullarray)
    # d_gcc_fullarray = d_gcc_fullarray

    # Build dataset to be saved as netcdf
    ds_cpl_fullarray = xr.Dataset(
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
    fpath = os.path.join(
        root_data,
        f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_fullarray.nc",
    )
    ds_cpl_fullarray.to_netcdf(fpath)


if __name__ == "__main__":

    nf = 40
    dx, dy = 20, 20
    # # Load rtf data
    # fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
    # ds = xr.open_dataset(fpath)

    # folder = f"fullsimu_dx{dx}m_dy{dy}m"
    # process_localisation_zhang2023(ds, folder, nf=nf)
    # plot_study_zhang2023(folder)

    fpath = os.path.join(ROOT_DATA, f"zhang_output_from_signal_dx{dx}m_dy{dy}m.nc")
    fpath = os.path.join(ROOT_DATA, "zhang_output_from_signal_dx20m_dy20m_nperseg11.nc")
    ds = xr.open_dataset(fpath)

    folder = f"from_signal_dx{dx}m_dy{dy}m"
    process_localisation_zhang2023(ds, folder, nf=nf)
    plot_study_zhang2023(folder)


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
