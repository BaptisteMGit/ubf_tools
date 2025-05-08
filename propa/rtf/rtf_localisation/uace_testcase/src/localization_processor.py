#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   localization_processor.py
@Time    :   2025/05/08 21:41:30
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to process localization from library / event features
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder

from misc import cast_matrix_to_target_shape
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
    get_axis_order,
)

from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast

# from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
#     study_perf_vs_snr,
#     plot_study_zhang2023,
# )


class LocalizationProcessor:
    """
    Class to process localization from library / event features
    """

    def __init__(self, simulation):
        """
        Constructor
        :param simulation: instance of the Simulation class
        """
        self.simulation = simulation
        self.fb = FeatureBuilder(simulation=simu)


    @staticmethod
    def get_axis_order(da, ax_names):
        # Get dims order to avoid potential confusions between axis
        ax_order = {}
        for name in ax_names:
            ax_order[name] = da.dims.index(name) if name in da.dims else None
        return ax_order

    def process_multiple_snrs(
        self,
        snrs,
        n_monte_carlo,
        dx=20,
        dy=20,
        nf=100,
        run_mode="a",
        subarrays_list=None,
        freq_draw_method="equally_spaced",
        antenna_type="zhang",
        library_stype="lfm",
        event_stype="wn",
        plot_args={},
        debug=False,
        verbose=False,
        check=False,
        ):

        fb = FeatureBuilder(
            root_name="zhang_output_from_signal",
            root_data=p.root_data,
            antenna_type=antenna_type,
            library_stype=library_stype,
            event_stype=event_stype,
            rtf_method="cs_eigve",
            gcc_method="scot",
            verbose=verbose,
            debug=debug,
            check=check,
        )

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
            subfolder_fullpath = os.path.join(p.root_data, subfolder)
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

                # Run simulation (one simulation = 1 noise generation)
                fb.build_features_from_time_signal(snr_dB=snr)

                # Load results
                fpath = os.path.join(
                    p.root_data,
                    f"zhang_output_from_signal_dx{dx}m_dy{dy}m_snr{snr:.1f}dB.nc",
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

                    msr, pos_hat = estimate_msr(ds_fa)
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

    def process(
        self,
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
        # _, _, _, grid, _, _ = params(debug=debug, antenna_type=antenna_type)

        # # Define folder to store data
        # root_data = os.path.join(p.root_data, folder)
        # if not os.path.exists(root_data):
        #     os.makedirs(root_data)

        # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
        # Match field processing #

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

        # Get dimension positions for more robustness and clarity
        da_tmp = ds.rtf_real.sel(idx_rcv_ref=0)
        ax_order = get_axis_order(da=da_tmp, ax_names=["idx_rcv", "f_rtf", "y", "x"])
        ax_rcv = ax_order["idx_rcv"]
        ax_f = ax_order["f_rtf"]
        ax_y = ax_order["y"]
        ax_x = ax_order["x"]

        # Set spatial coords order
        xy_dims = ["y", "x"] if ax_y < ax_x else ["x", "y"]

        # Define distance to use
        dist_func = D_hermitian_angle_fast

        dist_kwargs = {
            "ax_rcv": ax_rcv,
            "unit": "deg",
            "apply_mean": True,
            "ax_f": ax_f,
        }

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

            rtf_grid = (
                ds_cpl_rtf.rtf_real.values + 1j * ds_cpl_rtf.rtf_imag.values
            )  # (n_cpl=2, nf, ny, nx)
            rtf_event = (
                ds_cpl_rtf.rtf_event_real.values + 1j * ds_cpl_rtf.rtf_event_imag.values
            )  # (n_cpl=2, nf)

            theta = dist_func(rtf_event, rtf_grid, **dist_kwargs)

            # Add theta to dataset
            ds_cpl_rtf["theta"] = (xy_dims, theta)

            # Normalize
            d_rtf = normalize_metric_contrast(-ds_cpl_rtf.theta)

            # Convert to dB
            d_rtf = d_rtf.values
            d_rtf[d_rtf == 0] = p.min_val_log
            d_rtf = 10 * np.log10(d_rtf)
            ds_cpl_rtf["d_rtf"] = (xy_dims, d_rtf)

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

            # Normalize
            d_gcc = normalize_metric_contrast(d_gcc)

            # Convert to dB
            d_gcc = d_gcc
            d_gcc[d_gcc == 0] = p.min_val_log
            d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

            # Add d to dataset
            ds_cpl_gcc["d_gcc"] = (xy_dims, d_gcc)

            # Store d_gcc for full array incoherent processing
            # d_gcc_fullarray.append(d_gcc)

            # Build dataset to be saved as netcdf
            ds_cpl = xr.Dataset(
                data_vars=dict(
                    theta_rtf=(xy_dims, ds_cpl_rtf.theta.values),
                    d_rtf=(xy_dims, ds_cpl_rtf.d_rtf.values),
                    d_gcc=(xy_dims, ds_cpl_gcc.d_gcc.values),
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
            fpath = (
                self.simulation.localization_dataset_fpath.split(".nc")[0]
                + f"s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
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
            d_gcc = np.abs(np.sum(gcc_grid * np.conj(gcc_event) * df_gcc, axis=0))
            # d_gcc = d_gcc / np.max(d_gcc)

            # # Normalize
            d_gcc = normalize_metric_contrast(d_gcc)

            # # Convert to dB
            # d_gcc = d_gcc
            # d_gcc[d_gcc == 0] = p.min_val_log
            # d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

            # Store d_gcc for full array incoherent processing
            d_gcc_fullarray.append(d_gcc)

        ## RTF ##
        # Select reference receiver (by default the first receiver of the array is selected)
        i_ref = rcv_in_fullarray[0]
        ds_fa_rtf = ds_fa.sel(idx_rcv_ref=i_ref)

        rtf_grid = ds_fa_rtf.rtf_real.values + 1j * ds_fa_rtf.rtf_imag.values
        rtf_event = (
            ds_fa_rtf.rtf_event_real.values + 1j * ds_fa_rtf.rtf_event_imag.values
        )

        theta = dist_func(rtf_event, rtf_grid, **dist_kwargs)

        # Add theta to dataset
        ds_fa_rtf["theta"] = (xy_dims, theta)

        # # Convert theta to a metric between -1 and 1
        # theta_inv = (
        #     theta_max - ds_fa_rtf.theta
        # )  # So that the source position is the maximum value
        # d_rtf = (theta_inv - theta_max / 2) / (theta_max / 2)  # To lie between -1 and 1

        d_rtf = normalize_metric_contrast(-ds_fa_rtf.theta)  # q in [0, 1]

        #  Replace 0 by 1e-5 to avoid log(0) in dB conversion
        d_rtf = d_rtf.values
        d_rtf[d_rtf == 0] = p.min_val_log
        d_rtf = 10 * np.log10(d_rtf)  # Convert to dB
        ds_fa_rtf["d_rtf"] = (xy_dims, d_rtf)

        ## GCC ##
        d_gcc_fullarray = np.array(d_gcc_fullarray)
        d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

        # Normalize
        # d_gcc_fullarray = normalize_metric_contrast(d_gcc_fullarray)

        # # Convert to dB
        # d_gcc = d_gcc
        # d_gcc[d_gcc == 0] = p.min_val_log
        # d_gcc = 10 * np.log10(d_gcc)  # Convert to dB

        # # Convert back to linear scale before computing the mean
        # d_gcc_fullarray = 10 ** (d_gcc_fullarray / 10)
        # d_gcc_fullarray = np.mean(d_gcc_fullarray, axis=0)

        # Convert to dB
        d_gcc_fullarray[d_gcc_fullarray == 0] = p.min_val_log
        d_gcc_fullarray = 10 * np.log10(d_gcc_fullarray)
        # d_gcc_fullarray = d_gcc_fullarray

        # Build dataset to be saved as netcdf
        ds_fullarray = xr.Dataset(
            data_vars=dict(
                theta_rtf=(xy_dims, ds_fa_rtf.theta.values),
                d_rtf=(xy_dims, ds_fa_rtf.d_rtf.values),
                d_gcc=(xy_dims, d_gcc_fullarray),
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
            fpath = (
                self.simulation.localization_dataset_fpath.split(".nc")[0]
                + f"s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}dB.nc"
            )
        ds_fullarray.to_netcdf(fpath)
        ds_fullarray.close()
