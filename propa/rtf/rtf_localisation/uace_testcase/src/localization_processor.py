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
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from misc import compute_hyperbola, cast_matrix_to_target_shape
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast
from publication.publication_figure import PubFigure

import propa.rtf.rtf_localisation.uace_testcase.src.params as p

# Libraries to handle main lobe detection
from skimage import measure  # Import for contour detection
from scipy import ndimage as ndi
from skimage.filters import rank
from sklearn import preprocessing
from skimage.measure import label
from sklearn.cluster import KMeans
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from scipy.ndimage import binary_dilation, label

PubFigure(ticks_fontsize=22, use_tex=p.use_tex)


class LocalizationProcessor:
    """
    Class to process localization from library / event features
    """

    def __init__(self, simulation, plot_args: dict = p.plot_args):
        """
        Constructor
        :param simulation: instance of the Simulation class
        """
        self.simulation = simulation
        self.plot_args = plot_args

        self.n_monte_carlo = self.simulation.monte_carlo_iterations
        self.fb = FeatureBuilder(simulation=self.simulation)

        # Init filepath
        self.current_snr_foldername = None
        self.current_snr_folderpath = None
        self.current_snr_dataset_rootpath = None
        self.current_snr_fullarray_dataset_fpath = None

    @staticmethod
    def get_axis_order(da, ax_names):
        # Get dims order to avoid potential confusions between axis
        ax_order = {}
        for name in ax_names:
            ax_order[name] = da.dims.index(name) if name in da.dims else None
        return ax_order

    @staticmethod
    def get_array_label(rcv_idx):
        array_label = "_".join([f"s{i+1}" for i in rcv_idx])
        return array_label

    @classmethod
    def build_subarrays_args(cls, subarrays_list):
        subarrays_args = {
            index: {
                "idx_rcv": subarrays_list[index],
                "array_label": cls.get_array_label(subarrays_list[index]),
                "msr_filepath": None,
                "dr_pos_filepath": None,
            }
            for index in range(len(subarrays_list))
        }
        return subarrays_args

    def init_msr_file(self, run_mode, subarrays_args):

        root_msr = os.path.join(
            self.simulation.root_data, self.simulation.from_sig_foldername, "msr"
        )
        if not os.path.exists(root_msr):
            os.makedirs(root_msr)

        for index, sa_item in subarrays_args.items():

            msr_txt_filepath = os.path.join(
                root_msr, f"msr_snr_{sa_item['array_label']}.txt"
            )

            if (
                not os.path.exists(msr_txt_filepath) or run_mode == "w"
            ):  # To avoid writting over existing file
                header_line = "snr i_mc d_gcc d_rtf\n"
                with open(msr_txt_filepath, "w") as f:
                    f.write(header_line)

            sa_item["msr_filepath"] = msr_txt_filepath

    def init_dr_file(self, run_mode, subarrays_args):

        root_dr = os.path.join(
            self.simulation.root_data, self.simulation.from_sig_foldername, "dr_pos"
        )
        if not os.path.exists(root_dr):
            os.makedirs(root_dr)

        for index, sa_item in subarrays_args.items():

            dr_txt_filepath = os.path.join(
                root_dr, f"dr_pos_snr_{sa_item['array_label']}.txt"
            )

            if (
                not os.path.exists(dr_txt_filepath) or run_mode == "w"
            ):  # To avoid writting over existing file
                header_line = "snr i_mc dr_gcc dr_rtf\n"
                with open(dr_txt_filepath, "w") as f:
                    f.write(header_line)

            sa_item["dr_pos_filepath"] = dr_txt_filepath

    def process_multiple_snrs(
        self,
        snrs,
        run_mode="a",
        subarrays_list=None,
    ):

        #
        if subarrays_list is None:
            subarrays_list = np.atleast_2d(self.simulation.antenna.rcv_idx)  # Fullarray

        subarrays_args = self.build_subarrays_args(subarrays_list)
        self.init_dr_file(run_mode, subarrays_args)
        self.init_msr_file(run_mode, subarrays_args)

        for snr in snrs:
            self.current_snr_foldername = os.path.join(
                self.simulation.from_sig_foldername, f"snr_{snr:.1f}dB"
            )
            self.current_snr_folderpath = os.path.join(
                self.simulation.root_data, self.current_snr_foldername
            )
            if not os.path.exists(self.current_snr_folderpath):
                os.makedirs(self.current_snr_folderpath)

            self.current_snr_dataset_rootpath = os.path.join(
                self.current_snr_folderpath, self.simulation.localization_dataset_fname
            )

            # Init img folder
            self.current_snr_root_img = os.path.join(
                self.simulation.root_img, self.current_snr_foldername
            )
            if not os.path.exists(self.current_snr_root_img):
                os.makedirs(self.current_snr_root_img)

            # List existing files in subfolder
            snr_files = os.listdir(self.current_snr_folderpath)
            snr_files = [sfile for sfile in snr_files if "mc" in sfile]

            if run_mode == "a":  # Append mode -> do not overwrite existing values
                # Keep only fullarray files
                fa_snr_files = [sfile for sfile in snr_files if "fullarray" in sfile]
                # Parse i_mc and get max
                i_mcs = [
                    int(sfile.split("_")[5].split("mc")[1]) for sfile in fa_snr_files
                ]
                # Set offset
                i_mc_offset = max(i_mcs) + 1 if i_mcs else 0

            elif run_mode == "w":  # Write mode -> overwrite existing files
                # Remove file in subfolder
                for sfile in snr_files:
                    os.remove(os.path.join(self.current_snr_folderpath, sfile))
                # Set i_mc_offset to 0
                i_mc_offset = 0

            if self.simulation.verbose:
                print(
                    f"Start processing snr = {snr} dB (i_mc from {i_mc_offset} to {self.n_monte_carlo + i_mc_offset-1})"
                )

            plot_study = True
            # Run simulation self.n_monte_carlo times at the same snr to derive the mean MSR
            for i_mc in range(i_mc_offset, self.n_monte_carlo + i_mc_offset):

                if self.simulation.verbose:
                    print(f"i_mc = {i_mc}")

                # Run simulation (one simulation = 1 noise generation)
                self.fb.build_features_from_time_signal(snr_dB=snr)

                # Load results
                fpath = (
                    self.simulation.feature_dataset_fpath.split(".nc")[0]
                    + f"_snr_{snr:.1f}dB.nc"
                )
                ds = xr.open_dataset(fpath)

                # Process results
                # data_rootname = f"loc_zhang_dx{dx}m_dy{dy}m_snr{snr}dB_mc{i_mc}"
                self.current_mc_dataset_filename = f"localization_{self.simulation.grid_res_label}_snr_{snr:.1f}dB_mc{i_mc}.nc"

                # Loop over subarrays of interest
                for sa_idx, sa_item in subarrays_args.items():

                    rcv_in_fullarray = sa_item["idx_rcv"]
                    self.process(
                        ds,
                        rcv_in_fullarray=rcv_in_fullarray,
                    )

                    # Plot results
                    if plot_study:
                        self.plot()
                        plt.close("all")

                    # Load processed surface and derive msr
                    ds_fa = xr.open_dataset(self.current_snr_fullarray_dataset_fpath)
                    msr, pos_hat = self.estimate_msr(ds_fa)
                    ds_fa.close()

                    # Store MSR and DR
                    msr_txt_filepath = sa_item["msr_filepath"]
                    dr_txt_filepath = sa_item["dr_pos_filepath"]

                    # Save to text file for further analysis
                    newline = f"{snr} {i_mc} {msr['d_gcc']:.2f} {msr['d_rtf']:.2f}\n"
                    with open(msr_txt_filepath, "a") as f:
                        f.write(newline)

                    # Position error
                    delta_r_gcc = np.sqrt(
                        (pos_hat["d_gcc"]["x"] - self.simulation.event_ship_x) ** 2
                        + (pos_hat["d_gcc"]["y"] - self.simulation.event_ship_y) ** 2
                    )
                    delta_r_rtf = np.sqrt(
                        (pos_hat["d_rtf"]["x"] - self.simulation.event_ship_x) ** 2
                        + (pos_hat["d_rtf"]["y"] - self.simulation.event_ship_y) ** 2
                    )

                    # Save to text file for further analysis
                    newline = f"{snr} {i_mc} {delta_r_gcc:.2f} {delta_r_rtf:.2f}\n"
                    with open(dr_txt_filepath, "a") as f:
                        f.write(newline)

                plot_study = False

                # Check RTF estimation at a few grid points
                ds.close()

            self.study_perf_vs_snr(subarrays_list=subarrays_list)

    def process(
        self,
        ds,
        rcv_in_fullarray=None,
    ):

        # Compute distance between the RTF vector associated with the source and the RTF vector at each grid pixel
        # Match field processing #

        # Select a few frequencies
        if (
            self.simulation.frequency_drawing_method == "random"
        ):  # Same option as used by Zhang et al 2023 yet results (especially the msr are not reproductible from one run to another with the same input dataset)
            f_loc_rtf = np.random.choice(
                ds.f_rtf.values, self.simulation.number_of_drawn_frequencies
            )
            f_loc_gcc = np.random.choice(
                ds.f_gcc.values, self.simulation.number_of_drawn_frequencies
            )
        elif (
            self.simulation.frequency_drawing_method == "equally_spaced"
        ):  # Reproductible option used for msr study
            idx_f_loc = np.linspace(
                0,
                ds.sizes["f_rtf"] - 1,
                self.simulation.number_of_drawn_frequencies,
                dtype=int,
            )
            f_loc_rtf = ds.f_rtf.values[idx_f_loc]
            idx_f_loc = np.linspace(
                0,
                ds.sizes["f_gcc"] - 1,
                self.simulation.number_of_drawn_frequencies,
                dtype=int,
            )
            f_loc_gcc = ds.f_gcc.values[idx_f_loc]

        ds = ds.sel(f_rtf=f_loc_rtf)
        ds = ds.sel(f_gcc=f_loc_gcc)
        df_gcc = np.diff(ds.f_gcc.values)[0]

        # Get dimension positions for more robustness and clarity
        da_tmp = ds.rtf_real.sel(idx_rcv_ref=0)
        ax_order = self.get_axis_order(
            da=da_tmp, ax_names=["idx_rcv", "f_rtf", "y", "x"]
        )
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
        rcv_couples_fa = self.get_rcv_couples(idx_receivers=ds_fa.idx_rcv.values)

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
                self.current_snr_dataset_rootpath
                + f"_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
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
        array_label = self.get_array_label(rcv_in_fullarray)
        self.current_snr_fullarray_dataset_fpath = (
            self.current_snr_dataset_rootpath + f"_fullarray_{array_label}.nc"
        )
        ds_fullarray.to_netcdf(self.current_snr_fullarray_dataset_fpath)
        ds_fullarray.close()

    # ========================================================================================================================
    # Plotting functions
    # ========================================================================================================================
    def study_perf_vs_snr(self, subarrays_list):
        """Plot metrics (MSR, RMSE) vs SNR for both DCF and RTF"""

        root_img = os.path.join(
            self.simulation.root_img, self.simulation.from_sig_foldername, "perf_vs_snr"
        )
        if not os.path.exists(root_img):
            os.makedirs(root_img)

        # Load results (all available snrs)
        msr, dr, rmse = self.load_msr_rmse_res_subarrays(subarrays_list)

        for sa_key in msr.keys():

            # Extract info dataframes for current subarray
            rcv_ids = [f"{id[0]}_{id[1]}" for id in sa_key.split("_")]
            rcv_str = "$" + ", \,".join(rcv_ids) + "$"
            dr_sa = dr[sa_key]
            msr_sa = msr[sa_key]
            rmse_sa = rmse[sa_key]

            # Plot msr
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.errorbar(
                msr_sa.index,
                msr_sa.dcf_mean,
                yerr=msr_sa.dcf_std,
                fmt="o-",
                label="DCF",
            )
            ax.errorbar(
                msr_sa.index,
                msr_sa.rtf_mean,
                yerr=msr_sa.rtf_std,
                fmt="o-",
                label="RTF",
            )
            ax.set_xlabel("SNR [dB]")
            ax.set_ylabel("MSR [dB]")
            ax.legend()
            ax.grid()

            plt.suptitle(f"Receivers = ({rcv_str})")

            fpath = os.path.join(root_img, f"msr_snr_{sa_key}.png")
            plt.savefig(fpath)
            plt.close("all")

            # Plot dr
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.errorbar(
                dr_sa.index,
                dr_sa.dcf_mean,
                yerr=dr_sa.dcf_std,
                fmt="o-",
                label="DCF",
            )
            ax.errorbar(
                dr_sa.index,
                dr_sa.rtf_mean,
                yerr=dr_sa.rtf_std,
                fmt="o-",
                label="RTF",
            )
            plt.suptitle(f"Receivers = ({rcv_str})")
            ax.set_ylabel(r"$\Delta_r$" + " [m]")
            ax.set_xlabel("SNR [dB]")
            ax.legend()
            ax.grid()
            fpath = os.path.join(root_img, f"dr_pos_snr_{sa_key}.png")
            plt.savefig(fpath)

            # Plot rmse
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(rmse_sa.index, rmse_sa["dcf"], "o-", label="DCF")
            ax.plot(rmse_sa.index, rmse_sa["rtf"], "o-", label="RTF")
            plt.suptitle(f"Receivers = ({rcv_str})")
            ax.set_xlabel("SNR [dB]")
            ax.set_ylabel("RMSE [m]")
            ax.legend()
            ax.grid()
            fpath = os.path.join(root_img, f"rmse_snr_{sa_key}.png")
            plt.savefig(fpath)
            plt.close("all")

    def load_msr_rmse_res_subarrays(self, subarrays_list, snrs=None):
        subarrays_args = self.build_subarrays_args(subarrays_list)

        self.init_dr_file(run_mode="a", subarrays_args=subarrays_args)
        self.init_msr_file(run_mode="a", subarrays_args=subarrays_args)

        sa_labels = [sa["array_label"] for sa in list(subarrays_args.values())]

        msr_pd = {
            label: pd.DataFrame(
                [], index=None, columns=["rtf_mean", "dcf_mean", "rtf_std", "dcf_std"]
            )
            for label in sa_labels
        }
        dr_pd = {
            label: pd.DataFrame(
                [], index=None, columns=["rtf_mean", "dcf_mean", "rtf_std", "dcf_std"]
            )
            for label in sa_labels
        }
        rmse_pd = {
            label: pd.DataFrame([], index=None, columns=["rtf", "dcf"])
            for label in sa_labels
        }

        for sa_idx, sa_item in subarrays_args.items():
            sa_label = sa_item["array_label"]
            msr_txt_filepath = sa_item["msr_filepath"]
            dr_txt_filepath = sa_item["dr_pos_filepath"]

            # Load msr and position error results
            msr = pd.read_csv(msr_txt_filepath, sep=" ")
            dr = pd.read_csv(dr_txt_filepath, sep=" ")

            if snrs is None:
                snrs_to_keep = np.unique(dr.snr.values)
            else:
                snrs_to_keep = snrs

            # Keep only snrs of interest
            msr = msr[msr["snr"].isin(snrs_to_keep)]
            dr = dr[dr["snr"].isin(snrs_to_keep)]

            # Compute mean and std of msr for each snr
            msr_mean = msr.groupby("snr").mean()
            msr_std = msr.groupby("snr").std()

            msr_pd[sa_label]["rtf_std"] = msr_std[f"d_rtf"].values
            msr_pd[sa_label]["dcf_std"] = msr_std[f"d_gcc"].values
            msr_pd[sa_label]["rtf_mean"] = msr_mean[f"d_rtf"].values
            msr_pd[sa_label]["dcf_mean"] = msr_mean[f"d_gcc"].values
            msr_pd[sa_label].set_index(msr_std.index, inplace=True)

            # msr_mu.append(msr_mean)
            # msr_sig.append(msr_std)

            # Compute mean and std of position error for each snr
            dr_mean = dr.groupby("snr").mean()
            dr_std = dr.groupby("snr").std()

            dr_pd[sa_label]["rtf_std"] = dr_std[f"dr_rtf"].values
            dr_pd[sa_label]["dcf_std"] = dr_std[f"dr_gcc"].values
            dr_pd[sa_label]["rtf_mean"] = dr_mean[f"dr_rtf"].values
            dr_pd[sa_label]["dcf_mean"] = dr_mean[f"dr_gcc"].values
            dr_pd[sa_label].set_index(dr_std.index, inplace=True)

            # dr_mu.append(dr_mean)
            # dr_sig.append(dr_std)

            dr["d_gcc"] = dr["dr_gcc"] ** 2
            dr["d_rtf"] = dr["dr_rtf"] ** 2
            mse = dr.groupby("snr").mean()
            rmse = np.sqrt(mse)

            rmse_pd[sa_label]["rtf"] = rmse[f"d_rtf"].values
            rmse_pd[sa_label]["dcf"] = rmse[f"d_gcc"].values
            rmse_pd[sa_label].set_index(rmse.index, inplace=True)

            # rmse_.append(rmse)

        # return msr_mu, msr_sig, dr_mu, dr_sig, rmse_

        return msr_pd, dr_pd, rmse_pd

    def plot(self):
        # Extract plot flags
        plot_array = self.plot_args.get("plot_array", False)
        plot_single_cpl_surf = self.plot_args.get("plot_single_cpl_surf", False)
        plot_fullarray_surf = self.plot_args.get("plot_fullarray_surf", False)
        plot_cpl_surf_comparison = self.plot_args.get("plot_cpl_surf_comparison", False)
        plot_fullarray_surf_comparison = self.plot_args.get(
            "plot_fullarray_surf_comparison", False
        )
        plot_surf_dist_comparison = self.plot_args.get(
            "plot_surf_dist_comparison", False
        )
        plot_mainlobe_contour = self.plot_args.get("plot_mainlobe_contour", False)
        plot_msr_estimation = self.plot_args.get("plot_msr_estimation", False)

        ds_fa = xr.open_dataset(self.current_snr_fullarray_dataset_fpath)

        vmin_dB = np.round(
            np.max([ds_fa[dist].median() for dist in ["d_gcc", "d_rtf"]]), 0
        )

        # Define plot args for ambiguity surfaces
        plot_args_theta = {
            "dist": "theta_rtf",
            "root_img": self.current_snr_root_img,
            "testcase": "zhang_et_al_2023",
            "dist_label": r"$\theta$" + " [°]",
            "vmax": 50,
            "vmin": 0,
            "add_hyperbola": True,
        }

        plot_args_d_rtf = {
            "dist": "q_rtf",
            "root_img": self.current_snr_root_img,
            "testcase": "zhang_et_al_2023",
            # "dist_label": r"$d_{rtf}$",
            "dist_label": "[dB]",
            # "vmax": 1,
            # "vmin": 0,
            # dB scale
            "vmax": 0,
            "vmin": vmin_dB,
            "add_hyperbola": True,
        }

        plot_args_gcc = {
            "dist": "q_dcf",
            "root_img": self.current_snr_root_img,
            "testcase": "zhang_et_al_2023",
            # "dist_label": r"$d_{gcc}$",
            "dist_label": "[dB]",
            # "vmax": 1,
            # "vmin": 0,
            # dB scale
            "vmax": 0,
            "vmin": vmin_dB,
            "add_hyperbola": True,
        }

        if plot_array:
            # Plot antenna geometry and research area
            self.plot_antenna_and_search_area(
                root_img=self.current_snr_root_img,
                rcv_in_fullarray=list(ds_fa.idx_rcv),
            )

        ###### Two sensor pairs ######
        rcv_couples = self.get_rcv_couples(ds_fa.idx_rcv)

        if plot_single_cpl_surf:
            cpl_foldername = "ambiguity_surface_receivers_pair"
            # Select receivers to build the sub-array
            # rcv_couples = np.array([[0, 2], [1, 4], [3, 5]])  # s1s3, s2s5, s4s6

            for rcv_cpl in rcv_couples:
                # Load data for rcv_cpl
                fpath = (
                    self.current_snr_dataset_rootpath
                    + f"_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
                )
                ds_cpl = xr.open_dataset(fpath)

                # Update sub array args
                plot_args_theta["sub_array"] = rcv_cpl
                plot_args_d_rtf["sub_array"] = rcv_cpl
                plot_args_gcc["sub_array"] = rcv_cpl

                # Theta
                self.plot_ambiguity_surface(
                    amb_surf=ds_cpl.theta_rtf,
                    plot_args=plot_args_theta,
                    loc_arg="min",
                    folder_name=cpl_foldername,
                )

                # d_rtf
                self.plot_ambiguity_surface(
                    amb_surf=ds_cpl.d_rtf,
                    plot_args=plot_args_d_rtf,
                    loc_arg="max",
                    folder_name=cpl_foldername,
                )

                # d_gcc
                self.plot_ambiguity_surface(
                    amb_surf=ds_cpl.d_gcc,
                    plot_args=plot_args_gcc,
                    loc_arg="max",
                    folder_name=cpl_foldername,
                )

        ###### Full array ######
        fa_foldername = "ambiguity_surface_fullarray"
        if plot_fullarray_surf:
            # Update sub array args
            plot_args_theta["sub_array"] = ds_fa.attrs["idx_rcv"]
            plot_args_d_rtf["sub_array"] = ds_fa.attrs["idx_rcv"]
            plot_args_gcc["sub_array"] = ds_fa.attrs["idx_rcv"]
            plot_args_theta["add_circle"] = True
            plot_args_d_rtf["add_circle"] = True
            plot_args_gcc["add_circle"] = True
            plot_args_theta["add_hyperbola"] = False
            plot_args_d_rtf["add_hyperbola"] = False
            plot_args_gcc["add_hyperbola"] = False

            # hyperbola_cpls = [[0, 2], [0, 4], [1, 3], [3, 5]]
            hyperbola_cpls = [[2, 4], [3, 5]]
            plot_args_gcc["hyperbola_cpls"] = hyperbola_cpls

            # Theta
            self.plot_ambiguity_surface(
                amb_surf=ds_fa.theta_rtf,
                plot_args=plot_args_theta,
                loc_arg="min",
                folder_name=fa_foldername,
            )

            # d_rtf
            self.plot_ambiguity_surface(
                amb_surf=ds_fa.d_rtf,
                plot_args=plot_args_d_rtf,
                loc_arg="max",
                folder_name=fa_foldername,
            )

            # d_gcc
            self.plot_ambiguity_surface(
                amb_surf=ds_fa.d_gcc,
                plot_args=plot_args_gcc,
                loc_arg="max",
                folder_name=fa_foldername,
            )

        # Define plot args for ambiguity surfaces
        xticks_pos_km = [3.6, 4.0, 4.4]
        yticks_pos_km = [6.5, 6.9, 7.3]
        xticks_pos_m = [xt * 1e3 for xt in xticks_pos_km]
        yticks_pos_m = [yt * 1e3 for yt in yticks_pos_km]

        cmap = "jet"
        # vmax = 1
        # vmin = 0

        # dB scale
        vmax = 0
        vmin = vmin_dB

        x_src = self.simulation.event_ship_x
        y_src = self.simulation.event_ship_y

        ###### Figure 4 : Subplot in Zhang et al 2023 ######
        rcv_couples = self.get_rcv_couples(ds_fa.idx_rcv)

        if plot_cpl_surf_comparison:
            self.plot_subarrays_ambiguity_surfaces(
                rcv_couples,
                vmin,
                vmax,
                xticks_pos_m,
                yticks_pos_m,
                cmap=cmap,
            )

        ###### Figure 5 : Subplot in Zhang et al 2023 ######
        if plot_fullarray_surf_comparison:
            self.plot_fullarray_ambiguity_surfaces(
                ds_fa,
                vmin,
                vmax,
                xticks_pos_m,
                yticks_pos_m,
                cmap=cmap,
            )

        ###### Figure 5 distribution ######
        if plot_surf_dist_comparison:
            self.plot_ambiguity_surface_distribution(ds_fa)

        ###### Figure 5 showing pixels selected as the mainlobe ######
        if plot_mainlobe_contour:
            self.plot_ambiguity_surface_mainlobe_contour(
                ds_fa, vmin, vmax, xticks_pos_m, yticks_pos_m, cmap=cmap
            )

        # estimate_msr(ds=ds_fa, verbose=True)

    def plot_antenna_and_search_area(self, root_img, rcv_in_fullarray=[]):

        root_arrays = os.path.join(root_img, "arrays")
        if not os.path.exists(root_arrays):
            os.makedirs(root_arrays)

        area_square_x = [
            self.simulation.grid_x.min(),
            self.simulation.grid_x.min(),
            self.simulation.grid_x.max(),
            self.simulation.grid_x.max(),
            self.simulation.grid_x.min(),
        ]

        area_square_y = [
            self.simulation.grid_y.min(),
            self.simulation.grid_y.max(),
            self.simulation.grid_y.max(),
            self.simulation.grid_y.min(),
            self.simulation.grid_y.min(),
        ]
        rcv_x = np.append(self.simulation.antenna.x, self.simulation.antenna.x[0])
        rcv_y = np.append(self.simulation.antenna.y, self.simulation.antenna.y[0])

        x_src, y_src = self.simulation.event_ship_x, self.simulation.event_ship_y
        true_pos_label = (
            r"$X_{src} = ( "
            + f"{x_src:.0f}\,"
            + r"\textrm{m},\,"
            + f"{y_src:.0f}\,"
            + r"\textrm{m})$"
        )
        Lx = self.simulation.grid_x.max() - self.simulation.grid_x.min()
        Ly = self.simulation.grid_y.max() - self.simulation.grid_y.min()
        area_label = (
            r"$\mathcal{A} \,("
            + f"L_x = {Lx:.0f}\,"
            + r"\textrm{m},\,"
            + f"L_y = {Ly:.0f}\,"
            + r"\textrm{m})$"
        )

        plt.figure()
        plt.plot(
            rcv_x,
            rcv_y,
            color="k",
            linestyle="--",
            marker="o",
            markersize=10,
            label="Antenna",
        )

        label_offset_pts = (7, 7)  # Shift right and up in display units

        for i in range(self.simulation.antenna.n_elements):

            plt.annotate(
                f"$s_{i+1}$",
                xy=(self.simulation.antenna.x[i], self.simulation.antenna.y[i]),
                xycoords="data",
                xytext=label_offset_pts,
                textcoords="offset points",
                fontsize=18,
            )

        if rcv_in_fullarray:
            # Color selected antenna
            rcv_x_fa = np.append(
                self.simulation.antenna.x[rcv_in_fullarray],
                self.simulation.antenna.x[rcv_in_fullarray][0],
            )
            rcv_y_fa = np.append(
                self.simulation.antenna.y[rcv_in_fullarray],
                self.simulation.antenna.y[rcv_in_fullarray][0],
            )
            plt.plot(
                rcv_x_fa,
                rcv_y_fa,
                color="r",
                marker="o",
                markersize=10,
                linestyle="--",
            )

        plt.plot(area_square_x, area_square_y, color="r", label=area_label)
        plt.scatter(x_src, y_src, color="k", label=true_pos_label, marker="2", s=250)
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        array_label = self.get_array_label(rcv_in_fullarray)
        fpath = os.path.join(root_arrays, f"antenna_search_area_{array_label}.png")
        plt.savefig(fpath, dpi=300)

    def plot_subarrays_ambiguity_surfaces(
        self,
        rcv_couples,
        vmin,
        vmax,
        xticks_pos_m,
        yticks_pos_m,
        cmap="jet",
    ):

        true_pos_label = (
            r"$X_{src} = ( "
            + f"{self.simulation.event_ship_x:.0f}\,"
            + r"\textrm{m},\,"
            + f"{self.simulation.event_ship_y:.0f}\,"
            + r"\textrm{m})$"
        )

        f, axs = plt.subplots(
            2, rcv_couples.shape[0], figsize=(38, 20), sharex=True, sharey=True
        )
        if rcv_couples.shape[0] == 1:
            axs = np.atleast_2d(axs).T  # Ensure axs has necessary shape

        all_rcv_idx = []
        for i_cpl, rcv_cpl in enumerate(rcv_couples):

            # Load data
            # if data_fname is None:
            #     data_fname_cpl = f"loc_zhang_dx{grid['dx']}m_dy{grid['dy']}m_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
            # else:
            #     data_fname_cpl = f"{data_fname}_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"

            # fpath = os.path.join(
            #     root_data,
            #     data_fname_cpl,
            # )
            fpath = (
                self.current_snr_dataset_rootpath
                + f"_s{rcv_cpl[0]+1}_s{rcv_cpl[1]+1}.nc"
            )
            ds_cpl = xr.open_dataset(fpath)

            # Store all rcvs
            all_rcv_idx += list(ds_cpl.idx_rcv)

            if i_cpl == axs.shape[1] - 1:
                cbar_kwargs = {"label": r"$\textrm{[dB]}$"}
                add_colorbar = True
            else:
                cbar_kwargs = {}
                add_colorbar = False

            # Plot d_gcc and d_rtf
            for i, dist in enumerate(["d_gcc", "d_rtf"]):
                ax = axs[i, i_cpl]
                amb_surf = ds_cpl[dist]

                im = amb_surf.plot(
                    x="x",
                    y="y",
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extend="neither",
                    cbar_kwargs=cbar_kwargs,
                    add_colorbar=add_colorbar,
                )

                ax.scatter(
                    self.simulation.event_ship_x,
                    self.simulation.event_ship_y,
                    color="k",
                    # facecolors="none",
                    # edgecolors="k",
                    label=true_pos_label,
                    marker="2",
                    s=250,
                    linewidths=3,
                )

                ax.set_title(
                    r"$s_{"
                    + str(rcv_cpl[0] + 1)
                    + "} - s_{"
                    + str(rcv_cpl[1] + 1)
                    + r"}$"
                )
                if i == 1:
                    ax.set_xlabel(r"$x$" + " [m]")
                else:
                    ax.set_xlabel("")
                if i_cpl == 0:
                    ax.set_ylabel(r"$y$" + " [m]")
                else:
                    ax.set_ylabel("")

                # # Set xticks
                # ax.set_xticks(xticks_pos_m)
                # ax.set_yticks(yticks_pos_m)

        # Sup title with SNR
        all_rcv_idx = np.unique(all_rcv_idx)
        rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in all_rcv_idx]) + "$"
        plt.suptitle(f"SNR = {ds_cpl.snr} dB, Receivers = ({rcv_str})")

        # Save figure
        root_subarrays_comparison = os.path.join(
            self.current_snr_root_img, "subarrays_comparison"
        )
        if not os.path.exists(root_subarrays_comparison):
            os.makedirs(root_subarrays_comparison)

        rcv_lab = "_".join([f"s{id+1}" for id in all_rcv_idx])
        fpath = os.path.join(
            root_subarrays_comparison,
            f"{self.simulation.name}_snr{ds_cpl.snr}dB_rcvs_{rcv_lab}.png",
        )
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close("all")

    def plot_fullarray_ambiguity_surfaces(
        self,
        ds_fa,
        vmin,
        vmax,
        xticks_pos_m,
        yticks_pos_m,
        cmap="jet",
    ):

        true_pos_label = (
            r"$X_{src} = ( "
            + f"{self.simulation.event_ship_x:.0f}\,"
            + r"\textrm{m},\,"
            + f"{self.simulation.event_ship_y:.0f}\,"
            + r"\textrm{m})$"
        )

        titles = {"d_gcc": "DCF", "d_rtf": "RTF"}

        f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # Plot d_gcc and d_rtf
        for i, dist in enumerate(["d_gcc", "d_rtf"]):
            ax = axs[i]
            amb_surf = ds_fa[dist]

            im = amb_surf.plot(
                x="x",
                y="y",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                extend="neither",
                cbar_kwargs={"label": "q [dB]"},
                # robust=True,
                # cbar_kwargs={"label": dist_label},
            )

            # Add colorbar
            ax.scatter(
                self.simulation.event_ship_x,
                self.simulation.event_ship_y,
                color="k",
                label=true_pos_label,
                marker="2",
                s=200,
                linewidths=2,
            )

            ax.set_title(titles[dist])
            ax.set_xlabel(r"$x$" + " [m]")
            if i == 0:
                ax.set_ylabel(r"$y$" + " [m]")
            else:
                ax.set_ylabel("")

            # Set xticks
            # ax.set_xticks(xticks_pos_m)
            # ax.set_yticks(yticks_pos_m)

        root_fullarray_comparison = os.path.join(
            self.current_snr_root_img, "fullarray_comparison"
        )
        if not os.path.exists(root_fullarray_comparison):
            os.makedirs(root_fullarray_comparison)

        rcv_lab = "_".join([f"s{id+1}" for id in ds_fa.idx_rcv])
        fpath = os.path.join(
            root_fullarray_comparison,
            f"{self.simulation.name}_snr{ds_fa.snr}dB_rcvs_{rcv_lab}.png",
        )
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close("all")

    def plot_ambiguity_surface_mainlobe_contour(
        self, ds_fa, vmin, vmax, xticks_pos_m, yticks_pos_m, cmap="jet"
    ):
        # Find mainlobe contours
        masks = self.get_mainlobe_mask(ds_fa)
        mainlobe_contours = {
            dist: self.get_mainlobe_contours(ds_fa[dist], masks[dist])
            for dist in ["d_gcc", "d_rtf"]
        }

        f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        # Plot d_gcc and d_rtf
        for i, dist in enumerate(["d_gcc", "d_rtf"]):
            ax = axs[i]
            amb_surf = ds_fa[dist]

            ax_order = self.get_axis_order(da=amb_surf, ax_names=["x", "y"])

            amb_surf.plot(
                x="x",
                y="y",
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend="neither",
                cbar_kwargs={"label": "[dB]"},
            )

            contour = mainlobe_contours[dist]

            ax.plot(
                ds_fa["x"].values[contour[:, ax_order["x"]].astype(int)],
                ds_fa["y"].values[contour[:, ax_order["y"]].astype(int)],
                color="k",
                linewidth=2,
                label="Contour",
            )

            title = "DCF" if dist == "d_gcc" else "RTF"
            ax.set_title(title)
            ax.set_xlabel(r"$x$" + " [m]")
            if i == 0:
                ax.set_ylabel(r"$y$" + " [m]")
            else:
                ax.set_ylabel("")

            # Set xticks
            # ax.set_xticks(xticks_pos_m)
            # ax.set_yticks(yticks_pos_m)

        # Save figure
        root_mainlobe = os.path.join(self.current_snr_root_img, "mainlobe")
        if not os.path.exists(root_mainlobe):
            os.makedirs(root_mainlobe)

        rcv_lab = "_".join([f"s{id+1}" for id in ds_fa.idx_rcv])
        fpath = os.path.join(
            root_mainlobe,
            f"{self.simulation.name}_mainlobe__snr{ds_fa.snr}dB_rcvs_{rcv_lab}.png",
        )
        plt.legend()
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close("all")

    def plot_ambiguity_surface_distribution(self, ds_fa):
        """
        Plot the distribution of the ambiguity surfaces for d_gcc and d_rtf
        """
        f, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        percentile_threshold = 0.995
        bins = {"d_gcc": ds_fa["d_gcc"].size // 10, "d_rtf": ds_fa["d_rtf"].size // 10}

        # Plot d_gcc and d_rtf
        mainlobe_th = {}
        for i, dist in enumerate(["d_gcc", "d_rtf"]):
            ax = axs[i]
            amb_surf = ds_fa[dist]

            amb_surf.plot.hist(ax=ax, bins=bins[dist], alpha=0.5, color="b")

            # Vertical line representing the percentile threshold
            percentile = np.percentile(amb_surf.values, percentile_threshold * 100)
            mainlobe_th[dist] = percentile
            ax.axvline(
                percentile,
                color="r",
                linestyle="--",
                label=f"{percentile_threshold*100:.0f}th percentile",
            )

            ax.set_title("Full array")
            ax.set_xlim(-20, 0)
            ax.set_xlabel("[dB]")

        # Save figure
        root_dist = os.path.join(
            self.current_snr_root_img, "ambiguity_surf_distribution"
        )
        if not os.path.exists(root_dist):
            os.makedirs(root_dist)

        fpath = os.path.join(root_dist, f"{self.simulation.name}_dist.png")
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close("all")

    def get_amb_surf_cmap(loc_arg):
        if loc_arg == "max":
            cmap = "jet"
        elif loc_arg == "min":
            cmap = "jet_r"
        return cmap

    def plot_ambiguity_surface(self, amb_surf, plot_args, loc_arg, folder_name=""):

        dist = plot_args["dist"]
        testcase = plot_args["testcase"]
        root_img = plot_args["root_img"]
        dist_label = plot_args["dist_label"]
        vmax = plot_args["vmax"]
        vmin = plot_args["vmin"]
        sub_array = plot_args["sub_array"]

        # To plot the hyperbola corresponding to TDOA
        add_hyperbola = plot_args.get("add_hyperbola", False)
        hyperbola_cpls = plot_args.get("hyperbola_cpls", None)
        # To plot the circle centered on the center of the antenna array and passing through the source
        add_circle = plot_args.get("add_circle", False)

        # Source position
        x_src = self.simulation.event_ship_x
        y_src = self.simulation.event_ship_y

        # get estimated source position
        _, _, x_src_hat, y_src_hat = self.get_estimated_src_pos(
            amb_surf=amb_surf, loc_arg=loc_arg
        )

        if self.simulation.verbose:
            print("True source position: ", x_src, y_src)
            print(
                "Estimated source position: ",
                np.round(x_src_hat.values, 1),
                np.round(y_src_hat.values, 1),
            )

        cmap = self.get_amb_surf_cmap(loc_arg)

        plt.figure(figsize=(14, 12))
        amb_surf.plot(
            x="x",
            y="y",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # aspect="equal",
            extend="neither",
            # robust=True,
            cbar_kwargs={"label": dist_label},
            # cbar_kwargs={"label": r"$\textrm{[dB]}$"},
        )

        # Estimated source position
        estimated_pos_label = (
            r"$\hat{X}_{src} = ( "
            + f"{x_src_hat:.0f}\,"
            + r"\textrm{m},\,"
            + f"{y_src_hat:.0f}\,"
            + r"\textrm{m})$"
        )
        estimated_pos_label = (
            r"$\hat{X}_{src}" + f" = ({x_src_hat:.2f}, {y_src_hat:.2f})$"
        )
        plt.scatter(
            x_src_hat,
            y_src_hat,
            color="w",
            marker="o",
            label=estimated_pos_label,
            s=400,
            linewidths=4,
        )

        # True source position
        true_pos_label = (
            r"$X_{src} = ( "
            + f"{x_src:.0f}\,"
            + r"\textrm{m},\,"
            + f"{y_src:.0f}\,"
            + r"\textrm{m})$"
        )

        plt.scatter(
            x_src,
            y_src,
            color="k",
            # facecolors="none",
            # edgecolors="k",
            label=true_pos_label,
            marker="2",
            s=400,
            linewidths=4,
        )

        # # Add receiver positions
        # _, receivers, _, grid, _, _ = params(antenna_type=antenna_type)
        # x_rcv = np.concatenate([self.simulation.antenna.x, [self.simulation.antenna.x[0]]])
        # y_rcv = np.concatenate([self.simulation.antenna.y, [self.simulation.antenna.y[0]]])
        # plt.plot(
        #     x_rcv,
        #     y_rcv,
        #     color="k",
        #     marker="o",
        #     linestyle="--",
        #     markersize=7,
        #     # label=[f"$s_{i}$" for i in range(self.simulation.antenna.n_elements)],
        # )

        # txt_offset = 100
        # sgn_y = [-1, -1, 0, 0, 0, 0]
        # sgn_x = [0, 0, 1.5, 1.5, -1.5, -1.5]
        # for i, txt in enumerate([f"$s_{i+1}$" for i in range(self.simulation.antenna.n_elements)]):
        #     plt.annotate(
        #         txt,
        #         (self.simulation.antenna.x[i], self.simulation.antenna.y[i]),
        #         # (self.simulation.antenna.x[i] + sgn_x[i] * 50, self.simulation.antenna.y[i] + sgn_y[i] * 50),
        #         fontsize=16,
        #     )
        #     # plt.text(
        #     #     self.simulation.antenna.x[i] + sgn_x[i] * txt_offset,
        #     #     self.simulation.antenna.y[i] + sgn_y[i] * txt_offset,
        #     #     txt,
        #     #     fontsize=16,
        #     # )

        # Add hyperbola if required
        if add_hyperbola:
            # print("Add hyperbola")
            src_pos = (x_src, y_src)

            if hyperbola_cpls is None:
                # Compute hyperbola for each pair of receivers
                # default_cpls = [[0, 2], [1, 4], [3, 5]]
                hyperbola_cpls = self.get_rcv_couples(self.simulation.antenna.rcv_idx)

            for i, sa in enumerate(hyperbola_cpls):
                receiver1 = (
                    self.simulation.antenna.x[sa[0]],
                    self.simulation.antenna.y[sa[0]],
                )
                receiver2 = (
                    self.simulation.antenna.x[sa[1]],
                    self.simulation.antenna.y[sa[1]],
                )
                (right_branch, left_branch) = compute_hyperbola(
                    receiver1, receiver2, src_pos, num_points=1000, tmax=10
                )

                # Plot both branches
                plt.plot(
                    right_branch[0], right_branch[1], "k", linestyle="--", zorder=15
                )
                plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=16)

            # else:
            #     receiver1 = (
            #         self.simulation.antenna.x[hyperbola_cpls[0]],
            #         self.simulation.antenna.y[hyperbola_cpls[0]],
            #     )
            #     receiver2 = (
            #         self.simulation.antenna.x[hyperbola_cpls[1]],
            #         self.simulation.antenna.y[hyperbola_cpls[1]],
            #     )
            #     (right_branch, left_branch) = compute_hyperbola(
            #         receiver1, receiver2, src_pos, tmax=5
            #     )

            #     # Plot both branches
            #     plt.plot(right_branch[0], right_branch[1], "k", linestyle="--", zorder=10)
            #     plt.plot(left_branch[0], left_branch[1], "k", linestyle="--", zorder=10)

        # Add circle if required
        if add_circle:
            barycentre_x = np.mean(self.simulation.antenna.x)
            barycentre_y = np.mean(self.simulation.antenna.y)
            radius = np.sqrt((barycentre_x - x_src) ** 2 + (barycentre_y - y_src) ** 2)

            circle = plt.Circle(
                (barycentre_x, barycentre_y),
                radius,
                color="k",
                fill=False,
                linestyle="--",
                linewidth=2,
                label=r"$\mathcal{C}((\hat{x_r}, \hat{y_r}), r_{s})$",
            )
            plt.gca().add_artist(circle)

        # plt.xlim([self.simulation.grid_x.min(), self.simulation.grid_x.max()])
        # plt.ylim([self.simulation.grid_y.min(), self.simulation.grid_y.max()])

        # plt.axis("equal")
        sub_array = amb_surf.idx_rcv
        rcv_str = "$" + ", \,".join([f"s_{id+1}" for id in sub_array]) + "$"
        plt.title(f"SNR = {amb_surf.attrs['snr']} dB, Receivers = ({rcv_str})")
        plt.xlabel(r"$x$" + " [m]")
        plt.ylabel(r"$y$" + " [m]")
        plt.legend()

        # Save figure
        root_amb_surf = os.path.join(root_img, folder_name)
        if not os.path.exists(root_amb_surf):
            os.makedirs(root_amb_surf)

        sa_lab = (
            ""
            if sub_array is None
            else "_" + "_".join([f"s{sa+1}" for sa in sub_array])
        )
        fname = f"{testcase}_ambiguity_surface_{dist}{sa_lab}.png"
        fpath = os.path.join(root_amb_surf, fname)
        plt.savefig(fpath)
        plt.close("all")

    @classmethod
    def get_estimated_src_pos(cls, amb_surf, loc_arg):

        ax_order = cls.get_axis_order(da=amb_surf, ax_names=["x", "y"])

        # Estimated source position defined as one of the extremum of the ambiguity surface
        if loc_arg == "max":
            idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        elif loc_arg == "min":
            idx = np.unravel_index(np.argmin(amb_surf.values), amb_surf.shape)

        # Make sure we take coords in the right order
        x_idx = idx[ax_order["x"]]
        y_idx = idx[ax_order["y"]]

        # Extract estimated source pos
        x_src_hat = amb_surf.x[x_idx]
        y_src_hat = amb_surf.y[y_idx]

        return x_idx, y_idx, x_src_hat, y_src_hat

    @staticmethod
    def get_amb_surf_cmap(loc_arg):
        if loc_arg == "max":
            cmap = "jet"
        elif loc_arg == "min":
            cmap = "jet_r"
        return cmap

    @classmethod
    def get_mainlobe_mask(cls, ds):
        # Cast to uint8
        image_dcf = 10 ** (ds.d_gcc.values / 10)
        image_dcf = img_as_ubyte(image_dcf)

        image_rtf = 10 ** (ds.d_rtf.values / 10)
        image_rtf = img_as_ubyte(image_rtf)

        # Smooth images
        disk_size = 1
        image_dcf = rank.median(image_dcf, disk(disk_size))
        image_rtf = rank.median(image_rtf, disk(disk_size))

        # Number of clusters
        n_clusters = 5
        # Kmeans avec seulement les intensités des pixels
        X_dcf = image_dcf.flatten()[np.newaxis, :]
        X_rtf = image_rtf.flatten()[np.newaxis, :]

        # Apply K-means to get labels for each pixel
        labels_dcf = cls.apply_kmeans(X_dcf, n_clusters)
        labels_rtf = cls.apply_kmeans(X_rtf, n_clusters)

        # Reshape labels to 2d arrays
        labels_dcf = labels_dcf.reshape(ds.d_gcc.shape)
        labels_rtf = labels_rtf.reshape(ds.d_rtf.shape)

        # Define mask for regions belonging to the class of the estimated source position
        mask_dcf = labels_dcf == cls.get_src_pos_label(
            amb_surf=ds.d_gcc, labels=labels_dcf
        )
        mask_rtf = labels_rtf == cls.get_src_pos_label(
            amb_surf=ds.d_rtf, labels=labels_rtf
        )

        # New labels to get the region of interest
        mask_labeled_dcf, nlabel_dcf = label(mask_dcf)
        mask_labeled_rtf, nlabel_dcf = label(mask_rtf)

        # Get labels of the region of interest
        label_dcf = cls.get_src_pos_label(amb_surf=ds.d_gcc, labels=mask_labeled_dcf)
        label_rtf = cls.get_src_pos_label(amb_surf=ds.d_rtf, labels=mask_labeled_rtf)

        # Create new masks with only the region of interest
        mask_dcf_new = mask_labeled_dcf == label_dcf
        mask_rtf_new = mask_labeled_rtf == label_rtf

        # Remove holes
        mask_dcf_new = ndi.binary_fill_holes(mask_dcf_new)
        mask_rtf_new = ndi.binary_fill_holes(mask_rtf_new)

        # Expand regions
        mask_dcf_new = binary_dilation(mask_dcf_new, iterations=2)
        mask_rtf_new = binary_dilation(mask_rtf_new, iterations=2)
        masks = {"d_gcc": mask_dcf_new, "d_rtf": mask_rtf_new}

        return masks

    @staticmethod
    def apply_kmeans(X, n_clusters):
        # Normalize features
        X_norm = preprocessing.normalize(X).T
        # Apply kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(X_norm)

        # Get labels
        labels = kmeans.labels_

        return labels

    @classmethod
    def get_src_pos_label(cls, amb_surf, labels):
        x_idx, y_idx, _, _ = cls.get_estimated_src_pos(amb_surf=amb_surf, loc_arg="max")
        ax_order = cls.get_axis_order(da=amb_surf, ax_names=["x", "y"])
        idx_tuple = (y_idx, x_idx) if (ax_order["y"] == 0) else (x_idx, y_idx)
        src_hat_class = labels[idx_tuple]
        return src_hat_class

    def estimate_msr(self, ds):

        ### Define the mainlobe mask ###
        masks = self.get_mainlobe_mask(ds)

        ### Compute MSR ###
        msr = {}
        pos_hat = {}
        for i, dist in enumerate(["d_gcc", "d_rtf"]):

            amb_surf = ds[dist]
            # Source pos
            x_idx, y_idx, x_src_hat, y_src_hat = self.get_estimated_src_pos(
                amb_surf=amb_surf, loc_arg="max"
            )
            pos_hat[dist] = {"x": x_src_hat.values, "y": y_src_hat.values}

            mainlobe_mask = masks[dist]

            # Compute mainlobe to side lobe ratio
            main_lobe = np.max(amb_surf.values)
            side_lobe = np.max(amb_surf.values[~mainlobe_mask])
            msr[dist] = -(
                main_lobe - side_lobe
            )  # MSR = mainlobe_dB - side_lobe_dB  (- to fit with negative results presented by Zhang et al 2023)

            if self.simulation.verbose:
                print(f"MSR {dist} : {msr[dist]:.2f} dB")

        return msr, pos_hat

    @classmethod
    def get_estimated_src_pos(cls, amb_surf, loc_arg):

        ax_order = cls.get_axis_order(da=amb_surf, ax_names=["x", "y"])

        # Estimated source position defined as one of the extremum of the ambiguity surface
        if loc_arg == "max":
            idx = np.unravel_index(np.argmax(amb_surf.values), amb_surf.shape)
        elif loc_arg == "min":
            idx = np.unravel_index(np.argmin(amb_surf.values), amb_surf.shape)

        # Make sure we take coords in the right order
        x_idx = idx[ax_order["x"]]
        y_idx = idx[ax_order["y"]]

        # Extract estimated source pos
        x_src_hat = amb_surf.x[x_idx]
        y_src_hat = amb_surf.y[y_idx]

        return x_idx, y_idx, x_src_hat, y_src_hat

    @staticmethod
    def get_rcv_couples(idx_receivers):
        """
        Get all possible receiver couples
        """
        # rcv_couples = []
        # for i in idx_receivers:
        #     for j in idx_receivers:
        #         if j > i:
        #             rcv_couples.append([i, j])
        # rcv_couples = np.array(rcv_couples)

        rcv_couples = np.array(list(combinations(idx_receivers, 2)))
        rcv_couples = np.atleast_2d(rcv_couples)  # In case only two receivers

        return rcv_couples

    @classmethod
    def get_mainlobe_contours(cls, amb_surf, mask):
        x_idx, y_idx, _, _ = cls.get_estimated_src_pos(amb_surf=amb_surf, loc_arg="max")
        ax_order = cls.get_axis_order(da=amb_surf, ax_names=["x", "y"])

        # Find contours of src_hat_class and select the contour corresponding to the estimated position
        contours = measure.find_contours(mask, level=0.5)
        for contour in contours:
            # Check if src_hat is within the contour
            idx_x_min = np.min(contour[:, ax_order["x"]].astype(int))
            idx_x_max = np.max(contour[:, ax_order["x"]].astype(int))
            idx_y_min = np.min(contour[:, ax_order["y"]].astype(int))
            idx_y_max = np.max(contour[:, ax_order["y"]].astype(int))
            if (idx_x_min <= x_idx <= idx_x_max) and (idx_y_min <= y_idx <= idx_y_max):
                break

        return contour


if __name__ == "__main__":
    from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
    from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation

    debug = False
    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    simu = Simulation(debug=debug, antenna=antenna)

    lp = LocalizationProcessor(simulation=simu)

    snrs = [0, 10, 20]
    lp.process_multiple_snrs(snrs=snrs)
