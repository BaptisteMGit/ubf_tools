#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   Untitled-1
@Time    :   2026/09/29 08:57:41
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.io import loadmat
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

if os.name == "nt":  # Windows
    ROOT_PROGRAM = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
else:  # Linux
    ROOT_PROGRAM = "/home/program/ubf_tools"

from misc import equivalent_celerity
from publication.publication_figure import set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.acouplane_cst import *
from real_data_analysis.acouplane.src.acouplane_utils import *

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 4
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_INTERSECTION = os.path.join(ROOT_IMG_PORTION, "intersections")

ROOT_IMG_PORTION_FEATURES = os.path.join(ROOT_IMG_PORTION, "features")
ROOT_IMG_PORTION_FEATURES_RTF = os.path.join(ROOT_IMG_PORTION_FEATURES, "rtf")
ROOT_IMG_PORTION_FEATURES_DCF = os.path.join(ROOT_IMG_PORTION_FEATURES, "dcf")

ROOT_IMG_PORTION_INPUT_SIGNAL = os.path.join(ROOT_IMG_PORTION, "signal")
ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE = os.path.join(
    ROOT_IMG_PORTION_INPUT_SIGNAL, "pressure"
)
ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM = os.path.join(
    ROOT_IMG_PORTION_INPUT_SIGNAL, "spectrogram"
)

roots = [
    ROOT_IMG_PORTION,
    ROOT_IMG_PORTION_INTERSECTION,
    ROOT_IMG_PORTION_FEATURES,
    ROOT_IMG_PORTION_FEATURES_RTF,
    ROOT_IMG_PORTION_FEATURES_DCF,
    ROOT_IMG_PORTION_INPUT_SIGNAL,
    ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
    ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM,
]
for root in roots:
    os.makedirs(root, exist_ok=True)

c0 = 1508  # Equivalent celerity derived from real profile

nfft = 2**15
offset_ms = 100
signal_duration_ms = 3000

reference_rcv = "obs1"
obs_ids = [
    "obs1",
    "obs2",
    "obs4",
    "obs5",
    "obs6",
    "obs7",
]
# Frequency band
fmin = 10
fmax = 20

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
save_modification_to_portion = False
apply_obs_pos_correction = False
compute_cpsds = True
apply_mfp_strategy_intersections = True

plot_portion_position = False
plot_signal = False
plot_spectrogram = False
plot_signal_single_emission = False
plot_features = False
verbose = True

# ======================================================================================================================
# Load data
# ======================================================================================================================
if verbose:
    print(f"Loading data from portion {portion_id}...")
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = xr.open_datatree(portion_fpath)
print(portion)

obs_ids = np.asarray(obs_ids)

# NOTE : use corrected position (LMS estimation)
if apply_obs_pos_correction:
    if verbose:
        print("Correcting OBS position using LMS estimation...")
    apply_rcv_position_correction_to_portion(portion_tree=portion)

# ======================================================================================================================
# Plot portion positions
# ======================================================================================================================
if plot_portion_position:
    if verbose:
        print("Plotting portion positions...")
    fig, ax = plot_portion_tir_pos(portion_tree=portion)
    fpath = os.path.join(ROOT_IMG_PORTION, f"portion_{portion_id}_position.png")
    fig.savefig(fpath)

# ======================================================================================================================
# Plot signals
# ======================================================================================================================

single_fig = False
if plot_signal:
    if verbose:
        print("Plotting section signals...")
    for name, node in portion.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        fig, axs = plot_pressure_section(
            ds_pressure_section=ds_pressure_section, single_fig=single_fig
        )
        fig.suptitle(
            f"{ds_pressure_section.section_orientation} {ds_pressure_section.section_id}"
        )
        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
            f"pressure_signal_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)


# ======================================================================================================================
# Plot spectrograms
# ======================================================================================================================
if plot_spectrogram:
    if verbose:
        print("Plotting sections spectrograms...")
    for name, node in portion.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        fig, axs = plot_spectrogram_section(
            ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"
        )
        fig.suptitle(
            f"{ds_pressure_section.section_orientation} {ds_pressure_section.section_id}"
        )
        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM,
            f"pressure_spectrogram_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)

# ======================================================================================================================
# Derive propagation time
# ======================================================================================================================
if verbose:
    print("Propagation time estimation...")

derive_portion_propagation_time(portion_tree=portion, c0=c0)


# ======================================================================================================================
# Plot single emission and associated arrrival time
# ======================================================================================================================

# obs_ids = list(portion.pressure.children.items())[0][1].to_dataset().obs_id.values
# obs_ids = obs_ids[[0, 3, 5]]
if plot_signal_single_emission:
    if verbose:
        print("Plotting single emission...")

    for name, node in portion.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        # Associated ds_pos
        ds_pos_section = portion.position[name].to_dataset()
        tir_dt = ds_pos_section["tir_reception_datetime"].isel(src_datetime=slice(0, 1))

        fig, axs = plt.subplots(
            nrows=obs_ids.size, ncols=1, figsize=(16, 14), sharex=False
        )
        axs = axs.flatten()
        for i, rcv_id in enumerate(obs_ids):
            tir_dt_rcv = tir_dt.sel(rcv_id=rcv_id)
            # Select associated data window
            press_rcv = ds_pressure_section[f"pressure_{rcv_id}"].sel(
                {
                    f"time_{rcv_id}": slice(
                        tir_dt_rcv.values[0] - np.timedelta64(offset_ms, "ms"),
                        tir_dt_rcv.values[0]
                        + np.timedelta64(signal_duration_ms - offset_ms, "ms"),
                    )
                }
            )

            # Normalize
            press_rcv = press_rcv / press_rcv.max()
            press_rcv.plot(label=rcv_id.upper(), color=color(i), ax=axs[i])
            # Plot arrivals time
            axs[i].vlines(x=tir_dt_rcv, ymin=-1, ymax=1, color="k", linestyle="--")
            # Add legend
            axs[i].legend(loc="upper right")

        xlab = axs[0].get_xlabel()
        ylab = axs[0].get_ylabel()

        for ax in axs:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        fig.suptitle(
            f"{ds_pressure_section.section_orientation} {ds_pressure_section.section_id}"
        )
        fig.supxlabel(xlab)
        fig.supylabel(ylab)

        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
            f"pressure_signal_single_emission_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)

# ======================================================================================================================
# Plot single arrival and arrrival time
# ======================================================================================================================

if verbose:
    print("Derive CPSD...")

if compute_cpsds:
    derive_portion_cpsd(
        portion_tree=portion,
        offset_ms=offset_ms,
        signal_duration_s=int(signal_duration_ms * 1e-3),
        nfft=nfft,
        obs_ids=None,
        fmin=fmin,
        fmax=fmax,
    )
else:
    # Assert cpsd variable has already been computed
    assert "feature" in list(
        portion.children
    ), "No feature in portion dataset, use compute_cpsds = True !"


# ======================================================================================================================
# Plot features (RTF, DCF)
# ======================================================================================================================
if plot_features:

    if verbose:
        print("Plotting features (DCF, RTF)...")

    alpha_max = 0.5
    ref_rcv_id = reference_rcv

    # print(axs.shape)
    for name, node in portion.feature.children.items():
        ds_feature_section = node.to_dataset()

        # Slice frequency band
        ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

        sjj = ds_feature_section.sel(rcv_i=ref_rcv_id, rcv_j=ref_rcv_id).Sxy
        # Per convention : 2nd index = ref

        # RTF
        fig_rtf, axs_rtf = plt.subplots(
            nrows=obs_ids.size - 1, ncols=1, figsize=(16, 16), sharex=True, sharey=True
        )
        axs_rtf = axs_rtf.flatten()
        # RTF
        fig_dcf, axs_dcf = plt.subplots(
            nrows=obs_ids.size - 1, ncols=2, figsize=(16, 16), sharex=True, sharey=True
        )

        iplot = 0
        for i_rcv, rcv_id_i in enumerate(obs_ids):

            if rcv_id_i == ref_rcv_id:
                continue
            # Sii
            sii = ds_feature_section.sel(rcv_i=rcv_id_i, rcv_j=rcv_id_i).Sxy
            # Sij
            sij = ds_feature_section.sel(rcv_i=rcv_id_i, rcv_j=ref_rcv_id).Sxy

            # RTF
            denom = sjj.values
            denom[denom == 0] = np.nan  # Replace 0 by nan to avoid warnings
            pi_ij = sij.values / denom

            # pi_ij = sij.values / sjj.values
            gamma_ij = 20 * np.log10(np.abs(pi_ij))

            axs_rtf[iplot].pcolormesh(
                sij.freq,
                sij.src_datetime,
                gamma_ij,
                cmap="magma",
                vmin=-alpha_max * np.nanmax(gamma_ij),
                vmax=+alpha_max * np.nanmax(gamma_ij),
            )

            axs_rtf[iplot].set_xlabel("")
            axs_rtf[iplot].set_ylabel("")
            axs_rtf[iplot].set_title(rcv_id_i.upper(), fontsize=12)

            # DCF (sxy SCOT)
            denom = sii.values * sjj.values
            denom[denom == 0] = np.nan
            dcf_ij = sij.values / np.sqrt(denom)

            # dcf_ij = sij.values / np.sqrt(sii.values * sjj.values)
            axs_dcf[iplot, 0].pcolormesh(
                sij.freq,
                sij.src_datetime,
                np.real(dcf_ij),
                cmap="magma",
                vmin=-1.2,
                vmax=1.2,
            )
            axs_dcf[iplot, 1].pcolormesh(
                sij.freq,
                sij.src_datetime,
                np.imag(dcf_ij),
                cmap="magma",
                vmin=-1.2,
                vmax=1.2,
            )

            iplot += 1

        fig_rtf.supxlabel("Frequency [Hz]")
        fig_rtf.suptitle(
            f"{ds_feature_section.section_orientation} {ds_feature_section.section_id}"
        )

        fpath = os.path.join(
            ROOT_IMG_PORTION_FEATURES_RTF,
            f"rtf_map_{name}.png",
        )
        fig_rtf.savefig(fpath)
        plt.close(fig_rtf)

        fig_dcf.supxlabel("Frequency [Hz]")
        fig_dcf.suptitle(
            f"{ds_feature_section.section_orientation} {ds_feature_section.section_id}"
        )
        axs_dcf[0, 0].set_title("Real part")
        axs_dcf[0, 1].set_title("Imaginary part")

        fpath = os.path.join(
            ROOT_IMG_PORTION_FEATURES_DCF,
            f"dcf_map_{name}.png",
        )
        fig_dcf.savefig(fpath)
        plt.close(fig_dcf)


# ======================================================================================================================
# Derive features NOTE : should be before plot and plot should use the result
# ======================================================================================================================


def distances_single_intersection(ds_feature_section1, ds_feature_section2):

    # Compute real distance in meters between emissions from section 1 to emission from section 2
    dist_in_real_space_12 = get_distance_between_sections(
        ds_feature_section1, ds_feature_section2
    )

    # Section 1 vs 2
    dist_theta_arr = []
    dist_theta_gamma_arr = []
    dist_L2_gamma_arr = []
    dist_dcf_arr = []

    for i, rcv_id_i in enumerate(ds_feature_section1.rcv_id_feature.values):

        ds_feature_section1_rcv_i = ds_feature_section1.sel(rcv_id_feature=rcv_id_i)
        ds_feature_section2_rcv_i = ds_feature_section2.sel(rcv_id_feature=rcv_id_i)
        # Extract features
        features_section1 = dict(
            pi_ij=ds_feature_section1_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_section1_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_section1_rcv_i.sij_scot.values,
        )
        features_section2 = dict(
            pi_ij=ds_feature_section2_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_section2_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_section2_rcv_i.sij_scot.values,
        )
        # Compute distances between section 1 and 2
        dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf = (
            derive_feature_distance_between_sections(
                features_section1, features_section2
            )
        )

        # Store
        dist_theta_arr.append(dist_theta)
        dist_theta_gamma_arr.append(dist_theta_gamma)
        dist_L2_gamma_arr.append(dist_L2_gamma)
        dist_dcf_arr.append(dist_dcf)

    # Convert to arrays and group
    distance_section12 = dict(
        dist_in_real_space=dist_in_real_space_12,
        dist_theta=np.asarray(dist_theta_arr),
        dist_theta_gamma=np.asarray(dist_theta_gamma_arr),
        dist_L2_gamma=np.asarray(dist_L2_gamma_arr),
        dist_dcf=np.asarray(dist_dcf_arr),
    )  # General shape -> (nrcv, n_emission1, n_emission2)

    return distance_section12


def all_features_single_section(ds_feature_section, rcv_ref="obs1", fmin=5, fmax=5):

    rcv_i_arr = list(ds_feature_section.rcv_i.values)
    rcv_i_arr.remove(rcv_ref)

    # Section
    pi_1_arr = []
    gamma_1_arr = []
    sij_scot_1_arr = []

    for i, rcv_id_i in enumerate(rcv_i_arr):
        # Compute features (RTF, DCF) of the section
        pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
            ds_feature_section=ds_feature_section,
            fmin=fmin,
            fmax=fmax,
            rcv_ref=rcv_ref,
            rcv_i=rcv_id_i,
        )
        # Store
        pi_1_arr.append(pi_ij_a)
        gamma_1_arr.append(gamma_a)
        sij_scot_1_arr.append(sij_scot_a)

    # Convert to arrays and group
    features_section = dict(
        rcv_id=rcv_i_arr,
        pi_ij=np.asarray(pi_1_arr),
        gamma_ij=np.asarray(gamma_1_arr),
        sij_scot=np.asarray(sij_scot_1_arr),
    )  # General shape -> (nrcv, n_emission, nfreq)

    return features_section


def build_mfp_intersection_dataset(
    distance_section12,
    ds_feature_section1,
    ds_feature_section2,
    section_name1,
    section_name2,
):
    ds_mfp = xr.Dataset(
        data_vars=dict(
            d_real=(
                [
                    "src_datetime_section1",
                    "src_datetime_section2",
                ],
                distance_section12["dist_in_real_space"],
            ),
            d_theta=(
                [
                    "rcv_id_feature",
                    "src_datetime_section1",
                    "src_datetime_section2",
                ],
                distance_section12["dist_theta"],
            ),
            d_theta_gamma=(
                [
                    "rcv_id_feature",
                    "src_datetime_section1",
                    "src_datetime_section2",
                ],
                distance_section12["dist_theta_gamma"],
            ),
            d_L2_gamma=(
                [
                    "rcv_id_feature",
                    "src_datetime_section1",
                    "src_datetime_section2",
                ],
                distance_section12["dist_L2_gamma"],
            ),
            d_dcf=(
                [
                    "rcv_id_feature",
                    "src_datetime_section1",
                    "src_datetime_section2",
                ],
                distance_section12["dist_dcf"],
            ),
        ),
        coords=dict(
            rcv_id_feature=ds_feature_section1.rcv_id_feature.values,
            src_datetime_section1=ds_feature_section1.src_datetime.values,
            src_datetime_section2=ds_feature_section2.src_datetime.values,
        ),
        attrs=dict(
            description="MFP result for a single intersection.",
            campagne=ds_feature_section1.campagne,
            section1=section_name1,
            section2=section_name2,
            reference_receiver=ds_feature_section1.reference_receiver,
            obs_position=ds_feature_section1.obs_position,
            obs_position_filepath=ds_feature_section1.obs_position_filepath,
        ),
    )

    # Add attrs to variables
    ds_mfp.d_real.attrs = dict(
        units="m",
        long_name="Distance",
    )
    ds_mfp.d_theta.attrs = dict(
        # units="",
        long_name=r"$\theta_{\Pi}$",
        description="Hermitian angle between RTF vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_theta_gamma.attrs = dict(
        # units="",
        long_name=r"$\theta_{\gamma}$",
        description="Euclidean angle between RTF gamma vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_L2_gamma.attrs = dict(
        # units="",
        long_name=r"$L_{2_{\gamma}}$",
        description="Normalized euclidean distance (L2) between RTF gamma vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_dcf.attrs = dict(
        # units="",
        long_name=r"C_{el}",
        description="Normalized intercorrelation between intercorrelation vectors (DCF)",
        minval=0,
        maxval=1,
    )

    return ds_mfp


def apply_mfp_strategy_to_portion_intersections(
    portion, rcv_ref="obs1", fmin=5, fmax=50
):
    assert "feature" in list(
        portion.children
    ), "No feature in portion dataset, use compute_cpsds = True !"

    # Step 1 : list available sections
    sections = list(portion.feature.children)

    # Step 2 : derive all features to avoid duplicated call to derive_section_features
    for _, feature_node in portion.feature.children.items():
        ds_feature_section = feature_node.to_dataset()

        # Derive features
        features_section = all_features_single_section(
            ds_feature_section=ds_feature_section,
            rcv_ref=rcv_ref,
            fmin=fmin,
            fmax=fmax,
        )  # (nrcv, n_emission, nfreq)

        # Add coords
        ds_feature_section["rcv_id_feature"] = features_section["rcv_id"]

        # Store in dataset
        ds_feature_section["pi_ij"] = (
            ["rcv_id_feature", "src_datetime", "freq"],
            features_section["pi_ij"],
        )
        ds_feature_section["gamma_ij"] = (
            ["rcv_id_feature", "src_datetime", "freq"],
            features_section["gamma_ij"],
        )
        ds_feature_section["sij_scot"] = (
            ["rcv_id_feature", "src_datetime", "freq"],
            features_section["sij_scot"],
        )

        # Add attributes
        ds_feature_section.attrs["reference_receiver"] = rcv_ref

        # Update portion
        feature_node.ds = ds_feature_section

    # Step 3
    meridional_sections = [sec for sec in sections if sec.startswith("sn")]
    zonal_sections = [sec for sec in sections if sec.startswith("oe")]

    # Intersection are between zonal and meridional sections
    for section_name1 in meridional_sections:
        ds_feature_section1 = portion.feature[section_name1].to_dataset()

        for section_name2 in zonal_sections:
            ds_feature_section2 = portion.feature[section_name2].to_dataset()

            # Compute distance (in real space and in features space)
            distance_section12 = distances_single_intersection(
                ds_feature_section1=ds_feature_section1,
                ds_feature_section2=ds_feature_section2,
            )

            # Build dataset to save
            ds_mfp = build_mfp_intersection_dataset(
                distance_section12=distance_section12,
                ds_feature_section1=ds_feature_section1,
                ds_feature_section2=ds_feature_section2,
                section_name1=section_name1,
                section_name2=section_name2,
            )

            # Add to portion
            new_path = f"/mfp_intersection/{section_name1}/{section_name2}"
            portion[new_path] = ds_mfp


if apply_mfp_strategy_intersections:
    if verbose:
        print("Apply MFP strategy to each intersection...")
    apply_mfp_strategy_to_portion_intersections(portion)


if save_modification_to_portion:
    if verbose:
        print("Saving results...")
    portion.to_netcdf(portion_fpath)


# def combine_rcv(rcv_ref="obs1", rcv_i_arr=["obs2"], fmin=5, fmax=50):
#     ref_rcv_id = rcv_ref

#     for i, rcv_id_i in enumerate(rcv_i_arr):

#         pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
#             ds_feature_section=ds_feature_section1,
#             fmin=fmin,
#             fmax=fmax,
#             rcv_ref=ref_rcv_id,
#             rcv_i=rcv_id_i,
#         )
#         pi_ij_b, gamma_b, sij_scot_b = derive_section_features(
#             ds_feature_section=ds_feature_section2,
#             fmin=fmin,
#             fmax=fmax,
#             rcv_ref=ref_rcv_id,
#             rcv_i=rcv_id_i,
#         )

#         features_section1 = dict(pi_ij=pi_ij_a, gamma_ij=gamma_a, sij_scot=sij_scot_a)
#         features_section2 = dict(pi_ij=pi_ij_b, gamma_ij=gamma_b, sij_scot=sij_scot_b)

#         dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf = (
#             derive_feature_distance_between_sections(
#                 features_section1, features_section2
#             )
#         )

#         if i == 0:
#             combined_dist_theta = dist_theta
#             combined_dist_theta_gamma = dist_theta_gamma
#             combined_dist_L2_gamma = dist_L2_gamma
#             combined_dist_dcf = dist_dcf
#         else:
#             combined_dist_theta += dist_theta
#             combined_dist_theta_gamma += dist_theta_gamma
#             combined_dist_L2_gamma += dist_L2_gamma
#             combined_dist_dcf += dist_dcf

#     n_rcv_pairs = len(rcv_i_arr)
#     combined_dist_theta = combined_dist_theta / n_rcv_pairs
#     combined_dist_theta_gamma = combined_dist_theta_gamma / n_rcv_pairs
#     combined_dist_L2_gamma = combined_dist_L2_gamma / n_rcv_pairs
#     combined_dist_dcf = combined_dist_dcf / n_rcv_pairs

#     return (
#         combined_dist_theta,
#         combined_dist_theta_gamma,
#         combined_dist_L2_gamma,
#         combined_dist_dcf,
#     )
