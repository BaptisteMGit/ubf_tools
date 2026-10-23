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
import shutil
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
portion_id = 1
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
fmin = 0
fmax = 200

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
save_modification_to_portion = False
apply_obs_pos_correction = True
compute_cpsds = True
apply_mfp_strategy_intersections = True

plot_portion_position = True
plot_signal = False
plot_spectrogram = False
plot_signal_single_emission = False
plot_features = True
verbose = True

# ======================================================================================================================
# Load data
# ======================================================================================================================
if verbose:
    print(f"Loading data from portion {portion_id}...")
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = xr.open_datatree(portion_fpath, auto_complex=True)
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
# Derive features and apply MFP to each intersection
# ======================================================================================================================

if apply_mfp_strategy_intersections:
    if verbose:
        print("Apply MFP strategy to each intersection...")
    apply_mfp_strategy_to_portion_intersections(
        portion, rcv_ref=reference_rcv, fmin=fmin, fmax=fmax
    )

if save_modification_to_portion:
    if verbose:
        print("Saving results...")
    portion.load()
    # Save to tmp file
    tmp_portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}_tmp.nc"
    portion.to_netcdf(
        tmp_portion_fpath,
        auto_complex=True,
    )

    portion.close()  # NOTE should we keep this ?

    # Remove old file
    os.remove(portion_fpath)
    # Copy file
    os.rename(tmp_portion_fpath, portion_fpath)


# ======================================================================================================================
# Plot features (RTF, DCF)
# ======================================================================================================================

if plot_features:

    if verbose:
        print("Plotting features (DCF, RTF)...")

    alpha_max = 0.5

    for name, node in portion.feature.children.items():
        ds_feature_section = node.to_dataset()

        # Check if freqeuncy band match required one (in case of previous computation)
        if not (
            (fmin <= ds_feature_section.freq.min())
            and (fmax >= ds_feature_section.freq.max())
        ):
            print(
                f"Required frequency can not be matched, using default [{ds_feature_section.freq.min().values}, {ds_feature_section.freq.max().values}]"
            )
        # Slice frequency band
        ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

        # Check reference receiver
        if ds_feature_section.reference_receiver != reference_rcv:
            print(
                f"Reference receiver {reference_rcv} differs from the one store in dataset (ref = {ds_feature_section.reference_receiver})"
            )

        # RTF
        fig_gamma, axs_gamma = plt.subplots(
            nrows=obs_ids.size - 1, ncols=1, figsize=(16, 16), sharex=True, sharey=True
        )
        axs_gamma = axs_gamma.flatten()
        # RTF
        fig_dcf, axs_dcf = plt.subplots(
            nrows=obs_ids.size - 1, ncols=2, figsize=(16, 16), sharex=True, sharey=True
        )

        for iplot, rcv_id_i in enumerate(ds_feature_section.rcv_id_feature.values):

            ds_feat_rcv = ds_feature_section.sel(rcv_id_feature=rcv_id_i)

            # RTF Gamma
            ds_feat_rcv.gamma_ij.plot(
                x="freq",
                ax=axs_gamma[iplot],
                cmap="magma",
                vmin=-alpha_max * np.nanmax(ds_feat_rcv.gamma_ij.values),
                vmax=+alpha_max * np.nanmax(ds_feat_rcv.gamma_ij.values),
            )
            axs_gamma[iplot].set_xlabel("")
            axs_gamma[iplot].set_ylabel("")
            axs_gamma[iplot].set_title(rcv_id_i.upper(), fontsize=12)

            # DCF (sxy SCOT)
            re_sij = np.real(ds_feat_rcv.sij_scot)
            im_sij = np.imag(ds_feat_rcv.sij_scot)

            re_sij.plot(
                x="freq",
                ax=axs_dcf[iplot, 0],
                cmap="magma",
                vmin=-1.2,
                vmax=+1.2,
            )
            im_sij.plot(
                x="freq",
                ax=axs_dcf[iplot, 1],
                cmap="magma",
                vmin=-1.2,
                vmax=+1.2,
            )

            axs_dcf[iplot, 0].set_xlabel("")
            axs_dcf[iplot, 0].set_ylabel("")
            axs_dcf[iplot, 0].set_title(rcv_id_i.upper(), fontsize=12)
            axs_dcf[iplot, 1].set_xlabel("")
            axs_dcf[iplot, 1].set_ylabel("")
            axs_dcf[iplot, 1].set_title(rcv_id_i.upper(), fontsize=12)

        fig_gamma.supxlabel("Frequency [Hz]")
        fig_gamma.suptitle(
            f"{ds_feature_section.section_orientation} {ds_feature_section.section_id}"
        )

        fpath = os.path.join(
            ROOT_IMG_PORTION_FEATURES_RTF,
            f"rtf_gamma_map_{name}.png",
        )
        fig_gamma.savefig(fpath)
        plt.close(fig_gamma)

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
# Plot MFP results for each intersections
# ======================================================================================================================


def get_intersection_target_position(ds_mfp_intersection):
    """_summary_

    Parameters
    ----------
    ds_mfp_intersection : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    idx_target_pos = ds_mfp_intersection.d_real.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_dt1, intersection_dt2 = (
        ds_mfp_intersection.isel(idx_target_pos).src_datetime_section1.values,
        ds_mfp_intersection.isel(idx_target_pos).src_datetime_section2.values,
    )

    return intersection_dt1, intersection_dt2


def get_intersection_guess_positions(ds_mfp_intersection):

    # Theta
    idx_guess_pos_theta = ds_mfp_intersection.d_theta.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_theta_dt1, intersection_theta_dt2 = (
        ds_mfp_intersection.isel(idx_guess_pos_theta).src_datetime_section1,
        ds_mfp_intersection.isel(idx_guess_pos_theta).src_datetime_section2,
    )

    # Theta gamma
    idx_guess_pos_theta_gamma = ds_mfp_intersection.d_theta_gamma.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_theta_gamma_dt1, intersection_theta_gamma_dt2 = (
        ds_mfp_intersection.isel(idx_guess_pos_theta_gamma).src_datetime_section1,
        ds_mfp_intersection.isel(idx_guess_pos_theta_gamma).src_datetime_section2,
    )

    # L2 gamma
    idx_guess_pos_L2_gamma = ds_mfp_intersection.d_L2_gamma.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_L2_gamma_dt1, intersection_L2_gamma_dt2 = (
        ds_mfp_intersection.isel(idx_guess_pos_L2_gamma).src_datetime_section1,
        ds_mfp_intersection.isel(idx_guess_pos_L2_gamma).src_datetime_section2,
    )

    # DCF
    idx_guess_pos_dcf = ds_mfp_intersection.d_dcf.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_dcf_dt1, intersection_dcf_dt2 = (
        ds_mfp_intersection.isel(idx_guess_pos_dcf).src_datetime_section1,
        ds_mfp_intersection.isel(idx_guess_pos_dcf).src_datetime_section2,
    )

    # Build dictionnary to group intersections
    intersections = dict(
        theta=dict(dt1=intersection_theta_dt1, dt2=intersection_theta_dt2),
        theta_gamma=dict(
            dt1=intersection_theta_gamma_dt1, dt2=intersection_theta_gamma_dt2
        ),
        L2_gamma=dict(dt1=intersection_L2_gamma_dt1, dt2=intersection_L2_gamma_dt2),
        dcf=dict(dt1=intersection_dcf_dt1, dt2=intersection_dcf_dt2),
    )
    return intersections


print()

cmap = "Grays_r"
max_quantile = 0.95

# Iterate over meridional sections
for section1_label, meridional_node in portion.mfp_intersection.children.items():
    # Iterate over zonal sections
    for section2_label, node in meridional_node.children.items():
        # Get intersection dataset
        ds_mfp_intersection = node.to_dataset()

        # Build sections names
        section1_name = f"{portion.feature[section1_label].section_orientation} {portion.feature[section1_label].section_id}"
        section2_name = f"{portion.feature[section2_label].section_orientation} {portion.feature[section2_label].section_id}"

        # Define root to save img for current intersections
        root_img_intersection = os.path.join(
            ROOT_IMG_PORTION_INTERSECTION, f"{section1_label}_{section2_label}"
        )
        os.makedirs(root_img_intersection, exist_ok=True)

        # Extract intersection target position
        target_intersection_dt1, target_intersection_dt2 = (
            get_intersection_target_position(ds_mfp_intersection=ds_mfp_intersection)
        )

        # Extract intersection estimated pos
        intersection_guess = get_intersection_guess_positions(
            ds_mfp_intersection=ds_mfp_intersection
        )

        for rcv_id in ds_mfp_intersection.rcv_id_feature.values:

            # Extract result for current rcv couple (rcv_i, rcv_ref)
            ds_mfp_intersection_rcv = ds_mfp_intersection.sel(rcv_id_feature=rcv_id)

            # Extract intersection position estimation

            ########################################################################
            # Figure 1
            # Square plot with r(i, j), theta_gamma(i, j), theta(i, j), dcf(i,j)
            ########################################################################
            fig, axs = plt.subplots(
                nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True
            )
            # Plot ambiguity surfaces
            ds_mfp_intersection_rcv.d_real.plot(
                x="src_datetime_section1", cmap=cmap, ax=axs[0, 0], vmin=0, vmax=500
            )
            ds_mfp_intersection_rcv.d_theta_gamma.plot(
                x="src_datetime_section1",
                cmap=cmap,
                ax=axs[0, 1],
                vmin=ds_mfp_intersection_rcv.d_theta_gamma.min(),
                vmax=ds_mfp_intersection_rcv.d_theta_gamma.quantile(q=max_quantile),
            )
            ds_mfp_intersection_rcv.d_theta.plot(
                x="src_datetime_section1",
                cmap=cmap,
                ax=axs[1, 0],
                vmin=ds_mfp_intersection_rcv.d_theta.min(),
                vmax=ds_mfp_intersection_rcv.d_theta.quantile(q=max_quantile),
            )
            ds_mfp_intersection_rcv.d_dcf.plot(
                x="src_datetime_section1",
                cmap=cmap,
                ax=axs[1, 1],
                vmin=ds_mfp_intersection_rcv.d_dcf.min(),
                vmax=ds_mfp_intersection_rcv.d_dcf.quantile(q=max_quantile),
            )
            # Add target position
            for ax in axs.flatten():
                ax.scatter(
                    target_intersection_dt1,
                    target_intersection_dt2,
                    marker="s",
                    s=400,
                    facecolors="none",
                    edgecolors="r",
                    label="Target",
                    linewidth=3,
                )

            # Add guess position
            axs[0, 1].scatter(
                intersection_guess["theta_gamma"]["dt1"]
                .sel(rcv_id_feature=rcv_id)
                .values,
                intersection_guess["theta_gamma"]["dt2"]
                .sel(rcv_id_feature=rcv_id)
                .values,
                marker="o",
                s=200,
                facecolors="none",
                edgecolors="m",
                label="Guess",
                linewidth=3,
            )
            axs[1, 0].scatter(
                intersection_guess["theta"]["dt1"].sel(rcv_id_feature=rcv_id).values,
                intersection_guess["theta"]["dt2"].sel(rcv_id_feature=rcv_id).values,
                marker="o",
                s=200,
                facecolors="none",
                edgecolors="m",
                label="Guess",
                linewidth=3,
            )

            axs[1, 1].scatter(
                intersection_guess["dcf"]["dt1"].sel(rcv_id_feature=rcv_id).values,
                intersection_guess["dcf"]["dt2"].sel(rcv_id_feature=rcv_id).values,
                marker="o",
                s=200,
                facecolors="none",
                edgecolors="m",
                label="Guess",
                linewidth=3,
            )

            for ax in axs.flatten():
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("")

            set_subfigures_abc_labels(
                axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
            )

            fig.supxlabel(f"Time along {section1_name} trajectory")
            fig.supylabel(f"Time along {section2_name} trajectory")

            fpath = os.path.join(
                root_img_intersection,
                f"mfp_ambiguity_4_metrics_{rcv_id}.png",
            )
            fig.savefig(fpath)
            plt.close(fig)

            # ########################################################################
            # # Figure 2
            # # theta_gamma(i, j), theta(i, j), dcf(i,j) with common colorbar
            # ########################################################################
            # fig, axs = plt.subplots(
            #     nrows=3, ncols=1, figsize=(16, 16), sharex=True, sharey=True
            # )
            # # Plot ambiguity surfaces
            # ds_mfp_intersection_rcv.d_real.plot(
            #     x="src_datetime_section1", cmap=cmap, ax=axs[0, 0], vmin=0, vmax=500
            # )
            # ds_mfp_intersection_rcv.d_theta_gamma.plot(
            #     x="src_datetime_section1",
            #     cmap=cmap,
            #     ax=axs[0, 1],
            #     vmin=ds_mfp_intersection_rcv.d_theta_gamma.min(),
            #     vmax=ds_mfp_intersection_rcv.d_theta_gamma.quantile(q=max_quantile),
            # )
            # ds_mfp_intersection_rcv.d_theta.plot(
            #     x="src_datetime_section1",
            #     cmap=cmap,
            #     ax=axs[1, 0],
            #     vmin=ds_mfp_intersection_rcv.d_theta.min(),
            #     vmax=ds_mfp_intersection_rcv.d_theta.quantile(q=max_quantile),
            # )
            # ds_mfp_intersection_rcv.d_dcf.plot(
            #     x="src_datetime_section1",
            #     cmap=cmap,
            #     ax=axs[1, 1],
            #     vmin=ds_mfp_intersection_rcv.d_dcf.min(),
            #     vmax=ds_mfp_intersection_rcv.d_dcf.quantile(q=max_quantile),
            # )
            # # Add target position
            # for ax in axs.flatten():
            #     ax.scatter(
            #         target_intersection_dt1,
            #         target_intersection_dt2,
            #         marker="s",
            #         s=400,
            #         facecolors="none",
            #         edgecolors="r",
            #         label="Target",
            #         linewidth=3,
            #     )

            # # Add guess position
            # axs[0, 1].scatter(
            #     intersection_guess["theta_gamma"]["dt1"]
            #     .sel(rcv_id_feature=rcv_id)
            #     .values,
            #     intersection_guess["theta_gamma"]["dt2"]
            #     .sel(rcv_id_feature=rcv_id)
            #     .values,
            #     marker="o",
            #     s=200,
            #     facecolors="none",
            #     edgecolors="m",
            #     label="Guess",
            #     linewidth=3,
            # )
            # axs[1, 0].scatter(
            #     intersection_guess["theta"]["dt1"].sel(rcv_id_feature=rcv_id).values,
            #     intersection_guess["theta"]["dt2"].sel(rcv_id_feature=rcv_id).values,
            #     marker="o",
            #     s=200,
            #     facecolors="none",
            #     edgecolors="m",
            #     label="Guess",
            #     linewidth=3,
            # )

            # axs[1, 1].scatter(
            #     intersection_guess["dcf"]["dt1"].sel(rcv_id_feature=rcv_id).values,
            #     intersection_guess["dcf"]["dt2"].sel(rcv_id_feature=rcv_id).values,
            #     marker="o",
            #     s=200,
            #     facecolors="none",
            #     edgecolors="m",
            #     label="Guess",
            #     linewidth=3,
            # )

            # for ax in axs.flatten():
            #     ax.set_xlabel("")
            #     ax.set_ylabel("")
            #     ax.set_title("")

            # set_subfigures_abc_labels(
            #     axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
            # )

            # fig.supxlabel(f"Time along {section1_name} trajectory")
            # fig.supylabel(f"Time along {section2_name} trajectory")

            # fpath = os.path.join(
            #     root_img_intersection,
            #     f"mfp_ambiguity_4_metrics_{rcv_id}.png",
            # )
            # fig.savefig(fpath)
            # plt.close(fig)


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
