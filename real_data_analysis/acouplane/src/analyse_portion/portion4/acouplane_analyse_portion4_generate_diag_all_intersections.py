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

# ======================================================================================================================
# Load data
# ======================================================================================================================
fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = xr.open_datatree(fpath)
print(portion)


# NOTE : use corrected position (LMS estimation)
apply_obs_pos_correction = True

if apply_obs_pos_correction:
    apply_rcv_position_correction_to_portion(portion_tree=portion)
    plot_portion_tir_pos(portion_tree=portion)

# ======================================================================================================================
# Plot signals
# ======================================================================================================================

single_fig = False
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

for name, node in portion.position.children.items():
    ds_pos_section = node.to_dataset()
    # Propagation distance assuming strait line propagation path
    dpropa_direct = np.sqrt(
        ds_pos_section.src_rcv_geodesic_dist**2 + ds_pos_section.rcv_immersion**2
    )
    # Compute propagation time
    tps_propagation_s = dpropa_direct / c0
    # tps_propagation_s = ds_pos_section.src_rcv_geodesic_dist / c0
    ds_pos_section["tps_propagation_s"] = tps_propagation_s

    # Derive associated arrival dt
    tps_propagation_ns = (tps_propagation_s * 1e9).round()
    tps_propagation_td = tps_propagation_ns.astype("timedelta64[ns]")
    ds_pos_section["tir_reception_datetime"] = (
        tps_propagation_td + ds_pos_section.src_datetime
    )

    # Update tree
    node.ds = ds_pos_section


# ======================================================================================================================
# Plot single arrival and arrrival time
# ======================================================================================================================

nb_tir = 1
offset1 = 50
offset2 = 3000

obs_ids = list(portion.pressure.children.items())[0][1].to_dataset().obs_id.values
# obs_ids = obs_ids[[0, 3, 5]]
for name, node in portion.pressure.children.items():
    ds_pressure_section = node.to_dataset()
    # Associated ds_pos
    ds_pos_section = portion.position[name].to_dataset()
    tir_dt = ds_pos_section["tir_reception_datetime"].isel(
        src_datetime=slice(0, 0 + nb_tir)
    )

    fig, axs = plt.subplots(nrows=obs_ids.size, ncols=1, figsize=(16, 14), sharex=False)
    axs = axs.flatten()
    for i, rcv_id in enumerate(obs_ids):
        tir_dt_rcv = tir_dt.sel(rcv_id=rcv_id)
        # Select associated data window
        press_rcv = ds_pressure_section[f"pressure_{rcv_id}"].sel(
            {
                f"time_{rcv_id}": slice(
                    tir_dt_rcv.values[0] - np.timedelta64(offset1, "ms"),
                    tir_dt_rcv.values[-1] + np.timedelta64(offset2 - offset1, "ms"),
                )
            }
        )
        # Plot pressure signal
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

    plt.show()

# ======================================================================================================================
# Plot single arrival and arrrival time
# ======================================================================================================================

derive_portion_cpsd(
    portion_tree=portion, offset_ms=100, signal_duration_s=3, nfft=2**15, obs_ids=None
)

# ======================================================================================================================
# Plot features (RTF, DCF)
# ======================================================================================================================

# ref_rcv_id = "obs1"
# obs_ids_visu = obs_ids[[0, 1, 2, -1]]

# # print(axs.shape)
# for name, node in portion.feature.children.items():
#     ds_feature_section = node.to_dataset()

#     sjj = ds_feature_section.sel(rcv_i=ref_rcv_id, rcv_j=ref_rcv_id).Sxy
#     # Per convention : 2nd index = ref

#     fig, axs = plt.subplots(
#         nrows=obs_ids_visu.size - 1, ncols=1, figsize=(16, 16), sharex=True, sharey=True
#     )
#     axs = axs.flatten()
#     iplot = 0
#     for i_rcv, rcv_id_i in enumerate(obs_ids_visu):

#         if rcv_id_i == ref_rcv_id:
#             continue
#         # Sii
#         sii = ds_feature_section.sel(rcv_i=rcv_id_i, rcv_j=rcv_id_i).Sxy
#         # Sij
#         sij = ds_feature_section.sel(rcv_i=rcv_id_i, rcv_j=ref_rcv_id).Sxy

#         # RTF
#         pi_ij = sij.values / sjj.values
#         gamma_ij = 20 * np.log10(np.abs(pi_ij))
#         # np.abs(pi_ij).plot(ax=axs[i_rcv, 0])
#         # gamma_ij.plot(x="freq", ax=axs[iplot], cmap="magma", add_colorbar=False)
#         axs[iplot].pcolormesh(sij.freq, sij.src_datetime, gamma_ij, cmap="magma")

#         axs[iplot].set_xlabel("")
#         axs[iplot].set_ylabel("")
#         # axs[iplot].set_title("")
#         axs[iplot].set_title(rcv_id_i.upper(), fontsize=12)
#         iplot += 1

#     fig.supxlabel("Frequency [Hz]")
#     fig.suptitle(
#         f"{ds_feature_section.section_orientation} {ds_feature_section.section_id}"
#     )
#     # plt.xlim(100, 200)
#     plt.show()

#     # # DCF (sxy SCOT)
#     # dcf_ij = sij / np.sqrt(sii * sjj)
#     # np.real(dcf_ij).plot(ax=axs[i_rcv, 1], color=color(0))
#     # np.imag(dcf_ij).plot(ax=axs[i_rcv, 1], color=color(1))

#     # for ax in axs.flatten():
#     #     ax.set_title("")
#     #     ax.set_xlabel("")
#     #     ax.set_ylabel("")

#     # fig.supxlabel("Frequency [Hz]")
