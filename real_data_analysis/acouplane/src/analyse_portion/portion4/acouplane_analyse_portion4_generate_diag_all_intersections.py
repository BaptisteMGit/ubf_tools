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
