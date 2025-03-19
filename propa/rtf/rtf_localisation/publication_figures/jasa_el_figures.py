#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_figures.py
@Time    :   2025/02/26 11:37:40
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Create figures for JASA publication
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr


from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    get_rcv_couples,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import (
    ROOT_DATA,
    ROOT_IMG,
)

from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    plot_fullarray_ambiguity_surfaces_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi,
)

pfig = PubFigure()

ROOT_IMG_PUBLI = os.path.join(ROOT_IMG, "publication_rtf")
if not os.path.exists(ROOT_IMG_PUBLI):
    os.makedirs(ROOT_IMG_PUBLI)


def no_noise_amb_surf():
    # Params
    antenna_type = "zhang"
    _, _, source, grid, _, _ = params(antenna_type=antenna_type)
    dx = grid["dx"]
    dy = grid["dy"]

    # Full simu
    folder = f"fullsimu_dx{dx}m_dy{dy}m"
    root_data = os.path.join(ROOT_DATA, folder)

    array_label = "s1_s2_s3_s4_s5_s6"
    data_fname_fa = f"loc_zhang_dx{dx}m_dy{dy}m_fullarray_{array_label}.nc"
    fpath = os.path.join(root_data, data_fname_fa)
    ds_fa = xr.open_dataset(fpath)
    vmax = 0
    vmin = -8
    x_src = source["x"]
    y_src = source["y"]

    # Root img
    root_img = os.path.join(ROOT_IMG_PUBLI, "hexagonal_array_noise_free")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_fullarray_ambiguity_surfaces_publi(
        ds_fa,
        root_img,
        x_src,
        y_src,
        vmin,
        vmax,
        cmap="jet",
    )


def perf_vs_nb_rcv():
    # Root img
    root_img = os.path.join(ROOT_IMG_PUBLI, "performance_against_number_of_receivers")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_performance_vs_number_of_rcv_in_subarray_publi(
        root_img=root_img, snrs=[-15], dx=20, dy=20
    )


if __name__ == "__main__":
    # no_noise_amb_surf()
    perf_vs_nb_rcv()
