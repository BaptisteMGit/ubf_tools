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


from publication.publication_figure import PubFigure
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    params,
    get_rcv_couples,
)

import propa.rtf.rtf_localisation.zhang_et_al_testcase.src.params as p

from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    study_perf_vs_snr_publi,
    plot_fullarray_ambiguity_surfaces_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi,
)

# pfig = PubFigure()


def no_noise_amb_surf():
    # Params
    antenna_type = "zhang"
    _, _, source, grid, _, _ = params(antenna_type=antenna_type)
    dx = grid["dx"]
    dy = grid["dy"]

    # Full simu
    folder = f"fullsimu_dx{dx}m_dy{dy}m"
    root_data = os.path.join(p.root_data, folder)

    array_label = "s1_s2_s3_s4_s5_s6"
    data_fname_fa = f"loc_zhang_dx{dx}m_dy{dy}m_fullarray_{array_label}.nc"
    fpath = os.path.join(root_data, data_fname_fa)
    ds_fa = xr.open_dataset(fpath)
    vmax = 0
    vmin = -8
    x_src = source["x"]
    y_src = source["y"]

    # Root img
    root_img = os.path.join(p.root_img_publi, "hexagonal_array_noise_free")
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
    root_img = os.path.join(p.root_img_publi, "performance_against_number_of_receivers")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_performance_vs_number_of_rcv_in_subarray_publi(root_img=root_img, snrs=[0])


def perf_vs_snr(root_data):
    # Root img
    root_img = os.path.join(p.root_img_publi, "performance_against_snr")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    best_subarray = [0, 2, 5]
    worst_subarray = [2, 3, 5]
    subarrays_list = [best_subarray, worst_subarray]
    subarrays_list = [[0, 1, 2, 3, 4, 5]]

    study_perf_vs_snr_publi(subarrays_list, root_img, root_data=root_data)


if __name__ == "__main__":
    # no_noise_amb_surf()
    # perf_vs_nb_rcv()

    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\backups\rtf_zhang_backup_07041041\data"
    perf_vs_snr(root_data=root_data)
