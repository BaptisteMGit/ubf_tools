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
    perf_threshold,
    study_perf_vs_snr_publi,
    study_perf_vs_snr_publi_lfm_wgn,
    study_perf_vs_snr_compare_arrays_publi,
    plot_fullarray_ambiguity_surfaces_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi,
    plot_performance_vs_number_of_rcv_in_subarray_publi_violin,
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


def perf_vs_nb_rcv(snrs=[0], root_data=p.root_data):
    # Root img
    data_name = os.path.split(os.path.split(root_data)[0])[1]
    root_img = os.path.join(
        p.root_img_publi, "performance_against_number_of_receivers", data_name
    )
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    plot_performance_vs_number_of_rcv_in_subarray_publi(
        root_img=root_img, snrs=snrs, root_data=root_data
    )
    plot_performance_vs_number_of_rcv_in_subarray_publi_violin(
        root_img=root_img, snrs=snrs, root_data=root_data
    )


def perf_vs_snr(root_data):
    # Root img
    data_name = os.path.split(os.path.split(root_data)[0])[1]
    root_img = os.path.join(p.root_img_publi, "performance_against_snr", data_name)
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    best_subarray = [0, 1, 4]
    worst_subarray = [0, 1, 3]
    # subarrays_list = [best_subarray, worst_subarray]
    # subarrays_list = [[0, 1, 2, 3, 4, 5]]
    subarrays_list = [[0, 1, 2, 3, 4, 5], worst_subarray, best_subarray]

    perf_threshold(subarrays_list, root_data, root_res=root_img)

    study_perf_vs_snr_compare_arrays_publi(
        subarrays_list, root_img, root_data=root_data
    )
    study_perf_vs_snr_publi(subarrays_list, root_img, root_data=root_data)


def study_lfm_vs_wgn_snr(root_data_lfm, root_data_wgn):
    # Root img
    root_img = os.path.join(p.root_img_publi, "performance_against_snr", "lfm_vs_wgn")
    if not os.path.exists(root_img):
        os.makedirs(root_img)

    best_subarray = [0, 1, 4]
    worst_subarray = [0, 1, 3]
    subarrays_list = [[0, 1, 2, 3, 4, 5], worst_subarray, best_subarray]

    # study_perf_vs_snr_publi(subarrays_list, root_img, root_data=root_data)
    study_perf_vs_snr_publi_lfm_wgn(
        subarrays_list, root_img, root_data_lfm, root_data_wgn
    )


if __name__ == "__main__":
    # no_noise_amb_surf()

    root_backup = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data\backups"

    # Library source = LFM
    data_name = "rtf_zhang_backup_07041041"
    root_data_lfm = os.path.join(root_backup, data_name, "data")

    # Library source = WGN
    # data_name = "rtf_zhang_backup_11042025"
    data_name = "rtf_zhang_backup_05052025"
    root_data_wgn = os.path.join(root_backup, data_name, "data")

    p1 = np.arange(-15, -10, 1)
    p2 = np.arange(-5, 5, 1)
    p3 = np.arange(5, 15, 1)
    p4 = np.arange(-5, 15, 1)
    p5 = np.arange(-10, 1, 1)

    # perf_vs_nb_rcv(root_data=root_data, snrs=p1)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p2)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p3)
    # perf_vs_nb_rcv(root_data=root_data, snrs=p4)
    # perf_vs_nb_rcv(root_data=root_data, snrs=[-10])

    # perf_vs_snr(root_data=root_data)

    study_lfm_vs_wgn_snr(root_data_lfm=root_data_lfm, root_data_wgn=root_data_wgn)
