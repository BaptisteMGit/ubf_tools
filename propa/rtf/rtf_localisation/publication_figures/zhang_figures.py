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
    plot_subarrays_ambiguity_surfaces,
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
    fpath = os.path.join(ROOT_DATA, f"zhang_output_fullsimu_dx{dx}m_dy{dy}m.nc")
    ds = xr.open_dataset(fpath)

    folder = f"fullsimu_dx{dx}m_dy{dy}m"
    root_data = os.path.join(ROOT_DATA, folder)

    array_label = "s1_s2_s3_s4_s5_s6"
    data_fname_fa = f"loc_zhang_dx{dx}m_dy{dy}m_fullarray_{array_label}.nc"
    fpath = os.path.join(root_data, data_fname_fa)
    ds_fa = xr.open_dataset(fpath)

    xticks_pos_km = [3.6, 4.0, 4.4]
    yticks_pos_km = [6.5, 6.9, 7.3]
    xticks_pos_m = [xt * 1e3 for xt in xticks_pos_km]
    yticks_pos_m = [yt * 1e3 for yt in yticks_pos_km]

    vmax = 0
    vmin = -8
    # vmin = np.round(np.max([ds_fa[dist].median() for dist in ["d_gcc", "d_rtf"]]), 0)
    # # Define vmin as percentile instead of median
    # perc = 99
    # vmin = np.round(
    #     np.percentile(
    #         np.concatenate(
    #             [ds_fa["d_gcc"].values.flatten(), ds_fa["d_rtf"].values.flatten()]
    #         ),
    #         perc,
    #     ),
    #     0,
    # )

    x_src = source["x"]
    y_src = source["y"]

    # plot_fullarray_ambiguity_surfaces_publi(
    #     ds_fa,
    #     ROOT_IMG_PUBLI,
    #     x_src,
    #     y_src,
    #     vmin,
    #     vmax,
    #     xticks_pos_m,
    #     yticks_pos_m,
    #     cmap="jet",
    # )
    data_fname = f"loc_zhang_dx{dx}m_dy{dy}m"
    rcv_couples = get_rcv_couples(ds_fa.idx_rcv)
    plot_subarrays_ambiguity_surfaces(
        ROOT_IMG_PUBLI,
        rcv_couples,
        data_fname,
        root_data,
        grid,
        x_src,
        y_src,
        vmin,
        vmax,
        xticks_pos_m,
        yticks_pos_m,
        cmap="jet",
    )


if __name__ == "__main__":
    no_noise_amb_surf()
