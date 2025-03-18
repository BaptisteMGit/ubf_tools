#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_waveguide.py
@Time    :   2025/18/03 08:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Run test on TIM plateform
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    grid_dataset,
    build_signal,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_process_testcase import process_all_snr

nf = 100  # Nombre de points fréquentiel pour le calcul des grandeurs signantes (DCF, RTF)
dx, dy = 20, 20  # Taille des mailles de la grille de recherche
antenna_type = "zhang"  # Type d'antenne utilisée pour la simulation : antenne hexagonale (a = 250 m)
debug = False  
event_stype = "wn"  # Signal source à localiser : bruit blanc gaussien

### Build dataset ###
# grid_dataset(debug=debug)
build_signal(debug=debug)


# Paramètres graphiques pour la génération des figures
plot_args = {
    "plot_array": True,
    "plot_single_cpl_surf": True,
    "plot_fullarray_surf": True,
    "plot_cpl_surf_comparison": True,
    "plot_fullarray_surf_comparison": True,
    "plot_surf_dist_comparison": True,
    "plot_mainlobe_contour": True,
    "plot_msr_estimation": True,
}

subarrays_list = [[1, 2, 3]]
snrs = [0]
n_monte_carlo = 1

process_all_snr(
    snrs,
    n_monte_carlo,
    dx=dx,
    dy=dy,
    nf=nf,
    freq_draw_method="equally_spaced",
    run_mode="w",
    subarrays_list=subarrays_list,
    antenna_type=antenna_type,
    debug=debug,
    verbose=True,
    check=True,
    plot_args=plot_args,
)
