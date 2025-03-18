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
from dask.distributed import Client, LocalCluster
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_process_testcase import process_all_snr
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import N_WORKERS, MAX_RAM_PER_WORKER_GB


### Build dataset ###
# grid_dataset(debug=debug)
# build_signal(debug=debug)

### Open Dask client to manage ressources ###


nf = 100  # Nombre de points fréquentiel pour le calcul des grandeurs signantes (DCF, RTF)
dx, dy = 20, 20  # Taille des mailles de la grille de recherche
antenna_type = "zhang"  # Type d'antenne utilisée pour la simulation : antenne hexagonale (a = 250 m)
debug = False  
event_stype = "wn"  # Signal source à localiser : bruit blanc gaussien


def run_test():

    # Paramètres graphiques pour la génération des figures
    plot_args = {
        "plot_array": True,
        "plot_single_cpl_surf": False,
        "plot_fullarray_surf": False,
        "plot_cpl_surf_comparison": True,
        "plot_fullarray_surf_comparison": True,
        "plot_surf_dist_comparison": False,
        "plot_mainlobe_contour": False,
        "plot_msr_estimation": False,
    }

    subarrays_list = [[1, 2, 3]]
    snrs = [0]
    n_monte_carlo = 3

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

if __name__ == "__main__":
    #
    with Client(n_workers=N_WORKERS, threads_per_worker=1, memory_limit=f"{MAX_RAM_PER_WORKER_GB}GB") as client:
    # with Client(n_workers=1, threads_per_worker=1, memory_limit="4GB") as client:
        # Print dashboard link
        print("Dask Dashboard:", client.dashboard_link)
        run_test()
    
    # run_test()