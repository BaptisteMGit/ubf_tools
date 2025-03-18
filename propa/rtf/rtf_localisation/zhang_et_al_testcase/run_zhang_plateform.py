#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_waveguide.py
@Time    :   2025/18/03 08:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Run Zhang testcase on TIM plateform
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np 


from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_process_testcase import (
    process_all_snr,
    study_perf_vs_subarrays,
)
from dask.distributed import Client
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import get_subarrays
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_params import N_WORKERS, MAX_RAM_PER_WORKER_GB
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_build_datasets import (
    build_tf_dataset,
    grid_dataset,
    build_signal,
    build_features_fullsimu,
    build_features_from_time_signal,
)

"""
Etude des performances de la méthode RTF MFP dans le cadre du cas test proposée par Zhang et al. 2023 : plateforme TIM
Publication de référence : Zhang, T., Zhou, D., Cheng, L., & Xu, W. (2023). Correlation-based passive localization: Linear system modeling and sparsity-aware optimization. The Journal of the Acoustical Society of America, 154(1), 295–306. https://doi.org/10.1121/10.0020154

Functions are split so that they can be be run from two separated dockers at the same time 
"""


def run_test_1():
    ### 
    # Etape 2 : Simulation avec deux sous-antennes à trois capteurs de l'antenne hexagonale complète 
    ###
    """
    Le choix des deux sous-antennes à trois capteurs est basé sur les performances (critère hybride msr/rmse) de la méthode évaluée à un SNR = -15 dB sur l'ensemble des 57 sous-antennes de l'antenne hexagonale (sur 100 réalisations) : 
    -   **Antenne 1** (meilleure perf) : capteurs $s_1, s_3, s_6$
    -   **Antenne 2** (pire perf) : capteurs $s_3, s_4, s_6$

    ### Objectif
    L'objectif est de produire les figures de MSR et RMSE versus SNR pour les deux antennes, pour la zone de recherche complète et sur une large gamme de SNR. 

    ### Paramètres 
    -   **Nombre de simulations (Monte Carlo)** = 100
    -   **SNR** = de -30 à +20 dB par pas de 2.5 dB 
    build_features_fullsimu(debug=debug, antenna_type=antenna_type, event_stype=event_stype)

    """

    # Liste des sous antennes considérées 
    best_subarray = [0, 2, 5]
    worste_subarray = [2, 3, 5]
    subarrays_list = [best_subarray, worste_subarray]
    print(f"Number of subarrays = {len(subarrays_list)}")
    print("Subarrays list : ", subarrays_list)

    # Liste des SNR considérés
    snr_min = -30
    snr_max = 20
    snr_step = 2.5
    n_snr = int((snr_max - snr_min) / snr_step + 1)
    snrs = np.linspace(snr_min, snr_max, n_snr)
    print("SNRs : ", snrs)
    print(f"Number of SNRs = {n_snr}")

    # Nombre de simulations à réaliser pour chaque SNR
    n_monte_carlo = 100
    print(f"Number of Monte Carlo simulations = {n_monte_carlo}")

    # Derive expected cpu time for information 
    avg_cpu_t_per_iter = 200
    total_expected_cpu_time = n_snr * n_monte_carlo * avg_cpu_t_per_iter
    print(f"Expected cpu time = {np.round(total_expected_cpu_time, 0)} s = {np.round(total_expected_cpu_time/3600, 2)} h")

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

    # Calcul des MSR, RMSE pour chaque sous-antennes
    process_all_snr(
        snrs,
        n_monte_carlo,
        dx=dx,
        dy=dy,
        nf=nf,
        freq_draw_method=freq_draw_method,
        run_mode="a",
        subarrays_list=subarrays_list,
        antenna_type=antenna_type,
        debug=debug,
        verbose=True,
        check=True,
        plot_args=plot_args,
    )

def run_test_2():
    ###
    # Etape 3 : Etude des performances à faible SNR en fonction du nombre de récepteurs de l'antenne
    ###

    """
    ### Objectif
    L'objectif de cette simulation est d'étudier les performances de la méthode RTF vis à vis des performances de la méthode DCF à faible SNR. En particulier on cherche à mettre en lumière les avantages de la méthode RTF lorsque le nombre de récepteurs de l'antenne augmente. Pour cela on considère l'ensemble des sous antennes de l'antenne hexagonale et on cherche à représenter les métriques MSR et RMSE en fonction du nombre de capteur. On souhaite également représenter l'amplitude des variations de ces métriques pour un même nombre de capteur afin de voir si la configuration des capteurs à une influence importante sur les méthodes étudiées. 

    ### Paramètres 
    -   **Nombre de simulations (Monte Carlo)** = 100
    -   **SNR** = -15 dB
    """

    # Liste des sous antennes considérées : toutes les sous antennes possibles pour 2, 3, 4, 5 et 6 récepteurs
    subarrays_list = []
    n_rcv = [2, 3, 4, 5, 6]
    for i in n_rcv:
        subarrays_list += list(get_subarrays(nr_fullarray=6, nr_subarray=i))
    print(f"Number of subarrays = {len(subarrays_list)}")
    print("Subarrays list : ", subarrays_list)

    # Liste des SNR considérés
    snrs = [-15]

    # Nombre de simulations à réaliser pour chaque SNR
    n_monte_carlo = 100

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

    # Calcul des MSR, RMSE pour chaque sous-antennes
    process_all_snr(
        snrs,
        n_monte_carlo,
        dx=dx,
        dy=dy,
        nf=nf,
        freq_draw_method=freq_draw_method,
        run_mode="a",
        subarrays_list=subarrays_list,
        antenna_type=antenna_type,
        debug=debug,
        verbose=False,
        check=True,
        plot_args=plot_args,
    )

    # Etude des métriques vs nombre de capteurs dans la sous-antenne
    study_perf_vs_subarrays(subarrays_list, snrs, var="std")
    study_perf_vs_subarrays(subarrays_list, snrs, var="minmax")


if __name__ == "__main__":

    ### 
    # Etape 0 : définition des paramètres de l'étude 
    ### 
    """
    **antenna_type** -> type d'antenne considérée :
    -   "zhang" : antenne hexagonnale régulière de coté 250 m 
    -   "random" : antenne de 6 capteurs positionnés aléatoirement dans un cercle de rayon 1.5 km autour de l'origine du repère

    **event_stype** -> nature du signal émis par la source *event* à localiser :
    -   "wn" : bruit blanc gaussien 
    -   "lfm" : chirp linéaire

    **debug** -> mode d'execution du code :
    -   True : execution en mode debug sur une sous-zone de calcul (plus rapide) 
    -   False : execution sur la zone complète (1km x 1km) (plus lent)

    """
    antenna_type = "zhang"
    event_stype = "wn"
    debug = False

    freq_draw_method = "equally_spaced"
    nf = 100  # Nombre de points fréquentiel pour le calcul des grandeurs signantes (DCF, RTF)
    dx, dy = 20, 20  # Taille des mailles de la grille de recherche

    ### 
    # Etape 1 : génération du dataset de fonctions de transfert 
    ###

    ## Step 1 : Construction du dataset de fonctions de transfert avec Kraken
    # Not needed here, the correponding dataset should already be available in the ROOT_DATA folder with name : tf_zhang_dataset.nc
    # build_tf_dataset()     

    ## Step 2 : Interpolation des fonctions de transfert sur la grille de recherche 
    # A executer une unique fois en mode debug = False pour obtenir le dataset correspondant à la zone de recherche complète (1km x 1km)
    # grid_dataset(debug=debug, antenna_type=antenna_type)

    ## Step 3 : Calcul des signaux propagés depuis chacun des points de la grille 
    # A executer une unique fois
    # Step 3
    # build_signal(debug=debug, antenna_type=antenna_type, event_stype=event_stype)

    ## Step 4 : Calcul des vecteurs de RTF "théorique" directement à partir des fonctions de transfert
    # Principalement pour comparaison avec les vecteurs de RTF estimés par la méthode CS 

    ### Open Dask client to manage ressources ###
    with Client(n_workers=N_WORKERS, threads_per_worker=1, memory_limit=f"{MAX_RAM_PER_WORKER_GB}GB") as client:
        # Print dashboard link
        print("Dask Dashboard:", client.dashboard_link)

        # Uncomment target test 
        run_test_1()
        # run_test_2()

