#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   distance_distrubution_utils.py
@Time    :   2025/10/26 08:47:44
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Usefull functions to study the RTF distance distributions (theta, frobenius, etc.)
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import scipy.signal as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

from time import time
from scipy.linalg import eigh
from scipy.optimize import least_squares

from real_data_analysis.fiberscope_20.src import params
from real_data_analysis.fiberscope_20.src.utils import load_fiberscope_data
from real_data_analysis.fiberscope_20.src.fiberscope_manager import FiberscopeManager
from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
    FiberscopeSweep1,
    FiberscopeSweep2,
    FiberscopeDynamicRecording,
)
from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast

from publication.publication_figure import color
from misc import progression_bar, filter_outliers_iqr

# ======================================================================================================================
# Functions
# ======================================================================================================================


def build_distribution_dataframe(data):
    """
    Build a dataframe with distance distributions to be analyse using the seaborn library.

    TODO : describe data structure (dict)

    data = {
        records:
            {
            arr0,
            arr1,
            }
    }

    """

    rows = []

    for method, records in data.items():
        for record_name, arrays in records.items():
            for array_index, arr in enumerate(arrays):
                arr = np.array(arr).flatten()
                for val in arr:
                    rows.append(
                        {
                            "method": method,
                            "record": record_name,
                            "array_index": array_index,
                            "value": val,
                        }
                    )

    df = pd.DataFrame(rows)

    # Position name
    df["position"] = df["record"].str.extract(r"_(P\d+)_")

    return df


def filter_iqr_df(df, k=1.5, groupby_cols=("method", "position", "window_tf")):
    """
    Filtre les outliers par groupe en utilisant la règle IQR.
    Retourne (df_filtered, summary_df)
    - df_filtered : DataFrame sans les outliers
    - summary_df : DataFrame récapitulatif par groupe avec counts avant/apres et % supprimé
    """
    df = df.copy()

    # calculer Q1 et Q3 par ligne (transform renvoie un vecteur aligné avec df)
    g = df.groupby(list(groupby_cols))["value"]
    q1 = g.transform(lambda x: x.quantile(0.25))
    q3 = g.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    # si IQR == 0 (toutes valeurs égales) ou groupe très petit -> garder tout (éviter NaN)
    # Construire mask : True si dans les bornes (et si bornes non-NaN), sinon True pour garder
    mask = (df["value"] >= lower) & (df["value"] <= upper)

    # Remplacer les NaN dans mask par True (conserve les groupes où quantiles non définies)
    mask = mask.fillna(True)

    df_filtered = df[mask].reset_index(drop=True)

    # Résumé par groupe
    before = df.groupby(list(groupby_cols)).size().rename("n_before")
    after = df_filtered.groupby(list(groupby_cols)).size().rename("n_after")
    summary = pd.concat([before, after], axis=1).fillna(0).astype(int)
    summary["n_removed"] = summary["n_before"] - summary["n_after"]
    summary["pct_removed"] = 100 * summary["n_removed"] / summary["n_before"]
    summary = summary.reset_index()

    return df_filtered, summary


def plot_all_distribution_panel(
    df, value_label=r"$\theta$ [°]", filter=False, filter_k=3, plot_type="kde"
):

    if filter:
        # Filtrage des outliers (IQR)
        df, summary = filter_iqr_df(df, k=filter_k)
        print(f"Filtered data summary: {summary}")

    # Création de la grille de facettes
    g = sns.FacetGrid(
        df,
        row="position",
        col="window_tf",
        hue="method",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=5,
        aspect=1.2,
    )

    # Type de graphique : distribution (KDE, hist ou boxplot selon ton besoin)
    if plot_type == "kde":
        g.map(sns.kdeplot, "value", fill=True, alpha=0.4)
    elif plot_type == "hist":
        g.map(sns.histplot, "value", fill=True, alpha=0.4, stat="density")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Ajout de la légende et ajustement du layout
    g.add_legend(fontsize=30)
    g.set_axis_labels(value_label, "Density")
    g.figure.subplots_adjust(top=0.91)
    g.figure.suptitle(
        "Distribution des valeurs par position et par fenêtre temporelle", fontsize=35
    )

    return g


def plot_distribution_panel_single_window(
    df, value_label=r"$\theta$ [°]", filter=False, filter_k=3, plot_type="kde"
):

    if filter:
        # Filtrage des outliers (IQR)
        df, summary = filter_iqr_df(df, k=filter_k)
        print(f"Filtered data summary: {summary}")

    # Création de la grille de facettes
    g = sns.FacetGrid(
        df,
        # row="method",
        col="method",
        hue="position",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=5,
        aspect=1.2,
    )
    # Type de graphique : distribution (KDE, hist ou boxplot selon ton besoin)
    if plot_type == "kde":
        g.map(sns.kdeplot, "value", multiple="layer")
    elif plot_type == "hist":
        g.map(sns.histplot, "value", multiple="layer", stat="density")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Ajout de la légende et ajustement du layout
    g.add_legend()
    g.set_axis_labels(value_label, "Density")
    # g.figure.subplots_adjust(top=0.92, wspace=0.1)
    # g.figure.suptitle("Distribution des valeurs pour chaque fenêtre", fontsize=14)

    return g


def plot_distribution_panel_single_position(
    df,
    value_label=r"$\theta$ [°]",
    filter=False,
    filter_k=3,
    plot_type="kde",
):

    if filter:
        # Filtrage des outliers (IQR)
        df, summary = filter_iqr_df(df, k=filter_k)
        print(f"Filtered data summary: {summary}")

    # Création de la grille de facettes
    g = sns.FacetGrid(
        df,
        # row="method",
        col="method",
        hue="window_tf",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=5,
        aspect=1.2,
    )

    # Type de graphique : distribution (KDE, hist ou boxplot selon ton besoin)
    if plot_type == "kde":
        g.map(sns.kdeplot, "value", multiple="layer")
    elif plot_type == "hist":
        g.map(sns.histplot, "value", multiple="layer", stat="density")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Ajout de la légende et ajustement du layout
    g.add_legend()
    g.set_axis_labels(value_label, "Density")
    # g.figure.subplots_adjust(top=0.92)
    # g.figure.suptitle("Distribution des valeurs pour chaque position", fontsize=14)

    return g


def qq_data_rayleigh(data):
    """Retourne les quantiles empiriques et théoriques pour la loi de Rayleigh ajustée"""
    data = np.sort(data)
    n = len(data)

    # Ajustement des paramètres de la loi de Rayleigh
    loc, scale = stats.rayleigh.fit(data)

    # Calcul des probabilités cumulées empiriques
    probs = (np.arange(1, n + 1) - 0.5) / n

    # Quantiles théoriques correspondant
    theo = stats.rayleigh.ppf(probs, loc=loc, scale=scale)

    return pd.DataFrame({"empirical": data, "theoretical": theo})


def qq_data_gamma(data):
    """Retourne les quantiles empiriques et théoriques pour la loi Gamma ajustée"""
    data = np.sort(data)
    n = len(data)
    if n < 10:
        return pd.DataFrame(columns=["empirical", "theoretical"])

    # Ajustement des paramètres de la loi Gamma
    shape, loc, scale = stats.gamma.fit(data)

    # Calcul des probabilités empiriques
    probs = (np.arange(1, n + 1) - 0.5) / n

    # Quantiles théoriques
    theo = stats.gamma.ppf(probs, shape, loc=loc, scale=scale)

    return pd.DataFrame({"empirical": data, "theoretical": theo})


def qq_plot(df, law="rayleigh", filter=False, filter_k=3):

    # On prépare une structure pour tous les Q-Q plots
    qq_records = []

    for (pos, win, method), group in df.groupby(["position", "window_tf", "method"]):
        values = group["value"].values

        if filter:
            # Filtrage des outliers (IQR)
            values_filtered = filter_outliers_iqr(values, k=filter_k)
            values = values_filtered

        if law == "rayleigh":
            qq_df = qq_data_rayleigh(values)
        elif law == "gamma":
            qq_df = qq_data_gamma(values)
        else:
            print(f'Selectionned law "{law}" not handled yet.')

        qq_df["position"] = pos
        qq_df["window_tf"] = win
        qq_df["method"] = method
        qq_records.append(qq_df)

    qq_all = pd.concat(qq_records, ignore_index=True)

    # Visualisation avec FacetGrid
    g = sns.FacetGrid(
        qq_all,
        row="position",
        col="window_tf",
        hue="method",
        margin_titles=True,
        sharex=False,
        sharey=False,
        height=2.4,
    )

    g.map_dataframe(sns.scatterplot, x="theoretical", y="empirical", s=10, alpha=0.6)
    g.add_legend(fontsize=30)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Ligne de référence y=x
    for ax in g.axes.flat:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    g.figure.subplots_adjust(top=0.92, wspace=0.1)
    g.figure.suptitle(
        f"Q-Q plots : ajustement à la loi {law.capitalize()}", fontsize=35
    )

    return g


def ks_test_heatmap(df, law="rayleigh", filter=False, filter_k=3):

    methods = df["method"].unique()
    positions = df["position"].unique()
    window_values = sorted(df["window_tf"].unique())

    for method in methods:
        # Créer une matrice vide pour les p-values
        p_matrix = np.zeros((len(positions), len(window_values)))

        # Remplir la matrice
        for i, pos in enumerate(positions):
            for j, win in enumerate(window_values):
                group = df[
                    (df["method"] == method)
                    & (df["position"] == pos)
                    & (df["window_tf"] == win)
                ]
                if len(group) > 0:
                    values = group["value"].values

                    if filter:
                        # Filtrage des outliers (IQR)
                        values_filtered = filter_outliers_iqr(values, k=filter_k)
                        values = values_filtered

                    if law == "rayleigh":
                        # Ajustement des paramètres de la loi de Rayleigh
                        loc, scale = stats.rayleigh.fit(values)
                        args = (loc, scale)
                    elif law == "gamma":
                        # Ajustement Gamma
                        shape, loc, scale = stats.gamma.fit(values)
                        args = (shape, loc, scale)
                    else:
                        print(f'Selectionned law "{law}" not handled yet.')

                    # Test KS
                    ks_stat, ks_p = stats.kstest(values, law, args=args)
                    p_matrix[i, j] = ks_p
                else:
                    p_matrix[i, j] = np.nan  # si pas de données pour ce couple

        # Créer la heatmap
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(
            p_matrix,
            xticklabels=[f"{w}s" for w in window_values],
            yticklabels=positions,
            vmin=0,
            vmax=1,
            cmap="rocket_r",  # inverse pour que p≈1 = couleur claire
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "p-value KS test"},
        )
        ax.set_xlabel("Durée")
        ax.set_ylabel("Position")
        plt.title(
            f"Test KS pour la loi {law.capitalize()} - méthode {method}", fontsize=14
        )
        # plt.tight_layout()
        # plt.show()


def fit_th_distribution(df, law="rayleigh", win=10, filter=False, filter_k=3):

    methods = df["method"].unique()
    positions = df["position"].unique()

    if law == "rayleigh":
        rayleigh_loc = {method: [] for method in methods}
        rayleigh_scale = {method: [] for method in methods}
    elif law == "gamma":
        gamma_loc = {method: [] for method in methods}
        gamma_shape = {method: [] for method in methods}
        gamma_scale = {method: [] for method in methods}

    for method in methods:

        for i, pos in enumerate(positions):
            group = df[
                (df["method"] == method)
                & (df["position"] == pos)
                & (df["window_tf"] == win)
            ]
            values = group["value"].values

            if filter:
                # Filtrage des outliers (IQR)
                values_filtered = filter_outliers_iqr(values, k=filter_k)
                values = values_filtered

            if law == "rayleigh":
                # Ajustement des paramètres de la loi de Rayleigh
                loc, scale = stats.rayleigh.fit(values)

                rayleigh_loc[method].append(loc)
                rayleigh_scale[method].append(scale)

            elif law == "gamma":
                # Ajustement Gamma
                shape, loc, scale = stats.gamma.fit(values)

                gamma_shape[method].append(shape)
                gamma_loc[method].append(loc)
                gamma_scale[method].append(scale)

    if law == "rayleigh":
        return rayleigh_loc, rayleigh_scale

    elif law == "gamma":
        return gamma_shape, gamma_loc, gamma_scale

    else:
        return None


def plot_fitted_vs_observed(
    df,
    law,
    filter,
    method,
    win,
    pos=None,
    filter_k=3,
    params_stat="mean",
    value_label=r"$\theta$ [°]",
    plot_type="kde",
):

    df_group = df[(df["method"] == method) & (df["window_tf"] == win)]
    if pos is not None:
        df_group = df_group[(df_group["position"] == pos)]

    if filter:
        # Filtrage des outliers (IQR)
        df_group, summary = filter_iqr_df(df_group, k=filter_k)
        print(f"Filtered data summary: {summary}")

    fitted_params = fit_th_distribution(df_group, law=law, filter=filter, win=win)

    positions = df_group["position"].unique()

    if params_stat == "mean":
        f_stat = np.mean
    elif params_stat == "median":
        f_stat = np.median
    elif params_stat == "first":

        def take_first(arr):
            return np.take(arr, 0)

        f_stat = take_first
    else:
        pass

    x_fit = np.linspace(0, np.max(df_group["value"].values), int(1e4))

    if law == "rayleigh":
        loc = f_stat(fitted_params[0][method])
        scale = f_stat(fitted_params[1][method])
        f_x_fit = stats.rayleigh.pdf(x=x_fit, loc=loc, scale=scale)

    elif law == "gamma":
        shape = f_stat(fitted_params[0][method])
        loc = f_stat(fitted_params[1][method])
        scale = f_stat(fitted_params[2][method])
        print(f"Fitted params: {shape, scale, loc}")
        f_x_fit = stats.gamma.pdf(x=x_fit, a=shape, loc=loc, scale=scale)

    # Normaliser pour que l'aire = 1 (comme la KDE)
    # f_x_fit /= f_x_fit.sum() * (x_fit[1] - x_fit[0])

    plt.figure(figsize=(12, 6))

    # Tracer KDE par position
    for pos in positions:
        values = df[
            (df["method"] == method)
            & (df["position"] == pos)
            & (df["window_tf"] == win)
        ]["value"].values

        if filter:
            # Filtrage des outliers (IQR)
            values_filtered = filter_outliers_iqr(values, k=filter_k)
            values = values_filtered

        if plot_type == "kde":
            # KDE manuelle
            kde = stats.gaussian_kde(values)
            plt.plot(x_fit, kde(x_fit), label=f"{pos} KDE")
        elif plot_type == "hist":
            plt.hist(values, density=True, label=f"{pos} hist")

    plt.plot(x_fit, f_x_fit, "k-", label=f"Fitted {law.capitalize()}")
    plt.xlabel(value_label)
    plt.ylabel("Densité")
    plt.legend()


def plot_empirical_vs_gamma(
    df, method, positions, window_tfs, filter=False, figsize=(12, 4)
):
    """
    Trace la CDF empirique et théorique Gamma pour plusieurs couples (position, window_tf).

    Parameters:
    -----------
    df : pandas.DataFrame
        Contient les colonnes ["value", "position", "window_tf", "method"]
    method : str
        Méthode à sélectionner ("cs" ou "cs-evd")
    positions : list of str
        Liste des positions à tracer, ex: ["P1", "P2"]
    window_tfs : list of int/float
        Liste des durées window_tf à tracer, ex: [5, 10]
    figsize : tuple
        Taille de la figure matplotlib
    """

    n_plots = len(positions) * len(window_tfs)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for pos in positions:
        for win in window_tfs:
            ax = axes[plot_idx]

            # Filtrer les données
            values = df[
                (df["method"] == method)
                & (df["position"] == pos)
                & (df["window_tf"] == win)
            ]["value"].values

            if filter:
                # Filtrage des outliers (IQR)
                values_filtered = filter_outliers_iqr(values, k=3)
                values = values_filtered

            if len(values) == 0:
                ax.set_visible(False)
                plot_idx += 1
                continue

            # Ajustement Gamma
            shape, loc, scale = stats.gamma.fit(values)

            # CDF empirique
            sorted_values = np.sort(values)
            empirical_cdf = np.arange(1, len(values) + 1) / len(values)

            # CDF théorique Gamma
            theoretical_cdf = stats.gamma.cdf(
                sorted_values, a=shape, loc=loc, scale=scale
            )

            # Tracé
            ax.plot(
                sorted_values,
                empirical_cdf,
                marker=".",
                linestyle="none",
                label="Empirique",
            )
            ax.plot(
                sorted_values,
                theoretical_cdf,
                "r-",
                label=f"Gamma fit\nshape={shape:.2f}, scale={scale:.2f}",
            )
            ax.set_title(f"{pos}, T={win}s")
            ax.set_xlabel("Valeurs")
            ax.set_ylabel("CDF")
            ax.grid(True)
            ax.legend(fontsize=8)

            plot_idx += 1

    # Supprimer les axes vides
    for k in range(plot_idx, len(axes)):
        axes[k].set_visible(False)

    # plt.tight_layout()
    # plt.show()


def plot_empirical_vs_rayleigh(
    df, method, positions, window_tfs, filter=False, figsize=(12, 4)
):
    """
    Trace la CDF empirique et théorique Rayleigh pour plusieurs couples (position, window_tf).

    Parameters:
    -----------
    df : pandas.DataFrame
        Contient les colonnes ["value", "position", "window_tf", "method"]
    method : str
        Méthode à sélectionner ("cs" ou "cs-evd")
    positions : list of str
        Liste des positions à tracer, ex: ["P1", "P2"]
    window_tfs : list of int/float
        Liste des durées window_tf à tracer, ex: [5, 10]
    figsize : tuple
        Taille de la figure matplotlib
    """

    n_plots = len(positions) * len(window_tfs)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for pos in positions:
        for win in window_tfs:
            ax = axes[plot_idx]

            # Filtrer les données
            values = df[
                (df["method"] == method)
                & (df["position"] == pos)
                & (df["window_tf"] == win)
            ]["value"].values

            if filter:
                # Filtrage des outliers (IQR)
                values_filtered = filter_outliers_iqr(values, k=3)
                values = values_filtered

            if len(values) == 0:
                ax.set_visible(False)
                plot_idx += 1
                continue

            # Ajustement des paramètres de la loi de Rayleigh
            loc, scale = stats.rayleigh.fit(values)

            # CDF empirique
            sorted_values = np.sort(values)
            empirical_cdf = np.arange(1, len(values) + 1) / len(values)

            # CDF théorique Gamma
            theoretical_cdf = stats.rayleigh.cdf(sorted_values, loc=loc, scale=scale)

            # Tracé
            ax.plot(
                sorted_values,
                empirical_cdf,
                marker=".",
                linestyle="none",
                label="Empirique",
            )
            ax.plot(
                sorted_values,
                theoretical_cdf,
                "r-",
                label=f"Rayleigh fit\nloc={loc:.2f}, scale={scale:.2f}",
            )
            ax.set_title(f"{pos}, T={win}s")
            ax.set_xlabel("Valeurs")
            ax.set_ylabel("CDF")
            ax.grid(True)
            ax.legend(fontsize=8)

            plot_idx += 1

    # Supprimer les axes vides
    for k in range(plot_idx, len(axes)):
        axes[k].set_visible(False)

    # plt.tight_layout()
    # plt.show()
    fig.suptitle(method)
