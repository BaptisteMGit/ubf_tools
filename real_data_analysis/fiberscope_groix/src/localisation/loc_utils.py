#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   loc_utils.py
@Time    :   2026/01/20 18:45:09
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Localisation utils
"""


# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt

from publication.publication_figure import PubFigure, color
from misc import equivalent_celerity, calc_wls, draw_ellipse

# ======================================================================================================================
# LMS
# ======================================================================================================================


def f(X, X_s):
    """
    Parameters
    ----------
    X : array
        [y, x, z] position of the hydrophone to estimate in the local reference frame
    X_s : 2D array
        [y, x, z] positions of the source

    """
    x, y, z = X
    x_s, y_s, z_s = X_s[:, 0], X_s[:, 1], X_s[:, 2]
    d2 = (x - x_s) ** 2 + (y - y_s) ** 2 + (z - z_s) ** 2

    return d2


def jac(X, X_s):
    """
    Parameters
    ----------
    X : array
        [y, x, z] position of the hydrophone to estimate in the local reference frame
    X_s : 2D array
        [y, x, z] positions of the source
    """
    x, y, z = X
    x_s, y_s, z_s = X_s[:, 0], X_s[:, 1], X_s[:, 2]

    df_dx = 2 * (x - x_s)
    df_dy = 2 * (y - y_s)
    df_dz = 2 * (z - z_s)
    J = np.array([df_dx, df_dy, df_dz]).T

    return J


def inverse_obs_loc(df_arr, ds_gps, obs_id, min_f_score, plot=True, verbose=False):

    # Select sequence with sufficient score
    col_name = f"f_score OBS{obs_id}"
    df_obs = df_arr[df_arr[col_name] >= min_f_score]

    # Grandeur signante -> y = d**2 = (t_prop * c)**2
    c = 1500
    meas_prop_time = df_obs[f"Measured propagation time OBS{obs_id}"] - 27
    y = (meas_prop_time * c) ** 2

    # Position a priori
    e_obs = ds_gps.attrs[f"obs{obs_id}_e_apriori"]
    n_obs = ds_gps.attrs[f"obs{obs_id}_n_apriori"]
    u_obs = ds_gps.attrs[f"obs{obs_id}_u_apriori"]
    X_0 = np.array([e_obs, n_obs, u_obs])

    # Position succesives de la source
    e_pos = df_obs["Emission interpolated E GPS"].values
    n_pos = df_obs["Emission interpolated N GPS"].values
    u_pos = df_obs["Emission interpolated U GPS"].values
    X_s = np.stack((e_pos, n_pos, u_pos)).T

    # Weighting matrix
    # W = np.diag(1 / y_var)
    W = np.eye(len(y))  # Initialisation avec la matrice identité

    # Apply LMS
    X_hat, dY_hat, cost_hat, SigmaXhat, sigma02hat = calc_wls(
        Y=y, t=X_s, X_0=X_0, W=W, fct=f, jac=jac, tol=1e-6, verbose=False
    )
    # print(f"True position : x = {e}, y = {n}, z = {u}")
    if verbose:
        print(f"Estimated position : x = {X_hat[0]}, y = {X_hat[1]}, z = {X_hat[2]}")
        print(f"Residuals : {dY_hat}")
        print(f"Final cost : {cost_hat}")
        print(SigmaXhat)

    if plot:

        plt.figure()
        # Origine du repère locale
        # plt.scatter(
        #     0,
        #     0,
        #     label=fr"O ($\lambda$ = {{{np.round(pos0.lon, 1)}}}°, $\phi$ = {{{np.round(pos0.lat, 1)}}}°, $h$ = {{{np.round(pos0.h, 1)}}}m)"
        # )

        # Position a priori
        plt.scatter(
            X_0[0],
            X_0[1],
            marker="*",
            s=100,
            color="b",
            label=r"$X_0$",
            zorder=2,
        )

        # Position estimée
        plt.scatter(
            X_hat[0],
            X_hat[1],
            marker="*",
            s=100,
            color="r",
            label=r"$\hat{X}$",
            zorder=2,
        )

        # Ellipse de confiance
        alpha = 0.95
        # nombre d'observations
        nobs = y.shape[0]
        # nombre de paramètres inconnus
        nparams = 2
        ddl = nobs - nparams
        draw_ellipse(
            X=X_hat[0:2],
            SigmaX=SigmaXhat[0:2, 0:2],
            ddl=ddl,
            alpha=alpha,
            title="",
            color="k",
            fac=1,
        )

        ax = plt.gca()
        ax.axis("equal")
        plt.xlabel("E [m]")
        plt.ylabel("N [m]")
        plt.title(f"OBS{obs_id}")


if __name__ == "__main__":
    pass
