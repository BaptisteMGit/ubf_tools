#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   manifold_learning.py
@Time    :   2026/01/09 15:27:23
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt


from matplotlib import ticker
from sklearn import datasets, manifold

n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
# S_points, S_color = datasets.make_swiss_roll(n_samples, random_state=0)

# # set parameters
# length_phi = 15  # length of swiss roll in angular direction
# length_Z = 15  # length of swiss roll in z direction
# sigma = 0.1  # noise strength
# m = 10000  # number of samples

# # create dataset
# phi = length_phi * np.random.rand(m)
# xi = np.random.rand(m)
# Z = length_Z * np.random.rand(m)
# X = 1.0 / 6 * (phi + sigma * xi) * np.sin(phi)
# Y = 1.0 / 6 * (phi + sigma * xi) * np.cos(phi)

# S_points = np.array([X, Y, Z]).transpose()


def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    # plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    # plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"$\Psi_1$")
    ax.set_ylabel(r"$\Psi_2$")


# plot_3d(S_points, S_color, "Original S-curve samples")

n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
n_components = 2  # number of coordinates for the manifold

# isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
# S_isomap = isomap.fit_transform(S_points)

# plot_2d(S_isomap, S_color, "Isomap Embedding")

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def diffusion_map(X, n_components=2, epsilon=None, alpha=1.0, t=1):
    """
    Implémentation de Diffusion Maps.
    X : array (n_samples, n_features)
    n_components : nombre de dimensions de sortie
    epsilon : largeur du kernel. Si None, on l'estime.
    alpha : paramètre de normalisation
    t : nombre d'étapes de diffusion

    Retourne :
      embedding : array (n_samples, n_components)
      lambdas  : valeurs propres
      psi      : vecteurs propres
    """

    n = X.shape[0]

    # 1) calcul des distances et kernel Gaussien
    D2 = squareform(pdist(X, "sqeuclidean"))

    if epsilon is None:
        # heuristique moyenne des distances
        epsilon = np.mean(D2)

    K = np.exp(-D2 / epsilon)

    # 2) normalisation avec alpha
    q = np.sum(K, axis=1)
    q_alpha = q ** (-alpha)

    # normalisation bilatère
    K_tilde = (q_alpha[:, None]) * K * (q_alpha[None, :])

    # 3) matrice de diffusion stochastique
    D_tilde = np.sum(K_tilde, axis=1)
    P = K_tilde / D_tilde[:, None]

    # 4) décomposition spectrale sur P (matrice symétrisée si nécessaire)
    # on utilise eigh (symétrique) pour des données plus stables
    vals, vecs = eigh(P)

    # trier décroissant
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # 5) coordonnées de diffusion
    lambdas = vals[1 : n_components + 1]
    psi = vecs[:, 1 : n_components + 1]

    embedding = psi * (lambdas**t)

    return embedding, lambdas, psi


# Y_diff, lambdas, psi = diffusion_map(
#     S_points, n_components=n_components, epsilon=1, alpha=1.0, t=1
# )
# # Visualiser
# plot_2d(Y_diff, S_color, "Diffusion Embedding")


from pydiffmap import diffusion_map as dm

# initialize Diffusion map object.
neighbor_params = {"n_jobs": -1, "algorithm": "ball_tree"}

mydmap = dm.DiffusionMap.from_sklearn(
    n_evecs=2, k=200, epsilon="bgh", alpha=1.0, neighbor_params=neighbor_params
)
# fit to data and return the diffusion map.
dmap = mydmap.fit_transform(S_points)

print(mydmap.epsilon_fitted)


from pydiffmap.visualization import embedding_plot, data_plot

embedding_plot(mydmap, scatter_kwargs={"c": dmap[:, 0], "cmap": "Spectral"})
data_plot(mydmap, dim=3, scatter_kwargs={"cmap": "Spectral"})


Y_diff, lambdas, psi = diffusion_map(
    S_points, n_components=n_components, epsilon=mydmap.epsilon_fitted, alpha=1.0, t=1
)
# Visualiser
plot_2d(Y_diff, S_color, "Diffusion Embedding")


# embedding_plot(
#     mydmap, scatter_kwargs={"c": dmap[:, 0], "s": mydmap.q, "cmap": "Spectral"}
# )
# data_plot(mydmap, dim=3, scatter_kwargs={"cmap": "Spectral"})

# # Test pour différentes valeurs de epsilon
# epsilon = np.linspace(0.1, 2, 20)
# fig, axs = plt.subplots(figsize=(15, 10), ncols=5, nrows=4)

# idx = 0
# for eps in epsilon:
#     Y_diff, lambdas, psi = diffusion_map(
#         S_points, n_components=n_components, epsilon=eps, alpha=1.0, t=1
#     )

#     i = idx // 5
#     j = idx % 5
#     add_2d_scatter(axs[i, j], Y_diff, S_color, title=f"Diffusion Map (eps={eps:.2f})")

#     idx += 1

plt.show()
# def diffusion_map():
#     pass
