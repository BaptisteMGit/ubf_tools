#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   check_hamilton_sediment.py
@Time    :   2026/09/08 08:59:53
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

from publication.publication_figure import color
from illustration_rtf.src.sensitivity import celerity_density_Hamilton_Bachman_1982
from source.global_constants import sediments_EMODnet, sediments_TG

sediments = sediments_EMODnet.copy()
sediments.update(sediments_TG)

for sediment in sediments.values():
    # Display sediment properties
    print(
        f"{sediment['label']}: rho = {sediment['rho']} g.cm-3, c_p = {sediment['c_p']} m.s-1",
    )

rho2 = np.linspace(1.0, 2.5, 1000)

c2 = celerity_density_Hamilton_Bachman_1982(rho2)


# Plot all
err = []
ised = 0
plt.figure()
plt.plot(rho2, c2, color="k", label="Hamilton and Bachman (1982)")
for sed_label, sediment in sediments.items():
    plt.scatter(sediment["rho"], sediment["c_p"], label=sed_label, color=color(ised))
    # plt.scatter(
    #     sediment["rho"],
    #     celerity_density_Hamilton_Bachman_1982(sediment["rho"]),
    #     label=sed_label,
    #     color=color(ised),
    #     marker="x",
    # )

    err.append(
        sediment["c_p"] - celerity_density_Hamilton_Bachman_1982(sediment["rho"])
    )
    ised += 1
err = np.array(err)

rmse = np.sqrt(np.sum(np.abs(err) ** 2))
print(f"RMSE = {rmse} m.s-1")

y_mean = np.mean(np.array([sediment["c_p"] for sediment in sediments.values()]))
err_mean = np.array([sediment["c_p"] - y_mean for sediment in sediments.values()])
r2 = 1 - np.sum(err**2) / np.sum(err_mean**2)
print(f"Determination coef r2 = {r2}")

plt.xlabel(r"$\rho_2$ [g~cm$^{-3}$]")
plt.ylabel(r"$c_2$ [m~s$^{-1}$]")
plt.legend(ncols=3, fontsize=16)
plt.title(rf"$r^2 = {{{r2:.2f}}}$")

# Plot only sediments within error margins
err_th = 300
err = []
y = []
ised = 0
plt.figure()
plt.plot(rho2, c2, color="k", label="Hamilton and Bachman (1982)")
for sed_label, sediment in sediments.items():
    err_ = np.abs(
        sediment["c_p"] - celerity_density_Hamilton_Bachman_1982(sediment["rho"])
    )

    if err_ < err_th:
        plt.scatter(
            sediment["rho"], sediment["c_p"], label=sed_label, color=color(ised)
        )
        err.append(err_)
        y.append(sediment["c_p"])
        ised += 1
    else:
        print(
            f"Sediment {sed_label} discarded: error = {err_} m.s-1 > threshold = {err_th} m.s-1"
        )
err = np.array(err)
y = np.array(y)

rmse = np.sqrt(np.sum(err**2))
print(f"RMSE = {rmse} m.s-1")

err_mean = y - np.mean(y)
r2 = 1 - np.sum(err**2) / np.sum(err_mean**2)
print(f"Determination coef r2 = {r2}")

plt.xlabel(r"$\rho_2$ [g~cm$^{-3}$]")
plt.ylabel(r"$c_2$ [m~s$^{-1}$]")
plt.legend(ncols=3, fontsize=16)
plt.title(rf"$r^2 = {{{r2:.2f}}}$")

plt.show()
