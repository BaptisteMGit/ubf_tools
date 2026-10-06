#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   pva_cvo_proxy.py
@Time    :   2025/06/04 13:42:56
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import xarray as xr
import matplotlib.pyplot as plt


fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\cds\cds_swir\data_stream-oper_stepType-accum.nc"
ds_prec = xr.open_dataset(fpath)


plt.figure()
ds_prec.tp.isel(longitude=0, latitude=0).plot()
# plt.show()

fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\cds\cds_swir\data_stream-oper_stepType-instant.nc"
ds_wind = xr.open_dataset(fpath)

u_norm = (ds_wind.u10**2 + ds_wind.v10**2) ** 0.5
plt.figure()
ds_wind.u10.isel(longitude=0, latitude=0).plot(label="u10")
ds_wind.v10.isel(longitude=0, latitude=0).plot(label="v10")
u_norm.isel(longitude=0, latitude=0).plot(label=r"$|| u ||$")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Wind speed [m/s]")
# plt.show()

# Quiver plot
plt.figure()
ds_wind.isel(valid_time=5).plot.quiver(
    x="longitude",
    y="latitude",
    u="u10",
    v="v10",
)

fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

ds_wind.u10.isel(valid_time=5).plot(
    x="longitude",
    y="latitude",
    ax=axs[0],
    cmap="bwr",
    # add_colorbar=False,
)
ds_wind.v10.isel(valid_time=5).plot(
    x="longitude",
    y="latitude",
    ax=axs[1],
    cmap="bwr",
    # add_colorbar=False,
)

u_norm.isel(valid_time=5).plot(
    x="longitude",
    y="latitude",
    ax=axs[2],
    cmap="jet",
)

axs[1].set_ylabel("")
axs[2].set_ylabel("")

plt.show()

print()
