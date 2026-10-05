# %% [markdown]
# # Objectif
#
# Fichiers preprocessés :
#
# * Données de positions : acouplane_pos.nc
# * Données de pression : acouplane_raw_pressure.nc
#
# Date de création : 05/10/2026

# %%
import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.io import loadmat
from datetime import datetime, timedelta

# %%
ROOT_PROGRAM = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
sys.path.append(ROOT_PROGRAM)
from publication.publication_figure import set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.acouplane_cst import *
from real_data_analysis.acouplane.src.acouplane_utils import *

# %% [markdown]
# # Données de position

# %% [markdown]
# ## Chargement des données

# %%
ds_pos = xr.open_dataset(ACOUPLANE_POSITIONS_NC_FPATH)
ds_pos


# %%
def plot_obs_pos(ds_pos, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=id)
        plt.scatter(rcv.rcv_lon, rcv.rcv_lat, marker="d", label=f"{id}", s=200)

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("OBS positions")
    ax.legend()

    return ax


# %% [markdown]
# ### Trajectoire ATALANTE

# # %%
# plt.figure()
# ax = plt.gca()
# plt.scatter(ds_pos.atl_lon, ds_pos.atl_lat)
# plot_obs_pos(ds_pos, ax=ax)

# plt.show()

# %% [markdown]
# ### Tirs

# %%
plt.figure()
ax = plt.gca()
plt.scatter(ds_pos.tir_src_lon, ds_pos.tir_src_lat)
plot_obs_pos(ds_pos, ax=ax)

plt.show()

# %% [markdown]
# # Données de pression

# %% [markdown]
# ## Chargement des données

# %%
ds_pressure = xr.open_dataset(ACOUPLANE_PRESSURE_NC_FPATH)

# %% [markdown]
# ## Example de chargement des données à partir de ce fichier
#
# Les données stockées sont les données raw, il faut donc les convertir et fabriquer le vecteur de temps.

# %%
# Convert to volts
ds_pressure = convert_raw_pressure_to_physical_units(ds_pressure=ds_pressure)

# %%
ds_pressure

# %%
# Build a vector of datetime for each OBS signal
obs_datetimes = build_obs_datetime_vector(ds_pressure=ds_pressure)

# %%
# Slice the dataset to a specific time range
start_dt = datetime(2026, 2, 24, 14, 0, 0)
stop_dt = datetime(2026, 2, 24, 15, 20, 0)

obs1_slice_idx = (
    np.where(obs_datetimes["obs1"] >= start_dt)[0][0],
    np.where(obs_datetimes["obs1"] <= stop_dt)[0][-1],
)
obs2_slice_idx = (
    np.where(obs_datetimes["obs2"] >= start_dt)[0][0],
    np.where(obs_datetimes["obs2"] <= stop_dt)[0][-1],
)
pressure_slice = ds_pressure.isel(
    time_obs1=slice(
        obs1_slice_idx[0],
        obs1_slice_idx[1] + 1,
    ),
    time_obs2=slice(obs2_slice_idx[0], obs2_slice_idx[1] + 1),
)
pressure_slice["time_obs1"] = obs_datetimes["obs1"][
    obs1_slice_idx[0] : obs1_slice_idx[1] + 1
]
pressure_slice["time_obs2"] = obs_datetimes["obs2"][
    obs2_slice_idx[0] : obs2_slice_idx[1] + 1
]
pressure_slice["time_obs1"].attrs = {"unit": "UTC", "long_name": "Time"}
pressure_slice["time_obs2"].attrs = {"unit": "UTC", "long_name": "Time"}

# %%
plt.figure()
pressure_slice.pressure_obs1.plot(label="OBS1", color=color(0))
pressure_slice.pressure_obs2.plot(label="OBS2", color=color(1))
plt.legend()
plt.show()
