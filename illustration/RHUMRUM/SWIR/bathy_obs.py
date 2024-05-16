#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bathy_obs.py
@Time    :   2024/05/14 16:03:31
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos
from localisation.verlinden.plateform.init_dataset import init_grid
from publication.PublicationFigure import PubFigure
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as patches


pubfig = PubFigure()

# ======================================================================================================================
# Functions
# ======================================================================================================================


# Use custom colormap function from Earle
def custom_div_cmap(
    numcolors=11, name="custom_div_cmap", mincol="blue", midcol="white", maxcol="red"
):
    """Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    cmap = LinearSegmentedColormap.from_list(
        name=name, colors=[mincol, midcol, maxcol], N=numcolors
    )
    return cmap


def plot_swir_bathy():
    bathy_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm\PVA_RR48\GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
    ds_bathy = xr.open_dataset(bathy_path)

    blevels = [
        -6000,
        -5000,
        -4000,
        -3000,
        -2000,
        -1500,
    ]
    # N = len(blevels) - 1
    # cmap2 = custom_div_cmap(N, mincol="red", midcol="green", maxcol="yellow")
    # cmap2.set_over("0.7")  # light gray

    # bnorm = BoundaryNorm(blevels, ncolors=N, clip=False)

    # ds_bathy.elevation.plot(
    #     norm=bnorm,
    #     cmap=cmap2,
    #     extend="both",
    #     cbar_kwargs={
    #         "label": "Elevation [m]",
    #         "ticks": blevels[:-2] + [blevels[-1]],
    #         "spacing": "proportional",
    #     },
    # )
    ds_bathy.elevation.plot(cmap="jet")
    ds_bathy.elevation.plot.contour(levels=blevels, colors="k", linewidths=0.5)

    return ds_bathy


def plot_swir_obs(ds_bathy):
    rcv_info = {
        "id": ["RR41", "RR42", "RR43", "RR44", "RR45", "RR46", "RR47", "RR48"],
        "lons": [],
        "lats": [],
        "z": [],
        "z_bathy": [],
    }
    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)
        rcv_info["z"].append(pos_obs.elev)

        # z from bathy
        z_bathy = ds_bathy.sel(
            lon=pos_obs.lon, lat=pos_obs.lat, method="nearest"
        ).elevation.values
        rcv_info["z_bathy"].append(z_bathy)

    markers = ["o", "s", "D", "X", "P", "H", "v", "^"]
    for i_obs, obs_id in enumerate(rcv_info["id"]):
        plt.scatter(
            rcv_info["lons"][i_obs],
            rcv_info["lats"][i_obs],
            label=f"{obs_id}: {rcv_info['z'][i_obs]} m",
            s=130,
            zorder=10,
            color="k",
            marker=markers[i_obs],
        )

    plt.gca().axis("equal")

    return rcv_info


def plot_sim_area(rcv_info, minimum_distance_around_rcv, dx, dy):
    grid_info = init_grid(rcv_info, minimum_distance_around_rcv, dx, dy)
    width = grid_info["max_lon"] - grid_info["min_lon"]
    height = grid_info["max_lat"] - grid_info["min_lat"]
    rect = patches.Rectangle(
        (grid_info["min_lon"], grid_info["min_lat"]),
        width=width,
        height=height,
        linewidth=5,
        edgecolor="r",
        facecolor="none",
        label="Simulation area",
    )
    plt.gca().add_patch(rect)
    return grid_info


def plot_swir_bathy_obs():

    dx, dy = 100, 100
    minimum_distance_around_rcv = 50 * 1e3

    plt.figure()
    ds_bathy = plot_swir_bathy()
    rcv_info = plot_swir_obs(ds_bathy)
    grid_info = plot_sim_area(rcv_info, minimum_distance_around_rcv, dx, dy)

    plt.xlabel("Longitude [°E]")
    plt.ylabel("Latitude [°N]")
    plt.title("SWIR bathymetry and OBS position")
    plt.ylim(min(rcv_info["lats"]) - 0.5, max(rcv_info["lats"]) + 0.7)
    plt.xlim(min(rcv_info["lons"]) - 0.5, max(rcv_info["lons"]) + 0.7)
    plt.grid()

    plt.legend(ncol=3, fontsize=14)

    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\RHUMRUM\SWIR\swir_bathy_obs.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_swir_bathy_obs()
