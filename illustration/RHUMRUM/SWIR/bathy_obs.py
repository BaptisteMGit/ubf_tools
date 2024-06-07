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
import numpy as np
import matplotlib.pyplot as plt

from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos
from localisation.verlinden.plateform.init_dataset import init_grid
from publication.PublicationFigure import PubFigure
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as patches


pubfig = PubFigure(legend_fontsize=20)

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
    
    bathy_path = r"data/bathy/mmdpm/PVA_RR48/GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
    # bathy_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm\PVA_RR48\GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
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


def plot_swir_obs(ds_bathy, col=None):
    rcv_info = {
        # "id": ["RR41", "RR42", "RR43", "RR44", "RR45", "RR46", "RR47", "RR48"],
        # "id": ["RRpftim0", "RRpftim1", "RRpftim2"],
        "id": ["R1", "R2", "R3"],

        # "id": ["RR45", "RR48", "RR44"],
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
    if col is None:
        col = ["k"] * len(rcv_info["id"])
    for i_obs, obs_id in enumerate(rcv_info["id"]):
        plt.scatter(
            rcv_info["lons"][i_obs],
            rcv_info["lats"][i_obs],
            # label=f"{obs_id}: {rcv_info['z'][i_obs]} m",
            s=130,
            zorder=10,
            color="k",
            # color=col[i_obs],
            # marker=markers[i_obs],
        )

        # Plot obs_id next to the point
        plt.text(
            rcv_info["lons"][i_obs] + 0.01,
            rcv_info["lats"][i_obs] - 0.01,
            f"{obs_id}",
            fontsize=22,
            color="k",
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
    minimum_distance_around_rcv = 5 * 1e3
    deg_offset = 0.1

    plt.figure()
    ds_bathy = plot_swir_bathy()
    rcv_info = plot_swir_obs(ds_bathy)
    grid_info = plot_sim_area(rcv_info, minimum_distance_around_rcv, dx, dy)

    plt.xlabel("Longitude [°E]")
    plt.ylabel("Latitude [°N]")
    plt.title("SWIR bathymetry and OBS position")
    ymin, ymax = (
        min(rcv_info["lats"]) - deg_offset,
        max(rcv_info["lats"]) + deg_offset + 0.2,
    )
    xmin, xmax = (
        min(rcv_info["lons"]) - deg_offset,
        max(rcv_info["lons"]) + deg_offset + 0.2,
    )
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    # # Plot border
    # xmin, xmax = plt.gca().get_xlim()
    # ymin, ymax = plt.gca().get_ylim()
    # width = xmax - xmin
    # height = ymax - ymin
    # rect = patches.Rectangle(
    #     (xmin, ymin),
    #     width=width - 0.001,
    #     height=height - 0.001,
    #     linewidth=5,
    #     edgecolor="r",
    #     facecolor="none",
    #     # label="Simulation area",
    # )
    # plt.gca().add_patch(rect)

    # ax = plt.gca()
    # fig = plt.gcf()
    # # Largeur des bordures en points
    # border_width_points = 6  # largeur de la bordure blanche
    # border_width_inches = (
    #     border_width_points / 72
    # )  # convertir en pouces (1 pouce = 72 points)
    # border_width_fig = (
    #     border_width_inches / fig.get_size_inches()[0]
    # )  # convertir en unités de figure

    # # Ajouter la bordure pointillée noire (plus étroite)
    # add_dotted_border(ax, "black", border_width_points, (0, (5, 5)))

    # # # Ajuster la position des axes pour décaler la bordure à l'intérieur
    # pos = ax.get_position()
    # ax.set_position(
    #     [
    #         pos.x0 + border_width_fig,
    #         pos.y0 + border_width_fig,
    #         pos.width + 2 * border_width_fig,
    #         pos.height + 2 * border_width_fig,
    #     ]
    # )

    plt.grid()
    plt.legend(ncol=2)

    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\RHUMRUM\SWIR\swir_bathy_obs_no_border.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_swir_bathy_obs_src():
    dx, dy = 100, 100
    minimum_distance_around_rcv = 5 * 1e3

    ds_bathy = plot_swir_bathy()
    rcv_info = plot_swir_obs(ds_bathy)
    grid_info = plot_sim_area(rcv_info, minimum_distance_around_rcv, dx, dy)

    initial_ship_pos = {
        "lon": rcv_info["lons"][0],
        "lat": rcv_info["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    plt.figure()
    plot_swir_bathy()
    plot_swir_obs(ds_bathy, col=["k", "k", "k"])
    plt.scatter(
        initial_ship_pos["lon"],
        initial_ship_pos["lat"],
        # label="Ship position",
        s=400,
        zorder=10,
        color="r",
        marker="*",
    )

    plt.text(
        initial_ship_pos["lon"] + 0.01,
        initial_ship_pos["lat"] - 0.01,
        "ship",
        fontsize=22,
        color="k",
    )

    plt.xlabel("Longitude [°E]")
    plt.ylabel("Latitude [°N]")
    plt.title("SWIR bathymetry and OBS position")
    ymin, ymax = (
        grid_info["min_lat"],
        grid_info["max_lat"],
    )
    xmin, xmax = (
        grid_info["min_lon"],
        grid_info["max_lon"],
    )
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.grid()
    # plt.legend(ncol=2)

    # path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\RHUMRUM\SWIR\swir_bathy_obs_no_border_src.png"
    path = r"/home/program/ubf_tools/data/bathy/mmdpm/PVA_RR48/swir_bathy_obs_no_border_src.png"

    plt.savefig(path, dpi=300, bbox_inches="tight")
    # plt.show()


# Fonction pour ajouter une bordure pointillée
def add_dotted_border(ax, color, linewidth, linestyle):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)
        spine.set_linestyle(linestyle)
        spine.set_capstyle("butt")


def plot_bathy_profile():
    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\testcase_working_directory\testcase3_1\bathy.csv"
    df = pd.read_csv(path)
    r, z = df.iloc[:, 0], df.iloc[:, 1]
    plt.figure()
    plt.plot(r, z)
    plt.xlabel("Range [km]")
    plt.ylabel("Depth [m]")
    plt.title("Bathymetry profile")
    plt.grid()
    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\RHUMRUM\SWIR\bathy_profile.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")

    # Max variation
    variation = np.diff(z)
    max_var = np.max(np.abs(variation))


if __name__ == "__main__":
    # plot_swir_bathy_obs()
    plot_swir_bathy_obs_src()
    # plot_bathy_profile()
