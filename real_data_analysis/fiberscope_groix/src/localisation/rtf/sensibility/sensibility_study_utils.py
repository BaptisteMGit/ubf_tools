#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sensibility_study_utils.py
@Time    :   2026/04/15 13:33:43
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================


import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors


import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from scipy.special import kl_div
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import (
    wasserstein_distance,
    wasserstein_distance_nd,
    entropy,
    gaussian_kde,
)
from datetime import timedelta, datetime
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

from publication.publication_figure import (
    PubFigure,
    LargeFigure,
    color,
    set_subfigures_abc_labels,
)
from propa.rtf.rtf_utils import D_hermitian_angle_fast, D_euclidian


from real_data_analysis.fiberscope_groix.src.fiberscope_groix_manager import (
    # ActiveFiberscopeManager,
    # PassiveFiberscopeManager,
    BandFilter,
)

from real_data_analysis.fiberscope_groix.src.localisation.rtf.rtf_mfp import (
    RTF_MFP_Processor,
)
from misc import progression_bar


# ======================================================================================================================
# Plotting routines
# ======================================================================================================================
# Tracer des positions d'émission et des positions des OBS sur la carte
def plot_seq_replica_positions(df_seq, ds_gps, root_fig=None):
    """
    Plot the interpolated GPS positions of the source along the sequence, as well as the a priori positions of the OBS.
    Parameters
    ----------
    df_seq : pandas.DataFrame
        DataFrame containing the interpolated GPS positions of the source along the sequence.
    ds_gps : xarray.Dataset
        Dataset containing the a priori positions of the OBS.
    root_fig : str
        Root directory to save the figure.
    """
    pfig = PubFigure(size=(10, 8), legend_fontsize=16)
    # Plot
    plt.figure()

    seq_id = df_seq["sequence_id"].iloc[0]
    # Série de positions successives
    plt.scatter(
        df_seq["emission_interp_e_gps"],
        df_seq["emission_interp_n_gps"],
        marker="+",
        label=f"Event ({seq_id})",
        # c=np.arange(df_seq["emission_interp_e_gps"].size),
        c=df_seq["pulse_id"],
        cmap="jet",
    )
    plt.colorbar(label="Replica ID")

    keys = ["obs1", "obs2", "obs3"]
    label = {
        "obs1": "1S",
        "obs2": "2S",
        "obs3": "3S",
        "t1": "1",
        "t2": "2",
        "t3": "3",
        "t4": "4",
        "t5": "5",
    }
    for ik, k in enumerate(keys):
        e = ds_gps.attrs[f"{k}_e_apriori"]
        n = ds_gps.attrs[f"{k}_n_apriori"]
        plt.scatter(
            e,
            n,
            marker="D",
            label=label[k],
            zorder=10,
            color=color(ik),
            s=40,
        )

    plt.legend()
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    if root_fig is not None:
        fpath = os.path.join(root_fig, "emission_positions.png")
        plt.savefig(fpath)


def plot_seq_replica_positions_wgs84(
    df_seq, ds_gps, ds_bathy, root_fig=None, add_transect_points=False
):
    """
    Plot:
        - regional bathymetric context (main map)
        - zoom on replica sequence and OBS geometry (inset)

    Parameters
    ----------
    df_seq : pandas.DataFrame

        Must contain:
            emission_longitude_gps
            emission_latitude_gps
            sequence_id

    ds_gps : xr.Dataset

        Must contain attributes:
            obs1_lon_apriori
            obs1_lat_apriori
            obs2_lon_apriori
            obs2_lat_apriori
            obs3_lon_apriori
            obs3_lat_apriori

    ds_bathy : xr.Dataset

        Must contain:
            lon
            lat
            elevation

    root_fig : str, optional
    """

    # ======================================================================
    # OBS positions
    # ======================================================================

    obs_keys = ["obs1", "obs2", "obs3"]

    obs_lon = np.array(
        [
            ds_gps.attrs["obs1_lon_apriori"],
            ds_gps.attrs["obs2_lon_apriori"],
            ds_gps.attrs["obs3_lon_apriori"],
        ]
    )

    obs_lat = np.array(
        [
            ds_gps.attrs["obs1_lat_apriori"],
            ds_gps.attrs["obs2_lat_apriori"],
            ds_gps.attrs["obs3_lat_apriori"],
        ]
    )

    # ======================================================================
    # Bathymetry
    # ======================================================================

    # Main map extent
    all_lon = np.concatenate(
        [
            df_seq["emission_longitude_gps"].values,
            obs_lon,
        ]
    )

    all_lat = np.concatenate(
        [
            df_seq["emission_latitude_gps"].values,
            obs_lat,
        ]
    )

    dx = all_lon.max() - all_lon.min()
    dy = all_lat.max() - all_lat.min()

    pad = 10 * max(dx, dy)

    ds_bathy.sel(lon=slice(all_lon.min() - pad, all_lon.max() + pad)).sel(
        lat=slice(all_lat.min() - pad, all_lat.max() + pad)
    )

    # zoom_extent = [
    #     all_lon.min() - pad,
    #     all_lon.max() + pad,
    #     all_lat.min() - pad,
    #     all_lat.max() + pad,
    # ]

    lon = ds_bathy.lon.values
    lat = ds_bathy.lat.values
    z = -ds_bathy.elevation

    # ls = LightSource(
    #     azdeg=315,
    #     altdeg=45,
    # )

    # rgb = ls.shade(
    #     -z,
    #     cmap=plt.cm.Blues,
    #     vert_exag=8,
    #     blend_mode="overlay",
    # )

    # ----------------------------------------------------------------------
    # Land mask
    # ----------------------------------------------------------------------

    # land_mask = z >= 0

    # rgb_land = rgb.copy()

    # rgb_land[land_mask, :3] = [0.90, 0.90, 0.90]
    # rgb_land[~land_mask] = np.nan

    # ======================================================================
    # Zoom extent
    # ======================================================================

    # all_lon = np.concatenate(
    #     [
    #         df_seq["emission_longitude_gps"].values,
    #         obs_lon,
    #     ]
    # )

    # all_lat = np.concatenate(
    #     [
    #         df_seq["emission_latitude_gps"].values,
    #         obs_lat,
    #     ]
    # )

    # dx = all_lon.max() - all_lon.min()
    # dy = all_lat.max() - all_lat.min()

    pad = 0.75 * max(dx, dy)

    zoom_extent = [
        all_lon.min() - pad,
        all_lon.max() + pad,
        all_lat.min() - pad,
        all_lat.max() + pad,
    ]

    ds_bathy_zoom = ds_bathy.sel(lon=slice(zoom_extent[0], zoom_extent[1])).sel(
        lat=slice(zoom_extent[2], zoom_extent[3])
    )
    z_zoom = -ds_bathy_zoom.elevation

    # ======================================================================
    # Figure
    # ======================================================================

    fig = plt.figure(figsize=(14, 10))

    # ======================================================================
    # MAIN MAP : REGIONAL CONTEXT
    # ======================================================================

    ax = fig.add_axes(
        [0.05, 0.05, 0.90, 0.90],
        projection=ccrs.PlateCarree(),
    )

    # ----------------------------------------------------------------------
    # Bathymetry
    # ----------------------------------------------------------------------

    # im = ax.imshow(
    #     rgb,
    #     origin="lower",
    #     extent=[
    #         lon.min(),
    #         lon.max(),
    #         lat.min(),
    #         lat.max(),
    #     ],
    #     transform=ccrs.PlateCarree(),
    #     zorder=1,
    # )
    z_zoom_pos = z_zoom.where(z_zoom > 0)
    vmin = 0
    vmax = z_zoom_pos.max() * 0.9

    f = z.plot(
        x="lon",
        cmap=plt.cm.Blues,
        transform=ccrs.PlateCarree(),
        zorder=1,
        vmin=vmin,
        vmax=vmax,
        # cbar_kwargs={"fontsize": 22},
    )

    f.colorbar.ax.tick_params(labelsize=26)
    f.colorbar.set_label("Depth [m]", fontsize=28)
    # f.cbar.ax.tick_params(labelsize=22)
    # cb.set_label("Replica ID", fontsize=20, color="white")
    # cb.ax.tick_params(which="both", color="white", labelcolor="white", labelsize=20)

    levels = np.arange(
        np.floor(z.min() / 50) * 50,
        0,
        50,
    )

    ax.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # im = ax.imshow(
    #     rgb_land,
    #     origin="lower",
    #     extent=[
    #         lon.min(),
    #         lon.max(),
    #         lat.min(),
    #         lat.max(),
    #     ],
    #     transform=ccrs.PlateCarree(),
    #     zorder=3,
    # )

    ax.contour(
        lon,
        lat,
        z,
        levels=[0],
        colors="k",
        linewidths=1,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # # Add bathy colobar
    # # Create a normalized scalar mappable from bathymetry (not RGB image)
    # cmap = plt.cm.Blues
    # norm = plt.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))

    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])  # required for older matplotlib compatibility

    # cax = fig.add_axes([0.92, 0.15, 0.015, 0.60])

    # cb = plt.colorbar(sm, cax=cax)

    # ----------------------------------------------------------------------
    # OBS locations
    # ----------------------------------------------------------------------

    for ik, obs in enumerate(obs_keys):

        ax.scatter(
            ds_gps.attrs[f"{obs}_lon_apriori"],
            ds_gps.attrs[f"{obs}_lat_apriori"],
            marker="D",
            label=obs.upper(),
            s=70,
            color=color(ik),
            edgecolors="black",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

    if add_transect_points:
        transect_keys = ["t1", "t2", "t3", "t4", "t5"]
        offset_col = len(obs_keys)  # to avoid color overlap with OBS markers
        for ik, transect_pt in enumerate(transect_keys):

            ax.scatter(
                ds_gps.attrs[f"{transect_pt}_lon_apriori"],
                ds_gps.attrs[f"{transect_pt}_lat_apriori"],
                marker="o",
                label=transect_pt.upper(),
                s=70,
                color=color(ik + offset_col),
                edgecolors="black",
                linewidths=0.8,
                transform=ccrs.PlateCarree(),
                zorder=35,
            )

    # ----------------------------------------------------------------------
    # Sequence trajectory
    # ----------------------------------------------------------------------

    ax.plot(
        df_seq["emission_longitude_gps"],
        df_seq["emission_latitude_gps"],
        color="red",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        zorder=21,
        # label=f"Emissions",
    )

    # start point

    # ax.scatter(
    #     df_seq["emission_longitude_gps"].iloc[0],
    #     df_seq["emission_latitude_gps"].iloc[0],
    #     marker="o",
    #     s=80,
    #     color="lime",
    #     edgecolor="black",
    #     transform=ccrs.PlateCarree(),
    #     zorder=22,
    # )

    # ----------------------------------------------------------------------
    # Zoom rectangle
    # ----------------------------------------------------------------------

    rect = Rectangle(
        (zoom_extent[0], zoom_extent[2]),
        zoom_extent[1] - zoom_extent[0],
        zoom_extent[3] - zoom_extent[2],
        fill=False,
        edgecolor="red",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        zorder=25,
    )

    ax.add_patch(rect)

    # ----------------------------------------------------------------------
    # Regional extent
    # ----------------------------------------------------------------------

    ax.set_extent(
        [
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max(),
        ]
    )

    gl = ax.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.5,
    )

    gl.top_labels = False
    gl.right_labels = False

    seq_id = df_seq["sequence_id"].iloc[0]

    # ax.set_title(
    #     f"Sequence {seq_id} - Regional context",
    #     fontsize=16,
    # )

    ax.legend(fontsize=20, loc="upper right")
    gl.xlabel_style = {"size": 28}
    gl.ylabel_style = {"size": 28}

    # ======================================================================
    # INSET : ZOOM ON GEOMETRY
    # ======================================================================

    # axins = fig.add_axes(
    #     [0.47, 0.06, 0.35, 0.35],
    #     projection=ccrs.PlateCarree(),
    # )

    # inset_map_pos = [0.4, 0.06, 0.35, 0.35]
    # inset_map_pos = [0.38, 0.6, 0.35, 0.35]
    inset_map_pos = [0.075, 0.06, 0.35, 0.35]

    axins = fig.add_axes(
        inset_map_pos,
        projection=ccrs.PlateCarree(),
    )

    # ----------------------------------------------------------------------
    # Bathymetry
    # ----------------------------------------------------------------------

    z_zoom.plot(
        x="lon",
        cmap=plt.cm.Blues,
        transform=ccrs.PlateCarree(),
        zorder=1,
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
    )

    # axins.imshow(
    #     rgb,
    #     origin="lower",
    #     extent=[
    #         lon.min(),
    #         lon.max(),
    #         lat.min(),
    #         lat.max(),
    #     ],
    #     transform=ccrs.PlateCarree(),
    #     zorder=1,
    # )

    levels = np.arange(
        np.floor(z.min() / 10) * 10,
        0,
        10,
    )

    axins.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # axins.imshow(
    #     rgb_land,
    #     origin="lower",
    #     extent=[
    #         lon.min(),
    #         lon.max(),
    #         lat.min(),
    #         lat.max(),
    #     ],
    #     transform=ccrs.PlateCarree(),
    #     zorder=3,
    # )

    # axins.contour(
    #     lon,
    #     lat,
    #     z,
    #     levels=[0],
    #     colors="k",
    #     linewidths=1,
    #     transform=ccrs.PlateCarree(),
    #     zorder=4,
    # )

    # ----------------------------------------------------------------------
    # Replica positions
    # ----------------------------------------------------------------------

    sc = axins.scatter(
        df_seq["emission_longitude_gps"],
        df_seq["emission_latitude_gps"],
        c=np.arange(df_seq.shape[0]),
        # color="red",
        cmap="spring",
        marker="+",
        s=120,
        linewidths=2,
        transform=ccrs.PlateCarree(),
        zorder=30,
        label="Emissions",
    )

    axins.legend(fontsize=20, loc="upper left")

    # ----------------------------------------------------------------------
    # OBS positions
    # ----------------------------------------------------------------------

    for ik, obs in enumerate(obs_keys):

        axins.scatter(
            ds_gps.attrs[f"{obs}_lon_apriori"],
            ds_gps.attrs[f"{obs}_lat_apriori"],
            marker="D",
            s=120,
            color=color(ik),
            edgecolors="black",
            linewidths=1,
            transform=ccrs.PlateCarree(),
            zorder=40,
        )

    # ----------------------------------------------------------------------
    # Zoom extent
    # ----------------------------------------------------------------------

    axins.set_extent(
        zoom_extent,
        crs=ccrs.PlateCarree(),
    )

    gl2 = axins.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.3,
    )

    gl2.top_labels = False
    gl2.left_labels = False
    gl2.right_labels = False
    gl2.bottom_labels = False

    # axins.set_title(
    #     "Experiment geometry",
    #     fontsize=10,
    # )

    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2)

    # ======================================================================
    # Colorbar
    # ======================================================================
    # [0.4, 0.06, 0.35, 0.35],

    colorbar_pos = [
        inset_map_pos[0] + (inset_map_pos[-2] - 0.05),
        inset_map_pos[1],
        0.015,
        inset_map_pos[-1],
    ]
    cax = fig.add_axes(colorbar_pos)

    # ticks = np.linspace(
    #     np.arange(df_seq.shape[0])[0], np.arange(df_seq.shape[0])[-1], 4
    # )

    cb = plt.colorbar(
        sc,
        cax=cax,
        # ticks=ticks,
    )

    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.set_label("Replica ID", fontsize=24, color="white")
    cb.ax.tick_params(which="both", color="white", labelcolor="white", labelsize=24)

    # ======================================================================
    # Save
    # ======================================================================

    if root_fig is not None:

        os.makedirs(root_fig, exist_ok=True)

        fpath = os.path.join(
            root_fig,
            "emission_positions_context_and_zoom.png",
        )

        fig.savefig(
            fpath,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax, axins


# def plot_seq_replica_positions_wgs84(
#     df_seq,
#     ds_gps,
#     ds_bathy,
#     root_fig=None,
# ):
#     """
#     Plot replica positions and OBS locations over a hillshaded bathymetric map.

#     Parameters
#     ----------
#     df_seq : pandas.DataFrame
#         DataFrame containing:
#             - emission_longitude_gps
#             - emission_latitude_gps
#             - sequence_id

#     ds_gps : xarray.Dataset
#         Dataset whose attributes contain:
#             obs1_lon_apriori, obs1_lat_apriori,
#             obs2_lon_apriori, obs2_lat_apriori,
#             obs3_lon_apriori, obs3_lat_apriori

#     ds_bathy : xarray.Dataset
#         Bathymetry dataset containing:
#             - lon
#             - lat
#             - elevation

#     root_fig : str, optional
#         Output directory for saving the figure.
#     """

#     # =========================================================================
#     # Bathymetry
#     # =========================================================================

#     z = ds_bathy["elevation"].values
#     lon = ds_bathy["lon"].values
#     lat = ds_bathy["lat"].values

#     # Hillshade
#     ls = LightSource(
#         azdeg=315,
#         altdeg=45,
#     )

#     rgb = ls.shade(
#         -z,  # invert bathymetry
#         cmap=plt.cm.Blues,
#         vert_exag=8,
#         blend_mode="overlay",
#     )

#     # =========================================================================
#     # Figure
#     # =========================================================================

#     fig = plt.figure(figsize=(12, 10))

#     ax = fig.add_subplot(
#         111,
#         projection=ccrs.PlateCarree(),
#     )

#     # =========================================================================
#     # Bathymetry background
#     # =========================================================================

#     ax.imshow(
#         rgb,
#         origin="lower",
#         extent=[
#             lon.min(),
#             lon.max(),
#             lat.min(),
#             lat.max(),
#         ],
#         transform=ccrs.PlateCarree(),
#         zorder=1,
#     )

#     # Bathymetric contours

#     contour_step = 10

#     levels = np.arange(
#         np.floor(z.min() / contour_step) * contour_step,
#         0,
#         contour_step,
#     )

#     ax.contour(
#         lon,
#         lat,
#         z,
#         levels=levels,
#         colors="k",
#         linewidths=0.3,
#         alpha=0.4,
#         transform=ccrs.PlateCarree(),
#         zorder=2,
#     )

#     # =========================================================================
#     # Land mask
#     # =========================================================================

#     land_mask = z >= 0

#     rgb_land = rgb.copy()

#     rgb_land[land_mask, :3] = [0.90, 0.90, 0.90]
#     rgb_land[~land_mask] = np.nan

#     ax.imshow(
#         rgb_land,
#         origin="lower",
#         extent=[
#             lon.min(),
#             lon.max(),
#             lat.min(),
#             lat.max(),
#         ],
#         transform=ccrs.PlateCarree(),
#         zorder=3,
#     )

#     # Coastlines

#     # ax.add_feature(
#     #     cfeature.LAND,
#     #     facecolor="lightgray",
#     #     zorder=4,
#     # )

#     ax.contour(
#         ds_bathy.lon, ds_bathy.lat, ds_bathy.elevation, levels=[0], colors="k", zorder=4
#     )

#     # ax.coastlines(
#     #     resolution="10m",
#     #     linewidth=1,
#     #     zorder=4,
#     # )

#     # =========================================================================
#     # Replica positions
#     # =========================================================================

#     seq_id = df_seq["sequence_id"].iloc[0]

#     sc = ax.scatter(
#         df_seq["emission_longitude_gps"],
#         df_seq["emission_latitude_gps"],
#         c=np.arange(df_seq.shape[0]),
#         cmap="viridis",
#         marker="+",
#         s=100,
#         linewidths=2,
#         label=f"Event {seq_id}",
#         transform=ccrs.PlateCarree(),
#         zorder=20,
#     )

#     cbar = plt.colorbar(
#         sc,
#         ax=ax,
#         pad=0.02,
#     )

#     cbar.set_label("Replica ID")

#     # =========================================================================
#     # OBS positions
#     # =========================================================================

#     obs_keys = ["obs1", "obs2", "obs3"]

#     obs_labels = {
#         "obs1": "OBS1",
#         "obs2": "OBS2",
#         "obs3": "OBS3",
#     }

#     for ik, obs in enumerate(obs_keys):

#         lon_obs = ds_gps.attrs[f"{obs}_lon_apriori"]
#         lat_obs = ds_gps.attrs[f"{obs}_lat_apriori"]

#         ax.scatter(
#             lon_obs,
#             lat_obs,
#             marker="D",
#             s=150,
#             color=color(ik),  # assumes your custom color() function exists
#             edgecolors="black",
#             linewidths=1,
#             label=obs_labels[obs],
#             transform=ccrs.PlateCarree(),
#             zorder=30,
#         )

#     # =========================================================================
#     # Automatic extent
#     # =========================================================================

#     obs_lon = np.array(
#         [
#             ds_gps.attrs["obs1_lon_apriori"],
#             ds_gps.attrs["obs2_lon_apriori"],
#             ds_gps.attrs["obs3_lon_apriori"],
#         ]
#     )

#     obs_lat = np.array(
#         [
#             ds_gps.attrs["obs1_lat_apriori"],
#             ds_gps.attrs["obs2_lat_apriori"],
#             ds_gps.attrs["obs3_lat_apriori"],
#         ]
#     )

#     all_lon = np.concatenate(
#         [
#             df_seq["emission_longitude_gps"].values,
#             obs_lon,
#         ]
#     )

#     all_lat = np.concatenate(
#         [
#             df_seq["emission_latitude_gps"].values,
#             obs_lat,
#         ]
#     )

#     pad = 0.10

#     ax.set_extent(
#         [
#             all_lon.min() - pad,
#             all_lon.max() + pad,
#             all_lat.min() - pad,
#             all_lat.max() + pad,
#         ],
#         crs=ccrs.PlateCarree(),
#     )

#     # =========================================================================
#     # Gridlines
#     # =========================================================================

#     gl = ax.gridlines(
#         draw_labels=True,
#         linestyle=":",
#         linewidth=0.5,
#     )

#     gl.top_labels = False
#     gl.right_labels = False

#     # =========================================================================
#     # Labels
#     # =========================================================================

#     ax.set_title(
#         f"Emission sequence {seq_id}",
#         fontsize=16,
#     )

#     ax.legend(
#         loc="best",
#         fontsize=12,
#     )

#     # =========================================================================
#     # Save
#     # =========================================================================

#     if root_fig is not None:

#         os.makedirs(root_fig, exist_ok=True)

#         fpath = os.path.join(
#             root_fig,
#             "emission_positions_wgs84_bathy.png",
#         )

#         fig.savefig(
#             fpath,
#             dpi=100,
#             bbox_inches="tight",
#         )

#     return fig, ax


# def plot_seq_replica_positions_wgs84(df_seq, ds_gps, root_fig):
#     """
#     Plot the interpolated GPS positions of the source along the sequence, as well as the a priori positions of the OBS in WGS84 coordinates (latitude and longitude).
#     Parameters
#     ----------
#     df_seq : pandas.DataFrame
#         DataFrame containing the interpolated GPS positions of the source along the sequence.
#     ds_gps : xarray.Dataset
#         Dataset containing the a priori positions of the OBS.
#     root_fig : str
#         Root directory to save the figure.
#     """
#     pfig = PubFigure(size=(10, 8), legend_fontsize=16)
#     # Plot
#     plt.figure()

#     seq_id = df_seq["sequence_id"].iloc[0]
#     # Série de positions successives
#     plt.scatter(
#         df_seq["emission_longitude_gps"],
#         df_seq["emission_latitude_gps"],
#         marker="+",
#         label=f"Event ({seq_id})",
#         c=np.arange(df_seq["emission_longitude_gps"].size),
#         cmap="vanimo",
#         s=100,
#     )
#     plt.colorbar(label="Replica ID")

#     keys = ["obs1", "obs2", "obs3"]
#     label = {
#         "obs1": "OBS1",
#         "obs2": "OBS2",
#         "obs3": "OBS3",
#         "t1": "1",
#         "t2": "2",
#         "t3": "3",
#         "t4": "4",
#         "t5": "5",
#     }
#     for ik, k in enumerate(keys):
#         e = ds_gps.attrs[f"{k}_lon_apriori"]
#         n = ds_gps.attrs[f"{k}_lat_apriori"]
#         plt.scatter(
#             e,
#             n,
#             marker="D",
#             label=label[k],
#             zorder=10,
#             color=color(ik),
#             s=150,
#         )

#     plt.legend(fontsize=16)
#     plt.xlabel("Longitude [°]")
#     plt.ylabel("Latitude [°]")

#     if root_fig is not None:
#         fpath = os.path.join(root_fig, "emission_positions_wgs84.png")
#         plt.savefig(fpath)


def plot_passive_positions(ds_replica, ds_gps, root_fig):
    """
    Plot the interpolated GPS positions of the source along the sequence, as well as the a priori positions of the OBS.
    Parameters
    ----------
    ds_replica : xarray.Dataset
        Dataset containing the interpolated GPS positions of the source along the sequence.
    ds_gps : xarray.Dataset
        Dataset containing the a priori positions of the OBS.
    root_fig : str
        Root directory to save the figure.
    """
    pfig = PubFigure(size=(10, 8), legend_fontsize=16)
    # Plot
    plt.figure()

    # Série de positions successives
    plt.scatter(
        ds_replica.e_replica,
        ds_replica.n_replica,
        marker="+",
        c=np.arange(ds_replica.replica_id.size),
        cmap="jet",
    )
    plt.colorbar(label="Replica ID")

    keys = ["obs1", "obs2", "obs3"]
    label = {
        "obs1": "1S",
        "obs2": "2S",
        "obs3": "3S",
        "t1": "1",
        "t2": "2",
        "t3": "3",
        "t4": "4",
        "t5": "5",
    }
    for ik, k in enumerate(keys):
        e = ds_gps.attrs[f"{k}_e_apriori"]
        n = ds_gps.attrs[f"{k}_n_apriori"]
        plt.scatter(
            e,
            n,
            marker="D",
            label=label[k],
            zorder=10,
            color=color(ik),
            s=40,
        )

    plt.legend()
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    fpath = os.path.join(root_fig, "passive_emission_positions.png")
    plt.savefig(fpath)


def plot_speed_along_seq(df_seq, root_fig=None):
    """
    Plot the speed along the sequence, computed from the interpolated GPS positions.

    Parameters
    ----------
    df_seq : pandas.DataFrame
        DataFrame containing the interpolated GPS positions of the source along the sequence.
    root_fig : str
        Root directory to save the figure.
    """

    ve = df_seq["emission_interp_ve_gps"]
    vn = df_seq["emission_interp_vn_gps"]
    vs = np.vstack((ve, vn))
    vs_norm = np.linalg.norm(vs, axis=0)
    vs_norm_med = np.median(vs_norm)
    print(f"Median source speed along traj : vs = {np.median(vs_norm):.2f} m.s-1")
    print(f"Std source speed along traj : vs = {np.std(vs_norm):.2f} m.s-1")

    plt.figure()
    plt.plot(vs_norm, color=color(0), label=r"$\lVert \vec{v_{ship}} \rVert$")
    plt.axhline(
        vs_norm_med,
        color=color(1),
        linestyle="--",
        label=f"Median speed = {vs_norm_med:.2f} " + r"m.s$^{-1}$",
    )
    plt.xlabel("Replica ID")
    plt.ylabel(r"$\lVert \vec{v_{ship}} \rVert$")
    plt.legend()

    if root_fig is not None:
        fpath = os.path.join(root_fig, "speed_along_seq.png")
        plt.savefig(fpath, dpi=300)


def plot_sequence_spectrogram(
    ds_sig,
    ds_wav,
    nperseg=2**12,
    noverlap=2**11,
    fmin=200,
    fmax=900,
    fig_folder=None,
    fname="spectro_3obs",
):
    """
    Plot the spectrogram of the signal recorded by each OBS during the sequence.
    Parameters
    ----------
    ds_sig : xarray.Dataset
        Dataset containing the signal metadata (start datetime, datetime format, etc.).
    ds_wav : xarray.Dataset
        Dataset containing the raw signal recorded by each OBS.
    fig_folder : str
        Root directory to save the figure.
    """
    datetime_fmt = ds_sig.attrs["datetime_format"]
    t_start = ds_sig.attrs[f"start_datetime"]
    t_start = datetime.strptime(t_start, datetime_fmt)
    t_end = t_start + timedelta(seconds=int(np.ceil(np.max(ds_sig.time.values))))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    axs = np.atleast_1d(axs)

    sxx_plot = []

    for i, obs_id in enumerate([1, 2, 3]):

        datetime_fmt = ds_wav.attrs["datetime_format"]

        sig_varname = f"signal_obs{obs_id}"
        time_coordsname = f"time{obs_id}"
        signal = ds_wav[sig_varname]

        # Select a window of the signal
        fs = ds_wav.attrs[f"fs_obs{obs_id}"]

        # Start of recording
        t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
        t0 = datetime.strptime(t0, datetime_fmt)

        # Select window
        t_from_t0_start_s = (t_start - t0).total_seconds()
        n_start = int(t_from_t0_start_s * fs)
        t_from_t0_end_s = (t_end - t0).total_seconds()
        n_end = int(t_from_t0_end_s * fs)

        # Slice signal
        sig_win = signal.isel({time_coordsname: slice(n_start, n_end)})

        # Define datetime borders
        t0_slice = t0 + timedelta(seconds=n_start * 1 / fs)
        t1_slice = t0 + timedelta(seconds=n_end * 1 / fs)

        # Derive stft
        ff, tt, stft = sp.stft(
            sig_win.values,  # .values -> ici on charge les données en mémoire (un tout petit subset seulement)
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # U^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft))  # dB re 1uPa**2 / Hz ou dB re 1 (m/s)^2 / Hz
        # Associated datetime vector
        tt_datetime = pd.date_range(
            t0_slice,
            t0_slice + timedelta(seconds=tt[-1]),
            freq=f"{tt[1]-tt[0]}s",
            inclusive="both",
        )

        sxx_plot.append(sxx)

    # Plot
    cmap = "magma"
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    for i, obs_id in enumerate([1, 2, 3]):
        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(f"OBS{obs_id}")
        axs[i].set_ylim([fmin, fmax])

    clabel = r"dB re 1$\mu$Pa$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)
    plt.setp(axs[-1].get_xticklabels(), rotation=15, ha="right")

    fig.supylabel("Frequency [Hz]")
    fig.supxlabel("Time (UTC)")
    # fig.suptitle(f"OBS{obs_id}")

    set_subfigures_abc_labels(
        axs=axs,
        fontsize=14,
        x_pos=0.015,
        y_pos=1.02,
        ha="left",
        va="top",
    )

    if fig_folder is not None:
        fpath = os.path.join(fig_folder, f"{fname}.png")
        plt.savefig(fpath, bbox_inches="tight")


def plot_gamma_along_sequence(
    ds,
    dist_rcv,
    obs_cpa_idx,
    reps,
    replica_id_slice=slice(0, 10000),
    fmin=200,
    fmax=900,
    fig_folder=None,
    fname="gamma_along_traj",
):
    """
    Plot the gamma = 20 log10(mod(rtf)) along the sequence, for each OBS. The CPA positions of the OBS are also plotted as horizontal lines.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the RTF estimates and metadata.
    dist_rcv : numpy.ndarray
        Array containing the distance from each replica to each receiver.
    obs_cpa_idx : numpy.ndarray
        Array containing the index of the replica corresponding to the CPA position of each OBS.
    reps : numpy.ndarray
        Array containing the replica IDs.
    fig_folder : str
        Root directory to save the figure.
    """

    # n_rcv = ds.h_index.size
    idx_rcv_plot = [
        idx
        for idx in np.atleast_1d(ds.h_index.values)
        if idx != ds.reference_receiver_id
    ]

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(idx_rcv_plot),
        sharex=True,
        sharey="row",
        figsize=(8 * len(idx_rcv_plot), 10),
    )
    ax_mod = np.atleast_1d(axs)

    rtf_cs_evd_amp = ds.rtf_amp
    # Select frequency range
    rtf_cs_evd_amp = rtf_cs_evd_amp.sel(f_rtf=slice(fmin, fmax))

    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(ds.h_index.values - id_rcv))

        rtf_cs_evd_amp_rcv = rtf_cs_evd_amp.sel(h_index=id_rcv).sel(
            replica_id=replica_id_slice
        )
        gamma = 20 * np.log10(rtf_cs_evd_amp_rcv)
        gamma.plot(
            x="f_rtf",
            cmap="magma",
            ax=ax_mod[i_ax],
            vmin=np.percentile(gamma, 5),
            vmax=np.percentile(gamma, 95),
            # cbar_kwargs={"label": r"$\gamma(f, r)$"},
            # cbar_kwargs={"label": r"$\vert \Pi(f, r) \rvert$ [dB]"},
            cbar_kwargs={"label": r"$\vert \Pi \rvert^2$ [dB]"},
        )

        for i_rcv in range(dist_rcv.shape[0]):
            if (reps[obs_cpa_idx[i_rcv]] <= replica_id_slice.stop) and (
                reps[obs_cpa_idx[i_rcv]] >= replica_id_slice.start
            ):
                ax_mod[i_ax].axhline(
                    reps[obs_cpa_idx[i_rcv]],
                    color=color(2 + i_rcv),
                    label=f"CPA OBS{i_rcv+1}",
                    linestyle="--",
                    linewidth=5,
                    zorder=10,
                )

        ax_mod[i_ax].set_title(f"OBS {id_rcv}")
        ax_mod[i_ax].set_xlabel("")
        ax_mod[i_ax].set_ylabel("")
        ax_mod[i_ax].xaxis.set_major_locator(MaxNLocator(nbins=4))

        i_ax += 1

    set_subfigures_abc_labels(
        axs=axs, fontsize=22, x_pos=0.99, y_pos=1.01, ha="right", va="bottom"
    )
    fig.supxlabel("Frequency [Hz]")
    fig.supylabel("Replica ID")

    if fig_folder is not None:
        fpath = os.path.join(fig_folder, f"{fname}.png")
        plt.savefig(fpath)

    return fig, axs


def plot_rtf_mod_along_sequence(
    ds,
    dist_rcv,
    obs_cpa_idx,
    reps,
    fig_folder,
    replica_id_slice=slice(0, 10000),
    fmin=200,
    fmax=900,
    fname="rtf_along_traj_cs_evd",
):
    """
    Plot the module of the RTF along the sequence, for each OBS. The CPA positions of the OBS are also plotted as horizontal lines.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the RTF estimates and metadata.
    dist_rcv : numpy.ndarray
        Array containing the distance from each replica to each receiver.
    obs_cpa_idx : numpy.ndarray
        Array containing the index of the replica corresponding to the CPA position of each OBS.
    reps : numpy.ndarray
        Array containing the replica IDs.
    fig_folder : str
        Root directory to save the figure.
    """

    n_rcv = ds.h_index.size
    idx_rcv_plot = [idx for idx in ds.h_index.values if idx != ds.reference_receiver_id]

    fig, axs = plt.subplots(
        nrows=1, ncols=len(idx_rcv_plot), sharex=True, sharey="row", figsize=(16, 10)
    )
    ax_mod = axs

    rtf_cs_evd_amp = ds.rtf_amp
    # Select frequency range
    rtf_cs_evd_amp = rtf_cs_evd_amp.sel(f_rtf=slice(fmin, fmax))

    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(ds.h_index.values - id_rcv))

        rtf_cs_evd_amp_rcv = rtf_cs_evd_amp.sel(h_index=id_rcv).sel(
            replica_id=replica_id_slice
        )
        log_mod = 10 * np.log10(rtf_cs_evd_amp_rcv)
        log_mod.plot(
            x="f_rtf",
            cmap="magma",
            ax=ax_mod[i_ax],
            vmin=np.percentile(log_mod, 5),
            vmax=np.percentile(log_mod, 95),
            cbar_kwargs={"label": r"$\lvert \hat{\Pi} \rvert$ [dB]"},
        )

        for i_rcv in range(dist_rcv.shape[0]):
            if (reps[obs_cpa_idx[i_rcv]] <= replica_id_slice.stop) and (
                reps[obs_cpa_idx[i_rcv]] >= replica_id_slice.start
            ):
                ax_mod[i_ax].axhline(
                    reps[obs_cpa_idx[i_rcv]],
                    color=color(2 + i_rcv),
                    label=f"CPA OBS{i_rcv+1}",
                    linestyle="--",
                    zorder=10,
                )

        ax_mod[i_ax].set_title(f"OBS {id_rcv}")
        ax_mod[i_ax].set_xlabel("")
        ax_mod[i_ax].set_ylabel("")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    fig.supxlabel("Frequency [Hz]")
    fig.supylabel("Replica ID")

    fpath = os.path.join(fig_folder, f"{fname}.png")
    plt.savefig(fpath)


def plot_log_mod_distribution_along_sequence(
    log_mod_distribution, log_mod_distribution_distance, bin_centers, replica_ids
):
    """
    Plot the distribution of the log-module of the RTF along the sequence, for each OBS. The distance to a reference distribution is also plotted as a function of the replica ID.

    Parameters
    ----------
    log_mod_distribution : dict
        Dictionary containing the distribution of the log-module of the RTF for each OBS. The keys are the OBS IDs and the values are 2D arrays of shape (n_reps, n_bins).
    log_mod_distribution_distance : dict
        Dictionary containing the distance to a reference distribution for each OBS. The keys are the OBS IDs and the values are 1D arrays of shape (n_reps,).

    """

    idx_rcv_plot = list(log_mod_distribution.keys())
    fig, axs = plt.subplots(
        nrows=2, ncols=len(idx_rcv_plot), sharex=False, sharey="row", figsize=(16, 10)
    )
    ax_distribution, ax_distance = axs

    # bin_centers = log_mod_distribution["bin_centers"]
    # reps = log_mod_distribution["replica_id"]
    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:

        im = ax_distribution[i_ax].pcolormesh(
            bin_centers,
            replica_ids,
            log_mod_distribution[id_rcv],
            shading="auto",
            cmap="jet",
            vmin=0,
            vmax=np.percentile(log_mod_distribution[id_rcv], 90),
        )
        plt.colorbar(im, ax=ax_distribution[i_ax], label=r"$\mu$")

        ax_distribution[i_ax].set_xlabel(r"$\lvert \Pi_2(f, r) \rvert$")
        ax_distribution[i_ax].set_ylabel("")
        ax_distribution[i_ax].set_title(f"OBS {id_rcv}")

        # Plot distances
        idist = 0
        for d_name, d_arr in log_mod_distribution_distance.items():
            ax_distance[i_ax].plot(
                d_arr[id_rcv], replica_ids, label=d_name, color=color(idist)
            )
            idist += 1

        # ax_distance[i_ax].plot(wasserstein_dist_arr, reps, label="W", color=color(0))
        # ax_distance[i_ax].plot(jensenshannon_dist_arr, reps, label="JS", color=color(1))
        ax_distance[i_ax].set_xlabel("Distance")
        ax_distance[i_ax].set_ylabel("")
        ax_distance[i_ax].legend()
        ax_distance[i_ax].set_title(f"OBS {id_rcv}")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    # fig.supxlabel(r"$\lvert \Pi(f, r) \rvert$")
    fig.supylabel("Replica ID")

    # fpath = os.path.join(fig_folder, "rtf_along_traj_cs_evd.png")
    # plt.savefig(fpath)


def get_distance_label(dist_type):
    dist_type_label = {
        "wasserstein": "Wasserstein distance",
        "jensen_shannon": "Jensen-Shannon distance",
        "shannon_entropy": "Shannon entropy",
        "KL_divergence_2D": "Kullback-Leibler divergence (2D)",
    }
    dist_label = dist_type_label.get(dist_type, dist_type)
    return dist_label


def plot_distance_matrix_comparison_subroutine(
    spatial_dist_mat, dist_mat, dist_type, cpa_idx=None, scale="linear"
):

    dist_label = get_distance_label(dist_type)

    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(18, 10), constrained_layout=True
    )
    ax1, ax2 = axs

    # Spatial distance matrix
    vmax = np.nanpercentile(spatial_dist_mat, 75)
    im = ax1.pcolormesh(spatial_dist_mat, cmap="magma", vmax=vmax)
    fig.colorbar(im, ax=ax1, orientation="vertical", label=r"$d$ [m]")
    ax1.set_title(r"$M_{d}$")

    # Data distance matrix
    vmax = np.nanpercentile(dist_mat, 75)
    if scale == "linear":
        im = ax2.pcolormesh(dist_mat, cmap="magma", vmax=vmax)
    elif scale == "log":
        valid = dist_mat[np.isfinite(dist_mat) & (dist_mat > 0)]
        im = ax2.pcolormesh(
            dist_mat,
            cmap="magma",
            # vmax=vmax,
            # norm=colors.LogNorm(
            #     vmin=max(np.nanpercentile(dist_mat, 1), 1e-6), vmax=dist_mat.max()
            # ),
            norm=colors.LogNorm(vmin=valid.min(), vmax=valid.max()),
        )

    fig.colorbar(im, ax=ax2, orientation="vertical", label=dist_label)
    ax2.set_title(r"$M_{S}$")

    # Add CPA indication
    if cpa_idx is not None:
        for ax in axs:
            for i in range(len(cpa_idx)):
                ax.scatter(
                    cpa_idx[i],
                    cpa_idx[i],
                    marker="o",
                    s=150,
                    color=color(i),
                    label=f"OBS{i+1}",
                )

    fig.supxlabel("Replica i")
    fig.supylabel("Replica j")

    return fig, axs


def plot_distance_matrix_comparison(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    cpa_idx=None,
    scale="linear",
):
    # Plot matrix comparison
    fig, axs = plot_distance_matrix_comparison_subroutine(
        spatial_dist_mat=spatial_dist_mat,
        dist_mat=dist_mat,
        dist_type=dist_type,
        cpa_idx=cpa_idx,
        scale=scale,
    )

    # Add legend for CPA position
    ax1, ax2 = axs
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    fpath = os.path.join(
        fig_folder,
        f"distance_matrix_comparison_{dist_type}_{rcv_combinaison_strategy}.png",
    )
    plt.savefig(fpath)


def plot_distance_matrix_comparison_line_selected(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    cpa_idx=None,
    selected_lines=None,
    scale="linear",
):
    # Plot matrix comparison
    fig, axs = plot_distance_matrix_comparison_subroutine(
        spatial_dist_mat=spatial_dist_mat,
        dist_mat=dist_mat,
        dist_type=dist_type,
        cpa_idx=cpa_idx,
        scale=scale,
    )

    if selected_lines is None:
        i_ls = np.random.choice(
            np.arange(spatial_dist_mat.shape[0]), size=3, replace=False
        )
    else:
        i_ls = selected_lines

    for ax in axs:
        for k, i_l in enumerate(i_ls):
            ax.axhline(i_l + 0.5, color=color(k), label=rf"$l_{k}$ (j = {i_l})")

    # Add legend for CPA position
    ax1, ax2 = axs
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    fpath = os.path.join(
        fig_folder,
        f"distance_matrix_comparison_{dist_type}_{rcv_combinaison_strategy}_selected_lines.png",
    )
    plt.savefig(fpath)


def plot_distance_selected_lines(
    spatial_dist_mat,
    dist_mat,
    fig_folder,
    dist_type="shannon_entropy",
    rcv_combinaison_strategy="sum",
    selected_lines=None,
    scale="linear",
):

    dist_label = get_distance_label(dist_type)

    if selected_lines is None:
        i_ls = np.random.choice(
            np.arange(spatial_dist_mat.shape[0]), size=3, replace=False
        )
    else:
        i_ls = selected_lines

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))
    ax1, ax2 = axs

    for k, i_l in enumerate(i_ls):
        ax1.plot(
            spatial_dist_mat[i_l, :],
            marker="+",
            color=color(k),
            label=rf"$l_{k}$ (j = {i_l})",
        )
        ax2.plot(
            dist_mat[i_l, :], marker="+", color=color(k), label=rf"$l_{k}$ (j = {i_l})"
        )

    ax2.set_yscale(scale)

    ax1.set_ylabel(r"$d$ [m]")
    ax2.set_ylabel(dist_label)
    ax1.legend()
    ax2.legend()
    fig.supxlabel("Replica ID")

    fpath = os.path.join(
        fig_folder,
        f"distance_selected_lines_{dist_type}_{rcv_combinaison_strategy}.png",
    )
    plt.savefig(fpath)


def get_combine_distance(distance_dict, combine_method="product"):

    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_replicas = len(distance_dict[d_name_0][idx_rcv_plot[0]])
    if combine_method == "product":
        combined_distance_dict = {
            d_name: np.ones(n_replicas) for d_name in distance_dict.keys()
        }
    elif combine_method == "sum":
        combined_distance_dict = {
            d_name: np.zeros(n_replicas) for d_name in distance_dict.keys()
        }

    for id_rcv in idx_rcv_plot:
        for d_name in distance_dict.keys():
            if combine_method == "product":
                combined_distance_dict[d_name] *= distance_dict[d_name][id_rcv]
            elif combine_method == "sum":
                combined_distance_dict[d_name] += distance_dict[d_name][id_rcv]

    for d_name in combined_distance_dict.keys():
        if combine_method == "product":
            # Smallest distance corresponds to all distance being small, which means that the product of 1/d will be large, and thus 1/product will be small
            # combined_distance_dict[d_name] =
            pass
        elif combine_method == "sum":
            combined_distance_dict[d_name] *= 1 / len(
                idx_rcv_plot
            )  # Average over receivers

        # Normalize to fall into [0, 1] for comparison purpose
        combined_distance_dict[d_name] = normalize(arr=combined_distance_dict[d_name])

    return combined_distance_dict


def plot_dist_and_combined_dist(
    distance_dict, product_distance_dict, sum_distance_dict, replica_ids
):

    def plot_distance(ax, reps, distance, label, color, marker=None):
        ax.plot(reps, distance, label=label, color=color, marker=marker)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend()

    # Setup
    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_axes = len(idx_rcv_plot) + 2

    fig, axs = plt.subplots(
        nrows=1,
        ncols=n_axes,
        sharex=False,
        sharey="row",
        figsize=(16, 10),
    )

    i_ax = 0
    # --- Individual OBS plots ---
    for id_rcv in idx_rcv_plot:

        for i_d, d_name in enumerate(distance_dict.keys()):
            plot_distance(
                axs[i_ax],
                replica_ids,
                distance_dict[d_name][id_rcv],
                d_name,
                color(i_d),
                marker="+",
            )
        axs[i_ax].set_title(f"OBS {id_rcv}")

        i_ax += 1

    # Sum distances
    for i_d, d_name in enumerate(sum_distance_dict.keys()):
        plot_distance(
            axs[i_ax],
            replica_ids,
            sum_distance_dict[d_name],
            d_name,
            color(i_d),
            marker="+",
        )
    axs[i_ax].set_title(" + ".join([f"OBS {id_rcv}" for id_rcv in idx_rcv_plot]))

    i_ax += 1
    # Product distances
    for i_d, d_name in enumerate(product_distance_dict.keys()):
        plot_distance(
            axs[i_ax],
            replica_ids,
            product_distance_dict[d_name],
            d_name,
            color(i_d),
            marker="+",
        )
    axs[i_ax].set_title(" x ".join([f"OBS {id_rcv}" for id_rcv in idx_rcv_plot]))

    # --- Global formatting ---
    set_subfigures_abc_labels(
        axs=axs, fontsize=14, x_pos=0.015, y_pos=0.99, ha="left", va="top"
    )

    fig.supylabel("Distance")
    fig.supxlabel("Replica ID")
    plt.yscale("log")


# ======================================================================================================================
# Miscellaneous routines
# ======================================================================================================================
def get_dist_to_rcv(ds):
    e = ds["e_replica"].values
    n = ds["n_replica"].values

    rep_pos = np.column_stack((e, n))
    dist_to_rcv = []
    for i_rcv in ds.h_index.values:
        e_rcv = ds.attrs[f"obs{i_rcv}_e_apriori"]
        n_rcv = ds.attrs[f"obs{i_rcv}_n_apriori"]
        rcv_pos = np.column_stack((e_rcv, n_rcv))

        spatial_dist = cdist(rep_pos, rcv_pos, metric="euclidean")
        dist_to_rcv.append(spatial_dist)

    return np.array(dist_to_rcv)


def get_distribution_arr(rtf_module, n_bins=50, kde=False, dist_min_max=None):
    log_mod = 10 * np.log10(rtf_module)

    # Derive histogram
    # Define a common range of values for the histograms
    if dist_min_max is not None:
        log_mod_min, log_mod_max = dist_min_max
    else:
        log_mod_min = np.percentile(log_mod.values, 0.5)
        log_mod_max = np.percentile(log_mod.values, 99.5)

    reps = rtf_module.replica_id.values

    log_mod_hist_arr = np.zeros((len(reps), n_bins))
    for irep, rep in enumerate(reps):
        log_mod_r = log_mod.sel(replica_id=rep)
        log_mod_r_hist, bin_edges = np.histogram(
            log_mod_r.values,
            bins=n_bins,
            range=(log_mod_min, log_mod_max),
            density=True,
        )

        log_mod_hist_arr[irep, :] = log_mod_r_hist

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if kde:

        log_mod_kde_arr = np.zeros((len(reps), n_bins))

        for irep, rep in enumerate(reps):
            log_mod_r = log_mod.sel(replica_id=rep)
            # Ensure a common grid of values for the KDE
            log_mod_r_sorted = np.sort(log_mod_r.values)
            log_mod_r_idx_min = np.where(log_mod_r_sorted >= log_mod_min)[0][0]
            log_mod_r_idx_max = np.where(log_mod_r_sorted <= log_mod_max)[0][-1]
            log_mod_r = log_mod_r_sorted[log_mod_r_idx_min : log_mod_r_idx_max + 1]

            # Apply Gaussian KDE
            kde = gaussian_kde(log_mod_r)
            log_mod_kde_arr[irep, :] = kde.pdf(bin_centers)
    else:
        log_mod_kde_arr = None

    return log_mod_hist_arr, bin_centers, log_mod_kde_arr


def get_distribution_distance(
    log_mod_distribution, idx_rep_ref, distance=["shannon_entropy"]
):

    # Reference distribution to compare with
    mu_ref = log_mod_distribution[idx_rep_ref, :]

    nreps = log_mod_distribution.shape[0]
    distance_dict = {d_name: np.zeros(nreps) for d_name in distance}

    for irep in range(nreps):
        mu_r = log_mod_distribution[irep, :]

        if "wasserstein" in distance:
            distribution_support = np.arange(mu_ref.size)
            wd = wasserstein_distance(
                u_values=distribution_support,
                v_values=distribution_support,
                u_weights=mu_ref,
                v_weights=mu_r,
            )
            # wd = wasserstein_distance(u_values=mu_ref, v_values=mu_r)
            distance_dict["wasserstein"][irep] = wd

        if "jensen_shannon" in distance:
            js = jensenshannon(mu_ref, mu_r)
            distance_dict["jensen_shannon"][irep] = js

        if "shannon_entropy" in distance:
            mu_ref_s = mu_ref.copy()
            mu_ref_s[mu_ref_s == 0] = np.nan
            mu_r_s = mu_r.copy()
            mu_r_s[mu_r_s == 0] = np.nan
            s = entropy(mu_ref_s, mu_r_s, nan_policy="omit")
            distance_dict["shannon_entropy"][irep] = s

        # kld = kl_div(mu_ref, mu_r)

    return distance_dict


def get_distribution_distance_all_rcv(
    log_mod_distribution, idx_rep_ref, distances=["jensen_shannon"]
):

    distance_dict = {d_name: {} for d_name in distances}

    for id_rcv, log_mod_distribution_rcv in log_mod_distribution.items():
        distance_dict_id_rcv = get_distribution_distance(
            log_mod_distribution=log_mod_distribution_rcv,
            idx_rep_ref=idx_rep_ref,
            distance=distances,
        )
        for d_name in distances:
            d = distance_dict_id_rcv[d_name]
            d_norm = normalize(arr=d)
            # d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d))
            distance_dict[d_name][id_rcv] = d_norm

    return distance_dict


def get_log_module_distributions(ds, fmin, fmax, n_bins=50, dist_min_max=None):

    idx_rcv_plot = [idx for idx in ds.h_index.values if idx != ds.reference_receiver_id]

    log_mod_hist = {}
    log_mod_kde = {}
    for id_rcv in idx_rcv_plot:
        rtf_cs_evd_rcv = ds.rtf_amp.sel(h_index=id_rcv).sel(f_rtf=slice(fmin, fmax))

        log_mod_hist_arr, bin_centers, log_mod_kde_arr = get_distribution_arr(
            rtf_module=rtf_cs_evd_rcv,
            n_bins=n_bins,
            kde=True,
            dist_min_max=dist_min_max,
        )

        log_mod_hist[id_rcv] = log_mod_hist_arr
        log_mod_kde[id_rcv] = log_mod_kde_arr

    return (log_mod_hist, log_mod_kde, bin_centers)


def get_bootstrap_dist_matrix(
    ds,
    distances=["wasserstein", "jensen_shannon", "shannon_entropy"],
    input="histogram",
    fmin=400,
    fmax=800,
    n_bins=100,
    dist_min_max=None,
):

    log_mod_hist, log_mod_kde, bin_centers = get_log_module_distributions(
        ds, fmin=fmin, fmax=fmax, n_bins=n_bins, dist_min_max=dist_min_max
    )

    # Derive distributions distance matrix
    dist_mat_prod = {d_name: [] for d_name in distances}
    dist_mat_sum = {d_name: [] for d_name in distances}
    for i, id in enumerate(ds.replica_id.values):
        # Histogram
        if input == "histogram":
            distance_dict_hist = get_distribution_distance_all_rcv(
                log_mod_distribution=log_mod_hist,
                idx_rep_ref=id,
                distances=distances,
            )
            product_distance_dict = get_combine_distance(
                distance_dict=distance_dict_hist, combine_method="product"
            )
            sum_distance_dict = get_combine_distance(
                distance_dict=distance_dict_hist, combine_method="sum"
            )

        # KDE
        elif input == "kde":
            distance_dict_kde = get_distribution_distance_all_rcv(
                log_mod_distribution=log_mod_kde,
                idx_rep_ref=id,
                distances=distances,
            )
            product_distance_dict = get_combine_distance(
                distance_dict=distance_dict_kde, combine_method="product"
            )
            sum_distance_dict = get_combine_distance(
                distance_dict=distance_dict_kde, combine_method="sum"
            )

        else:
            raise ValueError(f"Unknown input {input}, must be 'histogram' or 'kde'.")

        for d_name in distances:
            dist_mat_prod[d_name].append(product_distance_dict[d_name])
            dist_mat_sum[d_name].append(sum_distance_dict[d_name])

    # Convert to np array
    for d_name in distances:
        dist_mat_prod[d_name] = np.array(dist_mat_prod[d_name])
        dist_mat_sum[d_name] = np.array(dist_mat_sum[d_name])

    # Normalize distance matrix to fall into [0, 1] for comparison purpose
    for d_name in distances:
        dist_mat_prod[d_name] = normalize(arr=dist_mat_prod[d_name])
        dist_mat_sum[d_name] = normalize(arr=dist_mat_sum[d_name])

    return dist_mat_prod, dist_mat_sum


def build_dist_matrix(ds):
    e = ds["e_replica"].values
    n = ds["n_replica"].values

    rep_pos = np.column_stack((e, n))
    spatial_dist = cdist(rep_pos, rep_pos, metric="euclidean")

    return spatial_dist


def normalize(arr):
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))


def get_combine_distance(distance_dict, combine_method="product"):

    d_name_0 = list(distance_dict.keys())[0]
    idx_rcv_plot = list(distance_dict[d_name_0].keys())
    n_replicas = len(distance_dict[d_name_0][idx_rcv_plot[0]])
    if combine_method == "product":
        combined_distance_dict = {
            d_name: np.ones(n_replicas) for d_name in distance_dict.keys()
        }
    elif combine_method == "sum":
        combined_distance_dict = {
            d_name: np.zeros(n_replicas) for d_name in distance_dict.keys()
        }

    for id_rcv in idx_rcv_plot:
        for d_name in distance_dict.keys():
            if combine_method == "product":
                combined_distance_dict[d_name] *= distance_dict[d_name][id_rcv]
            elif combine_method == "sum":
                combined_distance_dict[d_name] += distance_dict[d_name][id_rcv]

    for d_name in combined_distance_dict.keys():
        if combine_method == "product":
            # Smallest distance corresponds to all distance being small, which means that the product of 1/d will be large, and thus 1/product will be small
            # combined_distance_dict[d_name] =
            pass
        elif combine_method == "sum":
            combined_distance_dict[d_name] *= 1 / len(
                idx_rcv_plot
            )  # Average over receivers

        # Normalize to fall into [0, 1] for comparison purpose
        combined_distance_dict[d_name] = normalize(arr=combined_distance_dict[d_name])

    return combined_distance_dict


#
# CS-EVD perf


def interp_deconvolution(ds):
    f_interp = ds.f_rtf.values
    freq_deconv = ds.f_deconv.values

    # Interpolate rtf_deconv at rtf frequencies
    # Interp module
    interpolator_module = interp1d(
        freq_deconv, ds.rtf_amp_deconv.values, axis=1, fill_value="extrapolate"
    )
    rtf_deconvolution_module_interp = interpolator_module(f_interp)

    # Interp unwrapped phase
    unwrapped_phase_deconv = np.unwrap(ds.rtf_phase_deconv.values, axis=1)
    interpolator_phase = interp1d(
        freq_deconv, unwrapped_phase_deconv, axis=1, fill_value="extrapolate"
    )
    rtf_deconvolution_phase_unwrapped_interp = interpolator_phase(f_interp)
    # Wrap the interpolated unwrapped phase back to [-pi, pi]
    rtf_deconvolution_phase_interp = (
        rtf_deconvolution_phase_unwrapped_interp + np.pi
    ) % (2 * np.pi) - np.pi

    # Build interpolated complex RTF
    rtf_deconvolution_interp = rtf_deconvolution_module_interp * np.exp(
        1j * rtf_deconvolution_phase_interp
    )

    return f_interp, rtf_deconvolution_interp


def plot_features_ipulse(
    rtf_cs_evd,
    rtf_deconvolution_interp,
    freq,
    rcv_ref_id=2,
    i_pulse=0,
    rtf_deconvolution=None,
    freq_deconv=None,
    save=False,
    root_img=None,
):

    idx_rcv_plot = [idx for idx in rtf_cs_evd.h_index.values if idx != rcv_ref_id]

    fig, ax_mod = plt.subplots(
        nrows=len(idx_rcv_plot), ncols=1, sharex=True, sharey="row", figsize=(16, 10)
    )

    # Module
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(rtf_cs_evd.h_index.values - id_rcv))

        h_plot = i_rcv + 1

        ax_mod[i_ax].plot(
            freq,
            np.abs(rtf_cs_evd[i_rcv, :, i_pulse]),
            label="CS-EVD",
            color=color(1),
            zorder=2,
        )

        if rtf_deconvolution is not None and freq_deconv is not None:
            deconv_interp_label = "Deconvolution (linear interpolation)"
            ax_mod[i_ax].plot(
                freq_deconv,
                np.abs(rtf_deconvolution[i_rcv, :, i_pulse]),
                label="Deconvolution",
                color=color(2),
                linestyle="--",
                zorder=3,
            )
        else:
            deconv_interp_label = "Deconvolution"

        ax_mod[i_ax].plot(
            freq,
            np.abs(rtf_deconvolution_interp[i_rcv, :, i_pulse]),
            label=deconv_interp_label,
            color=color(0),
            linestyle="-",
            zorder=1,
        )

        ax_mod[i_ax].set_title(f"OBS {h_plot}")
        ax_mod[i_ax].set_ylim([0.001, 1000])
        ax_mod[i_ax].set_yscale("log")
        ax_mod[i_ax].set_xlabel("")
        ax_mod[i_ax].set_ylabel(r"$|\Pi|$")
        ax_mod[i_ax].legend(fontsize=10, ncol=3, loc="lower right")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=ax_mod, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    fig.supxlabel("Frequency [Hz]")

    if save and root_img is not None:
        fpath = os.path.join(
            root_img, f"rtf_deconv_vs_cs_evd_module_replica_{i_pulse}.png"
        )
        plt.savefig(fpath)

    # Wrapped phase
    fig, ax_phase_wrapped = plt.subplots(
        nrows=len(idx_rcv_plot), ncols=1, sharex=True, sharey="row", figsize=(16, 10)
    )
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(rtf_cs_evd.h_index.values - id_rcv))

        h_plot = i_rcv + 1

        ax_phase_wrapped[i_ax].plot(
            freq,
            np.angle(rtf_cs_evd[i_rcv, :, i_pulse]),
            label="CS-EVD",
            color=color(1),
            zorder=2,
        )

        if rtf_deconvolution is not None and freq_deconv is not None:
            deconv_interp_label = "Deconvolution (linear interpolation)"

            ax_phase_wrapped[i_ax].plot(
                freq_deconv,
                np.angle(rtf_deconvolution[i_rcv, :, i_pulse]),
                label="Deconvolution",
                color=color(2),
                linestyle="--",
                zorder=3,
            )
        else:
            deconv_interp_label = "Deconvolution"

        ax_phase_wrapped[i_ax].plot(
            freq,
            np.angle(rtf_deconvolution_interp[i_rcv, :, i_pulse]),
            label=deconv_interp_label,
            color=color(0),
            linestyle="-",
            zorder=1,
        )

        ax_phase_wrapped[i_ax].set_title(f"OBS {h_plot}")
        ax_phase_wrapped[i_ax].set_xlabel("")
        ax_phase_wrapped[i_ax].set_ylabel(r"Phase [rad]")
        ax_phase_wrapped[i_ax].legend(fontsize=10, ncol=3, loc="lower right")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=ax_phase_wrapped, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    fig.supxlabel("Frequency [Hz]")

    if save and root_img is not None:
        fpath = os.path.join(
            root_img, f"rtf_deconv_vs_cs_evd_phase_replica_{i_pulse}.png"
        )
        plt.savefig(fpath)

    # Unwrapped phase
    fig, ax_phase = plt.subplots(
        nrows=len(idx_rcv_plot), ncols=1, sharex=True, sharey="row", figsize=(16, 10)
    )
    i_ax = 0
    for id_rcv in idx_rcv_plot:
        i_rcv = np.argmin(np.abs(rtf_cs_evd.h_index.values - id_rcv))
        h_plot = i_rcv + 1

        ax_phase[i_ax].plot(
            freq,
            np.unwrap(np.angle(rtf_cs_evd[i_rcv, :, i_pulse])),
            label="CS-EVD",
            color=color(0),
        )

        if rtf_deconvolution is not None and freq_deconv is not None:
            deconv_interp_label = "Deconvolution (linear interpolation)"

            ax_phase[i_ax].plot(
                freq_deconv,
                np.unwrap(np.angle(rtf_deconvolution[i_rcv, :, i_pulse])),
                label="Deconvolution",
                color=color(2),
                linestyle="--",
            )
        else:
            deconv_interp_label = "Deconvolution"

        ax_phase[i_ax].plot(
            freq,
            np.unwrap(np.angle(rtf_deconvolution_interp[i_rcv, :, i_pulse])),
            label=deconv_interp_label,
            color=color(1),
            linestyle="-",
        )

        ax_phase[i_ax].set_title(f"OBS {h_plot}")
        ax_phase[i_ax].set_xlabel("")
        ax_phase[i_ax].set_ylabel(r"Phase [rad]")
        ax_phase[i_ax].legend(fontsize=10, ncol=3, loc="lower right")

        i_ax += 1

    set_subfigures_abc_labels(
        axs=ax_phase, fontsize=14, x_pos=0.015, y_pos=0.98, ha="left", va="top"
    )
    fig.supxlabel("Frequency [Hz]")

    if save and root_img is not None:
        fpath = os.path.join(
            root_img, f"rtf_deconv_vs_cs_evd_unwrapped_phase_replica_{i_pulse}.png"
        )
        plt.savefig(fpath)


def process_all_pulses(
    freq,
    rtf_cs_evd,
    rtf_deconvolution_interp,
    plot_feature=False,
    use_deconvolution_phase=False,
):

    dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": False, "apply_median": True}
    mod_dist_kwargs = {
        "ax_rcv": 0,
        "ax_f": 1,
        "apply_mean": False,
        "apply_median": True,
        "data_space": "real",
    }

    theta_dist = []
    theta_mod_dist = []
    theta_along_f_dist = []
    d_L1s = []
    d_L1_mod = []

    d_euc = []
    dist_dtw = []

    for i_pulse in range(rtf_cs_evd.shape[-1]):

        if plot_feature:
            plot_features_ipulse(
                rtf_cs_evd=rtf_cs_evd,
                rtf_deconvolution_interp=rtf_deconvolution_interp,
                freq=freq,
                i_pulse=i_pulse,
            )

        # # Compute theta distance
        rtf = np.abs(rtf_cs_evd[..., i_pulse])
        rtf_ref = np.abs(rtf_deconvolution_interp[..., i_pulse])
        theta_mod = D_hermitian_angle_fast(
            rtf_ref=rtf_ref,
            rtf=rtf,
            **mod_dist_kwargs,
        )
        # theta_mod = np.percentile(theta_mod, 50)
        theta_mod_dist.append(theta_mod)

        dist = D_hermitian_angle_fast(
            rtf_ref=rtf_deconvolution_interp[..., i_pulse],
            rtf=rtf_cs_evd[..., i_pulse],
            **dist_kwargs,
        )
        # dist = np.percentile(dist, 50)

        # Compute theta along f for each rcv
        dist_f = []
        for i_rcv in range(rtf_cs_evd.shape[0]):

            # rtf_ref = rtf_deconvolution_interp[i_rcv, :, i_pulse]
            # rtf = rtf_cs_evd[i_rcv, :, i_pulse]
            rtf_ref = np.abs(rtf_deconvolution_interp[i_rcv, :, i_pulse])
            rtf = np.abs(rtf_cs_evd[i_rcv, :, i_pulse])

            inner_prod = np.sum(rtf_ref.conj() * rtf)
            norm_ref = np.linalg.norm(rtf_ref)
            norm_rtf = np.linalg.norm(rtf)

            # Cosine of Hermitian angle, clipped for stability
            cos_angle = np.clip(inner_prod / (norm_ref * norm_rtf), -1.0, 1.0)
            d_f = np.arccos(cos_angle)
            d_f = np.rad2deg(d_f)
            # print(d_f)
            dist_f.append(d_f)

        dist_f = np.mean(dist_f)

        # Compute element wise L1 distance
        rtf_ref = rtf_deconvolution_interp[..., i_pulse]
        rtf = rtf_cs_evd[..., i_pulse]
        d_L1 = np.sum(
            np.abs(rtf_ref - rtf) / np.abs(rtf_ref), axis=0
        )  # Axis 0 = rcv axis
        # d_L1 = d_L1 / np.sum(np.abs(rtf_ref), axis=0) # Normalization
        d_L1 = np.median(d_L1, axis=0).squeeze()  # Median along f axis
        d_L1s.append(d_L1)

        # Compute element wise L1 distance on module
        rtf_ref = np.abs(rtf_deconvolution_interp[..., i_pulse])
        rtf = np.abs(rtf_cs_evd[..., i_pulse])
        d_L1_m = np.sum(
            np.abs(rtf_ref - rtf) / np.abs(rtf_ref), axis=0
        )  # Axis 0 = rcv axis
        # d_L1_m = d_L1_m / np.sum(np.abs(rtf_ref), axis=0) # Normalization
        d_L1_m = np.median(d_L1_m, axis=0).squeeze()  # Median along f axis
        d_L1_mod.append(d_L1_m)

        # Absolute diff in log (correspond to what we usually visualize)
        # d_mod = np.mean(np.linalg.norm(
        #     10 * np.log10(np.abs(rtf_deconvolution_interp[..., i_pulse])) - 10 * np.log10(np.abs(rtf_cs_evd[..., i_pulse])), axis=1
        # ))

        # d1_along_f = np.linalg.norm(
        #     np.abs(rtf_deconvolution_interp[..., i_pulse])
        #     - np.abs(rtf_cs_evd[..., i_pulse]),
        #     axis=1,
        # )
        # pi_norm_along_f = np.linalg.norm(rtf_cs_evd[..., i_pulse], axis=1)
        # d_mod = np.mean(d1_along_f / pi_norm_along_f)

        d = np.mean(
            np.linalg.norm(
                rtf_deconvolution_interp[..., i_pulse] - rtf_cs_evd[..., i_pulse],
                axis=1,
            )
        )

        # # Dynamic Time Warping
        # d_dtww = []
        # for id_rcv in [idx for idx in ds.h_index.values if idx != ds.reference_receiver_id]:
        #     i_rcv = np.argmin(np.abs(ds.h_index.values - id_rcv))

        #     d_dtw = dtw.distance(
        #         np.abs(rtf_deconvolution_interp[i_rcv, :, i_pulse]),
        #         np.abs(rtf_cs_evd[i_rcv, :, i_pulse]),
        #     )
        #     d_dtww.append(d_dtw)
        # d_dtw = np.mean(d_dtww)
        # print(d_dtw)

        if use_deconvolution_phase:
            dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
            rtf_test = np.abs(rtf_cs_evd) * np.exp(
                1j * np.angle(rtf_deconvolution_interp)
            )
            dist = D_hermitian_angle_fast(
                rtf_ref=rtf_deconvolution_interp[..., i_pulse],
                rtf=rtf_test[..., i_pulse],
                **dist_kwargs,
            )

        theta_dist.append(dist)
        theta_along_f_dist.append(dist_f)
        # d_euc.append(d)
        # dist_dtw.append(d_dtw)

    return (
        np.array(theta_dist),
        np.array(theta_mod_dist),
        np.array(theta_along_f_dist),
        np.array(d_L1s),
        np.array(d_L1_mod),
        np.array(dist_dtw),
    )


def test_single_cs_evd_param_set(
    rtf_mfp_processor,
    active_replicas_args,
    passive_replicas_args,
    nperseg,
    noverlap,
    id_library,
    verbose=True,
):

    if verbose:
        print(
            f"Initial params : \n\t nperseg = {rtf_mfp_processor.fsm_passive.cm.nperseg} \n\t noverlap = {rtf_mfp_processor.fsm_passive.cm.noverlap}"
        )

    # Update processor params
    rtf_mfp_processor.fsm_nperseg = nperseg
    rtf_mfp_processor.fsm_noverlap = noverlap

    # Update Active manager params
    rtf_mfp_processor.fsm_active.nperseg = rtf_mfp_processor.fsm_nperseg
    rtf_mfp_processor.fsm_active.noverlap = rtf_mfp_processor.fsm_noverlap

    # Update passive manager params
    rtf_mfp_processor.fsm_passive.nperseg = rtf_mfp_processor.fsm_nperseg
    rtf_mfp_processor.fsm_passive.noverlap = rtf_mfp_processor.fsm_noverlap

    # Run
    rtf_mfp_processor.compute_library(
        active_feature_args=active_replicas_args,
        passive_feature_args=passive_replicas_args,
        id=id_library,
    )

    if verbose:
        print(
            f"Updated params : \n\t nperseg = {rtf_mfp_processor.fsm_active.cm.nperseg} \n\t noverlap = {rtf_mfp_processor.fsm_active.cm.noverlap}"
        )


def run_sensibility_cs_evd(
    rtf_mfp_processor,
    active_replicas_args,
    passive_replicas_args,
    pulse_window_length_s,
    fs,
):
    # Define the range of the params to study
    ns = int(pulse_window_length_s * fs)  # Number of sample in each pulse window
    # Arbitrary but we now that stft snapshots should be longer than impulse response to ensure the multiplicative
    # transfert function assumption holds
    tau_ir = 0.2
    ns_tau_ir = int(tau_ir * fs)  # It should be larger than tau_ir
    n_stft_pow2_min = int(np.log2(ns_tau_ir))
    # n_stft_pow2_min = 6

    # STFT snapshot length can not exceed the total siganl snapshot duration
    n_stft_pow2_max = int(np.log2(ns))
    # n_stft_pow2_max = 7
    # Test all snapshot length between min and max
    n_stft_pow2 = np.arange(n_stft_pow2_min, n_stft_pow2_max + 1)
    # Overlap factor to test (usual sample cov matrix estimation assumes alpha_ov=0 for independence of segments)
    ov_factors = np.array([0, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    # ov_factors = np.arange(0.05, 0.95 + 0.05, step=0.05)
    # ov_factors = np.array([0.25])

    print(f"n_perseg_min = {n_stft_pow2_min}, n_perseg_max = {n_stft_pow2_max}")
    print(f"ov_factor = {ov_factors}")

    id_library_initial = 501
    i_test = 0
    prev_progress = 0
    n_test = n_stft_pow2.size * ov_factors.size

    ids = []
    params = []
    # Iterate over n_stft values
    for n_stft_p2 in n_stft_pow2:
        # Iterate over overlapping factor
        for ov_factor in ov_factors:

            prev_progress = progression_bar(
                index=i_test,
                index0=0,
                indexf=n_test - 1,
                prev_progress=prev_progress,
            )

            nperseg = 2**n_stft_p2
            noverlap = int(nperseg * ov_factor)

            id_library = id_library_initial + i_test

            test_single_cs_evd_param_set(
                rtf_mfp_processor,
                active_replicas_args=active_replicas_args,
                passive_replicas_args=passive_replicas_args,
                nperseg=nperseg,
                noverlap=noverlap,
                id_library=id_library,
            )

            ids.append(id_library)
            params.append((n_stft_p2, ov_factor))

            i_test += 1

    return ids, params


def sensibility_cs_evd_analysis(library_ids, fmin=400, fmax=800):
    # Load data
    root_lib = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\library"

    thetas = []
    theta_mods = []
    d_L1s = []
    d_L1_mods = []
    for id in library_ids:
        fpath = os.path.join(root_lib, f"library_{id}.nc")
        ds = xr.open_dataset(fpath)

        # Select frequency range
        ds = ds.sel(f_rtf=slice(fmin, fmax), f_deconv=slice(fmin, fmax))

        # Build complex RTFs
        rtf_cs_evd = ds.rtf_amp * np.exp(1j * ds.rtf_phase)
        f, rtf_deconvolution_interp = interp_deconvolution(ds)

        # Compute theta for the sequence
        theta_dist, theta_mod_dist, theta_along_f_dist, d_L1, d_L1_mod, dist_dtw = (
            process_all_pulses(
                freq=f,
                rtf_cs_evd=rtf_cs_evd.values,
                rtf_deconvolution_interp=rtf_deconvolution_interp,
                plot_feature=False,
                use_deconvolution_phase=False,
            )
        )

        thetas.append(theta_dist)
        theta_mods.append(theta_mod_dist)
        d_L1s.append(d_L1)
        d_L1_mods.append(d_L1_mod)

    return (
        np.array(thetas),
        np.array(theta_mods),
        np.array(d_L1s),
        np.array(d_L1_mods),
    )


if __name__ == "__main__":
    pass
