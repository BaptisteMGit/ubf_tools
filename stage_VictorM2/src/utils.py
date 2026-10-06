#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/07/02 11:54:13
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle

# ======================================================================================================================
# Plotting functions
# ======================================================================================================================


def plot_network_and_bathy(
    ds_bathy,
    ds_network_9R,
):
    # Extract data
    z = ds_bathy["elevation"].values
    lon = ds_bathy["lon"].values
    lat = ds_bathy["lat"].values

    # Hillshade
    ls = LightSource(
        azdeg=315, altdeg=45
    )  # illumination azimuth  # illumination elevation

    # Bathymetry is negative underwater.
    # Invert sign so hillshading behaves naturally.
    rgb = ls.shade(-z, cmap=plt.cm.Blues_r, vert_exag=5, blend_mode="overlay")

    # Figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Shaded bathymetry
    ax.imshow(
        rgb,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
    )

    # Optional depth contours
    levels = np.arange(np.floor(z.min() / 100) * 100, 0, 1000)

    cs = ax.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.4,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )

    ax.clabel(cs, fmt="%d m", fontsize=7)

    # Coastlines
    ax.coastlines(resolution="10m", linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # Grid
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("Shaded Bathymetry")

    rx_lon = ds_network_9R["longitude"].values
    rx_lat = ds_network_9R["latitude"].values
    rx_id = ds_network_9R["receiver_id"].values

    ax.scatter(
        rx_lon,
        rx_lat,
        s=60,
        c="red",
        edgecolors="white",
        linewidths=1,
        marker="^",
        transform=ccrs.PlateCarree(),
        zorder=10,
        label="Receivers",
    )

    for lon, lat, rid in zip(rx_lon, rx_lat, rx_id):
        ax.annotate(
            str(rid),
            xy=(lon, lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            zorder=11,
        )

    return fig, ax


def plot_ais_traj_over_bathy(ds_bathy, ds_network_9R, ds_ais):

    fig, ax = plot_network_and_bathy(ds_bathy, ds_network_9R)

    for mmsi in ds_ais.mmsi.values:

        lon_track = ds_ais.lon.sel(mmsi=mmsi).values
        lat_track = ds_ais.lat.sel(mmsi=mmsi).values

        mask = np.isfinite(lon_track) & np.isfinite(lat_track)

        ax.plot(
            lon_track[mask],
            lat_track[mask],
            color="red",
            linewidth=1,
            alpha=0.3,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

    # plt.tight_layout()
    # plt.show()


def get_ais_density(ds_ais):
    lon_all = ds_ais.lon.values.ravel()
    lat_all = ds_ais.lat.values.ravel()

    mask = np.isfinite(lon_all) & np.isfinite(lat_all)

    lon_all = lon_all[mask]
    lat_all = lat_all[mask]

    H, xedges, yedges = np.histogram2d(lon_all, lat_all, bins=350)

    H[H <= np.nanpercentile(H, 30)] = np.nan

    return xedges, yedges, H


def plot_ais_density_over_bathy(ds_bathy, ds_network_9R, ds_ais):

    # Extract data
    z = ds_bathy["elevation"].values
    lon = ds_bathy["lon"].values
    lat = ds_bathy["lat"].values

    # Receiver dataset
    rx_lon = ds_network_9R["longitude"].values
    rx_lat = ds_network_9R["latitude"].values
    rx_id = ds_network_9R["receiver_id"].values

    xedges, yedges, H = get_ais_density(ds_ais)

    # =============================================================================
    # DEFINE ZOOM AREA FROM RECEIVER NETWORK
    # =============================================================================

    pad = 0.05  # degrees

    zoom_extent = [
        rx_lon.min() - pad,
        rx_lon.max() + pad,
        rx_lat.min() - pad,
        rx_lat.max() + pad,
    ]

    # =============================================================================
    # HILLSHADE
    # =============================================================================

    ls = LightSource(
        azdeg=315,
        altdeg=45,
    )

    # Hillshade computed on bathymetry
    rgb = ls.shade(
        -z,  # invert bathymetry for proper relief
        cmap=plt.cm.Blues,
        vert_exag=8,
        blend_mode="overlay",
    )
    # import cmocean

    # rgb = ls.shade(-z, cmap=cmocean.cm.deep, vert_exag=8, blend_mode="soft")
    # rgb = ls.shade(-z, cmap=cmocean.cm.tempo, vert_exag=8, blend_mode="overlay")

    # =============================================================================
    # FIGURE
    # =============================================================================

    fig = plt.figure(figsize=(14, 10))

    # -----------------------------------------------------------------------------
    # MAIN MAP
    # -----------------------------------------------------------------------------

    ax = fig.add_axes(
        [0.05, 0.05, 0.90, 0.90],
        projection=ccrs.PlateCarree(),
    )

    ax.imshow(
        rgb,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
    )

    # Bathymetric contours
    levels = np.arange(
        np.floor(z.min() / 100) * 100,
        0,
        1000,
    )

    cs = ax.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )

    # # Receiver halo
    # ax.scatter(
    #     rx_lon,
    #     rx_lat,
    #     s=120,
    #     c="white",
    #     linewidths=0,
    #     transform=ccrs.PlateCarree(),
    #     zorder=9,
    # )

    # # Receiver symbols
    # ax.scatter(
    #     rx_lon,
    #     rx_lat,
    #     s=50,
    #     c="crimson",
    #     marker="^",
    #     edgecolors="black",
    #     linewidths=0.5,
    #     transform=ccrs.PlateCarree(),
    #     zorder=10,
    # )

    # Zoom rectangle
    rect = Rectangle(
        (zoom_extent[0], zoom_extent[2]),
        zoom_extent[1] - zoom_extent[0],
        zoom_extent[3] - zoom_extent[2],
        fill=False,
        edgecolor="red",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        zorder=20,
    )

    ax.add_patch(rect)

    # Grid
    gl = ax.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.5,
    )

    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        # "Bathymetry and Receiver Network",
        "9R network",
        fontsize=16,
    )

    # for mmsi in ds_ais.mmsi.values[0:100]:

    #     lon_track = ds_ais.lon.sel(mmsi=mmsi).values
    #     lat_track = ds_ais.lat.sel(mmsi=mmsi).values

    #     mask = np.isfinite(lon_track) & np.isfinite(lat_track)

    #     ax.plot(
    #         lon_track[mask],
    #         lat_track[mask],
    #         color="red",
    #         linewidth=1,
    #         alpha=0.3,
    #         transform=ccrs.PlateCarree(),
    #         zorder=15,
    #     )

    ax.pcolormesh(
        xedges,
        yedges,
        H.T,
        cmap="inferno",
        norm=LogNorm(vmin=1),
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=15,
    )

    # Create mask
    land_mask = z >= 0

    # Land color (RGB)
    land_color = np.array([0.9, 0.9, 0.9])

    # Replace land pixels by a uniform color
    rgb_land = rgb.copy()
    rgb_land[land_mask, :3] = land_color
    rgb_land[~land_mask] = np.nan

    ax.imshow(
        rgb_land,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
        zorder=16,
    )

    # Coastlines and land
    ax.coastlines(resolution="10m", linewidth=1, zorder=17)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=17)

    # =============================================================================
    # INSET MAP
    # =============================================================================

    axins = fig.add_axes(
        [0.55, 0.10, 0.30, 0.30],  # left, bottom, width, height
        projection=ccrs.PlateCarree(),
    )

    # Bathymetry in inset
    axins.imshow(
        rgb,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
    )

    # Bathymetric contours
    levels = np.arange(
        np.floor(z.min() / 100) * 100,
        0,
        500,
    )

    # Contours
    cs = axins.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    # axins.clabel(cs, fmt="%d m", fontsize=7)

    # Receivers
    axins.scatter(
        rx_lon,
        rx_lat,
        s=80,
        c="crimson",
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=40,
    )

    # Receiver labels
    for rid, x, y in zip(rx_id, rx_lon, rx_lat):
        axins.annotate(
            str(rid),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.7,
            ),
            zorder=41,
        )

    # Zoom extent
    axins.set_extent(zoom_extent)

    # Grid
    gl2 = axins.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.3,
    )

    gl2.top_labels = False
    gl2.right_labels = True
    gl2.left_labels = False

    axins.set_title(
        "Receiver Network",
        fontsize=10,
    )

    # =============================================================================
    # OPTIONAL: CONNECT INSET TO ZOOM RECTANGLE
    # =============================================================================

    # # Comment this section out if you don't like it

    # fig.lines.extend(
    #     [
    #         plt.Line2D(
    #             [0.53, 0.60],
    #             [0.60, 0.70],
    #             transform=fig.transFigure,
    #             color="red",
    #             lw=1,
    #             alpha=0.7,
    #         ),
    #         plt.Line2D(
    #             [0.53, 0.60],
    #             [0.50, 0.55],
    #             transform=fig.transFigure,
    #             color="red",
    #             lw=1,
    #             alpha=0.7,
    #         ),
    #     ]
    # )

    # =============================================================================

    # plt.show()


def plot_ais_traj_and_bathy_zoom_network(
    ds_bathy, ds_network_9R, ds_ais, lon_lat_extent_deg=0.5
):
    # Extract data
    z = ds_bathy["elevation"].values
    lon = ds_bathy["lon"].values
    lat = ds_bathy["lat"].values

    # Receiver dataset
    rx_lon = ds_network_9R["longitude"].values
    rx_lat = ds_network_9R["latitude"].values
    rx_id = ds_network_9R["receiver_id"].values

    # pad = 0.25  # degrees

    zoom_extent = [
        rx_lon.min() - lon_lat_extent_deg / 2,
        rx_lon.max() + lon_lat_extent_deg / 2,
        rx_lat.min() - lon_lat_extent_deg / 2,
        rx_lat.max() + lon_lat_extent_deg / 2,
    ]

    bathy_zoom = ds_bathy.sel(
        lon=slice(zoom_extent[0], zoom_extent[1]),
        lat=slice(zoom_extent[2], zoom_extent[3]),
    )

    z = bathy_zoom["elevation"].values
    lon = bathy_zoom["lon"].values
    lat = bathy_zoom["lat"].values

    # Hillshade
    ls = LightSource(
        azdeg=315, altdeg=45
    )  # illumination azimuth  # illumination elevation
    rgb = ls.shade(-z, cmap=plt.cm.Blues, vert_exag=5, blend_mode="overlay")

    # Figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Shaded bathymetry
    ax.imshow(
        rgb,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
    )

    # Optional depth contours
    levels = np.arange(np.floor(z.min() / 100) * 100, 0, 1000)

    cs = ax.contour(
        lon,
        lat,
        z,
        levels=levels,
        colors="k",
        linewidths=0.4,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    ax.clabel(cs, fmt="%d m", fontsize=7)

    # Coastlines
    ax.coastlines(resolution="10m", linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # Grid
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("")

    rx_lon = ds_network_9R["longitude"].values
    rx_lat = ds_network_9R["latitude"].values
    rx_id = ds_network_9R["receiver_id"].values

    # Receivers
    ax.scatter(
        rx_lon,
        rx_lat,
        s=80,
        c="crimson",
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=40,
    )

    # Receiver labels
    for rid, x, y in zip(rx_id, rx_lon, rx_lat):
        ax.annotate(
            str(rid),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.7,
            ),
            zorder=41,
        )

    ds_ais_day = ds_ais.sel(
        time="2023-01-02",
    )
    for mmsi in ds_ais_day.mmsi.values:

        lon_track = ds_ais_day.lon.sel(mmsi=mmsi).values
        lat_track = ds_ais_day.lat.sel(mmsi=mmsi).values

        mask = np.isfinite(lon_track) & np.isfinite(lat_track)

        ax.plot(
            lon_track[mask],
            lat_track[mask],
            color="red",
            linewidth=1,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )

    # Zoom on the area of interest
    ax.set_extent(zoom_extent)

    # plt.tight_layout()
    # plt.show()


# ======================================================================================================================
# Miscellaneous functions
# ======================================================================================================================


def progression_bar(index: int, index0: int, indexf: int, prev_progress: int) -> int:
    step = 1
    no_graduations = 100
    current_progress = int((index - index0) / (indexf - index0) * 100)

    if current_progress >= prev_progress + step:
        print("\r", end="")
        print(
            f"Progress: "
            + "\u2588" * int(current_progress)
            + "." * (no_graduations - current_progress)
            + f" {int(current_progress)}%",
            end="",
        )
        return current_progress

    return prev_progress


PROJECT_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
ROOT_BATHY_DATA = os.path.join(PROJECT_ROOT, "data", "bathy")


def load_bathy(
    box_center_lon,
    box_center_lat,
    dlon_box=0.25,
    dlat_box=0.25,
    root_bathy_data=ROOT_BATHY_DATA,
):
    # input_data_root = os.path.join(project_root, "data")
    bathy_fpath = os.path.join(root_bathy_data, "GEBCO_2021_sub_ice_topo.nc")

    # Load bathy data
    ds_bathy = xr.open_dataset(bathy_fpath)

    # Slice data to get the area of interest
    lat0 = box_center_lat
    lon0 = box_center_lon
    ds_bathy = ds_bathy.sel(
        lat=slice(
            lat0 - dlat_box / 2,
            lat0 + dlat_box / 2,
        ),
        lon=slice(
            lon0 - dlon_box / 2,
            lon0 + dlon_box / 2,
        ),
    )

    return ds_bathy


if __name__ == "__main__":
    pass
