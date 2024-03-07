import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure
from localisation.verlinden.verlinden_path import TC_WORKING_DIR
from get_data.bathymetry.bathy_profile_extraction import extract_bathy_profile

pfig = PubFigure()


def bathy_flat_seabed(
    testcase_name="testcase1",
    waveguide_depth=200,
    max_range=50,
    plot=False,
    bathy_path="",
):
    # Define bathymetry
    r = [0, max_range]
    h = [waveguide_depth, waveguide_depth]

    # Save bathymetry
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    # Plot bathy
    if plot:
        max_depth = waveguide_depth + 10
        plt.figure(figsize=(16, 8))
        plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, max_depth])
        plt.fill_between(r, h, max_depth, color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path)
        plt.close()


def bathy_sin_slope(
    testcase_name="testcase1",
    min_depth=150,
    max_range=50,
    theta=94,
    range_periodicity=6,
    plot=False,
    bathy_path="",
):
    # Define bathymetry
    fr = 1 / range_periodicity
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    alpha = 50
    h = min_depth - alpha * (
        -1
        + np.sin(
            2 * np.pi * r * np.cos(theta * np.pi / 180) / range_periodicity - np.pi / 2
        )
    )

    # Save bathymetry
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )
    if plot:
        max_depth = min_depth + 2 * alpha
        plt.figure(figsize=(16, 8))
        plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, max_depth])
        plt.fill_between(r, h, max_depth, color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path)
        plt.close()


def bathy_seamount(
    testcase_name="testcase1",
    min_depth=150,
    max_range=50,
    seamount_height=100,
    seamount_width=6,
    plot=False,
    bathy_path="",
):
    # Define bathymetry
    max_depth = min_depth + seamount_height
    fr = 1 / seamount_width
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    r_seamount = r.max() / 2
    r0 = r_seamount - seamount_width / 2
    r1 = r_seamount + seamount_width / 2

    h_seamount = min_depth
    h = np.ones(r.size) * max_depth

    alpha = (h_seamount - max_depth) / (r_seamount - r0)
    upslope = alpha * (r - r0) + max_depth
    downslope = -alpha * (r - r_seamount) + h_seamount

    idx_r_before = (r0 < r) * (r <= r_seamount)
    h[idx_r_before] = upslope[idx_r_before]
    idx_r_after = (r_seamount <= r) * (r < r1)
    h[idx_r_after] = downslope[idx_r_after]

    # Save bathymetry
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    if plot:
        max_depth = max_depth + 10
        plt.figure(figsize=(16, 8))
        plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, max_depth])
        plt.fill_between(r, h, max_depth, color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path)
        plt.close()


def mmdpm_profile(
    testcase_name,
    mmdpm_testname="PVA_RR48",
    azimuth=360,
    max_range_km=50,
    plot=False,
    bathy_path="",
):
    data_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm"
    fpath = os.path.join(
        data_dir, mmdpm_testname, f"mmdpm_test_{mmdpm_testname}_{azimuth}.csv"
    )
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)

    pd_data = pd.read_csv(fpath, sep=",", header=None)
    r = pd_data[0]
    h = pd_data[1]

    # Limit ranges to max_range_km
    idx = r <= max_range_km
    r = r[idx]
    h = h[idx]

    # Plot full bathymetry profile
    if plot:
        plt.figure(figsize=(16, 8))
        plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, h.max()])
        plt.fill_between(r, h, h.max(), color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path)
        plt.close()

    # Subsample profile to reduce cpu time
    r = r[::10]
    h = h[::10]

    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    if plot:
        plt.figure(figsize=(16, 8))
        plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, h.max()])
        plt.fill_between(r, h, h.max(), color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path.replace(".png", "_subsampled.png"))
        plt.close()


def extract_2D_bathy_profile(
    bathy_nc_path,
    obs_lon,
    obs_lat,
    testcase_name,
    azimuth=0,
    max_range_km=100,
    range_resolution=100,
    plot=False,
    bathy_path="",
):
    # Load bathymetry data
    ds_bathy = xr.open_dataset(bathy_nc_path)
    # Set bathymetry as positive towards the bottom
    ds_bathy["bathymetry"] = ds_bathy.elevation * -1

    # Extract profile
    range_along_profile, bathymetry_profile = extract_bathy_profile(
        xr_bathy=ds_bathy,
        start_lat=obs_lat,
        start_lon=obs_lon,
        azimuth=azimuth,
        range_resolution=range_resolution,
        max_range_m=max_range_km * 1e3,
    )

    # Save profile
    r_km = range_along_profile / 1e3
    h_m = bathymetry_profile

    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    pd.DataFrame({"r": np.round(r_km, 3), "h": np.round(h_m, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    if plot:
        plt.figure(figsize=(16, 8))
        plt.plot(r_km, h_m, color="k", linewidth=2, marker="o", markersize=2)
        plt.ylim([0, h_m.max()])
        plt.fill_between(r_km, h_m, h_m.max(), color="lightgrey")
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)", fontsize=pfig.label_fontsize)
        plt.ylabel("Depth (m)", fontsize=pfig.label_fontsize)
        pfig.apply_ticks_fontsize()
        plt.grid()
        plt.savefig(bathy_path)
        plt.close()
