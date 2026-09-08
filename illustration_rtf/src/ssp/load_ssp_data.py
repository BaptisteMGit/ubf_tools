#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   load_ssp_data.py
@Time    :   2026/09/08 11:01:44
@Author  :   Menetrier Baptiste
@Version :   1.1 (reviewed)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Build the real sound-speed profile (SSP) datasets that
             ssp_process_eof.py's EOF/PCA decomposition is later fit
             on. Starting from CMEMS temperature/salinity reanalysis
             data (1993-2026) and GEBCO bathymetry over the same box:
               1. Plot sea-surface temperature with bathymetric
                  contours and the 2 profile locations (a shallow-water
                  "SW" one and a deep-water "DW" one) for a sanity
                  check.
               2. Plot the temperature/salinity profile spread over
                  time at both locations.
               3. Derive sound speed from temperature+salinity
                  (Mackenzie 1981, via arlpy.uwa.soundspeed) at both
                  locations, and save the resulting SSP time series to
                  '.nc' (one file per location: "sw"/"dw").
               4. Split each location's SSP time series into its 4
                  meteorological seasons and save those too (one file
                  per location x season) -- so downstream,
                  ssp_process_eof.py's own '__main__' block has 10
                  input files total: (1 all-data + 4 seasonal) x 2
                  locations.

"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import arlpy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from publication.publication_figure import PubFigure

PubFigure()

root_ssp_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\img\ssp"
root_ssp_data = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration_rtf\data\ssp"
)
os.makedirs(root_ssp_img, exist_ok=True)
os.makedirs(root_ssp_data, exist_ok=True)

# ======================================================================================================================
# Load CMEMS temperature/salinity data and GEBCO bathymetry over the same box
# ======================================================================================================================
fname = "cmems_data_1993_2026.nc"
ds = xr.open_dataset(os.path.join(root_ssp_data, fname))

box_center_lon = ds.longitude.values.mean()
box_center_lat = ds.latitude.values.mean()
dlon_box = ds.longitude.values.max() - ds.longitude.values.min()
dlat_box = ds.latitude.values.max() - ds.latitude.values.min()

# Load bathy data
from source.utils.utils_bathy import load_bathy

bathy = load_bathy(
    box_center_lon=box_center_lon,
    box_center_lat=box_center_lat,
    dlon_box=dlon_box,
    dlat_box=dlat_box,
)
# Build bathy dataset
ds_bathy = xr.Dataset(
    data_vars=dict(
        elevation=(["lat", "lon"], bathy.elevation.values),
    ),
    coords=dict(
        lon=bathy.lon.values,
        lat=bathy.lat.values,
    ),
    attrs=dict(
        description="Bathymetry data from GEBCO 2021",
        geodesic_frame="WGS84",
    ),
)

# Add attributes to variables
ds_bathy.elevation.attrs["units"] = "m"
ds_bathy.lon.attrs["units"] = "°"
ds_bathy.lat.attrs["units"] = "°"
ds_bathy.elevation.attrs["long_name"] = "Elevation (WGS84)"
ds_bathy.lon.attrs["long_name"] = "Longitude"
ds_bathy.lat.attrs["long_name"] = "Latitude"

# Define profile coordinates
sw_profile = {
    "lon": 3.592,
    "lat": 43.177,
}
dw_profile = {
    "lon": 7.542,
    "lat": 42.305,
}

# ======================================================================================================================
# Step 1 : plot sea-surface temperature with bathymetric contours and profile locations
# ======================================================================================================================
ds_t0 = ds.isel(time=0)
ds_t0_z0 = ds_t0.sel(depth=0, method="nearest")

plt.figure(figsize=(16, 8))
ds_t0_z0.thetao.plot(x="longitude", y="latitude", cmap="jet")
# Bathymetric contours
levels = np.arange(
    np.floor(ds_bathy["elevation"].min() / 100) * 100,
    0,
    100,
)

# Contours
cs = plt.contour(
    ds_bathy.lon.values,
    ds_bathy.lat.values,
    ds_bathy.elevation.values,
    levels=levels,
    colors="k",
    linewidths=0.3,
    alpha=1,
    # transform=ccrs.PlateCarree(),
)

# Plot profile pos
plt.scatter(
    sw_profile["lon"], sw_profile["lat"], color="r", marker="o", label="SW profile"
)
plt.scatter(
    dw_profile["lon"], dw_profile["lat"], color="g", marker="o", label="DW profile"
)
plt.legend(loc="upper left")
plt.savefig(os.path.join(root_ssp_img, "sst_bathy_profiles_pos.png"))
# plt.show()

# ======================================================================================================================
# Step 2 : plot temperature and salinity profile spread over time, at both locations
# ======================================================================================================================

ds_sw = ds.sel(
    longitude=sw_profile["lon"], latitude=sw_profile["lat"], method="nearest"
)
ds_dw = ds.sel(
    longitude=dw_profile["lon"], latitude=dw_profile["lat"], method="nearest"
)


# Plot
f_sw, axs_sw = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
f_dw, axs_dw = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
for it in range(ds.sizes["time"]):
    # Shallow water profiles
    ds_sw.isel(time=it).so.plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_sw[0]
    )
    ds_sw.isel(time=it).thetao.plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_sw[1]
    )

    # Deep water profiles
    ds_dw.isel(time=it).so.plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_dw[0]
    )
    ds_dw.isel(time=it).thetao.plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_dw[1]
    )


f_sw.suptitle("Shallow water")
f_sw.supylabel("Depth [m]")
axs_sw[0].set_xlabel("Salinity [psu]")
axs_sw[1].set_xlabel("Temperature [°C]")
axs_sw[0].set_title("")
axs_sw[1].set_title("")
axs_sw[0].set_ylabel("")
axs_sw[1].set_ylabel("")

f_dw.suptitle("Deep water")
f_dw.supylabel("Depth [m]")
axs_dw[0].set_xlabel("Salinity [psu]")
axs_dw[1].set_xlabel("Temperature [°C]")
axs_dw[0].set_title("")
axs_dw[1].set_title("")
axs_dw[0].set_ylabel("")
axs_dw[1].set_ylabel("")

f_sw.savefig(os.path.join(root_ssp_img, "so_thetao_sw.png"))
f_dw.savefig(os.path.join(root_ssp_img, "so_thetao_dw.png"))

plt.close("all")


# ======================================================================================================================
# Step 3 : derive sound speed (Mackenzie 1981) from temperature + salinity
# ======================================================================================================================
def prep_dataset(ssp):
    """Rename an arlpy.uwa.soundspeed() output to "ssp" and attach CF-
    style metadata describing it, ready to be saved to NetCDF.

    Args:
        ssp (xr.DataArray): sound speed (m/s), as returned by
            arlpy.uwa.soundspeed().

    Returns:
        xr.DataArray: the same data, renamed and with attrs set.
    """
    # Rename ssp variable and update attributes
    ssp = ssp.rename("ssp")
    ssp.attrs["units"] = "m/s"
    ssp.attrs["units_label"] = r"m~s$^{-1}$"
    ssp.attrs["unit_long"] = "Meters per second"
    ssp.attrs["long_name"] = "Sound speed profile"
    ssp.attrs["standard_name"] = "sound_speed"
    ssp.attrs["description"] = (
        "Sound speed profile computed from temperature and salinity profiles using the Mackenzie 1981 equation."
    )
    ssp.attrs["source"] = "CMEMS data"
    ssp.attrs["history"] = "Created by load_ssp_data.py script"
    ssp.attrs["valid_min"] = 1400.0
    ssp.attrs["valid_max"] = 1600.0

    return ssp


ssp_sw = arlpy.uwa.soundspeed(
    temperature=ds_sw.thetao, salinity=ds_sw.so, depth=ds_sw.depth
)
ssp_sw = prep_dataset(ssp_sw)


ssp_dw = arlpy.uwa.soundspeed(
    temperature=ds_dw.thetao, salinity=ds_dw.so, depth=ds_dw.depth
)
ssp_dw = prep_dataset(ssp_dw)

# Save ssp profiles to netcdf
ssp_sw.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_sw.nc"))
ssp_dw.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_dw.nc"))

# Plot profiles
f_sw, axs_sw = plt.subplots(1, 1, figsize=(10, 8))
f_dw, axs_dw = plt.subplots(1, 1, figsize=(10, 8))
for it in range(ds.sizes["time"]):
    # Shallow water profiles
    ssp_sw.isel(time=it).plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_sw
    )

    # Deep water profiles
    ssp_dw.isel(time=it).plot(
        y="depth", yincrease=False, alpha=0.25, color="b", ax=axs_dw
    )

# Save figures
f_sw.suptitle("Shallow water")
axs_sw.set_ylabel("Depth [m]")
axs_sw.set_title("")

f_dw.suptitle("Deep water")
axs_dw.set_ylabel("Depth [m]")
axs_dw.set_title("")

f_sw.savefig(os.path.join(root_ssp_img, "ssp_sw.png"))
f_dw.savefig(os.path.join(root_ssp_img, "ssp_dw.png"))

# plt.show()


# ======================================================================================================================
# Step 4 : split each location's SSP time series into meteorological seasons
# ======================================================================================================================
# Extract profiles for each season
winter_months = [12, 1, 2]
spring_months = [3, 4, 5]
summer_months = [6, 7, 8]
automn_months = [9, 10, 11]

# Extract and save seasonal profiles
ssp_sw_winter = ssp_sw.sel(time=ssp_sw.time.dt.month.isin(winter_months))
ssp_sw_spring = ssp_sw.sel(time=ssp_sw.time.dt.month.isin(spring_months))
ssp_sw_summer = ssp_sw.sel(time=ssp_sw.time.dt.month.isin(summer_months))
ssp_sw_automn = ssp_sw.sel(time=ssp_sw.time.dt.month.isin(automn_months))
ssp_sw_winter.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_sw_winter.nc"))
ssp_sw_spring.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_sw_spring.nc"))
ssp_sw_summer.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_sw_summer.nc"))
ssp_sw_automn.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_sw_automn.nc"))

ssp_dw_winter = ssp_dw.sel(time=ssp_dw.time.dt.month.isin(winter_months))
ssp_dw_spring = ssp_dw.sel(time=ssp_dw.time.dt.month.isin(spring_months))
ssp_dw_summer = ssp_dw.sel(time=ssp_dw.time.dt.month.isin(summer_months))
ssp_dw_automn = ssp_dw.sel(time=ssp_dw.time.dt.month.isin(automn_months))
ssp_dw_winter.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_dw_winter.nc"))
ssp_dw_spring.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_dw_spring.nc"))
ssp_dw_summer.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_dw_summer.nc"))
ssp_dw_automn.to_netcdf(os.path.join(root_ssp_data, "ssp_profiles_dw_automn.nc"))


# Plot seasonal profiles
def plot_seasonal_profiles(ssp_season, axs, season_name):
    print(f"Season {season_name}: {ssp_season.sizes['time']} profiles")
    for it in range(ssp_season.sizes["time"]):
        ssp_season.isel(time=it).plot(
            y="depth", yincrease=False, alpha=0.25, color="b", ax=axs
        )
    axs.set_title(f"{season_name}")
    axs.set_xlabel("")
    axs.set_ylabel("")


f_sw, axs_sw = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
f_dw, axs_dw = plt.subplots(1, 4, figsize=(16, 8), sharey=True)

# Winter profiles
plot_seasonal_profiles(ssp_sw_winter, axs_sw[0], "Winter")
plot_seasonal_profiles(ssp_dw_winter, axs_dw[0], "Winter")
# Spring
plot_seasonal_profiles(ssp_sw_spring, axs_sw[1], "Spring")
plot_seasonal_profiles(ssp_dw_spring, axs_dw[1], "Spring")
# Summer
plot_seasonal_profiles(ssp_sw_summer, axs_sw[2], "Summer")
plot_seasonal_profiles(ssp_dw_summer, axs_dw[2], "Summer")
# Spring
plot_seasonal_profiles(ssp_sw_automn, axs_sw[3], "Automn")
plot_seasonal_profiles(ssp_dw_automn, axs_dw[3], "Automn")

f_sw.supylabel("Depth [m]")
f_sw.supxlabel("Sound speed [m s$^{-1}$]")
f_dw.supylabel("Depth [m]")
f_dw.supxlabel("Sound speed [m s$^{-1}$]")

# Save
f_sw.savefig(os.path.join(root_ssp_img, "ssp_per_season_sw.png"))
f_dw.savefig(os.path.join(root_ssp_img, "ssp_per_season_dw.png"))

plt.close("all")
