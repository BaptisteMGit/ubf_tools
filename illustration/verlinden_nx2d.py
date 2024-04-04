import numpy as np
import matplotlib.pyplot as plt
from publication.PublicationFigure import PubFigure


def plot_angle_repartition(ds, bathy_dlat, bathy_dlon):
    dmax = ds.r_from_rcv.max(dim=["lat", "lon", "idx_rcv"]).round(0).values
    delta = np.sqrt(bathy_dlat**2 + bathy_dlon**2)
    dtheta_th = np.arctan(delta / dmax)
    list_theta_th = np.arange(ds.az_propa.min(), ds.az_propa.max(), dtheta_th)

    print(f"Number of angles : {len(list_theta_th)}")
    print(f"Max distance from rcv = {dmax}m")
    print(f"Estimated angle step = {dtheta_th * 180 / np.pi}°")
    print(f"Estimated Number of angles : {len(list_theta_th)}")

    # Plot angles
    cmap = plt.cm.get_cmap("tab20", len(list_theta_th))
    # color_list = ["#%06X" % randint(0, 0xFFFFFF) for i in range(len(list_theta_th))]
    color_list = [cmap(i) for i in range(len(list_theta_th))]

    fig = PubFigure()
    plt.figure()
    plt.axis("equal")
    ds.az_propa.isel(idx_rcv=0).plot(cmap=cmap)

    print(f"Number of angles : {len(ds.az_propa.isel(idx_rcv=0).values)}")

    # Plot profiles
    # x_profile = np.arange(ds.lon.min() - 1e3, ds.x.max() + 1e3, 100)
    # y_profile_obs_0 = np.array(
    #     [
    #         np.tan(theta) * (x_profile - ds.lon_obs.isel(idx_obs=0).values)
    #         + ds.lat_obs.isel(idx_obs=0).values
    #         for theta in list_theta_th
    #     ]
    # )
    # llon, llat = np.meshgrid(grid_info["lons"], grid_info["lats"])
    # for itheta, theta in enumerate(list_theta_th):
    # plt.plot(
    #     x_profile,
    #     y_profile_obs_0[itheta, :],
    #     linestyle="-",
    #     label=f"{np.round(theta * 180/np.pi, 0)}°",
    #     color=color_list[itheta],
    # )
    # plt.plot(
    #     x_profile,
    #     y_profile_obs_0[itheta, :],
    #     linestyle="-.",
    #     color="k",
    #     dashes=(5, 10),
    # )
    # Derive grid points closest to the profile
    # closest_points_idx = np.abs(ds.az_rcv.sel(idx_obs=0) - theta) < dtheta_th / 2
    # plt.scatter(
    #     llon[closest_points_idx].flatten(),
    #     llat[closest_points_idx].flatten(),
    #     s=5,
    #     color=color_list[itheta],
    # )

    # Add bathy
    bathy_nc_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm\PVA_RR48\GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
    ds_bathy = xr.open_dataset(bathy_nc_path)
    ds_bathy = ds_bathy.sel(
        lon=slice(ds.lon.min(), ds.lon.max() * 1.0001),
        lat=slice(ds.lat.min() * 1.0001, ds.lat.max()),
    )
    llon, llat = np.meshgrid(ds_bathy.lon.values, ds_bathy.lat.values)
    plt.scatter(
        llon,
        llat,
        c=ds_bathy.elevation.values.flatten(),
        # s=1,
        # color="k",
    )
    # ds_bathy.elevation.plot.scatter()
    # ds_bathy.elevation.plot(cmap="viridis", add_colorbar=False)

    # plt.scatter(
    #     ds.lon_rcv.isel(idx_rcv=0), ds.lat_rcv.isel(idx_rcv=0), s=100, color="k"
    # )
    # plt.ylim(
    #     [
    #         min(ds.lat.min(), ds.lat_rcv.isel(idx_rcv=0)) - 0.01,
    #         max(ds.lat.max(), ds.lat_rcv.isel(idx_rcv=0)) + 0.01,
    #     ]
    # )
    # plt.xlim(
    #     [
    #         min(ds.lon.min(), ds.lon_rcv.isel(idx_rcv=0)) - 0.001,
    #         max(ds.lon.max(), ds.lon_rcv.isel(idx_rcv=0)) + 0.001,
    #     ]
    # )
    plt.legend(ncol=4, loc="lower right")

    fig.set_full_screen()
    plt.tight_layout()
    plt.savefig(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\Nx2D_verlinden\angle_repartition.png"
    )
    plt.close()


if __name__ == "__main__":
    import xarray as xr
    import os

    # path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\verlinden_process_output\testcase3_1\ship\not_on_grid\output_snr0dB.nc"
    path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\verlinden_process_output\testcase3_1\debug_pulse\not_on_grid\output_snr0dB.nc"
    ds = xr.open_dataset(path)

    lon, lat = 65.6019, -27.6581
    lat_rad = np.radians(lat)  # Latitude en radians
    lon_rad = np.radians(lon)  # Longitude en radians

    grid_size = 15 / 3600 * np.pi / 180  # 15" (secondes d'arc)
    lat_0 = lat_rad - grid_size
    lat_1 = lat_rad + grid_size
    lon_0 = lon_rad - grid_size
    lon_1 = lon_rad + grid_size

    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, dlat = geod.inv(
        lons1=lon,
        lats1=np.degrees(lat_0),
        lons2=lon,
        lats2=np.degrees(lat_1),
    )

    print("dlat = ", dlat)

    _, _, dlon = geod.inv(
        lons1=np.degrees(lon_0),
        lats1=lat,
        lons2=np.degrees(lon_1),
        lats2=lat,
    )

    plot_angle_repartition(ds, dlat, dlon)
