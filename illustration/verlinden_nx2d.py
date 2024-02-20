import numpy as np
import matplotlib.pyplot as plt
from publication.PublicationFigure import PubFigure


def plot_angle_repartition(ds, grid_info):
    dmax = ds.r_from_rcv.max(dim=["lat", "lon", "idx_rcv"]).round(0).values
    delta = min(ds.dx, ds.dy)
    dtheta_th = np.arctan(delta / dmax)
    list_theta_th = np.arange(ds.az_rcv.min(), ds.az_rcv.max(), dtheta_th)

    print(f"Number of angles : {len(list_theta_th)}")
    print(f"Max distance from rcv = {dmax}m")
    print(f"Estimated angle step = {dtheta_th * 180 / np.pi}°")
    print(f"Estimated Number of angles : {len(list_theta_th)}")

    # Plot angles
    cmap = plt.cm.get_cmap("bwr", len(list_theta_th))
    # color_list = ["#%06X" % randint(0, 0xFFFFFF) for i in range(len(list_theta_th))]
    color_list = [cmap(i) for i in range(len(list_theta_th))]

    fig = PubFigure()
    plt.figure()
    ds.az_propa.isel(idx_rcv=0).plot(cmap=cmap)

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

    plt.scatter(
        ds.lon_rcv.isel(idx_rcv=0), ds.lat_rcv.isel(idx_rcv=0), s=100, color="k"
    )
    plt.ylim(
        [
            min(ds.lat.min(), ds.lat_rcv.isel(idx_rcv=0)) - 0.1,
            max(ds.lat.max(), ds.lat_rcv.isel(idx_rcv=0)) + 0.1,
        ]
    )
    plt.xlim(
        [
            min(ds.lon.min(), ds.lon_rcv.isel(idx_rcv=0)) - 0.1,
            max(ds.lon.max(), ds.lon_rcv.isel(idx_rcv=0)) + 0.1,
        ]
    )
    plt.legend(ncol=4, loc="lower right")

    fig.set_full_screen()
    plt.savefig(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\Nx2D_verlinden\angle_repartition.png"
    )
    plt.close()
