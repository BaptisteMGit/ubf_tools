import numpy as np
import matplotlib.pyplot as plt
from publication.PublicationFigure import PubFigure


def plot_angle_repartition(ds, grid_x, grid_y):
    dmax = ds.r_from_obs.max(dim=["y", "x", "idx_obs"]).round(0).values
    delta = min(ds.dx, ds.dy)
    dtheta_th = np.arctan(delta / dmax)
    list_theta_th = np.arange(ds.theta_obs.min(), ds.theta_obs.max(), dtheta_th)

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
    ds.theta_propa.isel(idx_obs=0).plot(cmap=cmap)

    # Plot profiles
    x_profile = np.arange(ds.x.min() - 1e3, ds.x.max() + 1e3, 100)
    y_profile_obs_0 = np.array(
        [
            np.tan(theta) * (x_profile - ds.x_obs.isel(idx_obs=0).values)
            + ds.y_obs.isel(idx_obs=0).values
            for theta in list_theta_th
        ]
    )
    xx, yy = np.meshgrid(grid_x, grid_y)
    for itheta, theta in enumerate(list_theta_th):
        plt.plot(
            x_profile,
            y_profile_obs_0[itheta, :],
            linestyle="-",
            label=f"{np.round(theta * 180/np.pi, 0)}°",
            color=color_list[itheta],
        )
        plt.plot(
            x_profile,
            y_profile_obs_0[itheta, :],
            linestyle="-.",
            color="k",
            dashes=(5, 10),
        )
        # Derive grid points closest to the profile
        closest_points_idx = np.abs(ds.theta_obs.sel(idx_obs=0) - theta) < dtheta_th / 2
        plt.scatter(
            xx[closest_points_idx].flatten(),
            yy[closest_points_idx].flatten(),
            s=5,
            color=color_list[itheta],
        )

    plt.scatter(ds.x_obs.isel(idx_obs=0), ds.y_obs.isel(idx_obs=0))
    plt.ylim([ds.y_obs.isel(idx_obs=0), ds.y.max() + 1e3])
    plt.xlim([ds.x.min() - 3e3, ds.x.max() + 3e3])
    plt.legend(ncol=4, loc="lower right")

    fig.set_full_screen()
    plt.savefig(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\Nx2D_verlinden\angle_repartition.png"
    )
    plt.close()
