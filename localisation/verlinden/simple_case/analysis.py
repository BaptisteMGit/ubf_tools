import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal

from propa.kraken_toolbox.plot_utils import plotshd


root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"
# env_fname = "verlinden_1_ssp"
env_fname = "verlinden_1_test_case"

nc_path = os.path.join(root, env_fname + ".nc")
ds = xr.open_dataset(nc_path)

# Image folder
root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden\test_case\isotropic\range_independent"
root_img = os.path.join(root_img, ds.src_pos, f"dx{ds.dx}m_dy{ds.dy}m")
if not os.path.exists(root_img):
    os.makedirs(root_img)
img_basepath = os.path.join(root_img, env_fname + "_")

# n_instant_to_plot = ds.dims["src_trajectory_time"]
n_instant_to_plot = 25
n_instant_to_plot = min(n_instant_to_plot, ds.dims["src_trajectory_time"])

# # Plot one TL profile
# shd_fpath = os.path.join(root, env_fname + ".shd")
# for f in [10, 15, 20, 25, 40, 45]:
#     plotshd(shd_fpath, freq=f, units="km")
# plt.show()

# Plot ambiguity distribution
plt.figure(figsize=(10, 8))
ds.ambiguity_surface.isel(idx_obs_pairs=0, src_trajectory_time=0).plot.hist(bins=100)
# plt.show()


x_offset = 5000
y_offset = 5000

dx = 10  # m
dy = 10  # m
Lx = 100 * 1e3  # m
Ly = 80 * 1e3  # m
grid_x = np.arange(-Lx / 2, Lx / 2, dx, dtype=np.float32)
grid_y = np.arange(-Ly / 2, Ly / 2, dy, dtype=np.float32)
xx, yy = np.meshgrid(grid_x, grid_y)

rr_obs = np.array(
    [
        np.sqrt(
            (xx - ds.x_obs.sel(idx_obs=i_obs).values) ** 2
            + (yy - ds.y_obs.sel(idx_obs=i_obs).values) ** 2
        )
        for i_obs in range(ds.dims["idx_obs"])
    ]
)


delta_rr = rr_obs[0, ...] - rr_obs[1, ...]
delta_rr_ship = ds.r_obs_ship.sel(idx_obs=0) - ds.r_obs_ship.sel(idx_obs=1)

amb_surf = -10 * np.log10(ds.ambiguity_surface)

for i in range(n_instant_to_plot):
    plt.figure(figsize=(10, 8))

    amb_surf.isel(idx_obs_pairs=0, src_trajectory_time=i).plot(
        x="x", y="y", zorder=0, vmin=0, vmax=10, cmap="jet"
    )

    # plt.gca().get_images()[0].clim([0, 20])
    # ds.ambiguity_surface.isel(idx_obs_pairs=0, src_trajectory_time=i).plot(
    #     x="x", y="y", zorder=0, clim=[-1, 1]
    # )
    condition = np.abs(delta_rr - delta_rr_ship.isel(src_trajectory_time=i).values) < 10
    # plt.plot(
    #     xx[condition],
    #     yy[condition],
    #     "k",
    #     linestyle="--",
    #     label=r"$ || \overrightarrow{O_0M} || - || \overrightarrow{O_1M} || =  \Delta_i|| \overrightarrow{O_iX_{ship}} ||$",
    #     zorder=1,
    # )

    plt.scatter(
        ds.x_ship.isel(src_trajectory_time=i),
        ds.y_ship.isel(src_trajectory_time=i),
        color="k",
        marker="+",
        s=90,
        label=r"$X_{ship}$",
        zorder=2,
    )

    plt.scatter(
        ds.detected_pos_x.isel(idx_obs_pairs=0, src_trajectory_time=i),
        ds.detected_pos_y.isel(idx_obs_pairs=0, src_trajectory_time=i),
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=120,
        linewidths=2.2,
        label="Estimated position",
        zorder=3,
    )

    for i_obs in range(ds.dims["idx_obs"]):
        plt.scatter(
            ds.x_obs.isel(idx_obs=i_obs),
            ds.y_obs.isel(idx_obs=i_obs),
            marker="o",
            label=f"$O_{i_obs}$",
        )

        plt.xlim(
            [
                min(ds.x.min(), ds.x_obs.min()) - x_offset,
                max(ds.x.max(), ds.x_obs.max()) + x_offset,
            ]
        )
        plt.ylim(
            [
                min(ds.y.min(), ds.y_obs.min()) - y_offset,
                max(ds.y.max(), ds.y_obs.max()) + y_offset,
            ]
        )

    # plt.scatter(ds.x_obs, ds.y_obs, color="red")
    plt.tight_layout()
    plt.legend(ncol=2, loc="upper right")
    plt.savefig(img_basepath + f"ambiguity_surface_{i}.png")
    plt.close()

# Plot ship trajectory
plt.figure(figsize=(10, 8))
plt.plot(ds.x_ship, ds.y_ship, marker="+", color="red", label=r"$X_{ship}$")
for i_obs in range(ds.dims["idx_obs"]):
    plt.scatter(
        ds.x_obs.isel(idx_obs=i_obs),
        ds.y_obs.isel(idx_obs=i_obs),
        marker="o",
        label=f"$O_{i_obs}$",
    )
for i_pair in ds.idx_obs_pairs:
    plt.scatter(
        ds.detected_pos_x.sel(idx_obs_pairs=i_pair),
        ds.detected_pos_y.sel(idx_obs_pairs=i_pair),
        facecolors="none",
        edgecolors="blue",
        s=120,
        linewidths=2.2,
        label="Estimated position",
    )

plt.xlim(
    [
        ds.x_ship.min() - x_offset,
        ds.x_ship.min() + x_offset,
    ]
)
plt.ylim(
    [
        ds.y_ship.min() - y_offset,
        ds.y_ship.min() + y_offset,
    ]
)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(img_basepath + f"estimated_pos.png")
plt.close()

# Plot detection error
plt.figure(figsize=(10, 8))

for i_pair in ds.idx_obs_pairs:
    pos_error = np.sqrt(
        (ds.detected_pos_x.sel(idx_obs_pairs=i_pair) - ds.x_ship) ** 2
        + (ds.detected_pos_y.sel(idx_obs_pairs=i_pair) - ds.y_ship) ** 2
    )
    pos_error.plot()

plt.ylabel("Position error [m]")
plt.tight_layout()
plt.savefig(img_basepath + f"pos_error.png")
plt.close()


# Plot received signal
fig, axes = plt.subplots(
    n_instant_to_plot,
    ds.dims["idx_obs"],
    sharex=True,
    sharey=True,
)

axes = np.reshape(axes, (n_instant_to_plot, ds.dims["idx_obs"]))
for ax, col in zip(axes[0], [f"Receiver {i}" for i in ds.idx_obs]):
    ax.set_title(col)

for i_ship in range(n_instant_to_plot):
    for i_obs in range(ds.dims["idx_obs"]):
        ds.rcv_signal_event.isel(src_trajectory_time=i_ship, idx_obs=i_obs).plot(
            ax=axes[i_ship, i_obs], label="event"
        )

        ds.rcv_signal_library.sel(
            x=ds.x_ship.isel(src_trajectory_time=i_ship),
            y=ds.y_ship.isel(src_trajectory_time=i_ship),
            method="nearest",
        ).isel(idx_obs=i_obs).plot(ax=axes[i_ship, i_obs], label="library")

        axes[i_ship, i_obs].set_xlabel("")
        axes[i_ship, i_obs].set_ylabel("")
        axes[i_ship, i_obs].set_title("")
        # axes[i_ship, i_obs].set_ylim([-0.005, 0.005])

fig.supylabel("Received signal")
fig.supxlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig(img_basepath + f"rcv_signal.png")

plt.close()


# # F_f = np.fft.rfft(ds.rcv_signal_event.sel(idx_obs=rcv_pair[0]), axis=1)
# # G_f = np.fft.rfft(ds.rcv_signal_event.sel(idx_obs=rcv_pair[1]), axis=1)
# # G_f_conj = np.conj(G_f)
# # corr_vect = np.fft.irfft(F_f * G_f[:, -1:], axis=1)


# Plot correlation
fig, axes = plt.subplots(
    n_instant_to_plot,
    ds.dims["idx_obs_pairs"],
    sharex=True,
    sharey=True,
)
axes = np.reshape(
    axes, (n_instant_to_plot, ds.dims["idx_obs_pairs"])
)  # Ensure 2D axes array in case of single obs pair

for i_ship in range(n_instant_to_plot):
    for i_obs_pair in range(ds.dims["idx_obs_pairs"]):
        # lib_env = np.abs(
        #     signal.hilbert(
        #         ds.library_corr.sel(
        #             x=ds.x_ship.isel(src_trajectory_time=i_ship),
        #             y=ds.y_ship.isel(src_trajectory_time=i_ship),
        #             method="nearest",
        #         ).isel(idx_obs_pairs=i_obs_pair)
        #     )
        # )
        # event_env = np.abs(
        #     signal.hilbert(
        #         ds.event_corr.sel(idx_obs_pairs=i_pair).isel(src_trajectory_time=i_ship)
        #     )
        # )
        # plt.figure()
        # plt.plot(lib_env, label="lib")
        # plt.plot(event_env, label="event")
        # plt.legend()
        # plt.show()

        ds.event_corr.isel(src_trajectory_time=i_ship, idx_obs_pairs=i_obs_pair).plot(
            ax=axes[i_ship, i_obs_pair], label="event"
        )
        ds.library_corr.sel(
            x=ds.x_ship.isel(src_trajectory_time=i_ship),
            y=ds.y_ship.isel(src_trajectory_time=i_ship),
            method="nearest",
        ).isel(idx_obs_pairs=i_obs_pair).plot(
            ax=axes[i_ship, i_obs_pair], label="lib at ship pos"
        )
        axes[i_ship, i_obs_pair].set_xlabel("")
        axes[i_ship, i_obs_pair].set_ylabel("")
        axes[i_ship, i_obs_pair].set_title("")
        axes[i_ship, i_obs_pair].set_ylim([-1, 1])

plt.savefig(img_basepath + f"signal_corr.png")

# plt.show()
plt.legend()
plt.close()
