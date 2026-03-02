import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from publication.publication_figure import color, PubFigure, LargeFigure


# -----------------------------------------------------------------------------
# Figure style
# -----------------------------------------------------------------------------

pfig = LargeFigure(
    legend_fontsize=10,
    label_fontsize=18,
    title_fontsize=22,
    ticks_fontsize=16,
)


def rtf_mfp_animation(
    ds_theta,
    df_library,
    ds_gps_event,
    ds_ais_event,
    normalization_percentile=50,
    save=True,
    root_img="",
    output_fname="rtf_mfp_results",
    step=4,
    fps=10,
    dpi=80,
):

    # Normalize theta to get distances in [0, 1]
    max_at_each_seg = np.percentile(
        ds_theta.theta.values, normalization_percentile, axis=1
    )[:, np.newaxis]
    theta = (max_at_each_seg - ds_theta.theta) / max_at_each_seg
    mu = np.clip(theta, 0, 1)

    # -----------------------------------------------------------------------------
    # Trajectories
    # -----------------------------------------------------------------------------
    gps_event_seg_dt = ds_gps_event.sel(
        time=ds_theta.segment_dt.values, method="nearest"
    )
    ais_event_seg_dt = ds_ais_event.sel(
        time=ds_theta.segment_dt.values, method="nearest"
    )

    # Event
    event_traj = {}
    event_traj["jules_gps"] = {
        "e": gps_event_seg_dt.e.values,
        "n": gps_event_seg_dt.n.values,
        "name": "GPS Jules",
    }
    for mmsi in ds_ais_event.mmsi.values:
        ship = ais_event_seg_dt.sel(mmsi=mmsi)
        event_traj[mmsi] = {
            "e": ship.e.values,
            "n": ship.n.values,
            "name": f"AIS {mmsi}",
        }

    # Library
    library_pos = {
        "e": df_library["emission_interp_e_gps"].values,
        "n": df_library["emission_interp_n_gps"].values,
    }

    n_lib_replicas = library_pos["e"].size
    n_frames = ds_theta.segment_dt.size

    # -----------------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------------
    fig, ax_traj = plt.subplots(1, 1, constrained_layout=True)

    # -----------------------------------------------------------------------------
    # LEFT: probabilistic localisation
    # -----------------------------------------------------------------------------

    ax_traj.set_xlabel("E [m]")
    ax_traj.set_ylabel("N [m]")
    ax_traj.set_aspect("equal")

    # (line,) = ax_traj.plot([], [], lw=2, color="tab:blue", label="Trajectoire testée")
    # (point,) = ax_traj.plot([], [], "o", color="red", zorder=15)

    sc_lib = ax_traj.scatter(
        library_pos["e"],
        library_pos["n"],
        c=np.zeros(n_lib_replicas),
        cmap="magma_r",
        vmin=0,
        vmax=1,
        s=60,
        zorder=10,
        label="Référence (pondérée)",
    )

    cbar = fig.colorbar(sc_lib, ax=ax_traj, pad=0.01)
    cbar.set_label(r"$\mu$")

    # -------------------------------------------------------------------------
    # Event trajectories artists
    # -------------------------------------------------------------------------

    event_lines = {}
    event_points = {}

    for i, (key, traj) in enumerate(event_traj.items()):

        col = color(i)

        (line_i,) = ax_traj.plot(
            [],
            [],
            lw=2,
            color=col,
            label=traj["name"],
        )

        (point_i,) = ax_traj.plot(
            [],
            [],
            "o",
            color=col,
            zorder=15,
        )

        event_lines[key] = line_i
        event_points[key] = point_i

    ax_traj.legend(fontsize=10)

    lib_grid_offset = 500
    ax_traj.set_xlim(
        library_pos["e"].min() - lib_grid_offset,
        library_pos["e"].max() + lib_grid_offset,
    )

    ax_traj.set_ylim(
        library_pos["n"].min() - lib_grid_offset,
        library_pos["n"].max() + lib_grid_offset,
    )

    # -----------------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------------

    def init():

        sc_lib.set_array(np.zeros(n_lib_replicas))

        for key in event_traj.keys():
            event_lines[key].set_data([], [])
            event_points[key].set_data([], [])

        return (
            sc_lib,
            *event_lines.values(),
            *event_points.values(),
        )

    def update(frame):

        # Update probabilistic localisation
        sc_lib.set_array(mu[frame])

        # Update trajectories
        for key, traj in event_traj.items():

            e = traj["e"]
            n = traj["n"]

            event_lines[key].set_data(e[: frame + 1], n[: frame + 1])
            event_points[key].set_data([e[frame]], [n[frame]])

        return (
            sc_lib,
            *event_lines.values(),
            *event_points.values(),
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(ds_theta.segment_dt), step),
        init_func=init,
        interval=200,
        blit=True,
    )

    # -----------------------------------------------------------------------------
    # Save (optional)
    # -----------------------------------------------------------------------------
    if save:

        ani.save(
            os.path.join(root_img, f"{output_fname}.gif"),
            fps=fps,
            dpi=dpi,
        )
    else:
        plt.show()


if __name__ == "__main__":
    pass
