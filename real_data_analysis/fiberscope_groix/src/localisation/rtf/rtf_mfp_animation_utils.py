import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from publication.publication_figure import color, PubFigure
from propa.rtf.rtf_utils import D_hermitian_angle_fast
from real_data_analysis.fiberscope_groix.src.fiberscope_groix_manager import (
    FiberscopeManager,
    BandFilter,
)
from real_data_analysis.fiberscope_groix.src import params

# -----------------------------------------------------------------------------
# Figure style
# -----------------------------------------------------------------------------

pfig = PubFigure(
    legend_fontsize=10,
    label_fontsize=18,
    title_fontsize=22,
    ticks_fontsize=16,
)

# -----------------------------------------------------------------------------
# Paths & data loading
# -----------------------------------------------------------------------------

project_root = params.project_root
root_folder = os.path.join(project_root, "real_data_analysis", "fiberscope_groix")
data_folder = os.path.join(root_folder, "data")
img_folder = os.path.join(root_folder, "img")

ds_gps = xr.open_dataset(os.path.join(data_folder, "gps.nc"))
ds_arr = xr.open_dataset(os.path.join(data_folder, "processed_arrivals.nc"))
df_arr = ds_arr.to_dataframe()

# -----------------------------------------------------------------------------
# Fiberscope manager
# -----------------------------------------------------------------------------

bandfilter = BandFilter(order=4, lowcut=200, highcut=990)

fsm = FiberscopeManager(
    root_processed_data=data_folder,
    h_index_ref=1,
    plot_feature=True,
    bandfilter=bandfilter,
    tau_ir=3,
    process_pulse_one_by_one=True,
)

# -----------------------------------------------------------------------------
# Distance computation
# -----------------------------------------------------------------------------


def get_dists(df_arr, seq_id_ref, seq_id, fmin=600, fmax=800):

    dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}

    xr_seq_ref = xr.open_dataset(
        os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
    )
    df_seq_ref = df_arr.loc[df_arr["sequence_id"] == seq_id_ref]
    df_seq_ref = df_seq_ref.loc[df_seq_ref["pulse_id"].isin(xr_seq_ref.pulse_id.values)]

    ref_pos_e = df_seq_ref["emission_interp_e_gps"].values
    ref_pos_n = df_seq_ref["emission_interp_n_gps"].values

    rtf_ref = xr_seq_ref.rtf_amp_hat * np.exp(1j * xr_seq_ref.rtf_phase_hat)

    xr_seq = xr.open_dataset(
        os.path.join(fsm.root_data_sequence, f"sequence_{seq_id}_rtf.nc")
    )
    df_seq = df_arr.loc[df_arr["sequence_id"] == seq_id]
    df_seq = df_seq.loc[df_seq["pulse_id"].isin(xr_seq.pulse_id.values)]

    theta_distances = []

    for pulse_id in xr_seq.pulse_id.values:

        xr_seq_pulse = xr_seq.sel(pulse_id=pulse_id)
        rtf_pulse = xr_seq_pulse.rtf_amp_hat * np.exp(1j * xr_seq_pulse.rtf_phase_hat)

        rtf_ref_sel = rtf_ref.sel(f_rtf=slice(fmin, fmax))
        rtf_pulse = rtf_pulse.sel(f_rtf=slice(fmin, fmax))

        theta = []
        for pid_ref in xr_seq_ref.pulse_id.values:
            rtf_ref_i = rtf_ref_sel.sel(pulse_id=pid_ref)
            theta_i = D_hermitian_angle_fast(
                rtf_ref=rtf_ref_i.values,
                rtf=rtf_pulse.values,
                **dist_kwargs,
            )
            theta.append(theta_i)

        theta_distances.append(theta)

    return (
        np.array(theta_distances),
        xr_seq_ref,
        df_seq_ref,
        xr_seq,
        df_seq,
    )


# -----------------------------------------------------------------------------
# Compute distances
# -----------------------------------------------------------------------------

seq_ref = 144
seq_id = 147
fmin, fmax = 600, 800

theta_distances, xr_seq_ref, df_seq_ref, xr_seq, df_seq = get_dists(
    df_arr, seq_ref, seq_id, fmin, fmax
)

# Normalize → probability-like
theta_distances = (90 - theta_distances) / 90
theta_distances = np.clip(theta_distances, 0, 1)

# -----------------------------------------------------------------------------
# Trajectories
# -----------------------------------------------------------------------------

df_seq = df_seq.sort_values("emission_datetime")
E = df_seq["emission_interp_e_gps"].values
N = df_seq["emission_interp_n_gps"].values
t = df_seq["emission_datetime"].values

df_seq_ref = df_seq_ref.sort_values("emission_datetime")
E_ref = df_seq_ref["emission_interp_e_gps"].values
N_ref = df_seq_ref["emission_interp_n_gps"].values

# -----------------------------------------------------------------------------
# MAP estimation
# -----------------------------------------------------------------------------

map_indices = np.argmax(theta_distances, axis=1)
E_map = E_ref[map_indices]
N_map = N_ref[map_indices]

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
ax_traj, ax_map = axes

# -----------------------------------------------------------------------------
# LEFT: probabilistic localisation
# -----------------------------------------------------------------------------

# ax_traj.set_title("Localisation probabiliste (RTF)")
ax_traj.set_xlabel("E [m]")
ax_traj.set_ylabel("N [m]")
ax_traj.set_aspect("equal")

(line,) = ax_traj.plot([], [], lw=2, color="tab:blue", label="Trajectoire testée")
(point,) = ax_traj.plot([], [], "o", color="red", zorder=15)

sc_ref = ax_traj.scatter(
    E_ref,
    N_ref,
    c=np.zeros(len(E_ref)),
    cmap="jet",
    vmin=0,
    vmax=1,
    s=60,
    zorder=10,
    label="Référence (pondérée)",
)

cbar = fig.colorbar(sc_ref, ax=ax_traj, pad=0.01)
cbar.set_label(r"$\mu$")

# -----------------------------------------------------------------------------
# RIGHT: MAP trajectory
# -----------------------------------------------------------------------------

ax_map.set_title("Trajectoire MAP estimée")
ax_map.set_xlabel("E [m]")
ax_map.set_ylabel("N [m]")

(line_map,) = ax_map.plot([], [], lw=2, color="black", label="MAP")
(point_map,) = ax_map.plot([], [], "o", color="crimson")

ax_map.legend()

ax_map.set_xlim(E_ref.min() - 10, E_ref.max() + 10)
ax_map.set_ylim(N_ref.min() - 10, N_ref.max() + 10)
ax_map.set_aspect("equal")

# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------


def init():
    line.set_data([], [])
    point.set_data([], [])
    sc_ref.set_array(np.zeros(len(E_ref)))
    line_map.set_data([], [])
    point_map.set_data([], [])
    return line, point, sc_ref, line_map, point_map


def update(frame):

    line.set_data(E[: frame + 1], N[: frame + 1])
    point.set_data([E[frame]], [N[frame]])

    sc_ref.set_array(theta_distances[frame])

    line_map.set_data(E_map[: frame + 1], N_map[: frame + 1])
    point_map.set_data([E_map[frame]], [N_map[frame]])

    return line, point, sc_ref, line_map, point_map


ani = FuncAnimation(
    fig,
    update,
    frames=len(E),
    init_func=init,
    interval=200,
    blit=True,
)

plt.show()

# -----------------------------------------------------------------------------
# Save (optional)
# -----------------------------------------------------------------------------
# ani.save(os.path.join(img_folder, "localisation_MAP.gif"), fps=10, dpi=150)
