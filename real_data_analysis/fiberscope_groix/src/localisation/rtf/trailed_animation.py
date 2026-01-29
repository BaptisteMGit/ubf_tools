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

pfig = PubFigure(
    legend_fontsize=10, label_fontsize=18, title_fontsize=22, ticks_fontsize=16
)


project_root = params.project_root

root_groix_data = os.path.join(project_root, "data", "fiberscope_groix_oct_2025")
root_groix_wav = os.path.join(root_groix_data, "wav")
root_groix_metadata = os.path.join(root_groix_data, "metadata")

root_folder = os.path.join(project_root, "real_data_analysis", "fiberscope_groix")
data_folder = os.path.join(root_folder, "data")
img_folder = os.path.join(root_folder, "img")


ds_gps = xr.open_dataset(os.path.join(data_folder, "gps.nc"))

# Load arrivals dataset
fpath = os.path.join(
    data_folder,
    f"processed_arrivals.nc",
)
ds_arr = xr.open_dataset(fpath)
df_arr = ds_arr.to_dataframe()

seq_id = [144]  # , 134, 135, 136, 143]

df_seq = df_arr.loc[df_arr["Sequence_id"].isin(seq_id)]


# Load wav data from netcdf
nc_fpath = os.path.join(data_folder, "wav.nc")
ds_wav = xr.open_dataset(nc_fpath)

datetime_fmt = ds_wav.attrs["datetime_format"]
fs = ds_wav.attrs[f"fs_obs{1}"]
ts = 1 / fs


# Select window size to compute RTF
tau_ir_sim = 0.01  # Estimated impulse response duration (-30dB) on profile p1
tau_rtf_analysis = 10 * tau_ir_sim  # To ensure we include the entire response

# Number of samples corresponding to the assumed impulse response duration
n_rtf_analysis = int(tau_rtf_analysis * fs)
# Get closer power of 2
nperseg = 2 ** int(
    np.log2(n_rtf_analysis) + 1
)  # Number of sample per snapshot to use = closest power of two
alpha_overlap = 0.5
noverlap = int(nperseg * alpha_overlap)

print(f"nperseg = {nperseg}, noverlap = {noverlap}")

bandfilter = BandFilter(
    order=4,
    lowcut=200,
    highcut=990,
)
# bandfilter = None

h_index_ref = 1  # -> OBS 3 has the higher snr
# root_rtf_data = os.path.join(data_folder, "rtf")
plot_feature = True
process_pulse_one_by_one = True

fsm = FiberscopeManager(
    root_processed_data=data_folder,
    h_index_ref=h_index_ref,
    plot_feature=plot_feature,
    bandfilter=bandfilter,
    tau_ir=3,
    process_pulse_one_by_one=process_pulse_one_by_one,
)


# -----------------------------------------------------------------------------
# Paths & data loading
# -----------------------------------------------------------------------------

project_root = params.project_root
root_folder = os.path.join(project_root, "real_data_analysis", "fiberscope_groix")
data_folder = os.path.join(root_folder, "data")

ds_gps = xr.open_dataset(os.path.join(data_folder, "gps.nc"))
ds_arr = xr.open_dataset(os.path.join(data_folder, "processed_arrivals.nc"))
df_arr = ds_arr.to_dataframe()

# -----------------------------------------------------------------------------
# Distance computation
# -----------------------------------------------------------------------------


def get_dists(df_arr, seq_id_ref, seq_id, fmin=600, fmax=800):

    # dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}

    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id_ref}_rtf.nc")
    xr_seq_ref = xr.open_dataset(fpath)
    df_seq_ref = df_arr.loc[df_arr["Sequence_id"] == seq_id_ref]
    ref_pos_e = df_seq_ref["Emission interpolated E GPS"].iloc[0]
    ref_pos_n = df_seq_ref["Emission interpolated N GPS"].iloc[0]

    df_seq_i = df_arr.loc[df_arr["Sequence_id"] == seq_id]
    fpath = os.path.join(fsm.root_data_sequence, f"sequence_{seq_id}_rtf.nc")
    xr_seq_i = xr.open_dataset(fpath)

    # # TODO remove this
    # xr_seq_i = xr_seq_i.sel(pluse_id=slice(0, 300))
    # df_seq_i = xr_seq_i.loc[xr_seq_i["pulse_id"].isin(xr_seq_i.pulse_id.values)]

    spatial_dist = np.sqrt(
        (ref_pos_e - df_seq_i["Emission interpolated E GPS"]) ** 2
        + (ref_pos_n - df_seq_i["Emission interpolated N GPS"]) ** 2
    )

    xr_seq_i_ref = xr_seq_ref.sel(pulse_id=0)
    rtf_ref = xr_seq_i_ref.rtf_amp_hat * np.exp(1j * xr_seq_i_ref.rtf_phase_hat)
    rtf_pulse = xr_seq_i.rtf_amp_hat * np.exp(1j * xr_seq_i.rtf_phase_hat)

    rtf_ref = rtf_ref.sel(f_rtf=slice(fmin, fmax))
    rtf_pulse = rtf_pulse.sel(f_rtf=slice(fmin, fmax))

    theta_dist = []
    for pid in xr_seq_i.pulse_id.values:
        theta = D_hermitian_angle_fast(
            rtf_ref=rtf_ref.values,
            rtf=rtf_pulse.sel(pulse_id=pid).values,
            ax_rcv=0,
            ax_f=1,
            apply_mean=True,
        )
        theta_dist.append(theta)

    return np.array(spatial_dist), np.array(theta_dist), xr_seq_i, df_seq_i


# -----------------------------------------------------------------------------
# Compute theta distances
# -----------------------------------------------------------------------------

fmin, fmax = 600, 800
# seq_id = 144
seq_id = 144

seq_refs = [151, 116, 127]  # obs1, obs2, obs3

spatial_dist_obs = []
theta_dist_obs = []

for seq_id_ref in seq_refs:
    spatial_dist, theta_dist, xr_seq, df_seq = get_dists(
        df_arr=df_arr,
        seq_id_ref=seq_id_ref,
        seq_id=seq_id,
        fmin=fmin,
        fmax=fmax,
    )

    # TODO remove this
    spatial_dist = spatial_dist[:250]
    theta_dist = theta_dist[:250]

    # # Smooth with rolling average
    n_roll_avg = 10
    theta_dist = np.convolve(theta_dist, np.ones(n_roll_avg) / n_roll_avg, mode="same")
    # energy = np.convolve(energy, np.ones(n_roll_avg) / n_roll_avg, mode="same")

    spatial_dist_obs.append(spatial_dist)
    theta_dist_obs.append(theta_dist)


# Normalize theta dist to 1
theta_dist_obs = (90 - np.array(theta_dist_obs)) / 90
# Normalize each columns to sum to 1 (presence probability)
theta_dist_obs = theta_dist_obs / np.sum(theta_dist_obs, axis=0)

pulse_id = xr_seq.pulse_id.values[:250]
df_seq = df_seq.loc[df_seq["pulse_id"].isin(pulse_id)]
pulse_dt = df_seq["Emission datetime"].values

# -----------------------------------------------------------------------------
# Trajectory data
# -----------------------------------------------------------------------------

df_seq = df_seq.sort_values("Emission datetime")
E = df_seq["Emission interpolated E GPS"].values
N = df_seq["Emission interpolated N GPS"].values
t = df_seq["Emission datetime"].values

# -----------------------------------------------------------------------------
# Figure and subplots
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
ax_traj, ax_bar = axes

# -----------------------------------------------------------------------------
# Left subplot: trajectory
# -----------------------------------------------------------------------------

ax_traj.set_title(f"Trajectoire - Sequence {seq_id}")
ax_traj.set_xlabel("E [m]")
ax_traj.set_ylabel("N [m]")
ax_traj.set_aspect("equal")

ax_traj.set_xlim(E.min() - 100, E.max() + 100)
ax_traj.set_ylim(N.min() - 100, N.max() + 100)

# keys = ["obs1", "obs2", "obs3", "t1", "t2", "t3", "t4", "t5"]
keys = ["obs1", "obs2", "obs3"]

for ik, k in enumerate(keys):
    ax_traj.scatter(
        ds_gps.attrs[f"{k}_e_apriori"],
        ds_gps.attrs[f"{k}_n_apriori"],
        marker="D",
        color=color(ik),
        s=40,
        zorder=10,
        label=k.upper(),
    )

(line,) = ax_traj.plot([], [], lw=2, color="tab:blue")
(point,) = ax_traj.plot([], [], "o", color="red", zorder=15)

time_text = ax_traj.text(0.02, 0.95, "", transform=ax_traj.transAxes)
ax_traj.legend(loc="lower right")

# -----------------------------------------------------------------------------
# Right subplot: animated bar chart (θ distances)
# -----------------------------------------------------------------------------

obs_labels = ["OBS1", "OBS2", "OBS3"]
bar_colors = [color(0), color(1), color(2)]

bars = ax_bar.bar(
    obs_labels,
    [0, 0, 0],
    color=bar_colors,
)

# ax_bar.set_ylim(0, 1.05 * np.max(theta_dist_obs))
ax_bar.set_ylim(0, 1.05)

# ax_bar.set_ylabel(r"$\theta$ (Hermitian angle)")
ax_bar.set_ylabel(r"$\mu$")
ax_bar.set_title("Probabilité de présence instantanée")
ax_bar.grid(axis="y", alpha=0.3)

# -----------------------------------------------------------------------------
# Animation functions
# -----------------------------------------------------------------------------


def init():
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text("")
    for bar in bars:
        bar.set_height(0)
    return (line, point, time_text, *bars)


def update(frame):
    # Trajectory
    line.set_data(E[: frame + 1], N[: frame + 1])
    point.set_data([E[frame]], [N[frame]])

    # Time
    time_text.set_text(f"UTC : {pd.to_datetime(t[frame])}")

    # Update bars
    for i, bar in enumerate(bars):
        bar.set_height(theta_dist_obs[i][frame])

    return (line, point, time_text, *bars)


ani = FuncAnimation(
    fig,
    update,
    frames=len(E),
    init_func=init,
    interval=200,
    blit=True,
)

# -----------------------------------------------------------------------------
# Display / save
# -----------------------------------------------------------------------------

# plt.show()

fpath = os.path.join(img_folder, "trajectory_presence_probability_animation.gif")
ani.save(fpath, fps=10, dpi=150)
