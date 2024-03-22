import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
import xarray as xr

from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.post_process import (
    postprocess,
    process_broadband,
)

from tqdm import tqdm
from colorama import Fore

working_dir = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis"
)
# template_env = "MunkK"
template_env = "CalibSynthesis"
os.chdir(working_dir)


# Define ship trajecory
x_ship_begin = -3000
y_ship_begin = 1000
x_ship_end = 3000
y_ship_end = 0

z_ship = 5

v_ship = 50 / 3.6

Dtot = np.sqrt((x_ship_begin - x_ship_end) ** 2 + (y_ship_begin - y_ship_end) ** 2)

vx = v_ship * (x_ship_end - x_ship_begin) / Dtot
vy = v_ship * (y_ship_end - y_ship_begin) / Dtot

Ttot_ship = Dtot / v_ship + 3
T_one_pos = 20
t_ship = np.arange(0, Ttot_ship - T_one_pos, T_one_pos)
npos_ship = t_ship.size

x_ship_t = x_ship_begin + vx * t_ship
y_ship_t = y_ship_begin + vy * t_ship

# OBS position
x_obs = -1000
y_obs = 0

# Distance to receiver
r_ship_t = np.sqrt((x_ship_t - x_obs) ** 2 + (y_ship_t - y_obs) ** 2)

plt.figure()
plt.plot(t_ship, r_ship_t)
plt.xlabel("Time (s)")
plt.ylabel("Ship distance from receiver")
# plt.show()

# Define source signal
f0 = 15
fs = 8 * f0
t = np.arange(0, Ttot_ship, 1 / fs)
s = np.sin(2 * np.pi * f0 * t)
dt = t[1] - t[0]
source = AcousticSource(s, t)
source.z_src = z_ship
source.kraken_freq = np.array([f0 - 10, f0 - 5, f0, f0 + 5, f0 + 10])
source.display_source()

# process_broadband(fname=template_env, source=source, max_depth=300)
# # Plot pressure field
# plotshd(template_env + ".shd", 30)
# plt.scatter(
#     r_ship_t,
#     np.ones(r_ship_t.shape) * z_ship,
#     marker=">",
#     color="red",
#     s=20,
#     label="Receiver positions",
# )
# plt.legend()


# Receiver position
rcv_range = r_ship_t
rcv_depth = [source.z_src]

# Propagate signal from each ship position
n_per_pos = int(T_one_pos * fs) - 1
propa_overlap = 0.75

# full_s = np.empty((n_per_pos * npos_ship,))
# full_s = np.nan
full_s = []
full_t = []

bar_format = "%s{l_bar}%s{bar}%s{r_bar}%s" % (
    Fore.YELLOW,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.RESET,
)
for it in tqdm(
    range(1, npos_ship - 1),
    desc="Derive signal at each ship position",
    bar_format=bar_format,
):
    # tmin = it * T_one_pos
    # tmax = (it + 1) * T_one_pos
    # idx_t_pos = np.logical_and(t > tmin, t < tmax)
    idx_min = n_per_pos * it - int(n_per_pos * propa_overlap)
    idx_max = n_per_pos * (it + 1) + int(n_per_pos * propa_overlap)
    t_pos = t[idx_min:idx_max]
    s_pos = s[idx_min:idx_max]

    # Apply window to the extracted signal (to avoid discontinuities)
    win = signal.windows.hann(s_pos.size)
    s_pos = s_pos * win

    src_pos = AcousticSource(signal=s_pos, time=t_pos)
    t0 = time.time()
    time_vector, s_at_rcv_pos, Pos = postprocess(
        fname=template_env,
        source=src_pos,
        rcv_range=rcv_range[it : it + 1],
        rcv_depth=rcv_depth,
    )
    # c_time = time.time() - t0
    # print(f"postprocess c-time = {c_time}s")
    # print(f"Expected time = {c_time * npos_ship}s")
    # full_s.append(s_at_rcv_pos[:, 0, 0])
    if it == 1:
        full_t = time_vector
        full_s = s_at_rcv_pos[:, 0, 0]
    new_t = time_vector + max(full_t) + dt
    full_t = np.append(full_t, new_t)
    full_s = np.append(full_s, s_at_rcv_pos[:, 0, 0])

# Save data

ds = xr.Dataset(
    data_vars=dict(received_signal=(["time"], full_s)), coords=dict(time=full_t)
)
root_ds = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis"
)
filename_ds = f"doppler_test_{T_one_pos}s.nc"
ds.to_netcdf(os.path.join(root_ds, filename_ds))

# Plot full signal
plt.figure()
plt.plot(full_t, full_s)
plt.xlabel("Time (s)")

# Plot spectrogram of the received signal
nperseg = 512 * 2
overlap_window = 2 / 4
noverlap = int(nperseg * overlap_window)

f, t, Sxx = signal.spectrogram(
    full_s, source.fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
)

plt.figure()
plt.pcolormesh(t, f, 20 * np.log10(np.abs(Sxx)))
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.show()
