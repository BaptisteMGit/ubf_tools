import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal

from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal,
    process_broadband,
)

from signals import pulse, ship_noise
from propa.kraken_toolbox.plot_utils import plotshd

img_path = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\generation_signaux"
)
figsize = (16, 8)
label_fontsize = 26

working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\test_synthesis"
template_env = "CalibSynthesis"
# template_env = "MunkK"

max_depth = 100

os.chdir(working_dir)

T = 7.2
fc = 25
fs = 100
window = "hanning"
# window = None

# Receiver position
rcv_depth = [20]
rcv_range = np.array([20000])
delays = rcv_range / 1500

delay = list([rcv_range[0] / 1500]) * len(rcv_range)
s, t = ship_noise(T=T)

plt.figure(figsize=figsize)
plt.plot(t, s, color="b", label=r"$s(t)$")
plt.xlabel("Time [s]", fontsize=label_fontsize)
plt.ylabel("Amplitude", fontsize=label_fontsize)
plt.xlim(0, 5)
plt.yticks(fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)

plt.tight_layout()
plt.savefig(os.path.join(img_path, f"src_ship.png"))

# s, t = pulse(T=T, f=fc, fs=fs, t0=0)
s, t = ship_noise(T=T)
source = AcousticSource(s, t, name="source", waveguide_depth=max_depth, window=window)
# process_broadband(fname=template_env, source=source, max_depth=max_depth)

plt.figure(figsize=figsize)
plotshd(os.path.join(working_dir, template_env + ".shd"), freq=40)
plt.scatter(rcv_range, rcv_depth, marker=">", c="k", s=50)
plt.yticks(fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(img_path, f"src_shd.png"))


time_vector, s_at_rcv_pos, Pos = postprocess_received_signal(
    shd_fpath=os.path.join(working_dir, template_env + ".shd"),
    source=source,
    rcv_range=rcv_range,
    rcv_depth=rcv_depth,
    apply_delay=True,
    delay=delay,
)

plt.figure(figsize=figsize)
plt.plot(time_vector, s_at_rcv_pos[:, 0, 0], color="b", label=r"$s_r(t)$")

plt.legend()
plt.xlabel("Time (s)", fontsize=label_fontsize)
plt.ylabel("Received signal", fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.xlim(0, 7)
plt.tight_layout()
plt.savefig(os.path.join(img_path, f"src_ship_propagated_1.png"))
# plt.show()
