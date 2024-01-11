import os
import numpy as np
import matplotlib.pyplot as plt
from signals import pulse, pulse_train, ship_noise
from localisation.verlinden.AcousticComponent import AcousticSource

img_path = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\generation_signaux"
)

T = 7.2
fc = 50
fs = 200

sig = {}
time = {}
sig["pulse"], time["pulse"] = pulse(T=T, f=fc, fs=fs, t0=0.5 * T)
sig["pulse_train"], time["pulse_train"] = pulse_train(T=T, f=fc, fs=fs)
sig["ship"], time["ship"] = ship_noise(T=T)


depth = 150  # Verlinden test case
window = "hanning"

srcs = {}
for i, k in enumerate(sig.keys()):
    library_src = AcousticSource(
        signal=sig[k],
        time=time[k],
        name=k,
        waveguide_depth=depth,
        window=window,
        nfft=2 ** int(np.log2(sig[k].size) + 1),
    )
    srcs[k] = library_src

    library_src.display_source()

    if window is not None:
        fname = f"source_{k}_window_{window}.png"
    else:
        fname = f"source_{k}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, fname))

f, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

for i, k in enumerate(sig.keys()):
    srcs[k].plot_signal(ax=axs[i])
    axs[i].set_title(k)
    axs[i].set_xlabel("")
    axs[i].set_ylabel("")
    axs[i].set_ylim([-1.2, 1.2])
f.supxlabel("Time (s)")
f.supylabel("Amplitude")
plt.tight_layout()
plt.savefig(os.path.join(img_path, "signals_types.png"))
