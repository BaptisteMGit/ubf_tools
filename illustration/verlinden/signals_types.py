import os
import numpy as np
import matplotlib.pyplot as plt
from signals import pulse, pulse_train, ship_noise, generate_ship_signal
from localisation.verlinden.misc.AcousticComponent import AcousticSource

from publication.PublicationFigure import PubFigure

PFIG = PubFigure()

img_path = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\generation_signaux"
)


# Ship sig
T = 60 * 60
fc = 25
fs = 100
src_sig, t_src_sig = generate_ship_signal(
    Ttot=T,
    f0=5,
    std_fi=1 / 100,
    tau_corr_fi=1 / 100,
    fs=fs,
)

library_src = AcousticSource(
    signal=src_sig,
    time=t_src_sig,
    name="",
    waveguide_depth=150,
    # window="hamming",
    # nfft=2 ** int(np.log2(sig[k].size) + 1),
)
# library_src.display_source(plot_spectrum=False)

plt.figure()
plt.plot(t_src_sig / 3600, src_sig)
plt.xlabel("Time [h]")
plt.ylabel("Normalized amplitude")


# Spectro
import scipy.signal as sp

plt.figure()
freq, tt, stft = sp.stft(src_sig, fs=fs, window="hann", nperseg=512, noverlap=512 // 2)
plt.pcolormesh(
    tt / 3600,
    freq,
    20 * np.log10(abs(stft)),
    # shading="gouraud",
    vmin=-100,
    vmax=0,
)
plt.xlabel("Time [h]")
plt.ylabel("Frequency [Hz]")
# plt.show()

plt.figure()
idx_zoom = (t_src_sig > 12 * 60) & (t_src_sig < 12 * 60 + 10)
plt.plot(t_src_sig[idx_zoom], src_sig[idx_zoom])
plt.xlabel("Time [s]")
plt.ylabel("Normalized amplitude")

plt.show()

# T = 20
# fc = 25
# fs = 100

# sig = {}
# time = {}
# sig["pulse"], time["pulse"] = pulse(T=T, f=fc, fs=fs, t0=0.5 * T)
# # sig["pulse"], time["pulse"] = pulse(T=T, f=fc, fs=fs, t)
# sig["pulse_train"], time["pulse_train"] = pulse_train(T=T, f=fc, fs=fs)
# sig["ship"], time["ship"] = ship_noise(T=T)

# depth = 150  # Verlinden test case
# window = "hamming"
# # window = "hanning"
# window = None


# srcs = {}
# for i, k in enumerate(sig.keys()):
#     library_src = AcousticSource(
#         signal=sig[k],
#         time=time[k],
#         name=k,
#         waveguide_depth=depth,
#         window=window,
#         nfft=2 ** int(np.log2(sig[k].size) + 1),
#     )
#     srcs[k] = library_src

#     library_src.display_source(plot_spectrum=False)

#     if window is not None:
#         fname = f"source_{k}_window_{window}.png"
#     else:
#         fname = f"source_{k}.png"
#     plt.tight_layout()
#     plt.ylabel("PSD")
#     plt.savefig(os.path.join(img_path, fname))

# f, axs = plt.subplots(3, 1, sharex=True)

# for i, k in enumerate(sig.keys()):
#     srcs[k].plot_signal(ax=axs[i])
#     axs[i].set_title("")
#     axs[i].set_xlabel("")
#     axs[i].set_ylabel("")
#     axs[i].set_ylim([-1.2, 1.2])
# f.supxlabel("Time (s)")
# f.supylabel("Amplitude")
# plt.tight_layout()
# plt.savefig(os.path.join(img_path, "signals_types.png"))
