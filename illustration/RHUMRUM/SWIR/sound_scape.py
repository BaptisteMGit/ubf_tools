import os
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from publication.PublicationFigure import PubFigure

PFIG = PubFigure(
    # label_fontsize=25,
    # title_fontsize=25,
    # ticks_fontsize=25,
    # legend_fontsize=20,
)

fs = 100
nperseg = 2**12
noverlap = int(nperseg // 2)

root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\wav\RHUMRUM"

data = {
    "fpaths": [],
    "vars": ["BDH"],
    "stft": [],
    "tt": [],
    "freq": [],
    "signal": [],
}

for chnl in data["vars"]:
    fname = f"signal_{chnl}_RR44_2013-05-31.wav"
    data["fpaths"].append(os.path.join(root_data, fname))
    fs, sig = wavfile.read(data["fpaths"][-1])

    # Normalize data
    sig = sig / np.max(sig)
    data["signal"].append(sig)

    f, tt, stft = sp.stft(sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    data["stft"].append(stft)

data["freq"] = f
data["tt"] = tt
time = np.arange(0, len(sig) * 1 / fs, 1 / fs)
data["time"] = time


vmin = -100
vmax = 0

# Plot spectrograms
plt.figure(figsize=(14, 5))
im = plt.pcolormesh(
    data["tt"] / 3600,
    data["freq"],
    20 * np.log10(abs(data["stft"][0])),
    # shading="gouraud",
    vmin=vmin,
    vmax=vmax,
)

# Convert time to hours
# xticks = np.arange(0, data["tt"][-1], 3600)
# plt.xticks(xticks, [f"{int(x/3600)}" for x in xticks])

plt.xlabel("Time [h]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()
# plt.colorbar(im, label="Magnitude [dB]")
plt.show()
