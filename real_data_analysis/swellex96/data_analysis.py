import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sp

from publication.PublicationFigure import PubFigure

PubFigure()


nperseg = 2**14
noverlap = 2**13
fs = 3276.8
ts = 1 / fs
fmax = 400  # Max frequency emmited

data = {
    "S5": {"south": {}},
    # "S5": {"south": {}, "north": {}},
    # "S59": {"south": None, "north": None},
}
data_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\SwellEx96"
name_root = "J1312340.hla"
for event in data.keys():
    for side in data[event].keys():
        fname = f"{name_root}.{side}.mat"
        fpath = os.path.join(data_root, event, fname)
        xdata = sio.loadmat(fpath)["xdata"]
        data[event][side]["sig"] = xdata[:, 0:2]

        # Derive stft
        ff, tt, stft_xdata = sp.stft(
            xdata, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, axis=0
        )
        data[event][side]["stft"] = stft_xdata

nt = data["S5"]["south"]["sig"].shape[0]


t = np.arange(0, nt * ts, ts)

# Select channel to plot
ch = 0
for event in data.keys():
    plt.figure()
    for side in data[event].keys():
        plt.plot(t, data[event][side]["sig"][:, ch], label=side)
        plt.xlabel(r"$t \textrm{[s]}$")
        plt.title(event)
    plt.legend()


# Plot spectro
for event in data.keys():
    for side in data[event].keys():
        plt.figure()
        plt.ylim([0, fmax])
        plt.pcolormesh(
            tt,
            ff,
            np.abs(data[event][side]["stft"][:, ch, :]),
            shading="gouraud",
            vmin=0,
            vmax=100,
        )
        plt.xlabel(r"$t \textrm{[s]}$")
        plt.ylabel(r"$f \textrm{[Hz]}$")
        plt.title(f"{event} - {side}")
        plt.colorbar()

plt.show()
