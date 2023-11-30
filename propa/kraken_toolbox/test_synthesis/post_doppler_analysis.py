import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal

dopp_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis\doppler_test_0.5s"
ds = xr.open_dataset(dopp_path)

# Plot full signal
ds.received_signal.plot()

fs = 1 / ds.time.diff(dim="time")[0].values
# Plot spectrogram of the received signal
nperseg = 512 * 2
overlap_window = 2 / 4
noverlap = int(nperseg * overlap_window)

f, t, Sxx = signal.spectrogram(
    ds.received_signal, fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
)

plt.figure()
plt.pcolormesh(t, f, 20 * np.log10(np.abs(Sxx)), cmap="jet")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar()
plt.show()
print()
