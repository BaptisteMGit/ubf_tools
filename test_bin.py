import numpy as np
import matplotlib.pyplot as plt

f0 = 2
fs = 2000
ts = 1 / fs
T = 10
t = np.arange(0, T, ts)

x = np.random.randn(t.size)
tau = 1.13
shift = int(tau / ts)
y = np.roll(x, shift=-shift)

plt.figure()
plt.plot(t, x, label="x")
plt.plot(t, y, label="y")
plt.legend()


import scipy.signal as sp

cxy = sp.correlate(x, y, mode="full")
tau = sp.correlation_lags(x.size, y.size, mode="full") * ts


n = x.size
nfft = 2 * n - 1  # Zero pad to compute full correlation

x_fft = np.fft.fft(x, n=nfft, axis=0)
y_fft = np.fft.fft(y, n=nfft, axis=0)

c_xy = np.fft.ifft(x_fft * np.conj(y_fft), axis=0)
c_xy = np.real(c_xy)
cxy_fft = np.fft.fftshift(c_xy, axes=0)

plt.figure()
plt.plot(tau, np.abs(cxy))
plt.plot(tau, np.abs(cxy_fft))

plt.show()
