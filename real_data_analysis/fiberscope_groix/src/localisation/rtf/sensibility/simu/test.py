import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

# Test
fs = 2000
f0 = 5
omega0 = 2 * np.pi * f0
ts = 1 / fs
# t = np.arange(0, 3, ts)
# x = np.sin(omega0 * t)
# tau = 1 / f0 * 0.25
# y = np.sin(omega0 * (t - tau))

t = np.arange(0, 3, ts)
x = np.zeros_like(t)
win_idx = np.logical_and(t >= 0.5, t <= 1.5)
x[win_idx] = 1
tau = 1 / f0 * 0.25
win_idx = np.logical_and((t - tau) >= 0.5, (t - tau) <= 1.5)
y = np.zeros_like(t)
y[win_idx] = 1

fig, axs = plt.subplots(nrows=2)
axs[0].plot(t, x, label="x", color="b")
axs[0].plot(t, y, label=f"y (tau = {tau}s)", color="r")
axs[0].legend()
axs[0].set_xlabel("t [s]")

x_fft = np.fft.fft(x, axis=0)
y_fft = np.fft.fft(y, axis=0)
s_xy = x_fft * np.conj(y_fft)
c_xy = np.fft.ifft(s_xy)
c_xy = np.fft.fftshift(np.real(c_xy))
lags = sp.correlation_lags(x.size, y.size, mode="same") * ts


axs[1].plot(lags, c_xy, label="c_xy", color="k", marker="*")
axs[1].set_xlabel(r"$\tau$ [s]")

plt.show()
