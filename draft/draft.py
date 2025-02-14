import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


T = 100
fs = 1200
t = np.arange(0, T, 1 / fs)
nt = len(t)
s_e = np.random.normal(0, 1, nt)  # Unit var signal

nt = len(t)
S_f_event = np.fft.rfft(s_e)
f_event = np.fft.rfftfreq(nt, 1 / fs)

nperseg = 1024
print("size(t) = ", len(t))
dsp = sp.welch(s_e, fs, nperseg=nperseg, noverlap=nperseg // 2)
plt.figure()
plt.plot(dsp[0], 10 * np.log10(dsp[1]))
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [dB/Hz]")
plt.show()


for snr_dB in np.arange(-10, 5, 1):
    noise_da = derive_received_noise(
        s_library=ds_sig.s_l,
        s_event_rcv0=ds_sig.s_e.sel(idx_rcv=0),
        event_source=source,
        snr_dB=snr_dB,
    )
