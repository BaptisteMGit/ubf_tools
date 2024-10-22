#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   signals.py
@Time    :   2024/07/08 09:13:43
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Define signals for underwater acoustics.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.io as sio
import scipy.signal as sp


def pulse(T, f, fs, t0=0):
    """Generate pulse defined in Jensen et al. (2000)"""
    t = np.arange(0, T, 1 / fs)
    s = np.zeros(len(t))
    idx_tpulse = np.logical_and(0 < t - t0, t - t0 < 4 / f)
    t_pulse = t[idx_tpulse] - t0
    omega = 2 * np.pi * f
    s[idx_tpulse] = (
        1 / 2 * np.sin(omega * t_pulse) * (1 - np.cos(1 / 4 * omega * t_pulse))
    )

    # Normalize to 1
    s /= np.max(np.abs(s))

    return s, t


def pulse_train(T, f, fs, interpulse_delay=None):
    """Generate train of pulses"""
    pulse_duration = 4 / f
    if interpulse_delay is None:
        interpulse_delay = 0.5 * pulse_duration

    omega = 2 * np.pi * f
    t_train = np.arange(0, T, 1 / fs)
    s_train = np.zeros(len(t_train))
    nb_motif = int(np.ceil(T / (interpulse_delay + pulse_duration)))
    for i in range(nb_motif):
        t_pulse = t_train - i * (interpulse_delay + pulse_duration)
        s_pulse = np.zeros(len(t_pulse))

        idx_tpulse = np.logical_and(0 < t_pulse, t_pulse < pulse_duration)
        t_pulse = t_pulse[idx_tpulse]
        s_pulse[idx_tpulse] = (
            1 / 2 * np.sin(omega * t_pulse) * (1 - np.cos(1 / 4 * omega * t_pulse))
        )
        s_train += s_pulse

    # Normalize to 1
    s_train /= np.max(np.abs(s_train))

    return s_train, t_train


def sine_wave(f0, fs, T, A, phi):
    if fs < 2 * f0:
        raise ValueError(
            "Sampling frequency must be at least twice the frequency of the signal"
        )

    t = np.arange(0, T, 1 / fs)
    y = A * np.sin(2 * np.pi * f0 * t + phi)

    return y, t


def ship_noise(T):
    fpath = r"C:\Users\baptiste.menetrier\Desktop\ressource\ShipNoise_obs_SamuelPinson\ship_source_signal.mat"
    ship_d = sio.loadmat(fpath)
    t = ship_d["t"].squeeze()
    s = ship_d["sig_s_t"].squeeze()

    # Normalize to 1
    s /= np.max(np.abs(s))

    fs = 1 / (t[1] - t[0])
    nmax = int(fs * T)
    s = s[0:nmax]
    t = t[0:nmax]

    return s, t


def generate_ship_signal(
    Ttot,
    f0,
    std_fi=None,
    tau_corr_fi=None,
    fs=100,
    Nh=None,
    A_harmonics=None,
    normalize="max",
):

    if std_fi is None:
        std_fi = f0 * 1 / 100
    if tau_corr_fi is None:
        tau_corr_fi = 1 / f0

    # source signal parameters
    if Nh is None:
        Nh = int(np.floor(fs / 2 / f0) - 1)
    if A_harmonics is None:
        A_harmonics = np.ones(Nh)

    # signal variables
    t = np.arange(0, Ttot, 1 / fs)
    f = np.arange(0, fs, 1 / Ttot)
    Nt = len(t)

    # random instant frequency perturbation delta_fi with Gaussian power spectrum
    freq = np.fft.fftfreq(len(t), 1 / fs)
    fi_perturbation_psd = np.zeros_like(freq)
    fi_perturbation_psd = (
        np.sqrt(2 * np.pi)
        * tau_corr_fi
        * std_fi**2
        * np.exp(-2 * (np.pi * freq * tau_corr_fi) ** 2)
    )

    noise_phase = np.random.randn(len(t))
    random_phase = np.fft.fft(noise_phase)
    delta_fi = np.fft.ifft(
        random_phase * np.sqrt(fi_perturbation_psd * fs)
    )  # 19/09/2024 for clarity
    delta_ph = (
        np.cumsum(delta_fi) / fs
    )  # random instant phase perturbation (for the fs coef check comment la FFT est écrite pour avoir la bonne amplitude)

    # Derive ship signal from harmonics
    s = np.zeros_like(t, dtype=complex)
    for k in range(1, Nh + 1):
        s += (
            A_harmonics[k - 1]
            * ship_spectrum(f0 * k)
            * np.exp(1j * 2 * np.pi * k * (f0 * t + delta_ph))
        )

    # Real
    s = s.real
    if normalize == "max":
        # Normalize to 1
        s /= np.max(np.abs(s))
    elif normalize == "var":
        # Normalize to unit variance
        s /= np.std(s)

    return s, t


def ship_spectrum(f):
    f = np.array(f)
    fc = 15
    # fc = 20
    Q = 2
    Aship = 1 / (1 - f**2 / fc**2 + 1j * f / fc / Q)
    return Aship


def ricker_pulse(fc, fs, T, t0=0, center=True):
    """Ricker pulse"""
    t = np.arange(0, T, 1 / fs)
    if center:
        t0 = t.max() / 2

    s = (1 - 2 * (np.pi * fc * (t - t0)) ** 2) * np.exp(-((np.pi * fc * (t - t0)) ** 2))

    # Normalize to 1
    s /= np.max(np.abs(s))

    return s, t


def z_call(signal_args={}, model_args={}):
    """
    Z-call signal according to
    Socheleau, F.-X., Leroy, E., Carvallo Pecci, A., Samaran, F., Bonnel, J., & Royer, J.-Y. (2015). Automated detection of Antarctic blue whale calls. The Journal of the Acoustical Society of America, 138(5), 3105–3117. https://doi.org/10.1121/1.4934271
    The default parameters are the one proposed for z-calls recorded by the RHUM RUM array by :
    Bouffaut, L., Dréo, R., Labat, V., Boudraa, A.-O., & Barruol, G. (2018). Passive stochastic matched filter for Antarctic blue whale call detection. The Journal of the Acoustical Society of America, 144(2), 955–965. https://doi.org/10.1121/1.5050520

    Adapted from the original Matlab code provided by L. Bouffaut.

    About SL, depth and ICI :
    Bouffaut, L., Landrø, M., & Potter, J. R. (2021).
    Source level and vocalizing depth estimation of two blue whale subspecies in the western Indian Ocean from single sensor observations.
    The Journal of the Acoustical Society of America, 149(6), 4422–4436. https://doi.org/10.1121/10.0005281

    The SL and depth were estimated for the ABW at 188.5 +/- 2.1 dB and 25.0 +/- 3.7m
    ICI = 66.5 s

    Parameters
    ----------
    Tz : float
        Duration of the z-call signal in seconds.
    L : float
        Lower asymptote in Hz.
    U : float
        Upper asymptote in Hz.
    M : float
        Time at which the frequency is at the middle of the slope.
    alpha : float
        Slope of the Z-call.
    fc : float
        Central frequency of the Z-call.
    fs : int
        Sampling frequency.

    """

    # Unpack parametric model params
    fc = model_args.get("fc", 22.6)  # Central frequency of the Z-call.
    Tz = model_args.get("Tz", 20)  # Duration of a single z-call signal in seconds.
    L = model_args.get("L", -4.5)  # Lower asymptote in Hz.
    U = model_args.get("U", 3.2)  # Upper asymptote in Hz.
    M = model_args.get(
        "M", Tz / 2
    )  # Time at which the frequency is at the middle of the slope.
    alpha = model_args.get("alpha", 1.8)  # Slope of the Z-call.
    ici = model_args.get("ici", 66.5)  # Inter-Call Interval.

    # Unpack signal params
    fs = signal_args.get("fs", 100)  # Sampling frequency.
    nz = signal_args.get("nz", 1)  # Number of z-calls.
    start_offset_seconds = signal_args.get(
        "start_offset_seconds", 10
    )  # Delay before the first z-call.
    stop_offset_seconds = signal_args.get(
        "stop_offset_seconds", 10
    )  # Delay after the last z-call.

    # 1) Generate a single Z-call signal
    tz = np.arange(0, Tz, 1 / fs)  # axe du temps
    ns = len(tz)  # nombre d'échantillons

    fc = 22.6  # (Hz) fréquence centrale du Z-call
    alpha = 1.8  # pente

    L = -4.5  # (Hz) asymptote inférieure
    U = 3.2  # (Hz) asymptote supérieure
    M = Tz / 2  # (s) temps à la moitié de la pente

    # Estimation de la phase variable dans le temps
    adj = fc - 8.5  # Ajustement de la fréquence du Z-call
    L = L - adj  # Asymptote inférieure ajustée (Hz)
    U = U - adj  # Asymptote supérieure ajustée (Hz)

    # Calcul de la phase
    n = np.arange(ns)  # Axe des échantillons
    phase_whale = (
        2
        * np.pi
        * (
            L * n / fs
            + ((U - L) / alpha)
            * np.log((1 + np.exp(-alpha * M)) / (1 + np.exp(alpha * (n / fs - M))))
        )
    )
    phase_whale = phase_whale[::-1]  # Inverser dans le temps

    # Signal temporel
    single_z_call = np.exp(1j * phase_whale)
    single_z_call = np.real(single_z_call / np.max(np.abs(single_z_call)))

    # Variation d'amplitude dans le temps
    # amplitude = np.concatenate([np.ones(round(ns / 2)), np.ones(round(ns / 2)) * 0.95])
    amplitude = sp.windows.tukey(ns, alpha=0.2)
    single_z_call = amplitude * single_z_call
    single_z_call = single_z_call / np.max(np.abs(single_z_call))

    # 2) Generate desired signals containing nz z-calls separated by ICI = 66.5 s
    t_max = (
        start_offset_seconds + Tz * nz + ici * nz + stop_offset_seconds
    )  # Signal total duration
    t = np.arange(0, t_max, 1 / fs)
    s_whale = np.zeros(len(t))

    for i in range(nz):
        idx_start = int((start_offset_seconds + i * (Tz + ici)) * fs)
        idx_stop = int((start_offset_seconds + i * (Tz + ici) + Tz) * fs)
        s_whale[idx_start:idx_stop] = single_z_call

    return s_whale, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # s, t = ricker_pulse(fc=200, T=0.05, fs=20 * 200, center=True)
    # plt.figure()
    # plt.plot(t, s)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

    # # Exemple d'utilisation de la fonction
    # Ttot = 50  # Durée totale du signal en secondes
    # f0 = 0.5  # Fréquence fondamentale
    # std_fi = f0 * 1 / 100  # Écart-type de la fluctuation de fréquence
    # tau_corr_fi = 1 / f0  # frequency fluctuation correlation time

    # std_fi_1 = f0 * 10 / 100  # Écart-type de la fluctuation de fréquence

    # std_fi_2 = f0 * 1 / 100  # Écart-type de la fluctuation de fréquence
    # tau_corr_fi_2 = 0.0001 / f0  # frequency fluctuation correlation time

    # signal, time = generate_ship_signal(Ttot, f0, std_fi)
    # signal_1, time = generate_ship_signal(Ttot, f0, std_fi_1)
    # signal_2, time = generate_ship_signal(Ttot, f0, std_fi_2, tau_corr_fi_2)

    # plt.figure()
    # plt.plot(time, signal, label=f"std_fi={std_fi}")
    # plt.plot(time, signal_1, label=f"std_fi={std_fi_1}")
    # plt.plot(time, signal_2, label=f"std_fi={std_fi_2}, tau_corr_fi={tau_corr_fi_2}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()

    signal_args = {
        "fs": 100,
        "nz": 10,
        "start_offset_seconds": 5,
        "stop_offset_seconds": 15,
    }
    s, t = z_call(signal_args)
    plt.figure()
    plt.plot(t, s)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Derive stft
    import scipy.signal as sp

    nperseg = 2**9
    noverlap = int(0.8 * nperseg)
    print(f"nperseg = {nperseg}, noverlap = {noverlap}")
    f, t, stft = sp.stft(s, fs=100, nperseg=nperseg, noverlap=noverlap)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(stft))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Amplitude")

    plt.show()
