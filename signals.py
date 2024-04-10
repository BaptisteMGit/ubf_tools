import numpy as np
import scipy.io as sio


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


def generate_ship_signal(Ttot, f0, std_fi=None, tau_corr_fi=None, fs=100):

    if std_fi is None:
        std_fi = f0 * 1 / 100
    if tau_corr_fi is None:
        tau_corr_fi = 1 / f0

    # source signal parameters
    Nh = int(np.floor(fs / 2 / f0) - 1)
    A_harmonics = np.ones(Nh)

    # signal variables
    t = np.arange(0, Ttot, 1 / fs)
    f = np.arange(0, fs, 1 / Ttot)
    Nt = len(t)

    # random instant frequency perturbation delta_fi with Gaussian power spectrum
    Gaussian = np.zeros_like(f)

    Gaussian[0 : int(np.floor(Nt / 2))] = (
        np.sqrt(2 * np.pi)
        * tau_corr_fi
        * std_fi**2
        * np.exp(-2 * (np.pi * f[0 : int(np.floor(Nt / 2))] * tau_corr_fi) ** 2)
    )
    Gaussian[-1 : int(np.floor(Nt / 2)) : -1] = np.conj(
        Gaussian[1 : int(np.ceil(Nt / 2))]
    )
    delta_fi = np.random.randn(len(t))
    delta_fi = np.fft.ifft(np.fft.fft(delta_fi) * np.sqrt(Gaussian * Nt / Ttot))
    delta_ph = np.cumsum(delta_fi) / fs  # random instant phase perturbation

    # Derive ship signal from harmonics
    s = np.zeros_like(t, dtype=complex)
    for l in range(1, Nh + 1):
        s += (
            A_harmonics[l - 1]
            * ship_spectrum(f0 * l)
            * np.exp(1j * 2 * np.pi * l * (f0 * t + delta_ph))
        )

    # Real
    s = s.real
    # Normalize to 1
    s /= np.max(np.abs(s))

    return s, t


def ship_spectrum(f):
    f = np.array(f)
    fc = 15
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # s, t = ricker_pulse(fc=200, T=0.05, fs=20 * 200, center=True)
    # plt.figure()
    # plt.plot(t, s)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.show()

    # Exemple d'utilisation de la fonction
    Ttot = 50  # Durée totale du signal en secondes
    f0 = 0.5  # Fréquence fondamentale
    std_fi = f0 * 1 / 100  # Écart-type de la fluctuation de fréquence
    tau_corr_fi = 1 / f0  # frequency fluctuation correlation time

    std_fi_1 = f0 * 10 / 100  # Écart-type de la fluctuation de fréquence

    std_fi_2 = f0 * 1 / 100  # Écart-type de la fluctuation de fréquence
    tau_corr_fi_2 = 0.0001 / f0  # frequency fluctuation correlation time

    signal, time = generate_ship_signal(Ttot, f0, std_fi)
    signal_1, time = generate_ship_signal(Ttot, f0, std_fi_1)
    signal_2, time = generate_ship_signal(Ttot, f0, std_fi_2, tau_corr_fi_2)

    plt.figure()
    plt.plot(time, signal, label=f"std_fi={std_fi}")
    plt.plot(time, signal_1, label=f"std_fi={std_fi_1}")
    plt.plot(time, signal_2, label=f"std_fi={std_fi_2}, tau_corr_fi={tau_corr_fi_2}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
