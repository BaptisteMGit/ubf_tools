#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   signal_generator.py
@Time    :   2025/04/01 10:18:30
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Define class to handle signal generation and analysis
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.signal as sp
import scipy.special as special
import matplotlib.pyplot as plt

import source.global_constants as g

# ======================================================================================================================
# Class
# ======================================================================================================================


class SignalGenerator:
    """
    Class to handle signal generation and analysis
    """

    def __init__(self):
        pass

    @classmethod
    def colored_noise(cls, T, fs, noise_color="white"):
        """
        Generate colored noise

        Parameters
        ----------
        T : float
            Duration of the time serie
        fs : float
            Sampling frequency
        noise_color : str, optional
            Color of the noise.

        Returns
        -------
        t : np.array
            Time vector
        x_t : np.array
            Time serie
        """

        target_nt = int(T * fs)

        # Ensure nt is even
        if target_nt % 2 != 0:
            nt = target_nt + 1
        else:
            nt = target_nt

        # The number number of fft points if nt and  we have nt/2+1 points such that f>=0
        nf = nt // 2 + 1
        k = np.arange(
            1,
            nf + 1,
        )
        if noise_color == "white":
            # psd = np.ones(f.shape)
            alpha_psd = 0
        elif noise_color == "pink":
            # psd = 1 / f
            alpha_psd = -1
        elif noise_color == "brown":
            # psd = 1 / f**2
            alpha_psd = -2
        elif noise_color == "blue":
            # psd = f
            alpha_psd = 1
        elif noise_color == "purple":
            # psd = f**2
            alpha_psd = 2
        else:
            raise ValueError("Unknown noise color")

        # Convert psd slope into amplitude spectrum slope given that Sxx(f) = |X(f)|^2
        # We wish to get Sxx(f) = f ** alpha_psd which is equivalent to |X(f)| = f ** (alpha_psd / 2)
        alpha_spec = alpha_psd / 2

        # Generate WGN spectrum
        X_f = np.random.randn(nf) + 1j * np.random.randn(nf)

        # Apply the desired correction to get the right spectrum slope
        X_f = X_f * (k**alpha_spec)

        # Apply inverse fft to get the time signal
        x_t = np.fft.irfft(X_f, n=nt)

        # Normalize to unit variance and zero mean
        x_t = (x_t - np.mean(x_t)) / np.std(x_t)

        # Remove extra point if nt was odd
        if x_t.shape[0] > target_nt:
            x_t = x_t[:-1]  # Drop last point

        # Build associated time vector
        t = np.arange(0, target_nt) / fs

        return t, x_t

    @staticmethod
    def psd_to_timeserie(psd, df):
        """
        Generate a time series from a given power spectral density (PSD) using the inverse Fourier transform.
        Parameters
        ----------
        psd : np.ndarray
            Power spectral density (PSD) of the signal containing the psd components for positive frequencies (f=np.fft.rfftfreq(nt, 1/fs)).
        df : float
            Frequency resolution of the PSD.
        Returns
        -------
        t : np.ndarray
            Time vector of the signal.
        x_t : np.ndarray
            Time series generated from the PSD.
        """

        # Number of frequency components in psd
        nf = psd.shape[0]

        # Define module of the spectrum
        X_f_mod = np.sqrt(psd)  # Definition of PSD Sxx(f) = |X(f)|^2

        # Generate random phase to create the complexe spectrum
        phi_t = np.random.randn(2 * nf - 1)
        X_f_ang = np.angle(np.fft.rfft(phi_t))

        # Use random phase to create spectrum
        X_f = X_f_mod * np.exp(1j * X_f_ang)

        # Inverse fourier transform to get time signal
        x_t = np.fft.irfft(X_f)
        nt = x_t.shape[0]

        # Correct for rfft factor
        x_t *= nt * np.sqrt(df / 2)

        # Time vector
        fs = nt * df
        t = np.linspace(0, 1 / fs * (nt - 1), nt)

        return t, x_t

    @classmethod
    def z_call(cls, signal_args={}, model_args={}):
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
        ICI = 66.4 s

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
        ici = model_args.get("ici", 66.4)  # Inter-Call Interval.

        # Unpack signal params
        fs = signal_args.get("fs", 100)  # Sampling frequency.
        nz = signal_args.get("nz", 1)  # Number of z-calls.
        start_offset_seconds = signal_args.get(
            "start_offset_seconds", 10
        )  # Delay before the first z-call.
        stop_offset_seconds = signal_args.get(
            "stop_offset_seconds", 10
        )  # Delay after the last z-call.
        signal_duration = signal_args.get(
            "signal_duration", None
        )  # Duration of the signal.
        sl = signal_args.get("sl", 188.5)  # Source level in dB re 1 µPa @ 1m.

        # 1) Generate a single Z-call signal
        tz = np.arange(0, Tz, 1 / fs)  # axe du temps
        ns = len(tz)  # nombre d'échantillons

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
        amplitude = sp.windows.tukey(ns, alpha=0.2)
        # amplitude = np.hanning(ns)
        # amplitude = 1
        single_z_call = amplitude * single_z_call
        single_z_call = single_z_call / np.max(np.abs(single_z_call))

        # Normalize z-call to the desired source level
        single_z_call = cls.normalize_to_sl(single_z_call, sl)

        # 2) Generate desired signals containing nz z-calls separated by ICI = 66.5 s
        # nz = 0 to get maximum number of z-calls in the signal duration
        if nz == 0 and signal_duration is not None:
            nz = int(
                np.ceil(
                    (signal_duration - start_offset_seconds - stop_offset_seconds)
                    / (Tz + ici)
                )
            )
            t_max = signal_duration
        elif nz != 0 and signal_duration is not None:
            t_max = signal_duration
        else:
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

    @staticmethod
    def normalize_to_sl(sig, sl):
        """Normalize source signal to match desired Source Level sl"""
        target_std = (
            10 ** (sl / 20) * g.p0
        )  # Target rms amplitude to reach the desired SL
        sig = sig * target_std / np.std(sig)

        return sig

    @classmethod
    def generate_ship_signal(
        cls,
        Ttot,
        f0,
        std_fi=None,
        tau_corr_fi=None,
        fs=100,
        Nh=None,
        A_harmonics=None,
        normalize="max",
        sl=None,
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
                * cls.ship_spectrum(f0 * k)
                * np.exp(1j * 2 * np.pi * k * (f0 * t + delta_ph))
            )

        # Real
        s = s.real
        # Normalize
        s = cls.normalize_sig(s, normalize, sl)

        return s, t

    @staticmethod
    def ship_spectrum(f):
        f = np.array(f)
        fc = 15
        # fc = 20
        Q = 2
        Aship = 1 / (1 - f**2 / fc**2 + 1j * f / fc / Q)
        return Aship

    # ======================================================================================================================
    # Impulsive signals
    # ======================================================================================================================

    @classmethod
    def normalize_sig(cls, sig, normalize, sl=None):
        if normalize == "max":
            # Normalize to 1
            sig /= np.max(np.abs(sig))
        elif normalize == "var":
            # Normalize to unit variance
            sig /= np.std(sig)
        elif normalize == "sl" and sl is not None:
            # Normalize to desired source level
            sig = cls.normalize_to_sl(sig, sl)
        return sig

    @classmethod
    def dirac(
        cls,
        fs,
        T,
        t0=0,
        center=True,
        normalize="max",
        sl=None,
    ):

        t = np.arange(0, T, 1 / fs)
        if center:
            t0 = t.max() / 2
        idx_dirac = np.argmin(np.abs(t - t0))
        s = np.zeros(len(t))
        s[idx_dirac] = 1

        # Normalize
        s = cls.normalize_sig(s, normalize, sl)

        return s, t

    @classmethod
    def ricker_pulse(
        cls,
        fc,
        fs,
        T,
        t0=0,
        center=True,
        normalize="max",
        sl=None,
    ):
        """Ricker pulse"""
        t = np.arange(0, T, 1 / fs)
        if center:
            t0 = t.max() / 2

        s = (1 - 2 * (np.pi * fc * (t - t0)) ** 2) * np.exp(
            -((np.pi * fc * (t - t0)) ** 2)
        )

        # Normalize
        s = cls.normalize_sig(s, normalize, sl)

        return s, t

    @staticmethod
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

    @staticmethod
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

    # ======================================================================================================================
    # Chirp signals
    # ======================================================================================================================

    @staticmethod
    def lfm_chirp(f0, f1, fs, T, phi=0):
        """LFM chirp signal"""
        t = np.arange(0, T, 1 / fs)
        s = np.cos(2 * np.pi * (f0 + (f1 - f0) / (2 * T) * t) * t + phi)

        return s, t

    @staticmethod
    def lfm_chirp_train(f0, f1, fs, T_chirp, T, interpulse_delay=None, start_delay=0):

        if interpulse_delay is None:
            interpulse_delay = 0.5 * T_chirp

        t_chirp = np.arange(0, T_chirp, 1 / fs)
        s0 = sp.chirp(t=t_chirp, f0=f0, f1=f1, t1=T_chirp, method="linear")
        pad_before_t = np.arange(0, start_delay, 1 / fs)
        pad_after_t = np.arange(0, T - (T_chirp + start_delay), 1 / fs)
        s0 = np.pad(s0, (len(pad_before_t), len(pad_after_t)))

        n_lfm = int(np.ceil((T - start_delay) / (interpulse_delay + T_chirp)))
        s = np.zeros_like(s0)
        t = np.arange(0, T, 1 / fs)
        shift = int((interpulse_delay + T_chirp) * fs)
        print(shift * 1 / fs)
        for i in range(n_lfm):
            si = np.roll(s0, shift=i * shift)
            s += si

        return s, t

    @staticmethod
    def lfm_chirp_analytic_spectrum(alpha, freq):
        """LFM chirp spectrum analytic expression from [1] Eq.10 p.11"""
        S_f = (
            np.sqrt(np.pi / alpha)
            * np.exp(1j * np.pi / 4)
            * np.exp(1j * (2 * np.pi * freq) ** 2 / (4 * alpha))
        )
        return freq, S_f

    @classmethod
    def pulse_lfm_chirp_analytic_spectrum(cls, alpha, freq, Tp):
        """Pulse LFM chirp spectrum analytic expression from [1] Eq.23 p.14"""

        omega = 2 * np.pi * freq
        a = np.sqrt(np.pi / (2 * alpha))
        b = np.exp(-1j * omega**2 / (4 * alpha))
        c = cls.F_fresnel((omega + alpha * Tp) / (2 * np.sqrt(alpha))) - cls.F_fresnel(
            (omega - alpha * Tp) / (2 * np.sqrt(alpha))
        )
        S_f = a * b * c
        return freq, S_f

    @staticmethod
    def F_fresnel(z):
        S, C = special.fresnel(z)
        F = C + 1j * S
        return F

    # ======================================================================================================================
    # Plot functions
    # ======================================================================================================================

    @staticmethod
    def plot_signal(t, s, title="Signal", xlabel="Time [s]", ylabel="Amplitude"):
        """Plot signal"""
        plt.figure()
        plt.plot(t, s)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    @staticmethod
    def plot_spectrum(
        f, S_f, title="Spectrum", xlabel="Frequency [Hz]", ylabel="Amplitude"
    ):
        """Plot spectrum"""
        plt.figure()
        plt.plot(f, np.abs(S_f))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    @staticmethod
    def plot_spectrogram(
        t,
        f,
        S_tf,
        ax=None,
        title="Spectrogram",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    ):
        """Plot spectrogram"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(t, f, np.abs(S_tf))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        # plt.show()

    @staticmethod
    def plot_signal_spectrogram(
        t,
        f,
        s,
        S_tf,
        title="Signal and Spectrogram",
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
    ):
        """Plot signal and spectrogram"""
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, s)
        plt.title("Signal")
        plt.xlabel(xlabel)
        plt.ylabel("Amplitude")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.pcolormesh(t, f, np.abs(S_tf))
        plt.title("Spectrogram")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    @staticmethod
    def plot_psd(
        f,
        psd,
        title="Power Spectral Density",
        xlabel="Frequency [Hz]",
        ylabel="Amplitude [dB]",
    ):
        """Plot Power Spectral Density"""
        plt.figure()
        plt.plot(f, 10 * np.log10(psd))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Examples
    fs = 1200
    T = 10
    sg = SignalGenerator()
    t, x, f_, psd_ = sg.colored_noise(T, fs, noise_color="blue")

    # Plot time serie
    sg.plot_signal(t, x)

    # Derive and plot psd
    f, psd = sp.welch(x, fs=fs, nperseg=1024, noverlap=512)

    sg.plot_psd(
        f,
        psd,
        title="Power Spectral Density",
        xlabel="Frequency [Hz]",
        ylabel="Amplitude [dB]",
    )

    # Compare to target psd
    plt.figure()
    plt.plot(f, 10 * np.log10(psd), label="reached")
    plt.plot(f_, 10 * np.log10(psd_), label="target")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    # plt.show()

    # Compute and plot fft
    ff = np.fft.rfftfreq(len(x), 1 / fs)
    X_f = np.fft.rfft(x)

    plt.figure()
    plt.plot(ff, np.abs(X_f))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
