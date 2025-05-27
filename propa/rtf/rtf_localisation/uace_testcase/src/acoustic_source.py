#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ship_signal.py
@Time    :   2025/05/06 08:56:24
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle ship signal properties
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sp
from source.signal_generator import SignalGenerator

# import propa.rtf.rtf_localisation.uace_testcase.src.params as p


class AcousticSource:
    """Mother class for sources used in simulation"""

    def __init__(
        self,
        name: str,
        fs: float,
        duration: float,
        root_img: str = "",
        x: float = None,
        y: float = None,
        z: float = None,
    ):
        """
        Constructor
        :param name: Name of the source
        :param fs: Sampling frequency in Hz
        :param duration: Duration of the signal in seconds
        :param root_img: Path to the folder where to save the images
        :param x: X coordinate of the source in meters
        :param y: Y coordinate of the source in meters
        :param z: Z coordinate of the source in meters
        """
        self.name = name
        self.fs = fs
        self.ts = 1 / fs
        self.duration = duration

        # Position
        self.x = x
        self.y = y
        self.z = z

        # Ensure folder exists to store images
        if not os.path.exists(root_img):
            os.makedirs(root_img)
        self.root_img = root_img

        # Init signal generator
        self.sg = SignalGenerator()

        # Init usefull properties
        self.time = None
        self.signal = None
        self.n_samples = None
        self.freq = None
        self.spectrum = None

    def get_signal(self):
        pass

    def get_spectrum(self):
        """Get the signal frequency spectrum"""

        # Assert signal exists
        if self.signal is None:
            raise ValueError(
                "Signal not generated yet. Please call get_signal() first."
            )
        # Compute the spectrum
        self.spectrum = np.fft.rfft(self.signal)
        self.freq = np.fft.rfftfreq(self.n_samples, self.ts)

    def get_stft(self, window="hann", nperseg=2**8, noverlap=2**7):
        """Get the signal STFT"""
        # Assert signal exists
        if self.signal is None:
            raise ValueError(
                "Signal not generated yet. Please call get_signal() first."
            )

        ff, tt, Sxx = sp.stft(
            self.signal, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap
        )
        return ff, tt, Sxx

    def get_psd(self, window="hann", nperseg=2**8, noverlap=2**7):
        """Get the signal PSD"""
        # Assert signal exists
        if self.signal is None:
            raise ValueError(
                "Signal not generated yet. Please call get_signal() first."
            )

        f, Pxx = sp.welch(
            self.signal, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap
        )
        return f, Pxx

    def plot_signal(self, tmin=0, tmax=None, ax=None):
        """Plot the signal"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.time, self.signal)
        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid()

        if tmax is not None:
            ax.set_xlim(tmin, tmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_signal.png",
            dpi=300,
        )

    def plot_spectrum(self, fmin=0, fmax=None, ax=None):
        """Plot signal spectrum"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.freq, np.abs(self.spectrum))
        ax.set_title(self.name)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")
        ax.grid()

        if fmax is not None:
            ax.set_xlim(fmin, fmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_spectrum.png",
            dpi=300,
        )

    def plot_stft(
        self,
        window="hann",
        nperseg=2**8,
        noverlap=2**7,
        tmin=0,
        tmax=None,
        fmin=0,
        fmax=None,
        ax=None,
    ):
        """Plot the signal STFT"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ff, tt, sxx = self.get_stft(window=window, nperseg=nperseg, noverlap=noverlap)

        im = ax.pcolormesh(tt, ff, 10 * np.log10(np.abs(sxx)), shading="gouraud")
        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        plt.colorbar(im, label="Amplitude [dB]", ax=ax)
        ax.grid()

        if tmax is not None:
            ax.set_xlim(tmin, tmax)
        if fmax is not None:
            ax.set_ylim(fmin, fmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_stft.png",
            dpi=300,
        )

    def plot_psd(
        self, window="hann", nperseg=2**8, noverlap=2**7, fmin=0, fmax=None, ax=None
    ):
        """Plot the signal PSD"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        f, Pxx = self.get_psd(window=window, nperseg=nperseg, noverlap=noverlap)

        ax.plot(f, 10 * np.log10(Pxx))
        ax.set_title(self.name)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power Spectral Density [dB]")
        ax.grid()

        if fmax is not None:
            ax.set_xlim(fmin, fmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_psd.png",
            dpi=300,
        )


class Ship(AcousticSource):
    """
    Class to handle ship signal properties
    """

    def __init__(
        self,
        name: str,
        fs: float,
        duration: float,
        root_img: str,
        x: float = None,
        y: float = None,
        z: float = None,
        f0: float = None,
        std_fi: float = None,
        tau_corr_fi: float = None,
        n_harmonics: int = None,
    ):
        """
        Constructor
        :param name: Name of the ship
        :param fs: Sampling frequency in Hz
        :param duration: Duration of the signal in seconds
        :param root_img: Path to the folder where to save the images
        :param f0: Fondamental frequency of the ship signal in Hz
        :param std_fi: Standard deviation of the frequency8 fluctuation in Hz
        :param tau_corr_fi: Correlation time of the frequency flucutation in seconds
        """
        super().__init__(name, fs, duration, root_img, x, y, z)

        # Ship specific properties
        self.f0 = f0
        self.std_fi = std_fi
        self.tau_corr_fi = tau_corr_fi
        self.n_harmonics = n_harmonics

        # Generate signal
        self.get_signal()
        # Derive spectrum
        self.get_spectrum()

    def get_signal(self):
        """Get the ship signal"""
        t, s = self.sg.ship_signal(
            f0=self.f0,
            fs=self.fs,
            T=self.duration,
            std_fi=self.std_fi,
            tau_corr_fi=self.tau_corr_fi,
            Nh=self.n_harmonics,
        )

        self.time = t
        self.signal = s
        self.n_samples = len(s)

    @staticmethod
    def plot_demo_ship(root_img: str):
        # Properties of the demo signal (essentially for publication purposes)
        # f0 = 4.629
        # std_fi = 0.072 * f0
        # tau_corr_fi = 0.304 * 1 / f0

        f0 = 4.5
        std_fi = 0.07 * f0
        tau_corr_fi = 0.3 * 1 / f0

        # f0 = 4.889
        # std_fi = 0.058 * f0
        # tau_corr_fi = 0.067 * 1 / f0

        duration = 60 * 30  # 30 minutes
        fs = 100
        ship = Ship(
            name="DemoShip",
            f0=f0,
            fs=fs,
            duration=duration,
            std_fi=std_fi,
            tau_corr_fi=tau_corr_fi,
            root_img=root_img,
        )

        # Plot PSD and spectrogram side by side
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        f, Pxx = ship.get_psd(window="hann", nperseg=2**11, noverlap=2**10)
        axs[0].plot(10 * np.log10(Pxx), f)
        axs[0].set_xlabel("Power Spectral Density [dB]")
        axs[0].set_ylabel("Frequency [Hz]")
        axs[0].set_title("(a)")

        ff, tt, sxx = ship.get_stft(window="hann", nperseg=2**11, noverlap=2**10)
        axs[1].pcolormesh(tt, ff, 10 * np.log10(np.abs(sxx)), shading="gouraud")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_title("(b)")

        fpath = os.path.join(
            root_img,
            f"{ship.name}_psd_stft_{f0:.2f}_{std_fi/f0:.2f}_{tau_corr_fi*f0:.2f}.png",
        )
        plt.savefig(fpath)


class ZcallInterferer(AcousticSource):
    """Class representing an interference source emmiting a Z-call signal"""

    def __init__(
        self,
        name: str,
        fs: float,
        duration: float,
        root_img: str = "",
        x: float = None,
        y: float = None,
        z: float = None,
        fc: float = 22.6,
        Tz: float = 20,
        L: float = -4.5,
        U: float = 3.2,
        M: float = 10,
        alpha: float = 1.8,
        ici: float = 66.4,
        nz: int = 1,
        start_offset_seconds: float = 3,
        stop_offset_seconds: float = 3,
        sl: float = 188.5,
    ):

        super().__init__(name, fs, duration, root_img, x, y, z)

        # Parametric model params
        self.fc = fc
        self.Tz = Tz
        self.L = L
        self.U = U
        self.M = M
        self.alpha = alpha
        self.ici = ici
        self.nz = nz
        self.sl = sl
        self.start_offset_seconds = start_offset_seconds
        self.stop_offset_seconds = stop_offset_seconds

        # Build args
        self.model_args = None
        self.signal_args = None
        self.build_args()

        # Generate signal
        self.get_signal()
        # Derive spectrum
        self.get_spectrum()

    def build_args(self):
        """Build the arguments to pass to the signal generator"""
        self.model_args = {
            "fc": self.fc,
            "Tz": self.Tz,
            "L": self.L,
            "U": self.U,
            "M": self.M,
            "alpha": self.alpha,
            "ici": self.ici,
        }

        self.signal_args = {
            "fs": self.fs,
            "nz": self.nz,
            "start_offset_seconds": self.start_offset_seconds,
            "stop_offset_seconds": self.stop_offset_seconds,
            "signal_duration": self.duration,
            "sl": self.sl,
        }

    def get_signal(self):
        """Load z-call zignal"""
        t, s = self.sg.z_call(signal_args=self.signal_args, model_args=self.model_args)
        self.time = t
        self.signal = s
        self.n_samples = len(s)

    @staticmethod
    def plot_demo_zcall(root_img: str):
        duration = 20  # 20 s
        fs = 100
        abw = ZcallInterferer(
            name="Demo_ZcallInterferer",
            fs=fs,
            duration=duration,
            root_img=root_img,
            nz=0,
            start_offset_seconds=0,
            stop_offset_seconds=0,
        )

        # Print Source Level
        print(f"Expected Source Level: {abw.sl} dB re 1 uPa @ 1 m")
        p_rms = np.std(abw.signal)
        p0 = 1e-6
        sl = 20 * np.log10(p_rms / p0)
        print(f"Source Level: {sl} dB re 1 uPa @ 1 Hz")

        # Add noise to the signal
        abw.signal = (
            1 / 10 * np.std(abw.signal) * np.random.normal(0, 1, abw.n_samples)
            + abw.signal
        )

        # Plot PSD and spectrogram side by side
        nperseg = 2**7
        noverlap = int(0.5 * nperseg)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        f, Pxx = abw.get_psd(window="hann", nperseg=nperseg, noverlap=noverlap)
        axs[0].plot(10 * np.log10(Pxx), f)
        axs[0].set_xlabel("Power Spectral Density [dB]")
        axs[0].set_ylabel("Frequency [Hz]")
        axs[0].set_title("(a)")

        ff, tt, sxx = abw.get_stft(window="hann", nperseg=nperseg, noverlap=noverlap)
        min_val = np.min(np.abs(sxx)[np.abs(sxx) > 0])
        abs_sxx = np.abs(sxx)
        abs_sxx[abs_sxx < min_val] = min_val
        im = axs[1].pcolormesh(
            tt,
            ff,
            10 * np.log10(abs_sxx),
            # shading="gouraud",
            cmap="jet",
        )
        axs[1].set_xlabel("Time [s]")
        axs[1].set_title("(b)")
        plt.colorbar(im, label="[dB]")

        fpath = os.path.join(
            root_img,
            f"{abw.name}_psd_stft.png",
        )
        plt.savefig(fpath)

        # Plot time signal
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(abw.time, abw.signal)
        ax.set_title(abw.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        fpath = os.path.join(
            root_img,
            f"{abw.name}_sig.png",
        )
        plt.savefig(fpath)


if __name__ == "__main__":
    from publication.publication_figure import PubFigure

    pfig = PubFigure(label_fontsize=22, ticks_fontsize=20)
    root_publi = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\uace_testcase\publication\uace_proceedings_ship_signal_illustration"

    # Demo signals
    Ship.plot_demo_ship(root_publi)
    # ZcallInterferer.plot_demo_zcall(root_publi)
