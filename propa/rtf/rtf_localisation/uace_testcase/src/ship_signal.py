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


class ShipSignal:
    """
    Class to handle ship signal properties
    """

    def __init__(
        self,
        name: str,
        f0: float,
        fs: float,
        duration: float,
        std_fi: float = None,
        tau_corr_fi: float = None,
        root_img: str = "",
    ):
        """
        Constructor
        :param fmin: Minimum frequency
        :param fmax: Maximum frequency
        :param fs: Sampling frequency
        :param duration: Duration of the signal
        """
        self.name = name
        self.f0 = f0
        self.fs = fs
        self.ts = 1 / fs
        self.duration = duration
        self.std_fi = std_fi
        self.tau_corr_fi = tau_corr_fi

        if not os.path.exists(root_img):
            os.makedirs(root_img)
        self.root_img = root_img

        self.sg = SignalGenerator()

        self.time = None
        self.signal = None
        self.n_samples = None
        self.freq = None
        self.spectrum = None

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
        )

        # t, s = self.sg.lfm_chirp(f0=5, f1=50, fs=self.fs, T=self.duration)
        self.time = t
        self.signal = s
        self.n_samples = len(s)

    def get_spectrum(self):
        """Get the signal frequency spectrum"""
        S_f = np.fft.rfft(self.signal)
        f = np.fft.rfftfreq(self.n_samples, self.ts)

        self.freq = f
        self.spectrum = S_f

    def get_stft(self, window="hann", nperseg=2**8, noverlap=2**7):
        """Get the signal STFT"""
        ff, tt, Sxx = sp.stft(
            self.signal, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap
        )
        return ff, tt, Sxx

    def get_psd(self, window="hann", nperseg=2**8, noverlap=2**7):
        """Get the signal PSD"""
        f, Pxx = sp.welch(
            self.signal, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap
        )
        return f, Pxx

    def plot_signal(self, tmin=0, tmax=None):
        """Plot the signal"""

        plt.figure()
        plt.plot(self.time, self.signal)
        plt.title(self.name)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()

        if tmax is not None:
            plt.xlim(tmin, tmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_signal.png",
            dpi=300,
        )

    def plot_spectrum(self, fmin=0, fmax=None):

        plt.figure()
        plt.plot(self.freq, np.abs(self.spectrum))
        plt.title(self.name)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()

        if fmax is not None:
            plt.xlim(fmin, fmax)

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
    ):
        """Plot the signal STFT"""
        ff, tt, sxx = self.get_stft(window=window, nperseg=nperseg, noverlap=noverlap)
        plt.figure()
        plt.pcolormesh(tt, ff, 10 * np.log10(np.abs(sxx)), shading="gouraud")
        plt.title(self.name)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Amplitude [dB]")
        plt.grid()

        if tmax is not None:
            plt.xlim(tmin, tmax)
        if fmax is not None:
            plt.ylim(fmin, fmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_stft.png",
            dpi=300,
        )

    def plot_psd(self, window="hann", nperseg=2**8, noverlap=2**7, fmin=0, fmax=None):
        """Plot the signal PSD"""
        f, Pxx = self.get_psd(window=window, nperseg=nperseg, noverlap=noverlap)
        plt.figure()
        plt.plot(f, 10 * np.log10(Pxx))
        plt.title(self.name)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power Spectral Density [dB]")
        plt.grid()

        if fmax is not None:
            plt.xlim(fmin, fmax)

        plt.savefig(
            f"{self.root_img}/{self.name}_psd.png",
            dpi=300,
        )


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
    ship = ShipSignal(
        name="DemoShipSignal",
        f0=f0,
        fs=fs,
        duration=duration,
        std_fi=std_fi,
        tau_corr_fi=tau_corr_fi,
        root_img=root_img,
    )

    # Plot PSD and spectrom side by side
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


if __name__ == "__main__":
    from publication.publication_figure import PubFigure

    pfig = PubFigure(label_fontsize=22, ticks_fontsize=20)
    root_publi = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\uace_testcase\publication\uace_proceedings_ship_signal_illustration"
    plot_demo_ship(root_publi)
