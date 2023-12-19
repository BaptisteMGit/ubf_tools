import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from propa.kraken_toolbox.utils import waveguide_cutoff_freq


class AcousticSource:
    """Acoustic source class. The source is defined by the signal recorded 1m away from the source."""

    def __init__(
        self, signal, time, name="", waveguide_depth=100, z_src=5, kraken_freq=None
    ):
        self.signal = signal  # Source signal
        self.time = time  # Time vector
        self.name = name  # Source name
        self.waveguide_depth = waveguide_depth  # Waveguide depth (m)
        self.z_src = z_src  # Source depth (m)
        self.kraken_freq = (
            kraken_freq  # Use limited number of frequencies for kraken run
        )

        self.ns = len(signal)  # Number of samples
        self.fs = 1 / (time[1] - time[0])  # Sampling frequency
        self.positive_freq = None  # Frequency vector
        self.positive_spectrum = None  # Source spectrum
        self.df = None  # Source spectrum frequency resolution

        self.energetic_freq = None  # Energetic frequency domain
        self.min_freq = None  # Minimum frequency of the energetic domain
        self.max_freq = None  # Maximum frequency of the energetic domain
        self.get_spectrum()

    def get_spectrum(self, nfft=None):
        """Derive spectrum from source signal"""

        self.psd_freq, self.psd = signal.welch(
            self.signal, fs=self.fs, window="hamming", scaling="spectrum"
        )

        if nfft is None:
            # self.nfft = 2 ** int(np.log2(self.ns) + 1)  # Next power of 2
            self.nfft = 2**12

        max_nfreq_kraken = 1000  # Maximum number of frequencies allowed for kraken run
        self.positive_freq = np.fft.rfftfreq(self.nfft, 1 / self.fs)
        self.kraken_freq = self.positive_freq[
            self.positive_freq > waveguide_cutoff_freq(max_depth=self.waveguide_depth)
        ]

        # Ensure that the number of frequencies is not too large for kraken run
        while self.kraken_freq.size > max_nfreq_kraken:
            self.nfft = self.nfft // 2  # Previous power of 2
            self.positive_freq = np.fft.rfftfreq(self.nfft, 1 / self.fs)
            self.kraken_freq = self.positive_freq[
                self.positive_freq
                > waveguide_cutoff_freq(max_depth=self.waveguide_depth)
            ]

        # Real FFT
        self.positive_spectrum = np.fft.rfft(
            self.signal,
            n=self.nfft,
        )
        self.df = self.positive_freq[1] - self.positive_freq[0]

        self.analyse_spectrum()

    def analyse_spectrum(self):
        """Analyse source spectrum to derive frequency bounds of the energetic domain"""

        # TODO: might need to update this criteria depending on the source spectrum considered
        # Another simple criteria could be the bandwith B = 1 / T

        min_energy = 0.01 * np.max(np.abs(self.positive_spectrum))
        idx_energetic_freq = np.abs(self.positive_spectrum) >= min_energy
        self.energetic_freq = self.positive_freq[idx_energetic_freq]
        self.energetic_spectrum = self.positive_spectrum[idx_energetic_freq]
        self.min_freq = np.min(self.energetic_freq)
        self.max_freq = np.max(self.energetic_freq)

    def set_kraken_freq(self, fmin, fmax, df):
        """Set kraken frequency vector"""
        self.kraken_freq = np.arange(fmin, fmax, df)

    def plot_psd(self, ax=None):
        """Plot source psd"""
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.semilogy(self.psd_freq, self.psd)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (Pa²/Hz))")
        ax.set_title("Source spectrum")

    def plot_spectrum_magnitude(self, ax=None):
        """Plot source psd"""
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(self.positive_freq, np.abs(self.positive_spectrum))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|RFFT(S)|")
        ax.set_title("Source spectrum")

    def plot_signal(self, ax=None):
        """Plot source signal"""
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(self.time, self.signal)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Source signal")

    def display_source(self, ax_signal=None, ax_spectrum=None):
        """Display source signal and spectrum"""
        if ax_signal is not None and ax_spectrum is not None:
            self.plot_signal(ax=ax_signal)
            self.plot_psd(ax=ax_spectrum)
            plt.tight_layout()
        else:
            __, ax = plt.subplots(1, 3, figsize=(10, 8))
            self.plot_signal(ax=ax[0])
            self.plot_psd(ax=ax[1])
            self.plot_spectrum_magnitude(ax=ax[2])
            plt.tight_layout()
