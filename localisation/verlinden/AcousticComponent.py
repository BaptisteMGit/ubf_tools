import numpy as np
import matplotlib.pyplot as plt


class AcousticSource:
    """Acoustic source class. The source is defined by the signal recorded 1m away from the source."""

    def __init__(self, signal, time, z_src=5, kraken_freq=None):
        self.signal = signal  # Source signal
        self.time = time  # Time vector
        self.z_src = z_src  # Source depth (m)
        self.kraken_freq = (
            kraken_freq  # Use limited number of frequencies for kraken run
        )

        self.ns = len(signal)  # Number of samples
        self.fs = 1 / (time[1] - time[0])  # Sampling frequency
        self.freq = None  # Frequency vector
        self.spectrum = None  # Source spectrum
        self.spectrum_df = None  # Source spectrum frequency resolution

        self.energetic_freq = None  # Energetic frequency domain
        self.min_freq = None  # Minimum frequency of the energetic domain
        self.max_freq = None  # Maximum frequency of the energetic domain
        self.get_spectrum()

    def get_spectrum(self, nfft=None):
        """Source spectrum derived from time serie"""
        if nfft is None:
            nfft = self.ns
        self.spectrum = np.fft.rfft(self.signal, n=nfft)
        self.freq = np.fft.rfftfreq(nfft, 1 / self.fs)
        self.spectrum_df = self.freq[1] - self.freq[0]
        self.analyse_spectrum()

    def analyse_spectrum(self):
        """Analyse source spectrum to derive frequency bounds of the energetic domain"""

        # TODO: might need to update this cretiria depending on the source spectrum considered
        # Another simple criteria could be the bandwith B = 1 / T

        min_energy = 0.01 * np.max(np.abs(self.spectrum))
        idx_energetic_freq = np.abs(self.spectrum) >= min_energy
        self.energetic_freq = self.freq[idx_energetic_freq]
        self.energetic_spectrum = self.spectrum[idx_energetic_freq]
        self.min_freq = np.min(self.energetic_freq)
        self.max_freq = np.max(self.energetic_freq)

    def set_kraken_freq(self, fmin, fmax, df):
        """Set kraken frequency vector"""
        self.kraken_freq = np.arange(fmin, fmax, df)

    def plot_spectrum(self, ax=None):
        """Plot source spectrum"""
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(self.freq, np.abs(self.spectrum))
        ax.vlines(
            [self.min_freq, self.max_freq], 0, np.max(np.abs(self.spectrum)), "r", "--"
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|FFT(s(t))|")
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
            self.plot_spectrum(ax=ax_spectrum)
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 8))
            self.plot_signal(ax=ax[0])
            self.plot_spectrum(ax=ax[1])
            plt.tight_layout()


class Receiver:
    n_rcv = 0

    def __init__(self, name, pos):
        self.__class__.n_rcv += 1
        self.name = name
        self.position = pos
        self.idx = self.n_rcv - 1


class ReceiverArray:
    def __init__(self, receiver_names, receiver_positions):
        self.type = "unspecified"
        self.receiver_names = receiver_names
        self.receiver_positions = np.array(receiver_positions)
        self.create_receivers()

    def create_receivers(self):
        self.receivers = []
        for i_r, rcv_name in enumerate(self.receiver_names):
            rcv_i = Receiver(rcv_name, self.receiver_positions[i_r])
            self.receivers.append(rcv_i)
        self.n_rcv = self.receivers[0].n_rcv

    def plot_array(self, fig=None):
        if fig is None:
            plt.figure()

        for rcv in self.receivers:
            plt.plot(rcv.position[0], rcv.position[1], "o", label=rcv.name)
        plt.legend(self.receiver_names)


class LinearReceiverArray(ReceiverArray):
    def __init__(self, receiver_names, receiver_positions):
        super().__init__(receiver_names, receiver_positions)

        # For the moment only two elements array is supported
        if self.n_rcv != 2:
            raise ValueError("Only two elements array is supported")

        self.type = "linear"
        self.array_length = np.linalg.norm(
            self.receiver_positions[0] - self.receiver_positions[-1]
        )
