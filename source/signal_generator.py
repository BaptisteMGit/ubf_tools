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


# ======================================================================================================================
# Class
# ======================================================================================================================


class SignalGenerator:
    """
    Class to handle signal generation and analysis
    """

    def __init__(self):
        pass

    def psd_to_timeserie(self, psd, df):
        """
        Generate a time serie from its Power Spectral Density (PSD).

        Parameters
        ----------
        psd : np.ndarray
            Power Spectral Density
        df : float
            Frequency resolution

        Returns
        -------
        t : np.array
            Time vector
        x_t : np.array
            Time serie
        """

        # Set f=0 and f=fs/2 to 0   -> psd has exactly nf=np.fft.rfftfreq points
        psd = np.concatenate(([0], psd, [0]))

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

        # We can assert x_t has the required psd
        # fs = nt * df
        # ff, sxx = sp.welch(x_t, fs=fs)
        # assert np.allclose(psd, np.abs(X_f))

        return t, x_t

    def colored_noise(self, T, fs, noise_color="white"):
        """
        Generate colored noise

        Parameters
        ----------
        T : float
            Duration of the time serie
        fs : float
            Sampling frequency
        noise_color : str
            Color of the noise (white or pink)

        Returns
        -------
        t : np.array
            Time vector
        x_t : np.array
            Time serie
        """
