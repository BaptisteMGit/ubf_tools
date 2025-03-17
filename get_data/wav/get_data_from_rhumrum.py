#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   get_data_from_rhumrum.py
@Time    :   2025/02/26 10:24:37
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Module to download data from the RESIF database for the RHUM-RUM network
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import pandas as pd
import scipy.io.wavfile as wavfile

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# ======================================================================================================================
# Functions
# ======================================================================================================================


def get_rhumrum_data(
    station_id,
    date,
    duration_sec,
    fmin=8.0,
    fmax=46.0,
    filter_type="bandpass",
    filter_corners=2,
    channels=["BDH", "BHZ", "BH1", "BH2"],
    save=True,
    root_wav=None,
    plot=False,
):
    """
    Download data from the RESIF database for the RHUM-RUM network.
    The data is filtered and corrected for the sensor response.

    Parameters
    ----------
    station_id : str
        Station ID.
    date : str
        Date and time of the beginning of the recording.
    duration_sec : int
        Duration of the recording in seconds.
    fmin : float, optional
        Minimum frequency for the bandpass filter.
    fmax : float, optional
        Maximum frequency for the bandpass filter.
    filter_type : str, optional
        Type of filter. Default is 'bandpass'.
    filter_corners : int, optional
        Number of corners for the filter. Default is 2.
    channels : list, optional
        List of channels to download. Default is ['BDH', 'BHZ', 'BH1', 'BH2'].
    save : bool, optional
        Save the corrected signals as .wav files. Default is True.
    root_wav : str, optional
        Root directory to save the .wav files. Default is None.
    plot : bool, optional
        Plot the raw, filtered and corrected signals. Default is False.

    Returns
    -------
    raw_signal : dict
        Raw signals.
    filt_signal : dict
        Filtered signals.
    corr_signal : dict
        Corrected signals.

    Examples
    --------
    >>> date = "2013-05-31T16:30:00"
    >>> station_id = "RR44"
    >>> duration_sec = 2 * 60 * 60
    >>> raw_sig, filt_sig, corr_sig = get_rhumrum_data(
    ...     station_id=station_id, date=date, duration_sec=duration_sec, plot=True
    ... )

    """

    client = Client("RESIF")
    t = UTCDateTime(date)

    raw_signal = {}
    filt_signal = {}
    corr_signal = {}

    for chnl in channels:
        # Load raw data
        raw_sig = client.get_waveforms(
            "YV", station_id, "00", chnl, t, t + duration_sec
        )
        # Filter signal
        filt_sig = raw_sig.filter(
            filter_type,
            freqmin=fmin,
            freqmax=fmax,
            corners=filter_corners,
            zerophase=True,
        )
        # Load station metadata
        inv = client.get_stations(
            network="YV",
            station=station_id,
            channel=chnl,
            level="response",
            format="xml",
        )

        # Correct sensor response
        corr_sig = filt_sig[0].copy()
        corr_sig = corr_sig.remove_response(
            inventory=inv
        )  # on corrige de la reponse du capteur

        # Plot signals
        if plot:
            raw_sig.plot()  # raw
            filt_sig.plot()
            corr_sig.plot()

        if save:
            if root_wav == None:
                root_wav = (
                    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\wav\RHUMRUM"
                )

                assert os.path.exists(root_wav), f"Directory {root_wav} does not exist."

            # Build filename
            fmt = "%Y-%m-%d_%H-%M-%S"
            date_start = pd.to_datetime(date).strftime(fmt)
            date_end = (
                pd.to_datetime(date) + pd.Timedelta(seconds=duration_sec)
            ).strftime(fmt)
            fname = f"signal_{chnl}_{station_id}_{date_start}_{date_end}.wav"  # on sauvegarde les signaux corriges dans un fichier .wav
            fpath = os.path.join(root_wav, fname)

            # Save corrected signal
            wavfile.write(
                fpath,
                100,
                corr_sig.data,
            )

        raw_signal[chnl] = raw_sig
        filt_signal[chnl] = filt_sig
        corr_signal[chnl] = corr_sig

    return raw_signal, filt_signal, corr_signal

    # signal43 = signal43tr[:]


# signal44tr = signal44_brut[0].copy()
# signal44tr = signal44tr.remove_response(inventory=inv44)
# signal44tr.plot()
# signal44 = signal44tr[:]


if __name__ == "__main__":
    date = "2013-05-31T16:30:00"  # Date and time of the beginning of the recording

    station_id = "RR44"
    duration_sec = 2 * 60 * 60

    raw_sig, filt_sig, corr_sig = get_rhumrum_data(
        station_id=station_id, date=date, duration_sec=duration_sec, plot=True
    )
