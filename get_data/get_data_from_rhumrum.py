import os
import numpy as np
import scipy.io.wavfile as wavfile

from obspy import UTCDateTime
from matplotlib import pyplot as plt
from obspy.clients.fdsn import Client


def get_rhumrum_data(
    station_id,
    date,
    duration_sec,
    fmin=8.0,
    fmax=46.0,
    channels=["BDH", "BHZ", "BH1", "BH2"],
    save=True,
    plot=False,
):

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
            "bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True
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

        root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\wav\RHUMRUM"
        fname = f"signal_{chnl}_{station_id}_{date[0:9]}.wav"  # on sauvegarde les signaux corriges dans un fichier .wav
        fpath = os.path.join(root_data, fname)
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
    date = "2013-05-31T6:00:00"  # Date and time of the beginning of the recording

    station_id = "RR44"
    duration_sec = 60 * 60

    raw_sig, filt_sig, corr_sig = get_rhumrum_data(
        station_id=station_id, date=date, duration_sec=duration_sec, plot=False
    )
