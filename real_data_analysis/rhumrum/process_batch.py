#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   process_batch.py
@Time    :   2025/02/08 18:29:54
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

from real_data_analysis.real_data_utils import *
from publication.PublicationFigure import PubFigure

PubFigure(label_fontsize=22, title_fontsize=24, legend_fontsize=16, ticks_fontsize=20)


def process_batch(
    dates_to_process, station_id, freq_properties, ch=["BDH"], save=True, root_wav=None
):

    for date in dates_to_process:
        data = load_wav_data(
            date=date,
            rcv_id=station_id,
            duration_s=duration_seconds,
            ch=ch,
            freq_properties=freq_properties,
            save=save,
            root_wav=root_wav,
        )

        # Plot stft
        data["root_img"] = root_wav
        plot_stft(data, save=save)


if __name__ == "__main__":
    dates = []

    save = True
    ch = ["BDH"]
    station_id = "RR46"
    yyyy_mm = "2013-04"
    duration_seconds = 2 * 60 * 60

    # Def frequency properties
    nperseg = 2**12
    noverlap = int(nperseg * 3 / 4)

    fmin = 4
    fmax = 46
    filter_type = "bandpass"
    filter_corners = 4

    freq_properties = {
        "fmin": fmin,
        "fmax": fmax,
        "filter_type": filter_type,
        "filter_corners": filter_corners,
        "noverlap": noverlap,
        "nperseg": nperseg,
    }

    # for i_d in range(1, 32):
    #     for i_h in range(0, 12):
    #         date = f"{yyyy_mm}-{i_d:02d} {(i_h*2):02d}:00:00"
    #         dates.append(date)

    dates = [
        "2013-05-05 18:00:00",
        "2013-05-15 14:00:00",
        "2013-05-15 22:00:00",
        "2013-05-16 20:00:00",
    ]
    root_wav = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\wav\RHUMRUM\extract_abdel_05022025"

    if not os.path.exists(root_wav):
        os.makedirs(root_wav)

    process_batch(
        dates_to_process=dates,
        station_id=station_id,
        freq_properties=freq_properties,
        ch=ch,
        save=save,
        root_wav=root_wav,
    )
