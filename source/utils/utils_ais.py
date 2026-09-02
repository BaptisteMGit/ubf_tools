#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils_ais.py
@Time    :   2025/12/11 16:01:30
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from misc import progression_bar
from source.utils.utils_gps import interpolate_gps


def interpolate_ais(ais, time_step="10s", subset_idx=None, mmsi_to_include=None):
    mmsi = ais.mmsi.unique()
    if subset_idx is not None:
        mmsi = mmsi[subset_idx]
        # Ensure mmsi_jules is in mmsi in case we preprocess a subset of all the mmsis
        if mmsi_to_include not in mmsi:
            mmsi = np.append(mmsi, mmsi_to_include)

    ais_interp = pd.DataFrame()

    # Test progress bar
    index0 = 0
    indexf = mmsi.size - 1
    prev_progress = 0

    for i, mmsi_i in enumerate(mmsi):
        prev_progress = progression_bar(i, index0, indexf, prev_progress)

        ais_mmsi = ais.loc[ais["mmsi"] == mmsi_i]
        ais_mmsi_interp = interpolate_gps(df_gps=ais_mmsi, time_step=time_step)

        # Add mmsi column
        ais_mmsi_interp["mmsi"] = mmsi_i

        # Concatenate to ais_interp
        ais_interp = pd.concat([ais_interp, ais_mmsi_interp], ignore_index=True)

    return ais_interp, mmsi


# -------------------------------------------------------------
# Check si payload AIS valide (caractères OK)
# -------------------------------------------------------------
def is_valid_ais_payload(payload):
    for c in payload:
        code = ord(c)
        if code < 48 or code > 119:  # hors plage AIS
            return False
    return True


# -------------------------------------------------------------
def sixbit_ascii_to_bits(payload):
    bitstring = ""
    for c in payload:
        v = ord(c) - 48
        if v > 40:
            v -= 8
        bitstring += f"{v:06b}"
    return bitstring


def get_int(bits, start, length, signed=False):
    if start + length > len(bits):
        return None
    v = int(bits[start : start + length], 2)
    if signed and v & (1 << (length - 1)):
        v -= 1 << length
    return v


# -------------------------------------------------------------
# Décode positions AIS messages 1/2/3/18/19
# -------------------------------------------------------------
def decode_ais_position(bits):
    msg_type = get_int(bits, 0, 6)
    if msg_type not in (1, 2, 3, 18, 19):
        return None

    mmsi = get_int(bits, 8, 30)

    if msg_type in (1, 2, 3):
        lon = get_int(bits, 61, 28, signed=True)
        lat = get_int(bits, 89, 27, signed=True)
        sog = get_int(bits, 50, 10)
        cog = get_int(bits, 116, 12)
    else:
        lon = get_int(bits, 57, 28, signed=True)
        lat = get_int(bits, 85, 27, signed=True)
        sog = get_int(bits, 46, 10)
        cog = get_int(bits, 112, 12)

    if lon is None or lat is None:
        return None

    return {
        "type": msg_type,
        "mmsi": mmsi,
        "lat": lat / 600000.0,
        "lon": lon / 600000.0,
        "sog": sog / 10.0 if sog is not None else None,
        "cog": cog / 10.0 if cog is not None else None,
    }


# ============================================================
# 4. DECODAGE DE L'HEURE GPS ($GPRMC)
# ============================================================


def parse_gprmc(line):
    fields = line.split(",")
    if len(fields) < 10 or fields[2] != "A":
        return None

    hhmmss = fields[1]
    ddmmyy = fields[9]

    try:
        hour = int(hhmmss[0:2])
        minute = int(hhmmss[2:4])
        second = int(hhmmss[4:6])

        day = int(ddmmyy[0:2])
        month = int(ddmmyy[2:4])
        year = 2000 + int(ddmmyy[4:6])
    except:
        print("Erreur parsing GPRMC:", line)

    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# -------------------------------------------------------------
def decode_nmea_ais_line(line):
    if not line.startswith(("!AIVDM", "!AIVDO")):
        return None

    fields = line.split(",")

    # ---- Vérification minimum ----
    if len(fields) < 6:
        return None

    payload = fields[5]

    # ---- Vérifie payload valide ----
    if not is_valid_ais_payload(payload):
        return None

    # ---- Vérifie présence du champ pad/checksum ----
    if len(fields) < 7 or "*" not in fields[6]:
        return None  # trame incomplète → on ignore

    # ---- Extraction pad ----
    try:
        pad = int(fields[6].split("*")[0])
    except ValueError:
        return None

    # ---- Conversion en bits ----
    bits = sixbit_ascii_to_bits(payload)
    if pad > 0:
        bits = bits[:-pad]

    # ---- Taille minimale pour message position ----
    if len(bits) < 168:
        return None

    return decode_ais_position(bits)


# -------------------------------------------------------------
def ais_file_to_csv(input_file, output_file="ais_positions.csv"):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            msg = decode_nmea_ais_line(line)
            if msg:
                results.append(msg)

    # Écriture CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["mmsi", "type", "lat", "lon", "sog", "cog"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"✔ {len(results)} positions AIS extraites.")
    print(f"✔ CSV généré : {output_file}")


# def ais_file_to_csv(input_file, output_file):
#     current_time = None

#     with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, open(
#         output_file, "w", newline="", encoding="utf-8"
#     ) as fout:

#         writer = csv.writer(fout)
#         writer.writerow(
#             ["time_utc", "mmsi", "latitude", "longitude", "sog_knots", "cog_deg"]
#         )

#         for line in fin:
#             line = line.strip()

#             # Heure GPS
#             if line.startswith("$GPRMC"):
#                 t = parse_gprmc(line)
#                 if t is not None:
#                     current_time = t
#                 continue

#             # Message AIS
#             if line.startswith("!AIVDM") or line.startswith("!AIVDO"):
#                 if current_time is None:
#                     continue

#                 msg = decode_nmea_ais_line(line)
#                 if msg is None:
#                     continue

#                 writer.writerow(
#                     [
#                         current_time.isoformat(),
#                         msg["mmsi"],
#                         msg["lat"],
#                         msg["lon"],
#                         msg["sog"],
#                         msg["cog"],
#                     ]
#                 )


# -------------------------------------------------------------
# Exécution directe
# -------------------------------------------------------------
if __name__ == "__main__":
    import os

    root_input = r"C:\Users\baptiste.menetrier\Desktop\ressource\XP_Fiberscope_Groix_092025\Jules\ais\14_10_25"
    fname = "14_10_25"
    # fname = "extract_MMSI_226916000"
    fpath_in = os.path.join(root_input, fname + ".txt")

    root_output = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\ais"
    fpath_out = os.path.join(root_output, f"ais_pos_{fname}.csv")
    if not os.path.exists(root_output):
        os.makedirs(root_output)

    ais_file_to_csv(input_file=fpath_in, output_file=fpath_out)
