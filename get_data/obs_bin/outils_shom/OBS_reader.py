#!/usr/bin/env python
import os

# import time
# from datetime import datetime, timedelta
import Elobs_bin_reader as obsr
import matplotlib

matplotlib.use("TkAgg", force=True)
from matplotlib import pyplot as plt

# import librosa
import numpy as np

# import sys
# import scipy as sp
import pandas as pd

# from scipy.signal.ShortTimeFFT import stft


def convAcc2Vit(acc, fe=2000):
    """Convertit un signal d'accélération (m/s²) en vitesse (m/s)
    Entrées:
        - acc: signal en m/s² - calibré
        - fe: fréquence d'échantillonage
    sortie:
        - signal converti en m/s
    """

    # Conversion d'un signal large bande dans le domaine fréquentiel
    npts = len(acc)
    if (npts % 2) == 0:
        nptsFFT = int(npts / 2 + 1)
    else:
        nptsFFT = int((npts + 1) / 2)
    # Calcul de la vélocité à partir de l'accélération par division par i.2.PI.F dans le domaine fréquentiel
    accf = np.fft.rfft(acc, norm="ortho")
    # Suppression composante continue et division par 2.PI.F pour conversion en m/s
    vf = np.zeros_like(accf)
    for i in range(1, nptsFFT):
        f = i / npts * fe
        if f > FMIN_INTEG:
            # On ne conserve que la bande d'intérêt
            vf[i] = accf[i] / (2 * np.pi * f) * (1j)
    # retour dans le domaine temporel et passage en Pa
    v = np.fft.irfft(vf, npts, norm="ortho")
    return v.real


def read_tilt_from_health(
    folder, time
):  ### FAUT-IL INTEGRE CE MODULE A LA CLASSE OBS ???
    A = os.listdir(folder)
    healthfile = A[np.where(["Health_ELOBS" in a for a in A])[0][0]]
    HthData = pd.read_csv(
        folder + "/" + healthfile, sep=";", header="infer", index_col=False
    )
    Date = HthData["Date"]
    TiltI = HthData["Test tilt 2 result (deg)"]
    TiltC = HthData["Test tilt 3 result (deg)"]
    TiltMystere = HthData["Test tilt 1 result (deg)"]
    return TiltI, TiltC, Date, TiltMystere


class OBSdata:
    def __init__(self):
        self.fe = 2000
        self.p_raw = []
        self.p_calib = []
        self.ax_raw = []
        self.ay_raw = []
        self.az_raw = []
        self.vx_raw = []
        self.vy_raw = []
        self.vz_raw = []
        self.ax = []
        self.ay = []
        self.az = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.ax_meta = "not_charged"
        self.ay_meta = "not_charged"
        self.az_meta = "not_charged"
        self.vx_meta = "not_charged"
        self.vy_meta = "not_charged"
        self.vz_meta = "not_charged"
        self.time = []
        self.Lon = []
        self.Lat = []
        self.imm = []
        self.type = []
        self.name = []
        self.sn = []

    def extract_raw_OBS(self, folder, period, channel="all", Acc=True):
        # type = 'D' for digital or 'A' for Analogic
        # period contient le temps de début et fin de la séquence à traiter en datetime.UTC
        t0 = period[0]
        t1 = period[1]
        sig, timestamps, lh, sn = obsr.readOBS(
            folder, t0, t1, channel="H", calib=True, SeismoVar="acc"
        )
        self.p_raw = sig
        self.time = timestamps
        self.sn = sn
        if Acc:
            if channel == "all":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="X", calib=True, SeismoVar="acc"
                )
                self.ax = sig
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Y", calib=True, SeismoVar="acc"
                )
                self.ay = sig
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Z", calib=True, SeismoVar="acc"
                )
                self.az = sig
                self.ax_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
                self.ay_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
                self.az_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
            elif channel == "X":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="X", calib=True, SeismoVar="acc"
                )
                self.ax = sig
                self.time = timestamps
                self.ax_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
            elif channel == "Y":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Y", calib=True, SeismoVar="acc"
                )
                self.ay = sig
                self.time = timestamps
                self.ay_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
            elif channel == "Z":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Z", calib=True, SeismoVar="acc"
                )
                self.az = sig
                self.time = timestamps
                self.az_meta = "calibration statique - pas de correction de tilt - pas de correction azimutale"
        else:
            if channel == "all":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="H", calib=True, SeismoVar="vit"
                )
                self.p_raw = sig
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="X", calib=True, SeismoVar="vit"
                )
                self.vx_raw = sig
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Y", calib=True, SeismoVar="vit"
                )
                self.vy_raw = sig
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Z", calib=True, SeismoVar="vit"
                )
                self.vz_raw = sig
            elif channel == "X":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="X", calib=True, SeismoVar="vit"
                )
                self.vx_raw = sig
                self.time = timestamps
            elif channel == "Y":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Y", calib=True, SeismoVar="vit"
                )
                self.vy_raw = sig
                self.time = timestamps
            elif channel == "Z":
                sig, timestamps, lh, sn = obsr.readOBS(
                    folder, t0, t1, channel="Z", calib=True, SeismoVar="vit"
                )
                self.vz_raw = sig
                self.time = timestamps
        return

    def calib_H(self, sh=-160):
        if len(self.p_calib) == len(self.p_raw):
            print("La données est déjà calibrée")
            return
        else:
            self.p_calib = self.p_raw / np.power(10, sh / 20)
            print(
                "La sensibilité de l'hydrophone à ?? Hz a été prise en compte pour tout le spectre."
            )
            return

    def compute_v(self, chan):
        # ne marche que sur accélération calibrée
        if chan == "X":
            if len(self.ax) != 0:
                self.vx = convAcc2Vit(self.ax, self.fe)
                self.vx_meta = self.ax_meta
        if chan == "Y":
            if len(self.ay) != 0:
                self.vy = convAcc2Vit(self.ay, self.fe)
                self.vy_meta = self.ay_meta
        if chan == "Z":
            if len(self.az) != 0:
                self.vz = convAcc2Vit(self.az, self.fe)
                self.vz_meta = self.az_meta
        if chan == "all":
            if (len(self.ax) != 0) & (len(self.ay) != 0) & (len(self.az) != 0):
                self.vx = convAcc2Vit(self.ax, self.fe)
                self.vy = convAcc2Vit(self.ay, self.fe)
                self.vz = convAcc2Vit(self.az, self.fe)
                self.vx_meta = self.ax_meta
                self.vy_meta = self.ay_meta
                self.vz_meta = self.az_meta
            return

    def compute_tilt_corr(self, folder):
        # réaligne la composante Z avec la direction du vecteur g mesuré et remet les composantes horizontales dans le plan horizontal.
        if "- pas de correction de tilt -" in self.ax_meta:
            # semble marcher avec des angles nuls
            TiltI, TiltC, Date, TiltMystR = read_tilt_from_health(
                folder, self.time
            )  # CONTROLER  si le tilt change sur la série temporelle
            # print ('Date : ')
            # print(Date)
            GPS_Date = [
                obsr.ConvertStrUTC2GPSDate(d, "%Y-%m-%d %H:%M:%S") for d in Date
            ]
            SelectIndX = np.where(
                (np.array(GPS_Date) <= self.time[-1])
                * (np.array(GPS_Date) >= self.time[0])
            )[0]
            print("Select indexes : \r")
            print(SelectIndX)
            # Test variabilité de I et de C
            if (np.std([TiltI[i] for i in SelectIndX]) <= 0.001) * (
                np.std([TiltC[i] for i in SelectIndX]) <= 0.001
            ):
                I = TiltI[SelectIndX[0]] / 180 * np.pi
                C = TiltC[SelectIndX[0]] / 180 * np.pi
            elif len(SelectIndX) == 0:
                print("std = nan")
                SIndX = np.where(
                    np.abs(np.array(GPS_Date) - self.time[-1])
                    == np.min(np.abs(np.array(GPS_Date) - self.time[-1]))
                )[0][0]
                I = TiltI[SIndX] / 180 * np.pi
                C = TiltC[SIndX] / 180 * np.pi
            else:
                GPSdate = [GPS_Date[i] for i in SelectIndX]
                TI = [TiltI[i] for i in SelectIndX]
                TC = [TiltC[i] for i in SelectIndX]
                fig = plt.figure(figsize=(15, 15))
                titre = "Variation des tilts pendant la mesure"
                ax0 = plt.subplot(2, 1, 1)
                ax0.plot(GPSdate, TI, "+", label="TiltI")
                ax1 = plt.subplot(2, 1, 2)
                ax1.plot(GPSdate, TC, "+", label="Tiltc")
                plt.legend()
                print(np.std([TiltI[i] for i in SelectIndX]))
                print(np.std([TiltC[i] for i in SelectIndX]))
                print(
                    np.std([TiltI[i] for i in SelectIndX])
                    <= 0.001 + np.std([TiltC[i] for i in SelectIndX])
                    <= 0.001
                )
                # print('l\'OBS a bougé pendant cette mesure, pas de correction effectuée')
                # return
                raise Exception("l'OBS a bougé significativement pendant cette mesure")

            if np.abs(C) != np.pi / 2:
                sC = np.cos(C) / np.abs(np.cos(C))
            else:
                sC = 1
            if np.abs(I) != np.pi / 2:
                sI = np.cos(I) / np.abs(np.cos(I))
            else:
                sI = 1
            M = np.zeros((3, 3))
            M[0, 0] = sC * sI * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2)
            M[0, 1] = -np.sin(I)
            M[0, 2] = np.sin(C)
            M[1, 0] = sC * np.tan(I) * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2)
            M[1, 1] = np.cos(I)
            M[1, 2] = np.sin(C) * np.tan(I)
            M[2, 0] = -np.sin(C) / np.cos(I)
            M[2, 2] = sC * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2) / np.cos(I)
            Vect = [
                self.az,
                self.ax,
                self.ay,
            ]  # !!! A CONTROLER EN FONCTION DES DENOMINATIONS CHAN 0 CHAN 1 CHAN 2 CHAN 3 dans la lecture -> ok contrôlé
            Vect_tilt_corr = np.dot(M, Vect)  # [Z_corr, X_corr, Y_corr]
            self.az = Vect_tilt_corr[0]
            self.ax = Vect_tilt_corr[1]
            self.ay = Vect_tilt_corr[2]
            meta = "calibration statique - tilt corrigé - pas de correction azimutale"
            self.ax_meta = meta
            self.ay_meta = meta
            self.az_meta = meta
            return
        else:
            print("le tilt a déjà été corrigé dans les données")
            return


### TILT CORRECTION ###
if __name__ == "__main__":

    # Test continuité des fichiers (fichier 2 et 3 de OBS 1)
    fich_2 = "ELOBS_D-2744775_TB_1456025622500000-1456050626249500_channel3_2026-03-02-08-45-30_002"
    gps_date = fich_2.split("_")[3]
    start_date_gps = float(gps_date.split("-")[0]) / 1e6
    end_date_gps = float(gps_date.split("-")[1]) / 1e6
    start_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=start_date_gps)
    print("start_utc_from_file_code : ", start_utc_from_file_code)

    end_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=end_date_gps)
    print("end_utc_from_file_code : ", end_utc_from_file_code)

    fich_3 = "ELOBS_D-2744775_TB_1456050626250000-1456075631249500_channel3_2026-03-02-08-45-30_002"
    gps_date = fich_3.split("_")[3]
    start_date_gps = float(gps_date.split("-")[0]) / 1e6
    end_date_gps = float(gps_date.split("-")[1]) / 1e6
    start_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=start_date_gps)
    print("start_utc_from_file_code : ", start_utc_from_file_code)

    end_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=end_date_gps)
    print("end_utc_from_file_code : ", end_utc_from_file_code)

    ### Adaptation pour test données ACOUPLANE ###
    folderfile = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ACOUPLANE\DATA\PRESSURE\BIN\OBS1"

    # ELOBS_D-3042301_TB_1455972318000000-1455997319874500_channel3_2026-03-02-09-52-44_002
    # 1455972318000000 to 1455997319874500
    # D'après le code matlab de JM B. le code GPS de début fin est contenu dans les 10 premiers éléments
    start_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=1455972318)
    print("start_utc_from_file_code : ", start_utc_from_file_code)

    end_utc_from_file_code = obsr.ConvertGPSDate2String(gpsDate=1455997319)
    print("end_utc_from_file_code : ", end_utc_from_file_code)

    # arrêt du canon
    start_date = "24/02/2026 12:45:00"  # en UTC !!!
    stop_date = "24/02/2026 19:41:41"  # en UTC !!!
    # charger des données longues
    GPS_start_time = obsr.ConvertStrUTC2GPSDate(start_date, "%d/%m/%Y %H:%M:%S")
    print(GPS_start_time)
    data_length = obsr.SecBetweenDates(start_date, stop_date, "%d/%m/%Y %H:%M:%S")
    # # set channel to extract
    # channel = "all"
    # OBS7 = OBSdata()
    # OBS7.extract_raw_OBS(folderfile, [start_date, stop_date], channel, True)
    # start_date, stop_date = None, None
    sig, timestamps, lh, sn = obsr.readOBS(
        folderfile, start_date, stop_date, channel="H", calib=True, SeismoVar="acc"
    )
    p_raw = sig
    time = timestamps
    print("p_raw : ", p_raw)
    print("time : ", time)
    print("lh : ", lh)
    print("sn : ", sn)

    print(p_raw.shape)
    # OBS7 = OBSdata()
    # OBS7.extract_raw_OBS(folderfile, [start_date, stop_date], channel="H", Acc=True)

    plt.figure()
    plt.plot(time, p_raw)
    plt.show()


    ### Code fourni par Myriam L. ###
    # FMIN_INTEG = 1
    # folderfile = "E:\ACOUPLANE\ACOUPLANE_2026\OBS4\DATA"
    # # arrêt du canon
    # start_date = "26/02/2026 20:20:00"  # en UTC !!!
    # stop_date = "26/02/2026 20:40:00"  # en UTC !!!
    # # charger des données longues
    # GPS_start_time = obsr.ConvertStrUTC2GPSDate(start_date, "%d/%m/%Y %H:%M:%S")
    # data_length = obsr.SecBetweenDates(start_date, stop_date, "%d/%m/%Y %H:%M:%S")
    # # set channel to extract
    # channel = "all"
    # OBS7 = OBSdata()
    # OBS7.extract_raw_OBS(folderfile, [start_date, stop_date], channel, True)
    # # gérer les variations du tilt
    # TiltI, TiltC, Date, TiltMystR = read_tilt_from_health(
    #     folderfile, OBS7.time
    # )  # CONTROLER  si le tilt change sur la série temporelle
    # GPS_Date = [obsr.ConvertStrUTC2GPSDate(d, "%Y-%m-%d %H:%M:%S") for d in Date]
    # SelectIndX = np.where(
    #     (np.array(GPS_Date) <= GPS_start_time + data_length)
    #     * (np.array(GPS_Date) >= GPS_start_time)
    # )[0]
    # # Test variabilité de I et de C
    # if np.std([TiltI[i] for i in SelectIndX]) == 0 + np.std(
    #     [TiltC[i] for i in SelectIndX]
    # ):
    #     I = TiltI[SelectIndX[0]] / 180 * np.pi
    #     C = TiltC[SelectIndX[0]] / 180 * np.pi
    #     M = TiltMystR[SelectIndX[0]] / 180 * np.pi
    # else:
    #     # David.exit('l\'OBS a bougé pendant cette mesure')
    #     raise Exception("l'OBS a bougé pendant cette mesure")
    # if np.abs(C) != np.pi / 2:
    #     sC = np.cos(C) / np.abs(np.cos(C))
    # else:
    #     sC = 1
    # if np.abs(I) != np.pi / 2:
    #     sI = np.cos(I) / np.abs(np.cos(I))
    # else:
    #     sI = 1

    # # charger le tilt en conséquence
    # # qualifier l'offset sur les voies X Y Z
    # M = np.zeros((3, 3))
    # M[0, 0] = sC * sI * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2)
    # M[0, 1] = -np.sin(I)
    # M[0, 2] = np.sin(C)
    # M[1, 0] = sC * np.tan(I) * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2)
    # M[1, 1] = np.cos(I)
    # M[1, 2] = np.sin(C) * np.tan(I)
    # M[2, 0] = -np.sin(C) / np.cos(I)
    # M[2, 2] = sC * np.sqrt(np.cos(C) ** 2 - np.sin(I) ** 2) / np.cos(I)
    # Vect = [
    #     OBS7.az,
    #     OBS7.ax,
    #     OBS7.ay,
    # ]  # !!! A CONTROLER EN FONCTION DES DENOMINATIONS CHAN 0 CHAN 1 CHAN 2 CHAN 3 dans la lecture -> ok contrôlé

    # # faire la correction de tilt
    # Vect_tilt_corr = np.dot(M, Vect)  # [Z_corr, X_corr, Y_corr]
    # OBS7.az = Vect_tilt_corr[0]
    # OBS7.ax = Vect_tilt_corr[1]
    # OBS7.ay = Vect_tilt_corr[2]
    # meta = "calibration statique - tilt corrigé - pas de correction azimutale"
    # print(meta)
    # OBS7.ax_meta = meta
    # OBS7.ay_meta = meta
    # OBS7.az_meta = meta
