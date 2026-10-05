#!/usr/bin/python3
# -*- coding: latin-1 -*-

"""
Converts seismic data from OBNmanager into wav files

python3 ElobsBin2Wav.py -d <directory containing the .bin files>
                        -e <device type (VS for Vector Sensor, M for MICROBS, EA for ELOBS STAND.)

Optional:
                       -t <start timestamp (GPS time) in seconds>
                       -l <signal duration in seconds>
                       -f <sampling frequency in Hz (if no XML file)>
                       -g <analog gain in dB (if no XML file)>
                       -s <serial number>
                       -m <timeslice in seconds>
                       -c   to apply the frequency response of the sensor (Vector Sensor only)
                       -v   to convert accelero data into velocity (DIGITAL sensor only)

The directory should have an XML file with the acquisition parameters (supplied by OBNmanager).
Without this file, the options -f and -g must be used.
If option -t is not used, the lowest timestamp is used.
If option -l is not used, the maximal duration is used.
Files will be cut according to -m option.

Version 1.0 03/02/2022
Version 1.1 11/02/2022
    Add leap seconds management
    Suppress the display of the list of found files
Version 2.0 12/10/2022:
    add merge of files spaced by 1min30s. max (tilt measure on DIGITAL)
    add timeslice option
Version 2.1 07/02/2024:
    In function ConvertGPSDate2String replace localtime to gmtime

"""

import argparse
import datetime
import math
import numpy as np
import os
import parse
import pathlib
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
import sys
import time as dt
import datetime as dtt
from xml.dom import minidom
# Nombre de voies
NbVoies = 4
import calendar as cld

# format des fichiers OBIT
ObitBinFileFmt = ["ELOBS_{type}-{sn}_TB_{start}-{stop}_channel{chan}_{date}.bin",
                  "{mission}_ELOBS_{type}-{sn}_TB_{start}-{stop}_channel{chan}_{date}.bin"]
ObitXMLFileFmt = ["ELOBS_{type}-{sn}_TB_{start}-{stop}_{date}.xml",
                  "{mission}_ELOBS_{type}-{sn}_TB_{start}-{stop}_{date}.xml"]

# Axes du VectorSensor correspondant aux voies suivant version firmware
# Avant V4: ICVH, après V4: VICH
VS_Axes = ['Z', 'Y', 'X', 'H']
VS_Axes_V4 = ['X', 'Z', 'Y', 'H']

# Axes génériques d'ELOBS A / D correspondant aux voies suivant version firmware
# Avant V4: ICVH, après V4: VICH
Gen_Axes_A = ['X', 'Y', 'Z', 'H']
Gen_Axes_A_V4 = ['Z', 'X', 'Y', 'H']
Gen_Axes_D = ['X', 'Z', 'Y', 'H']
Gen_Axes_D_V4 = ['Z', 'X', 'Y', 'H']  # ELOBS SHOM

# Fullscale SiDel3 en Vpp: +/-2,25V avec gain de 0dB
FS_SIGDEL3_V = 4.52

# FullScale QS3 en m/s²: +/-5m/s². La conversion bin vers wav retourne directement en accélération
FS_QS3_V = 10

# Ecart en seconde entre temps UNIX et temps GPS
GPSfromUTC = (datetime.datetime(1980, 1, 6) - datetime.datetime(1970, 1, 1)).total_seconds()

# Réponse en fréquence du capteur vectoriel externe en eau
VSEau_RepFreq_Axial = [[50, 100, 200, 300, 400, 500, 600],
                       [-6, -4, -2, -3, 6, 2, 4]]
VSEau_RepFreq_Radial = [[50, 100, 200, 300, 400, 500, 600, 700],
                        [-5.5, -4.5, -3.5, -3.5, -2.5, 2, 4, 6]]

# Fréquence de coupure basse en Hz
LOW_CUT = 0

# Fréquence minimale pour l'intégration en vitesse en Hertz
FMIN_INTEG = 1

# Durée maximum d'acquisition autorisée: 1 an
DUREE_MAX_ACQ = 3600 * 24 * 365

# Nombre courant de leap seconds
LeapSeconds = 18

# Ecart maximal entre fichiers en secondes pour rabouter plusieurs fichiers
# notamment dans le cas d'un arrêt pour une mesure de tilt
ECART_MAX_FICHIERS_S = 90


def fileListGet(path, ext="bin"):
    """ Retourne la liste des fichiers .bin triée par ordre alphabétique
        Entrées:
            - répertoire des fichiers
            - ext: extension des fichiers
        Sortie: liste des fichiers
    """
    glob_path = pathlib.Path(path)
    List = [str(pp) for pp in glob_path.glob("*ELOBS*." + ext)]
    List.sort()
    return List


def parseELOBSfilename(f, ext="bin"):
    """ Parse le nom du fichier ELOBS
        Entrées:
            - f: fichier bin
            - ext: extension bin ou xml
        Sortie: fichier parsé
    """
    if ext == "bin":
        fileFmt = ObitBinFileFmt
    elif ext == "xml":
        fileFmt = ObitXMLFileFmt
    else:
        fileFmt = []
    for i in fileFmt:
        fp = parse.parse(i, os.path.split(f)[1])
        if fp is not None:
            return fp
    return None


def ConvertGPSDate2String(gpsDate,format = "%b %d %Y %H:%M:%S UTC"):
    return dt.strftime(format, dt.gmtime(int(gpsDate) + GPSfromUTC - LeapSeconds))


def ConvertStrUTC2GPSDate(StrDate, dateformat):
    ladate = dt.strptime(StrDate,dateformat)
    return int(cld.timegm(ladate) - GPSfromUTC + LeapSeconds)

def SecBetweenDates(StrDate1,StrDate2,dateformat):
    Sdate = dt.strptime(StrDate1, dateformat)
    Edate = dt.strptime(StrDate2, dateformat)
    return (dt.mktime(Edate)-dt.mktime(Sdate))


def getDataFromBin(file, fs, offset=0, duree=-1, fullScale=FS_SIGDEL3_V):
    """  Lit les données sismiques à partir d'un fichier bin. Les convertit en volts
         Entrées:
             - file: fichier bin
             - fs: fréquence d'échantillonnage
             - offset: offset de temps en seconde
             - duree: durée de l'extraction en seconde (-1 si jusqu'à la fin)
             - fullScale: dynamique max pic/pic (4,5 , 2,25 ou 1,125 pour SD3, 10 pour MEMs)
        Sortie: données sismiques lues
    """
    rawData = np.fromfile(file, np.int32, -1, "")
    if (duree == -1) or ((offset + duree) * fs > len(rawData)):
        seismicDataFloat = rawData[int(round(offset * fs)):] * float(fullScale) / 2 ** 32
    else:
        seismicDataFloat = rawData[int(round(offset * fs)):int(round((offset + duree) * fs))] * float(
            fullScale) / 2 ** 32

    return seismicDataFloat


def getParamsFromXMLFile(rep, sn, timestamp):
    """ Récupère les paamètres du fichier XML associé aux fichiers bin
        Entrées:
             - rep: répertoire contenant les fichiers bin
             - sn: numéro de série
             - timestamp: date de début
        Sortie: (fréquence d'écantillonnage en Hz, gain en dB, version firmware majeure)
    """
    lf = fileListGet(rep, "xml")
    fs = 0
    g = 0
    v = 0
    for f in lf:
        fp = parseELOBSfilename(f, "xml")
        if fp is not None:
            if int(fp['sn']) == sn and timestamp >= float(fp['start']) / 1e6 and timestamp <= float(fp['stop']) / 1e6:
                # Lecture XML
                xp = minidom.parse(f)
                xsr = xp.getElementsByTagName('Sampling_Rate')
                # Fréquence d'échantillonnage
                sfs = xsr[0].firstChild.data
                fsp = parse.parse("{sr}ms ({fs} Hz)", sfs)
                if fsp == None:
                    print("Erreur format XML (sampling rate)")
                    return (0, 0)
                fs = float(fsp['fs'])
                # Gain
                xg = xp.getElementsByTagName('Gain_Geo')
                sg = xg[0].firstChild.data
                gp = parse.parse("{g} dB - {fs} mV RMS", sg)
                if gp == None:
                    print("Erreur format XML (gain)")
                    return (0, 0)
                g = int(gp['g'])
                # Version firmware majeure
                xv = xp.getElementsByTagName('Main')
                sv = xv[0].firstChild.data
                gv = parse.parse("{vmaj}.{vmin}.{rel}", sv)
                if gv == None:
                    print("Erreur format XML (version firmware)")
                    return (0, 0)
                v = int(gv['vmaj'])
                break

    if fs == 0:
        print("SN %d, timestamp %f: aucun fichier XML trouvé" % (sn, timestamp))

    return (fs, g, v)


def getListeBinFiles(rep, sn, channel, timestamp, duree, fs):
    """ Retourne la liste des fichiers correspondant au timestamp et la durée demandée
        Entrées:
             - rep: répertoire contenant les fichiers bin
             - sn: numéro de série
             - channel: voie, 'channel0' à 'channel3'
             - fs: fréquence d'échantillonnage
             - timestamp: timestamp de début en secondes
             - duree: durée de l'extraction en secondes
             - fullScale: fullScale SD3 (1600, 400 ou 100)
             - fs: fréquence d'échantillonnage
        Sortie:
            - (fichiers, timestamps de début associés, durées associées, trous dans l'acquisition')
    """
    lb = fileListGet(rep)
    lf = []
    lts = []
    ld = []
    lh = []
    for f in lb:
        fp = parseELOBSfilename(f)
        if fp is not None:
            if (int(fp['sn']) == sn) and (int(fp['chan']) == channel):
                tstart = float(fp['start']) / 1e6
                # Le timestamp de stop n'est pas bon sur les anciennes versions => calcul sur taille du fichier
                tstop = tstart + os.path.getsize(f) / 4 / fs
                if (tstart > timestamp) and (tstart <= (timestamp + ECART_MAX_FICHIERS_S)) and (
                        duree > (tstart - timestamp)):
                    # Petit trou dans la donnée, on prend quand même le fichier suivant. On corrige timestamp
                    duree -= tstart - timestamp
                    lh.append([timestamp, tstart])
                    timestamp = tstart
                if (tstart <= timestamp) and (tstop > timestamp):
                    lf.append(f)
                    lts.append(tstart)
                    if duree == -1:
                        # On prend le premier fichier jusqu'à la fin
                        duree = tstop - timestamp
                    dint = min(duree, tstop - timestamp)
                    ld.append(dint)
                    duree -= dint
                    timestamp += dint
                    if duree < 1 / fs:
                        break

    return (lf, lts, ld, lh)


def getMinTimestamp(rep, sn, ts):
    """ Retourne le timestamp minimal à partir du paramètre timestamp permettant
        d'avoir les 4 voies
        Entrées:
            - rep: répertorie contenant les fichiers bin
            - sn: numéro de série
            - ts: timestamp demandé
        Sortie: timestamp minimal permettant d'avoir les 4 voies
    """
    ts_int = np.ones(4) * -1
    lf = fileListGet(rep)
    for f in lf:
        fp = parseELOBSfilename(f)
        if int(fp['sn']) == sn:
            chan = int(fp['chan'])
            if ts_int[chan] == -1:
                start = float(fp['start']) * 1e-6
                stop = float(fp['stop']) * 1e-6
                if ts == -1:
                    # Pas de date de début spécifié => on prend le début du fichier
                    ts_int[chan] = start
                elif ts >= start and ts <= stop:
                    # date de début dans le fichier => on garde cette date
                    ts_int[chan] = ts
                elif ts < start:
                    # date de début avant le début du fichier => on prend le début du fichier
                    ts_int[chan] = start
    # Si aucune voie détectée, on passe
    if np.isin(ts_int, -1).all():
        return -1
    if np.isin(-1, ts_int).any():
        # Manque au moins une voie => erreur
        print("Voie manquante")
        return -1
    # On retourne la date la lus élevée pour avoir les 4 voies
    return np.max(ts_int)


def getDataFromBinFiles(rep, sn, channel, fs, timestamp, duree, fullScale=FS_SIGDEL3_V):
    """  Lit des données à partir de fichiers bin contenus dans un répertoire.
         Unité: Volt
         Entrées:
             - rep: répertoire contenant les fichiers bin
             - sn: numéro de série
             - channel: voie, 'channel0' à 'channel3'
             - fs: fréquence d'échantillonnage
             - timestamp: timestamp de debut en secondes
             - duree: durée de l'extraction en secondes
             - fullScale: fullScale SD3 (1600, 400 ou 100)
         Sortie: tableau de données lues, trous détectés dans l'acquisition'
    """
    # Liste des fichiers à parcourir
    lf, lts, ld, lh = getListeBinFiles(rep, sn, channel, timestamp, duree, fs)
    ts = timestamp
    seismicDataFloat = np.array([])
    print('debut d\'extraction pour le channel ',channel)
    for i, f in enumerate(lf):
        print('fichier ',f)
        if (i != 0) and (ts < lts[i]):
            # Petite rupture, on remplit de 0 le fichier sauf au début
            seismicDataFloat = np.append(seismicDataFloat, np.zeros(int((lts[i] - ts) * fs)))
            duree -= lts[i] - ts
            ts = lts[i]
        if duree == -1:
            duree = ld[i]
        df = min(duree, ld[i])
        seismicDataFloat = np.append(seismicDataFloat, getDataFromBin(f, fs, ts - lts[i], df, fullScale))
        duree -= df
        ts += df

    return seismicDataFloat, lh


def egalisation(sig, fs, rf):
    """ Corrige le signal de la réponse en fréquence du capteur
        Entrées:
            - sig: signal
            - fs: fréquence d'échantillonnage
            - rf: réponse en fréquence (tableau 2D (f(Hz), g(dB))
        Sortie: signal corrigé
    """
    # Conversion réponse en fréquence en tableau numpy
    nrf = np.array(rf)
    # Il faut passer dans le domaine fréquentiel pour pouvoir appliquer la réponse en fréquence
    sf = np.fft.rfft(sig)
    # Fonction d'interpolation linéaire
    fi = interp1d(nrf[0, :], np.power(10, nrf[1][:] / 20), bounds_error=False,
                  fill_value=(np.power(10, nrf[1][0] / 20), np.power(10, nrf[1][-1] / 20)))
    nbpts = len(sf)
    '''for i in range(nbpts):
        f = i*fs/nbpts
        sf[i] /= fi(f)'''
    sf /= fi(np.linspace(0, fs / 2, nbpts))
    # Retour dans le domaine fréquentiel
    sig = np.fft.irfft(sf)
    return sig


def convAcc2Vit(acc, fe=2000):
    """ Convertit un signal d'accélération (m/s²) en vitesse (m/s)
        Entrées:
            - acc: signal en m/s²
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
    return v


def getSerialNumbers(rep, sn_user):
    """ Récupère les numéros de série des fichiers listés ainsi que le type
        (ANALOG ou DIGITAL)
        Entrées:
            - rep: répertoire des fichiers bin
            - sn: numéro de série spécifique (-1 si non spécifié)
        Sortie: liste de tuplets (numéros de série, type)
    """
    lsn = []
    lf = fileListGet(rep)
    for f in lf:
        fp = parseELOBSfilename(f)
        if fp is not None:
            sn = int(fp['sn'])
            if sn not in [x[0] for x in lsn] and (sn_user == -1 or sn_user == sn):
                lsn.append((sn, fp['type']))
    lsn.sort()
    print("Numéros de série trouvés: " + str([x[0] for x in lsn]))
    return lsn


def writeHoles(csvname, channel, lh):
    """ Ecrit la liste des trous détectés dans le fichier csv
        Entrées:
            - csvname: chemin d'accès complet du fichier csv
            - channel: voie concernée
            - lh: liste des trous
        Sortie: fichier csv rempli
    """
    # Ecriture des trous trouvés dans l'acquisition
    with open(csvname, 'a') as csvfile:
        if csvfile.tell() == 0:
            # Ecriture entête
            csvfile.write('channel;start;stop\n')
        for h in lh:
            csvfile.write('%d;%.4f;%.4f\n' % (channel, h[0], h[1]))

def readOBS(folder,start,stop,channel='H',calib=True,SeismoVar='acc'):

    # folder = str de type : 'F:\/24-Classif_2024\OBS\OBS5\/test_extract'
    # start = date de début de la donnée à charger au format str : jj/mm/yyy hh:mm:ss
    # stop = date de fin de la donnée à charger au format str : jj/mm/yyy hh:mm:ss
    # channel = str : 'H' 'Y'...
    # calib = boolean
    # OBStype = 'A' ou 'D'
    # SeismoVar = 'acc' ou 'vit'

    parser = argparse.ArgumentParser("Converts seismic data from OBNmanager into WAV files")
    GPS_start_time = ConvertStrUTC2GPSDate(start,"%d/%m/%Y %H:%M:%S")
    data_length = SecBetweenDates(start,stop,"%d/%m/%Y %H:%M:%S")

    parser.add_argument("--dir", "-d"
                        , required=True
                        , action="store"
                        , type=str
                        , dest="dir"
                        , help="Directory containing the bin files")

    parser.add_argument("--device", "-e"
                        , required=True
                        , action="store"
                        , dest="device"
                        , help="device type (VectorSensor: VS, MICROBS: M, ELOBS STAND.: ES")

    parser.add_argument("--ts", "-t"
                        , default=-1
                        , action="store"
                        , type=float
                        , dest="ts"
                        , help="start timestamp (GPS time) in seconds")

    parser.add_argument("--len", "-l"
                        , default=DUREE_MAX_ACQ
                        , action="store"
                        , type=float
                        , dest="len"
                        , help="signal duration in seconds")

    parser.add_argument("--calib", "-c"
                        , action="store_true"
                        , dest="calib"
                        , help="frequency response of sensor is applied (Vector Sensor only)")

    parser.add_argument("--freq", "-f"
                        , default=1000
                        , type=int
                        , action="store"
                        , dest="freq"
                        , help="sampling frequency in Hz (if no XML file)")

    parser.add_argument("--gain", "-g"
                        , default=0
                        , type=int
                        , action="store"
                        , dest="gain"
                        , help="gain in dB (if no XML file)")

    parser.add_argument("--sn", "-s"
                        , default=-1
                        , type=int
                        , action="store"
                        , dest="sn"
                        , help="serial number")

    parser.add_argument("--velocity", "-v"
                        , action="store_true"
                        , dest="velocity"
                        , help="accelero data are converted in velocity")

    parser.add_argument("--timeslice", "-m"
                        , default=-1
                        , type=int
                        , action="store"
                        , dest="timeslice"
                        , help="timeslice in seconds")

    # store parser arguments
    if calib:
        if SeismoVar=='vit':
            args = parser.parse_args(['-d', folder,  # path
                                      '-e', 'M',
                                      '-v','-c',
                                      '-m', str(25000)])
        elif SeismoVar=='acc':
            args = parser.parse_args(['-d', folder,  # path
                                      '-e', 'M',
                                      '-c',
                                      '-m', str(25000)])
    else:
        if SeismoVar=='vit':
            args = parser.parse_args(['-d', folder,  # path
                                      '-e', 'M',
                                      '-v',
                                      '-m', str(25000)])
        elif SeismoVar=='acc':
            args = parser.parse_args(['-d', folder,  # path
                                      '-e', 'M',
                                      '-m', str(25000)])
    if GPS_start_time:
        if data_length:
            args.ts = GPS_start_time
            args.len = data_length
    if not os.access(args.dir, os.R_OK):
        print("Directory not found")
        exit(0)

    # Récupère les numéros de série avec le type DIGITAL ou ANALOG
    lsn = getSerialNumbers(args.dir, args.sn)
    # Liste les fichiers correspondant aux heures demandées
    for sn in lsn:
        ts = getMinTimestamp(args.dir, sn[0], args.ts)
        if ts > GPS_start_time:
            GPS_start_time = ts
        print(ts)
        lb = []
        # Lecture paramètres d'acquisition
        (fs, g, v) = getParamsFromXMLFile(args.dir, sn[0], ts)
        # Détermination des axes
        if args.device == "VS":
            if v >= 4:
                axes = VS_Axes_V4
            else:
                axes = VS_Axes
        else:
            if sn[1] == 'A':
                if v >= 4:
                    axes = Gen_Axes_A_V4
                else:
                    axes = Gen_Axes_A
            else:
                if v >= 4:
                    axes = Gen_Axes_D_V4
                else:
                    axes = Gen_Axes_D
        if fs == 0:
            fs = args.freq
            g = args.gain
        print("S/N %d  %s (V%d): Fs = %dHz, G = %ddB" % (sn[0], sn[1], v, fs, g))
        for chan in range(NbVoies):
            print(chan)
            if sn[1] == 'A' or chan == 3:
                fullScale = FS_SIGDEL3_V / np.power(10, g / 20)
            else:
                fullScale = FS_QS3_V
            # if channel == 'all':
            #     s, lh = getDataFromBinFiles(args.dir, sn[0], chan, fs, ts, args.len, fullScale)
            #     if chan == 0:
            #         s4 = s
            #     elif chan == 1:
            #         l = min(len(s4), len(s))
            #         s4 = np.vstack((s4[0:l], s[0:l]))
            #     else:
            #         l = min(np.shape(s4)[1], len(s))
            #         s4 = np.vstack((s4[:, 0:l], s[0:l]))
            #     if len(lh) != 0:
            #         writeHoles(
            #             os.path.join(args.dir, "ELOBS_%s-SN%d-TS%.4f-D%.4f-Holes.csv" % (sn[1], sn[0], ts, args.len)), chan,
            #             lh)
            #         # Application de la réponse en fréquence du capteur
            #         if sn[1] == 'D' and args.device == "VS" and args.calib:
            #             s4[axes.index('X'), :] = egalisation(s4[axes.index('X'), :], fs, VSEau_RepFreq_Radial)
            #             s4[axes.index('Y'), :] = egalisation(s4[axes.index('Y'), :], fs, VSEau_RepFreq_Radial)
            #             s4[axes.index('Z'), :] = egalisation(s4[axes.index('Z'), :], fs, VSEau_RepFreq_Axial)
            #         elif args.calib:
            #             print("Frequency response not applied (only for Vector Sensor)")
            #             # Intégration en vitesse avec filtrage passe-haut
            #         if sn[1] == 'D' and args.velocity:
            #             s4[axes.index('X'), :] = convAcc2Vit(s4[axes.index('X'), :], fs)
            #             s4[axes.index('Y'), :] = convAcc2Vit(s4[axes.index('Y'), :], fs)
            #             s4[axes.index('Z'), :] = convAcc2Vit(s4[axes.index('Z'), :], fs)
            #     TimeStamps = np.linspace(GPS_start_time, data_length, len(s4[0]))
            # else:
            if axes[chan] == channel:
                s4, lh = getDataFromBinFiles(args.dir, sn[0], chan, fs, ts, args.len, fullScale)
                TimeStamps = np.linspace(GPS_start_time, GPS_start_time+data_length, len(s4))
                # Application de la réponse en fréquence du capteur
                if sn[1] == 'D' and args.device == "VS" and args.calib and channel!='H':
                    if channel!='Z':
                        s4 = egalisation(s4, fs, VSEau_RepFreq_Radial)
                    elif channel == 'Z':
                        s4 = egalisation(s4, fs, VSEau_RepFreq_Axial)
                    if args.velocity:
                        s4[axes.index(channel), :] = convAcc2Vit(s4[axes.index(channel), :], fs)
                elif args.calib:
                    print("Frequency response not applied (only for Vector Sensor)")
                    # Intégration en vitesse avec filtrage passe-haut
        Sig = s4
    return Sig, TimeStamps, lh, sn
