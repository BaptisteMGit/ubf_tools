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
# Avant V4: ICVH, apr�s V4: VICH
VS_Axes = ['Z', 'Y', 'X', 'H']
VS_Axes_V4 = ['X', 'Z', 'Y', 'H']

# Axes g�n�riques d'ELOBS A / D correspondant aux voies suivant version firmware
# Avant V4: ICVH, apr�s V4: VICH
Gen_Axes_A = ['X', 'Y', 'Z', 'H']
Gen_Axes_A_V4 = ['Z', 'X', 'Y', 'H']
Gen_Axes_D = ['X', 'Z', 'Y', 'H']
Gen_Axes_D_V4 = ['Z', 'X', 'Y', 'H']  # ELOBS SHOM

# Fullscale SiDel3 en Vpp: +/-2,25V avec gain de 0dB
FS_SIGDEL3_V = 4.52

# FullScale QS3 en m/s�: +/-5m/s�. La conversion bin vers wav retourne directement en acc�l�ration
FS_QS3_V = 10

# Ecart en seconde entre temps UNIX et temps GPS
GPSfromUTC = (datetime.datetime(1980, 1, 6) - datetime.datetime(1970, 1, 1)).total_seconds()

# R�ponse en fr�quence du capteur vectoriel externe en eau
VSEau_RepFreq_Axial = [[50, 100, 200, 300, 400, 500, 600],
                       [-6, -4, -2, -3, 6, 2, 4]]
VSEau_RepFreq_Radial = [[50, 100, 200, 300, 400, 500, 600, 700],
                        [-5.5, -4.5, -3.5, -3.5, -2.5, 2, 4, 6]]

# Fr�quence de coupure basse en Hz
LOW_CUT = 0

# Fr�quence minimale pour l'int�gration en vitesse en Hertz
FMIN_INTEG = 1

# Dur�e maximum d'acquisition autoris�e: 1 an
DUREE_MAX_ACQ = 3600 * 24 * 365

# Nombre courant de leap seconds
LeapSeconds = 18

# Ecart maximal entre fichiers en secondes pour rabouter plusieurs fichiers
# notamment dans le cas d'un arr�t pour une mesure de tilt
ECART_MAX_FICHIERS_S = 90


def fileListGet(path, ext="bin"):
    """ Retourne la liste des fichiers .bin tri�e par ordre alphab�tique
        Entr�es:
            - r�pertoire des fichiers
            - ext: extension des fichiers
        Sortie: liste des fichiers
    """
    glob_path = pathlib.Path(path)
    List = [str(pp) for pp in glob_path.glob("*ELOBS*." + ext)]
    List.sort()
    return List


def parseELOBSfilename(f, ext="bin"):
    """ Parse le nom du fichier ELOBS
        Entr�es:
            - f: fichier bin
            - ext: extension bin ou xml
        Sortie: fichier pars�
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
    """  Lit les donn�es sismiques � partir d'un fichier bin. Les convertit en volts
         Entr�es:
             - file: fichier bin
             - fs: fr�quence d'�chantillonnage
             - offset: offset de temps en seconde
             - duree: dur�e de l'extraction en seconde (-1 si jusqu'� la fin)
             - fullScale: dynamique max pic/pic (4,5 , 2,25 ou 1,125 pour SD3, 10 pour MEMs)
        Sortie: donn�es sismiques lues
    """
    rawData = np.fromfile(file, np.int32, -1, "")
    if (duree == -1) or ((offset + duree) * fs > len(rawData)):
        seismicDataFloat = rawData[int(round(offset * fs)):] * float(fullScale) / 2 ** 32
    else:
        seismicDataFloat = rawData[int(round(offset * fs)):int(round((offset + duree) * fs))] * float(
            fullScale) / 2 ** 32

    return seismicDataFloat


def getParamsFromXMLFile(rep, sn, timestamp):
    """ R�cup�re les paam�tres du fichier XML associ� aux fichiers bin
        Entr�es:
             - rep: r�pertoire contenant les fichiers bin
             - sn: num�ro de s�rie
             - timestamp: date de d�but
        Sortie: (fr�quence d'�cantillonnage en Hz, gain en dB, version firmware majeure)
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
                # Fr�quence d'�chantillonnage
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
        print("SN %d, timestamp %f: aucun fichier XML trouv�" % (sn, timestamp))

    return (fs, g, v)


def getListeBinFiles(rep, sn, channel, timestamp, duree, fs):
    """ Retourne la liste des fichiers correspondant au timestamp et la dur�e demand�e
        Entr�es:
             - rep: r�pertoire contenant les fichiers bin
             - sn: num�ro de s�rie
             - channel: voie, 'channel0' � 'channel3'
             - fs: fr�quence d'�chantillonnage
             - timestamp: timestamp de d�but en secondes
             - duree: dur�e de l'extraction en secondes
             - fullScale: fullScale SD3 (1600, 400 ou 100)
             - fs: fr�quence d'�chantillonnage
        Sortie:
            - (fichiers, timestamps de d�but associ�s, dur�es associ�es, trous dans l'acquisition')
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
                # NOTE : B. Menetrier -> 
                # os.path.getsize(f) donne la taille du fichier en octets (bytes)
                # Le facteur 4 correspond à la taille d'un int32 en octets, et le facteur fs correspond au nombre d'échantillons par seconde. 
                # Donc, os.path.getsize(f) / 4 / fs calcule la durée du fichier en secondes.
                tstop = tstart + os.path.getsize(f) / 4 / fs
                if (tstart > timestamp) and (tstart <= (timestamp + ECART_MAX_FICHIERS_S)) and (
                        duree > (tstart - timestamp)):
                    # Petit trou dans la donn�e, on prend quand m�me le fichier suivant. On corrige timestamp
                    duree -= tstart - timestamp
                    lh.append([timestamp, tstart])
                    timestamp = tstart
                if (tstart <= timestamp) and (tstop > timestamp):
                    lf.append(f)
                    lts.append(tstart)
                    if duree == -1:
                        # On prend le premier fichier jusqu'� la fin
                        duree = tstop - timestamp
                    dint = min(duree, tstop - timestamp)
                    ld.append(dint)
                    duree -= dint
                    timestamp += dint
                    if duree < 1 / fs:
                        break

    return (lf, lts, ld, lh)


def getMinTimestamp(rep, sn, ts):
    """ Retourne le timestamp minimal � partir du param�tre timestamp permettant
        d'avoir les 4 voies
        Entr�es:
            - rep: r�pertorie contenant les fichiers bin
            - sn: num�ro de s�rie
            - ts: timestamp demand�
        Sortie: timestamp minimal permettant d'avoir les 4 voies
    """
    ts_int = np.ones(4) * -1
    lf = fileListGet(rep)
    for f in lf:
        fp = parseELOBSfilename(f)
        print(f, fp)
        if int(fp['sn']) == sn:
            chan = int(fp['chan'])
            if ts_int[chan] == -1:
                start = float(fp['start']) * 1e-6
                stop = float(fp['stop']) * 1e-6
                print(start, stop, ts, ts_int)
                if ts == -1:
                    # Pas de date de d�but sp�cifi� => on prend le d�but du fichier
                    ts_int[chan] = start
                elif ts >= start and ts <= stop:
                    # date de d�but dans le fichier => on garde cette date
                    ts_int[chan] = ts
                elif ts < start:
                    # date de d�but avant le d�but du fichier => on prend le d�but du fichier
                    ts_int[chan] = start
    # NOTE : B. Menetrier -> dans notre cas les fichiers bins ne contiennent volontairement que la voie hydro (3) et donc cette fonction renvoie toujours -1. 
    # Si aucune voie d�tect�e, on passe
    if np.isin(ts_int, -1).all():
        return -1
    if np.isin(-1, ts_int).any():
        # Manque au moins une voie => erreur
        print("Voie manquante")
        return -1
    # On retourne la date la lus �lev�e pour avoir les 4 voies
    return np.max(ts_int)


def getDataFromBinFiles(rep, sn, channel, fs, timestamp, duree, fullScale=FS_SIGDEL3_V):
    """  Lit des donn�es � partir de fichiers bin contenus dans un r�pertoire.
         Unit�: Volt
         Entr�es:
             - rep: r�pertoire contenant les fichiers bin
             - sn: num�ro de s�rie
             - channel: voie, 'channel0' � 'channel3'
             - fs: fr�quence d'�chantillonnage
             - timestamp: timestamp de debut en secondes
             - duree: dur�e de l'extraction en secondes
             - fullScale: fullScale SD3 (1600, 400 ou 100)
         Sortie: tableau de donn�es lues, trous d�tect�s dans l'acquisition'
    """
    # Liste des fichiers � parcourir
    lf, lts, ld, lh = getListeBinFiles(rep, sn, channel, timestamp, duree, fs)
    ts = timestamp
    seismicDataFloat = np.array([])
    print('debut d\'extraction pour le channel ',channel)
    for i, f in enumerate(lf):
        print('fichier ',f)
        if (i != 0) and (ts < lts[i]):
            # Petite rupture, on remplit de 0 le fichier sauf au d�but
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
    """ Corrige le signal de la r�ponse en fr�quence du capteur
        Entr�es:
            - sig: signal
            - fs: fr�quence d'�chantillonnage
            - rf: r�ponse en fr�quence (tableau 2D (f(Hz), g(dB))
        Sortie: signal corrig�
    """
    # Conversion r�ponse en fr�quence en tableau numpy
    nrf = np.array(rf)
    # Il faut passer dans le domaine fr�quentiel pour pouvoir appliquer la r�ponse en fr�quence
    sf = np.fft.rfft(sig)
    # Fonction d'interpolation lin�aire
    fi = interp1d(nrf[0, :], np.power(10, nrf[1][:] / 20), bounds_error=False,
                  fill_value=(np.power(10, nrf[1][0] / 20), np.power(10, nrf[1][-1] / 20)))
    nbpts = len(sf)
    '''for i in range(nbpts):
        f = i*fs/nbpts
        sf[i] /= fi(f)'''
    sf /= fi(np.linspace(0, fs / 2, nbpts))
    # Retour dans le domaine fr�quentiel
    sig = np.fft.irfft(sf)
    return sig


def convAcc2Vit(acc, fe=2000):
    """ Convertit un signal d'acc�l�ration (m/s�) en vitesse (m/s)
        Entr�es:
            - acc: signal en m/s�
            - fe: fr�quence d'�chantillonage
        sortie:
            - signal converti en m/s
    """

    # Conversion d'un signal large bande dans le domaine fr�quentiel
    npts = len(acc)
    if (npts % 2) == 0:
        nptsFFT = int(npts / 2 + 1)
    else:
        nptsFFT = int((npts + 1) / 2)
    # Calcul de la v�locit� � partir de l'acc�l�ration par division par i.2.PI.F dans le domaine fr�quentiel
    accf = np.fft.rfft(acc, norm="ortho")
    # Suppression composante continue et division par 2.PI.F pour conversion en m/s
    vf = np.zeros_like(accf)
    for i in range(1, nptsFFT):
        f = i / npts * fe
        if f > FMIN_INTEG:
            # On ne conserve que la bande d'int�r�t
            vf[i] = accf[i] / (2 * np.pi * f) * (1j)
    # retour dans le domaine temporel et passage en Pa
    v = np.fft.irfft(vf, npts, norm="ortho")
    return v


def getSerialNumbers(rep, sn_user):
    """ R�cup�re les num�ros de s�rie des fichiers list�s ainsi que le type
        (ANALOG ou DIGITAL)
        Entr�es:
            - rep: r�pertoire des fichiers bin
            - sn: num�ro de s�rie sp�cifique (-1 si non sp�cifi�)
        Sortie: liste de tuplets (num�ros de s�rie, type)
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
    # print("Num�ros de s�rie trouv�s: " + str([x[0] for x in lsn]))
    return lsn


def writeHoles(csvname, channel, lh):
    """ Ecrit la liste des trous d�tect�s dans le fichier csv
        Entr�es:
            - csvname: chemin d'acc�s complet du fichier csv
            - channel: voie concern�e
            - lh: liste des trous
        Sortie: fichier csv rempli
    """
    # Ecriture des trous trouv�s dans l'acquisition
    with open(csvname, 'a') as csvfile:
        if csvfile.tell() == 0:
            # Ecriture ent�te
            csvfile.write('channel;start;stop\n')
        for h in lh:
            csvfile.write('%d;%.4f;%.4f\n' % (channel, h[0], h[1]))

def readOBS(folder,start,stop,channel='H',calib=True,SeismoVar='acc'):

    # folder = str de type : 'F:\/24-Classif_2024\OBS\OBS5\/test_extract'
    # start = date de d�but de la donn�e � charger au format str : jj/mm/yyy hh:mm:ss
    # stop = date de fin de la donn�e � charger au format str : jj/mm/yyy hh:mm:ss
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

    # R�cup�re les num�ros de s�rie avec le type DIGITAL ou ANALOG
    lsn = getSerialNumbers(args.dir, args.sn)
    # Liste les fichiers correspondant aux heures demand�es
    for sn in lsn:
        ts = getMinTimestamp(args.dir, sn[0], args.ts)
        if ts > GPS_start_time:
            GPS_start_time = ts
        print(ts)
        lb = []
        # Lecture param�tres d'acquisition
        (fs, g, v) = getParamsFromXMLFile(args.dir, sn[0], ts)
        # D�termination des axes
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
            #         # Application de la r�ponse en fr�quence du capteur
            #         if sn[1] == 'D' and args.device == "VS" and args.calib:
            #             s4[axes.index('X'), :] = egalisation(s4[axes.index('X'), :], fs, VSEau_RepFreq_Radial)
            #             s4[axes.index('Y'), :] = egalisation(s4[axes.index('Y'), :], fs, VSEau_RepFreq_Radial)
            #             s4[axes.index('Z'), :] = egalisation(s4[axes.index('Z'), :], fs, VSEau_RepFreq_Axial)
            #         elif args.calib:
            #             print("Frequency response not applied (only for Vector Sensor)")
            #             # Int�gration en vitesse avec filtrage passe-haut
            #         if sn[1] == 'D' and args.velocity:
            #             s4[axes.index('X'), :] = convAcc2Vit(s4[axes.index('X'), :], fs)
            #             s4[axes.index('Y'), :] = convAcc2Vit(s4[axes.index('Y'), :], fs)
            #             s4[axes.index('Z'), :] = convAcc2Vit(s4[axes.index('Z'), :], fs)
            #     TimeStamps = np.linspace(GPS_start_time, data_length, len(s4[0]))
            # else:
            if axes[chan] == channel:
                # NOTE: BM
                fs = 2000
                s4, lh = getDataFromBinFiles(args.dir, sn[0], chan, fs, ts, args.len, fullScale)
                TimeStamps = np.linspace(GPS_start_time, GPS_start_time+data_length, len(s4))
                # Application de la r�ponse en fr�quence du capteur
                if sn[1] == 'D' and args.device == "VS" and args.calib and channel!='H':
                    if channel!='Z':
                        s4 = egalisation(s4, fs, VSEau_RepFreq_Radial)
                    elif channel == 'Z':
                        s4 = egalisation(s4, fs, VSEau_RepFreq_Axial)
                    if args.velocity:
                        s4[axes.index(channel), :] = convAcc2Vit(s4[axes.index(channel), :], fs)
                elif args.calib:
                    print("Frequency response not applied (only for Vector Sensor)")
                    # Int�gration en vitesse avec filtrage passe-haut
        Sig = s4
    return Sig, TimeStamps, lh, sn
