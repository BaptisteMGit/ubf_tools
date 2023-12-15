import numpy as np


def readshd(filename, xs=None, ys=None, freq=None):
    """Read a shade file produce by FIELD.exe and return the data in a dictionary.

    Usage : PlotTitle, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure = read_shd(filename, xs, ys, freq)

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    if freq is None:
        if xs is None:
            (
                PlotTitle,
                PlotType,
                freqVec,
                freq0,
                read_freq,
                atten,
                Pos,
                pressure,
            ) = readshd_bin(filename=filename)
        else:
            (
                PlotTitle,
                PlotType,
                freqVec,
                freq0,
                read_freq,
                atten,
                Pos,
                pressure,
            ) = readshd_bin(filename=filename, xs=xs, ys=ys)
    else:
        (
            PlotTitle,
            PlotType,
            freqVec,
            freq0,
            read_freq,
            atten,
            Pos,
            pressure,
        ) = readshd_bin(filename=filename, freq=freq)

    # else:
    #     raise ValueError("Unrecognized file extension")

    return PlotTitle, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure


def readshd_bin(filename, xs=None, ys=None, freq=None):
    """Read a '.shd' binary file.

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    try:
        fid = open(filename, "rb")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"readshd_bin.py: No shade file with the name {filename} exists"
        )

    recl = int(np.fromfile(fid, dtype=np.int32, count=1))  # record length in bytes
    title = fid.read(80).decode("utf-8").strip()  # read and decode the title

    fid.seek(4 * recl)  # reposition to end of first record
    PlotType = fid.read(10).decode("utf-8").strip()  # read and decode the PlotType

    fid.seek(2 * 4 * recl)  # reposition to end of second record
    Nfreq = int(np.fromfile(fid, dtype=np.int32, count=1))
    Ntheta = int(np.fromfile(fid, dtype=np.int32, count=1))
    Nsx = int(np.fromfile(fid, dtype=np.int32, count=1))
    Nsy = int(np.fromfile(fid, dtype=np.int32, count=1))
    Nsz = int(np.fromfile(fid, dtype=np.int32, count=1))
    Nrz = int(np.fromfile(fid, dtype=np.int32, count=1))
    Nrr = int(np.fromfile(fid, dtype=np.int32, count=1))
    freq0 = float(np.fromfile(fid, dtype=np.float64, count=1))
    atten = float(np.fromfile(fid, dtype=np.float64, count=1))

    fid.seek(3 * 4 * recl)  # reposition to end of record 3
    freqVec = np.fromfile(fid, dtype=np.float64, count=Nfreq)

    fid.seek(4 * 4 * recl)  # reposition to end of record 4
    Pos = {}
    Pos["theta"] = np.fromfile(fid, dtype=np.float64, count=Ntheta)

    if PlotType.strip() != "TL":
        fid.seek(5 * 4 * recl)  # reposition to end of record 5
        Pos["s"] = {}
        Pos["s"]["x"] = np.fromfile(fid, dtype=np.float64, count=Nsx)

        fid.seek(6 * 4 * recl)  # reposition to end of record 6
        Pos["s"]["y"] = np.fromfile(fid, dtype=np.float64, count=Nsy)
    else:
        fid.seek(5 * 4 * recl)  # reposition to end of record 5
        Pos["s"] = {}
        Pos["s"]["x"] = np.fromfile(fid, dtype=np.float64, count=2)
        Pos["s"]["x"] = np.linspace(Pos["s"]["x"][0], Pos["s"]["x"][1], Nsx)

        fid.seek(6 * 4 * recl)  # reposition to end of record 6
        Pos["s"]["y"] = np.fromfile(fid, dtype=np.float64, count=2)
        Pos["s"]["y"] = np.linspace(Pos["s"]["y"][0], Pos["s"]["y"][1], Nsy)

    fid.seek(7 * 4 * recl)  # reposition to end of record 7
    Pos["s"]["z"] = np.fromfile(fid, dtype=np.float32, count=Nsz)

    fid.seek(8 * 4 * recl)  # reposition to end of record 8
    Pos["r"] = {}
    Pos["r"]["z"] = np.fromfile(fid, dtype=np.float32, count=Nrz)

    fid.seek(9 * 4 * recl)  # reposition to end of record 9
    Pos["r"]["r"] = np.fromfile(fid, dtype=np.float64, count=Nrr)

    if PlotType == "rectilin  ":
        Nrcvrs_per_range = Nrz
    elif PlotType == "irregular ":
        Nrcvrs_per_range = 1
    else:
        Nrcvrs_per_range = Nrz

    if freq is None:
        nread_freq = 1
    else:
        freq = np.array(freq)  # Ensure freq is a np array
        freq = np.reshape(
            freq, (freq.size,)
        )  # Ensure freq as one dimension (to avoid issue when freq is given as a scalar)
        nread_freq = freq.size

    pressure = np.zeros((nread_freq, Ntheta, Nsz, Nrcvrs_per_range, Nrr), dtype=complex)

    if xs is None or ys is None:
        if freq is not None:
            # Old version (prior 15/12/2023)
            # freqdiff = np.abs(freqVec - freq)
            # freq_idx = np.argmin(freqdiff)

            # Updated version to handle multiple frequencies reading at the same time
            freq_idx = np.array([np.argmin(np.abs(freqVec - f)) for f in freq])
        else:
            freq_idx = np.array([0])
        read_freq = freqVec[freq_idx]

        for ifreq in freq_idx:
            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = (
                            10
                            + ifreq * Ntheta * Nsz * Nrcvrs_per_range
                            + itheta * Nsz * Nrcvrs_per_range
                            + isz * Nrcvrs_per_range
                            + irz
                        )

                        status = fid.seek(recnum * 4 * recl)
                        if status == -1:
                            raise ValueError(
                                "Seek to specified record failed in readshd_bin"
                            )

                        temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                        pressure[ifreq, itheta, isz, irz, :] = (
                            temp[0::2] + 1j * temp[1::2]
                        )
        # Get rid of the useless first dimension in case of single frequency (mainly for coherence with other functions plotshd ...)
        if nread_freq == 1:
            pressure = pressure[0, ...]

    else:
        # TODO: this part of the function is inherited from the MATLAB function from AT and might not work anymore
        read_freq = None
        xdiff = np.abs(Pos["s"]["x"] - xs * 1000)
        idxX = np.argmin(xdiff)
        ydiff = np.abs(Pos["s"]["y"] - ys * 1000)
        idxY = np.argmin(ydiff)

        for itheta in range(Ntheta):
            for isz in range(Nsz):
                for irz in range(Nrcvrs_per_range):
                    recnum = (
                        10
                        + idxX * Nsy * Ntheta * Nsz * Nrz
                        + idxY * Ntheta * Nsz * Nrz
                        + itheta * Nsz * Nrz
                        + isz * Nrz
                        + irz
                    )

                    status = fid.seek(recnum * 4 * recl)
                    if status == -1:
                        raise ValueError(
                            "Seek to specified record failed in readshd_bin"
                        )

                    temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                    pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

    fid.close()

    return title, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure
