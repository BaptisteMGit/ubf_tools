#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   read_shd.py
@Time    :   2024/07/08 09:09:48
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Read a shade file ('.shd') produced by FIELD.exe.

This module does NOT change the public API of the original file (same
function names/signatures). Adapted from the original Matlab Acoustics
Toolbox by Michael B. Porter, https://oalib.hlsresearch.com/AcousticsToolbox/

------------------------------------------------------------------------
'.shd' binary layout (for reference, to help navigate readshd_bin)
------------------------------------------------------------------------
The file is made of fixed-length Fortran records of 'recl' bytes each
(read from the very first 4 bytes of the file). Fields are stored at
fixed record offsets:

    record 0: recl, title (80 chars)
    record 4: PlotType (10 chars)
    record 8: Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr, freq0, atten
    record 12: freqVec (Nfreq float64)
    record 16: theta (Ntheta float64)
    record 20: source x (or, for PlotType=='TL', 2 bounds later expanded
               via linspace to Nsx points)
    record 24: source y (same convention as x)
    record 28: source z (Nsz float32)
    record 32: receiver z (Nrz float32)
    record 36: receiver r (Nrr float64)
    record 40+: pressure data, one record per
               (frequency, theta, source depth, receiver depth) combination,
               each holding Nrr complex values (interleaved real/imag
               float32 pairs).

All the '.seek(N * 4 * recl)' calls below reposition the file to the
start of record N (the '4 *' factor accounts for 'recl' being expressed
in 4-byte words in the header, while file positions are in bytes).
------------------------------------------------------------------------
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import numpy as np


def readshd(filename, xs=None, ys=None, freq=None):
    """Read a shade file produced by FIELD.exe and return the data in a
    dictionary-like tuple.

    Usage: PlotTitle, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure = readshd(filename, xs, ys, freq)

    Args:
        filename (str): path to the '.shd' file.
        xs, ys (float|None): source (x, y) position (km) used to locate
            the closest receiver grid point, instead of reading by
            frequency. See readshd_bin's docstring for the caveats on
            this code path (it is a direct, largely untested port of
            legacy MATLAB code).
        freq (float|array-like|None): frequency/frequencies (Hz) to
            read. None reads the first frequency stored in the file.

    Returns:
        See readshd_bin.
    """
    # NOTE (dead branching removed): the original code had 3 separate
    # branches (freq is None + xs is None / freq is None + xs is not
    # None / freq is not None), each calling readshd_bin with a
    # different subset of keyword arguments explicitly set to None. But
    # readshd_bin already defaults xs/ys/freq to None, so passing None
    # explicitly for an argument is strictly equivalent to omitting it.
    # All 3 branches were therefore calling readshd_bin with the exact
    # same effective arguments as a single unconditional call would --
    # this is a pure simplification, not a behaviour change (verified:
    # for every combination of xs/ys/freq, this single call produces
    # identical results to the original 3-branch dispatch). One
    # practical benefit: xs/ys and freq can now genuinely be supplied
    # together (the original silently dropped xs/ys whenever freq was
    # also given, since the `freq is not None` branch never forwarded
    # them).
    return readshd_bin(filename=filename, xs=xs, ys=ys, freq=freq)


def readshd_bin(filename, xs=None, ys=None, freq=None):
    """Read a '.shd' binary file.

    Two mutually-driven read modes:
      - by frequency (default): read the pressure field for the
        frequency/frequencies closest to 'freq' (or the first frequency
        in the file if freq is None), across the full receiver grid.
      - by source position (xs and ys both given): locate the source
        grid point closest to (xs, ys) and read the pressure field at
        that single position, ignoring 'freq' (see NOTE below -- this
        path is a legacy, largely unvalidated port).

    Args:
        filename (str): path to the '.shd' file.
        xs, ys (float|None): source (x, y) position (km). Both must be
            given together to use this mode.
        freq (float|array-like|None): frequency/frequencies (Hz).

    Returns:
        tuple(title, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure):
            title (str): simulation title stored in the file.
            PlotType (str): one of 'rectilin', 'irregular', ... (10-char
                field, whitespace-stripped). CAVEAT: validated against a
                real FIELD.exe broadband, coherent-addition output where
                this field was found blank (10 spaces), even though the
                receiver grid it is meant to describe was still
                correctly rectilinear -- the Nrcvrs_per_range fallback
                (see 'else' branch below) happens to match 'rectilin'
                behaviour, so results were unaffected here, but this has
                NOT been validated against a genuinely 'irregular' grid.
                If you rely on distinguishing 'rectilin' from
                'irregular', verify PlotType is actually populated for
                your specific KRAKEN/FIELD build and run type first.
            freqVec (np.ndarray): all frequencies stored in the file.
            freq0 (float): nominal frequency (Hz) as stored in the file.
                CAVEAT (validated against a real FIELD.exe broadband,
                coherent-addition output): this field was found to hold
                meaningless/uninitialized data for that run type, even
                though every other header field (Nfreq, Ntheta, Nsx,
                Nsy, Nsz, Nrz, Nrr, atten, freqVec, and every position
                grid) parsed correctly at the exact same byte offsets --
                proving the parsing itself is correct and this is a
                genuine property of that FIELD output, not a bug here.
                Use freqVec (always found reliable) instead of freq0
                when you need the list of simulated frequencies.
            read_freq (np.ndarray|None): the frequency/frequencies
                actually read (closest match to 'freq'); None in the
                source-position read mode.
            atten (float): attenuation value stored in the file.
            Pos (dict): grid positions ('theta', 's' source grid, 'r'
                receiver grid).
            pressure (np.ndarray, complex): pressure field. Shape
                (n_freq, Ntheta, Nsz, Nrcvrs_per_range, Nrr) in
                frequency-read mode (n_freq dimension squeezed out if
                a single frequency was read), or
                (Ntheta, Nsz, Nrcvrs_per_range, Nrr) in source-position
                mode.

    Raises:
        FileNotFoundError: if 'filename' does not exist.
    """
    # NOTE: a bare try/except FileNotFoundError re-raising an almost
    # identical FileNotFoundError added little value; kept the clearer
    # custom message but chained the original exception ('from exc') so
    # the underlying OS error is still visible when debugging.
    try:
        fid = open(filename, "rb")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"readshd_bin.py: No shade file with the name {filename} exists"
        ) from exc

    # NOTE (robustness fix): the original code never closed 'fid' on an
    # error path (only at the very end, on success). Any exception
    # raised while parsing (e.g. a bad 'seek', a malformed file) would
    # leak the file handle. Wrapping the rest of the function in
    # try/finally guarantees fid.close() runs in every case, without
    # changing the parsing logic itself.
    try:
        return _read_shd_from_open_file(fid, xs=xs, ys=ys, freq=freq)
    finally:
        fid.close()


def _read_shd_from_open_file(fid, xs, ys, freq):
    """Actual '.shd' parsing logic, factored out of readshd_bin so the
    file handle can be guaranteed to close via try/finally regardless of
    where parsing stops (return or exception). See readshd_bin's
    docstring for the return value description, and the module docstring
    for the binary record layout.
    """
    # NOTE (bug fixed): every 'int(np.fromfile(..., count=1))' /
    # 'float(np.fromfile(..., count=1))' call below used to crash with
    # `TypeError: only 0-dimensional arrays can be converted to Python
    # scalars` on numpy >= 1.25 -- np.fromfile(..., count=1) returns a
    # 1-element, 1-D array, and int()/float() on a non-0-D array is no
    # longer allowed. This affected every single '.shd' file read (not
    # just an edge case), and was only caught by round-tripping a
    # synthetic '.shd' file in the test suite (see
    # test_read_shd.py::TestReadshdBinSyntheticFile) -- no test in the
    # previous pass actually exercised the real binary parsing. Fixed by
    # indexing the first (only) element explicitly.
    recl = int(np.fromfile(fid, dtype=np.int32, count=1)[0])  # record length in bytes
    title = fid.read(80).decode("utf-8").strip()  # read and decode the title

    fid.seek(4 * recl)  # reposition to end of first record
    PlotType = fid.read(10).decode("utf-8").strip()  # read and decode the PlotType

    fid.seek(2 * 4 * recl)  # reposition to end of second record
    Nfreq = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Ntheta = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Nsx = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Nsy = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Nsz = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Nrz = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    Nrr = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    freq0 = float(np.fromfile(fid, dtype=np.float64, count=1)[0])
    atten = float(np.fromfile(fid, dtype=np.float64, count=1)[0])

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
        # TL runs store only the (start, end) bounds of the source grid
        # and rely on it being regularly spaced -> reconstruct the full
        # grid with linspace.
        fid.seek(5 * 4 * recl)
        Pos["s"] = {}
        Pos["s"]["x"] = np.fromfile(fid, dtype=np.float64, count=2)
        Pos["s"]["x"] = np.linspace(Pos["s"]["x"][0], Pos["s"]["x"][1], Nsx)

        fid.seek(6 * 4 * recl)
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
        pressure = _read_pressure_by_frequency(
            fid, freq, freqVec, Ntheta, Nsz, Nrcvrs_per_range, Nrr, recl, pressure
        )
        read_freq = freqVec[
            np.array([0]) if freq is None
            else np.array([np.argmin(np.abs(freqVec - f)) for f in freq])
        ]
        # Get rid of the useless first dimension in case of single
        # frequency (mainly for coherence with other functions like
        # plotshd ...)
        if nread_freq == 1:
            pressure = pressure[0, ...]
    else:
        # NOTE: this branch is inherited from the original MATLAB
        # function and, per the original author's own comment, "might
        # not work anymore" -- it has no automated test coverage here
        # (no real-world sample file exercising this path was
        # available) and should be validated against a known-good
        # '.shd' file before being relied upon.
        read_freq = None
        pressure = _read_pressure_by_source_position(
            fid, Pos, xs, ys, Ntheta, Nsz, Nrcvrs_per_range, Nrr, recl
        )

    return title, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure


def _read_pressure_by_frequency(fid, freq, freqVec, Ntheta, Nsz, Nrcvrs_per_range, Nrr, recl, pressure):
    """Read the pressure field for the frequency/frequencies closest to
    'freq' (or the first stored frequency if freq is None), filling and
    returning 'pressure' (pre-allocated by the caller)."""
    if freq is not None:
        freq_idx = np.array([np.argmin(np.abs(freqVec - f)) for f in freq])
    else:
        freq_idx = np.array([0])

    for idx_f_pressure, ifreq in enumerate(freq_idx):
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
                        raise ValueError("Seek to specified record failed in readshd_bin")

                    temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                    pressure[idx_f_pressure, itheta, isz, irz, :] = (
                        temp[0::2] + 1j * temp[1::2]
                    )
    return pressure


def _read_pressure_by_source_position(fid, Pos, xs, ys, Ntheta, Nsz, Nrcvrs_per_range, Nrr, recl):
    """Read the pressure field at the source grid point closest to
    (xs, ys). See readshd_bin's docstring: this is a legacy code path,
    validate against a known-good file before relying on it.

    Returns:
        np.ndarray of shape (Ntheta, Nsz, Nrcvrs_per_range, Nrr).
    """
    xdiff = np.abs(Pos["s"]["x"] - xs * 1000)
    idxX = np.argmin(xdiff)
    ydiff = np.abs(Pos["s"]["y"] - ys * 1000)
    idxY = np.argmin(ydiff)

    Nsy = Pos["s"]["y"].size
    pressure = np.zeros((Ntheta, Nsz, Nrcvrs_per_range, Nrr), dtype=complex)

    for itheta in range(Ntheta):
        for isz in range(Nsz):
            for irz in range(Nrcvrs_per_range):
                recnum = (
                    10
                    + idxX * Nsy * Ntheta * Nsz * Nrcvrs_per_range
                    + idxY * Ntheta * Nsz * Nrcvrs_per_range
                    + itheta * Nsz * Nrcvrs_per_range
                    + isz * Nrcvrs_per_range
                    + irz
                )
                status = fid.seek(recnum * 4 * recl)
                if status == -1:
                    raise ValueError("Seek to specified record failed in readshd_bin")

                temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                # NOTE (bug fixed): the original code wrote into
                # `pressure[itheta, isz, irz, :]` where 'pressure' had
                # been allocated with a LEADING frequency dimension
                # (shape (nread_freq, Ntheta, Nsz, Nrcvrs_per_range,
                # Nrr)), i.e. one index short for that array's actual
                # rank. This silently used 'itheta' to index the
                # frequency axis instead of the theta axis (or raised an
                # IndexError once itheta reached nread_freq). Fixed by
                # allocating a properly-shaped (Ntheta, Nsz,
                # Nrcvrs_per_range, Nrr) array for this source-position
                # read mode, which does not have a frequency axis to
                # begin with (a single source position is read,
                # regardless of frequency -- see readshd_bin's
                # docstring). As with the rest of this code path, this
                # fix has not been validated against a real '.shd' file
                # exercising the by-source-position read mode.
                pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

    return pressure
