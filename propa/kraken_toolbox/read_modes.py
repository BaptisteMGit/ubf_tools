#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   read_modes.py
@Time    :   2024/07/08 09:09:27
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Read the modes produced by KRAKEN from a '.mod' binary file.

This module does NOT change the public API of the original file (same
function names/signatures). Adapted from the original Matlab Acoustics
Toolbox by Michael B. Porter, https://oalib.hlsresearch.com/AcousticsToolbox/

BUGS FIXED COMPARED TO THE ORIGINAL CODE:
  1. readmodes(): the extension-resolution logic used
     `os.path.basename(modfil).split(".")[0]` to strip the extension,
     which truncates the file root at the FIRST dot rather than the
     last one -- a file named e.g. "run.v2.mod" would be turned into
     "run" + ".mod" = "run.mod", silently losing the ".v2" part. Fixed
     with `os.path.splitext`, which only strips the last extension.
  2. readmodes_bin(): the file handle 'fid' was never closed (no
     'fid.close()' anywhere in the function) -- a resource leak on
     every call. The `if not hasattr(readmodes_bin, "fid"):` guard,
     presumably meant to cache/reuse an already-open handle across
     calls, never actually worked: `readmodes_bin.fid` is never
     assigned anywhere, so `hasattr(readmodes_bin, "fid")` is always
     False and the "first open" branch runs on every single call
     regardless. Since the rest of the function always seeks to
     absolute byte offsets (it does not rely on the file's current
     position from a previous call), removing this non-functional
     caching attempt changes nothing observable; the file is now
     properly opened and closed via a `with` block on every call.
  3. readmodes_bin(): `modes <= Modes["M"]` and `modes - 1` require
     'modes' to be a numpy array; passing a plain Python list for the
     'modes' argument crashed with a TypeError. Fixed by coercing
     'modes' to a numpy array immediately after the "read all modes"
     default is resolved.
  4. readmodes(): Top halfspace wavenumber used
     `Modes["freqVec"][0]` (always the FIRST frequency in the file)
     while the structurally identical Bottom halfspace calculation used
     `Modes["freqVec"][freq_index]` (the frequency actually matching
     the user's request). This asymmetry looks like a copy-paste slip;
     fixed to use freq_index consistently for both. NOTE: this fix
     could not be validated against a real '.mod' file with more than
     one stored frequency (none was available), so please double-check
     the Top halfspace wavenumber against a known-good reference before
     relying on it for a genuinely multi-frequency mode file.
  5. readmodes_bin(): Top/Bottom halfspace 'rho' was stored as a
     1-element numpy array when read from the file (acousto-elastic
     boundary), but as a plain Python float (1.0) in readmodes()'s
     vacuum-boundary fallback. Normalized to always be a plain float,
     so downstream code does not need to special-case the boundary
     condition just to read this value.

These are documented inline with "NOTE (bug ...)" wherever they occur.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np


def readmodes(modfil, freq=0, modes=None):
    """Read the modes produced by KRAKEN from a '.mod' binary file.

    Usage: Modes = readmodes(filename, freq, modes)
    filename can include the extension or not (any extension is
    replaced with '.mod').

    Args:
        modfil (str): path to the '.mod' file (extension optional).
        freq (float): frequency (Hz) to read; the closest frequency
            actually stored in the file is used. 0 selects the first
            frequency in the file.
        modes (array-like|None): optional list/array of 1-based mode
            indices to read. None reads every mode.

    Returns:
        dict: 'Modes', see readmodes_bin's docstring for its content,
        plus (if at least one mode was found) 'Top'/'Bot' halfspace
        wavenumber info ('k2', 'gamma', 'phi') added by this function.

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter
    https://oalib.hlsresearch.com/AcousticsToolbox/
    """
    # NOTE (bug fixed): the original extension-resolution logic used
    # `os.path.basename(modfil).split(".")[0]`, which truncates at the
    # FIRST dot in the filename rather than the last one -- a file named
    # e.g. "run.v2.mod" would lose the ".v2" part. `os.path.splitext`
    # only strips the last extension, which is what "give me a '.mod'
    # file no matter what extension was passed" actually requires.
    file_root, _ext = os.path.splitext(modfil)
    modfil = file_root + ".mod"

    Modes = readmodes_bin(modfil, freq, modes)

    # Identify the index of the frequency closest to the user-specified value
    freq_diff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freq_diff)

    # Calculate wavenumbers in halfspaces (if there are any modes)
    if Modes["M"] != 0:
        if Modes["Top"]["BC"] == "A":  # Top
            # NOTE (bug fixed): used `Modes["freqVec"][0]` (always the
            # first stored frequency) instead of `Modes["freqVec"][freq_index]`
            # (the frequency actually requested/matched) -- inconsistent
            # with the structurally identical Bottom calculation just
            # below. Not validated against a genuinely multi-frequency
            # '.mod' file (see module docstring).
            Modes["Top"]["k2"] = (
                2 * np.pi * Modes["freqVec"][freq_index] / Modes["Top"]["cp"]
            ) ** 2
            gamma2 = Modes["k"] ** 2 - Modes["Top"]["k2"]
            Modes["Top"]["gamma"] = np.sqrt(gamma2)  # Vertical wavenumber
            Modes["Top"]["phi"] = Modes["phi"][0, :]  # Mode value at halfspace
        else:
            Modes["Top"]["rho"] = 1.0
            Modes["Top"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Top"]["phi"] = np.zeros_like(Modes["phi"][0, :])

        if Modes["Bot"]["BC"] == "A":  # Bottom
            Modes["Bot"]["k2"] = (
                2 * np.pi * Modes["freqVec"][freq_index] / Modes["Bot"]["cp"]
            ) ** 2
            gamma2 = Modes["k"] ** 2 - Modes["Bot"]["k2"]
            Modes["Bot"]["gamma"] = np.sqrt(gamma2)  # Vertical wavenumber
            Modes["Bot"]["phi"] = Modes["phi"][-1, :]  # Mode value at halfspace
        else:
            Modes["Bot"]["rho"] = 1.0
            Modes["Bot"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Bot"]["phi"] = np.zeros_like(Modes["phi"][-1, :])

    return Modes


def readmodes_bin(filename, freq=0, modes=None):
    """Read the modes '.mod' binary file.

    Args:
        filename (str): path to the '.mod' file (exact name, extension
            included -- see readmodes() for extension resolution).
        freq (float): frequency (Hz) to read; the closest frequency
            actually stored in the file is used.
        modes (array-like|None): optional list/array of 1-based mode
            indices to read. None reads every mode found at the
            selected frequency.

    Returns:
        dict with (amongst others): 'title', 'Nfreq', 'Nmedia', 'N'
        (points per medium), 'Mater' (material per medium), 'depth',
        'rho', 'freqVec', 'z', 'M' (number of modes at the selected
        frequency), 'selected_modes', 'nb_selected_modes', 'Top', 'Bot'
        (halfspace info), 'phi' (mode shapes), 'k' (complex
        wavenumbers).

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter
    https://oalib.hlsresearch.com/AcousticsToolbox/
    """
    # NOTE (bug fixed): the original code guarded the file-opening block
    # with `if not hasattr(readmodes_bin, "fid"):`, apparently intending
    # to cache/reuse an already-open file handle across repeated calls.
    # This never worked: `readmodes_bin.fid` is not assigned anywhere in
    # the function, so the attribute never exists and the guard's
    # condition is always True -- the "first open" branch ran on every
    # call regardless, with no actual caching taking place. Combined
    # with the complete absence of a matching `fid.close()`, every call
    # leaked one open file handle. Since the rest of the function always
    # seeks to absolute byte offsets (never relies on a handle's
    # position left over from a previous call), removing this
    # non-functional caching attempt changes no observable behaviour;
    # a single `with open(...) as fid:` now guarantees the file is
    # always closed, on every exit path (including the early
    # `if Ntot < 0: return Modes` below).
    with open(filename, "rb") as fid:
        return _read_modes_from_open_file(fid, freq=freq, modes=modes)


def _read_modes_from_open_file(fid, freq, modes):
    """Actual '.mod' parsing logic, factored out of readmodes_bin so the
    file handle can be guaranteed to close via the caller's 'with'
    block regardless of where parsing stops (return or exception)."""
    iRecProfile = 1  # (first time only)
    lrecl = (
        4 * np.fromfile(fid, dtype=np.int32, count=1)[0]
    )  # This is converted to bytes. Fortran versions use words instead

    rec = iRecProfile - 1
    fid.seek(rec * lrecl + 4)

    Modes = {}

    Modes["title"] = fid.read(80).decode("utf-8").strip()
    Modes["Nfreq"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    Modes["Nmedia"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    Ntot = np.fromfile(fid, dtype=np.int32, count=1)[0]
    NMat = np.fromfile(fid, dtype=np.int32, count=1)[0]

    if Ntot < 0:
        return Modes

    # N and Mater
    rec = iRecProfile
    fid.seek(rec * lrecl)

    Modes["N"] = np.zeros(Modes["Nmedia"], dtype=np.int32)
    Modes["Mater"] = np.empty(Modes["Nmedia"], dtype=object)

    for Medium in range(Modes["Nmedia"]):
        Modes["N"][Medium] = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Mater = fid.read(8).decode("utf-8").strip()
        Modes["Mater"][Medium] = Mater

    # Depth and density
    rec = iRecProfile + 1
    fid.seek(rec * lrecl)

    bulk = np.fromfile(fid, dtype=np.float32, count=2 * Modes["Nmedia"]).reshape(
        2, Modes["Nmedia"]
    )
    Modes["depth"] = bulk[0, :]
    Modes["rho"] = bulk[1, :]

    # Frequencies
    rec = iRecProfile + 2
    fid.seek(rec * lrecl)
    Modes["freqVec"] = np.fromfile(fid, dtype=np.float64, count=Modes["Nfreq"])

    # z
    rec = iRecProfile + 3
    fid.seek(rec * lrecl)
    Modes["z"] = np.fromfile(fid, dtype=np.float32, count=Ntot)

    # Identify the index of the frequency closest to the user-specified value
    freqdiff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freqdiff)

    iRecProfile = iRecProfile + 4
    rec = iRecProfile

    # Skip through the mode file to get to the chosen frequency
    for ifreq in range(freq_index + 1):
        fid.seek(rec * lrecl)
        Modes["M"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

        if ifreq < freq_index:
            iRecProfile = (
                iRecProfile + 3 + Modes["M"] + int(4 * (2 * Modes["M"] - 1) / lrecl)
            )  # Advance to the next profile
            rec = iRecProfile

    if modes is None:
        modes = np.arange(1, Modes["M"] + 1)  # Read all modes if the user didn't specify
    else:
        # NOTE (bug fixed): 'modes <= Modes["M"]' and 'modes - 1' further
        # below both require 'modes' to be a numpy array. A plain Python
        # list (a perfectly reasonable thing to pass given the
        # docstring says "array-like") raised a TypeError. The
        # 'np.arange(...)' branch above already produces an array, so
        # only the user-supplied case needed coercing.
        modes = np.atleast_1d(np.asarray(modes))

    # Don't try to read modes that don't exist
    ii = modes <= Modes["M"]
    modes = modes[ii]

    Modes["selected_modes"] = modes
    Modes["nb_selected_modes"] = len(modes)

    # Read top and bottom halfspace info

    # Top
    rec = iRecProfile + 1
    fid.seek(rec * lrecl)
    Modes["Top"] = {}
    Modes["Top"]["BC"] = fid.read(1).decode("utf-8")
    cp_real, cp_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Top"]["cp"] = complex(cp_real, cp_imag)
    cs_real, cs_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Top"]["cs"] = complex(cs_real, cs_imag)
    # NOTE (bug fixed): 'rho'/'depth' used to be stored as 1-element
    # numpy arrays here, while readmodes()'s vacuum-boundary fallback
    # sets 'rho' to a plain float (1.0) -- an inconsistency that forced
    # any downstream code to special-case the boundary condition just to
    # read this value. Normalized to a plain float/consistent scalar.
    Modes["Top"]["rho"] = float(np.fromfile(fid, dtype=np.float32, count=1)[0])
    Modes["Top"]["depth"] = float(np.fromfile(fid, dtype=np.float32, count=1)[0])

    # Bottom
    Modes["Bot"] = {}
    Modes["Bot"]["BC"] = fid.read(1).decode("utf-8")
    cp_real, cp_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Bot"]["cp"] = complex(cp_real, cp_imag)
    cs_real, cs_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Bot"]["cs"] = complex(cs_real, cs_imag)
    Modes["Bot"]["rho"] = float(np.fromfile(fid, dtype=np.float32, count=1)[0])
    Modes["Bot"]["depth"] = float(np.fromfile(fid, dtype=np.float32, count=1)[0])

    # Read the modes (eigenfunctions, then eigenvalues)
    rec = iRecProfile
    fid.seek(rec * lrecl)

    if Modes["M"] == 0:
        Modes["phi"] = np.array([])  # No modes
        Modes["k"] = np.array([])
    else:
        Modes["phi"] = np.zeros((NMat, len(modes)), dtype=np.complex64)  # Number of modes

        for ii in range(len(modes)):
            rec = iRecProfile + 1 + modes[ii]
            fid.seek(rec * lrecl)
            phi = np.fromfile(fid, dtype=np.float32, count=2 * NMat).reshape(NMat, 2)
            phi_real, phi_imag = phi[:, 0], phi[:, 1]
            phi = phi_real + 1j * phi_imag
            Modes["phi"][:, ii] = phi

        rec = iRecProfile + 2 + Modes["M"]
        fid.seek(rec * lrecl)

        Modes["k"] = np.zeros(Modes["M"], dtype=np.complex64)
        k = np.fromfile(fid, dtype=np.float32, count=2 * Modes["M"]).reshape(
            Modes["M"], 2
        )
        k_real, k_imag = k[:, 0], k[:, 1]
        Modes["k"] = k_real + 1j * k_imag
        Modes["k"] = Modes["k"][modes - 1]

    iRecProfile = (
        iRecProfile + 4 + Modes["M"] + int(4 * (2 * Modes["M"] - 1) / lrecl)
    )  # Advance to the next profile

    return Modes
