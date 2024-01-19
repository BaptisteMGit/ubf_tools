import os
import numpy as np


def readmodes(modfil, freq=0, modes=None):
    """Read the modes produced by KRAKEN from a '.mod' binary file.

    Usage: Modes = read_modes(filename, freq, modes)
    filename can include the extension or not

    freq is the frequency, 0 selects the first frequency in the file
    modes is an optional vector of mode indices

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    # Identify the file type
    file_root = os.path.join(
        os.path.dirname(modfil), os.path.basename(modfil).split(".")[0]
    )
    ext = "." + os.path.basename(modfil).split(".")[-1]
    # file_path, file_root, ext = os.path.splitext(modfil)

    if not ext or ext != ".mod":
        ext = ".mod"  # Use this as the default extension if none is specified

    modfil = file_root + ext

    # Read the modal data
    Modes = readmodes_bin(modfil, freq, modes)

    # Identify the index of the frequency closest to the user-specified value
    freq_diff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freq_diff)

    # Calculate wavenumbers in halfspaces (if there are any modes)
    if Modes["M"] != 0:
        if Modes["Top"]["BC"] == "A":  # Top
            Modes["Top"]["k2"] = (
                2 * np.pi * Modes["freqVec"][0] / Modes["Top"]["cp"]
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

    Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

    """

    if not hasattr(readmodes_bin, "fid"):
        fid = open(filename, "rb")
        if not fid:
            raise Exception("Mode file does not exist")

        iRecProfile = 1  # (first time only)
        lrecl = (
            4 * np.fromfile(fid, dtype=np.int32, count=1)[0]
        )  # This is converted to bytes. Fortran versions use words instead

    rec = iRecProfile - 1
    fid.seek(rec * lrecl + 4)

    # Initialize Modes dictionary
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

    # Skip through frequencies to get to the selected set

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
        modes = np.arange(
            1, Modes["M"] + 1
        )  # Read all modes if the user didn't specify

    # Don't try to read modes that don't exist
    ii = modes <= Modes["M"]
    modes = np.array(modes)[ii]

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
    Modes["Top"]["rho"] = np.fromfile(fid, dtype=np.float32, count=1)
    Modes["Top"]["depth"] = np.fromfile(fid, dtype=np.float32, count=1)

    # Bottom
    Modes["Bot"] = {}
    Modes["Bot"]["BC"] = fid.read(1).decode("utf-8")
    cp_real, cp_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Bot"]["cp"] = complex(cp_real, cp_imag)
    cs_real, cs_imag = np.fromfile(fid, dtype=np.float32, count=2)
    Modes["Bot"]["cs"] = complex(cs_real, cs_imag)
    Modes["Bot"]["rho"] = np.fromfile(fid, dtype=np.float32, count=1)
    Modes["Bot"]["depth"] = np.fromfile(fid, dtype=np.float32, count=1)

    # Read the modes (eigenfunctions, then eigenvalues)
    rec = iRecProfile
    fid.seek(rec * lrecl)

    if Modes["M"] == 0:
        Modes["phi"] = np.array([])  # No modes
        Modes["k"] = np.array([])
    else:
        Modes["phi"] = np.zeros(
            (NMat, len(modes)), dtype=np.complex64
        )  # Number of modes

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
