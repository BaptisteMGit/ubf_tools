import os
import numpy as np
from cst import C0
from propa.kraken_toolbox.read_shd import readshd


def get_component(Modes, comp):
    components = {"H": 0, "V": 1, "T": 2, "N": 3}

    if comp not in components:
        raise Exception("Fatal Error in get_component: Unknown component")

    comp_index = components[comp]
    num_modes = Modes["phi"].shape[1]

    phi = np.zeros((len(Modes["z"]), num_modes), dtype=np.complex64)

    k = 0

    for medium in range(Modes["Nmedia"]):
        for ii in range(len(Modes["z"])):
            if k >= Modes["phi"].shape[0]:
                return phi

            material = Modes["Mater"][medium]

            if material == "ACOUSTIC":
                phi[ii] = Modes["phi"][k]
                k += 1
            elif material == "ELASTIC":
                phi[ii] = Modes["phi"][k + comp_index]
                k += 4
            else:
                raise Exception("Fatal Error in get_component: Unknown material type")

    return phi


def align_var_description(var_line, desc):
    """
    Add variable description at the end of the line
    :param var_line:
    :param desc:
    :return:
    """

    n_align = 55
    blank_space = (max(n_align - len(var_line), 3)) * " "
    return var_line + blank_space + f" ! {desc}\n"


def default_nb_rcv_z(fmax, max_depth, n_per_l=7):
    # Jensen et al 2000 : advise between 5 and 10 points per wavelength p.446
    if n_per_l < 5:
        n_per_l = 5
    elif n_per_l > 10:
        n_per_l = 10

    lmin = C0 / fmax
    nz = int(np.ceil(max_depth / lmin * n_per_l))
    return nz


def waveguide_cutoff_freq(max_depth, c0=C0):
    fc = c0 / (4 * max_depth)
    return fc


def get_rcv_pos_idx(
    kraken_range=None,
    kraken_depth=None,
    shd_fpath=None,
    rcv_depth=None,
    rcv_range=None,
):
    if kraken_range is None and kraken_depth is None:
        # Dummy read to get frequencies used by kraken and field grid information
        _, _, _, _, _, _, field_pos, pressure = readshd(filename=shd_fpath, freq=0)
        nr = pressure.shape[2]
        nz = pressure.shape[1]
        kraken_range = field_pos["r"]["r"]
        kraken_depth = field_pos["r"]["z"]
    else:
        nr = kraken_range.size
        nz = kraken_depth.size
        field_pos = None

    if rcv_range is not None:
        # No need to process the entire grid : extract pressure field at desired positions
        rcv_pos_idx_r = [
            np.nanargmin(np.abs(kraken_range - rcv_r)) for rcv_r in rcv_range
        ]
    else:
        rcv_pos_idx_r = range(nr)

    if rcv_depth is not None:
        rcv_pos_idx_z = [
            np.nanargmin(np.abs(kraken_depth - rcv_z)) for rcv_z in rcv_depth
        ]
    else:
        rcv_pos_idx_z = range(nz)

    rr, zz = np.meshgrid(rcv_pos_idx_r, rcv_pos_idx_z)

    return rr, zz, field_pos
