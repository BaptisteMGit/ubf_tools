import os
import numpy as np
from cst import C0


def runkraken(filename):
    # Run Fortran version of Kraken
    os.system(f"kraken {filename}")

    # Run Fortran version of Field
    os.system(f"field {filename}")


def runfield(filename):
    # Run Fortran version of Field
    os.system(f"field {filename}")


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

    n_align = 35
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
