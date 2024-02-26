import os
import numpy as np
from cst import C0
from propa.kraken_toolbox.read_shd import readshd
from scipy.optimize import minimize


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


""" Find optimal intervals for parallel processing """


def find_optimal_intervals(fmin, fmax, nf, n_workers, mean_cpu_time=None, z=None):
    """
    Find optimal intervals for parallel processing based on cpu time. The function finds optimal frequency intervals
     bounds so that the cpu time is balanced between workers."""

    # If z is not provided, use known polynomial fit coef (computed on the range 0 - 100 Hz)
    if z is None:
        # z = [0.0016158, -0.01881528, 0.49200085]  # win = 20
        # z = [0.00173842, -0.05340321, 1.02745305]  # win = 40
        # z = [0.00168555, -0.04056279, 0.79734153]  # win = 30
        z = [0.00181359, -0.06963699, 1.38239787]  # win = 50

    if mean_cpu_time is None:
        mean_cpu_time = 5.2  # Mean cpu time for the range 0 - 100 Hz (s)

    # Initial guess
    fi = np.linspace(fmin, fmax, n_workers + 1)
    fi = fi[1:-1]
    alpha = mean_cpu_time * nf
    x0 = np.array([alpha, *fi])

    # Bounds
    bounds = [(0, alpha), *[(fmin, fmax)] * len(fi)]
    res = minimize(objective_function, x0, args=(z, fmin, fmax), bounds=bounds)
    expected_cpu_time = res.x[0]
    freq_bounds = [fmin, *res.x[1:], fmax]

    return expected_cpu_time, freq_bounds


def objective_function(x, z, fmin, fmax):
    Y = build_y(x, z=z, fmin=fmin, fmax=fmax)
    return np.sum(Y**2)


def build_y(x, z, fmin, fmax):
    alpha = x[0]
    fi = x[1:]
    Y = np.array([g(fi, alpha, k, z, fmin, fmax) for k in range(len(fi) + 1)])
    # print(f"alpha = {alpha}")
    # print(f"fi = {fi}")
    # print(f"Y = {Y}")
    return Y


def g(fi, alpha, k, z, fmin, fmax):
    x = [fmin, *fi, fmax]
    a, b, c = z
    gk = (
        a / 3 * (x[k + 1] ** 3 - x[k] ** 3)
        + b / 2 * (x[k + 1] ** 2 - x[k] ** 2)
        + c * (x[k + 1] - x[k])
        - alpha
    )
    return gk
