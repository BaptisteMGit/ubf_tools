import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from propa.kraken_toolbox.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenEnv,
    KrakenFlp,
    Bathymetry,
)
from propa.kraken_toolbox.utils import default_nb_rcv_z
from localisation.verlinden.misc.AcousticComponent import AcousticSource

from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE


"""
Test case based on the paper Computation of Broadband Sound Signal Propagation in a Shallow Water Environment
with Sinusoidal Bottom using a 3-D PE Model by Frédéric Sturm 
"""

ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\sturm"
SOURCE_DEPTH = 25  # m
MAX_DEPTH = 200  # m
MIN_DEPTH = 100  # m
MAX_RANGE = 20  # km
FMAX = 40  # Hz

RCV_DEPTH = np.linspace(10, 200, 20)  # m
RCV_RANGE = [20000]  # m
DELAYS = np.array(RCV_RANGE) / 1500

# Bathymetry
THETA = 94 * np.pi / 180
RANGE_PERIODICITY = 6  # km
FR = 1 / RANGE_PERIODICITY
DR = 1 / (20 * FR)


def generate_bathy():
    # Define bathymetry
    r = np.arange(0, MAX_RANGE, DR)
    # h = np.linspace(100, 200, len(r))
    h = 150 - 50 * np.sin(2 * np.pi * r * np.cos(THETA) / RANGE_PERIODICITY - np.pi / 2)
    # h = 150 - 50 * np.cos(2 * np.pi * r / RANGE_PERIODICITY)

    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        "bathy.csv", index=False, header=False
    )

    plt.figure(figsize=(16, 8))
    plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=10)
    plt.ylim([0, 200])
    plt.fill_between(r, h, 200, color="lightgrey")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (km)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Depth (m)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.grid()
    plt.savefig("bathy.png")


def signal():
    # Define Gaussian pulse signal
    fc = 25  # Hz
    fs = 4 * fc  # Hz
    T = 0.4  # s
    t = np.arange(-T / 2, T / 2, 1 / fs)
    s = np.cos(2 * np.pi * fc * t) * np.exp(-((5 * np.pi * t) ** 2))

    plt.figure(figsize=(16, 8))
    plt.plot(t, s)
    plt.xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Amplitude", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.grid()
    plt.tight_layout()
    plt.savefig("signal.png")

    return s, t


def testcase():
    # Top halfspace
    top_hs = KrakenTopHalfspace()
    # SSP
    ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values

    medium = KrakenMedium(
        ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=500
    )

    # Attenuation
    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    # Field
    n_rcv_z = default_nb_rcv_z(FMAX, MAX_DEPTH, n_per_l=15)
    field = KrakenField(
        src_depth=SOURCE_DEPTH,
        phase_speed_limits=[0, 20000],
        n_rcv_z=n_rcv_z,
        rcv_z_max=MAX_DEPTH,
    )

    # Range dependent bathymetry
    bathy = Bathymetry(
        data_file=os.path.join(ROOT, "bathy.csv"),
        interpolation_method="linear",
        units="km",
    )

    # Source signal
    fc = 25  # Hz
    fs = 4 * fc
    s, t = signal()
    Tr = 4
    nfft = int(fs * Tr)  # Number of points for FFT
    source = AcousticSource(s, t, waveguide_depth=bathy.bathy_depth.min(), nfft=nfft)
    source.display_source()
    plt.savefig("source.png")

    bott_hs_properties = {
        "rho": 1.5,
        "c_p": 1700.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s)
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    env_filename = "sturm_testcase"
    env = KrakenEnv(
        title="STURM test case",
        env_root=ROOT,
        env_filename=env_filename,
        freq=source.kraken_freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
        rModes=np.arange(0, MAX_RANGE, DR),
    )

    env.write_env()
    flp = KrakenFlp(
        env=env,
        src_depth=SOURCE_DEPTH,
        mode_theory="coupled",
        rcv_r_max=MAX_RANGE,
        rcv_z_max=MAX_DEPTH,
        nb_modes=50,
    )
    flp.write_flp()

    return env, flp, source


if __name__ == "__main__":
    os.chdir(ROOT)
    generate_bathy()
    testcase()
    # pd.read_csv("bathy.csv", header=None, names=["r", "h"]).plot(x="r", y="h")
    # plt.show()
    # signal()
