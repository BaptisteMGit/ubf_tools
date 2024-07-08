import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from signals import ricker_pulse
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
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.plot_utils import plotshd_from_pressure_field, plotshd
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal,
    postprocess_received_signal_from_broadband_pressure_field,
)

ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\saclantcen"
SOURCE_DEPTH = 100  # m
MAX_DEPTH = 200  # m
MIN_DEPTH = 100  # m
MAX_RANGE = 10  # km
FMAX = 1000  # Hz

RCV_DEPTH = [20]  # m
RCV_RANGE = [5000, 10000]  # m
DELAYS = np.array(RCV_RANGE) / 1480
DR = 0.1  # km


def common_features():
    # Top halfspace
    top_hs = KrakenTopHalfspace()
    # SSP
    ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values

    medium = KrakenMedium(
        ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=5000
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
        data_file=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\saclantcen\bathy_data.csv",
        interpolation_method="linear",
        units="km",
    )

    return top_hs, medium, att, field, z_ssp, bathy


def testcase_1():
    """Test case 1 : flat bottom 200m deep problem."""
    # Source signal
    fc = 200
    fs = 4 * fc
    s, t = ricker_pulse(fc=fc, fs=fs, T=0.05, center=True)
    Tr = 1
    nfft = fs * Tr  # Number of points for FFT
    source = AcousticSource(s, t, waveguide_depth=MIN_DEPTH, nfft=nfft)
    source.display_source()

    top_hs, medium, att, field, z_ssp, __ = common_features()
    medium.nmesh_ = 0

    bott_hs_properties = {
        "rho": 2.0,
        "c_p": 1600.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s)
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    env_filename = "saclantcen_testcase_1"
    env = KrakenEnv(
        title="SACLANTCEN test case 1",
        env_root=ROOT,
        env_filename=env_filename,
        freq=source.kraken_freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
    )

    env.write_env()
    flp = KrakenFlp(
        env=env,
        src_depth=SOURCE_DEPTH,
        mode_theory="coupled",
        rcv_r_max=MAX_RANGE,
        rcv_z_max=MAX_DEPTH,
        nb_modes=47,
    )
    flp.write_flp()

    return env, flp, source


def testcase_2():
    """Test case 2 : symmetric upslope/downslope environment."""
    # Source signal
    fc = 200
    fs = 4 * fc
    s, t = ricker_pulse(fc=fc, fs=fs, T=0.05, center=True)
    Tr = 1
    nfft = int(fs * Tr)  # Number of points for FFT
    source = AcousticSource(s, t, waveguide_depth=MIN_DEPTH, nfft=nfft)
    source.display_source()

    top_hs, medium, att, field, z_ssp, bathy = common_features()
    bott_hs_properties = {
        "rho": 2.0,
        "c_p": 1600.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s)
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    env_filename = "saclantcen_testcase_2"
    env = KrakenEnv(
        title="SACLANTCEN test case 2",
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
        nb_modes=47,
    )
    flp.write_flp()

    return env, flp, source


def testcase_3():
    """Test case 3 : symmetric upslope/downslope environment."""
    # Source signal
    fc = 200
    s, t = ricker_pulse(fc=fc, fs=4 * fc, T=0.05, center=True)
    source = AcousticSource(s, t, waveguide_depth=MIN_DEPTH)
    source.display_source()

    top_hs, medium, att, field, z_ssp, bathy = common_features()

    bott_hs_properties = {
        "rho": 2.0,
        "c_p": 1800.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s)
        "a_p": 0.1,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    env_filename = "saclantcen_testcase_3"
    env = KrakenEnv(
        title="SACLANTCEN test case 3",
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
        nb_modes=155,
    )
    flp.write_flp()

    return env, flp, source


def run_testcase(testcase=1, freq_to_plot=200):
    if testcase == 1:
        env, flp, source = testcase_1()
    elif testcase == 2:
        env, flp, source = testcase_2()
    else:
        env, flp, source = testcase_3()

    if testcase == 1:
        runkraken(env.filename)
        time_vector, s_at_rcv_pos, Pos = postprocess_received_signal(
            shd_fpath=os.path.join(working_dir, env.filename + ".shd"),
            source=source,
            rcv_range=RCV_RANGE,
            rcv_depth=RCV_DEPTH,
            apply_delay=True,
            delay=DELAYS,
        )
    else:
        broadband_pressure_field = runkraken(
            env=env,
            flp=flp,
            frequencies=source.kraken_freq,
        )

        (
            time_vector,
            s_at_rcv_pos,
            Pos,
        ) = postprocess_received_signal_from_broadband_pressure_field(
            shd_fpath=os.path.join(working_dir, env.filename + ".shd"),
            broadband_pressure_field=broadband_pressure_field,
            frequencies=source.kraken_freq,
            source=source,
            rcv_range=RCV_RANGE,
            rcv_depth=RCV_DEPTH,
            apply_delay=True,
            minimum_waveguide_depth=env.bathy.bathy_depth.min(),
        )

    ds = xr.Dataset(
        {
            "s_at_rcv_pos": (
                ["time", "rcv_depth", "rcv_range"],
                s_at_rcv_pos.astype(np.float32),
            ),
            "broadband_pressure_field_real": (
                ["frequency", "Ntheta", "Nsz", "z", "r"],
                np.real(broadband_pressure_field.astype(np.float32)),
            ),
            "broadband_pressure_field_imag": (
                ["frequency", "Ntheta", "Nsz", "z", "r"],
                np.imag(broadband_pressure_field.astype(np.float32)),
            ),
        },
        coords={
            "time": time_vector.astype(np.float32),
            "rcv_depth": RCV_DEPTH,
            "rcv_range": RCV_RANGE,
            "frequency": source.kraken_freq.astype(np.float32),
            "Ntheta": [0],  # Dummy params to match kraken output
            "Nsz": [0],  # Dummy params to match kraken output
            "z": Pos["r"]["z"],
            "r": Pos["r"]["r"],
        },
    )

    ds.to_netcdf(f"saclantcen_testcase{testcase}.nc")

    if testcase == 1:
        plotshd(
            env.filename + ".shd",
            title=f"{env.simulation_title} - f = {freq_to_plot} Hz",
            freq=freq_to_plot,
        )
    else:
        ifreq = np.argmin(np.abs(source.kraken_freq - freq_to_plot))
        freq_to_plot = source.kraken_freq[ifreq]
        pressure_field_to_plot = broadband_pressure_field[ifreq, ...]
        plotshd_from_pressure_field(
            env.filename + ".shd",
            pressure_field=pressure_field_to_plot,
            title=f"{env.simulation_title} - f = {freq_to_plot} Hz",
            freq=freq_to_plot,
        )
        plt.savefig(f"saclantcen_testcase{testcase}_f{freq_to_plot}Hz.png")

    for ir, r in enumerate(RCV_RANGE):
        for iz, z in enumerate(RCV_DEPTH):
            plt.figure(figsize=(16, 8))
            plt.plot(time_vector, s_at_rcv_pos[:, iz, ir])
            plt.xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
            plt.ylabel("Pressure [Pa]", fontsize=LABEL_FONTSIZE)
            plt.xticks(fontsize=TICKS_FONTSIZE)
            plt.yticks(fontsize=TICKS_FONTSIZE)
            plt.title(
                f"Received signal (r={r}m, z={z}m)\n",
                fontsize=TITLE_FONTSIZE,
            )

            plt.tight_layout()
            plt.savefig(f"saclantcen_testcase{testcase}_r{r}m_z{z}m.png")

    # plt.show()


if __name__ == "__main__":
    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\saclantcen"
    os.chdir(working_dir)

    run_testcase(testcase=2, freq_to_plot=200)
