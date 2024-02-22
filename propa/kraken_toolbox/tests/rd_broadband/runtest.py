import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.plot_utils import plotshd_from_pressure_field
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

from propa.kraken_toolbox.post_process import (
    postprocess_received_signal,
    postprocess_received_signal_from_broadband_pressure_field,
)
from signals import pulse
from localisation.verlinden.AcousticComponent import AcousticSource


def define_test_env():
    top_hs = KrakenTopHalfspace()

    ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values
    medium = KrakenMedium(
        ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=5000
    )

    bott_hs_properties = {
        "rho": 1.5,
        "c_p": 1506.50,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }  # stepK properties

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
    field = KrakenField(
        src_depth=18,
        phase_speed_limits=[0, 20000],
        n_rcv_z=5001,
        rcv_z_max=3000,
    )

    # Range dependent bathymetry
    bathy = Bathymetry(
        data_file=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband\bathy_data.csv",
        interpolation_method="linear",
        units="km",
    )

    env_filename = "test_kraken_rd_broadband"

    return top_hs, medium, att, bott_hs, field, bathy, env_filename


def test():
    rcv_range = np.array([5000, 30000])
    rcv_depth = [20, 600, 2000]

    fc = 25
    T = 10
    s, t = pulse(T=1, f=fc, fs=4 * fc)

    window = np.hanning(s.size)
    s *= window

    dr = 1

    top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env()
    source = AcousticSource(s, t, waveguide_depth=bathy.bathy_depth.min())

    env = KrakenEnv(
        title="Test de la classe KrakenEnv",
        env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband",
        env_filename=env_filename,
        freq=100,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
        rModes=np.arange(1, 30, dr),
    )

    freqs = source.kraken_freq
    src_depth = 18
    rcv_r_max = 30
    rcv_z_max = 3000

    flp = KrakenFlp(
        env=env,
        src_depth=src_depth,
        mode_theory="coupled",
        rcv_r_max=rcv_r_max,
        rcv_z_max=rcv_z_max,
    )

    broadband_pressure_field, _ = runkraken(env=env, flp=flp, frequencies=freqs)

    # for ifreq in range(len(freqs)):
    for ifreq in [0, 250, 400]:
        pressure_field_to_plot = broadband_pressure_field[ifreq, ...]
        plotshd_from_pressure_field(
            env_filename + ".shd",
            pressure_field=pressure_field_to_plot,
            tl_min=60,
            tl_max=110,
            title=f"Step K - f = {freqs[ifreq]} Hz",
            freq=freqs[ifreq],
        )
    plt.show()

    (
        time_vector,
        s_at_rcv_pos,
        Pos,
    ) = postprocess_received_signal_from_broadband_pressure_field(
        shd_fpath=os.path.join(working_dir, env_filename + ".shd"),
        broadband_pressure_field=broadband_pressure_field,
        frequencies=freqs,
        source=source,
        rcv_range=rcv_range,
        rcv_depth=rcv_depth,
        apply_delay=True,
        minimum_waveguide_depth=bathy.bathy_depth.min(),
    )

    pd_dict = {"time": np.round(time_vector, 3)}
    for ir, r in enumerate(rcv_range):
        for iz, z in enumerate(rcv_depth):
            key = f"r{r}m_z{z}m"
            pd_dict[key] = s_at_rcv_pos[:, iz, ir]

            plt.figure()
            plt.plot(time_vector, s_at_rcv_pos[:, iz, ir])
            plt.xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
            plt.ylabel("Pressure [Pa]", fontsize=LABEL_FONTSIZE)
            plt.xticks(fontsize=TICKS_FONTSIZE)
            plt.yticks(fontsize=TICKS_FONTSIZE)
            plt.title(
                f"Received signal (r={r}m, z={z}m)\n" + r"$\Delta_r$" + f"={dr*1e3}m",
                fontsize=TITLE_FONTSIZE,
            )

    pd.DataFrame(pd_dict).to_csv(
        f"propagated_signal_dr_{dr}.csv",
        index=False,
    )

    ds = xr.Dataset(
        {
            "s_at_rcv_pos": (["time", "rcv_depth", "rcv_range"], s_at_rcv_pos),
            "broadband_pressure_field_real": (
                ["frequency", "Ntheta", "Nsz", "z", "r"],
                np.real(broadband_pressure_field),
            ),
            "broadband_pressure_field_imag": (
                ["frequency", "Ntheta", "Nsz", "z", "r"],
                np.imag(broadband_pressure_field),
            ),
        },
        coords={
            "time": time_vector,
            "rcv_depth": rcv_depth,
            "rcv_range": rcv_range,
            "frequency": freqs,
            "Ntheta": [0],  # Dummy params to match kraken output
            "Nsz": [0],  # Dummy params to match kraken output
            "z": Pos["r"]["z"],
            "r": Pos["r"]["r"],
        },
    )

    ds.to_netcdf("broadband_pressure_field.nc")


if __name__ == "__main__":
    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband"
    os.chdir(working_dir)

    test()
