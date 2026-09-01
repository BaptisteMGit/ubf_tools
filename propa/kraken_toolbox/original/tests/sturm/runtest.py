import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
from propa.kraken_toolbox.run_kraken import (
    runkraken_broadband_range_dependent,
    runkraken,
)
from propa.kraken_toolbox.plot_utils import plotshd_from_pressure_field
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal_from_broadband_pressure_field,
)

from propa.kraken_toolbox.tests.sturm.testcase import testcase

ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\sturm"
IMG_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\sturm\img"
SOURCE_DEPTH = 25  # m
MAX_DEPTH = 200  # m
MIN_DEPTH = 100  # m
MAX_RANGE = 10  # km
FMAX = 40  # Hz

RCV_DEPTH = np.linspace(10, 200, 20)  # m
RCV_RANGE = [20000]  # m
CREF = 1450
DELAYS = np.array(RCV_RANGE) / CREF

# Bathymetry
RANGE_PERIODICITY = 6  # km
FR = 1 / RANGE_PERIODICITY


def run_test():
    env, flp, source = testcase()
    # runkraken(env.filename)
    broadband_pressure_field = runkraken_broadband_range_dependent(
        range_dependent_env=env,
        flp=flp,
        frequencies=source.kraken_freq,
    )

    (
        time_vector,
        s_at_rcv_pos,
        Pos,
    ) = postprocess_received_signal_from_broadband_pressure_field(
        shd_fpath=os.path.join(ROOT, env.filename + ".shd"),
        broadband_pressure_field=broadband_pressure_field,
        frequencies=source.kraken_freq,
        source=source,
        rcv_range=RCV_RANGE,
        rcv_depth=RCV_DEPTH,
        apply_delay=False,
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

    try:
        ds.to_netcdf(f"sturm_testcase.nc")
    except:
        ds = ds.drop_vars(
            ["broadband_pressure_field_real", "broadband_pressure_field_imag"]
        )
        ds.to_netcdf(f"sturm_testcase.nc")

    ifreq = np.argmin(np.abs(source.kraken_freq - 25))
    freq_to_plot = source.kraken_freq[ifreq]
    pressure_field_to_plot = broadband_pressure_field[ifreq, ...]
    plotshd_from_pressure_field(
        env.filename + ".shd",
        pressure_field=pressure_field_to_plot,
        title=f"{env.simulation_title} - f = {freq_to_plot:.2f} Hz",
        freq=freq_to_plot,
        bathy=env.bathy,
    )
    plt.savefig(f"sturm_testcase_f{freq_to_plot:.2f}Hz.png")

    for ir, r in enumerate(RCV_RANGE):
        plt.figure(figsize=(12, 10))
        for iz, z in enumerate(RCV_DEPTH):
            sig = ds.s_at_rcv_pos.sel(rcv_depth=z, rcv_range=r) - z * 5 * 1e-6
            sig.plot(color="k")

        depth_label = [f"{z:.0f}" for z in ds.rcv_depth.values]
        depth_pos = [-z * 5 * 1e-6 for z in ds.rcv_depth.values]
        ax = plt.gca()
        ax.set_yticks(depth_pos[::2])
        ax.set_yticklabels(depth_label[::2])
        plt.xlabel(f"Time [s] - r/{CREF}", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.title(
            f"Received signal (r={r}m)\n",
            fontsize=TITLE_FONTSIZE,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_ROOT, f"sturm_testcase_r{r}m.png"))


def re_analyse():
    ds = xr.open_dataset(f"sturm_testcase.nc")

    # Stack series
    for ir, r in enumerate(RCV_RANGE):
        plt.figure(figsize=(12, 10))
        for iz, z in enumerate(RCV_DEPTH):
            sig = (
                ds.s_at_rcv_pos.sel(rcv_depth=z, rcv_range=r).sel(time=slice(0, 2.5))
                - z * 5 * 1e-6
            )
            sig.plot(color="k")

        depth_label = [f"{z:.0f}" for z in ds.rcv_depth.values]
        depth_pos = [-z * 5 * 1e-6 for z in ds.rcv_depth.values]
        ax = plt.gca()
        ax.set_yticks(depth_pos[::2])
        ax.set_yticklabels(depth_label[::2])
        plt.xlabel(f"Time [s] - r/{CREF}", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.title(
            f"Received signal (r={r}m)\n",
            fontsize=TITLE_FONTSIZE,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_ROOT, f"sturm_testcase_r{r}m.png"))


if __name__ == "__main__":
    os.chdir(ROOT)
    run_test()
    # re_analyse()
