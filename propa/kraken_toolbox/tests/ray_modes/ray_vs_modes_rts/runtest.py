import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from propa.kraken_toolbox.post_process import (
    postprocess_received_signal,
    postprocess_received_signal_from_broadband_pressure_field,
)
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
from propa.kraken_toolbox.plot_utils import plotmode
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.utils import default_nb_rcv_z
from localisation.verlinden.misc.AcousticComponent import AcousticSource

from cst import TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE

CREF = 1450
IMG_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\ray_vs_modes_rts\result"

RUN = False
z_src = 200
rcv_range = np.array([30000])
rcv_depth = np.array([z_src + i * 10 for i in range(-10, 10)])
delays = rcv_range / CREF
Tr = 4

if RUN:
    """Run KRAKEN with broadband src"""
    t0 = time.time()
    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\ray_vs_modes_rts\env_kraken"
    os.chdir(working_dir)

    # Source
    fpath = r"C:\Users\baptiste.menetrier\MATLAB\Projects\Localisation_Verlinden\Test_timeserie\src_pulse_rts.mat"
    src_sig = sio.loadmat(fpath)
    t = src_sig["t_sts"].squeeze()
    s = src_sig["sts"].squeeze()
    name = src_sig["PulseTitle"]
    fs = 1 / (t[1] - t[0])

    nfft = int(fs * Tr)  # Number of points for FFT
    src = AcousticSource(signal=s, time=t, name=name, waveguide_depth=1000, nfft=nfft)

    src.display_source()
    plt.savefig(os.path.join(IMG_ROOT, "src.png"))

    # Top halfspace
    top_hs = KrakenTopHalfspace()
    # SSP
    ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values

    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    # Attenuation
    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    # Field
    n_rcv_z = default_nb_rcv_z(max(src.kraken_freq), 1000, n_per_l=15)
    field = KrakenField(
        src_depth=z_src,
        phase_speed_limits=[0, 20000],
        n_rcv_z=n_rcv_z,
        rcv_z_max=1000,
    )

    bott_hs_properties = {
        "rho": 1.5,
        "c_p": 1600.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s)
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    env_filename = "calib_rts_modes"
    env = KrakenEnv(
        title="STURM test case",
        env_root=working_dir,
        env_filename=env_filename,
        freq=src.kraken_freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
    )

    env.write_env()
    flp = KrakenFlp(
        env=env,
        src_depth=z_src,
        mode_theory="adiabatic",
        rcv_r_max=30,
        rcv_z_max=1000,
        nb_modes=50,
    )
    flp.write_flp()

    pressure_field, _ = runkraken(env, flp, src.kraken_freq)

    (
        t_obs,
        s_obs,
        Pos,
    ) = postprocess_received_signal_from_broadband_pressure_field(
        shd_fpath=env.shd_fpath,
        broadband_pressure_field=pressure_field,
        frequencies=src.kraken_freq,
        source=src,
        rcv_range=rcv_range,
        rcv_depth=rcv_depth,
        apply_delay=True,
        delay=delays,
        minimum_waveguide_depth=1000,
    )
    t_kraken = time.time() - t0

    print(f"Kraken rd time: {t_kraken:.2f} s")

    ds = xr.Dataset(
        {
            "s_at_rcv_pos": (
                ["time", "rcv_depth", "rcv_range"],
                s_obs.astype(np.float32),
            ),
        },
        coords={
            "time": t_obs.astype(np.float32),
            "rcv_depth": rcv_depth,
            "rcv_range": rcv_range,
        },
    )

    ds.to_netcdf(f"modes_rts_r{rcv_range[0]}.nc")

else:
    """Load KRAKEN results"""
    fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\ray_vs_modes_rts\env_kraken"
    fname = f"modes_rts_r{rcv_range[0]}.nc"
    ds = xr.open_dataset(os.path.join(fpath, fname))

max_amplitude = np.round(ds.s_at_rcv_pos.max().values, 5)
z_offset = max_amplitude + 1e-5
for ir, r in enumerate(rcv_range):
    plt.figure(figsize=(12, 10))
    for iz, z in enumerate(rcv_depth):
        sig = ds.s_at_rcv_pos.sel(rcv_depth=z, rcv_range=r) - iz * z_offset
        if z == z_src:
            sig.plot(color="r", label="Source depth")
        else:
            sig.plot(color="k")

    depth_label = [f"{z:.0f}" for z in ds.rcv_depth.values]
    depth_pos = [-iz * z_offset for iz in range(ds.dims["rcv_depth"])]
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
    plt.legend(fontsize=LABEL_FONTSIZE, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_ROOT, f"modes_r{r}m.png"))


# Plot modes
fname = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\ray_vs_modes_rts\env_kraken\calib_rts_modes"
plotmode(fname, freq=50, modes=[1, 2, 3, 4])
plt.savefig(os.path.join(IMG_ROOT, "modes.png"))

# plt.figure()
# plt.plot(pressure_filed)
