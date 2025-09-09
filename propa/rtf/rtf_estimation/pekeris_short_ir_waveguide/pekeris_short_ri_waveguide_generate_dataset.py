# %% [markdown]
# # Objectif
#
# L'objectif de ce notebook est de définir le guide d'onde de Pekeris étudié dans ce dossier. L'objectif de ce guide d'onde est de fournir un cas test simple pour l'évaluation de différentes propriétés relatives au vecteur de RTF :
#
# * Performances des estimateurs
# * Performances de la méthodes MFP - RTF
#
# L'environnement considéré est un guide d'onde de Pekeris dont les paramètres sont les suivants :
#
# * Dans la colonne d'eau :
#     * $c_0 = 1500 m/s$
#     * $\rho_0 = 1000 kg/m^3$
# * Dans le sédiment :
#     * $c_1 = 1600 m/s$
#     * $\rho_1 = 1500 kg/m^3$
#     * $\alpha_1 = 0.2 dB/\lambda$
#
# Ce guide d'onde est identique à celui utiliser pour illustrer les propriétés de bases de la théorie des modes normaux (Cf notebook basic_modal_features.ipynb)
#

# %% [markdown]
# ## Choix des calculs à effectuer
# Lorsque que les flags sont à False on se contente de charger les données pré-existantes.

# %%
# Kraken compute opion
run_kraken = False
# run_kraken = True

# %%
import os
import sys
import numpy as np
import xarray as xr

sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd")

# Load usefull functions
from propa.kraken_toolbox.src.kraken_testcase import (
    KrakenTestCase,
    DomainProperties,
    SourceProperties,
    ReceiverProperties,
    KrakenProperties,
)
from propa.kraken_toolbox.src.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox.utils import default_nb_rcv_z
from signals.AcousticComponent import AcousticSource
from source.signal_generator import SignalGenerator
from source.ssp_profiles import SSPProfile

import propa.rtf.rtf_estimation.pekeris_short_ir_waveguide.pekeris_short_ri_waveguide_params as params

# %% [markdown]
# # Génération du jeux de données pour l'estimation des performances des méthodes de RTF

# %% [markdown]
# ## Paramètres de la source

# %%
output_fs = 4 * params.src_fs  # Output sampling frequency after propagation

# Derive N_ir <=> length of the impulse duration
N_ir = int(params.tau_th * output_fs)
print(f"Impulse response length: {N_ir} samples")

# Derive asssociated length of STFT analysis window N_stft = m * N_ir
m = 5  # Avargel and Cohen 2007 N_opti = 32 * Nh
N_stft = m * N_ir
# Get closer power of 2
N_stft = 2 ** int(np.log2(N_stft) + 1)
print(f"STFT analysis window length: {N_stft} samples")

# Define the target STFT overlap
alpha_ov = 0.75  # STFT overlap factor
R_stft = int((1 - alpha_ov) * N_stft)
print(f"STFT Block shift: {R_stft} samples")

# Set the number of expected snapshots L
L_stft = 20  # Number of STFT snapshots to estimate the RTF
print(f"Number of STFT snapshots: {L_stft}")

# Derive the signal duration to get L STFT snapshots
signal_duration = (L_stft - 1) * R_stft / output_fs + N_stft / output_fs
print(f"Signal duration: {signal_duration} s")


# %%
src_depth = params.waveguide_depth - 1  # Reciprocity -> receiver depth

# Create dummy signal to make it easy to run kraken simulation
fc = 25
sg = SignalGenerator()
s, t = sg.pulse(T=signal_duration, fc=fc, fs=params.src_fs, t0=0)
s = sg.normalize_sig(s, normalize="max")

src = AcousticSource(
    signal=s,
    time=t,
    name="Pulse",
    waveguide_depth=params.waveguide_depth,
    window=None,
    nfft=2 ** int(np.log2(s.size) + 1),
)

# %% [markdown]
# ## Paramètres du domaine de calcul

# %%
max_range_km = 45
min_range_km = 15

# %% [markdown]
# ## Définition du cas test avant calcul Kraken

# %%
name = "perekis_short_ir_waveguide_perf"
title = (
    "Pekeris waveguide with short impulse response - for RTF methods performance study"
)

# Common properties
zmin = 0
zmax = params.waveguide_depth
rmax = max_range_km * 1e3
rmin = min_range_km * 1e3

min_phase_speed = 1000
max_phase_speed = 20000

bott_props = {
    "rho": params.rho_sediment * 1e-3,  # Density (g/cm^3)
    "c_p": params.c_sediment,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s)
    "a_p": params.alpha_sediment,  # Compression wave attenuation (dB/wavelength)
    "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
}

# Set domain properties
domain_properties = DomainProperties(
    zmin=zmin, zmax=zmax, rmin=rmin, rmax=rmax, unit="m"
)

# Set source properties
src_properties = SourceProperties(
    src_type="point_source", src_depth=src_depth, freq=src.kraken_freq
)
# Set receiver properties : needs to cover the whole water domain
rcv_z_min = zmin
rcv_z_max = params.waveguide_depth

# Number of receiver depths / ranges (flp file) : sufficient resolution for later use (can be easily downsampled afterwards)
dr = 50
dz = 5
nr_flp = int((rmax - rmin) / dr) + 1
nz_flp = int(rcv_z_max / dz) + 1

rcv_properties = ReceiverProperties(
    zmin=rcv_z_min, zmax=rcv_z_max, rmin=rmin, rmax=rmax, unit="m"
)

# Set kraken properties
nmedia = 2

top_hs = KrakenTopHalfspace(
    boundary_condition="vacuum",
    halfspace_properties=None,
    twersky_scatter_properties=None,
)

bott_hs = KrakenBottomHalfspace(
    boundary_condition="acousto_elastic",
    sigma=0.0,
    halfspace_properties=bott_props,
    fmin=src.kraken_freq.min(),
    alpha_wavelength=10,
)
# Set attenuation properties
att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
# Set SSP profile
z = [0, zmax]
c = [params.c_water, params.c_water]  # Constant celerity profile
ssp = SSPProfile(z=z, c=c)

# Create the medium = water column layer
medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=ssp.z,
    c_p=ssp.c,
    c_s=0.0,
    rho=1.0,
    a_p=0.0,
    a_s=0.0,
    nmesh=0,
    sigma=0.0,
)

bott_hs.derive_sedim_layer_max_depth(domain_properties.zmax_m)
max_rcv_depth = bott_hs.sedim_layer_max_depth
n_rcv_z = default_nb_rcv_z(
    fmax=src.kraken_freq.max(), max_depth=max_rcv_depth, n_per_l=10
)
field = KrakenField(
    phase_speed_limits=[min_phase_speed, max_phase_speed],
    src_depth=[src_properties.depth],
    n_rcv_z=n_rcv_z,
    rcv_z_min=0,
    rcv_z_max=max_rcv_depth,
    rcv_r_max=0.0,
)

kraken_properties = KrakenProperties(
    mode_coupling="coupled",
    mode_addition="coherent",
    n_mode=100,
    nr=nr_flp,
    nz=nz_flp,
    nmedia=nmedia,
    top_hs=top_hs,
    bott_hs=bott_hs,
    att=att,
    medium=medium,
    field=field,
)


k_tc = KrakenTestCase(
    name=name,
    title=title,
    root_dir=params.tc_root_dir,
    domain_properties=domain_properties,
    src_properties=src_properties,
    rcv_properties=rcv_properties,
    kraken_properties=kraken_properties,
)

# %%
if run_kraken:
    km = KrakenManager()

    # Because of the large number of frequencies we need to split the calculation into frequency chunks -> field does not work with broadband simulation using
    # nf > 1000 frequencies
    all_freqs = k_tc.src.freq
    chunk_size = 950  # Number of frequencies per chunk
    n_chunks = int(np.ceil(len(all_freqs) / chunk_size))
    pressure_field = []
    for ichunk in range(n_chunks):
        # Get the frequencies for the current chunk
        start_freq = ichunk * chunk_size
        end_freq = min((ichunk + 1) * chunk_size, len(all_freqs))
        freq_chunk = all_freqs[start_freq:end_freq]

        # Run Kraken for the current chunk
        print(
            f"Running Kraken for frequencies {start_freq} to {end_freq - 1} (chunk {ichunk + 1}/{n_chunks})"
        )

        # update env freq
        k_tc.env.freq = freq_chunk
        # Run Kraken and get the pressure field
        p, field_pos = km.runkraken(env=k_tc.env, flp=k_tc.flp, frequencies=freq_chunk)

        # Append the pressure field for the current chunk
        pressure_field.append(p)

    # Concatenate the pressure fields from all chunks
    pressure_field = np.concatenate(pressure_field, axis=0)

    # pressure_field, field_pos = km.runkraken(
    #     env=k_tc.env, flp=k_tc.flp, frequencies=k_tc.src.freq
    # )

    # Store pressure field as netcdf using xarray
    pressure_field = np.squeeze(pressure_field)  # Remove singleton dimensions if any
    ds_tf = xr.Dataset(
        data_vars=dict(
            tf_real=(["f", "z", "r"], np.real(pressure_field)),
            tf_imag=(["f", "z", "r"], np.imag(pressure_field)),
        ),
        coords=dict(
            f=k_tc.src.freq,
            z=field_pos["r"]["z"],
            r=field_pos["r"]["r"],
        ),
        attrs=dict(
            title="Transfer functions for Pekeris waveguide",
            description="Transfer functions computed using Kraken for a Pekeris waveguide with short impulse response.",
            # date_created=np.datetime64("now"),
            note="Dataset for rtf performance analysis purpose: lower spatial resolution, reduced range coverage and long signal duration.",
            type="perf",
            fs=params.src_fs,
            signal_duration=signal_duration,
            waveguide_depth=params.waveguide_depth,
            src_depth=src_depth,
        ),
    )
    # Save to netcdf
    ds_tf.to_netcdf(params.tf_perf_fpath)
else:
    # Load transfer functions from netcdf
    ds_tf = xr.open_dataset(params.tf_perf_fpath)
