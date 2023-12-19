import numpy as np
import time

from cst import C0, RHO_W
from misc import mult_along_axis
from propa.kraken_toolbox.utils import runkraken, waveguide_cutoff_freq
from propa.kraken_toolbox.read_shd import readshd
from scipy.interpolate import interp1d


def postprocess_ir(shd_fpath, source, rcv_range, rcv_depth):
    """Post process Kraken run to derive ocean waveguide impulse response."""

    # Dummy read to get frequencies used by kraken and field grid information
    _, _, freqVec, _, _, _, field_pos, _ = readshd(filename=shd_fpath, freq=0)

    # Load pressure field for all frequencies
    _, _, _, _, read_freq, _, _, pressure = readshd(filename=shd_fpath, freq=freqVec)
    pressure_field = np.squeeze(pressure, axis=(1, 2))  # 3D array (nfreq, nz, nr)

    # No need to process the entire grid :  extract pressure field at desired positions
    rcv_pos_idx_r = [
        np.argmin(np.abs(field_pos["r"]["r"] - rcv_r)) for rcv_r in rcv_range
    ]
    rcv_pos_idx_z = [
        np.argmin(np.abs(field_pos["r"]["z"] - rcv_z)) for rcv_z in rcv_depth
    ]
    rr, zz = np.meshgrid(rcv_pos_idx_r, rcv_pos_idx_z)
    pressure_field = pressure_field[:, zz, rr]

    # Get rid of frequencies below the cutoff frequency
    fc = waveguide_cutoff_freq(max_depth=field_pos["r"]["z"].max())
    propagating_freq = source.positive_freq[source.positive_freq > fc]

    if (
        read_freq.size < propagating_freq.size
    ):  # Sparse frequency vector used for kraken
        pressure_field = interp_frequency_domain_ir(
            read_freq, pressure_field, propagating_freq
        )

    elif read_freq.size == propagating_freq.size:
        freqdiff = read_freq - propagating_freq
        if freqdiff.max() > 0.5:  # Different frequency sampling
            pressure_field = interp_frequency_domain_ir(
                read_freq, pressure_field, propagating_freq
            )

    return propagating_freq, pressure_field, field_pos


def postprocess_received_signal(
    shd_fpath, source, rcv_range, rcv_depth, apply_delay=True, delay=None
):
    # Derive broadband impulse response of the ocean waveguide
    propagating_freq, pressure_field, field_pos = postprocess_ir(
        shd_fpath, source, rcv_range, rcv_depth
    )

    # Extract propagating spectrum from entire spectrum
    fc = waveguide_cutoff_freq(max_depth=field_pos["r"]["z"].max())
    propagating_spectrum = source.positive_spectrum[source.positive_freq > fc]

    # TODO : check which factor to apply
    # norm_factor = (
    #     RHO_W / np.pi * np.exp(1j * (2 * np.pi * positive_fft_freq / C0 - np.pi / 2))
    # )
    norm_factor = 1

    # Received signal spectrum resulting from the convolution of the source signal and the impulse response
    transmited_field_f = mult_along_axis(
        pressure_field, propagating_spectrum * norm_factor, axis=0
    )

    # Apply corresponding delay to the signal
    if apply_delay:
        for ir, rcv_r in enumerate(rcv_range):  # TODO: remove loop for efficiency
            if delay is None:
                tau = rcv_r / C0
            else:
                tau = delay[ir]

            delay_f = np.exp(1j * 2 * np.pi * tau * propagating_freq)

            transmited_field_f[..., ir] = mult_along_axis(
                transmited_field_f[..., ir], delay_f, axis=0
            )

    # Fourier synthesis of the received signal -> time domain
    nfft_inv = (
        4 * source.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    received_signal_t = np.fft.irfft(transmited_field_f, axis=0, n=nfft_inv)
    transmited_field_t = np.real(received_signal_t)

    T_tot = 1 / source.df
    dt = T_tot / received_signal_t.shape[0]
    time_vector = np.arange(0, T_tot, dt)

    return (
        time_vector,
        transmited_field_t,
        field_pos,
    )


def interp_frequency_domain_ir(pf_p, f_p, f):
    unwrapped_phase = np.unwrap(np.angle(pf_p), axis=0)
    magnitude = np.abs(pf_p)

    interpolated_pf = np.empty_like(pf_p)

    # Loop over physical dimensions (not efficient but np interp can't deal with multi dimensionnal arrays)
    for iz in range(pf_p.shape[1]):
        for ir in range(pf_p.shape[2]):
            # Interp phase
            interpolated_unwrapped_phase = np.interp(f, f_p, unwrapped_phase[:, iz, ir])
            interpolated_wraped_phases = (interpolated_unwrapped_phase + np.pi) % (
                2 * np.pi
            ) - np.pi

            # Interp magnitude
            interpolated_magnitude = np.interp(f, f_p, magnitude[:, iz, ir])

            # Reconstruct field at pos iz, ir
            interpolated_pf[:, iz, ir] = interpolated_magnitude * np.exp(
                1j * interpolated_wraped_phases
            )

    return interpolated_pf


def process_broadband(fname, source, max_depth):
    """Write frequencies to env file and run KRAKEN"""

    if source.kraken_freq is None:
        kraken_freq = source.positive_freq
    else:
        kraken_freq = source.kraken_freq

    # Get rid of frequencies below the cutoff frequency
    fc = C0 / (4 * max_depth)
    idx_freq = kraken_freq > fc
    kraken_freq = kraken_freq[idx_freq]
    # source_spectrum = source_spectrum[idx_freq]

    # Write env file with frequencies of interest
    with open(fname + ".env", "r") as f:
        lines = f.readlines()
        lines[-2] = f"{int(len(kraken_freq))}              ! Nfreq\n"
        kraken_freqvec = " ".join([str(f) for f in kraken_freq])
        lines[-1] = f" {kraken_freqvec} /      ! freqVec( 1 : Nfreq )"

    with open(fname + ".env", "w") as f:
        f.writelines(lines)

    # Run kraken for the frequencies of interest
    runkraken(fname)


def fourier_synthesis_kraken(fname, source, max_depth):
    # OUTDATED functin to be removed
    """Derive time serie of the received signal through Fourier synthesis.

    Note that according to Jensen et al. (2000), the sampling frequency must be greater than 8 * fmax for visual inspection of the propagated pulse.
    """

    # TODO: add args to define the freqeuncies to use
    # It can be very usefull to reduce the number of frequencies to compute the Fourier synthesis
    # For instance using a non unigorm frequency vector can help catching the entire signal complexity while significantly reducing the
    # computing effort --> example : signal with dirac like harmonics

    # freq_ = source.energetic_freq[source.energetic_freq > 0]  # Option to reduce computing effort (need to update time vector accordingly)

    positive_fft_freq = source.freq[source.freq > 0]
    if source.kraken_freq is None:
        kraken_freq = source.freq[source.freq > 0]
    else:
        kraken_freq = source.kraken_freq
        # source_spectrum = np.interp(freq_, source.freq, source.spectrum)

    positive_source_spectrum = source.spectrum[source.freq > 0]

    # Get rid of frequencies below the cutoff frequency
    fc = C0 / (4 * max_depth)
    idx_freq = kraken_freq > fc
    kraken_freq = kraken_freq[idx_freq]
    # source_spectrum = source_spectrum[idx_freq]

    # Write env file with frequencies of interest
    # TODO: update section below with appropriate write_env function ?
    with open(fname + ".env", "r") as f:
        lines = f.readlines()
        lines[-2] = f"{int(len(kraken_freq))}              ! Nfreq\n"
        kraken_freqvec = " ".join([str(f) for f in kraken_freq])
        lines[-1] = f" {kraken_freqvec} /      ! freqVec( 1 : Nfreq )"

    with open(fname + ".env", "w") as f:
        f.writelines(lines)

    # Run kraken for the frequencies of interest
    runkraken(fname)

    env_filename = fname + ".shd"

    rcv_signal_f = []
    for ifreq, freq in enumerate(positive_fft_freq):  # Iterate over all fft frequencies
        # Load pressure field for closest freq
        _, _, _, _, fread, _, field_pos, pressure = readshd(
            filename=env_filename, freq=freq
        )

        # Normalisation factor to get potential velocity
        norm_factor = 4 * RHO_W * np.exp(1j * (2 * np.pi * freq / C0 - np.pi / 2))

        # Potential velocity field for freq f <-> spatial transfert function
        p_f = np.squeeze(pressure, axis=(0, 1)) * norm_factor

        # Received signal in the frequency domain
        s_f = positive_source_spectrum[ifreq] * p_f
        # s_f = source.spectrum[ifreq] * p_f

        # Set approximated time of arrival
        # c_group_max = 1500
        # if ifreq == 0:
        #     d_direct_path = np.sqrt(rcv_range**2 + (rcv_depth - Pos["s"]["z"]) ** 2)
        #     tau = d_direct_path / c_group_max
        # s_f *= np.exp(1j * 2 * np.pi * tau * freq)  # Apply delay to the signal

        rcv_signal_f.append(s_f)

    received_signal_f = np.array(rcv_signal_f)

    # real inverse FFT to exploit conjugate symmetry of the transfert function (see Jensen et al. 2000 p.612-613)
    received_signal_t = np.fft.irfft(received_signal_f, axis=0)

    transmited_field = np.real(received_signal_t)
    time_vector = np.arange(0, transmited_field.shape[0] / source.fs, 1 / source.fs)

    return time_vector, transmited_field, field_pos


if __name__ == "__main__":
    pass

    # plt.figure()
    # plt.vlines(freqVec, ymin=-0.01, ymax=0.01, color="k", linestyle="--")
    # plt.plot(positive_fft_freq, np.real(transmited_field_f[:, 0, 0]), label="real")
    # plt.plot(positive_fft_freq, np.imag(transmited_field_f[:, 0, 0]), label="imag")
    # plt.xlabel("Frequency (Hz)")
    # plt.show()

    # # from scipy import interpolate
    # f_interp = np.arange(positive_fft_freq.min(), positive_fft_freq.max(), 0.1)
    # interp_phases = np.interp(
    #     f_interp,
    #     positive_fft_freq,
    #     np.unwrap(np.angle(transmited_field_f[:, 0, 0])),
    # )

    # # phase_interpolator = interpolate.interp1d(
    # #     positive_fft_freq,
    # #     np.unwrap(np.angle(transmited_field_f[:, 0, 0])),
    # #     kind="cubic",
    # # )
    # # interp_phases = phase_interpolator(f_interp)
    # wraped_phases = (interp_phases + np.pi) % (2 * np.pi) - np.pi

    # plt.figure()
    # plt.plot(
    #     positive_fft_freq,
    #     np.unwrap(np.angle(transmited_field_f[:, 0, 0])),
    #     label="original",
    # )
    # plt.plot(f_interp, interp_phases, label="interp")
    # plt.ylabel("Phase (°)")
    # plt.xlabel("Frequency (Hz)")
    # plt.legend()

    # plt.figure()
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Phase (rads)")
    # plt.plot(
    #     positive_fft_freq,
    #     np.angle(transmited_field_f[:, 0, 0]),
    #     label="original",
    #     marker="+",
    # )
    # plt.plot(f_interp, wraped_phases, label="interp", marker="+")
    # plt.show()
