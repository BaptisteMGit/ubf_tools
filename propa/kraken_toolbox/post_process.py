import numpy as np
import time

from cst import C0, RHO_W
from utils import mult_along_axis
from propa.kraken_toolbox.utils import runkraken
from propa.kraken_toolbox.read_shd import readshd


def postprocess(shd_fpath, source, rcv_range, rcv_depth):
    """Post process Kraken run to derive time serie of the received signal through Fourier synthesis."""

    # Get frequencies
    positive_fft_freq = source.freq[source.freq > 0]
    positive_source_spectrum = source.spectrum[source.freq > 0]

    _, _, freqVec, _, _, _, field_pos, _ = readshd(filename=shd_fpath, freq=0)
    # Receiver position in the grid
    rcv_pos_idx_r = [
        np.argmin(np.abs(field_pos["r"]["r"] - rcv_r)) for rcv_r in rcv_range
    ]
    rcv_pos_idx_z = [
        np.argmin(np.abs(field_pos["r"]["z"] - rcv_z)) for rcv_z in rcv_depth
    ]
    rr, zz = np.meshgrid(rcv_pos_idx_r, rcv_pos_idx_z)

    """ Quicker implementation with """
    # nf0
    nf_arr = [
        len(positive_fft_freq[positive_fft_freq <= (freqVec[0] + freqVec[1]) / 2])
    ]

    nf_arr += [
        len(
            positive_fft_freq[
                np.logical_and(
                    positive_fft_freq > (freqVec[i_f] + freqVec[i_f + 1]) / 2,
                    positive_fft_freq <= (freqVec[i_f + 1] + freqVec[i_f + 2]) / 2,
                )
            ]
        )
        for i_f in range(0, freqVec.size - 2)
    ]

    # nfmax
    nf_arr.append(
        len(positive_fft_freq[positive_fft_freq > (freqVec[-2] + freqVec[-1]) / 2])
    )

    pf_array = np.empty(
        (positive_source_spectrum.size, len(rcv_depth), len(rcv_range)), dtype=complex
    )
    prev_n = 0
    next_n = 0
    for ifk, f_k in enumerate(freqVec):
        next_n += nf_arr[ifk]
        # Load pressure field
        _, _, _, _, _, _, _, pressure = readshd(filename=shd_fpath, freq=f_k)

        # Potential velocity field for freq f <-> spatial transfert function
        p_f = np.squeeze(pressure, axis=(0, 1))
        pf_array[prev_n:next_n, ...] = np.tile(p_f[zz, rr], (nf_arr[ifk], 1, 1))
        prev_n = next_n

    # pressure field given by field.exe is the transmission loss pressure field : p/p0(r=1) with p0(r) = exp(ik0r)/(4*pi*r))
    norm_factor = (
        RHO_W / np.pi * np.exp(1j * (2 * np.pi * positive_fft_freq / C0 - np.pi / 2))
    )
    received_signal_f = mult_along_axis(
        pf_array, positive_source_spectrum * norm_factor, axis=0
    )

    # Apply corresponding delay to the signal
    for ir, rcv_r in enumerate(rcv_range):
        delay = np.exp(-1j * 2 * np.pi * rcv_r / C0 * positive_fft_freq)
        received_signal_f[..., ir] = mult_along_axis(
            received_signal_f[..., ir], delay, axis=0
        )

    # real inverse FFT to exploit conjugate symmetry of the transfert function (see Jensen et al. 2000 p.612-613)
    received_signal_t = np.fft.irfft(received_signal_f, axis=0, n=source.ns)

    transmited_field = np.real(received_signal_t)
    time_vector = np.arange(0, transmited_field.shape[0] / source.fs, 1 / source.fs)

    return time_vector, transmited_field, field_pos


def process_broadband(fname, source, max_depth):
    """Write frequencies to env file and run KRAKEN"""

    if source.kraken_freq is None:
        kraken_freq = source.freq[source.freq > 0]
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
