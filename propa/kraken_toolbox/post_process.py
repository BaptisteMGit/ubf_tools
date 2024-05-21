import numpy as np
import time

from cst import C0, RHO_W
from misc import mult_along_axis
from propa.kraken_toolbox.utils import waveguide_cutoff_freq, get_rcv_pos_idx
from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.read_shd import readshd
from scipy.interpolate import interp1d

""" Post processor for KRAKEN runs using shd files. """


def postprocess_ir(shd_fpath, source, rcv_range, rcv_depth):
    """Read kraken results to get the transfert function."""

    # Dummy read to get frequencies used by kraken and field grid information
    _, _, freqVec, _, _, _, field_pos, _ = readshd(filename=shd_fpath, freq=0)

    # Load pressure field for all frequencies
    _, _, _, _, read_freq, _, _, pressure = readshd(filename=shd_fpath, freq=freqVec)
    pressure_field = np.squeeze(pressure, axis=(1, 2))  # 3D array (nfreq, nz, nr)

    # No need to process the entire grid :  extract pressure field at desired positions
    rr, zz, field_pos = get_rcv_pos_idx(shd_fpath, rcv_depth, rcv_range)
    pressure_field = pressure_field[:, zz, rr]

    # Get rid of frequencies below the cutoff frequency
    fc = waveguide_cutoff_freq(waveguide_depth=field_pos["r"]["z"].max())
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
    fc = waveguide_cutoff_freq(waveguide_depth=field_pos["r"]["z"].max())
    propagating_spectrum = source.positive_spectrum[source.positive_freq > fc]

    # TODO : check which factor to apply
    norm_factor = (
        RHO_W / np.pi * np.exp(1j * (2 * np.pi * propagating_freq / C0 - np.pi / 2))
    )

    # Received signal spectrum resulting from the convolution of the source signal and the impulse response
    transmited_field_f = mult_along_axis(
        pressure_field, propagating_spectrum * norm_factor, axis=0
    )

    nfft_inv = (
        4 * source.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / source.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt)

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
    received_signal_t = np.fft.irfft(transmited_field_f, axis=0, n=nfft_inv)
    transmited_field_t = np.real(received_signal_t)

    return (
        time_vector,
        transmited_field_t,
        field_pos,
    )


""" Post processor for KRAKEN runs pre-loaded broadband pressure field array. Needed for broadband range dependent simulations."""


def postprocess_ir_from_broadband_pressure_field(
    broadband_pressure_field,
    frequencies,
    source,
    kraken_range=None,
    kraken_depth=None,
    shd_fpath=None,
    rcv_range=None,
    rcv_depth=None,
    minimum_waveguide_depth=1000,
    squeeze=True,
):
    """Post process Kraken run to derive ocean waveguide impulse response."""

    if squeeze:
        pressure_field = np.squeeze(
            broadband_pressure_field, axis=(1, 2)
        )  # 3D array (nfreq, nz, nr)
    else:
        pressure_field = broadband_pressure_field

    # No need to process the entire grid :  extract pressure field at desired positions
    rr, zz, field_pos = get_rcv_pos_idx(
        kraken_range=kraken_range,
        kraken_depth=kraken_depth,
        rcv_depth=rcv_depth,
        rcv_range=rcv_range,
        shd_fpath=shd_fpath,
    )
    pressure_field = pressure_field[:, zz, rr]

    # Get rid of frequencies below the cutoff frequency
    fc = waveguide_cutoff_freq(waveguide_depth=minimum_waveguide_depth)
    propagating_freq = source.positive_freq[source.positive_freq > fc]

    # # TODO interpolation to match source frequencies not implemented yet
    # if (
    #     frequencies.size < propagating_freq.size
    # ):  # Sparse frequency vector used for kraken
    #     pressure_field = interp_frequency_domain_ir(
    #         frequencies, pressure_field, propagating_freq
    #     )

    # elif frequencies.size == propagating_freq.size:
    #     freqdiff = frequencies - propagating_freq
    #     if freqdiff.max() > 0.5:  # Different frequency sampling
    #         pressure_field = interp_frequency_domain_ir(
    #             frequencies, pressure_field, propagating_freq
    #         )

    return propagating_freq, pressure_field, field_pos


def postprocess_received_signal_from_broadband_pressure_field(
    broadband_pressure_field,
    frequencies,
    source,
    rcv_range,
    rcv_depth,
    minimum_waveguide_depth=1000,
    kraken_range=None,
    kraken_depth=None,
    shd_fpath=None,
    apply_delay=True,
    delay=None,
    squeeze=True,
):
    # Derive broadband impulse response of the ocean waveguide
    (
        propagating_freq,
        pressure_field,
        field_pos,
    ) = postprocess_ir_from_broadband_pressure_field(
        broadband_pressure_field=broadband_pressure_field,
        frequencies=frequencies,
        source=source,
        shd_fpath=shd_fpath,
        kraken_range=kraken_range,
        kraken_depth=kraken_depth,
        rcv_range=rcv_range,
        rcv_depth=rcv_depth,
        minimum_waveguide_depth=minimum_waveguide_depth,
        squeeze=squeeze,
    )

    # Extract propagating spectrum from entire spectrum
    fc = waveguide_cutoff_freq(waveguide_depth=minimum_waveguide_depth)
    propagating_spectrum = source.positive_spectrum[source.positive_freq > fc]

    k0 = 2 * np.pi * propagating_freq / C0
    norm_factor = np.exp(1j * k0) / (4 * np.pi)

    # Received signal spectrum resulting from the convolution of the source signal and the impulse response
    transmited_field_f = mult_along_axis(
        pressure_field, propagating_spectrum * norm_factor, axis=0
    )

    nfft_inv = (
        4 * source.nfft
    )  # according to Jensen et al. (2000) p.616 : dt < 1 / (8 * fmax) for visual inspection of the propagated pulse
    T_tot = 1 / source.df
    dt = T_tot / nfft_inv
    time_vector = np.arange(0, T_tot, dt)

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
    received_signal_t = np.fft.irfft(transmited_field_f, axis=0, n=nfft_inv)
    transmited_field_t = np.real(received_signal_t)

    return (
        time_vector,
        transmited_field_t,
        field_pos,
    )


def interp_frequency_domain_ir(pf_p, f_p, f):
    # TODO : debug this function
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


if __name__ == "__main__":
    pass
