#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   data_builder.py
@Time    :   2025/04/07 11:48:20
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to build simulation data for Zhang et al. test case
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr

import source.global_constants as g
import propa.rtf.rtf_localisation.zhang_et_al_testcase.src.params as p
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from misc import cast_matrix_to_target_shape, mult_along_axis
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    library_src_spectrum,
    event_src_spectrum,
    params,
)


class DataBuilder:
    """
    Class to build simulation data for Zhang et al. test case
    """

    def __init__(
        self,
        root_tmp: str = p.root_tmp,
        root_data: str = p.root_data,
        fmin: float = p.fmin,
        fmax: float = p.fmax,
        fs: float = p.fs,
        signal_duration: float = p.duration,
        antenna_type: str = "zhang",
        library_stype: str = "lfm",
        event_stype: str = "wn",
        debug: bool = False,
        verbose: bool = False,
    ):
        """
        Constructor
        """
        self.root_tmp = root_tmp
        self.root_data = root_data
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.signal_duration = signal_duration

        self.antenna_type = antenna_type
        self.library_stype = library_stype
        self.event_stype = event_stype

        # Usefull flags
        self.debug = debug
        self.verbose = verbose

    def build_tf_dataset(self):
        """Step 1 : use Kraken propagation model to derive the broadband transfert function of the testcase waveguide"""

        km = KrakenManager()

        _, _, freq, idx_in_band = library_src_spectrum(
            stype=self.library_stype,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            T=self.signal_duration,
            plot=False,
        )

        f = freq[idx_in_band]

        # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
        n_subband = 900
        i_subband = 1
        f0 = f[0]
        f1 = f[n_subband]

        fname = "testcase_zhang2023"
        # working_dir = os.path.join(ROOT, "tmp")
        working_dir = self.root_tmp
        env_file = os.path.join(working_dir, f"{fname}.env")

        # Read env file
        with open(env_file, "r") as file:
            lines = file.readlines()

        first_iter = True
        while f0 < f[-1]:

            # Frequency subband
            f_kraken = f[(f < f1) & (f >= f0)]
            # print(i_subband, f0, f1, len(f_kraken))
            pad_before = np.sum(f < f0)
            pad_after = np.sum(f >= f1)

            # Modify number of frequencies
            nb_freq = f"{len(f_kraken)}                                                     ! Number of frequencies\n"
            lines[-2] = nb_freq
            # Replace frequencies in the env file
            new_freq_line = " ".join([f"{fi:.2f}" for fi in f_kraken])
            new_freq_line += "    ! Frequencies (Hz)"
            lines[-1] = new_freq_line

            # Write new env file
            with open(env_file, "w") as file:
                file.writelines(lines)

            # Run kraken and field
            os.chdir(working_dir)
            km.run_kraken_exec(fname)
            km.run_field_exec(fname)

            # Read shd from previously run kraken
            shdfile = f"{fname}.shd"

            _, _, _, _, read_freq, _, field_pos, pressure_field = km.readshd(
                filename=shdfile, freq=f_kraken
            )
            tf_subband = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)

            if first_iter:
                nf = len(f)
                nr = tf_subband.shape[1]
                h_grid = np.zeros((nf, nr), dtype=complex)
                first_iter = False
            h_grid += np.pad(tf_subband, ((pad_before, pad_after), (0, 0)))

            # Update frequency subband
            i_subband += 1
            f0 = f1
            f1 = f[min(n_subband * i_subband, len(f) - 1)]

        # return f, r, z, h_grid

        # Pad h_grid with 0 for frequencies outside the 50 - 550 Hz band
        pad_before = np.sum(freq < 50)
        pad_after = np.sum(freq > 550)
        h_grid = np.pad(h_grid, ((pad_before, pad_after), (0, 0)))

        # Build xarray dataset
        library_zhang = xr.Dataset(
            data_vars=dict(
                tf_real=(
                    ["f", "r"],
                    np.real(h_grid),
                ),
                tf_imag=(["f", "r"], np.imag(h_grid)),
            ),
            coords={
                "f": freq,
                "r": field_pos["r"]["r"],
            },
        )

        # Save as netcdf
        fpath = os.path.join(self.root_data, "tf_zhang_dataset.nc")
        library_zhang.to_netcdf(fpath)
        library_zhang.close()

    def grid_dataset(self):
        """Step 2 : Associate each grid pixel to the corresponding broadband transfert function caracterized by the range to the receiver.

        -   Kraken tf : H(f, r)
        -   Grid : r(x, y)
        -   Gridded tf : H(f, x, y) = H(f, r(x, y))
        """

        # Load dataset
        fpath = os.path.join(self.root_data, "tf_zhang_dataset.nc")
        ds = xr.open_dataset(fpath)

        # Load param
        _, receivers, _, grid, _, _ = params(
            debug=self.debug, antenna_type=self.antenna_type
        )

        # Create new dataset
        ds_grid = xr.Dataset(
            coords=dict(
                f=ds.f.values,
                y=grid["y"][:, 0],
                x=grid["x"][0, :],
                idx_rcv=range(len(receivers["x"])),
            ),
            attrs=dict(
                df=ds.f.diff("f").values[0],
                dx=grid["dx"],
                dy=grid["dy"],
                testcase="zhang_et_al_2023",
            ),
        )

        # Grid tf to the desired resolution
        # Preprocess tf to decrease the number of point for further interpolation
        r_grid_all_rcv = np.array(
            [grid["r"][i_rcv].flatten() for i_rcv in range(len(receivers["x"]))]
        )
        r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

        tf_vect = ds.tf_real.sel(
            r=r_grid_all_rcv_unique, method="nearest"
        ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")

        gridded_tf = []
        grid_shape = (ds_grid.sizes["f"],) + grid["r"].shape[
            1:
        ]  # Try to fix search grid issues 11/02/202    (nf, ny, nx)
        for i_rcv in range(len(receivers["x"])):
            r_grid = grid["r"][i_rcv].flatten()
            tf_ircv = tf_vect.sel(r=r_grid, method="nearest")

            tf_grid = tf_ircv.values.reshape(grid_shape)  # (nf, ny, nx)
            gridded_tf.append(tf_grid)

        gridded_tf = np.array(gridded_tf)  # (nr, nf, ny, nx)
        # Add to dataset
        grid_coords = ["idx_rcv", "f", "y", "x"]  # Fix 11/02/2025
        ds_grid["tf_real"] = (grid_coords, np.real(gridded_tf))
        ds_grid["tf_imag"] = (grid_coords, np.imag(gridded_tf))

        # Save dataset
        if self.debug:
            fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m_debug.nc"
        else:
            fname = f"tf_zhang_grid_dx{ds_grid.dx}m_dy{ds_grid.dy}m.nc"

        fpath = os.path.join(self.root_data, fname)
        ds_grid.to_netcdf(fpath)
        ds_grid.close()

    def build_signal(self):
        """Step 3 : derive signal received from each grid pixel using library source spectrum and gridded transfert functions.

        -   Gridded tf : H(x, y, f)
        -   Source spectrum : S(f)
        -   Gridded spectrum : Y(x, y, f) = S(f) H(x, y, f)
        -   Gridded signal : y(x, y, t) = FFT_inv(Y(x, y, f))

        """

        # Load params
        depth, receivers, source, grid, frequency, _ = params(
            debug=self.debug, antenna_type=self.antenna_type
        )

        # Load gridded dataset
        if self.debug:
            fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m_debug.nc"
        else:
            fname = f"tf_zhang_grid_dx{grid['dx']}m_dy{grid['dy']}m.nc"

        fpath = os.path.join(self.root_data, fname)
        ds_gridded_tf = xr.open_dataset(fpath)  # (nf, ny, nx, nrcv)

        # Limit max frequency to speed up
        # fs_target = 1200
        # fmax = fs_target / 2
        ds_gridded_tf = ds_gridded_tf.sel(f=slice(0, self.fs / 2))

        # Load library spectrum
        # f = ds_gridded_tf.f.values
        library_props, S_f_library, _, _ = library_src_spectrum(
            stype=self.library_stype,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            T=self.signal_duration,
            plot=True,
        )

        # Load event spectrum
        _, S_f_event, _ = event_src_spectrum(
            stype=self.event_stype,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            T=self.signal_duration,
            plot=True,
        )

        # Derive delay for each receiver
        cmin = 1488  # Min speed in SwellEx96 mean profile
        delay_rcv = grid["r"] / cmin  # (nrcv, ny, nx)

        # Same delay is applied to each receiver : the receiver with the minimum delay is taken as the time reference
        # (we are only interested in relative time difference)
        # tau = ds_gridded_tf.delay_rcv.min(dim="idx_rcv")
        f = ds_gridded_tf.f
        tau = np.min(delay_rcv, axis=0)  # (ny, nx)
        da_tau = xr.DataArray(
            data=tau, coords=dict(y=ds_gridded_tf.y, x=ds_gridded_tf.x)
        )
        tau_event = da_tau.sel(y=source["y"], x=source["x"], method="nearest")
        delay_event = np.exp(1j * 2 * np.pi * tau_event * f)

        # Cast tau gridded tf shape
        tau_lib = cast_matrix_to_target_shape(
            tau, ds_gridded_tf.tf_real.shape[1:]
        )  # (nf, ny, nx)

        y_t_event = []
        y_t_library = []
        for i_rcv in range(len(receivers["x"])):

            tf_library = ds_gridded_tf.tf_real.sel(
                idx_rcv=i_rcv
            ) + 1j * ds_gridded_tf.tf_imag.sel(
                idx_rcv=i_rcv
            )  # (nf, ny, nx)
            tf_event = tf_library.sel(
                y=source["y"], x=source["x"], method="nearest"
            )  # (nf,)

            # Derive received spectrum (Y = SH)
            k0 = 2 * np.pi * f / g.c0
            norm_factor = np.exp(1j * k0) / (4 * np.pi)

            y_f_library = mult_along_axis(tf_library, S_f_library * norm_factor, axis=0)
            y_f_event = tf_event * S_f_event * norm_factor

            # Derive delay factor to take into account the propagation time
            tau_vec = mult_along_axis(tau_lib, f, axis=0)
            delay_library = np.exp(1j * 2 * np.pi * tau_vec)

            # Apply delay
            y_f_library *= delay_library  # (nf, ny, nx)
            y_f_event *= delay_event  # (nf,)

            # FFT inv to get signal
            y_t_l = np.fft.irfft(y_f_library, axis=0)  # (nt, ny, nx)
            y_t_e = np.fft.irfft(y_f_event)  # (nt,)

            # Store for current receiver
            y_t_library.append(y_t_l)
            y_t_event.append(y_t_e)

        y_t_library = np.array(y_t_library)  # (nrcv, nt, ny, nx)
        y_t_event = np.array(y_t_event)  # (nrcv, nt)

        # Build dataset to save
        t = np.arange(0, library_props["T"], 1 / library_props["fs"])

        ds_sig = xr.Dataset(
            coords=dict(
                idx_rcv=ds_gridded_tf.idx_rcv,
                t=t,
                y=ds_gridded_tf.y,
                x=ds_gridded_tf.x,
            ),
            data_vars=dict(
                s_l=(["idx_rcv", "t", "y", "x"], y_t_library),
                s_e=(["idx_rcv", "t"], y_t_event),
            ),
            attrs=ds_gridded_tf.attrs,
        )

        # Save dataset
        if self.debug:
            fname = f"zhang_library_dx{grid['dx']}m_dy{grid['dy']}m_debug.nc"
        else:
            fname = f"zhang_library_dx{grid['dx']}m_dy{grid['dy']}m.nc"

        fpath = os.path.join(self.root_data, fname)
        ds_sig.to_netcdf(fpath)

    def derive_received_noise(
        self,
        s_library,
        s_event,
        event_source,
        snr_dB=10,
        noise_model="gaussian",
    ):
        """
        Function to derive noise signals according to target SNR.

        Event signal and library signal at event source position do not have the exact same power due to the different nature of the source signal
        (even if the source signal are both normalized to unit variance). To account for that we need to use different noise power for library and
        event signals to ensure both reach the target SNR.

        """

        if noise_model == "gaussian":
            ## Library
            s_library_src_pos_rcv0 = s_library.sel(idx_rcv=0).sel(
                x=event_source["x"], y=event_source["y"], method="nearest"
            )
            # Library signal power at receiver n°0 and source position used as reference
            sigma_rcv_ref_library = np.std(s_library_src_pos_rcv0.values)
            # Normalize to account for the reference signal power to reach required snr at receiver n°0
            sigma_v_library = sigma_rcv_ref_library * np.sqrt(10 ** (-snr_dB / 10))
            # We assume that the noise is due to ambiant noise (hence it does not depend on the source position within the search grid) and is the same at each receiver position (receiver electronic noise )
            noise_library = np.random.normal(
                loc=0, scale=sigma_v_library, size=s_library.shape
            )

            ## Event ##
            s_event_rcv0 = s_event.sel(idx_rcv=0)
            # Event signal power at receiver n°0
            sigma_rcv_ref_event = np.std(s_event_rcv0.values)
            # Normalize to account for the reference signal power to reach required snr at receiver n°0
            sigma_v_event = sigma_rcv_ref_event * np.sqrt(10 ** (-snr_dB / 10))
            # We assume that the noise is due to ambiant noise (hence it does not depend on the source position within the search grid)
            # and is the same at each receiver position (receiver electronic noise negligible)
            noise_event = np.random.normal(
                loc=0, scale=sigma_v_event, size=s_event.shape
            )

        else:
            raise ValueError(
                f"Noise model {noise_model} not implemented. Only gaussian noise is available."
            )

        # Create dataset to store noise signals
        ds_noise = xr.Dataset(
            data_vars=dict(
                n_l=(["idx_rcv", "t", "y", "x"], noise_library),
                n_e=(["idx_rcv", "t"], noise_event),
            ),
            coords=dict(
                t=s_library.t,
                x=s_library.x,
                y=s_library.y,
                idx_rcv=s_library.idx_rcv,
            ),
            attrs=dict(
                std_ref_event=sigma_rcv_ref_event,
                std_ref_library=sigma_rcv_ref_library,
                snr=snr_dB,
            ),
        )

        if self.verbose:
            # Check SNR
            snr_rcv0_event = 10 * np.log10(
                np.var(s_event_rcv0.values) / np.var(ds_noise.n_e.sel(idx_rcv=0).values)
            )
            snr_rcv0_library = 10 * np.log10(
                np.var(s_library_src_pos_rcv0.values)
                / np.var(ds_noise.n_l.sel(idx_rcv=0).values)
            )
            print(
                f"SNR event signal at receiver n°0 : {np.round(snr_rcv0_event, 2)} dB (required {snr_dB}dB)"
            )
            print(
                f"SNR library signal at receiver n°0 : {np.round(snr_rcv0_library, 2)} dB (required {snr_dB}dB)"
            )

        return ds_noise


if __name__ == "__main__":
    # Test the class
    debug = True
    db = DataBuilder(debug=debug)
    db.build_tf_dataset()
    db.grid_dataset()
    db.build_signal()

    # Load dataset
    if debug:
        fname = f"tf_zhang_grid_dx{20}m_dy{20}m_debug.nc"
    else:
        fname = f"tf_zhang_grid_dx{20}m_dy{20}m.nc"
    fpath = os.path.join(db.root_data, fname)
    ds_gridded_tf = xr.open_dataset(fpath)  # (nf, ny, nx, nrcv)
