#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   data_builder.py
@Time    :   2025/05/05 11:48:20
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to build simulation data uace test cases
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import Antenna, SparseAntenna

import source.global_constants as g
from source.signal_generator import SignalGenerator
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from misc import cast_matrix_to_target_shape, mult_along_axis


class DataBuilder:
    """
    Class to build simulation data for Zhang et uace test case
    """

    def __init__(self, simulation: Simulation = None):
        self.simulation = simulation

    # =======================================================================================================================
    # Build data
    # =======================================================================================================================
    def build_tf_dataset(self):
        """
        Wrapper to build_tf_dataset_range_dependent or build_tf_dataset_range_independent
        depending on the simulation range dependency
        """
        if self.simulation.is_range_dependent:
            self.build_tf_dataset_range_dependent()
        else:
            self.build_tf_dataset_range_independent()

    def build_tf_dataset_range_dependent(self):
        km = KrakenManager(parallel=True, n_workers=p.n_workers, verbose=True)

        freq = self.simulation.library_ship[0].freq
        f = freq[(freq >= self.simulation.fmin) & (freq <= self.simulation.fmax)]
        pressure_field, field_pos = km.runkraken(
            env=self.simulation.kraken_env,
            flp=self.simulation.kraken_flp,
            frequencies=f,
        )

        tf = np.squeeze(pressure_field, axis=(1, 2, 3))  # (nf, nr)
        # Pad h_grid with 0 for frequencies outside the fmin, fmax band
        pad_before = np.sum(freq < self.simulation.fmin)
        pad_after = np.sum(freq > self.simulation.fmax)
        h_grid = np.pad(tf, ((pad_before, pad_after), (0, 0)))

        # Build xarray dataset
        tf_dataset = xr.Dataset(
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
        tf_dataset.to_netcdf(self.simulation.tf_dataset_fpath)
        tf_dataset.close()

    def build_tf_dataset_range_independent(self):
        """Step 1 : use Kraken propagation model to derive the broadband transfert function of the testcase waveguide"""

        km = KrakenManager()

        freq = self.simulation.library_ship.freq
        f = freq[(freq >= self.simulation.fmin) & (freq <= self.simulation.fmax)]

        # For too long frequencies vector field fails to compute -> we will iterate over frequency subband to compute the transfert function
        n_subband = 900
        i_subband = 1
        idx_start = 0
        idx_end = min(n_subband * i_subband, len(f) - 1)

        # Read env file
        with open(self.simulation.env_file, "r") as file:
            lines = file.readlines()

        cwd = os.getcwd()
        os.chdir(self.simulation.tmp_folder)

        first_iter = True
        while idx_start < len(f):

            # Set interval
            f0 = f[idx_start]
            f1 = f[idx_end]

            # Frequency subband
            f_kraken = f[(f >= f0) & (f <= f1)]
            # print(i_subband, f0, f1, len(f_kraken))
            pad_before = np.sum(f < f0)
            pad_after = np.sum(f > f1)

            # Modify number of frequencies
            nb_freq = f"{len(f_kraken)}                                                     ! Number of frequencies\n"
            lines[-2] = nb_freq
            # Replace frequencies in the env file
            new_freq_line = " ".join([f"{fi:.2f}" for fi in f_kraken])
            new_freq_line += "    ! Frequencies (Hz)"
            lines[-1] = new_freq_line

            # Write new env file
            with open(self.simulation.env_file, "w") as file:
                file.writelines(lines)

            # Run kraken and field
            km.run_kraken_exec(self.simulation.name)
            km.run_field_exec(self.simulation.name)

            # Read shd from previously run kraken
            shdfile = f"{self.simulation.name}.shd"
            _, _, _, _, _, _, field_pos, pressure_field = km.readshd(
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
            idx_start = idx_end + 1
            idx_end = min(n_subband * i_subband, len(f) - 1)

        os.chdir(cwd)

        # Pad h_grid with 0 for frequencies outside the fmin, fmax band
        pad_before = np.sum(freq < self.simulation.fmin)
        pad_after = np.sum(freq > self.simulation.fmax)
        h_grid = np.pad(h_grid, ((pad_before, pad_after), (0, 0)))

        # Build xarray dataset
        tf_dataset = xr.Dataset(
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

        # TODO remove
        # import matplotlib.pyplot as plt

        # tf = tf_dataset.sel(r=5 * 1e3).tf_real + 1j * tf_dataset.sel(r=5 * 1e3).tf_imag
        # plt.figure()
        # np.abs(tf).sel(f=slice(0, 55)).plot()
        # plt.savefig("test")

        # Save as netcdf
        tf_dataset.to_netcdf(self.simulation.tf_dataset_fpath)
        tf_dataset.close()

    def grid_dataset(self):
        """Step 2 : Associate each grid pixel to the corresponding broadband transfert function caracterized by the range to the receiver.

        -   Kraken tf : H(f, r)
        -   Grid : r(x, y)
        -   Gridded tf : H(f, x, y) = H(f, r(x, y))
        """

        # Load dataset
        ds = xr.open_dataset(self.simulation.tf_dataset_fpath)

        # # TODO remove
        # import matplotlib.pyplot as plt

        # tf = ds.tf_real + 1j * ds.tf_imag
        # tl = np.abs(tf)
        # tl = xr.where(tl == 0, 1e-20, tl)
        # tl = 10 * np.log10(tl)
        # plt.figure()
        # tl.sel(f=45).plot()
        # plt.savefig("test_tf.png")

        # Create new dataset
        ds_grid = xr.Dataset(
            coords=dict(
                f=ds.f.values,
                y=self.simulation.grid_y,
                x=self.simulation.grid_x,
                idx_rcv=self.simulation.antenna.rcv_idx,
            ),
            attrs=dict(
                df=ds.f.diff("f").values[0],
                dx=self.simulation.dx,
                dy=self.simulation.dy,
                testcase=self.simulation.name,
            ),
        )

        # Grid tf to the desired resolution
        # Preprocess tf to decrease the number of point for further interpolation
        # r_grid_all_rcv = np.array(
        #     [
        #         self.simulation.grid_ranges_from_rcv[i_rcv].flatten()
        #         for i_rcv in self.simulation.antenna.rcv_idx
        #     ]
        # )
        # r_grid_all_rcv_unique = np.unique(np.round(r_grid_all_rcv.flatten(), 0))

        # tf_vect = ds.tf_real.sel(
        #     r=r_grid_all_rcv_unique, method="nearest"
        # ) + 1j * ds.tf_imag.sel(r=r_grid_all_rcv_unique, method="nearest")
        tf_vect = ds.tf_real + 1j * ds.tf_imag  # No preprocessing = longuer but safer

        gridded_tf = []
        grid_shape = (ds_grid.sizes["f"],) + self.simulation.grid_ranges_from_rcv.shape[
            1:
        ]  # Try to fix search grid issues 11/02/202    (nf, ny, nx)
        for i_rcv in self.simulation.antenna.rcv_idx:
            r_grid = self.simulation.grid_ranges_from_rcv[i_rcv].flatten()
            tf_ircv = tf_vect.sel(r=r_grid, method="nearest")

            tf_grid = tf_ircv.values.reshape(grid_shape)  # (nf, ny, nx)
            gridded_tf.append(tf_grid)

        gridded_tf = np.array(gridded_tf)  # (nr, nf, ny, nx)
        # Add to dataset
        grid_coords = ["idx_rcv", "f", "y", "x"]  # Fix 11/02/2025
        ds_grid["tf_real"] = (grid_coords, np.real(gridded_tf))
        ds_grid["tf_imag"] = (grid_coords, np.imag(gridded_tf))

        # # TODO remove
        # # Plot gridded tf
        # tf_g = ds_grid.tf_real + 1j * ds_grid.tf_imag
        # tl = np.abs(tf_g)
        # tl = xr.where(tl == 0, 1e-20, tl)
        # tl = 10 * np.log10(tl)
        # plt.figure()
        # tl.isel(f=490, idx_rcv=0).plot()
        # plt.savefig("test_tf_grid.png")

        # plt.figure()
        # np.abs(tf_g).isel(idx_rcv=0, x=5, y=5).plot()
        # plt.xlim([0, 50])
        # plt.savefig("test_tf_grid_2.png")

        # Save dataset
        ds_grid.to_netcdf(self.simulation.tf_grid_dataset_fpath)
        ds_grid.close()

    def build_signal(self):
        """Step 3 : derive signal received from each grid pixel using library source spectrum and gridded transfert functions.

        -   Gridded tf : H(x, y, f)
        -   Source spectrum : S(f)
        -   Gridded spectrum : Y(x, y, f) = S(f) H(x, y, f)
        -   Gridded signal : y(x, y, t) = FFT_inv(Y(x, y, f))

        """
        # Load gridded dataset
        ds_gridded_tf = xr.open_dataset(
            self.simulation.tf_grid_dataset_fpath
        )  # (nf, ny, nx, nrcv)

        # Limit max frequency to speed up
        # fs_target = 1200
        # fmax = fs_target / 2
        ds_gridded_tf = ds_gridded_tf.sel(f=slice(0, self.simulation.fs / 2))

        # Library / event spectrum
        # S_f_library = self.simulation.library_ship.spectrum

        # Update 17/05/2025 -> multiple library sources
        # Build 3d array by randomly picking library ship from the given list of ship
        target_shape = ds_gridded_tf.tf_real.sel(idx_rcv=0).shape
        nf, ny, nx = target_shape
        num_lib_sources = self.simulation.library_ship.shape[0]
        # Step 1: generate random indices for each (x, y)
        random_indices = np.random.randint(
            0, num_lib_sources, size=target_shape[1:]
        )  # shape (ny, nx)

        # Plot ship distributuion for analysis
        if self.simulation.plot_library_ship_distribution:
            # Define associated dataset for easy plot
            ds_rng_idx = xr.Dataset(
                coords=dict(
                    y=ds_gridded_tf.y,
                    x=ds_gridded_tf.x,
                ),
                data_vars=dict(
                    library_ship_id=(["y", "x"], random_indices),
                ),
            )
            # Plot distribution
            plt.figure()
            ds_rng_idx.library_ship_id.plot(x="x", y="y")
            plt.xlabel(r"$x$" + " [m]")
            plt.ylabel(r"$y$" + " [m]")
            plt.gca().set_aspect("equal")
            fpath = os.path.join(
                self.simulation.img_folder, "library_ship_distribution.png"
            )
            plt.savefig(fpath)

            # Save distibution for further analysis
            ds_rng_idx.to_netcdf(
                os.path.join(
                    self.simulation.data_folder, "library_ship_distribution.nc"
                )
            )

        # Step 2: reshape for indexing — flatten the index grid
        flat_indices = random_indices.ravel()  # shape (ny*nx,)

        # Step 3: extract the selected vectors — result shape: (ny*nx, nf)
        spectrums_vector = np.array(
            [l_ship.spectrum for l_ship in self.simulation.library_ship]
        )
        selected_vectors = spectrums_vector[flat_indices]  # shape (ny*nx, nf)

        # Step 4: reshape to (nf, ny, nx)
        S_f_library_mat = np.moveaxis(
            selected_vectors.reshape(ny, nx, nf), 2, 0
        )  # (nf, ny, nx)

        S_f_event = self.simulation.event_ship.spectrum

        # Derive delay for each receiver
        delay_rcv = (
            self.simulation.grid_ranges_from_rcv / self.simulation.cmin
        )  # (nrcv, ny, nx)

        # Same delay is applied to each receiver : the receiver with the minimum delay is taken as the time reference
        # (we are only interested in relative time difference)
        # tau = ds_gridded_tf.delay_rcv.min(dim="idx_rcv")
        f = ds_gridded_tf.f
        tau = np.min(delay_rcv, axis=0)  # (ny, nx)
        # Aplly small offset to avoid wrapping a small part of the signal around
        tau_offset = 0.2
        tau = tau - tau_offset
        da_tau = xr.DataArray(
            data=tau, coords=dict(y=ds_gridded_tf.y, x=ds_gridded_tf.x)
        )
        tau_event = da_tau.sel(
            y=self.simulation.event_ship_y,
            x=self.simulation.event_ship_x,
            method="nearest",
        )
        delay_event = np.exp(1j * 2 * np.pi * tau_event * f)

        # Cast tau gridded tf shape
        tau_lib = cast_matrix_to_target_shape(
            tau, ds_gridded_tf.tf_real.shape[1:]
        )  # (nf, ny, nx)

        y_t_event = []
        y_t_library = []
        for i_rcv in self.simulation.antenna.rcv_idx:

            tf_library = ds_gridded_tf.tf_real.sel(
                idx_rcv=i_rcv
            ) + 1j * ds_gridded_tf.tf_imag.sel(
                idx_rcv=i_rcv
            )  # (nf, ny, nx)
            tf_event = tf_library.sel(
                y=self.simulation.event_ship_y,
                x=self.simulation.event_ship_x,
                method="nearest",
            )  # (nf,)

            # Derive received spectrum (Y = SH)
            k0 = 2 * np.pi * f / g.c0
            norm_factor = np.exp(1j * k0) / (4 * np.pi)

            # y_f_library = mult_along_axis(tf_library, S_f_library * norm_factor, axis=0)
            y_f_library = mult_along_axis(
                tf_library * S_f_library_mat, norm_factor, axis=0
            )

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

        # TODO : think about adding propagated interference signal here !

        # Build dataset to save
        # t = np.arange(0, self.simulation., 1 / library_props["fs"])
        t = self.simulation.library_ship[0].time  # (nt,)

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
        ds_sig.to_netcdf(self.simulation.library_dataset_fpath)

    def derive_received_noise(
        self,
        s_library: xr.DataArray,
        s_event: xr.DataArray,
        snr_dB: float = 10,
        noise_model: str = "gaussian",
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
                x=self.simulation.event_ship_x,
                y=self.simulation.event_ship_y,
                method="nearest",
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

        if self.simulation.verbose:
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
    # # Test the class
    # debug = False
    # antenna = SparseAntenna(
    #     name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    # )
    # simu = Simulation(debug=debug, antenna=antenna)
    # db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # db.grid_dataset()
    # db.build_signal()

    # Test build tf broadband and rangeindependent
    from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
        DeepWaterRealEnv,
    )

    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )

    debug = False
    check = True
    n_mc = 1
    use_weighted_rtf = True
    name = "dw_real_env"

    # Test multiple library ship
    from propa.rtf.rtf_localisation.uace_testcase.src.ship_signal import ShipSignal

    nl_ship = 10
    library_ship = []
    for iship in range(nl_ship):
        f0_l = np.random.uniform(low=p.f0_min, high=p.f0_max)
        std_fi_l = np.random.uniform(low=p.std_fi_min, high=p.std_fi_max) * f0_l
        tau_corr_fi_l = (
            np.random.uniform(low=p.tau_corr_fi_min, high=p.tau_corr_fi_max) * 1 / f0_l
        )
        library_ship_i = ShipSignal(
            name="library_ship",
            f0=f0_l,
            fs=p.fs,
            duration=p.duration,
            std_fi=std_fi_l,
            tau_corr_fi=tau_corr_fi_l,
            root_img=p.root_img_ship_sigs,
        )
        library_ship.append(library_ship_i)

    search_area_length = 0.5 * 1e3
    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        library_ship=library_ship,
        monte_carlo_iterations=n_mc,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name="dw_real_env")
    db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # db.grid_dataset()
    db.build_signal()

    # # Load dataset
    # if debug:
    #     fname = f"tf_zhang_grid_dx{20}m_dy{20}m_debug.nc"
    # else:
    #     fname = f"tf_zhang_grid_dx{20}m_dy{20}m.nc"
    # fpath = os.path.join(db.root_data, fname)
    # ds_gridded_tf = xr.open_dataset(fpath)  # (nf, ny, nx, nrcv)
