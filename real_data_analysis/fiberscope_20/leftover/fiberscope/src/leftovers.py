    def split_signal_noise(
        self,
        xr_data,
        signal,
        split_method: str = "rolling_power",
        alpha_th: float = 1e-4,
        plot: bool = False,
    ):

        # Unpack usefull properties
        t_interp_pulse = signal.interp_pulse_period
        n_em = signal.n_sweep
        f0 = signal.fmin
        f1 = signal.fmax

        n_hydro = xr_data.sizes["h_index"]
        ts = xr_data.ts

        # Derive the split time for the first emission received by the hydrophone n°4 (the one which is likely to received the highest number of echoes)
        # The goal of choosing a common split time is that the simultaneity of received signals is preserved (essential to derive csdm)
        hydro_for_split_time_calibration = 4
        y = xr_data.signal.sel(time=slice(0, t_interp_pulse)).sel(
            h_index=hydro_for_split_time_calibration
        )

        split_method = "rolling_power"
        if split_method == "rolling_power":
            rolling_power = y.rolling(time=1000, center=True).var().dropna("time")
            power_threshold = rolling_power.max() * alpha_th
            # power_threshold = rolling_power[-1]
            power_threshold = rolling_power[-1] * (1 + 0.5)
            # Detect last instant where the power is above the threshold
            time_with_power_above_threshold = rolling_power.time.values[
                rolling_power.values > power_threshold.values
            ]
            split_time_0 = 0
            split_time_1 = time_with_power_above_threshold[-1]

        elif split_method == "band_energy":
            # Other approach : use the energy level in the frequency band of interest to split signal and noise
            # 1) Derive the stft of the original signal (signal of interest + noise)
            nperseg = 2**12
            noverlap = nperseg // 2
            ff, tt, stft = sp.stft(
                y.values,
                fs=1 / ts,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="psd",
            )
            # 2) Compute the energy in the frequency band of interest
            f0_idx = np.argmin(np.abs(ff - f0))
            f1_idx = np.argmin(np.abs(ff - f1))
            energy_band = np.sum(np.abs(stft[f0_idx:f1_idx, :]) ** 2, axis=0) * (
                ff[1] - ff[0]
            )  # Integrate over the frequency band -> energy in V^2

            # 3) Split signal and noise based on the energy level in the frequency band of interest
            # energy_threshold = energy_band.max() * alpha_th
            energy_threshold = np.max(energy_band[tt > tt.max() * 2 / 3]) * 4
            # print(f"energy_threshold = {energy_threshold}")
            time_with_energy_above_threshold = tt[energy_band > energy_threshold]
            split_time_0 = time_with_energy_above_threshold[0]
            split_time_1 = time_with_energy_above_threshold[-1]

        # Derive the number of pulse in the signal
        y = xr_data.signal.sel(h_index=hydro_for_split_time_calibration)
        # 1) Derive the stft of the original signal (signal of interest + noise)
        nperseg = 2**12
        noverlap = nperseg // 2
        ff, tt, stft = sp.stft(
            y.values,
            fs=1 / ts,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",
        )

        # 2) Compute the energy in the frequency band of interest
        f0_idx = np.argmin(np.abs(ff - f0))
        f1_idx = np.argmin(np.abs(ff - f1))
        energy_band = np.sum(np.abs(stft[f0_idx:f1_idx, :]) ** 2, axis=0) * (
            ff[1] - ff[0]
        )  # Integrate over the frequency band -> energy in V^2

        # Detect impulsions : from below to above threshold
        energy_threshold = np.max(energy_band[tt > tt.max() * 0.9]) * 4
        impulsions = energy_band > energy_threshold

        # Count impulsions
        n_em = np.sum(np.diff(impulsions.astype(int)) == 1)

        signal_plus_noise = []
        only_noise = []
        # Loop over each hydrophone to extract signal and noise
        for i_hydro in range(n_hydro):
            sig_array = []
            noise_array = []
            # Process each emission
            for i_em in range(n_em):
                # Extract the emission
                y = xr_data.signal.sel(
                    time=slice(i_em * t_interp_pulse - ts, (i_em + 1) * t_interp_pulse)
                ).isel(h_index=i_hydro)

                # Ensure y is not empty
                if y.size == 0:
                    continue

                # Update split time for the current emission
                split_time_i_em_0 = split_time_0 + y.time.min().values
                split_time_i_em_1 = split_time_1 + y.time.min().values

                sig = y.sel(time=slice(split_time_i_em_0, split_time_i_em_1))
                noise = xr.concat(
                    [
                        y.sel(time=slice(0, split_time_i_em_0)),
                        y.sel(time=slice(split_time_i_em_1, y.time.max())),
                    ],
                    dim="time",
                )

                # Apply window at the very edge to avoid high frequency effects when combining with other emissions
                alpha_tukey = 0.01
                sig = sig * sp.windows.tukey(len(sig), alpha_tukey)
                noise = noise * sp.windows.tukey(len(noise), alpha_tukey)

                # Add signal and noise to arrays
                sig_array.append(sig.values)
                noise_array.append(noise.values)

            # Combine all emissions into a single signal and noise array
            sig_array = np.concatenate(sig_array)
            noise_array = np.concatenate(noise_array)

            # Store the signal and noise in the dedicated arrays
            signal_plus_noise.append(sig_array)
            only_noise.append(noise_array)

        # Pad with zeros to ensure all arrays have the same length
        max_len_sig = max([sig.size for sig in signal_plus_noise])
        max_len_noise = max([noise.size for noise in only_noise])
        for i_hydro in range(n_hydro):
            signal_plus_noise[i_hydro] = np.pad(
                signal_plus_noise[i_hydro],
                (0, max_len_sig - signal_plus_noise[i_hydro].size),
            )
            only_noise[i_hydro] = np.pad(
                only_noise[i_hydro], (0, max_len_noise - only_noise[i_hydro].size)
            )

        # Convert to arrays
        signal_plus_noise = np.array(signal_plus_noise)
        only_noise = np.array(only_noise)

        # Create new time vectors
        signal_plus_noise_time = np.linspace(
            0, signal_plus_noise.shape[1] * ts, signal_plus_noise.shape[1]
        )
        only_noise_time = np.linspace(0, only_noise.shape[1] * ts, only_noise.shape[1])

        # Add coordinate to the dataset
        xr_data["signal_plus_noise_time"] = signal_plus_noise_time
        xr_data["only_noise_time"] = only_noise_time

        # Add signal and noise to the dataset
        xr_data["signal_plus_noise"] = (
            ["h_index", "signal_plus_noise_time"],
            signal_plus_noise,
        )
        xr_data["only_noise"] = (
            ["h_index", "only_noise_time"],
            only_noise,
        )

        # Derive SNR in the frequency band of interest
        noise_fft = np.fft.rfft(only_noise, axis=1)
        signal_fft = np.fft.rfft(signal_plus_noise, axis=1)
        f_noise = np.fft.rfftfreq(only_noise.shape[1], d=ts)
        f_signal = np.fft.rfftfreq(signal_plus_noise.shape[1], d=ts)
        f_in_band_noise = np.logical_and(f_noise >= f0, f_noise <= f1)
        f_in_band_sig = np.logical_and(f_signal >= f0, f_signal <= f1)
        snr = np.sum(np.abs(signal_fft[:, f_in_band_sig]) ** 2, axis=1) / np.sum(
            np.abs(noise_fft[:, f_in_band_noise]) ** 2, axis=1
        )
        snr = 10 * np.log10(snr)

        # Add snr to the dataset
        xr_data["snr"] = (
            ["h_index"],
            snr,
        )

        if plot:
            plt.figure()
            xr_data.signal.plot(x="time", hue="h_index")
            plt.title("Original signal")
            plt.savefig(os.path.join(xr_data.attrs["img_path"], "original_signal.png"))

            plt.figure()
            xr_data.signal_plus_noise.plot(x="signal_plus_noise_time", hue="h_index")
            plt.title("Signal")
            plt.savefig(
                os.path.join(xr_data.attrs["img_path"], "signal_plus_noise.png")
            )

            plt.figure()
            xr_data.only_noise.plot(x="only_noise_time", hue="h_index")
            plt.title("Noise")
            plt.savefig(os.path.join(xr_data.attrs["img_path"], "only_noise.png"))
            plt.close("all")

        return xr_data