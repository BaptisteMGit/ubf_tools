# # ### Coherence spectrogram directly derived from STFT

# # Plot coherence between receiver couples
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])

# for i, rcv_couple in enumerate(crosscorr_data.keys()):
#     for j, mmsi in enumerate(mmsi_selected):
#         df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
#         ship_name = df_mmsi["shipName"].values[0]
#         ax[0, j].set_title(f"{ship_name}")

#         coh = coh_xy[rcv_couple][mmsi]["coh_xy"]
#         f = coh_xy[rcv_couple][mmsi]["f"]
#         tt = coh_xy[rcv_couple][mmsi]["tt"]
#         tt_hour = tt / 3600
#         im = ax[i, j].pcolormesh(tt_hour, f, coh, cmap="jet", shading="gouraud")

#         ax[i, j].legend(loc="upper right", fontsize=12)

#         # Add colorbar
#         fig.colorbar(im, ax=ax[i, j])


# fig.supxlabel("Time [h]")
# fig.supylabel("Frequency [Hz]")

# # Save figure
# fig_name = f"coh_from_stft_T{delta_t_stat//60}min"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# Adjust space for colorbar and ensure it doesn't overlap with subplots
# fig.subplots_adjust(
#     right=0.85, hspace=0.4, wspace=0.4
# )  # Adjust right space for the colorbar

# # Add colorbar spanning all subplots, positioned to the right of the subplots
# cbar_ax = fig.add_axes(
#     [0.86, 0.15, 0.03, 0.7]
# )  # Adjust position to fit the colorbar outside
# fig.colorbar(im, cax=cbar_ax)

# Derive coherence for the entire signal
# coh, sxx, syy, sxy = compute_dsps(wav_data, rcv_info)

# # %%
# # Plot DSPs
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])

# for i, rcv_couple in enumerate(sxx.keys()):
#     for j, mmsi in enumerate(mmsi_selected):
#         df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
#         ship_name = df_mmsi["shipName"].values[0]
#         ax[0, j].set_title(f"{ship_name}")

#         Sxx = sxx[rcv_couple][mmsi]["Sxx"]
#         Syy = syy[rcv_couple][mmsi]["Syy"]
#         Sxy = sxy[rcv_couple][mmsi]["Sxy"]
#         f = sxx[rcv_couple][mmsi]["f"]
#         im = ax[i, j].plot(f, 10 * np.log10(Sxx), label="Sxx")
#         ax[i, j].plot(f, 10 * np.log10(Syy), label="Syy")
#         ax[i, j].plot(f, 10 * np.log10(np.abs(Sxy)), label="Sxy")

#         ax[i, j].set_xlim(fmin - 2, fmax + 2)
#         ax[i, j].set_ylim(-100, 10)

#         # Add text with the name of the receiver couple
#         ax[i, j].text(
#             0.015,
#             0.85,
#             rcv_couple,
#             transform=ax[i, j].transAxes,
#             ha="left",
#             fontsize=10,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

#     ax[0, j].legend(
#         fontsize=12, ncol=n_couple, bbox_to_anchor=(1, 1.05), loc="upper right"
#     )

# fig.supxlabel("Frequency [Hz]")
# fig.supylabel("Power [dB]")

# # Save figure
# fig_name = "dsps"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# # %%
# # Plot coherence spectrum
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])

# for i, rcv_couple in enumerate(coh.keys()):
#     for j, mmsi in enumerate(mmsi_selected):
#         df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
#         ship_name = df_mmsi["shipName"].values[0]
#         ax[0, j].set_title(f"{ship_name}")

#         coh_xy = coh[rcv_couple][mmsi]["coh_xy"]
#         f = coh[rcv_couple][mmsi]["f"]
#         im = ax[i, j].plot(f, coh_xy, label=f"{rcv_couple}")
#         # ax[i, j].plot(f, coh[rcv_couple][mmsi]["coh_xy_sp"], label=f"{rcv_couple} sp")

#         # ax[i, j].set_xlabel("Frequency [Hz]")
#         # ax[i, j].set_ylabel(r"$\gamma_{xy}$")
#         # ax[i, j].legend(loc="upper right", fontsize=12)

#         # Add text with the name of the receiver couple
#         ax[i, j].text(
#             0.015,
#             0.85,
#             rcv_couple,
#             transform=ax[i, j].transAxes,
#             ha="left",
#             fontsize=10,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

#         ax[i, j].set_xlim(fmin - 2, fmax + 2)
#         ax[i, j].set_ylim(0, 0.1)

# fig.supxlabel("Frequency [Hz]")
# fig.supylabel(r"$\gamma_{xy}$")
# plt.suptitle("Coherence")

# # Define a threshold on coherence to detect ship rays
# coh_threshold = 0.01

# # Plot threshold on coherence
# for i, rcv_couple in enumerate(coh.keys()):
#     for j, mmsi in enumerate(mmsi_selected):
#         ax[i, j].axhline(coh_threshold, color="r", linestyle="--", label="Threshold")

# # Store frequencies where coherence is above threshold
# f_detected = {}
# for i, rcv_couple in enumerate(coh.keys()):
#     f_detected[rcv_couple] = {}
#     for mmsi in mmsi_selected:
#         # coh_above_th = coh[rcv_couple][mmsi]["coh_xy"] > coh_threshold

#         # f_d = coh[rcv_couple][mmsi]["f"][coh[rcv_couple][mmsi]["coh_xy"] > coh_threshold]
#         f_peaks = sp.find_peaks(coh[rcv_couple][mmsi]["coh_xy"], height=coh_threshold)[
#             0
#         ]
#         f_detected[rcv_couple][mmsi] = {}
#         f_detected[rcv_couple][mmsi]["idx"] = f_peaks
#         f_detected[rcv_couple][mmsi]["f"] = coh[rcv_couple][mmsi]["f"][f_peaks]

# # Plot detected frequencies
# for i, rcv_couple in enumerate(coh.keys()):
#     for j, mmsi in enumerate(mmsi_selected):
#         ax[i, j].scatter(
#             f_detected[rcv_couple][mmsi]["f"],
#             coh[rcv_couple][mmsi]["coh_xy"][f_detected[rcv_couple][mmsi]["idx"]],
#             color="r",
#             label="Detected frequencies",
#         )

# # Save figure
# fig_name = "coh_entire_sig"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))


# # %%
# # Derive coherence over time using Welch method
# n_period_per_sub_segment = 30
# max_signal_period = 1 / fmin
# nperseg_sub = int(max_signal_period * n_period_per_sub_segment * fs)
# noverlap_sub = int(nperseg_sub * 2 / 3)


# coh_perio = {}
# for mmsi in mmsi_selected:
#     coh_perio[mmsi] = {}
#     for i, rcv_id1 in enumerate(rcv_info["id"]):
#         for j, rcv_id2 in enumerate(rcv_info["id"]):
#             if i < j:
#                 rcv_couple_id = f"{rcv_id1}_{rcv_id2}"

#                 s1 = wav_data[rcv_id1][mmsi]
#                 s2 = wav_data[rcv_id2][mmsi]
#                 fs = s1["sig"].meta.sampling_rate

#                 nperseg = np.floor(Tcorr[mmsi] * fs).astype(int)  # Window length
#                 noverlap = int(nperseg * 2 / 3)

#                 frequencies, time, coherence_matrix = compute_coherence_spectrogram(
#                     s1["data"],
#                     s2["data"],
#                     fs,
#                     nperseg,
#                     noverlap,
#                     nperseg_sub,
#                     noverlap_sub,
#                 )

#                 coh_perio[mmsi][rcv_couple_id] = {}
#                 coh_perio[mmsi][rcv_couple_id]["tt"] = time
#                 coh_perio[mmsi][rcv_couple_id]["f"] = frequencies
#                 coh_perio[mmsi][rcv_couple_id]["coh"] = coherence_matrix

# # %%
# # Plot coherence spectrogram
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])

# for j, mmsi in enumerate(mmsi_selected):
#     df_mmsi = ais_data[ais_data["mmsi"] == mmsi]
#     ship_name = df_mmsi["shipName"].values[0]
#     ax[0, j].set_title(f"{ship_name}")
#     for i, rcv_couple in enumerate(coh_perio[mmsi].keys()):
#         tt = coh_perio[rcv_couple][mmsi]["tt"]
#         tt_hour = tt / 3600
#         f = coh_perio[rcv_couple][mmsi]["f"]
#         coh = coh_perio[rcv_couple][mmsi]["coh"]
#         im = ax[i, j].pcolormesh(
#             tt_hour, f, 10 * np.log10(coh), shading="gouraud", cmap="jet", vmin=-15
#         )

#         ax[i, j].text(
#             0.015,
#             0.85,
#             rcv_couple,
#             transform=ax[i, j].transAxes,
#             ha="left",
#             fontsize=10,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

#         ax[i, j].set_ylim(fmin, fmax)

#         # Overlay the cross spectrum with different x-axis
#         # f = sxy[rcv_couple][mmsi]["f"]
#         # Sxy = sxy[rcv_couple][mmsi]["Sxy"]

#         # ax2 = ax[i, j].twiny()
#         # # shift = 1 - np.max(10*np.log10(np.abs(Sxy)))
#         # ax2.plot(10*np.log10(np.abs(Sxy)), f, color="white", linestyle="--", linewidth=2)
#         # ax2.set_ylim(fmin, fmax)
#         # ax2.set_xlim(-80, 10)
#         # ax2.set_xticklabels([])

#         # Add colorbar
#         cbar = fig.colorbar(im, ax=ax[i, j])

#         # Add detected frequencies as dotted lines
#         f_detected_couple = f_detected[rcv_couple][mmsi]["f"]
#         for f_d in f_detected_couple:
#             ax[i, j].axhline(
#                 f_d,
#                 color="w",
#                 linestyle="--",
#             )


# fig.supxlabel("Time [h]")
# fig.supylabel("Frequency [Hz]")

# # Save figure
# fig_name = "coh_perio"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# %% [markdown]
# ## Study coherence
# Try to study TDOA using coherence. To do so :
# - Select one ship
# - Select a portion of the signal excluding the CPA (either before or after CPA)
# - Derive coherence for this portion of signal
# - Derive coherence for the same portion of signal by shifting the signal from the second receiver by the TDOA
# - Derive coherence for a set of shifting time
#

# %% [markdown]
# ### Select signal portion

# # %%
# # Select one of the ship to analyze
# mmsi = mmsi_selected[0]
# ais_mmsi = ais_data[ais_data["mmsi"] == mmsi]
# wav_mmsi = {rcv_k: {mmsi: wav_data[rcv_k][mmsi].copy()} for rcv_k in wav_data.keys()}

# # fig, ax = plot_spectrograms(wav_data=wav_mmsi, ais_data=ais_mmsi)

# wav_data = wav_mmsi
# ais_data = ais_mmsi
# delta_f = 2
# rcv_0 = list(wav_data.keys())[0]
# available_mmsi = list(wav_data[rcv_0].keys())
# available_stations = list(wav_data.keys())

# # %%
# idx_time_trajectory_start = 10
# idx_time_trajectory_end = 500

# # %%
# # %matplotlib inline
# vmin = -100
# vmax = 0
# fig, ax = plt.subplots(len(available_mmsi), len(available_stations), sharex=True)

# # Reshape if only one ship
# if len(available_mmsi) == 1:
#     ax = np.array([ax])

# for i, rcv_id in enumerate(available_stations):
#     for j, mmsi in enumerate(available_mmsi):
#         stft = wav_data[rcv_id][mmsi]["stft"]
#         f = wav_data[rcv_id][mmsi]["f"]
#         tt = wav_data[rcv_id][mmsi]["tt"]
#         ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

#         tt_hour = tt / 3600
#         # ax[j, i].pcolormesh(tt, f, 20*np.log10(np.abs(stft)), shading="gouraud", vmin=vmin, vmax=vmax)
#         ax[j, i].pcolormesh(
#             tt_hour,
#             f,
#             20 * np.log10(np.abs(stft)),
#             shading="gouraud",
#             vmin=vmin,
#             vmax=vmax,
#         )

#         # ax[j, 0].set_ylabel("Frequency [Hz]")
#         ax[j, i].text(
#             0.02,
#             0.9,
#             ship_name,
#             transform=ax[j, i].transAxes,
#             ha="left",
#             fontsize=12,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )
#         # ax[j, i].set_title(f"{ship_name}")
#         ax[j, i].legend(loc="upper right", fontsize=7)
#         ax[j, i].set_ylim([fmin - delta_f, fmax + delta_f])

#     # ax[-1, i].set_xlabel("Time [s]")
#     ax[0, i].set_title(f"Station {rcv_id}")
#     # ax[-1, i].set_xlabel("Time [h]")

# fig.supxlabel("Time [h]")
# fig.supylabel("Frequency [Hz]")


# # ax = plt.gca()
# # Add arrow pointing the begining of the selected signal portion
# for i, rcv_id in enumerate(wav_data.keys()):

#     # Add arrow pointing the beginning of the selected window
#     tt = wav_mmsi[rcv_id][mmsi]["tt"]
#     tt_hour = tt / 3600
#     ax[0, i].annotate(
#         "Start",
#         xy=(tt_hour[idx_time_trajectory_start], 2),
#         xytext=(tt_hour[idx_time_trajectory_start], 0),
#         arrowprops=dict(facecolor="red", edgecolor="k", width=2),
#         ha="center",
#     )
#     # Add vertical dotted line
#     ax[0, i].axvline(
#         tt_hour[idx_time_trajectory_start],
#         color="r",
#         linestyle="--",
#     )

#     # Add arrow pointing the end of the selected window
#     ax[0, i].annotate(
#         "End",
#         xy=(tt_hour[idx_time_trajectory_end], 2),
#         xytext=(tt_hour[idx_time_trajectory_end], 0),
#         arrowprops=dict(facecolor="red", edgecolor="k", width=2),
#         ha="center",
#     )
#     # Add vertical dotted line
#     ax[0, i].axvline(
#         tt_hour[idx_time_trajectory_end],
#         color="r",
#         linestyle="--",
#     )

#     # Draw horizontal arrow from the middlde of the previous dotted line to the end of the recording
#     ax[0, i].annotate(
#         "",
#         xy=(tt_hour[idx_time_trajectory_end], 10),
#         xytext=(tt_hour[idx_time_trajectory_start], 10),
#         arrowprops=dict(facecolor="red", edgecolor="k", width=2),
#         ha="center",
#     )

#     ax[0, i].set_title(f"Station {rcv_id}")

# # Save figure
# fig_name = f"stft_{ship_name}_selected_portion"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# # %%
# # Convert idx_time_traj into idx relative to the data vector
# tt_start = wav_mmsi[rcv_id][mmsi]["tt"][idx_time_trajectory_start]
# tt_end = wav_mmsi[rcv_id][mmsi]["tt"][idx_time_trajectory_end]

# # Limit data to the selected time window
# for i, rcv_id in enumerate(wav_data.keys()):
#     time = wav_mmsi[rcv_id][mmsi]["sig"].times()
#     idx_of_interest = (time >= tt_start) & (time <= tt_end)
#     wav_mmsi[rcv_id][mmsi]["data"] = wav_mmsi[rcv_id][mmsi]["data"][idx_of_interest]
#     # wav_mmsi[rcv_id][mmsi]["tt"] = wav_mmsi[rcv_id][mmsi]["tt"][
#     #     idx_time_trajectory_start:idx_time_trajectory_end
#     # ]
#     # wav_mmsi[rcv_id][mmsi]["stft"] = wav_mmsi[rcv_id][mmsi]["stft"][
#     #     idx_time_trajectory_start:idx_time_trajectory_end
#     # ]

# # %%
# # Derive coherence without lag
# coh_no_lag, sxx, syy, sxy = compute_dsps(wav_mmsi, rcv_info)

# # Plot this coherence
# n_couple = len(coh_no_lag[mmsi].keys())
# n_ships = len(coh_no_lag.keys())
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])
# if n_ships == 1:
#     ax = np.array([ax])

# for j, mmsi_j in enumerate(coh_no_lag.keys()):
#     for i, rcv_couple in enumerate(coh_no_lag[mmsi_j].keys()):
#         coh_xy = coh_no_lag[mmsi_j][rcv_couple]["coh_xy"]
#         f = coh_no_lag[mmsi_j][rcv_couple]["f"]
#         ax[i, j].plot(f, coh_xy, label=f"{rcv_couple}")
#         ax[i, j].set_xlim(fmin - 2, fmax + 2)

#         idx_f_in_band = (f >= fmin) & (f <= fmax)
#         max_coh = np.max(coh_xy[idx_f_in_band])
#         ax[i, j].set_ylim(0, max_coh * 1.01)

#         # Add text with the name of the receiver couple
#         ax[i, j].text(
#             0.015,
#             0.85,
#             rcv_couple,
#             transform=ax[i, j].transAxes,
#             ha="left",
#             fontsize=10,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

# fig.supxlabel("Frequency [Hz]")
# fig.supylabel(r"$\gamma_{xy}$")

# # Save figure
# fig_name = "coh_no_lag"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# # %%
# # Derive coherence with the shift corresponding to the theoretical time delay between the two receivers
# idx_mmsi = np.argmin(
#     np.abs(np.array([mmsi_selected[i] - mmsi for i in range(len(mmsi_selected))]))
# )

# rcv_id1 = list(wav_data.keys())[0]
# rcv_id2 = list(wav_data.keys())[1]
# # Select the time delay corresponding to the middle of the selected signal portion
# tt_middle_portion = wav_mmsi[rcv_id1][mmsi]["tt"][
#     len(wav_mmsi[rcv_id1][mmsi]["tt"]) // 2
# ]
# # Convert to datetime using start time of the recording
# dt_middle_portion = start_times[idx_mmsi] + pd.Timedelta(seconds=tt_middle_portion)
# # Find the closest time in the AIS data
# idx_closest = np.argmin(np.abs(ais_data["time"] - dt_middle_portion))
# # Select the time delay corresponding to the closest time
# th_delay = time_delay[rcv_couple][mmsi][idx_closest]
# # Convert to index shift
# idx_shift = int(th_delay * fs)


# print(f"Time delay : {th_delay} s")
# print(f"Index shift : {idx_shift}")

# if idx_shift > 0:
#     idx_rcv_to_shift = 0
# else:
#     idx_rcv_to_shift = 1
#     idx_shift = -idx_shift

# # Shift the second time series by idx_shift
# for i, rcv_id in enumerate(wav_data.keys()):
#     if i == idx_rcv_to_shift:
#         # Do not shift the first time serie but remove the last points to keep sizes equal
#         wav_mmsi[rcv_id][mmsi]["data"] = wav_mmsi[rcv_id][mmsi]["data"][:-idx_shift]
#         # wav_mmsi[rcv_id][mmsi]["tt"] = wav_mmsi[rcv_id][mmsi]["tt"][:-idx_shift]
#         # wav_mmsi[rcv_id][mmsi]["stft"] = wav_mmsi[rcv_id][mmsi]["stft"][:-idx_shift]
#     else:
#         # Shift the second time serie and remove first points to keep sizes equal
#         wav_mmsi[rcv_id][mmsi]["data"] = wav_mmsi[rcv_id][mmsi]["data"][idx_shift:]
#         # wav_mmsi[rcv_id][mmsi]["tt"] = wav_mmsi[rcv_id][mmsi]["tt"][idx_shift:]
#         # wav_mmsi[rcv_id][mmsi]["stft"] = wav_mmsi[rcv_id][mmsi]["stft"][idx_shift:]


# # %%
# # Compute coherence with the shifted signal
# coh_shifted, sxx, syy, sxy = compute_dsps(wav_mmsi, rcv_info)

# # Detect peak frequencies in coherences
# coh_threshold = 0.04
# f_detected_no_lag = {}
# f_detected_shifted = {}
# for j, rcv_couple in enumerate(coh_no_lag[mmsi].keys()):
#     f_detected_no_lag[rcv_couple] = {}
#     f_detected_shifted[rcv_couple] = {}
#     f_detected_no_lag[rcv_couple]["idx"] = sp.find_peaks(
#         coh_no_lag[rcv_couple][mmsi]["coh_xy"], height=coh_threshold
#     )[0]
#     f_detected_shifted[rcv_couple]["idx"] = sp.find_peaks(
#         coh_shifted[rcv_couple][mmsi]["coh_xy"], height=coh_threshold
#     )[0]
#     f_detected_no_lag[rcv_couple]["f"] = coh_no_lag[rcv_couple][mmsi]["f"][
#         f_detected_no_lag[rcv_couple]["idx"]
#     ]
#     f_detected_shifted[rcv_couple]["f"] = coh_shifted[rcv_couple][mmsi]["f"][
#         f_detected_shifted[rcv_couple]["idx"]
#     ]

# # plt.show()

# # Plot the coherence with and without shift
# fig, ax = plt.subplots(n_couple, n_ships, sharey=True, sharex=True)

# # Reshape ax if only one couple
# if n_couple == 1:
#     ax = np.array([ax])
# if n_ships == 1:
#     ax = np.array([ax])

# for j, mmsi_j in enumerate(coh_no_lag.keys()):
#     for i, rcv_couple in enumerate(coh_no_lag[mmsi_j].keys()):
#         coh_xy = coh_no_lag[mmsi_j][rcv_couple]["coh_xy"]
#         f = coh_no_lag[mmsi_j][rcv_couple]["f"]
#         ax[i, j].plot(f, coh_xy, label=f"{rcv_couple} no shift", color="r")
#         coh_xy = coh_shifted[mmsi_j][rcv_couple]["coh_xy"]
#         f = coh_shifted[mmsi_j][rcv_couple]["f"]
#         ax[i, j].plot(f, coh_xy, label=f"{rcv_couple} shift", color="b")
#         ax[i, j].set_xlim(fmin - 2, fmax + 2)

#         idx_f_in_band = (f >= fmin) & (f <= fmax)
#         max_coh = max(
#             np.max(coh_no_lag[mmsi_j][rcv_couple]["coh_xy"][idx_f_in_band]),
#             np.max(coh_shifted[mmsi_j][rcv_couple]["coh_xy"][idx_f_in_band]),
#         )
#         ax[i, j].set_ylim(0, max_coh * 1.01)

#         # Add text with the name of the receiver couple
#         ax[i, j].text(
#             0.015,
#             0.85,
#             rcv_couple,
#             transform=ax[i, j].transAxes,
#             ha="left",
#             fontsize=10,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

#         # Add detected peaks as scatter points
#         f_detected_no_lag_couple = f_detected_no_lag[rcv_couple]["f"]
#         f_detected_shifted_couple = f_detected_shifted[rcv_couple]["f"]

#         ax[i, j].scatter(
#             f_detected_no_lag_couple,
#             coh_no_lag[mmsi_j][rcv_couple]["coh_xy"][
#                 f_detected_no_lag[rcv_couple]["idx"]
#             ],
#             color="r",
#             label="Detected frequencies no shift",
#         )
#         ax[i, j].scatter(
#             f_detected_shifted_couple,
#             coh_shifted[mmsi_j][rcv_couple]["coh_xy"][
#                 f_detected_shifted[rcv_couple]["idx"]
#             ],
#             color="b",
#             label="Detected frequencies shift",
#         )

#         ax[i, j].legend(loc="upper right", fontsize=12)
# fig.supxlabel("Frequency [Hz]")
# fig.supylabel(r"$\gamma_{xy}$")

# # Save figure
# fig_name = "coh_no_lag_vs_shifted"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# # Plot dsp
# plt.figure()
# plt.plot(
#     sxx[rcv_couple][mmsi]["f"], 10 * np.log10(sxx[rcv_couple][mmsi]["Sxx"]), label="Sxx"
# )
# plt.plot(
#     syy[rcv_couple][mmsi]["f"], 10 * np.log10(syy[rcv_couple][mmsi]["Syy"]), label="Syy"
# )
# plt.plot(
#     sxy[rcv_couple][mmsi]["f"],
#     10 * np.log10(np.abs(sxy[rcv_couple][mmsi]["Sxy"])),
#     label="Sxy",
# )
# plt.xlim(fmin - 2, fmax + 2)
# plt.legend()

# # Save figure
# fig_name = "dsps_shifted"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))

# # %%
# # Plot spectrograms and detected frequencies

# # First : spectrograms and detected frequencies without shift
# wav_data = wav_mmsi
# ais_data = ais_mmsi
# delta_f = 2
# rcv_0 = list(wav_data.keys())[0]
# available_mmsi = list(wav_data[rcv_0].keys())
# available_stations = list(wav_data.keys())

# fig, ax = plt.subplots(len(available_mmsi), len(available_stations), sharex=True)

# # Reshape if only one ship
# if len(available_mmsi) == 1:
#     ax = np.array([ax])

# for i, rcv_id in enumerate(available_stations):
#     for j, mmsi in enumerate(available_mmsi):
#         stft = wav_data[rcv_id][mmsi]["stft"]
#         f = wav_data[rcv_id][mmsi]["f"]
#         tt = wav_data[rcv_id][mmsi]["tt"]
#         ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

#         tt_hour = tt / 3600
#         ax[j, i].pcolormesh(
#             tt_hour,
#             f,
#             20 * np.log10(np.abs(stft)),
#             shading="gouraud",
#             vmin=vmin,
#             vmax=vmax,
#         )

#         ax[j, i].text(
#             0.02,
#             0.9,
#             ship_name,
#             transform=ax[j, i].transAxes,
#             ha="left",
#             fontsize=12,
#             bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
#         )

#         # Add detected frequencies as dotted lines
#         rcv_couple = list(coh_no_lag[mmsi].keys())[0]
#         f_detected_couple = f_detected_no_lag[rcv_couple]["f"]
#         f_detected_couple_shift = f_detected_shifted[rcv_couple]["f"]
#         for f_d in f_detected_couple:
#             ax[j, i].axhline(
#                 f_d,
#                 color="w",
#                 linestyle="--",
#             )
#         for f_d in f_detected_couple_shift:
#             ax[j, i].axhline(
#                 f_d,
#                 color="r",
#                 linestyle="--",
#             )

#         ax[j, i].legend(loc="upper right", fontsize=7)
#         ax[j, i].set_ylim([fmin - delta_f, fmax + delta_f])

#     ax[0, i].set_title(f"Station {rcv_id}")

# fig.supxlabel("Time [h]")
# fig.supylabel("Frequency [Hz]")

# # Save figure
# fig_name = "spectro_no_lag_vs_shifted"
# plt.savefig(os.path.join(img_root, img_folder_name, f"{fig_name}.png"))
