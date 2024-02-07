import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load global report
# path_global_report = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden_process_analysis\verlinden_1_test_case\pulse\not_on_grid\global_report.txt"
# data = pd.read_csv(path_global_report, sep=",")

# data.loc[data["SNR"] == " None", "SNR"] = 20
# data["SNR"] = data["SNR"].astype(float)

# metrics_to_plot = data.columns[2:]
# metrics_to_plot = ["RMSE", "STD"]

# fig, axes = plt.subplots(
#     nrows=len(metrics_to_plot), ncols=1, figsize=(10, 6 * len(metrics_to_plot))
# )

# for i, metric in enumerate(metrics_to_plot):
#     axes[i].bar(data["SNR"], data[metric], label="intercorr0", alpha=0.7)
#     axes[i].bar(data["SNR"] + 0.25, data[metric], label="lstsquares", alpha=0.7)
#     axes[i].bar(
#         data["SNR"] + 0.5, data[metric], label="hilbert_env_intercorr0", alpha=0.7
#     )

#     # Ajouter des étiquettes et une légende
#     axes[i].set_xlabel("SNR")
#     axes[i].set_ylabel(metric)
#     # axes[i].set_title(f"{metric} en fonction du SNR pour chaque métrique de détection")
#     axes[i].legend()

# # Ajuster l'espacement entre les sous-graphiques
# plt.tight_layout()

# # Afficher le graphique
# plt.show()


def plot_localisation_performance(
    data, detection_metric_list, metrics_to_plot, img_path
):
    # Sort data by SNR
    noisy_data = data[data["SNR"] != " None"].astype({"SNR": np.float32})
    noisy_data = noisy_data.sort_values(by=["SNR"])
    data = pd.concat([noisy_data, data[data["SNR"] == " None"]])

    plt.figure(figsize=(10, 6))
    bar_width = 0.2

    # Define positions of the bars
    positions = np.arange(len(data["SNR"].unique()))
    m = len(detection_metric_list)
    alpha = np.arange(-(m // 2), (m // 2) + m % 2, 1) + ((m + 1) % 2) / 2

    # Create bar plot
    bar_values = np.array([])
    all_pos = np.array([])
    colors = []
    for i_m, metric in enumerate(metrics_to_plot):
        for i_dm, detection_metric in enumerate(detection_metric_list):
            bar = data[data["Detection metric"] == detection_metric][metric]
            if not bar.empty:
                b = plt.bar(
                    positions + alpha[i_dm] * bar_width,
                    bar.values,
                    width=bar_width,
                    label=f"{detection_metric} - {metric}",
                )

                # Add value labels to bars
                # +3 / 4 * bar_width * np.sign(alpha[i_dm])
            pos = (
                positions
                + alpha[i_dm] * bar_width
                + 3 / 4 * bar_width * np.sign(alpha[i_dm])
            )
            all_pos = np.concatenate((all_pos, pos))
            bar_values = np.concatenate((bar_values, bar.values))
            colors += [b.patches[0].get_facecolor()] * len(bar)

    val_offset = 0.03 * max(bar_values)
    for i in range(len(bar_values)):
        plt.text(
            all_pos[i],
            bar_values[i] + val_offset,
            bar_values[i],
            ha="center",
            # color=colors[i],
            bbox=dict(facecolor=colors[i], alpha=0.8),
        )
    plt.ylim([0, max(bar_values) + 2 * val_offset])
    img_name = f"localisation_performance_" + "".join(metric) + ".png"
    plt.xlabel("SNR")
    plt.ylabel("m")
    plt.title("Verlinden process localisation performance")
    plt.xticks(positions, data["SNR"].unique())
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, img_name))
