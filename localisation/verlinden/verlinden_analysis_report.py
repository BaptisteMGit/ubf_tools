import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load global report
path_global_report = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\localisation\verlinden_process_analysis\verlinden_1_test_case\pulse\not_on_grid\global_report.txt"
data = pd.read_csv(path_global_report, sep=",")

data.loc[data["SNR"] == " None", "SNR"] = 20
data["SNR"] = data["SNR"].astype(float)

metrics_to_plot = data.columns[2:]
metrics_to_plot = ["RMSE", "STD"]

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


def plot_localisation_performance(data, metrics_to_plot, img_path):
    plt.figure(figsize=(10, 6))
    bar_width = 0.2

    # Positions des barres pour chaque SNR
    positions = np.arange(len(data["SNR"].unique()))

    # Créer un graphique à barres groupées pour chaque métrique
    for metric in metrics_to_plot:
        if not data[data["Detection metric"] == "intercorr0"][metric].empty:
            plt.bar(
                positions - bar_width,
                data[data["Detection metric"] == "intercorr0"][metric],
                width=bar_width,
                label=f"intercorr0 - {metric}",
            )

        if not data[data["Detection metric"] == "lstsquares"][metric].empty:
            plt.bar(
                positions,
                data[data["Detection metric"] == "lstsquares"][metric],
                width=bar_width,
                label=f"lstsquares - {metric}",
            )

        if not data[data["Detection metric"] == "hilbert_env_intercorr0"][metric].empty:
            plt.bar(
                positions + bar_width,
                data[data["Detection metric"] == "hilbert_env_intercorr0"][metric],
                width=bar_width,
                label=f"hilbert_env_intercorr0 - {metric}",
            )

    plt.xlabel("SNR")
    plt.ylabel("m")
    plt.title("Verlinden process localisation performance")
    plt.xticks(positions, sorted(data["SNR"].unique()))
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, f"global_performance.png"))
