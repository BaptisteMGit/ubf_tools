import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    plt.ylim([0, max(bar_values) + 5 * val_offset])
    img_name = f"localisation_performance_" + "".join(metric) + ".png"
    plt.xlabel("SNR")
    plt.ylabel("m")
    plt.title("Verlinden process localisation performance")
    plt.xticks(positions, data["SNR"].unique())
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, img_name))
