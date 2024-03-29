import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from publication.PublicationFigure import PubFigure

PFIG = PubFigure()


def plot_localisation_performance(
    data, testcase_name, similarity_metrics, metrics_to_plot, img_path
):
    # Sort data by SNR
    noisy_data = data[data["SNR"] != " None"].astype({"SNR": np.float32})
    noisy_data = noisy_data.sort_values(by=["SNR"])
    data = pd.concat([noisy_data, data[data["SNR"] == " None"]])

    plt.figure(figsize=PFIG.size)
    PFIG.set_all_fontsize()
    bar_width = 0.2

    # Define positions of the bars
    positions = np.arange(len(data["SNR"].unique()))
    m = len(similarity_metrics)
    alpha = np.arange(-(m // 2), (m // 2) + m % 2, 1) + ((m + 1) % 2) / 2

    # Create bar plot
    bar_values = np.array([])
    all_pos = np.array([])
    colors = []
    for i_m, metric in enumerate(metrics_to_plot):
        for i_dm, similarity_metric in enumerate(similarity_metrics):
            bar = data[data["Detection metric"] == similarity_metric][metric]
            if not bar.empty:
                b = plt.bar(
                    positions + alpha[i_dm] * bar_width,
                    bar.values,
                    width=bar_width,
                    label=similarity_metric,
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

    title = f"{testcase_name} - performance analysis"
    if metrics_to_plot[0] == "95_percentile":
        ylabel = "Position error 95 % percentile [m]"
    elif metrics_to_plot[0] == "99_percentile":
        ylabel = "Position error 99 % percentile [m]"
    elif metrics_to_plot[0] == "MEDIAN":
        ylabel = "Position median error [m]"
    elif metrics_to_plot[0] == "MEAN":
        ylabel = "Position mean error [m]"
    elif metrics_to_plot[0] == "STD":
        ylabel = "Position error standard deviation [m]"
    elif metrics_to_plot[0] == "RMSE":
        ylabel = "Position error RMSE [m]"
    elif metrics_to_plot[0] == "MIN":
        ylabel = "Position error min [m]"
    elif metrics_to_plot[0] == "MAX":
        ylabel = "Position error max [m]"
    elif metrics_to_plot[0] == "dynamic_range":
        ylabel = "Ambiguity surface dynamic range [dB]"
    else:
        pass

    plt.ylim([0, max(bar_values) + 5 * val_offset])
    img_name = f"localisation_performance_" + "".join(metric) + ".png"
    plt.xlabel("SNR")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions, data["SNR"].unique())
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, img_name))


if __name__ == "__main__":
    pass
