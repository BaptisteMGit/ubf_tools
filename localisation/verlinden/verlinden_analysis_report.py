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

    plt.figure()

    # Create bar plot
    # bar_width = 0.2

    # # Define positions of the bars
    # positions = np.arange(len(data["SNR"].unique()))
    # m = len(similarity_metrics)
    # alpha = np.arange(-(m // 2), (m // 2) + m % 2, 1) + ((m + 1) % 2) / 2

    # bar_values = np.array([])
    # all_pos = np.array([])
    # colors = []
    # for i_m, metric in enumerate(metrics_to_plot):
    #     for i_dm, similarity_metric in enumerate(similarity_metrics):
    #         bar = data[data["Detection metric"] == similarity_metric][metric]
    #         if not bar.empty:
    #             b = plt.bar(
    #                 positions + alpha[i_dm] * bar_width,
    #                 bar.values,
    #                 width=bar_width,
    #                 label=similarity_metric,
    #             )

    #             # Add value labels to bars
    #             # +3 / 4 * bar_width * np.sign(alpha[i_dm])
    #         pos = (
    #             positions
    #             + alpha[i_dm] * bar_width
    #             + 3 / 4 * bar_width * np.sign(alpha[i_dm])
    #         )
    #         all_pos = np.concatenate((all_pos, pos))
    #         bar_values = np.concatenate((bar_values, bar.values))
    #         colors += [b.patches[0].get_facecolor()] * len(bar)

    # val_offset = 0.03 * max(bar_values)
    # for i in range(len(bar_values)):
    #     plt.text(
    #         all_pos[i],
    #         bar_values[i] + val_offset,
    #         bar_values[i],
    #         ha="center",
    #         # color=colors[i],
    #         bbox=dict(facecolor=colors[i], alpha=0.8),
    #     )

    # Plot values
    for metric in metrics_to_plot:
        plt.figure()
        for similarity_metric in similarity_metrics:
            val = data[data["Detection metric"] == similarity_metric][metric]
            plt.plot(data["SNR"].unique(), val, label=similarity_metric, marker="o")

        # title = f"{testcase_name} - performance analysis"
        if metric == "95_percentile":
            title = f"95 % percentile position error ({testcase_name})"
            ylabel = r"$\epsilon_{95\%}$ [m]"
        elif metric == "99_percentile":
            title = f"99 % percentile position error ({testcase_name})"
            ylabel = r"$\epsilon_{99\%}$ [m]"
            # ylabel = "Position error 99 % percentile [m]"
        elif metric == "MEDIAN":
            title = f"Median position error ({testcase_name})"
            # ylabel = "Position median error [m]"
            ylabel = r"$\epsilon_{50\%}$ [m]"
        elif metric == "MEAN":
            title = f"Mean position error ({testcase_name})"
            # ylabel = "Position mean error [m]"
            ylabel = r"$\mu_{\epsilon}$ [m]"
        elif metric == "STD":
            title = f"Standard deviation position error ({testcase_name})"
            # ylabel = "Position error standard deviation [m]"
            ylabel = r"$\sigma_{\epsilon}$ [m]"
        elif metric == "RMSE":
            title = f"Root mean square error position error ({testcase_name})"
            # ylabel = "Position error RMSE [m]"
            ylabel = r"$\epsilon_{RMSE}$ [m]"
        elif metric == "MIN":
            title = f"Minimum position error ({testcase_name})"
            # ylabel = "Position error min [m]"
            ylabel = r"$\epsilon_{min}$ [m]"
        elif metric == "MAX":
            title = f"Maximum position error ({testcase_name})"
            # ylabel = "Position error max [m]"
            ylabel = r"$\epsilon_{max}$ [m]"
        elif metric == "dynamic_range":
            title = f"Dynamic range ({testcase_name})"
            ylabel = "DR [dB]"
        else:
            pass

        img_name = f"localisation_performance_" + "".join(metric) + ".png"
        plt.xlabel("SNR [dB]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(ncol=2)
        fpath = os.path.join(img_path, img_name)
        plt.savefig(fpath)
        plt.close()


if __name__ == "__main__":
    pass
