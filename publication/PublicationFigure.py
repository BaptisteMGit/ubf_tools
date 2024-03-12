""" Define classes to handle figures properties for publication purpose """

import matplotlib as mpl
import matplotlib.pyplot as plt


class PubFigure:
    def __init__(
        self,
        size=(16, 8),
        label_fontsize=20,
        ticks_fontsize=20,
        title_fontsize=20,
        legend_fontsize=16,
        suplabel_fontsize=22,
        titlepad=20,
        labelpad=15,
        dpi=100,
    ):
        self.size = size
        self.label_fontsize = label_fontsize
        self.ticks_fontsize = ticks_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.suplabel_fontsize = suplabel_fontsize
        self.titlepad = titlepad
        self.labelpad = labelpad
        self.dpi = dpi
        self.set_all_fontsize()

    def apply_ticks_fontsize(self):
        plt.xticks(fontsize=self.ticks_fontsize)
        plt.yticks(fontsize=self.ticks_fontsize)

    def set_full_screen(self):
        mpl.rcParams["figure.max_open_warning"] = 0
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")

    def set_all_fontsize(self):
        params = {
            "legend.fontsize": self.legend_fontsize,
            "figure.figsize": self.size,
            "figure.titlesize": self.title_fontsize,
            "axes.labelsize": self.label_fontsize,
            "axes.titlesize": self.title_fontsize,
            "xtick.labelsize": self.ticks_fontsize,
            "ytick.labelsize": self.ticks_fontsize,
            "axes.titlepad": self.titlepad,
            "axes.labelpad": self.labelpad,
        }
        plt.rcParams.update(params)
