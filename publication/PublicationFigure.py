""" Define classes to handle figures properties for publication purpose """

import matplotlib as mpl
import matplotlib.pyplot as plt


class PubFigure:
    def __init__(
        self,
        size=(16, 8),
        label_fontsize=30,
        ticks_fontsize=30,
        title_fontsize=30,
        legend_fontsize=20,
        suplabel_fontsize=22,
        titlepad=20,
        labelpad=15,
        pad_inches=10,
        dpi=100,
        fmt="png",
        subplot_hspace=0.1,
        subplot_wspace=0.1,
        constrained_layout_h_pad=0.15,
        constrained_layout_w_pad=0.15,
        constrained_layout_hspace=0.1,
        constrained_layout_wspace=0.1,
    ):
        self.size = size
        self.label_fontsize = label_fontsize
        self.ticks_fontsize = ticks_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.suplabel_fontsize = suplabel_fontsize
        self.titlepad = titlepad
        self.labelpad = labelpad
        self.pad_inches = pad_inches

        # Constrained layout
        self.constrained_layout_h_pad = constrained_layout_h_pad
        self.constrained_layout_w_pad = constrained_layout_w_pad
        self.constrained_layout_hspace = constrained_layout_hspace
        self.constrained_layout_wspace = constrained_layout_wspace

        # Unconstrained subplot params
        self.subplot_hspace = subplot_hspace
        self.subplot_wspace = subplot_wspace

        self.dpi = dpi
        self.fmt = fmt
        self.set_all_params()

    def set_full_screen(self):
        mpl.rcParams["figure.max_open_warning"] = 0
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")

    def set_all_params(self):
        params = {
            "legend.fontsize": self.legend_fontsize,
            "figure.figsize": self.size,
            "figure.titlesize": self.title_fontsize,
            "figure.labelsize": self.label_fontsize,
            "figure.dpi": self.dpi,
            "figure.subplot.hspace": self.subplot_hspace,
            "figure.subplot.wspace": self.subplot_wspace,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.hspace": self.constrained_layout_hspace,
            "figure.constrained_layout.wspace": self.constrained_layout_wspace,
            "figure.constrained_layout.h_pad": self.constrained_layout_h_pad,
            "figure.constrained_layout.w_pad": self.constrained_layout_w_pad,
            "axes.labelsize": self.label_fontsize,
            "axes.titlesize": self.title_fontsize,
            "xtick.labelsize": self.ticks_fontsize,
            "ytick.labelsize": self.ticks_fontsize,
            "axes.titlepad": self.titlepad,
            "axes.labelpad": self.labelpad,
        }
        plt.rcParams.update(params)
