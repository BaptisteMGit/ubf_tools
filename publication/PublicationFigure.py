""" Define classes to handle figures properties for publication purpose """

import matplotlib.pyplot as plt


class PubFigure:
    def __init__(
        self,
        size=(16, 8),
        label_fontsize=20,
        ticks_fontsize=20,
        title_fontsize=20,
        legend_fontsize=18,
        suplabel_fontsize=22,
        dpi=300,
    ):
        self.size = size
        self.label_fontsize = label_fontsize
        self.ticks_fontsize = ticks_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.suplabel_fontsize = suplabel_fontsize
        self.dpi = dpi

    def apply_ticks_fontsize(self):
        plt.xticks(fontsize=self.ticks_fontsize)
        plt.yticks(fontsize=self.ticks_fontsize)
