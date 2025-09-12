#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   publication_figure.py
@Time    :   2025/04/07 16:15:41
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle figures properties.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt


def color(n: int):
    """Returns a tuple of RGB values used as a figure color.

    Args:
      - n (int): color index

    Returns:
      - tuple: RGB tuple
    """
    figcolors = [
        (0, 0.4470, 0.7410),  # 0, Royal blue
        (0.8500, 0.3250, 0.0980),  # 1, Orange
        (0.9290, 0.6940, 0.1250),  # 2, Fade yellow
        (0.4940, 0.1840, 0.5560),  # 3, Purple
        (0.4660, 0.6740, 0.1880),  # 4, Green
        (0.3010, 0.7450, 0.9330),  # 5, Sky blue
        (0.6350, 0.0780, 0.1840),  # 6, Bordeaux
        (1, 0, 0),  # 7, Red
        (0, 1, 0),  # 8, Lime green
        (0, 0, 1),  # 9, Blue
        (0, 1, 1),  # 10, Cyan
        (1, 0, 1),  # 11, Magenta
        (1, 1, 0),  # 12, Yellow
        (0, 0, 0),
    ]  # 13, Black

    k = n % len(figcolors)
    return figcolors[k][:]


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
        use_tex=True,
        disable_backend=False,
        language="en",
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
        self.use_tex = use_tex
        self.disable_backend = disable_backend
        self.set_all_params()

        # Define useful axis labels that we usually need
        self.language = language
        self.f_label = FrequencyLabel(language=language)
        self.r_label = RangeLabel(language=language)
        self.rkm_label = RangeLabel(language=language, unit="km")
        self.pl_label = PropagationLossLabel(language=language)
        self.tfmod_label = TransfertFunctionModuleLabel(language=language)
        self.tfphase_label = TransfertFunctionPhaseLabel(language=language)
        self.rtfmod_label = RTFModuleLabel(language=language)

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
            "text.usetex": self.use_tex,
            "font.family": "serif",
        }
        if self.disable_backend:
            params["backend"] = "Agg"

        plt.rcParams.update(params)

    def set_better_axis(axis, fontsize=13):
        """Remove top and right border of axis, add arrow on left and bottom border and set left and bottom label fontsize

        Args:
            - axis (Axes): matplotlib axis
            - fontsize (float, optional): label fontsize . Defaults to 13.
        """
        axis.spines["left"].set_position(("data", 0))
        axis.spines["bottom"].set_position(("data", 0))
        axis.plot(1, 0, ">k", transform=axis.get_yaxis_transform(), clip_on=False)
        axis.plot(0, 1, "^k", transform=axis.get_xaxis_transform(), clip_on=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params("x", labelsize=fontsize)
        axis.tick_params("y", labelsize=fontsize)


class LargeFigure(PubFigure):
    """
    Class to handle large figures properties.

    Large figures are defined as page wide figures (to be used with width=\textwith in LaTeX).

    The aim is to ensure to get the same font sizes as the Latex document.

    """

    def __init__(
        self,
        size=(16, 5.25),
        label_fontsize=30,
        legend_fontsize=30,
        suplabel_fontsize=30,
        title_fontsize=30,
        ticks_fontsize=30,
        **kwargs,
    ):
        super().__init__(
            size=size,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            suplabel_fontsize=suplabel_fontsize,
            title_fontsize=title_fontsize,
            ticks_fontsize=ticks_fontsize,
            **kwargs,
        )


class SmallFigure(PubFigure):
    """
    Class to handle small figures properties.

    Small figures are defined as page narrow figures (to be used with width=\textwidth in LaTeX).

    The aim is to ensure to get the same font sizes as the Latex document.

    """

    def __init__(self, **kwargs):
        super().__init__(
            size=(7, 5.25),
            label_fontsize=30,
            legend_fontsize=30,
            suplabel_fontsize=30,
            title_fontsize=30,
            ticks_fontsize=30,
            **kwargs,
        )


class AxisLabel:

    def __init__(
        self,
        name: str = "name",
        unit: str = "unit",
        fmt: str = "{} [{}]",
        axis: str = "x",
        name_fr: str = "nom par défaut",
        name_en: str = "default name",
        language: str = "en",
    ):
        self.name = name
        self.unit = unit
        self.fmt = fmt
        self.axis = axis

        self.name_fr = name_fr
        self.name_en = name_en
        self.language = language

        self.label = ""

    @property
    def label(self):
        self._label = self.fmt.format(self.name, self.unit)
        return self._label

    @label.setter
    def label(self, value):
        """
        Set label string
        :param value: label string
        """
        self._label = value

    @property
    def fmt(self):
        if len(self.unit) > 0:
            return self._fmt
        else:
            self.fmt = "{}{}"
        return self._fmt

    @fmt.setter
    def fmt(self, value):
        """
        Set fmt string
        :param value: fmt string
        """
        self._fmt = value

    def set_axis_label(self, axis=None):
        # Get current axis
        ax = plt.gca()
        # Set axis label
        if axis is None:
            axis = self.axis
        if axis == "x":
            ax.set_xlabel(self.label)
        elif axis == "y":
            ax.set_ylabel(self.label)
        else:
            raise ValueError(f"Unknown axis {self.axis}.")

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        """
        Set language string and update name accordingly
        :param value: language code ("fr" or "en")
        """
        # Set code
        self._language = value
        self.update_name()

    def update_name(self):
        """Update name according to the selected language"""
        if self.language == "fr":
            self.name = self.name_fr
        elif self.language == "en":
            self.name = self.name_en
        else:
            raise ValueError(f"Unknown language {self.language}.")


class FrequencyLabel(AxisLabel):

    def __init__(
        self,
        unit: str = "Hz",
        name_fr: str = "Fréquence",
        name_en: str = "Frequency",
        axis: str = "x",
        language: str = "fr",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


class PropagationLossLabel(AxisLabel):

    def __init__(
        self,
        unit=r"dB re 1 $\mu$Pa",
        name_fr="Perte par propagation",
        name_en="Propagation loss",
        axis: str = "x",
        language: str = "en",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


class RangeLabel(AxisLabel):

    def __init__(
        self,
        unit: str = "m",
        name_fr: str = "Distance horizontale",
        name_en: str = "Range",
        axis: str = "x",
        language: str = "en",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


class TransfertFunctionModuleLabel(AxisLabel):

    def __init__(
        self,
        unit: str = "",
        name_fr: str = r"$|H(f)|$",
        name_en: str = r"$|H(f)|$",
        axis: str = "x",
        language: str = "en",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


class TransfertFunctionPhaseLabel(AxisLabel):

    def __init__(
        self,
        unit: str = "",
        name_fr: str = r"$\phi_H(f)$",
        name_en: str = r"$\phi_H(f)$",
        axis: str = "x",
        language: str = "en",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


class RTFModuleLabel(AxisLabel):

    def __init__(
        self,
        unit: str = "",
        name_fr: str = r"$|\Pi(f)|$",
        name_en: str = r"$|\Pi(f)|$",
        axis: str = "x",
        language: str = "en",
    ):
        super().__init__(
            unit=unit,
            name_fr=name_fr,
            name_en=name_en,
            axis=axis,
            language=language,
        )


if __name__ == "__main__":
    # xlab = AxisLabel(name="Fréquence", unit="Hz", axis="x")
    # print(xlab.label)
    # plt.figure()
    # xlab.set_axis_label()

    flab = FrequencyLabel(language="en")
    pl_label = PropagationLossLabel(language="en")
    rlab = RangeLabel(language="en", axis="x")
    rlab_km = RangeLabel(language="en", axis="y", unit="km")
    print(flab.label)
    plt.figure()
    flab.set_axis_label()
    pl_label.set_axis_label("y")

    plt.figure()
    rlab.set_axis_label()
    rlab_km.set_axis_label()
    plt.show()
