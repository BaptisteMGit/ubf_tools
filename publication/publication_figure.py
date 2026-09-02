#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   publication_figure.py
@Time    :   2025/04/07 16:15:41
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Classes to handle figure properties for publication-/
             report-ready output: consistent fonts, sizes, and
             LaTeX-rendered labels, plus a few axis-label presets and
             (new in this version) figure-type presets tailored to the
             KRAKEN toolbox's own plots (waveguide/environment overview,
             transmission-loss maps, mode shapes).

This module does NOT change the public API of the original file (same
class/function names and signatures; two bugs fixed, three new classes
added at the end -- see below).

BUGS FIXED COMPARED TO THE ORIGINAL FILE:
  1. PubFigure.set_better_axis(axis, fontsize=13): defined inside the
     class but WITHOUT 'self' as its first parameter -- its body never
     references 'self' either, so it was clearly meant to be a
     @staticmethod. Without that decorator, calling it the normal way
     (`my_pubfig.set_better_axis(ax)`) implicitly passes 'self' as the
     first positional argument, silently binding `axis=my_pubfig` and
     `fontsize=ax` instead. Confirmed: this raised
     `AttributeError: 'PubFigure' object has no attribute 'spines'`
     (trying to read `.spines` off the PubFigure instance, not the
     Axes). Calling it via the class instead of an instance
     (`PubFigure.set_better_axis(ax)`) happened to work by accident,
     which is presumably how this went unnoticed. Fixed by adding
     @staticmethod.
  2. AxisLabel.__init__: the 'language' setter always calls
     update_name(), which unconditionally overwrites 'self.name' with
     'self.name_fr' or 'self.name_en'. Since '__init__' sets
     'self.language = language' AFTER 'self.name = name', any caller
     passing an explicit 'name' was silently ignored (confirmed:
     `AxisLabel(name="Custom", name_fr="A", name_en="B",
     language="en").name` returned "B", not "Custom"). No existing
     subclass in this file (FrequencyLabel, RangeLabel, etc.) ever
     passes 'name' explicitly, so this did not surface as a visible bug
     in practice -- but it made the constructor's 'name' parameter
     silently useless for any future/direct AxisLabel(...) call. Fixed
     by remembering an explicitly-passed name and preferring it over
     the language-based default, while leaving every existing call
     site (which never passes 'name') behaving exactly as before.

ROBUSTNESS ADDED: PubFigure(use_tex=True) enables matplotlib's LaTeX
text rendering, which requires a complete LaTeX installation (in
particular, packages like 'cm-super' / 'texlive-fonts-extra' -- a
surprisingly common gap even on machines that do have a working 'latex'
executable: this was confirmed in this exact environment, where
'text.usetex=True' failed on the very first plot with a cryptic
'! LaTeX Error: File `type1ec.sty' not found.'). Left as-is, this turns
every single subsequent plot call into an opaque, hard-to-diagnose
crash. PubFigure now runs a one-time, cached smoke test when
use_tex=True is requested, and falls back to use_tex=False with a clear
warning (naming the likely missing packages) if LaTeX rendering isn't
actually usable, rather than leaving every plot call to fail later.

NEW: WaveguideFigure, TLFigure and ModeShapeFigure -- LargeFigure
presets sized to match the KRAKEN toolbox's own environment-overview
(KrakenEnv.plot_env, 3 panels), transmission-loss
(plot_utils.plotshd/plotshd_from_pressure_field, 1 wide panel) and
mode-shape (plot_utils.plotmode/plotmode_several_freqs, up to 10 narrow
panels) figures, respectively. Usage: instantiate the preset BEFORE
calling the corresponding plot_utils function -- PubFigure works by
updating matplotlib's global rcParams (fonts, sizes, LaTeX), so no
change to plot_utils.py itself is required for the two to work
together:

    from publication.publication_figure import WaveguideFigure
    from propa.kraken_toolbox.src.kraken_env import KrakenEnv

    WaveguideFigure()  # apply report-ready styling globally
    fig = env.plot_env(plot_src=True, src_depth=25)
    fig.savefig("environment.pdf")
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import string
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_subfigures_abc_labels(
    axs,
    x_pos=0.02,
    y_pos=0.98,
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top",
):
    """Annotate each subplot in 'axs' with '(a)', '(b)', '(c)', ... in
    its top-left corner (or wherever x_pos/y_pos, in axes-fraction
    coordinates, place it) -- the standard convention for multi-panel
    figures in a scientific publication.

    Args:
        axs: a single Axes, or any (nested) array/list of Axes (e.g.
            the return value of plt.subplots(...)).
        x_pos, y_pos (float): label position, in axes-fraction
            coordinates (0-1).
        fontsize (float): label font size.
        fontweight (str): label font weight.
        ha, va (str): horizontal/vertical text alignment.
    """
    labels = list(string.ascii_lowercase)

    # Ensure axs is a 1D array of axes
    axs = np.atleast_1d(axs).flatten()

    # Iterate over subplots and add labels
    for i in range(axs.size):
        # Subfigure annotation
        axs[i].text(
            x_pos,
            y_pos,
            f"({labels[i]})",
            transform=axs[i].transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            va=va,
            clip_on=False,
        )


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


# ======================================================================================================================
# LaTeX availability smoke test (see module docstring: "ROBUSTNESS ADDED")
# ======================================================================================================================
_latex_usable_cache = None


def _latex_usable():
    """Return True if matplotlib can actually render text with
    'text.usetex=True' in this environment, False otherwise. The result
    is computed once (a real LaTeX invocation, so not free) and cached
    for the rest of the process.
    """
    global _latex_usable_cache
    if _latex_usable_cache is not None:
        return _latex_usable_cache

    previous = mpl.rcParams["text.usetex"]
    try:
        mpl.rcParams["text.usetex"] = True
        fig = plt.figure()
        try:
            fig.text(0.5, 0.5, r"$x^2$")
            fig.canvas.draw()
            _latex_usable_cache = True
        except Exception:
            _latex_usable_cache = False
        finally:
            plt.close(fig)
    except Exception:
        _latex_usable_cache = False
    finally:
        mpl.rcParams["text.usetex"] = previous

    return _latex_usable_cache


class PubFigure:
    """Apply a consistent, report-ready matplotlib style (fonts, sizes,
    optionally LaTeX-rendered text) globally, via matplotlib's rcParams.

    This works by updating matplotlib's GLOBAL configuration (not a
    per-figure setting): instantiate a PubFigure (or one of its
    subclasses/presets) once, before creating any figure, and every
    subsequent plot -- whether made directly with matplotlib/pyplot or
    through a helper such as propa.kraken_toolbox.plot_utils's
    functions -- will use this styling, with no further code changes
    needed on the plotting side.
    """

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
        """Maximize the current figure window. Only works with GUI
        backends that expose a Tk-style window manager (e.g. TkAgg);
        no-ops (with a warning) on any other backend, including the
        headless 'Agg' backend used in batch/report-generation
        pipelines and CI."""
        mpl.rcParams["figure.max_open_warning"] = 0
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state("zoomed")
        except AttributeError:
            warnings.warn(
                "set_full_screen() has no effect with the current matplotlib "
                "backend (no Tk-style window manager available -- this is "
                "expected for headless/'Agg'-backend or non-Tk GUI use)."
            )

    def set_all_params(self):
        # NOTE (robustness added): 'text.usetex=True' requires a
        # complete LaTeX installation (see module docstring). Run a
        # cheap, cached smoke test first and fall back to
        # use_tex=False with a clear warning rather than let every
        # subsequent plot fail with a cryptic LaTeX subprocess error.
        if self.use_tex and not _latex_usable():
            warnings.warn(
                "PubFigure(use_tex=True) was requested, but matplotlib could "
                "not actually render text with LaTeX in this environment "
                "(often caused by a missing 'cm-super' / "
                "'texlive-fonts-extra' package, even when a 'latex' "
                "executable is present). Falling back to use_tex=False for "
                "this and any other PubFigure created in this process."
            )
            self.use_tex = False

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
        if self.use_tex:
            params["text.latex.preamble"] = r"\usepackage{amsmath}"

        plt.rcParams.update(params)

    @staticmethod
    def set_better_axis(axis, fontsize=13):
        """Remove the top/right spines, add an arrow tip on the
        left/bottom spines, and set the tick label font size -- a
        common "cleaner axis" presentation style.

        NOTE (bug fixed): this used to be defined without '@staticmethod'
        even though its body never uses 'self', which broke the normal
        `my_pubfig.set_better_axis(ax)` call form (see module
        docstring). Now a proper static method, callable both as
        `PubFigure.set_better_axis(ax)` and `my_pubfig.set_better_axis(ax)`.

        Args:
            - axis (Axes): matplotlib axis
            - fontsize (float, optional): label fontsize. Defaults to 13.
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

    Large figures are defined as page wide figures (to be used with width=\\textwith in LaTeX).

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

    Small figures are defined as page narrow figures (to be used with width=\\textwidth in LaTeX).

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


# ======================================================================================================================
# NEW: figure-type presets tailored to the KRAKEN toolbox's own plots
# (propa.kraken_toolbox.plot_utils) -- see module docstring for usage.
# ======================================================================================================================
class WaveguideFigure(LargeFigure):
    """Preset for KrakenEnv.plot_env() / KrakenMedium.plot_medium() /
    KrakenBottomHalfspace.plot_bottom_halfspace(): a wide, 3-panel
    (sound speed, attenuation, density vs. depth) environment overview.
    Sized to match those functions' own figsize=(15, 8) so applying this
    preset does not change the figure's proportions, only its fonts/
    LaTeX rendering.
    """

    def __init__(self, size=(15, 8), title_fontsize=18, label_fontsize=16, ticks_fontsize=13, **kwargs):
        super().__init__(
            size=size,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            ticks_fontsize=ticks_fontsize,
            **kwargs,
        )


class TLFigure(LargeFigure):
    """Preset for plot_utils.plotshd() / plotshd_from_pressure_field():
    a wide, single-panel transmission-loss (range vs. depth) map with a
    colorbar. Sized to match those functions' own figsize=(16, 8).
    """

    def __init__(self, size=(16, 8), title_fontsize=18, label_fontsize=16, ticks_fontsize=13, **kwargs):
        super().__init__(
            size=size,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            ticks_fontsize=ticks_fontsize,
            **kwargs,
        )


class ModeShapeFigure(LargeFigure):
    """Preset for plot_utils.plotmode() / plotmode_several_freqs(): a
    wide row of up to 10 narrow mode-shape panels sharing a depth axis.
    Sized to match those functions' own figsize=(15, 5); tick labels
    default a bit smaller than WaveguideFigure/TLFigure since up to 10
    panels share the same width.
    """

    def __init__(self, size=(15, 5), title_fontsize=18, label_fontsize=14, ticks_fontsize=11, **kwargs):
        super().__init__(
            size=size,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            ticks_fontsize=ticks_fontsize,
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
        # NOTE (bug fixed): 'self.language = language' below (via its
        # setter -> update_name()) used to UNCONDITIONALLY overwrite
        # 'self.name' with 'name_fr'/'name_en', silently discarding
        # whatever 'name' was passed in explicitly (confirmed:
        # AxisLabel(name="Custom", name_fr="A", name_en="B",
        # language="en").name returned "B", not "Custom"). No call site
        # in this file ever passes 'name' explicitly (they only set
        # name_fr/name_en), so this never surfaced as a visible bug in
        # existing code -- but it silently broke the constructor's
        # documented 'name' parameter for any direct AxisLabel(...)
        # call that did use it. '_explicit_name' remembers such an
        # override so update_name() can honour it; every existing call
        # site (which leaves 'name' at its default "name"/unset) is
        # unaffected.
        self._explicit_name = name if name not in (None, "name") else None

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
        """Update name according to the selected language, unless an
        explicit 'name' was passed to the constructor (see the
        '_explicit_name' note in __init__)."""
        if self._explicit_name is not None:
            self.name = self._explicit_name
        elif self.language == "fr":
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
