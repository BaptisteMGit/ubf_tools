#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for publication/publication_figure.py.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests

NOTE: these tests force use_tex=False (or rely on the automatic
fallback) throughout, so they run identically whether or not the host
machine has a complete LaTeX installation -- see
publication_figure._latex_usable()'s docstring for why that check
exists at all.
"""
import unittest
import warnings
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication import publication_figure as pf


def _close_all_figures():
    plt.close("all")


# ======================================================================
# PubFigure.set_better_axis -- regression test for the fixed bug
# ======================================================================
class TestSetBetterAxis(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_works_as_instance_method(self):
        # NOTE: regression test for the fixed bug -- set_better_axis was
        # defined without @staticmethod despite never using 'self',
        # so calling it the normal way (`pubfig.set_better_axis(ax)`)
        # bound `axis=pubfig` and `fontsize=ax`, raising
        # `AttributeError: 'PubFigure' object has no attribute 'spines'`.
        pubfig = pf.PubFigure(use_tex=False)
        fig, ax = plt.subplots()
        pubfig.set_better_axis(ax)  # must not raise
        self.assertFalse(ax.spines["top"].get_visible())
        self.assertFalse(ax.spines["right"].get_visible())

    def test_works_as_class_method_too(self):
        fig, ax = plt.subplots()
        pf.PubFigure.set_better_axis(ax)  # already worked before the fix
        self.assertFalse(ax.spines["top"].get_visible())

    def test_fontsize_is_applied(self):
        fig, ax = plt.subplots()
        pf.PubFigure.set_better_axis(ax, fontsize=7)
        for label in ax.get_xticklabels():
            self.assertEqual(label.get_fontsize(), 7)


# ======================================================================
# AxisLabel -- regression test for the fixed 'name' bug
# ======================================================================
class TestAxisLabel(unittest.TestCase):
    def test_explicit_name_is_respected(self):
        # NOTE: regression test for the fixed bug -- an explicitly
        # passed 'name' used to be silently discarded, overwritten by
        # 'name_fr'/'name_en' via the language setter.
        lbl = pf.AxisLabel(
            name="Custom Name", name_fr="Nom FR", name_en="Name EN", language="en"
        )
        self.assertEqual(lbl.name, "Custom Name")
        self.assertIn("Custom Name", lbl.label)

    def test_default_behaviour_unaffected_fr(self):
        lbl = pf.AxisLabel(name_fr="Nom FR", name_en="Name EN", language="fr")
        self.assertEqual(lbl.name, "Nom FR")

    def test_default_behaviour_unaffected_en(self):
        lbl = pf.AxisLabel(name_fr="Nom FR", name_en="Name EN", language="en")
        self.assertEqual(lbl.name, "Name EN")

    def test_switching_language_updates_name_when_no_explicit_name(self):
        lbl = pf.AxisLabel(name_fr="Nom FR", name_en="Name EN", language="en")
        self.assertEqual(lbl.name, "Name EN")
        lbl.language = "fr"
        self.assertEqual(lbl.name, "Nom FR")

    def test_switching_language_keeps_explicit_name(self):
        lbl = pf.AxisLabel(
            name="Always This", name_fr="Nom FR", name_en="Name EN", language="en"
        )
        lbl.language = "fr"
        self.assertEqual(lbl.name, "Always This")

    def test_unknown_language_raises(self):
        with self.assertRaises(ValueError):
            pf.AxisLabel(language="de")

    def test_label_format_with_unit(self):
        lbl = pf.AxisLabel(name="Range", unit="km", language="en", name_en="Range")
        self.assertEqual(lbl.label, "Range [km]")

    def test_label_format_without_unit(self):
        lbl = pf.AxisLabel(name="Range", unit="", language="en", name_en="Range")
        self.assertEqual(lbl.label, "Range")

    def test_set_axis_label_x(self):
        fig, ax = plt.subplots()
        plt.sca(ax)
        lbl = pf.RangeLabel(language="en", axis="x")
        lbl.set_axis_label()
        self.assertEqual(ax.get_xlabel(), lbl.label)
        _close_all_figures()

    def test_set_axis_label_y(self):
        fig, ax = plt.subplots()
        plt.sca(ax)
        lbl = pf.RangeLabel(language="en", axis="y")
        lbl.set_axis_label()
        self.assertEqual(ax.get_ylabel(), lbl.label)
        _close_all_figures()

    def test_set_axis_label_invalid_axis_raises(self):
        fig, ax = plt.subplots()
        plt.sca(ax)
        lbl = pf.RangeLabel(language="en", axis="z")
        with self.assertRaises(ValueError):
            lbl.set_axis_label()
        _close_all_figures()


# ======================================================================
# Preset axis-label subclasses: sanity checks
# ======================================================================
class TestAxisLabelPresets(unittest.TestCase):
    def test_frequency_label_default_french(self):
        lbl = pf.FrequencyLabel()
        self.assertEqual(lbl.label, "Fréquence [Hz]")

    def test_range_label_km(self):
        lbl = pf.RangeLabel(language="en", unit="km")
        self.assertEqual(lbl.label, "Range [km]")

    def test_propagation_loss_label(self):
        lbl = pf.PropagationLossLabel(language="en")
        self.assertIn("Propagation loss", lbl.label)


# ======================================================================
# LaTeX availability smoke test / fallback
# ======================================================================
class TestLatexFallback(unittest.TestCase):
    def setUp(self):
        # Force a clean slate: the real check is cached at module level.
        pf._latex_usable_cache = None

    def tearDown(self):
        pf._latex_usable_cache = None
        _close_all_figures()

    def test_fallback_when_latex_unusable(self):
        with mock.patch.object(pf, "_latex_usable", return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pubfig = pf.PubFigure(use_tex=True)
        self.assertFalse(pubfig.use_tex)
        self.assertTrue(any("use_tex" in str(w.message) for w in caught))

    def test_no_fallback_when_latex_usable(self):
        with mock.patch.object(pf, "_latex_usable", return_value=True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pubfig = pf.PubFigure(use_tex=True)
        self.assertTrue(pubfig.use_tex)
        self.assertFalse(any("use_tex" in str(w.message) for w in caught))

    def test_use_tex_false_never_checks_latex(self):
        with mock.patch.object(pf, "_latex_usable") as spy:
            pf.PubFigure(use_tex=False)
        spy.assert_not_called()

    def test_result_is_cached(self):
        calls = []

        def fake_check():
            calls.append(1)
            return False

        with mock.patch("matplotlib.pyplot.figure", side_effect=RuntimeError("boom")):
            pf._latex_usable()  # first call: hits the (failing) real check once
        first_cache_value = pf._latex_usable_cache
        self.assertFalse(first_cache_value)
        # Second call must NOT re-invoke the expensive check.
        with mock.patch("matplotlib.pyplot.figure", side_effect=AssertionError("should not be called")):
            result = pf._latex_usable()
        self.assertEqual(result, first_cache_value)


# ======================================================================
# Figure-size presets (new classes)
# ======================================================================
class TestFigurePresets(unittest.TestCase):
    def tearDown(self):
        pf._latex_usable_cache = None
        _close_all_figures()

    def test_waveguide_figure_size_matches_plot_env(self):
        pf.WaveguideFigure(use_tex=False)
        fig = plt.figure()
        self.assertEqual(tuple(fig.get_size_inches()), (15.0, 8.0))

    def test_tl_figure_size_matches_plotshd(self):
        pf.TLFigure(use_tex=False)
        fig = plt.figure()
        self.assertEqual(tuple(fig.get_size_inches()), (16.0, 8.0))

    def test_mode_shape_figure_size_matches_plotmode(self):
        pf.ModeShapeFigure(use_tex=False)
        fig = plt.figure()
        self.assertEqual(tuple(fig.get_size_inches()), (15.0, 5.0))

    def test_presets_integrate_with_plot_utils(self):
        # End-to-end: applying a preset, then calling a real
        # plot_utils function, must not raise and must produce a
        # figure of the preset's size.
        import os
        from propa.kraken_toolbox import plot_utils as pu

        fixtures_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures"
        )
        shd_path = os.path.join(fixtures_dir, "real_field.shd")
        if not os.path.exists(shd_path):
            self.skipTest("real_field.shd fixture not present")

        pf.TLFigure(use_tex=False)
        fig = pu.plotshd(shd_path, freq=30, units="km")
        self.assertEqual(tuple(fig.get_size_inches()), (16.0, 8.0))


# ======================================================================
# set_subfigures_abc_labels / color
# ======================================================================
class TestMiscHelpers(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_set_subfigures_abc_labels(self):
        fig, axs = plt.subplots(1, 3)
        pf.set_subfigures_abc_labels(axs)
        texts = [t.get_text() for ax in axs for t in ax.texts]
        self.assertEqual(texts, ["(a)", "(b)", "(c)"])

    def test_set_subfigures_abc_labels_single_axis(self):
        fig, ax = plt.subplots()
        pf.set_subfigures_abc_labels(ax)
        self.assertEqual(ax.texts[0].get_text(), "(a)")

    def test_color_cycles(self):
        c0 = pf.color(0)
        c_wrap = pf.color(14)  # wraps around (14 colors defined, 0-13)
        self.assertEqual(c0, c_wrap)

    def test_color_returns_rgb_tuple(self):
        c = pf.color(1)
        self.assertEqual(len(c), 3)


if __name__ == "__main__":
    unittest.main()
