#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/plot_utils.py.

These tests focus on the bugs found and fixed (see plot_utils.py's
module docstring), using the real KRAKEN/FIELD fixture files already in
propa/kraken_toolbox/tests/fixtures/ (see test_real_kraken_files.py)
wherever real mode-count variation across frequencies matters -- no
synthetic construction can substitute for that.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from propa.kraken_toolbox import plot_utils as pu

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MOD_PATH = os.path.join(FIXTURES_DIR, "real_kraken.mod")
SHD_PATH = os.path.join(FIXTURES_DIR, "real_field.shd")


def _close_all_figures():
    plt.close("all")


# ======================================================================
# plotmode (unified single/multi-frequency mode-shape grid)
# ======================================================================
@unittest.skipUnless(os.path.exists(MOD_PATH), "real_kraken.mod fixture not present")
class TestPlotmode(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    # ------------------------------------------------------------------
    # Basic behaviour, single frequency
    # ------------------------------------------------------------------
    def test_single_mode_does_not_crash(self):
        # 10 Hz has exactly 1 mode in this file -> a 1x1 grid.
        fig = pu.plotmode(MOD_PATH, freq=10, n_modes=6)
        self.assertIsNotNone(fig)

    def test_multiple_modes_does_not_crash(self):
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=6)  # 7 modes available at 50 Hz
        self.assertIsNotNone(fig)

    def test_title_is_a_string_not_a_list(self):
        fig = pu.plotmode(MOD_PATH, freq=10)
        title_obj = fig._suptitle
        self.assertIsInstance(title_obj.get_text(), str)
        self.assertNotIn("[", title_obj.get_text())

    def test_no_modes_raises(self):
        with mock.patch.object(pu, "readmodes", return_value={"M": 0}):
            with self.assertRaises(Exception):
                pu.plotmode(MOD_PATH, freq=10)

    # ------------------------------------------------------------------
    # Per-user-request behaviour: automatic grid, titles, single legend,
    # bathymetry line on every panel.
    # ------------------------------------------------------------------
    def test_n_modes_caps_the_number_of_panels(self):
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=3)  # 7 available, only 3 requested
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(used_axes), 3)

    def test_n_modes_is_capped_by_availability(self):
        # Only 1 mode exists at 10 Hz: requesting 6 must not error, and
        # must not create phantom panels beyond what's available.
        fig = pu.plotmode(MOD_PATH, freq=10, n_modes=6)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(used_axes), 1)

    def test_grid_is_chosen_to_keep_the_overall_figure_landscape(self):
        # NOTE: per user request -- the grid selection criterion now
        # optimizes for an overall LANDSCAPE figure (wider than tall),
        # given that each individual panel is itself taller than wide.
        # A naive "roughly square in cell count" grid (e.g. 3x3 for 7
        # panels) would actually produce a PORTRAIT figure once each
        # cell's own portrait shape is accounted for; 2 rows x 4
        # columns is the layout that comes closest to landscape here.
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=7)
        self.assertEqual(len(fig.axes), 8)
        used = [ax for ax in fig.axes if ax.get_title()]
        unused = [ax for ax in fig.axes if not ax.get_title()]
        self.assertEqual(len(used), 7)
        self.assertEqual(len(unused), 1)
        for ax in unused:
            self.assertFalse(ax.axison)
        width, height = fig.get_size_inches()
        self.assertGreater(width, height)

    def test_explicit_ncols_override(self):
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=6, ncols=2)
        # 6 panels, 2 columns -> 3 rows -> 6 axes total, all used.
        self.assertEqual(len(fig.axes), 6)

    def test_mode_number_is_the_subplot_title_not_an_xlabel(self):
        # NOTE: per user request -- "Mode N" must be the subplot TITLE,
        # not an x-axis label.
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=3)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        titles = sorted(ax.get_title() for ax in used_axes)
        self.assertEqual(titles, ["Mode 1", "Mode 2", "Mode 3"])
        for ax in used_axes:
            self.assertNotIn("Mode", ax.get_xlabel())

    def test_explicit_modes_parameter(self):
        fig = pu.plotmode(MOD_PATH, freq=50, modes=[1, 3, 7])
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        titles = sorted(ax.get_title() for ax in used_axes)
        self.assertEqual(titles, ["Mode 1", "Mode 3", "Mode 7"])

    def test_single_legend_for_the_whole_figure(self):
        # NOTE: per user request -- one legend for the whole figure, not
        # one per subplot.
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=6)
        legends_found = sum(1 for ax in fig.axes if ax.get_legend() is not None)
        self.assertEqual(legends_found, 0)  # none of the per-axes legends are used
        self.assertIsNotNone(fig.legends)
        self.assertEqual(len(fig.legends), 1)

    def test_bathymetry_line_is_on_every_panel(self):
        # NOTE: regression test for the fixed bug -- in the old
        # 'plotmode_several_freqs', the seafloor line was only added
        # while processing the FIRST frequency, inside a loop bounded by
        # THAT frequency's own (possibly smaller) mode count, so panels
        # that only existed thanks to a later, richer frequency never
        # got the line at all.
        freqs = np.array([10, 20, 30, 40, 50])  # 1, 3, 4, 5, 7 modes respectively
        fig = pu.plotmode(MOD_PATH, freq=freqs, n_modes=7, bathy_depth=100.0)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(used_axes), 7)
        for ax in used_axes:
            hlines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--" and ln.get_color() == "r"]
            self.assertGreaterEqual(
                len(hlines), 1,
                msg=f"seafloor line missing on '{ax.get_title()}'",
            )

    def test_legend_lists_every_frequency_once(self):
        freqs = np.array([10, 20, 30])
        fig = pu.plotmode(MOD_PATH, freq=freqs, n_modes=3)
        legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
        for f in freqs:
            self.assertIn(f"{f:g} Hz", legend_labels)
        self.assertIn("Real part", legend_labels)
        self.assertIn("Imag part", legend_labels)

    def test_single_frequency_legend_has_no_frequency_entries(self):
        # With a single frequency there is nothing to disambiguate by
        # color, so only the Real/Imag entries are needed.
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=3)
        legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertEqual(set(legend_labels), {"Real part", "Imag part"})

    def test_legend_includes_seafloor_entry_when_bathy_given(self):
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=3, bathy_depth=100.0)
        legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertIn("Seafloor", legend_labels)

    def test_normalize_mode_scales_to_unit_range(self):
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=2, normalize_mode=True)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        for ax in used_axes:
            xmin, xmax = ax.get_xlim()
            self.assertLessEqual(xmax, 1.21)
            self.assertGreaterEqual(xmin, -1.21)

    # ------------------------------------------------------------------
    # Per-user-request behaviour, second round: correct mode shapes (no
    # more utils.get_component), taller panels, x-axis centered on 0.
    # ------------------------------------------------------------------
    def test_does_not_use_get_component(self):
        # NOTE: regression test for the fixed bug -- plotmode() used to
        # extract mode shapes via utils.get_component(Modes, "N"),
        # which was confirmed (on this exact real fixture) to silently
        # zero-pad most of the mode shape: Modes["N"] (mesh subdivision
        # count from the '.env' file, [25] here) does not match the
        # actual output grid size (2601 points here), so get_component
        # only filled the first 25 rows. plotmode() must not import or
        # call it at all anymore -- it reads Modes["phi"] directly.
        self.assertFalse(hasattr(pu, "get_component"))

    def test_mode_shapes_match_direct_phi_indexing(self):
        # The definitive correctness check: values plotted for a given
        # mode must be EXACTLY Modes["phi"][:, local_idx].real/.imag,
        # depth for depth -- matching the user's own, confirmed-correct
        # reference implementation (which never used get_component).
        from propa.kraken_toolbox.read_modes import readmodes

        Modes = readmodes(MOD_PATH, freq=50)
        expected_real = Modes["phi"][:, 0].real  # mode 1

        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=1)
        ax = fig.axes[0]
        real_line = next(ln for ln in ax.get_lines() if ln.get_linestyle() == "-")
        np.testing.assert_allclose(real_line.get_xdata(), expected_real)
        np.testing.assert_allclose(real_line.get_ydata(), Modes["z"])

    def test_elastic_mode_file_raises_clear_error(self):
        # plotmode() only supports ACOUSTIC-only mode files (see
        # docstring): Modes["phi"] must have exactly one row per depth.
        # Simulate an ELASTIC file (4 phi rows per depth) via a mock.
        fake_modes = {
            "M": 2,
            "z": np.array([0.0, 1.0, 2.0]),
            "phi": np.zeros((12, 2), dtype=complex),  # 4 rows x 3 depths != 3
            "selected_modes": np.array([1, 2]),
            "nb_selected_modes": 2,
            "freqVec": np.array([50.0]),
            "title": "fake",
        }
        with mock.patch.object(pu, "readmodes", return_value=fake_modes):
            with self.assertRaises(ValueError):
                pu.plotmode(MOD_PATH, freq=50, n_modes=2)

    def test_panels_are_taller_than_wide(self):
        # NOTE: per user request -- each individual panel should be a
        # "portrait" (taller-than-wide) rectangle. With only 1 panel,
        # the overall figure necessarily has the same shape as that one
        # panel, so it stays portrait here -- this is the one
        # unavoidable exception to "the overall figure is landscape"
        # (see test_grid_is_chosen_to_keep_the_overall_figure_landscape
        # for the >1-panel case).
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=1)
        width, height = fig.get_size_inches()
        self.assertGreater(height, width)

    def test_legend_is_within_the_saved_figure_bounds(self):
        # NOTE: regression test for the fixed bug -- the legend used to
        # be placed at bbox_to_anchor=(1.0, 0.5), OUTSIDE the figure's
        # own canvas. Saving without bbox_inches="tight" (as every
        # example script in this project does) then silently clipped
        # the legend out of the file entirely. Checks that the legend's
        # bounding box, in figure-fraction coordinates, is fully
        # contained within [0, 1] x [0, 1] -- i.e. within what a plain
        # `savefig(path)` (no bbox_inches="tight") actually saves.
        import io

        fig = pu.plotmode(MOD_PATH, freq=[10, 20], n_modes=3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # deliberately no bbox_inches="tight"
        fig.canvas.draw()
        legend = fig.legends[0]
        bbox = legend.get_window_extent()
        fig_bbox = fig.bbox
        self.assertGreaterEqual(bbox.x0, fig_bbox.x0 - 1)  # small tolerance
        self.assertLessEqual(bbox.x1, fig_bbox.x1 + 1)
        self.assertGreaterEqual(bbox.y0, fig_bbox.y0 - 1)
        self.assertLessEqual(bbox.y1, fig_bbox.y1 + 1)

    def test_xaxis_is_centered_on_zero(self):
        # NOTE: per user request -- each panel's x-axis must be
        # symmetric around 0, not matplotlib's default autoscale.
        fig = pu.plotmode(MOD_PATH, freq=50, n_modes=3)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        for ax in used_axes:
            xmin, xmax = ax.get_xlim()
            self.assertAlmostEqual(xmin, -xmax, places=6)


@unittest.skipUnless(os.path.exists(MOD_PATH), "real_kraken.mod fixture not present")
class TestPlotmodeFromData(unittest.TestCase):
    """plotmode_from_data(): the counterpart to plotmode() for a
    broadband + range-dependent run, where per-frequency Modes dicts
    are collected in memory (see KrakenManager.last_modes) rather than
    re-read from a single '.mod' file on disk (see
    plotmode_from_data()'s docstring for why no such file exists)."""

    def tearDown(self):
        _close_all_figures()

    def _read_real_modes(self, freqs):
        from propa.kraken_toolbox.read_modes import readmodes
        return [readmodes(MOD_PATH, freq=f) for f in freqs]

    def test_matches_plotmode_for_the_same_underlying_data(self):
        # Consistency check: reading the SAME frequencies' Modes ahead
        # of time and plotting them via plotmode_from_data() must
        # produce the same grid/titles/legend as plotmode() reading
        # them from the file directly.
        freqs = [10, 20, 30]
        all_modes = self._read_real_modes(freqs)

        fig_direct = pu.plotmode(MOD_PATH, freq=freqs, n_modes=3)
        fig_from_data = pu.plotmode_from_data(all_modes, freq=freqs, n_modes=3)

        titles_direct = sorted(ax.get_title() for ax in fig_direct.axes if ax.get_title())
        titles_from_data = sorted(ax.get_title() for ax in fig_from_data.axes if ax.get_title())
        self.assertEqual(titles_direct, titles_from_data)

        legend_direct = sorted(t.get_text() for t in fig_direct.legends[0].get_texts())
        legend_from_data = sorted(t.get_text() for t in fig_from_data.legends[0].get_texts())
        self.assertEqual(legend_direct, legend_from_data)

    def test_mode_values_match_direct_phi_indexing(self):
        all_modes = self._read_real_modes([50])
        fig = pu.plotmode_from_data(all_modes, freq=50, n_modes=1)
        ax = fig.axes[0]
        real_line = next(ln for ln in ax.get_lines() if ln.get_linestyle() == "-")
        np.testing.assert_allclose(real_line.get_xdata(), all_modes[0]["phi"][:, 0].real)

    def test_bathymetry_line_on_every_panel(self):
        freqs = [10, 20, 30, 40, 50]  # 1, 3, 4, 5, 7 modes respectively
        all_modes = self._read_real_modes(freqs)
        fig = pu.plotmode_from_data(all_modes, freq=freqs, n_modes=7, bathy_depth=100.0)
        used_axes = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(used_axes), 7)
        for ax in used_axes:
            hlines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--" and ln.get_color() == "r"]
            self.assertGreaterEqual(len(hlines), 1, msg=f"seafloor line missing on '{ax.get_title()}'")

    def test_mismatched_lengths_raise(self):
        all_modes = self._read_real_modes([10, 20])
        with self.assertRaises(ValueError):
            pu.plotmode_from_data(all_modes, freq=[10, 20, 30], n_modes=2)

    def test_elastic_data_raises_clear_error(self):
        fake_modes = [{
            "M": 2,
            "z": np.array([0.0, 1.0, 2.0]),
            "phi": np.zeros((12, 2), dtype=complex),  # 4 rows x 3 depths != 3
            "selected_modes": np.array([1, 2]),
            "nb_selected_modes": 2,
            "freqVec": np.array([50.0]),
            "title": "fake",
        }]
        with self.assertRaises(ValueError):
            pu.plotmode_from_data(fake_modes, freq=50, n_modes=2)


# ======================================================================
# plotshd
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotshd(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_does_not_alter_filename_case(self):
        # NOTE: regression test for the fixed bug -- the original code
        # did `filename = filename.lower()` unconditionally, which
        # breaks on case-sensitive filesystems for any real path
        # containing uppercase characters. We can't rely on the actual
        # fixture path being mixed-case, so we spy on readshd() to
        # check the exact string it receives.
        with mock.patch.object(pu, "readshd", wraps=pu.readshd) as spy:
            pu.plotshd(SHD_PATH, freq=30)
        called_filename = spy.call_args.kwargs.get("filename", spy.call_args.args[0] if spy.call_args.args else None)
        self.assertEqual(called_filename, SHD_PATH)

    def test_basic_call_returns_the_created_figure(self):
        # NOTE: regression test for the fixed bug -- plotshd() used to
        # return None whenever (m, n, p) were not all given, even
        # though it always creates and draws into a figure regardless.
        # Every caller expecting to do `fig.savefig(...)` on the
        # result (the common, no-subplot usage) got an AttributeError.
        fig = pu.plotshd(SHD_PATH, freq=30, units="km")
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_subplot_mode_returns_figure(self):
        fig = pu.plotshd(SHD_PATH, freq=30, m=1, n=1, p=1)
        self.assertIsNotNone(fig)

    def test_axis_argument_is_used(self):
        fig, ax = plt.subplots()
        result = pu.plotshd(SHD_PATH, freq=30, axis=ax)
        self.assertIs(result, fig)
        self.assertTrue(len(ax.collections) > 0)  # pcolor drew into this axis


# ======================================================================
# plotshd_from_pressure_field
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotshdFromPressureField(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def _get_real_pressure(self, freq=30):
        from propa.kraken_toolbox.read_shd import readshd
        *_, pressure = readshd(SHD_PATH, freq=freq)
        return pressure

    def test_default_title_does_not_crash(self):
        # NOTE: regression test for the fixed bug -- the default-title
        # branch used to do `title = PlotTitle.replace(...)` then, on a
        # separate line, `+f'...'` -- never assigned back, and unary
        # '+' on a string raises `TypeError: bad operand type for
        # unary +: 'str'`. This means calling this function without an
        # explicit 'title' ALWAYS crashed.
        pressure = self._get_real_pressure()
        pu.plotshd_from_pressure_field(SHD_PATH, pressure_field=pressure, freq=30)
        # If we get here without raising, the bug is fixed.

    def test_explicit_title_still_works(self):
        pressure = self._get_real_pressure()
        pu.plotshd_from_pressure_field(
            SHD_PATH, pressure_field=pressure, freq=30, title="My custom title"
        )

    def test_returns_the_created_figure(self):
        # NOTE: regression test for the fixed bug -- this function had
        # the exact same None-return issue as plotshd() (see
        # TestPlotshd.test_basic_call_returns_the_created_figure).
        pressure = self._get_real_pressure()
        fig = pu.plotshd_from_pressure_field(SHD_PATH, pressure_field=pressure, freq=30)
        self.assertIsInstance(fig, plt.Figure)

    def test_pos_argument_bypasses_file_read_entirely(self):
        # NOTE: this is the fix for the same class of bug as
        # plotmode_from_data() -- after a broadband + range-dependent
        # run, no single '.shd' file exists on disk with every
        # frequency's grid metadata (see KrakenManager.runkraken's
        # docstring). Passing 'pos' directly (typically the 'field_pos'
        # KrakenManager.runkraken() already returns) must work with
        # filename=None, i.e. with NO file access of any kind.
        pos = {
            "r": {"r": np.linspace(0, 10000, 10), "z": np.linspace(0, 100, 21)},
            "s": {"z": np.array([25.0])},
        }
        pressure = (np.random.rand(21, 10) + 1j * np.random.rand(21, 10)) * 1e-3
        with mock.patch.object(pu, "readshd") as spy:
            fig = pu.plotshd_from_pressure_field(
                None, pressure_field=pressure, freq=50.0, pos=pos,
                base_title="My title", units="km",
            )
        spy.assert_not_called()
        self.assertIsInstance(fig, plt.Figure)

    def test_pos_argument_produces_a_sensible_default_title(self):
        pos = {
            "r": {"r": np.linspace(0, 10000, 10), "z": np.linspace(0, 100, 21)},
            "s": {"z": np.array([25.0])},
        }
        pressure = (np.random.rand(21, 10) + 1j * np.random.rand(21, 10)) * 1e-3
        fig = pu.plotshd_from_pressure_field(
            None, pressure_field=pressure, freq=50.0, pos=pos, base_title="My title",
        )
        ax = fig.axes[0]
        self.assertIn("My title", ax.get_title())
        self.assertIn("25.0", ax.get_title())


# ======================================================================
# plot_tl_profile / plot_tl_profile_multi_freq (new functions)
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotTlProfile(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_single_frequency_profile(self):
        fig = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="km")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_multi_freq_profile_overlays_one_line_per_frequency(self):
        fig = pu.plot_tl_profile_multi_freq(
            SHD_PATH, freqs=[10, 20, 30], rcv_depth=0.0, units="km"
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)

    # ------------------------------------------------------------------
    # New: optional spherical/cylindrical spreading-loss reference curves
    # ------------------------------------------------------------------
    def test_no_reference_curves_by_default(self):
        fig = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="km")
        self.assertEqual(len(fig.axes[0].lines), 1)

    def test_spherical_loss_reference_curve(self):
        fig = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="km", show_spherical_loss=True,
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)
        labels = [ln.get_label() for ln in ax.lines]
        self.assertTrue(any("Spherical" in lab for lab in labels))

    def test_cylindrical_loss_reference_curve(self):
        fig = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="km", show_cylindrical_loss=True,
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)
        labels = [ln.get_label() for ln in ax.lines]
        self.assertTrue(any("Cylindrical" in lab for lab in labels))

    def test_both_reference_curves_together(self):
        fig = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="km",
            show_spherical_loss=True, show_cylindrical_loss=True,
        )
        self.assertEqual(len(fig.axes[0].lines), 3)

    def test_spherical_curve_matches_20log10r_formula(self):
        fig = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="m", show_spherical_loss=True,
        )
        ax = fig.axes[0]
        ref_line = next(ln for ln in ax.lines if "Spherical" in ln.get_label())
        r = ref_line.get_xdata()
        tl = ref_line.get_ydata()
        # skip r=0 (deliberately left as a gap, not 20*log10(0)=-inf)
        nonzero = r > 0
        np.testing.assert_allclose(tl[nonzero], 20 * np.log10(r[nonzero]))

    def test_cylindrical_curve_matches_10log10r_formula(self):
        fig = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="m", show_cylindrical_loss=True,
        )
        ax = fig.axes[0]
        ref_line = next(ln for ln in ax.lines if "Cylindrical" in ln.get_label())
        r = ref_line.get_xdata()
        tl = ref_line.get_ydata()
        nonzero = r > 0
        np.testing.assert_allclose(tl[nonzero], 10 * np.log10(r[nonzero]))

    def test_reference_curve_uses_meters_regardless_of_display_units(self):
        # The formula is always 20/10*log10(r_METERS), even when the
        # x-axis itself is displayed in km.
        fig_km = pu.plot_tl_profile(
            SHD_PATH, freq=30, rcv_depth=0.0, units="km", show_spherical_loss=True,
        )
        ax = fig_km.axes[0]
        ref_line = next(ln for ln in ax.lines if "Spherical" in ln.get_label())
        r_km = ref_line.get_xdata()
        tl_km_axis = ref_line.get_ydata()
        nonzero = r_km > 0
        np.testing.assert_allclose(tl_km_axis[nonzero], 20 * np.log10(r_km[nonzero] * 1000.0))

    def test_reference_curves_on_multi_freq_variant(self):
        fig = pu.plot_tl_profile_multi_freq(
            SHD_PATH, freqs=[10, 20], rcv_depth=0.0, units="km",
            show_spherical_loss=True, show_cylindrical_loss=True,
        )
        # 2 frequency curves + 2 reference curves
        self.assertEqual(len(fig.axes[0].lines), 4)

    def test_reference_curves_on_multi_freq_from_data_variant(self):
        field_pos = {"r": {"r": np.linspace(0, 10000, 10), "z": np.linspace(0, 100, 21)}}
        pressure_field = (np.random.rand(2, 21, 10) + 1j * np.random.rand(2, 21, 10)) * 1e-3
        fig = pu.plot_tl_profile_multi_freq_from_data(
            pressure_field, [10.0, 20.0], field_pos, rcv_depth=0.0, units="km",
            show_spherical_loss=True, show_cylindrical_loss=True,
        )
        self.assertEqual(len(fig.axes[0].lines), 4)

    def test_units_km_is_smaller_than_units_m(self):
        # Sanity check tying this module's TL-profile helpers to the
        # m/km fix already validated in read_shd.py / test_read_shd.py.
        fig_km = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="km")
        r_km = fig_km.axes[0].lines[0].get_xdata()
        fig_m = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="m")
        r_m = fig_m.axes[0].lines[0].get_xdata()
        np.testing.assert_allclose(np.asarray(r_km) * 1000.0, r_m)


class TestPlotTlProfileMultiFreqFromData(unittest.TestCase):
    """plot_tl_profile_multi_freq_from_data(): the counterpart to
    plot_tl_profile_multi_freq() for a broadband + range-dependent run,
    where the aggregated pressure field is already in memory (see
    KrakenManager.runkraken()'s return value) rather than re-readable
    from a single '.shd' file on disk (see the function's docstring for
    why no such file exists)."""

    def tearDown(self):
        _close_all_figures()

    def _fake_field_pos(self, n_r=10, n_z=21):
        return {"r": {"r": np.linspace(0, 10000, n_r), "z": np.linspace(0, 100, n_z)}}

    def test_overlays_one_line_per_frequency(self):
        field_pos = self._fake_field_pos()
        freqs = np.array([10.0, 20.0, 30.0])
        pressure_field = (np.random.rand(3, 21, 10) + 1j * np.random.rand(3, 21, 10)) * 1e-3

        fig = pu.plot_tl_profile_multi_freq_from_data(
            pressure_field, freqs, field_pos, rcv_depth=0.0, units="km"
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)
        labels = [ln.get_label() for ln in ax.lines]
        self.assertEqual(labels, ["10 Hz", "20 Hz", "30 Hz"])

    def test_does_not_require_any_file_access(self):
        field_pos = self._fake_field_pos()
        freqs = np.array([10.0, 20.0])
        pressure_field = (np.random.rand(2, 21, 10) + 1j * np.random.rand(2, 21, 10)) * 1e-3

        with mock.patch.object(pu, "readshd") as spy:
            pu.plot_tl_profile_multi_freq_from_data(
                pressure_field, freqs, field_pos, rcv_depth=0.0, units="km"
            )
        spy.assert_not_called()

    def test_matches_file_based_version_for_the_same_underlying_data(self):
        # Consistency check against the real fixture: reading the SAME
        # frequencies' pressure fields ahead of time and plotting them
        # via the "_from_data" variant must produce the same curves as
        # plot_tl_profile_multi_freq() reading them from the file
        # directly.
        from propa.kraken_toolbox.read_shd import readshd

        freqs = [10, 20, 30]
        _, _, _, _, _, _, field_pos, pressure_field = readshd(SHD_PATH, freq=freqs)

        fig_file = pu.plot_tl_profile_multi_freq(SHD_PATH, freqs=freqs, rcv_depth=0.0, units="km")
        fig_data = pu.plot_tl_profile_multi_freq_from_data(
            pressure_field, freqs, field_pos, rcv_depth=0.0, units="km"
        )
        for ln_file, ln_data in zip(fig_file.axes[0].lines, fig_data.axes[0].lines):
            np.testing.assert_allclose(ln_file.get_ydata(), ln_data.get_ydata())


# ======================================================================
# plot_ssp / plot_attenuation / plot_density
# ======================================================================
class TestPlotSsp(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_shear_wave_curve_is_plotted_when_present(self):
        # NOTE: regression test for the fixed bug -- an erroneous
        # `cs = 0` right before the "should I hide the S-wave curve?"
        # check discarded the real cs array, so the S-wave curve was
        # hidden EVERY TIME cp was not all-zero (i.e. for any normal
        # environment with real elastic shear-wave data).
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        cp = np.array([1500.0, 1550.0, 1600.0])
        cs = np.array([200.0, 250.0, 300.0])  # real, non-zero S-wave data
        pu.plot_ssp(cp_ssp=cp, cs_ssp=cs, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertIn("S-wave", labels)
        self.assertIn("C-wave", labels)

    def test_shear_wave_curve_hidden_when_genuinely_zero(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        cp = np.array([1500.0, 1550.0, 1600.0])
        cs = np.zeros(3)  # a fluid medium: genuinely no shear waves
        pu.plot_ssp(cp_ssp=cp, cs_ssp=cs, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertNotIn("S-wave", labels)
        self.assertIn("C-wave", labels)

    def test_scalar_inputs_are_broadcast(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_ssp(cp_ssp=1500.0, cs_ssp=0.0, z=z, ax=ax)  # must not raise

    def test_z_bottom_shades_domains(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_ssp(cp_ssp=1500.0, cs_ssp=0.0, z=z, z_bottom=50.0, ax=ax)
        # color_domains draws 2 fill_between collections when z_bottom is given
        self.assertGreaterEqual(len(ax.collections), 2)


class TestPlotAttenuationAndDensity(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_plot_attenuation_basic(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_attenuation(ap=0.5, ash=0.0, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertIn("C-wave", labels)
        self.assertNotIn("S-wave", labels)

    def test_plot_density_basic(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_density(rho=1.5, z=z, ax=ax)
        self.assertEqual(len(ax.get_lines()), 1)


class TestPlotBathymetry(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="plot_bathymetry_test_")

    def tearDown(self):
        _close_all_figures()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_basic_plot(self):
        from propa.kraken_toolbox.src.kraken_env import Bathymetry

        path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(path, "w") as f:
            f.write("0,100\n5,80\n10,110\n")
        bathy = Bathymetry(path, units="km")

        fig = pu.plot_bathymetry(bathy)
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 5, 10])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [100, 80, 110])

    def test_axis_argument_is_used(self):
        from propa.kraken_toolbox.src.kraken_env import Bathymetry

        path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(path, "w") as f:
            f.write("0,100\n5,80\n")
        bathy = Bathymetry(path, units="km")

        fig, ax = plt.subplots()
        result = pu.plot_bathymetry(bathy, ax=ax)
        self.assertIs(result, fig)
        self.assertEqual(len(ax.lines), 1)


if __name__ == "__main__":
    unittest.main()
