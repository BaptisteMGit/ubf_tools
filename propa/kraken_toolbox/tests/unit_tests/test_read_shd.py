#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/read_shd.py.

These tests build a minimal, synthetic '.shd' binary file matching the
record layout documented in read_shd.py's module docstring, and read it
back with readshd_bin/readshd -- this is what caught the numpy
compatibility bug documented in read_shd.py (int()/float() on
np.fromfile results), since no test in the previous refactor pass
actually exercised the real binary parsing.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from propa.kraken_toolbox.read_shd import readshd, readshd_bin


def write_synthetic_shd(
    path,
    recl_words=200,
    title="synthetic test",
    plot_type="rectilin  ",
    freqVec=np.array([50.0, 100.0]),
    freq0=50.0,
    atten=0.0,
    theta=np.array([0.0]),
    sx=np.array([0.0]),
    sy=np.array([0.0]),
    sz=np.array([10.0]),
    rz=np.array([0.0, 50.0, 100.0]),
    rr=np.array([100.0, 200.0, 300.0, 400.0]),
):
    """Build a minimal '.shd' file matching the record layout described
    in read_shd.py's module docstring: record N starts at byte offset
    N * 4 * recl_words.

    Returns a dict with the grid sizes and a {(ifreq, itheta, isz, irz):
    complex_array} map of the exact pressure values written, so tests
    can assert on them without re-deriving the binary layout.
    """
    Nfreq = freqVec.size
    Ntheta = theta.size
    Nsx, Nsy, Nsz = sx.size, sy.size, sz.size
    Nrz, Nrr = rz.size, rr.size
    lrecl = 4 * recl_words

    def seek_to_record(f, n):
        f.seek(n * lrecl)

    with open(path, "wb") as f:
        # record 0: recl (words) + title
        seek_to_record(f, 0)
        f.write(np.array([recl_words], dtype=np.int32).tobytes())
        f.write(title.encode("utf-8")[:80].ljust(80, b" "))

        # record 1: PlotType
        seek_to_record(f, 1)
        f.write(plot_type.encode("utf-8")[:10].ljust(10, b" "))

        # record 2: Nfreq..atten
        seek_to_record(f, 2)
        for val in (Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr):
            f.write(np.array([val], dtype=np.int32).tobytes())
        f.write(np.array([freq0], dtype=np.float64).tobytes())
        f.write(np.array([atten], dtype=np.float64).tobytes())

        # record 3: freqVec
        seek_to_record(f, 3)
        f.write(freqVec.astype(np.float64).tobytes())

        # record 4: theta
        seek_to_record(f, 4)
        f.write(theta.astype(np.float64).tobytes())

        # record 5: source x
        seek_to_record(f, 5)
        f.write(sx.astype(np.float64).tobytes())

        # record 6: source y
        seek_to_record(f, 6)
        f.write(sy.astype(np.float64).tobytes())

        # record 7: source z
        seek_to_record(f, 7)
        f.write(sz.astype(np.float32).tobytes())

        # record 8: receiver z
        seek_to_record(f, 8)
        f.write(rz.astype(np.float32).tobytes())

        # record 9: receiver r
        seek_to_record(f, 9)
        f.write(rr.astype(np.float64).tobytes())

        # records 10+: pressure, one per (freq, theta, sz, rz)
        Nrcvrs_per_range = Nrz  # rectilin
        rec_idx = 10
        pressure_ref = {}
        for ifreq in range(Nfreq):
            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        seek_to_record(f, rec_idx)
                        vals = np.array(
                            [complex(ifreq * 1000 + itheta * 100 + isz * 10 + irz, ir) for ir in range(Nrr)]
                        )
                        pressure_ref[(ifreq, itheta, isz, irz)] = vals
                        interleaved = np.empty(2 * Nrr, dtype=np.float32)
                        interleaved[0::2] = vals.real.astype(np.float32)
                        interleaved[1::2] = vals.imag.astype(np.float32)
                        f.write(interleaved.tobytes())
                        rec_idx += 1

    return dict(
        Nfreq=Nfreq, Ntheta=Ntheta, Nsx=Nsx, Nsy=Nsy, Nsz=Nsz, Nrz=Nrz, Nrr=Nrr,
        freqVec=freqVec, pressure_ref=pressure_ref, title=title, plot_type=plot_type,
    )


def _assert_shd_results_equal(test, result_a, result_b):
    """Compare two readshd()/readshd_bin() return tuples field by field.
    'Pos' (a dict of numpy arrays) needs element-wise array comparison
    rather than a plain '==', which is ambiguous for arrays."""
    for a, b in zip(result_a, result_b):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            test.assertEqual(set(a.keys()), set(b.keys()))
            for key in a:
                if isinstance(a[key], dict):
                    for subkey in a[key]:
                        np.testing.assert_array_equal(a[key][subkey], b[key][subkey])
                else:
                    np.testing.assert_array_equal(a[key], b[key])
        else:
            test.assertEqual(a, b)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="read_shd_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestReadshdBinSyntheticFile(TempDirTestCase):
    def _write_default_shd(self):
        path = os.path.join(self.tmp_dir, "synthetic.shd")
        meta = write_synthetic_shd(path)
        return path, meta

    def test_header_fields_are_read_correctly(self):
        # NOTE: regression test for the fixed bug -- every
        # int(np.fromfile(..., count=1)) / float(np.fromfile(...,
        # count=1)) call in the header parsing used to raise
        # `TypeError: only 0-dimensional arrays can be converted to
        # Python scalars` with numpy >= 1.25, since np.fromfile(...,
        # count=1) returns a 1-element 1-D array, not a 0-D one. This
        # broke EVERY '.shd' read, not just an edge case.
        path, meta = self._write_default_shd()
        title, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure = readshd_bin(
            path, freq=[50.0]
        )
        self.assertEqual(title, meta["title"])
        self.assertEqual(PlotType, meta["plot_type"].strip())
        np.testing.assert_array_equal(freqVec, meta["freqVec"])

    def test_single_frequency_read_matches_written_values(self):
        path, meta = self._write_default_shd()
        *_, pressure = readshd_bin(path, freq=[50.0])
        # single-frequency axis is squeezed out
        self.assertEqual(pressure.shape, (meta["Ntheta"], meta["Nsz"], meta["Nrz"], meta["Nrr"]))
        np.testing.assert_allclose(pressure[0, 0, 0, :], meta["pressure_ref"][(0, 0, 0, 0)])
        np.testing.assert_allclose(pressure[0, 0, 2, :], meta["pressure_ref"][(0, 0, 0, 2)])

    def test_multi_frequency_read_matches_written_values(self):
        path, meta = self._write_default_shd()
        *_, pressure = readshd_bin(path, freq=[50.0, 100.0])
        self.assertEqual(
            pressure.shape, (2, meta["Ntheta"], meta["Nsz"], meta["Nrz"], meta["Nrr"])
        )
        np.testing.assert_allclose(pressure[0, 0, 0, 0, :], meta["pressure_ref"][(0, 0, 0, 0)])
        np.testing.assert_allclose(pressure[1, 0, 0, 0, :], meta["pressure_ref"][(1, 0, 0, 0)])

    def test_default_freq_reads_first_frequency(self):
        path, meta = self._write_default_shd()
        _, _, _, _, read_freq, _, _, pressure = readshd_bin(path)
        np.testing.assert_array_equal(read_freq, meta["freqVec"][:1])
        np.testing.assert_allclose(pressure[0, 0, 0, :], meta["pressure_ref"][(0, 0, 0, 0)])

    def test_nearest_frequency_match(self):
        path, meta = self._write_default_shd()
        # 60 Hz is closer to 50 Hz than to 100 Hz in meta['freqVec']
        _, _, _, _, read_freq, _, _, pressure = readshd_bin(path, freq=[60.0])
        np.testing.assert_array_equal(read_freq, [50.0])
        np.testing.assert_allclose(pressure[0, 0, 0, :], meta["pressure_ref"][(0, 0, 0, 0)])

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            readshd_bin(os.path.join(self.tmp_dir, "does_not_exist.shd"))


class TestReadshdWrapper(TempDirTestCase):
    def test_readshd_matches_readshd_bin_directly(self):
        # NOTE: regression test -- the original readshd() had 3
        # redundant branches that were functionally identical to a
        # single direct call to readshd_bin (see read_shd.py's
        # docstring). This confirms the simplified version still
        # produces identical results.
        path = os.path.join(self.tmp_dir, "synthetic.shd")
        meta = write_synthetic_shd(path)

        result_wrapper = readshd(path, freq=[50.0, 100.0])
        result_direct = readshd_bin(path, freq=[50.0, 100.0])
        _assert_shd_results_equal(self, result_wrapper, result_direct)

    def test_readshd_default_args_match(self):
        path = os.path.join(self.tmp_dir, "synthetic.shd")
        write_synthetic_shd(path)

        result_wrapper = readshd(path)
        result_direct = readshd_bin(path)
        _assert_shd_results_equal(self, result_wrapper, result_direct)


if __name__ == "__main__":
    unittest.main()
