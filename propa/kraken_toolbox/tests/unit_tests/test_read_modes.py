#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/read_modes.py.

These tests build a minimal, synthetic '.mod' binary file matching the
record layout parsed by readmodes_bin, and read it back -- this is what
validates the parsing logic beyond a purely static code review (no
real-world '.mod' sample file was available in this environment).

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

from propa.kraken_toolbox.read_modes import readmodes, readmodes_bin


def write_synthetic_mod(
    path,
    recl_words=50,
    title="synthetic modes",
    Nmedia=1,
    Nfreq=1,
    freqVec=np.array([50.0]),
    Ntot=3,
    z=np.array([0.0, 50.0, 100.0]),
    N_per_medium=np.array([3], dtype=np.int32),
    Mater=("ACOUSTIC",),
    depth_rho=np.array([[100.0], [1.0]], dtype=np.float32),
    M=1,
    NMat=3,
    top_bc="V",
    top_cp=(1500.0, 0.0),
    top_cs=(0.0, 0.0),
    top_rho=1.0,
    top_depth=0.0,
    bot_bc="A",
    bot_cp=(1700.0, 0.1),
    bot_cs=(0.0, 0.0),
    bot_rho=1.9,
    bot_depth=100.0,
    phi_values=None,
    k_values=None,
):
    """Build a minimal '.mod' file matching the record layout parsed by
    read_modes.readmodes_bin. Record N (byte-addressed) starts at
    N * lrecl_bytes, where lrecl_bytes = 4 * recl_words.

    Returns a dict with the exact phi/k values written, for assertions.
    """
    lrecl = 4 * recl_words

    if phi_values is None:
        phi_values = np.array([[complex(i, i + 0.5) for i in range(NMat)] for _ in range(M)]).T
    if k_values is None:
        k_values = np.array([complex(1.0 + 0.1 * m, 0.01) for m in range(M)])

    def seek_to_record(f, n):
        f.seek(n * lrecl)

    with open(path, "wb") as f:
        # record 0: recl (words) + title + Nfreq + Nmedia + Ntot + NMat
        seek_to_record(f, 0)
        f.write(np.array([recl_words], dtype=np.int32).tobytes())
        f.write(title.encode("utf-8")[:80].ljust(80, b" "))
        f.write(np.array([Nfreq], dtype=np.int32).tobytes())
        f.write(np.array([Nmedia], dtype=np.int32).tobytes())
        f.write(np.array([Ntot], dtype=np.int32).tobytes())
        f.write(np.array([NMat], dtype=np.int32).tobytes())

        # record 1: N[medium] (int32) + Mater[medium] (8 chars), per medium
        seek_to_record(f, 1)
        for i in range(Nmedia):
            f.write(np.array([N_per_medium[i]], dtype=np.int32).tobytes())
            f.write(Mater[i].encode("utf-8")[:8].ljust(8, b" "))

        # record 2: depth (Nmedia float32) then rho (Nmedia float32)
        seek_to_record(f, 2)
        f.write(depth_rho.astype(np.float32).tobytes())

        # record 3: freqVec (Nfreq float64)
        seek_to_record(f, 3)
        f.write(freqVec.astype(np.float64).tobytes())

        # record 4: z (Ntot float32)
        seek_to_record(f, 4)
        f.write(z.astype(np.float32).tobytes())

        # record 5: M (int32) -- single-frequency file (freq_index always 0)
        seek_to_record(f, 5)
        f.write(np.array([M], dtype=np.int32).tobytes())

        # record 6: Top + Bot halfspace info
        seek_to_record(f, 6)
        f.write(top_bc.encode("utf-8")[:1])
        f.write(np.array(top_cp, dtype=np.float32).tobytes())
        f.write(np.array(top_cs, dtype=np.float32).tobytes())
        f.write(np.array([top_rho], dtype=np.float32).tobytes())
        f.write(np.array([top_depth], dtype=np.float32).tobytes())
        f.write(bot_bc.encode("utf-8")[:1])
        f.write(np.array(bot_cp, dtype=np.float32).tobytes())
        f.write(np.array(bot_cs, dtype=np.float32).tobytes())
        f.write(np.array([bot_rho], dtype=np.float32).tobytes())
        f.write(np.array([bot_depth], dtype=np.float32).tobytes())

        # records (5+1+mode_number): phi for each mode (1-based)
        for mode_number in range(1, M + 1):
            seek_to_record(f, 5 + 1 + mode_number)
            col = phi_values[:, mode_number - 1]
            interleaved = np.empty(2 * NMat, dtype=np.float32)
            interleaved[0::2] = col.real.astype(np.float32)
            interleaved[1::2] = col.imag.astype(np.float32)
            f.write(interleaved.tobytes())

        # record (5+2+M): eigenvalues k (M complex values)
        seek_to_record(f, 5 + 2 + M)
        interleaved_k = np.empty(2 * M, dtype=np.float32)
        interleaved_k[0::2] = k_values.real.astype(np.float32)
        interleaved_k[1::2] = k_values.imag.astype(np.float32)
        f.write(interleaved_k.tobytes())

    return dict(phi_values=phi_values, k_values=k_values, M=M, NMat=NMat, title=title)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="read_modes_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestReadmodesBinSyntheticFile(TempDirTestCase):
    def _write_default_mod(self):
        path = os.path.join(self.tmp_dir, "test.mod")
        meta = write_synthetic_mod(path)
        return path, meta

    def test_basic_fields_are_read_correctly(self):
        path, meta = self._write_default_mod()
        Modes = readmodes_bin(path, freq=50.0)
        self.assertEqual(Modes["title"], meta["title"])
        self.assertEqual(Modes["Nfreq"], 1)
        self.assertEqual(Modes["Nmedia"], 1)
        self.assertEqual(Modes["M"], meta["M"])
        np.testing.assert_array_equal(Modes["N"], [3])
        self.assertEqual(list(Modes["Mater"]), ["ACOUSTIC"])
        np.testing.assert_allclose(Modes["z"], [0.0, 50.0, 100.0])

    def test_phi_and_k_match_written_values(self):
        path, meta = self._write_default_mod()
        Modes = readmodes_bin(path, freq=50.0)
        np.testing.assert_allclose(Modes["phi"], meta["phi_values"], atol=1e-5)
        np.testing.assert_allclose(Modes["k"], meta["k_values"], atol=1e-5)

    def test_top_and_bottom_halfspace_fields(self):
        path, meta = self._write_default_mod()
        Modes = readmodes_bin(path, freq=50.0)
        self.assertEqual(Modes["Top"]["BC"], "V")
        self.assertEqual(Modes["Bot"]["BC"], "A")
        # NOTE: regression test for the fixed bug -- Top/Bot 'rho' used
        # to be a 1-element numpy array when read from the file, while
        # readmodes()'s vacuum fallback set it to a plain float. Both
        # should now be plain floats regardless of boundary condition.
        self.assertIsInstance(Modes["Bot"]["rho"], float)
        self.assertAlmostEqual(Modes["Bot"]["rho"], 1.9, places=4)
        self.assertAlmostEqual(Modes["Bot"]["cp"].real, 1700.0, places=1)

    def test_modes_as_python_list_does_not_crash(self):
        # NOTE: regression test for the fixed bug -- 'modes <= Modes["M"]'
        # and 'modes - 1' required a numpy array; a plain Python list
        # raised a TypeError.
        path, _meta = self._write_default_mod()
        Modes = readmodes_bin(path, freq=50.0, modes=[1])
        np.testing.assert_array_equal(Modes["selected_modes"], [1])

    def test_modes_beyond_available_count_are_dropped(self):
        path, meta = self._write_default_mod()
        Modes = readmodes_bin(path, freq=50.0, modes=[1, 2, 3])  # only 1 mode exists
        np.testing.assert_array_equal(Modes["selected_modes"], [1])

    def test_negative_ntot_returns_early(self):
        path = os.path.join(self.tmp_dir, "empty.mod")
        write_synthetic_mod(path, Ntot=-1)
        Modes = readmodes_bin(path, freq=50.0)
        self.assertNotIn("z", Modes)  # returned before the 'z' block was read


class TestReadmodesWrapper(TempDirTestCase):
    def test_extension_is_forced_to_mod(self):
        path = os.path.join(self.tmp_dir, "test.mod")
        write_synthetic_mod(path)
        # Ask for the same base name but with a different extension:
        # readmodes() must still resolve it to 'test.mod'.
        Modes = readmodes(os.path.join(self.tmp_dir, "test.env"), freq=50.0)
        self.assertEqual(Modes["title"], "synthetic modes")

    def test_multi_dot_filename_is_resolved_correctly(self):
        # NOTE: regression test for the fixed bug -- the original
        # extension-resolution logic used split(".")[0], truncating at
        # the FIRST dot. A file named 'run.v2.mod' would have been
        # mistakenly looked up as 'run.mod'.
        path = os.path.join(self.tmp_dir, "run.v2.mod")
        write_synthetic_mod(path, title="multi dot file")
        Modes = readmodes(os.path.join(self.tmp_dir, "run.v2.env"), freq=50.0)
        self.assertEqual(Modes["title"], "multi dot file")

    def test_acousto_elastic_bottom_gets_wavenumber_info(self):
        path = os.path.join(self.tmp_dir, "test.mod")
        write_synthetic_mod(path, bot_bc="A")
        Modes = readmodes(path, freq=50.0)
        self.assertIn("k2", Modes["Bot"])
        self.assertIn("gamma", Modes["Bot"])

    def test_vacuum_top_gets_default_fallback_values(self):
        path = os.path.join(self.tmp_dir, "test.mod")
        write_synthetic_mod(path, top_bc="V")
        Modes = readmodes(path, freq=50.0)
        self.assertEqual(Modes["Top"]["rho"], 1.0)
        self.assertNotIn("k2", Modes["Top"])

    def test_file_handle_is_closed_after_read(self):
        # NOTE: regression test for the fixed bug -- the original code
        # never closed the file handle (a broken, always-false
        # hasattr-based "cache" guard meant the open() branch always ran,
        # with no matching close()). Calling readmodes_bin repeatedly on
        # the same path must not raise or leak resources.
        path = os.path.join(self.tmp_dir, "test.mod")
        write_synthetic_mod(path)
        for _ in range(50):
            readmodes_bin(path, freq=50.0)  # must not raise


if __name__ == "__main__":
    unittest.main()
