# Kraken toolbox refactor — summary

Scope: `kraken_env.py`, `kraken_manager.py`, `kraken_testcase.py`.
Public API (class names, method names, constructor signatures, attribute
names) is **unchanged** — these files are drop-in replacements. All
comments/docstrings are in English, as requested.

## How this was validated

I don't have `kraken.exe`/`field.exe` in this sandbox, so I couldn't run
an end-to-end simulation. Instead:

1. I reconstructed your real dependency modules (`utils.py`, `read_shd.py`,
   `run_kraken.py`, `read_modes.py`, plus `source.global_constants`,
   `cst`, `publication.publication_figure`, etc.) to get faithful imports.
2. I ran the **original** code first to capture reference `.env`/`.flp`
   output and behaviour (this is how the bugs below were found — the
   original code doesn't even run as-is in a modern numpy/pandas
   environment).
3. I wrote a 79-test unit suite (`test_kraken_*.py`, stdlib `unittest`,
   also runnable with `pytest`) covering the pure logic: letter-code
   selection, validation, numeric computations, file writing,
   mutable-default-argument regressions, frequency distribution, shell
   command construction, `.env`/`.flp` content. All 79 pass against the
   refactored code.
4. I diffed the refactored `.env`/`.flp` output against the (bug-fixed)
   original for both flat-bottom and range-dependent cases — identical
   except one intentional formatting simplification (see below).

Running the tests:
```bash
python -m unittest discover -s propa/kraken_toolbox/tests -v
# or, if pytest is installed in your project:
pytest propa/kraken_toolbox/tests
```

## Bugs found and fixed (all flagged inline with "NOTE (bug ...)")

1. **`KrakenField.phase_speed_limits` default was silently discarded.**
   `self.phase_speed_limits = np.array(phase_speed_limits)` ran right
   after the `if phase_speed_limits is None: self.phase_speed_limits = [...]`
   block and re-converted the *original* argument (still `None`) instead
   of the default just assigned. Result: `np.array(None)`, a 0-D object
   array → `IndexError` the moment `phase_speed_limits` wasn't passed
   explicitly (which is the common case).

2. **`KrakenEnv.__init__` crashed whenever no bathymetry was supplied.**
   `self.bottom_hs.derive_sedim_layer_max_depth(z_max=self.bathy.bathy_depth.max())`
   ran unconditionally, but the matching guard (`if self.bathy.use_bathy:`)
   was present only as a comment. `Bathymetry()` with no data file has no
   `bathy_depth` attribute → `AttributeError`. Fixed by falling back to
   `medium.z_ssp.max()` for the flat-bottom case.

3. **In-place `.sort()` on a pandas-derived array breaks with modern
   pandas.** `self.modes_range.sort()` sorted a `DataFrame.values` array
   in place. With pandas' Copy-on-Write (default since 2.x, mandatory in
   3.x), this array is read-only → `ValueError: sort array is
   read-only`. Fixed with `np.sort` (out-of-place).

4. **`float(self.freq)` breaks with modern numpy for a single
   frequency.** `float()` on a non-0-D array (even size-1) raises
   `TypeError` since numpy ≥ 1.25. Fixed by indexing the first element
   explicitly.

5. **Malformed `np.append` call in range-dependent SSP extension.**
   `np.append(depth, medium_copy.z_ssp, medium_copy.ash)` passes 3
   positional args; `np.append`'s signature is `(arr, values)`. Would
   have raised `TypeError` the moment `cs`/`rho`/`a_p`/`a_s` was given as
   a full array (rather than scalar) in a range-dependent run needing
   profile extension. Fixed to append the correctly-extended value.

6. **Mutable default arguments in `kraken_testcase.py` (the big one).**
   `KrakenProperties.__init__` and `KrakenTestCase.__init__` used
   *object instances* as default parameter values
   (`field=KrakenField(...)`, `kraken_properties=KrakenProperties()`,
   etc.). Python evaluates defaults once, at function-definition time —
   so every call that didn't pass these explicitly **shared the same
   object**. Since `set_bathy()` mutates `kraken.field.n_rcv_z` /
   `rcv_depth_max` and `domain.zmax_m` in place, creating two
   `KrakenTestCase`s in a row silently corrupted the second one with the
   first one's settings. I reproduced this concretely: in the original
   code, `tc1.kraken is tc2.kraken` evaluates to `True`, and a plain
   flat-bottom test case created *after* a variable-bathymetry one
   inherited the wrong `n_rcv_z`/`rcv_depth_max`. Fixed by resolving
   `None` defaults inside the constructor body (a fresh object per
   call). Regression tests: `test_kraken_testcase.py::TestKrakenPropertiesMutableDefaults`,
   `::TestKrakenTestCaseMutableDefaults`.

7. **`DomainProperties`/`ReceiverProperties`**: an unsupported `unit`
   value raised an opaque `UnboundLocalError` (alpha_z/alpha_r only
   defined inside the 'm'/'km' branches). Now raises a clear `ValueError`.

## Duplication found (flagged, not touched)

- `KrakenManager.readshd`/`readshd_bin` reimplemented, verbatim, the
  ~150 lines already living in `read_shd.py` (already imported at the
  top of the same file). Now these two methods just delegate to
  `read_shd.readshd`/`readshd_bin` — same public API, single
  implementation to maintain.
- `kraken_manager.py` (class-based) and `run_kraken.py` (procedural)
  contain near-identical logic (`runkraken`, `assign_frequency_intervalls`,
  `run_exec`, `runkraken_broadband_range_dependent`...). I didn't touch
  `run_kraken.py` since it wasn't part of the requested refactor, but
  you likely want to retire one of the two eventually — a bug fixed in
  one won't apply to the other.
- `kraken_testcase.py` had two unused imports (`socket`,
  `get_subprocess_working_dir`) — removed.

## One intentional behaviour/format change (not a bug fix)

In `KrakenMedium.write_lines`, the original code used a Fortran
shorthand (`"100.00 1500.00 / \n"`) for SSP rows when density/attenuation
are constant scalars, relying on list-directed I/O to carry over the
previous row's remaining values. The refactor always writes the full,
explicit 6-column line. This is physically equivalent for KRAKEN and
more readable/robust, but it means regenerated `.env` files won't be
byte-identical to old ones for that case. If you rely on diffing `.env`
files against a historical baseline, say so and I'll restore the
shorthand.

## Other clarifications documented in the code (not changed)

- `KrakenTestCase(mode="run")` does **not** automatically call `.run()`
  — you still need to call it explicitly. `mode` only controls whether
  diagnostic plots are generated (`mode="demo"`). This was already the
  original behaviour; now documented explicitly.
- `KrakenAttenuation`: `available_units` lists `"neper_per_m"` but the
  code that maps it to a letter code checks `"nepers_per_m"` (with an
  s) — so the documented value never actually matches. Kept as-is
  (flagged) since changing it is a behaviour change I can't validate
  without knowing which spelling your callers actually use.
