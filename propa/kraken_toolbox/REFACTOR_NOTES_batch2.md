# Second batch — utils.py / read_shd.py / run_kraken.py / read_modes.py

Same approach and guarantees as the first batch (`REFACTOR_NOTES.md`):
public API unchanged (drop-in replacements), every fix documented inline
with "NOTE (bug ...)", 52 new tests (134 total across both batches, all
passing). Two of the four files (`read_shd.py`, `read_modes.py`) parse
binary formats with no real sample file available, so I built synthetic
binary files byte-by-byte matching the documented record layout and
round-tripped them through the real parsing code — this is what caught
the most serious bug below.

## Most important finding: `read_shd.py` was broken for every read

`int(np.fromfile(fid, dtype=np.int32, count=1))` / the `float(...)`
equivalent — used ~10 times in the header-parsing code — raise
`TypeError: only 0-dimensional arrays can be converted to Python
scalars` with numpy ≥ 1.25 (`np.fromfile(..., count=1)` returns a
1-element 1-D array, not a 0-D one). **This affects every single `.shd`
file read**, not an edge case — it's the exact same class of bug already
found and fixed in `kraken_env.py`'s frequency handling, just missed
here in the first pass because nothing exercised the real binary
parsing (the `kraken_manager`/`run_kraken` tests only mocked `readshd`).
Fixed by indexing `[0]` explicitly everywhere. If you're on an older
numpy where this silently worked, you'll now notice it working
everywhere, including versions where it previously didn't.

## Other bugs found and fixed

**`utils.py`**
- `get_component`: the inner loop bound for each medium used
  `len(Modes["z"])` (the *total* point count across every medium)
  instead of `Modes["N"][medium]` (that medium's own count). With mixed
  ACOUSTIC/ELASTIC media this misattributes points at medium boundaries
  and overwrites already-computed rows. Reproduced concretely: a
  2-medium (ACOUSTIC + ELASTIC) synthetic case returned `[3, 1, 2]`
  instead of `[0, 1, 2]`. Fixed with a proper per-medium point counter.
- `get_rcv_pos_idx`: supplying only one of `kraken_range`/`kraken_depth`
  (instead of both, or neither) crashed with
  `AttributeError: 'NoneType' object has no attribute 'size'`. Now
  raises a clear `ValueError`. Also added a clear error when neither the
  grids nor `shd_fpath` are given (previously would fail deep inside
  `readshd` with `filename=None`).
- `default_nb_rcv_z`/`waveguide_cutoff_freq`: added explicit
  `ValueError` for `fmax <= 0` / `depth <= 0` (previously a silent
  division by zero or nonsensical negative result).

**`read_shd.py`**
- The numpy header-parsing bug above (the big one).
- `readshd()`'s 3-branch dispatch was pure dead complexity: all 3
  branches call `readshd_bin` with arguments that are — once you account
  for its `None` defaults — identical to just forwarding everything
  unconditionally. Collapsed to one line, **zero behaviour change**
  (verified for every combination of `xs`/`ys`/`freq`). One side
  benefit: `xs`/`ys` are no longer silently dropped when `freq` is also
  given (the original never forwarded them together).
- The by-source-position read branch (`xs`/`ys` given) wrote into
  `pressure[itheta, isz, irz, :]`, but `pressure` had been allocated
  with a *leading frequency axis* — one index short. Fixed by allocating
  a properly-shaped array without the frequency axis for this branch
  (which doesn't use frequency at all). **Caveat**: this whole branch is
  explicitly flagged by the original author as "inherited from MATLAB
  and might not work anymore" / "TODO: check if still working" — I
  fixed the concrete indexing bug I could see, but had no real
  position-indexed `.shd` file to validate the fix end-to-end. Treat it
  as still provisional.
- File handle now closed via `try/finally` on every exit path (the
  original only closed it on success).

**`run_kraken.py`** (this file duplicates `kraken_manager.KrakenManager`
almost entirely — see the note in `kraken_manager.py`'s docstring; not
merged here, just fixed in place)
- `runkraken_broadband_range_dependent`: the per-frequency `KrakenEnv`
  reconstruction never passed `nmedia=env.nmedia`, silently resetting it
  to the default (1) every time — a real regression that the
  `KrakenManager` version of the same method never had. Confirmed with a
  test that spies on the `KrakenEnv` constructor across the frequency
  loop.
- `runkraken`'s parallel branch built a `multiprocessing.Pool` without a
  `with` block; an exception during `pool.starmap(...)` skipped
  `pool.close()`/`pool.join()` entirely, leaking worker processes. Fixed
  with `with multiprocessing.Pool(...) as pool:`.
- Bare `except:` around the per-frequency FIELD run narrowed to
  `except Exception as exc:`, with the error message now printed.

**`read_modes.py`**
- `readmodes()`'s extension resolution used
  `os.path.basename(modfil).split(".")[0]`, truncating at the *first*
  dot — a file named `run.v2.mod` would resolve to `run.mod`, silently
  losing `.v2`. Fixed with `os.path.splitext` (strips only the last
  extension). Verified with a synthetic `run.v2.mod` file.
- `readmodes_bin`: the file handle was never closed anywhere in the
  function. The `if not hasattr(readmodes_bin, "fid"):` guard, seemingly
  meant to cache/reuse an open handle across calls, never actually
  worked (`readmodes_bin.fid` is never assigned), so it silently always
  took the "first open" branch — with no matching close. Since nothing
  else in the function relies on a handle position left over from a
  previous call, replacing this with a single `with open(...) as fid:`
  changes nothing observable and fixes the leak.
- `modes <= Modes["M"]` / `modes - 1` require a numpy array; passing a
  plain Python list for the documented "array-like" `modes` argument
  raised a `TypeError`. Now coerced explicitly.
- Top halfspace wavenumber used `Modes["freqVec"][0]` (always the first
  stored frequency) while the structurally identical Bottom calculation
  correctly used `Modes["freqVec"][freq_index]`. Fixed for consistency.
  **Not validated against a real multi-frequency `.mod` file** (none
  available) — please double check before relying on it.
- Top/Bottom `rho`/`depth` were 1-element numpy arrays when read from
  file, but plain floats in the vacuum-boundary fallback. Normalized to
  plain floats everywhere.

## What's still duplicated (flagged, not merged)

`run_kraken.py` and `kraken_manager.KrakenManager` remain two
near-identical implementations of the same orchestration logic. Both
are now individually correct, but a future bug fix still has to be
applied twice. Worth consolidating when you have a moment — happy to
help with that whenever you're ready.
