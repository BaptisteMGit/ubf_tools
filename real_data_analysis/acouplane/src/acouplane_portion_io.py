#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_portion_io.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Open, save and browse an ACOUPLANE portion (lazy access), selection of intersections and receivers.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import json
import os
import xarray as xr

# Name of the root attribute that stores the settings of the computation that produced a saved result
COMPUTATION_SETTINGS_ATTR = "computation_settings"


def open_portion(portion_fpath, chunks=None):
    """Open a portion NetCDF file as a lazy DataTree.

    Nothing is read into memory at opening time: variables are only read from
    disk when their values are actually accessed (e.g. ``.values``, plotting,
    arithmetic). The file stays open until ``portion.close()`` is called.

    Parameters
    ----------
    portion_fpath : str
        Path to the portion NetCDF file (``acouplane_portion_<id>.nc``).
    chunks : dict or None, optional
        Dask chunks passed to ``xr.open_datatree``. Requires ``dask`` to be
        installed. If None (default), plain lazy indexing is used.

    Returns
    -------
    xarray.DataTree
        Lazy portion tree with groups ``position``, ``pressure`` and, if
        already computed, ``feature`` and ``mfp_intersection``.

    Raises
    ------
    FileNotFoundError
        If ``portion_fpath`` does not exist.
    """
    if not os.path.isfile(portion_fpath):
        raise FileNotFoundError(
            f"Portion file not found: '{portion_fpath}'. "
            "Build it first with build_and_save_portion_tree()."
        )
    return xr.open_datatree(portion_fpath, auto_complex=True, chunks=chunks)


def save_portion_inplace(portion_tree, portion_fpath):
    """Save a (possibly modified) portion tree over the file it was opened from.

    The tree is first written to a temporary file next to the original one. The
    data is NOT loaded in memory beforehand: variables still backed by the
    original file are streamed from it to the temporary file. Once the
    temporary file is completely written, the tree is closed and the original
    file is atomically replaced.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree, opened with ``open_portion``. It is closed by this
        function, so it must not be used afterwards.
    portion_fpath : str
        Path of the original portion file, which is overwritten.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Any error raised while writing is re-raised after the temporary file
        has been removed. The original file is left untouched in that case.
    """
    base, ext = os.path.splitext(portion_fpath)
    tmp_fpath = f"{base}_tmp{ext}"

    try:
        portion_tree.to_netcdf(tmp_fpath, auto_complex=True)
    except Exception:
        if os.path.exists(tmp_fpath):
            os.remove(tmp_fpath)
        raise

    # The source file must be released before being replaced (mandatory on Windows)
    portion_tree.close()
    os.replace(tmp_fpath, portion_fpath)


def section_name_of(node):
    """Build the display name of a section, e.g. ``"South/North 0"``.

    Parameters
    ----------
    node : xarray.DataTree
        Node of a section (in the ``position``, ``pressure`` or ``feature`` group).

    Returns
    -------
    str
        ``"<orientation> <id>"``.
    """
    return f"{node.attrs['section_orientation']} {node.attrs['section_id']}"


def select_intersections(sections, intersections=None):
    """List the intersections (meridional x zonal section pairs) to process.

    Parameters
    ----------
    sections : list of str
        Labels of the available sections (e.g. ``["sn0", "oe1", "oe2"]``).
        Meridional sections start with ``"sn"`` and zonal sections with ``"oe"``.
    intersections : list of tuple of str or None, optional
        Requested ``(meridional_label, zonal_label)`` pairs. If None, all the
        meridional x zonal pairs are returned.

    Returns
    -------
    list of tuple of str
        The ``(meridional_label, zonal_label)`` pairs to process, without
        duplicates, in the order requested (or meridional-major order if
        ``intersections`` is None).

    Raises
    ------
    ValueError
        If there is no meridional or no zonal section, if an element of
        ``intersections`` is not a pair of labels, if a label is unknown, or if
        a pair is not (meridional, zonal). The message lists the valid
        intersections.
    """
    meridional_sections = [sec for sec in sections if sec.startswith("sn")]
    zonal_sections = [sec for sec in sections if sec.startswith("oe")]
    if not meridional_sections or not zonal_sections:
        raise ValueError(
            "At least one meridional ('sn*') and one zonal ('oe*') section are required "
            f"to build intersections (sections found: {list(sections)})."
        )
    all_pairs = [(m, z) for m in meridional_sections for z in zonal_sections]

    if intersections is None:
        return all_pairs

    valid_msg = f"Valid intersections: {all_pairs}."
    selected = []
    for item in intersections:
        if isinstance(item, str) or len(item) != 2:
            raise ValueError(
                f"Invalid intersection {item!r}: expected a (meridional, zonal) pair, "
                f"e.g. ('sn0', 'oe1'). {valid_msg}"
            )
        pair = (item[0], item[1])
        if pair not in all_pairs:
            unknown = [lab for lab in pair if lab not in sections]
            reason = (
                f"unknown section(s) {unknown}"
                if unknown
                else "the first section must be meridional ('sn*') and the second zonal ('oe*')"
            )
            raise ValueError(f"Invalid intersection {pair}: {reason}. {valid_msg}")
        if pair not in selected:
            selected.append(pair)

    if not selected:
        raise ValueError(f"The list of intersections to process is empty. {valid_msg}")

    return selected


def list_stored_intersections(portion):
    """List the intersections stored in the ``mfp_intersection`` group of a portion.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree.

    Returns
    -------
    list of tuple of str
        ``(meridional_label, zonal_label)`` of each stored intersection; empty
        if the portion has no ``mfp_intersection`` group.
    """
    # No intersection has been computed yet
    if "mfp_intersection" not in portion.children:
        return []

    # Meridional sections are the first level of the group, zonal sections the second
    return [
        (section1, section2)
        for section1, node in portion.mfp_intersection.children.items()
        for section2 in node.children
    ]


def restrict_dataset_to_receivers(ds, rcv_ids, reference=None):
    """Keep the receivers of a dataset that are listed in ``rcv_ids``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the dimension ``rcv_id_feature``.
    rcv_ids : sequence of str
        Receivers wished.
    reference : str or None, optional
        Reference receiver, never kept (it has no pair with itself).

    Returns
    -------
    ds_selected : xarray.Dataset
        Dataset restricted to the receivers found, in the stored order.
    missing : list of str
        Wished receivers that are not stored in ``ds``.

    Raises
    ------
    ValueError
        If none of the wished receivers is stored in ``ds``.
    """
    stored = [str(rcv_id) for rcv_id in ds["rcv_id_feature"].values]
    wished = [str(rcv_id) for rcv_id in rcv_ids if str(rcv_id) != str(reference)]

    missing = [rcv_id for rcv_id in wished if rcv_id not in stored]
    indices = [i for i, rcv_id in enumerate(stored) if rcv_id in wished]
    if not indices:
        raise ValueError(
            f"None of the receivers {wished} is stored in the results "
            f"(stored receivers: {stored})."
        )

    return ds.isel(rcv_id_feature=indices), missing


# ======================================================================================================================
# Saved intermediate results (CPSD, ambiguity surfaces): safe saving and reuse
# ======================================================================================================================
def file_fingerprint(fpath):
    """Describe a file by its name, size and modification time.

    Used in the settings of a stored result, so that the result is recomputed
    if the file it was computed from has changed.

    Parameters
    ----------
    fpath : str
        Path of an existing file.

    Returns
    -------
    dict
        ``name``, ``size`` (bytes) and ``mtime_ns``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    stat = os.stat(fpath)
    return dict(name=os.path.basename(fpath), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def save_tree_atomic(tree, fpath, settings):
    """Save a tree with the settings of the computation, without ever leaving a partial file.

    The tree is written to a temporary file which atomically replaces ``fpath``
    only once completely written. A crash (or an interruption) while writing
    therefore never leaves a truncated file that a later run would reuse.

    Parameters
    ----------
    tree : xarray.DataTree
        Tree to save (complex variables are supported). Its root attribute
        ``computation_settings`` is set.
    fpath : str
        Destination file. An existing file is replaced; it must not be open.
    settings : dict
        Settings of the computation (JSON serializable, other values are
        converted to strings). Compared by ``load_tree_if_settings_match``.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Any error raised while writing is re-raised after the temporary file
        has been removed; the existing file is then left untouched.
    """
    tree.attrs[COMPUTATION_SETTINGS_ATTR] = json.dumps(settings, default=str, sort_keys=True)

    base, ext = os.path.splitext(fpath)
    tmp_fpath = f"{base}_tmp{ext}"
    try:
        tree.to_netcdf(tmp_fpath, auto_complex=True)
    except Exception:
        if os.path.exists(tmp_fpath):
            os.remove(tmp_fpath)
        raise
    os.replace(tmp_fpath, fpath)


def _flatten_settings(settings, prefix=""):
    """Flatten nested settings: ``{"a": {"b": 1}}`` -> ``{"a.b": 1}`` (clearer list of the changed settings)."""
    flat = {}
    for key, value in settings.items():
        if isinstance(value, dict):
            flat.update(_flatten_settings(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def load_tree_if_settings_match(fpath, settings, validate_unrecorded=None):
    """Open a stored result (lazily) if it was computed with the requested settings.

    Parameters
    ----------
    fpath : str
        File saved with ``save_tree_atomic``.
    settings : dict
        Settings of the computation that is about to be run.
    validate_unrecorded : callable or None, optional
        Only used for a file saved without settings (by an older version of the
        code). If None, such a file is not reused. Otherwise it is called as
        ``validate_unrecorded(tree)`` and must return ``(problems, notes)``, two
        lists of strings: the inconsistencies found in the file (the file is
        then not reused) and what could not be verified (displayed only).

    Returns
    -------
    tree : xarray.DataTree or None
        Lazy tree (the file stays open until ``tree.close()``), or None if there
        is nothing reusable: no file, unreadable file, file without settings, or
        different settings.
    reason : str
        Explanation to display; for different settings, the list of the settings
        that changed (stored value / requested value).
    """
    if not os.path.isfile(fpath):
        return None, f"no stored file '{fpath}'"

    try:
        tree = open_portion(fpath)
    except Exception as error:  # corrupted or partially written file
        return None, f"stored file unreadable ({type(error).__name__}: {error})"

    stored_json = tree.attrs.get(COMPUTATION_SETTINGS_ATTR)
    if stored_json is None:
        # File saved by an older version of the code: reused only if explicitly allowed and consistent
        if validate_unrecorded is None:
            tree.close()
            return None, "stored file has no recorded settings (saved by an older version of the code)"
        problems, notes = validate_unrecorded(tree)
        if problems:
            tree.close()
            return None, "stored file without recorded settings, inconsistent with the request (" + "; ".join(problems) + ")"
        return tree, "stored file without recorded settings accepted after checks" + (
            " (not verifiable from the file: " + "; ".join(notes) + ")" if notes else ""
        )

    stored = _flatten_settings(json.loads(stored_json))
    requested = _flatten_settings(json.loads(json.dumps(settings, default=str, sort_keys=True)))
    changed = [key for key in sorted(set(stored) | set(requested)) if stored.get(key) != requested.get(key)]
    if changed:
        tree.close()
        details = "; ".join(f"{key}: stored {stored.get(key)} / requested {requested.get(key)}" for key in changed)
        return None, f"settings changed ({details})"

    return tree, "settings identical"
