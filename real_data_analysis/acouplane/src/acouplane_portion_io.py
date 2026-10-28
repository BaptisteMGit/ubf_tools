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
import os
import xarray as xr


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
