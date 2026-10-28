#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/09/28 21:54:27
@Author  :   Menetrier Baptiste
@Version :   1.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Useful functions for the acouplane project
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import re
import functools
import warnings
import numpy as np
import xarray as xr
import pandas as pd

import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import timedelta

from publication.publication_figure import set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_DATA_CELERITY_SIPPICANS_DIR,
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_DATA_DIR,
)
from get_data.obs_bin.obs_reader_acouplane import convert_data_raw_to_physical
from source.utils.utils_geo import geodetic_to_ecef, ecef_to_enu

# Maximum number of elements of the temporary 3D arrays built when comparing two sections
# (see derive_feature_distance_between_sections). Keeps the RAM usage bounded.
_MAX_BROADCAST_ELEMENTS = 20_000_000

# ======================================================================================================================
# Portion file handling
# ======================================================================================================================


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


# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================


def build_obs_datetime_vector(ds_pressure):
    """Build the UTC datetime vector of each OBS from its start time and sampling rate.

    The time step is computed in integer nanoseconds from ``ds_pressure.fs`` to
    avoid the accumulation of floating point errors in the date range.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        Raw pressure dataset. Must contain the coordinate ``obs_id``, the
        variable ``start_datetime`` (dimension ``obs_id``), the dimensions
        ``time_<obs_id>`` and the attribute ``fs`` (sampling frequency in Hz).

    Returns
    -------
    dict of pandas.DatetimeIndex
        One datetime vector per OBS, keyed by obs_id. The vector of each OBS has
        the same length as its ``time_<obs_id>`` dimension.
    """
    datetime_obs_dict = {}
    step_ns = int(round(1e9 / ds_pressure.fs))

    for obs_id in ds_pressure.obs_id.values:
        start_datetime_obs = pd.to_datetime(
            ds_pressure.start_datetime.sel(obs_id=obs_id).values
        ).to_pydatetime()

        datetime_obs_dict[obs_id] = pd.date_range(
            start=start_datetime_obs,
            freq=f"{step_ns}ns",
            inclusive="left",
            periods=ds_pressure.sizes[f"time_{obs_id}"],
        )

    return datetime_obs_dict


def convert_raw_pressure_to_physical_units(ds_pressure):
    """Convert the raw (ADC counts) pressure of every OBS to volts.

    The dataset is modified in place: the variables ``pressure_<obs_id>`` are
    replaced by their converted version and their ``units`` attribute is set to
    ``"V"``.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        Dataset with the variables ``pressure_<obs_id>`` and the attribute
        ``fullscale`` (full scale of the converter).

    Returns
    -------
    xarray.Dataset
        The same dataset, with pressures expressed in volts.
    """
    for obs_id in ds_pressure.obs_id.values:
        raw_pressure = ds_pressure[f"pressure_{obs_id}"]
        pressure = convert_data_raw_to_physical(
            raw_data=raw_pressure, fullScale=ds_pressure.fullscale
        )
        ds_pressure[f"pressure_{obs_id}"] = pressure
        ds_pressure[f"pressure_{obs_id}"].attrs["units"] = "V"

    return ds_pressure


def build_section_orientation(sections_orientation_code):
    """Build human readable orientations and unique labels for a list of sections.

    Parameters
    ----------
    sections_orientation_code : list of str
        Orientation code of each section: ``"sn"`` (south/north) or ``"oe"``
        (west/east). Any code other than ``"sn"`` is interpreted as West/East.

    Returns
    -------
    sections_orientation : list of str
        ``"South/North"`` or ``"West/East"`` for each section.
    sections_orientation_label : list of str
        Unique label of each section, built as ``<code><index>`` (e.g.
        ``"sn0"``, ``"oe1"``). These labels are used as group names in the
        portion tree.
    """
    sections_orientation = [
        "South/North" if code == "sn" else "West/East"
        for code in sections_orientation_code
    ]
    sections_orientation_label = [
        f"{code}{i}" for i, code in enumerate(sections_orientation_code)
    ]
    return sections_orientation, sections_orientation_label


# ======================================================================================================================
# Celerity data preprocessing
# ======================================================================================================================


def edf_parser_convert_value(value):
    """Convert a string read in an EDF header to a suitable Python type.

    Parameters
    ----------
    value : str
        Raw value, possibly surrounded by whitespaces.

    Returns
    -------
    None or int or float or str
        None for empty values and for ``"none"``, ``"n/a"``, ``"null"``
        (case insensitive); an int for integer-looking values; a float for
        decimal or scientific notation values; the stripped string otherwise.
    """
    value = value.strip()

    # Empty values
    if not value:
        return None

    # Explicit missing values
    if value.lower() in {"none", "n/a", "null"}:
        return None

    # Integer or float, including scientific notation
    try:
        number = float(value)

        if number.is_integer() and not any(
            char in value.lower() for char in (".", "e")
        ):
            return int(number)

        return number

    except ValueError:
        return value


def parse_edf_header(lines):
    """Parse the header of a SIPPICANS ``*.edf`` file into a nested dictionary.

    Lines starting with ``//`` open a new section (the section name is
    lower-cased and spaces are replaced by underscores). The ``// Data``
    marker ends the header. Other lines of the form ``key : value`` are stored
    in the current section (``"general"`` before the first section heading).

    Parameters
    ----------
    lines : list of str
        All lines of the file, as returned by ``file.readlines()``.

    Returns
    -------
    header : dict
        Nested dictionary ``{section: {key: value}}``.
    header_length : int or None
        Index of the ``// Data`` line, i.e. the number of lines to skip before
        the data table. None if no ``// Data`` marker was found.
    """
    header = {}
    section = "general"

    header_length = None

    for line_idx, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        # Detect section headings
        if line.startswith("//"):
            section_name = line.lstrip("/ ").strip()

            if section_name:
                if section_name != "Data":
                    section = section_name.lower().replace(" ", "_")
                    header.setdefault(section, {})
                else:
                    header_length = line_idx

            continue

        # Parse key-value pairs
        match = re.match(r"^([^:]+?)\s*:\s*(.*)$", line)

        if not match:
            continue

        key, value = match.groups()
        key = key.strip()
        value = edf_parser_convert_value(value)

        header.setdefault(section, {})[key] = value

    return header, header_length


def dmm_to_decimal(coord):
    """Convert a coordinate in degrees and decimal minutes to decimal degrees.

    Example: ``'43 9.88329N'`` -> ``43.1647215``.

    Parameters
    ----------
    coord : str
        Coordinate formatted as ``"<degrees> <decimal minutes><N|S|E|W>"``.

    Returns
    -------
    float
        Coordinate in decimal degrees, negative for the S and W hemispheres.

    Raises
    ------
    ValueError
        If ``coord`` does not match the expected format.
    """
    match = re.fullmatch(r"\s*(\d+)\s+(\d+(?:\.\d+)?)\s*([NSEW])\s*", coord.upper())

    if not match:
        raise ValueError(
            f"Invalid coordinate format: '{coord}'. "
            "Expected '<degrees> <decimal minutes><N|S|E|W>', e.g. '43 9.88329N'."
        )

    degrees = float(match.group(1))
    minutes = float(match.group(2))
    hemisphere = match.group(3)

    decimal = degrees + minutes / 60

    if hemisphere in ("S", "W"):
        decimal = -decimal

    return decimal


def read_sippicans(data_folder=ACOUPLANE_DATA_CELERITY_SIPPICANS_DIR, verbose=True):
    """Read every SIPPICANS XBT file of a folder (sound speed profiles).

    For each file, the encoding is guessed, the header is parsed, the data table
    is read and the default value ``-99`` is replaced by NaN in the temperature
    and sound velocity columns.

    Parameters
    ----------
    data_folder : str, optional
        Folder containing the SIPPICANS ``*.edf`` files.
    verbose : bool, optional
        If True, print the file being read and its detected encoding.

    Returns
    -------
    dict
        One entry per file, keyed by the raw data filename found in the header.
        Each entry is a dict with the keys ``longitude`` and ``latitude``
        (decimal degrees), ``surface_temperatude_degC``, ``pression_atmo_hPa``,
        ``salinity_ppt`` and ``data`` (pandas.DataFrame of the profile).

    Raises
    ------
    FileNotFoundError
        If ``data_folder`` does not exist.
    ValueError
        If the encoding of a file cannot be detected, or if its header does not
        contain the expected sections.
    """
    from charset_normalizer import from_bytes

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"SIPPICANS folder not found: '{data_folder}'.")

    # List files in SIPPICANS folder
    sip_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f))
    ]

    celerity_data = {}

    default_value = -99
    for fpath in sip_files:
        # Step 1: read data in binary format and guess encoding
        with open(fpath, "rb") as f:
            data = f.read()
        result = from_bytes(data).best()
        if result is None:
            raise ValueError(f"Could not detect the encoding of file '{fpath}'.")
        encoding = result.encoding

        if verbose:
            print(f"[SIPPICANS] Reading '{fpath}' (encoding: {encoding})")

        with open(fpath, "r", encoding=encoding) as f:
            lines = f.readlines()

        # Read metadata
        metadata, header_length = parse_edf_header(lines)
        for required in ("data_fields", "file_information", "launch_information"):
            if required not in metadata:
                raise ValueError(
                    f"Section '{required}' not found in the header of '{fpath}' "
                    f"(sections found: {list(metadata)})."
                )
        if header_length is None:
            raise ValueError(f"No '// Data' marker found in '{fpath}'.")

        # Read data
        names = [
            metadata["data_fields"][f"Field{i}"]
            for i in range(1, metadata["data_fields"]["Num Data Fields"] + 1)
        ]
        df_celerity = pd.read_csv(
            fpath, header=header_length, delimiter="\t", encoding=encoding, names=names
        )

        # Replace invalid values by nan
        df_celerity.loc[
            df_celerity["Temperature (°C)"] == default_value, "Temperature (°C)"
        ] = np.nan
        df_celerity.loc[
            df_celerity["Sound Velocity (m/s)"] == default_value, "Sound Velocity (m/s)"
        ] = np.nan

        # Store
        f_id = metadata["file_information"]["Raw Data Filename"]
        celerity_data[f_id] = {}  # Init
        celerity_data[f_id]["longitude"] = np.round(
            dmm_to_decimal(metadata["launch_information"]["Longitude"]), 7
        )
        celerity_data[f_id]["latitude"] = np.round(
            dmm_to_decimal(metadata["launch_information"]["Latitude"]), 7
        )
        celerity_data[f_id]["surface_temperatude_degC"] = metadata[
            "launch_information"
        ]["T°surf_(dd.d)"]
        celerity_data[f_id]["pression_atmo_hPa"] = metadata["launch_information"][
            "P_atm(pppp.p)"
        ]
        celerity_data[f_id]["salinity_ppt"] = float(
            metadata["information_-_xbt"]["Salinity"].split(" ")[0]
        )
        celerity_data[f_id]["data"] = df_celerity

    return celerity_data


# ======================================================================================================================
# Data processing
# ======================================================================================================================


def extract_pressure_section(section_start, section_end, ds_pressure, obs_datetimes):
    """Extract the part of the pressure recordings between two datetimes.

    The extraction is done independently for each OBS (they do not share the
    same time axis). The time coordinates of the returned dataset are replaced
    by UTC datetimes.

    Parameters
    ----------
    section_start, section_end : datetime-like
        Start and end (both included) of the section, in UTC.
    ds_pressure : xarray.Dataset
        Raw pressure dataset (variables ``pressure_<obs_id>`` with dimensions
        ``time_<obs_id>``).
    obs_datetimes : dict of pandas.DatetimeIndex
        Datetime vector of each OBS, as returned by ``build_obs_datetime_vector``.

    Returns
    -------
    xarray.Dataset
        Pressure dataset restricted to the section.

    Raises
    ------
    ValueError
        If, for at least one OBS, the recording does not overlap the section.
    """
    # Build slice for each rcv
    obs_slice_idx = []
    for obs_id in ds_pressure.obs_id.values:
        idx_start = np.where(obs_datetimes[obs_id] >= section_start)[0]
        idx_end = np.where(obs_datetimes[obs_id] <= section_end)[0]
        if idx_start.size == 0 or idx_end.size == 0 or idx_start[0] > idx_end[-1]:
            raise ValueError(
                f"OBS '{obs_id}' has no data between {section_start} and {section_end} "
                f"(recording spans {obs_datetimes[obs_id][0]} -> {obs_datetimes[obs_id][-1]})."
            )
        obs_slice_idx.append((idx_start[0], idx_end[-1]))

    # Build slicing dict
    isel_dict = {
        f"time_{obs_id}": slice(obs_slice_idx[i][0], obs_slice_idx[i][1] + 1)
        for i, obs_id in enumerate(ds_pressure.obs_id.values)
    }
    # Slice pressure dataset
    pressure_slice = ds_pressure.isel(isel_dict)

    # Convert time to UTC
    for i, obs_id in enumerate(ds_pressure.obs_id.values):
        pressure_slice[f"time_{obs_id}"] = obs_datetimes[obs_id][
            obs_slice_idx[i][0] : obs_slice_idx[i][1] + 1
        ]
        # Update attributes accordingly
        pressure_slice[f"time_{obs_id}"].attrs = {"unit": "UTC", "long_name": "Time"}

    return pressure_slice


def extract_and_plot_pressure_section(
    section_start, section_end, ds_pressure, obs_datetimes, single_fig=False
):
    """Extract a pressure section, convert it to volts and plot it.

    Parameters
    ----------
    section_start, section_end : datetime-like
        Start and end of the section, in UTC.
    ds_pressure : xarray.Dataset
        Raw pressure dataset.
    obs_datetimes : dict of pandas.DatetimeIndex
        Datetime vector of each OBS (see ``build_obs_datetime_vector``).
    single_fig : bool, optional
        If True, all OBS are drawn on a single axis, otherwise one subplot per
        OBS is used.

    Returns
    -------
    xarray.Dataset
        Pressure dataset of the section, expressed in volts.
    """
    # Extract section of interest
    ds_pressure_section = extract_pressure_section(
        section_start, section_end, ds_pressure, obs_datetimes
    )
    # Convert to volts
    ds_pressure_section = convert_raw_pressure_to_physical_units(
        ds_pressure=ds_pressure_section
    )

    # Plot
    plot_pressure_section(
        ds_pressure_section=ds_pressure_section, single_fig=single_fig
    )

    return ds_pressure_section


def select_pos_section(section_start, section_end, ds_pos):
    """Select the positions (ATL, AIS and source) between two datetimes.

    Parameters
    ----------
    section_start, section_end : datetime-like
        Start and end of the section.
    ds_pos : xarray.Dataset
        Position dataset with the coordinates ``atl_datetime``, ``ais_datetime``
        and ``src_datetime``.

    Returns
    -------
    xarray.Dataset
        Position dataset restricted to the section.
    """
    return ds_pos.sel(
        atl_datetime=slice(section_start, section_end),
        ais_datetime=slice(section_start, section_end),
        src_datetime=slice(section_start, section_end),
    )


def select_and_plot_section(section_start, section_end, ds_pos):
    """Select a section of the position dataset and plot the shot positions.

    Shots are colored by emission time and the OBS positions are overlaid.

    Parameters
    ----------
    section_start, section_end : datetime-like
        Start and end of the section.
    ds_pos : xarray.Dataset
        Position dataset (see ``select_pos_section``).

    Returns
    -------
    xarray.Dataset
        Position dataset restricted to the section.
    """
    ds_pos_section = select_pos_section(section_start, section_end, ds_pos)

    fig, ax = plt.subplots()

    dates = ds_pos_section.src_datetime.values
    date_num = mdates.date2num(dates)

    sc = ax.scatter(
        ds_pos_section.tir_src_lon,
        ds_pos_section.tir_src_lat,
        c=date_num,
        cmap="jet",
    )

    plot_obs_pos(ds_pos_section, ax=ax)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Source datetime")
    cbar.ax.yaxis.set_major_locator(mdates.DayLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return ds_pos_section


def compute_src_rcv_geodesic_dist(ds_pos_sections):
    """Add the horizontal source-receiver geodesic distance to position datasets.

    Distances are computed on the WGS84 ellipsoid. The datasets are modified in
    place (and also returned for convenience).

    Parameters
    ----------
    ds_pos_sections : list of xarray.Dataset
        Position datasets, each with the variables ``tir_src_lon``,
        ``tir_src_lat`` (dimension ``src_datetime``) and ``rcv_lon``,
        ``rcv_lat`` (dimension ``rcv_id``).

    Returns
    -------
    list of xarray.Dataset
        The same datasets, with the new variable ``src_rcv_geodesic_dist`` in
        meters, of dimensions ``(rcv_id, src_datetime)``.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    for ds_pos_section in ds_pos_sections:

        horizontal_distance = []

        slon, slat = (
            ds_pos_section.tir_src_lon.values,
            ds_pos_section.tir_src_lat.values,
        )
        for rcv_id in ds_pos_section.rcv_id.values:
            rlon = float(ds_pos_section.sel(rcv_id=rcv_id).rcv_lon.values)
            rlat = float(ds_pos_section.sel(rcv_id=rcv_id).rcv_lat.values)

            # Cast to src pos shape
            rlon_ = np.full(slon.size, rlon)
            rlat_ = np.full(slon.size, rlat)
            # Compute geodesic distance (only the distance is kept)
            _, _, horiz_d = geod.inv(lons1=slon, lats1=slat, lons2=rlon_, lats2=rlat_)
            horizontal_distance.append(horiz_d)

        horizontal_distance = np.array(horizontal_distance)

        # Add to dataset
        ds_pos_section["src_rcv_geodesic_dist"] = (
            ["rcv_id", "src_datetime"],
            horizontal_distance,
        )

    return ds_pos_sections


def build_and_save_portion_tree(
    ds_pos_sections,
    ds_pressure_sections,
    sections_orientation,
    sections_label,
    portion_id,
    root_fpath=ACOUPLANE_PORTION_NC_ROOT_FPATH,
):
    """Group position and pressure sections in a DataTree and save it as NetCDF.

    The inputs are copied, so they are not modified. The source-receiver
    geodesic distance is added to the position datasets and traceability
    attributes (portion id, section orientation, id and label) are added to
    every section.

    Parameters
    ----------
    ds_pos_sections : list of xarray.Dataset
        Position dataset of each section.
    ds_pressure_sections : list of xarray.Dataset
        Pressure dataset of each section, in the same order.
    sections_orientation : list of str
        Orientation of each section (see ``build_section_orientation``).
    sections_label : list of str
        Unique label of each section, used as group name.
    portion_id : int
        Identifier of the portion.
    root_fpath : str, optional
        Root of the output path; the file is saved as ``<root_fpath>_<portion_id>.nc``.

    Returns
    -------
    xarray.DataTree
        The portion tree, with the groups ``/position/<label>`` and
        ``/pressure/<label>``.

    Raises
    ------
    ValueError
        If the lists do not all have the same length.
    """
    n_sections = len(ds_pos_sections)
    lengths = {
        "ds_pos_sections": n_sections,
        "ds_pressure_sections": len(ds_pressure_sections),
        "sections_orientation": len(sections_orientation),
        "sections_label": len(sections_label),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent number of sections between inputs: {lengths}.")

    # Copy to avoid muting the same object
    ds_pos_sections = [ds.copy() for ds in ds_pos_sections]
    ds_pressure_sections = [ds.copy() for ds in ds_pressure_sections]

    # Add horizontal distance info to dataset
    ds_pos_sections = compute_src_rcv_geodesic_dist(ds_pos_sections=ds_pos_sections)

    # Add metadata to improve traceability
    for i in range(n_sections):
        for ds in (ds_pos_sections[i], ds_pressure_sections[i]):
            ds.attrs["portion_id"] = portion_id
            ds.attrs["section_orientation"] = sections_orientation[i]
            ds.attrs["section_id"] = i
            ds.attrs["section_label"] = sections_label[i]

    # Build tree dict
    tree_dict_positions = {
        f"/position/{sections_label[i]}": ds_pos_sections[i] for i in range(n_sections)
    }
    tree_dict_pressure = {
        f"/pressure/{sections_label[i]}": ds_pressure_sections[i]
        for i in range(n_sections)
    }
    tree_dict = tree_dict_positions | tree_dict_pressure

    # Build tree
    portion_tree = xr.DataTree.from_dict(tree_dict)

    # Save as netcdf
    fpath = f"{root_fpath}_{portion_id}.nc"
    portion_tree.attrs["portion_filepath"] = fpath
    portion_tree.to_netcdf(fpath)

    return portion_tree


def get_portion_tir_mean_pos(portion_tree):
    """Compute the mean shot position of a portion.

    The mean is the average of the per-section mean positions (each section has
    the same weight, whatever its number of shots).

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group.

    Returns
    -------
    mean_lon, mean_lat : float
        Mean longitude and latitude, in degrees.
    """
    mean_lon = []
    mean_lat = []
    for _, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        mean_lon.append(ds_pos_section.tir_src_lon.mean().values)
        mean_lat.append(ds_pos_section.tir_src_lat.mean().values)

    return np.mean(mean_lon), np.mean(mean_lat)


def get_portion_tir_arr_dt(portion_tree, offset1=200, offset2=1500, obs_ids=None):
    """Measure the arrival time of every shot on every OBS by peak detection.

    For each shot and OBS, a window ``[theoretical arrival - offset1,
    theoretical arrival + offset2]`` is extracted. The first sample exceeding 5
    times the standard deviation of the last 100 samples of the window (noise
    estimate) is taken as the arrival. The propagation time is then the
    difference between this arrival and the emission time.

    The result is stored in each ``position`` section as the variable
    ``tps_propagation_measured_s`` (dimensions ``(rcv_id, src_datetime)``),
    NaN where no arrival is found, where the window falls outside the recording
    or where the OBS is not in ``obs_ids``. Requires
    ``derive_portion_propagation_time`` to have been called beforehand.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with the groups ``position`` and ``pressure``.
    offset1 : int, optional
        Window length before the theoretical arrival, in ms.
    offset2 : int, optional
        Window length after the theoretical arrival, in ms.
    obs_ids : list of str or None, optional
        OBS to process. If None, all OBS of the portion are processed.

    Returns
    -------
    None
        The portion tree is modified in place.
    """
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        # Associated pressure section
        ds_pressure_section = portion_tree.pressure[name].to_dataset()

        if "tir_reception_datetime" not in ds_pos_section:
            raise KeyError(
                f"Variable 'tir_reception_datetime' missing in section '{name}': "
                "call derive_portion_propagation_time() first."
            )

        if obs_ids is None:
            obs_ids = ds_pos_section.rcv_id.values

        tps_propa_measured_s = []
        # iterate over tirs
        for i_tir in range(ds_pos_section.sizes["src_datetime"]):
            # Theoretical arrival datetime
            arr_dt_th = ds_pos_section["tir_reception_datetime"].isel(
                src_datetime=slice(i_tir, i_tir + 1)
            )
            # Emission datetime
            emission_dt = ds_pos_section["src_datetime"].isel(
                src_datetime=slice(i_tir, i_tir + 1)
            )

            tps_propa_measured_s_rcv = []
            for rcv_id in ds_pos_section.rcv_id.values:

                if rcv_id not in obs_ids:
                    propagation_time_measured_s = np.nan

                else:

                    arr_dt_th_rcv = arr_dt_th.sel(rcv_id=rcv_id)

                    # Select associated data window
                    press_rcv = ds_pressure_section[f"pressure_{rcv_id}"].sel(
                        {
                            f"time_{rcv_id}": slice(
                                arr_dt_th_rcv.values[0] - np.timedelta64(offset1, "ms"),
                                arr_dt_th_rcv.values[0] + np.timedelta64(offset2, "ms"),
                            )
                        }
                    )

                    # We might fall at the border -> skip to avoid issues
                    if press_rcv.size == 0:
                        propagation_time_measured_s = np.nan

                    else:
                        # Normalize and center
                        press_rcv = press_rcv / press_rcv.max()
                        press_rcv = press_rcv - press_rcv.mean()

                        # Find peak (5 times the noise std seems a reasonable threshold)
                        peak_height = 5 * np.std(press_rcv.values[-100:])
                        peaks_idx, _ = sp.find_peaks(press_rcv, height=peak_height)
                        if peaks_idx.size > 0:
                            # Get arrival datetime
                            arr_dt_measured = press_rcv[f"time_{rcv_id}"].values[
                                peaks_idx[0]
                            ]
                            # Associated propagation time
                            propagation_time_measured_ns = arr_dt_measured - emission_dt
                            propagation_time_measured_s = (
                                propagation_time_measured_ns.values[0]
                                / np.timedelta64(1, "s")
                            )
                        else:
                            propagation_time_measured_s = np.nan

                tps_propa_measured_s_rcv.append(propagation_time_measured_s)

            tps_propa_measured_s.append(np.array(tps_propa_measured_s_rcv))

        # Add to portion
        ds_pos_section["tps_propagation_measured_s"] = (
            ["rcv_id", "src_datetime"],
            np.vstack(tps_propa_measured_s).T,
        )

        # Update attribute
        ds_pos_section.tps_propagation_measured_s.attrs = {
            "units": "s",
            "long_name": "Estimated propagation time",
        }

        # Update tree
        node.ds = ds_pos_section


def apply_rcv_position_correction_to_portion(
    portion_tree, rcv_pos_corr_fname="rcv_position_corrected.csv"
):
    """Replace the OBS positions of a portion by corrected (LMS inversion) ones.

    The corrected positions are read from a CSV file with the columns ``lon``
    and ``lat``, one row per OBS in the same order as the ``rcv_id`` coordinate
    of the portion. The source-receiver geodesic distance is recomputed and the
    attributes ``obs_position`` and ``obs_position_filepath`` are updated.

    Only the ``position`` group is updated. Features (``feature`` group)
    already computed with the previous positions are NOT updated and must be
    recomputed.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group (modified in place).
    rcv_pos_corr_fname : str, optional
        Name of the CSV file, located in ``ROOT_ACOUPLANE_PROCESSED_DATA_DIR``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV file lacks the ``lon``/``lat`` columns or does not contain
        one row per receiver.
    """
    rcv_pos_fpath = os.path.join(ROOT_ACOUPLANE_PROCESSED_DATA_DIR, rcv_pos_corr_fname)
    if not os.path.isfile(rcv_pos_fpath):
        raise FileNotFoundError(
            f"Corrected OBS position file not found: '{rcv_pos_fpath}'."
        )
    df_rcv_pos_corr = pd.read_csv(rcv_pos_fpath)

    missing_cols = {"lon", "lat"} - set(df_rcv_pos_corr.columns)
    if missing_cols:
        raise ValueError(
            f"Columns {sorted(missing_cols)} missing in '{rcv_pos_fpath}' "
            f"(columns found: {list(df_rcv_pos_corr.columns)})."
        )

    # Update tree
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()

        n_rcv = ds_pos_section.sizes["rcv_id"]
        if len(df_rcv_pos_corr) != n_rcv:
            raise ValueError(
                f"'{rcv_pos_fpath}' has {len(df_rcv_pos_corr)} rows but section '{name}' "
                f"has {n_rcv} receivers ({list(ds_pos_section.rcv_id.values)}): "
                "one row per receiver, in the same order, is expected."
            )

        # Update ds_pos_section
        ds_pos_section.update(
            {
                "rcv_lon": ("rcv_id", df_rcv_pos_corr["lon"].to_numpy()),
                "rcv_lat": ("rcv_id", df_rcv_pos_corr["lat"].to_numpy()),
            }
        )
        # No need to update immersion as it is not changed (2D LMS estimation)

        # Update geodetic distance between source and receiver
        ds_pos_section = compute_src_rcv_geodesic_dist(
            ds_pos_sections=[ds_pos_section]
        )[0]

        # Add attributes to track modifications
        ds_pos_section.attrs["obs_position"] = (
            "OBS position corrected using LMS inversion."
        )
        ds_pos_section.attrs["obs_position_filepath"] = rcv_pos_fpath

        # Update tree
        node.ds = ds_pos_section


def derive_portion_propagation_time(portion_tree, c0=1508):
    """Compute the theoretical propagation time and reception datetime of every shot.

    A straight line propagation path at constant celerity ``c0`` is assumed:
    ``t = sqrt(horizontal_distance**2 + rcv_immersion**2) / c0``. The results
    are stored in each ``position`` section as ``tps_propagation_s`` (seconds)
    and ``tir_reception_datetime``, both of dimensions ``(rcv_id, src_datetime)``.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree whose ``position`` sections contain
        ``src_rcv_geodesic_dist`` and ``rcv_immersion`` (modified in place).
    c0 : float, optional
        Equivalent sound celerity in m/s.

    Returns
    -------
    None
    """
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()

        for var in ("src_rcv_geodesic_dist", "rcv_immersion"):
            if var not in ds_pos_section:
                raise KeyError(
                    f"Variable '{var}' missing in position section '{name}' "
                    f"(variables found: {list(ds_pos_section.data_vars)})."
                )

        # Propagation distance assuming strait line propagation path
        dpropa_direct = np.sqrt(
            ds_pos_section.src_rcv_geodesic_dist**2 + ds_pos_section.rcv_immersion**2
        )
        # Compute propagation time
        tps_propagation_s = dpropa_direct / c0
        ds_pos_section["tps_propagation_s"] = tps_propagation_s
        ds_pos_section["tps_propagation_s"].attrs = dict(
            units="s", long_name="Propagation time"
        )

        # Derive associated arrival dt
        tps_propagation_ns = (tps_propagation_s * 1e9).round()
        tps_propagation_td = tps_propagation_ns.astype("timedelta64[ns]")
        ds_pos_section["tir_reception_datetime"] = (
            tps_propagation_td + ds_pos_section.src_datetime
        )
        ds_pos_section["tir_reception_datetime"].attrs = dict(
            long_name="Reception datetime"
        )

        # Update tree
        node.ds = ds_pos_section


# ======================================================================================================================
# ENU <-> WGS84 conversion
# ======================================================================================================================


def transform_pos_wgs84_enu(
    lon, lat, local_frame_origin, N_geoid_undulation, zeta=0, sensor_h_above_sea_level=5
):
    """Convert WGS84 positions to a local East-North-Up (ENU) frame.

    The ellipsoidal height is ``N_geoid_undulation + zeta +
    sensor_h_above_sea_level``. Positions are converted to ECEF and then to ENU
    around ``local_frame_origin``.

    Parameters
    ----------
    lon, lat : array_like
        Longitude and latitude in degrees.
    local_frame_origin : dict
        Origin of the ENU frame, with the keys ``lon`` and ``lat`` (degrees)
        and ``h`` (ellipsoidal height, meters).
    N_geoid_undulation : float
        Geoid undulation (height of the mean sea level above the ellipsoid), in meters.
    zeta : float, optional
        Sea surface elevation above the mean sea level, in meters.
    sensor_h_above_sea_level : float, optional
        Height of the sensor above the sea surface, in meters (negative when
        immersed).

    Returns
    -------
    e, n, u : numpy.ndarray
        East, North and Up coordinates in meters.
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    # GPS h above WGS84 ellipsoid
    h = N_geoid_undulation + zeta + sensor_h_above_sea_level
    # WGS84 -> ECEF
    x, y, z = geodetic_to_ecef(lat, lon, h)
    # ECEF -> ENU
    lat0, lon0, h0 = (
        np.deg2rad(local_frame_origin["lat"]),
        np.deg2rad(local_frame_origin["lon"]),
        local_frame_origin["h"],
    )
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, h0)

    return e, n, u


def _enu_tir_pos(ds, local_frame_origin, N_geoid_undulation=48.5):
    """Add the ENU coordinates of the shots to a dataset (in place).

    The source is assumed to be 5 m below the sea surface. Adds the variables
    ``tir_src_e``, ``tir_src_n`` and ``tir_src_u`` (dimension ``src_datetime``).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``tir_src_lon`` and ``tir_src_lat``.
    local_frame_origin : dict
        Origin of the ENU frame (keys ``lon``, ``lat``, ``h``).
    N_geoid_undulation : float, optional
        Geoid undulation in meters.

    Returns
    -------
    None
    """
    tir_src_e, tir_src_n, tir_src_u = transform_pos_wgs84_enu(
        lon=ds.tir_src_lon,
        lat=ds.tir_src_lat,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=0,
        sensor_h_above_sea_level=-5,  # Source is approximately 5 meter below sea surface = MSL
    )
    # Add to portion
    ds["tir_src_e"] = (["src_datetime"], tir_src_e)
    ds["tir_src_n"] = (["src_datetime"], tir_src_n)
    ds["tir_src_u"] = (["src_datetime"], tir_src_u)
    # Update attribute
    ds.tir_src_e.attrs = {"units": "m", "long_name": "E"}
    ds.tir_src_n.attrs = {"units": "m", "long_name": "N"}
    ds.tir_src_u.attrs = {"units": "m", "long_name": "U"}


def get_enu_tir_pos(portion_tree, local_frame_origin, N_geoid_undulation=48.5):
    """Add the ENU shot coordinates to the ``position`` and ``feature`` groups.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree (modified in place).
    local_frame_origin : dict
        Origin of the ENU frame (keys ``lon``, ``lat``, ``h``).
    N_geoid_undulation : float, optional
        Geoid undulation in meters.

    Returns
    -------
    None
    """
    for group in ("position", "feature"):
        if group not in portion_tree.children:
            continue
        for _, node in portion_tree[group].children.items():
            ds_section = node.to_dataset()
            _enu_tir_pos(
                ds_section, local_frame_origin, N_geoid_undulation=N_geoid_undulation
            )
            # Update tree
            node.ds = ds_section


def _enu_rcv_pos(ds, local_frame_origin, N_geoid_undulation=48.5):
    """Add the ENU coordinates of the receivers to a dataset (in place).

    Adds the variables ``rcv_e``, ``rcv_n`` and ``rcv_u`` (dimension ``rcv_id``).
    The receiver height is minus its immersion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``rcv_lon``, ``rcv_lat`` and ``rcv_immersion``.
    local_frame_origin : dict
        Origin of the ENU frame (keys ``lon``, ``lat``, ``h``).
    N_geoid_undulation : float, optional
        Geoid undulation in meters.

    Returns
    -------
    None
    """
    rcv_e, rcv_n, rcv_u = transform_pos_wgs84_enu(
        lon=ds.rcv_lon,
        lat=ds.rcv_lat,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=0,
        sensor_h_above_sea_level=-ds.rcv_immersion,
    )
    # Add to portion
    ds["rcv_e"] = (["rcv_id"], rcv_e)
    ds["rcv_n"] = (["rcv_id"], rcv_n)
    ds["rcv_u"] = (["rcv_id"], rcv_u)
    # Update attribute
    ds.rcv_e.attrs = {"units": "m", "long_name": "E"}
    ds.rcv_n.attrs = {"units": "m", "long_name": "N"}
    ds.rcv_u.attrs = {"units": "m", "long_name": "U"}


def get_enu_rcv_pos(portion_tree, local_frame_origin, N_geoid_undulation=48.5):
    """Add the ENU receiver coordinates to the ``position`` and ``feature`` groups.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree (modified in place).
    local_frame_origin : dict
        Origin of the ENU frame (keys ``lon``, ``lat``, ``h``).
    N_geoid_undulation : float, optional
        Geoid undulation in meters.

    Returns
    -------
    None
    """
    for group in ("position", "feature"):
        if group not in portion_tree.children:
            continue
        for _, node in portion_tree[group].children.items():
            ds_section = node.to_dataset()
            _enu_rcv_pos(
                ds_section, local_frame_origin, N_geoid_undulation=N_geoid_undulation
            )
            # Update tree
            node.ds = ds_section


def get_vertical_dist(portion_tree, z_src=5):
    """Compute the vertical source-receiver distance for every shot and receiver.

    Two estimates are returned: one from the receiver immersion and the nominal
    source depth (WGS84), and one from the ENU "Up" coordinates. Requires
    ``get_enu_tir_pos`` and ``get_enu_rcv_pos`` to have been called.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group.
    z_src : float, optional
        Nominal source depth below the surface, in meters.

    Returns
    -------
    v_dist_wgs84 : numpy.ndarray
        Vertical distance from immersions, shape ``(n_rcv, n_shots_total)``.
    v_dist_enu : numpy.ndarray
        Vertical distance from ENU coordinates, same shape.
    """
    v_dist_wgs84 = []
    v_dist_enu = []
    for _, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        source_z_below_surface = np.ones_like(ds_pos_section.tir_src_u) * z_src
        vd_wgs84 = (
            ds_pos_section.rcv_immersion.values[:, np.newaxis]
            - source_z_below_surface[np.newaxis, :]
        )  # (nrcv, npos)
        v_dist_wgs84.append(vd_wgs84)
        vd_enu = (
            ds_pos_section.rcv_u.values[:, np.newaxis]
            - ds_pos_section.tir_src_u.values[np.newaxis, :]
        )  # (nrcv, npos)
        v_dist_enu.append(vd_enu)

    v_dist_wgs84 = np.hstack(v_dist_wgs84)
    v_dist_enu = -np.hstack(v_dist_enu)

    return v_dist_wgs84, v_dist_enu


# ======================================================================================================================
# Plot functions
# ======================================================================================================================


### Positions ###
def plot_obs_pos(ds_pos, ax=None, add_legend=True, add_title=True):
    """Plot the OBS positions in geographic coordinates (longitude / latitude).

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Dataset with ``rcv_lon`` and ``rcv_lat`` (dimension ``rcv_id``).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.
    add_legend : bool, optional
        If True, add a legend (at most 4 columns).
    add_title : bool, optional
        If True, add the title "OBS positions".

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for obs_id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=obs_id)
        ax.scatter(
            rcv.rcv_lon, rcv.rcv_lat, marker="d", label=f"{obs_id.upper()}", s=200
        )

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        _, labels = ax.get_legend_handles_labels()
        ncols = min(len(labels), 4)
        ax.legend(ncols=ncols)

    return ax


def plot_obs_pos_enu(ds_pos, ax=None, add_legend=True, add_title=True):
    """Plot the OBS positions in the local ENU frame.

    Requires ``get_enu_rcv_pos`` to have been called.

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Dataset with ``rcv_e`` and ``rcv_n`` (dimension ``rcv_id``).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.
    add_legend : bool, optional
        If True, add a legend (at most 4 columns).
    add_title : bool, optional
        If True, add the title "OBS positions".

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for obs_id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=obs_id)
        ax.scatter(rcv.rcv_e, rcv.rcv_n, marker="d", label=f"{obs_id.upper()}", s=200)

    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        _, labels = ax.get_legend_handles_labels()
        ncols = min(len(labels), 4)
        ax.legend(ncols=ncols)

    return ax


def plot_portion_tir_pos_enu(portion_tree, ax=None):
    """Plot the shot positions of every section, and the OBS, in the ENU frame.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group containing ENU coordinates (see
        ``get_enu_tir_pos`` and ``get_enu_rcv_pos``).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the axis.
    ax : matplotlib.axes.Axes
        Axis used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ds_pos_section = None
    for _, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        ax.scatter(
            ds_pos_section.tir_src_e,
            ds_pos_section.tir_src_n,
            color=color(ds_pos_section.attrs["section_id"]),
            label=f"{ds_pos_section.attrs['section_orientation']} {ds_pos_section.attrs['section_id']}",
        )

    if ds_pos_section is not None:
        # OBS positions are shared by all sections
        plot_obs_pos_enu(ds_pos_section, ax=ax)

    return fig, ax


def plot_pressure_section(ds_pressure_section, single_fig=False):
    """Plot the pressure signal of every OBS of a section.

    Parameters
    ----------
    ds_pressure_section : xarray.Dataset
        Pressure dataset of a section (variables ``pressure_<obs_id>``).
    single_fig : bool, optional
        If True, all OBS are drawn on a single axis. Otherwise one subplot per
        OBS is created, sharing the x and y axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    axs : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        The single axis if ``single_fig`` is True, otherwise a flat array with
        one axis per OBS.
    """
    if single_fig:
        plt.figure()
        for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{obs_id}"].plot(
                label=obs_id.upper(), color=color(i)
            )
        plt.legend(loc="upper right")
        return plt.gcf(), plt.gca()

    fig, axs = plt.subplots(
        nrows=ds_pressure_section.obs_id.size,
        ncols=1,
        figsize=(16, 14),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axs = axs.flatten()
    for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
        ds_pressure_section[f"pressure_{obs_id}"].plot(
            label=obs_id.upper(), color=color(i), ax=axs[i]
        )
        axs[i].legend(loc="upper right")

    xlab = axs[0].get_xlabel()
    ylab = axs[0].get_ylabel()

    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    fig.supxlabel(xlab)
    fig.supylabel(ylab)

    return fig, axs


def plot_spectrogram_section(
    ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"
):
    """Plot the spectrogram of every OBS of a section (one subplot per OBS).

    The STFT is computed with a Hann window and the power spectral density is
    displayed in dB re 1 V^2/Hz. All subplots share the same color scale (10th
    to 99th percentile of all spectrograms) and the time axis of the first OBS.

    Parameters
    ----------
    ds_pressure_section : xarray.Dataset
        Pressure dataset of a section, in volts.
    nperseg : int, optional
        Number of samples per STFT segment.
    noverlap : int, optional
        Number of overlapping samples between segments.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array with one axis per OBS.
    """
    sxx_plot = []
    tt_datetime = None
    ff = None

    for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
        pressure = ds_pressure_section[f"pressure_{obs_id}"]
        # Derive stft
        ff, tt, stft = sp.stft(
            pressure.values,
            fs=ds_pressure_section.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # V^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft) + 1e-6)  # dB re V**2 / Hz
        sxx_plot.append(sxx)

        # Datetime vector of the first OBS is used for the shared time axis
        if i == 0:
            t0 = pd.to_datetime(
                ds_pressure_section[f"time_{obs_id}"].values[0]
            ).to_pydatetime()
            tt_datetime = pd.to_datetime(t0) + pd.to_timedelta(tt, unit="s")

    # Plot
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    fig, axs = plt.subplots(
        nrows=ds_pressure_section.obs_id.size,
        ncols=1,
        figsize=(16, 14),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axs = axs.flatten()
    for i, _ in enumerate(ds_pressure_section.obs_id.values):
        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
    clabel = r"dB re 1V$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )

    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    fig.supxlabel("Time")
    fig.supylabel("Frequency [Hz]")

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    return fig, axs


def plot_portion_tir_pos(portion_tree, ax=None):
    """Plot the shot positions of every section, and the OBS, in lon / lat.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the axis.
    ax : matplotlib.axes.Axes
        Axis used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ds_pos_section = None
    for _, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        ax.scatter(
            ds_pos_section.tir_src_lon,
            ds_pos_section.tir_src_lat,
            color=color(ds_pos_section.attrs["section_id"]),
            label=f"{ds_pos_section.attrs['section_orientation']} {ds_pos_section.attrs['section_id']}",
        )

    if ds_pos_section is not None:
        # OBS positions are shared by all sections
        plot_obs_pos(ds_pos_section, ax=ax)

    return fig, ax


# ======================================================================================================================
# Feature estimation (for RTF / DCF estimation)
# ======================================================================================================================


def compute_cpsd_matrix(signal_arr, nfft, freq_band_mask):
    """Compute the cross power spectral density matrix of several signals.

    ``Sxy[i, j] = X_i * conj(X_j)`` where ``X`` is the zero-padded FFT of the
    signals, restricted to the frequency band selected by ``freq_band_mask``.
    The diagonal is the PSD of each signal and ``Sxy[j, i] = conj(Sxy[i, j])``.
    No averaging is performed (single window estimate).

    Parameters
    ----------
    signal_arr : numpy.ndarray
        Signals of the receivers, shape ``(n_rcv, n_samples)``.
    nfft : int
        FFT length (must be at least ``n_samples``).
    freq_band_mask : numpy.ndarray of bool
        Mask of the kept FFT bins, shape ``(nfft // 2 + 1,)``.

    Returns
    -------
    numpy.ndarray
        CPSD matrix of dtype complex64, shape ``(n_rcv, n_rcv, n_freq)``, where
        ``n_freq`` is the number of True values in ``freq_band_mask``.

    Raises
    ------
    ValueError
        If ``nfft`` is smaller than the number of samples (the FFT would
        silently truncate the signals).
    """
    n_samples = signal_arr.shape[1]
    if nfft < n_samples:
        raise ValueError(
            f"nfft={nfft} is smaller than the signal length ({n_samples} samples): "
            "the signals would be truncated. Use nfft >= signal length."
        )

    rfft = np.fft.rfft(signal_arr, n=nfft, axis=1)  # (nrcv, nfft // 2 + 1)
    rfft = rfft[:, freq_band_mask]  # (nrcv, nf)

    # Sxy[i, j, f] = X_i(f) * conj(X_j(f))
    sxy = rfft[:, np.newaxis, :] * np.conj(rfft[np.newaxis, :, :])
    return sxy.astype(np.complex64)


def derive_portion_cpsd(
    portion_tree,
    offset_ms=100,
    signal_duration_s=3,
    nfft=None,
    obs_ids=None,
    fmin=None,
    fmax=None,
):
    """Compute the CPSD matrix of every shot of every section and store it as features.

    For each shot, a window of ``signal_duration_s`` seconds starting
    ``offset_ms`` before the theoretical reception time is extracted on each
    OBS, centered, and the CPSD matrix is computed (see ``compute_cpsd_matrix``).
    Shots whose window is incomplete on at least one OBS are discarded.

    The result is stored in the group ``/feature/<section>`` of the tree, as the
    variable ``Sxy`` of dimensions ``(src_datetime, rcv_i, rcv_j, freq)``,
    together with the position information of the section. Requires
    ``derive_portion_propagation_time`` to have been called beforehand.

    Only the pressure windows are read from disk, but the resulting ``Sxy``
    arrays are held in memory until the tree is saved.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with the groups ``position`` and ``pressure`` (modified in
        place: the group ``feature`` is added).
    offset_ms : int, optional
        Time before the theoretical reception included in the window, in ms.
    signal_duration_s : float, optional
        Window duration, in seconds.
    nfft : int or None, optional
        FFT length. If None, the number of samples of the window is used.
    obs_ids : array_like of str or None, optional
        OBS used. If None, all OBS of the first pressure section are used.
    fmin, fmax : float or None, optional
        Frequency band kept, in Hz. If None, the lowest / highest frequency is used.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If ``tir_reception_datetime`` is missing from a position section.
    ValueError
        If ``nfft`` is too small, or if no shot has a complete window in a section.
    """
    signal_duration_ms = int(round(signal_duration_s * 1000))

    first_section_name = list(portion_tree.pressure.children)[0]
    if obs_ids is None:
        obs_ids = portion_tree.pressure[first_section_name].to_dataset().obs_id.values
    obs_ids = np.asarray(obs_ids)

    fs = portion_tree.pressure[first_section_name].to_dataset().attrs["fs"]
    n_samples_expected = int(signal_duration_s * fs)

    if nfft is None:
        nfft = n_samples_expected
    if nfft < n_samples_expected:
        raise ValueError(
            f"nfft={nfft} is smaller than the window length "
            f"({n_samples_expected} samples = {signal_duration_s} s at {fs} Hz)."
        )
    # Init freq vector
    freq = np.fft.rfftfreq(nfft, d=1 / fs)

    if fmin is None:
        fmin = np.min(freq)
    if fmax is None:
        fmax = np.max(freq)

    # Slice required frequency band
    freq_band_mask = np.logical_and(freq >= fmin, freq <= fmax)
    if not freq_band_mask.any():
        raise ValueError(
            f"No frequency bin in [{fmin}, {fmax}] Hz (frequency range: "
            f"{freq.min()} - {freq.max()} Hz, resolution {freq[1] - freq[0]:.4g} Hz)."
        )
    freq = freq[freq_band_mask]

    # 1) Iterate over portion's sections
    for name, node in portion_tree.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        # Associated ds_pos
        ds_pos_section = portion_tree.position[name].to_dataset()

        if "tir_reception_datetime" not in ds_pos_section:
            raise KeyError(
                f"Variable 'tir_reception_datetime' missing in section '{name}': "
                "call derive_portion_propagation_time() first."
            )

        Sxy_data = []
        valid_emission_dt = []
        # 2) Iterate over emissions
        for tir_emission_dt in ds_pos_section.src_datetime.values:
            tir_reception_dt = ds_pos_section["tir_reception_datetime"].sel(
                src_datetime=tir_emission_dt
            )

            # 3) Build list of signal_i
            signal_arr = []
            cropped_signal = False
            for rcv_id_i in obs_ids:
                # Reception datetime for current rcv i
                tir_reception_dt_rcvi = tir_reception_dt.sel(rcv_id=rcv_id_i)

                # Select associated data window
                signal_i = ds_pressure_section[f"pressure_{rcv_id_i}"].sel(
                    {
                        f"time_{rcv_id_i}": slice(
                            tir_reception_dt_rcvi.data
                            - np.timedelta64(offset_ms, "ms"),
                            tir_reception_dt_rcvi.data
                            + np.timedelta64(signal_duration_ms - offset_ms, "ms"),
                        )
                    }
                )

                if signal_i.size != n_samples_expected:
                    cropped_signal = True
                    break

                # Ensure signal is centered
                signal_i = signal_i - np.nanmean(signal_i)

                # Store in array
                signal_arr.append(signal_i.values)

            if cropped_signal:
                continue

            valid_emission_dt.append(tir_emission_dt)
            signal_arr = np.asarray(signal_arr)  # (nrcv, nt)

            # Store for current tir
            Sxy_data.append(compute_cpsd_matrix(signal_arr, nfft, freq_band_mask))

        if len(Sxy_data) == 0:
            raise ValueError(
                f"No shot of section '{name}' has a complete {signal_duration_s} s window "
                f"on all of the OBS {list(obs_ids)}."
            )
        Sxy_data = np.array(Sxy_data)  # (ntir, nr, nr, nf)

        # Store in ds_feature
        ds_feature = ds_pos_section.copy()
        ds_feature = ds_feature.drop_vars(
            [
                "atl_lon",
                "atl_lat",
                "atl_datetime",
                "ais_lon",
                "ais_lat",
                "ais_datetime",
                "tir_SEGD",
                "tir_ECOS",
                "tir_vessel_lon",
                "tir_vessel_lat",
            ],
            errors="ignore",
        )
        attrs = dict(ds_feature.attrs)
        attrs.pop("interpolation_time_step", None)
        attrs["description"] = "Localisation features"
        ds_feature.attrs = attrs

        # Sel only valid tir
        ds_feature = ds_feature.sel(src_datetime=valid_emission_dt)

        # Add coords rcv_i, rcv_j and freq
        ds_feature = ds_feature.assign_coords(
            dict(rcv_i=obs_ids, rcv_j=obs_ids, freq=freq)
        )  # NOTE we need to declare new coords because we cant use duplicated coords

        ds_feature["Sxy"] = (["src_datetime", "rcv_i", "rcv_j", "freq"], Sxy_data)

        # Update tree
        portion_tree[f"/feature/{name}"] = ds_feature


def get_distance_between_sections(ds_feature_section1, ds_feature_section2):
    """Compute the geodesic distance between every shot of two sections.

    Distances are computed on the WGS84 ellipsoid between the source positions
    ``tir_src_lon`` / ``tir_src_lat`` of the two sections.

    Parameters
    ----------
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of two sections (dimension ``src_datetime``).

    Returns
    -------
    numpy.ndarray
        Distances in meters, shape ``(n_emission1, n_emission2)``.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    src_lon1, src_lat1 = (
        ds_feature_section1.tir_src_lon.values,
        ds_feature_section1.tir_src_lat.values,
    )
    src_lon2, src_lat2 = (
        ds_feature_section2.tir_src_lon.values,
        ds_feature_section2.tir_src_lat.values,
    )

    dist_12 = []
    for i in range(src_lon2.size):

        # Cast to src pos shape
        lon2 = np.full(src_lon1.size, src_lon2[i])
        lat2 = np.full(src_lon1.size, src_lat2[i])

        # Compute geodesic distance
        _, _, horiz_d = geod.inv(lons1=src_lon1, lats1=src_lat1, lons2=lon2, lats2=lat2)

        dist_12.append(horiz_d)

    dist_12 = np.asarray(dist_12)  # (n_emission2, n_emission1)
    dist_12 = dist_12.T  # (n_emission1, n_emission2)

    return dist_12


def _gamma_distances_block(gamma_a, gamma_b):
    """Compute the unnormalized distances between gamma (RTF level) vectors.

    Inputs are already broadcastable: ``gamma_a`` has shape ``(nt1, 1, nf)`` and
    ``gamma_b`` has shape ``(1, nt2, nf)``. NaN values are ignored (``nansum``).

    Parameters
    ----------
    gamma_a, gamma_b : numpy.ndarray
        RTF level in dB (real).

    Returns
    -------
    dist_theta_gamma : numpy.ndarray
        Euclidean angle between the gamma vectors, rescaled to [0, 1], shape
        ``(nt1, nt2)``.
    dist_L2_gamma_raw : numpy.ndarray
        L2 distance between the gamma vectors, shape ``(nt1, nt2)``. Not yet
        normalized to [0, 1].
    """
    # L2 distance (normalized by the caller)
    dist_L2_gamma = np.sqrt(np.nansum(np.abs(gamma_a - gamma_b) ** 2, axis=-1))

    # Theta
    inner_prod = np.nansum(gamma_a * gamma_b, axis=-1)
    norm_a = np.sqrt(np.nansum(gamma_a**2, axis=-1))
    norm_b = np.sqrt(np.nansum(gamma_b**2, axis=-1))
    # Clip to [-1, 1] for stability
    cos_angle = np.clip(inner_prod / (norm_a * norm_b), -1.0, 1.0)
    dist_theta_gamma = (1 - cos_angle) / 2  # [-1, 1] to [0, 1]

    return dist_theta_gamma, dist_L2_gamma


def _feature_distances_block(pi_a, gamma_a, sij_scot_a, pi_b, gamma_b, sij_scot_b):
    """Compute the unnormalized feature distances between two sets of shots.

    Inputs are already broadcastable: ``*_a`` have shape ``(nt1, 1, nf)`` and
    ``*_b`` have shape ``(1, nt2, nf)``. NaN values are ignored (``nansum``).

    Parameters
    ----------
    pi_a, pi_b : numpy.ndarray
        Complex RTF.
    gamma_a, gamma_b : numpy.ndarray
        RTF level in dB (real).
    sij_scot_a, sij_scot_b : numpy.ndarray
        Complex SCOT-weighted normalized cross spectrum (DCF).

    Returns
    -------
    dist_theta, dist_L2_gamma_raw, dist_theta_gamma, dist_dcf : numpy.ndarray
        Distance matrices of shape ``(nt1, nt2)``. ``dist_L2_gamma_raw`` is not
        yet normalized to [0, 1].
    """
    # Compute theta with pi
    inner_prod = np.nansum(pi_a * np.conj(pi_b), axis=-1)
    norm_a = np.sqrt(np.nansum(pi_a * np.conj(pi_a), axis=-1))
    norm_b = np.sqrt(np.nansum(pi_b * np.conj(pi_b), axis=-1))
    # Clip to [0, 1] for stability
    cos_angle = np.clip(np.abs(inner_prod) / np.abs(norm_a * norm_b), 0, 1.0)
    dist_theta = 1 - cos_angle  # to [0, 1]

    # Distances on gamma (L2 is normalized by the caller)
    dist_theta_gamma, dist_L2_gamma = _gamma_distances_block(gamma_a, gamma_b)

    # DCF
    dist_dcf = np.abs(np.nansum(sij_scot_a * np.conj(sij_scot_b), axis=-1))
    dist_dcf = 1 - dist_dcf / (
        np.sqrt(np.nansum(np.abs(sij_scot_a) ** 2, axis=-1))
        * np.sqrt(np.nansum(np.abs(sij_scot_b) ** 2, axis=-1))
    )

    return dist_theta, dist_L2_gamma, dist_theta_gamma, dist_dcf


def derive_feature_distance_between_sections(features_section1, features_section2):
    """Compute the distance in feature space between all shots of two sections.

    Four distances are computed between every shot of section 1 and every shot
    of section 2, for a single receiver pair:

    - ``dist_theta``: Hermitian angle between the complex RTF vectors, in [0, 1];
    - ``dist_theta_gamma``: Euclidean angle between the RTF level (gamma)
      vectors, rescaled to [0, 1];
    - ``dist_L2_gamma``: L2 distance between the gamma vectors, divided by its
      maximum so that it lies in [0, 1];
    - ``dist_dcf``: 1 minus the normalized correlation between the DCF vectors.

    To bound the memory usage, section 1 is processed by blocks of rows so that
    the temporary arrays never exceed about ``_MAX_BROADCAST_ELEMENTS``
    elements. The results are identical to a fully broadcast computation.

    Parameters
    ----------
    features_section1, features_section2 : dict
        Features of each section, with the keys ``pi_ij`` (complex RTF),
        ``gamma_ij`` (RTF level in dB) and ``sij_scot`` (DCF), each of shape
        ``(n_emission, n_freq)``. Both sections must share the same ``n_freq``.

    Returns
    -------
    dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf : numpy.ndarray
        Distance matrices of shape ``(n_emission1, n_emission2)``.

    Raises
    ------
    ValueError
        If the two sections do not have the same number of frequency bins.
    """
    pi_ij_a, gamma_a, sij_scot_a = (
        features_section1["pi_ij"],
        features_section1["gamma_ij"],
        features_section1["sij_scot"],
    )
    pi_ij_b, gamma_b, sij_scot_b = (
        features_section2["pi_ij"],
        features_section2["gamma_ij"],
        features_section2["sij_scot"],
    )

    if pi_ij_a.shape[-1] != pi_ij_b.shape[-1]:
        raise ValueError(
            "The two sections do not have the same number of frequency bins "
            f"({pi_ij_a.shape[-1]} vs {pi_ij_b.shape[-1]}): recompute the CPSD "
            "of both sections with the same fmin / fmax / nfft."
        )

    nt1, nf = pi_ij_a.shape
    nt2 = pi_ij_b.shape[0]

    # Reshape for broadcasting
    pi_b = pi_ij_b[np.newaxis, :, :]  # (1, nt2, nf)
    gamma_b = gamma_b[np.newaxis, :, :]  # (1, nt2, nf)
    sij_scot_b = sij_scot_b[np.newaxis, :, :]  # (1, nt2, nf)

    block = max(1, _MAX_BROADCAST_ELEMENTS // max(1, nt2 * nf))

    dist_theta = np.empty((nt1, nt2))
    dist_L2_gamma = np.empty((nt1, nt2))
    dist_theta_gamma = np.empty((nt1, nt2))
    dist_dcf = np.empty((nt1, nt2))

    # 0/0 occurs for shots with no valid frequency (all NaN): the distance is then NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        for start in range(0, nt1, block):
            sl = slice(start, min(start + block, nt1))
            d_theta, d_l2, d_theta_gamma, d_dcf = _feature_distances_block(
                pi_ij_a[sl, np.newaxis, :],  # (block, 1, nf)
                gamma_a[sl, np.newaxis, :],
                sij_scot_a[sl, np.newaxis, :],
                pi_b,
                gamma_b,
                sij_scot_b,
            )
            dist_theta[sl] = d_theta
            dist_L2_gamma[sl] = d_l2
            dist_theta_gamma[sl] = d_theta_gamma
            dist_dcf[sl] = d_dcf

    # Normalize L2 to [0, 1]
    dist_L2_gamma = _normalize_by_max(dist_L2_gamma)

    return dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf


def _non_finite_to_nan(arr):
    """Replace the non finite values (inf, -inf, NaN) of an array by NaN.

    Parameters
    ----------
    arr : numpy.ndarray
        Real or complex array.

    Returns
    -------
    numpy.ndarray
        Copy of ``arr`` where every non finite element (for a complex array, an
        element with a non finite real or imaginary part) is NaN.
    """
    arr = np.array(arr, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _normalize_by_max(dist):
    """Divide a distance matrix by its maximum, ignoring NaN values.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance matrix, possibly containing NaN.

    Returns
    -------
    numpy.ndarray
        ``dist / nanmax(dist)``. If ``dist`` has no valid value, the result is
        all NaN. No warning is emitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN matrix
        dmax = np.nanmax(dist)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dist / dmax


def _center_along_freq(gamma):
    """Remove from each shot the mean of its gamma over the frequencies.

    NaN values are ignored in the mean (``np.nanmean``) and stay NaN in the
    result. A shot with only NaN stays entirely NaN.

    Parameters
    ----------
    gamma : numpy.ndarray
        RTF level in dB, shape ``(n_emission, n_freq)``.

    Returns
    -------
    numpy.ndarray
        Centered gamma, same shape.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        return gamma - np.nanmean(gamma, axis=1, keepdims=True)


def derive_gamma_distance_between_sections(gamma_section1, gamma_section2):
    """Compute the gamma distances (theta_gamma and L2) between all shots of two sections.

    Same computation as in ``derive_feature_distance_between_sections`` but
    restricted to the gamma (RTF level) vectors, which allows comparing variants
    of gamma (e.g. centered along frequency) without storing them. Processed by
    blocks of rows to bound the memory usage.

    Parameters
    ----------
    gamma_section1, gamma_section2 : numpy.ndarray
        RTF level in dB of each section, shape ``(n_emission, n_freq)``.

    Returns
    -------
    dist_theta_gamma, dist_L2_gamma : numpy.ndarray
        Distance matrices of shape ``(n_emission1, n_emission2)``;
        ``dist_L2_gamma`` is normalized by its maximum.

    Raises
    ------
    ValueError
        If the two sections do not have the same number of frequency bins.
    """
    if gamma_section1.shape[-1] != gamma_section2.shape[-1]:
        raise ValueError(
            "The two sections do not have the same number of frequency bins "
            f"({gamma_section1.shape[-1]} vs {gamma_section2.shape[-1]})."
        )
    nt1, nf = gamma_section1.shape
    nt2 = gamma_section2.shape[0]
    gamma_b = gamma_section2[np.newaxis, :, :]
    block = max(1, _MAX_BROADCAST_ELEMENTS // max(1, nt2 * nf))

    dist_theta_gamma = np.empty((nt1, nt2))
    dist_L2_gamma = np.empty((nt1, nt2))
    # 0/0 occurs for shots with no valid frequency (all NaN): the distance is then NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        for start in range(0, nt1, block):
            sl = slice(start, min(start + block, nt1))
            dist_theta_gamma[sl], dist_L2_gamma[sl] = _gamma_distances_block(
                gamma_section1[sl, np.newaxis, :], gamma_b
            )

    return dist_theta_gamma, _normalize_by_max(dist_L2_gamma)


def derive_section_features(
    ds_feature_section, fmin=5, fmax=50, rcv_ref="obs1", rcv_i="obs2"
):
    """Derive the RTF and DCF features of a section for one receiver pair.

    With ``i = rcv_i`` and ``j = rcv_ref``, for every shot and frequency:

    - RTF: ``pi_ij = S_ij / S_jj`` and its level ``gamma_ij = 20 log10 |pi_ij|``;
    - DCF: ``S_ij / sqrt(S_ii * S_jj)`` (SCOT weighting).

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset of a section, with the variable ``Sxy`` of dimensions
        ``(src_datetime, rcv_i, rcv_j, freq)``.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    rcv_ref : str, optional
        Reference receiver (``j``).
    rcv_i : str, optional
        Other receiver of the pair (``i``).

    Returns
    -------
    pi_ij : numpy.ndarray
        Complex RTF, shape ``(n_emission, n_freq)``.
    gamma_ij : numpy.ndarray
        RTF level in dB, same shape.
    sij_scot : numpy.ndarray
        Complex SCOT-weighted cross spectrum, same shape.

    Notes
    -----
    Non finite values (``inf`` / ``-inf`` / ``NaN``), which appear at frequency
    bins where a spectrum is exactly zero (division by zero, ``log10(0)``), are
    all replaced by NaN in the three outputs, so that they are consistently
    ignored by the distance computations.

    Raises
    ------
    ValueError
        If ``rcv_ref`` or ``rcv_i`` is not a receiver of the dataset.
    """
    available = list(ds_feature_section.rcv_i.values)
    for rcv in (rcv_ref, rcv_i):
        if rcv not in available:
            raise ValueError(
                f"Receiver '{rcv}' not found in the feature dataset "
                f"(available receivers: {available})."
            )

    freq_slice = slice(fmin, fmax)
    # Sii, Sjj and Sij
    sii = ds_feature_section.sel(rcv_i=rcv_i, rcv_j=rcv_i, freq=freq_slice).Sxy
    sjj = ds_feature_section.sel(rcv_i=rcv_ref, rcv_j=rcv_ref, freq=freq_slice).Sxy
    sij = ds_feature_section.sel(rcv_i=rcv_i, rcv_j=rcv_ref, freq=freq_slice).Sxy

    # Frequency bins where a spectrum is exactly zero give 0/0 or log10(0): these
    # invalid values are expected and are replaced by NaN below
    with np.errstate(divide="ignore", invalid="ignore"):
        # RTF
        pi_ij = sij.values / sjj.values
        gamma_ij = 20 * np.log10(np.abs(pi_ij))  # (nt, nf)

        # DCF
        sij_scot = sij.values / np.sqrt(sii.values * sjj.values)

    return (
        _non_finite_to_nan(pi_ij),
        _non_finite_to_nan(gamma_ij),
        _non_finite_to_nan(sij_scot),
    )


def find_min_distance_position(dt1, dt2, dist):
    """Find the pair of coordinates where a 2D distance map is minimal.

    Parameters
    ----------
    dt1, dt2 : array_like
        Coordinates along the two axes of the map, of length ``dist.shape[1]``
        and ``dist.shape[0]`` respectively (``np.meshgrid`` convention).
    dist : numpy.ndarray
        Distance map.

    Returns
    -------
    xmin, ymin : scalar
        Coordinates of the minimum.

    Notes
    -----
    The map is flattened consistently with ``np.meshgrid(dt1, dt2)``, so
    ``dist`` must have shape ``(len(dt2), len(dt1))``.
    """
    if np.shape(dist) != (len(dt2), len(dt1)):
        raise ValueError(
            f"dist has shape {np.shape(dist)} but (len(dt2), len(dt1)) = "
            f"{(len(dt2), len(dt1))} is expected."
        )
    ind_min_dist = np.argmin(dist)
    xx, yy = np.meshgrid(dt1, dt2)
    xmin, ymin = xx.flatten()[ind_min_dist], yy.flatten()[ind_min_dist]
    return xmin, ymin


def derive_features_array(portion, fmin=5, fmax=50, rcv_ref="obs1", rcv_i="obs2"):
    """Stack the RTF and DCF features of all sections of a portion.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree with a ``feature`` group.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    rcv_ref : str, optional
        Reference receiver.
    rcv_i : str, optional
        Other receiver of the pair.

    Returns
    -------
    data_pi : numpy.ndarray
        Complex RTF of all shots, shape ``(sum of n_emission, n_freq)``.
    data_gamma : numpy.ndarray
        RTF level in dB, same shape.
    data_sij_scot : numpy.ndarray
        DCF, same shape.
    """
    pi_ij_arr = []
    gamma_arr = []
    sij_scot_arr = []
    for _, feature_node in portion.feature.children.items():
        ds_feature_section = feature_node.to_dataset()

        # Derive feature for current section
        pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
            ds_feature_section=ds_feature_section,
            fmin=fmin,
            fmax=fmax,
            rcv_ref=rcv_ref,
            rcv_i=rcv_i,
        )
        pi_ij_arr.append(pi_ij_a)
        gamma_arr.append(gamma_a)
        sij_scot_arr.append(sij_scot_a)

    # Build data (n_sample, n_features) = (sum(nti), nfreq)
    data_pi = np.concatenate(pi_ij_arr, axis=0)
    data_gamma = np.concatenate(gamma_arr, axis=0)
    data_sij_scot = np.concatenate(sij_scot_arr, axis=0)

    return data_pi, data_gamma, data_sij_scot


# ======================================================================================================================
# Derive features
# ======================================================================================================================


def distances_single_intersection(
    ds_feature_section1, ds_feature_section2, compare_centered_gamma=False
):
    """Compute all distances between the shots of two intersecting sections.

    The geodesic distance between shots ("real space") and, for each receiver
    pair ``(rcv_id_feature, reference receiver)``, the four feature distances of
    ``derive_feature_distance_between_sections`` are computed.

    Optionally, the gamma distances are also computed with a ``gamma`` centered
    along frequency (mean of each shot removed, NaN values ignored in the mean,
    see ``_center_along_freq``). The centered gamma is computed on the fly and
    is never stored. ``theta``
    (complex RTF) and ``dcf`` do not depend on ``gamma`` and are therefore
    unchanged by the centering.

    Parameters
    ----------
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of two sections, with the variables ``pi_ij``,
        ``gamma_ij`` and ``sij_scot`` of dimensions
        ``(rcv_id_feature, src_datetime, freq)`` (see
        ``apply_mfp_strategy_to_portion_intersections``).
    compare_centered_gamma : bool, optional
        If True, also compute ``dist_theta_gamma_centered`` and
        ``dist_L2_gamma_centered``.

    Returns
    -------
    dict
        Keys ``dist_in_real_space`` (shape ``(n_emission1, n_emission2)``, m)
        and ``dist_theta``, ``dist_theta_gamma``, ``dist_L2_gamma``,
        ``dist_dcf`` (shape ``(n_rcv, n_emission1, n_emission2)``). If
        ``compare_centered_gamma`` is True, also ``dist_theta_gamma_centered``
        and ``dist_L2_gamma_centered`` (same shape).
    """
    # Compute real distance in meters between emissions from section 1 to emission from section 2
    dist_in_real_space_12 = get_distance_between_sections(
        ds_feature_section1, ds_feature_section2
    )

    dist_theta_arr = []
    dist_theta_gamma_arr = []
    dist_L2_gamma_arr = []
    dist_dcf_arr = []
    dist_theta_gamma_c_arr = []
    dist_L2_gamma_c_arr = []

    for rcv_id_i in ds_feature_section1.rcv_id_feature.values:

        ds_feature_section1_rcv_i = ds_feature_section1.sel(rcv_id_feature=rcv_id_i)
        ds_feature_section2_rcv_i = ds_feature_section2.sel(rcv_id_feature=rcv_id_i)
        # Extract features
        features_section1 = dict(
            pi_ij=ds_feature_section1_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_section1_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_section1_rcv_i.sij_scot.values,
        )
        features_section2 = dict(
            pi_ij=ds_feature_section2_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_section2_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_section2_rcv_i.sij_scot.values,
        )
        # Compute distances between section 1 and 2
        dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf = (
            derive_feature_distance_between_sections(
                features_section1, features_section2
            )
        )

        # Store
        dist_theta_arr.append(dist_theta)
        dist_theta_gamma_arr.append(dist_theta_gamma)
        dist_L2_gamma_arr.append(dist_L2_gamma)
        dist_dcf_arr.append(dist_dcf)

        # Same gamma distances with gamma centered along frequency (not stored)
        if compare_centered_gamma:
            gamma1_c = _center_along_freq(features_section1["gamma_ij"])
            gamma2_c = _center_along_freq(features_section2["gamma_ij"])
            d_theta_gamma_c, d_L2_gamma_c = derive_gamma_distance_between_sections(
                gamma1_c, gamma2_c
            )
            dist_theta_gamma_c_arr.append(d_theta_gamma_c)
            dist_L2_gamma_c_arr.append(d_L2_gamma_c)

    # Convert to arrays and group (nrcv, n_emission1, n_emission2)
    distance_section12 = dict(
        dist_in_real_space=dist_in_real_space_12,
        dist_theta=np.asarray(dist_theta_arr),
        dist_theta_gamma=np.asarray(dist_theta_gamma_arr),
        dist_L2_gamma=np.asarray(dist_L2_gamma_arr),
        dist_dcf=np.asarray(dist_dcf_arr),
    )
    if compare_centered_gamma:
        distance_section12["dist_theta_gamma_centered"] = np.asarray(
            dist_theta_gamma_c_arr
        )
        distance_section12["dist_L2_gamma_centered"] = np.asarray(dist_L2_gamma_c_arr)

    return distance_section12


def all_features_single_section(ds_feature_section, rcv_ref="obs1", fmin=5, fmax=50):
    """Derive the RTF and DCF features of a section for all receivers but the reference.

    The RTF level ``gamma`` is NOT centered along the frequency axis (see
    ``distances_single_intersection`` to compare with a centered ``gamma``).

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset of a section (variable ``Sxy``).
    rcv_ref : str, optional
        Reference receiver.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.

    Returns
    -------
    dict
        ``rcv_id`` (list of the non-reference receivers) and the arrays
        ``pi_ij``, ``gamma_ij`` and ``sij_scot`` of shape
        ``(n_rcv - 1, n_emission, n_freq)``.

    Raises
    ------
    ValueError
        If ``rcv_ref`` is not a receiver of the dataset.
    """
    rcv_i_arr = list(ds_feature_section.rcv_i.values)
    if rcv_ref not in rcv_i_arr:
        raise ValueError(
            f"Reference receiver '{rcv_ref}' not found in the feature dataset "
            f"(available receivers: {rcv_i_arr})."
        )
    rcv_i_arr.remove(rcv_ref)

    pi_1_arr = []
    gamma_1_arr = []
    sij_scot_1_arr = []

    for rcv_id_i in rcv_i_arr:
        # Compute features (RTF, DCF) of the section
        pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
            ds_feature_section=ds_feature_section,
            fmin=fmin,
            fmax=fmax,
            rcv_ref=rcv_ref,
            rcv_i=rcv_id_i,
        )

        # Store
        pi_1_arr.append(pi_ij_a)
        gamma_1_arr.append(gamma_a)
        sij_scot_1_arr.append(sij_scot_a)

    # Convert to arrays and group (nrcv, n_emission, nfreq)
    features_section = dict(
        rcv_id=rcv_i_arr,
        pi_ij=np.asarray(pi_1_arr),
        gamma_ij=np.asarray(gamma_1_arr),
        sij_scot=np.asarray(sij_scot_1_arr),
    )

    return features_section


def build_mfp_intersection_dataset(
    distance_section12,
    ds_feature_section1,
    ds_feature_section2,
):
    """Build the dataset holding the MFP ambiguity surfaces of one intersection.

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``.
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of the two intersecting sections (section 1 is the
        meridional one, section 2 the zonal one).

    Returns
    -------
    xarray.Dataset
        Dataset with the receiver information, the shot positions of both
        sections, the real distance ``d_real`` (dimensions
        ``(src_datetime_section1, src_datetime_section2)``) and the feature
        distances ``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and ``d_dcf``
        (dimensions ``(rcv_id_feature, src_datetime_section1,
        src_datetime_section2)``). The attribute values that are missing from
        the feature datasets are set to ``"unknown"``.
    """
    attrs1 = ds_feature_section1.attrs

    ds_mfp = xr.Dataset(
        data_vars=dict(
            rcv_lon=ds_feature_section1.rcv_lon,
            rcv_lat=ds_feature_section1.rcv_lat,
            rcv_immersion=ds_feature_section1.rcv_immersion,
            tir_src_lon_section1=(
                ["src_datetime_section1"],
                ds_feature_section1.tir_src_lon.values,
            ),
            tir_src_lat_section1=(
                ["src_datetime_section1"],
                ds_feature_section1.tir_src_lat.values,
            ),
            tir_src_lon_section2=(
                ["src_datetime_section2"],
                ds_feature_section2.tir_src_lon.values,
            ),
            tir_src_lat_section2=(
                ["src_datetime_section2"],
                ds_feature_section2.tir_src_lat.values,
            ),
            d_real=(
                ["src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_in_real_space"],
            ),
            d_theta=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_theta"],
            ),
            d_theta_gamma=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_theta_gamma"],
            ),
            d_L2_gamma=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_L2_gamma"],
            ),
            d_dcf=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_dcf"],
            ),
        ),
        coords=dict(
            rcv_id_feature=ds_feature_section1.rcv_id_feature.values,
            src_datetime_section1=ds_feature_section1.src_datetime.values,
            src_datetime_section2=ds_feature_section2.src_datetime.values,
        ),
        attrs=dict(
            description="MFP result for a single intersection",
            campagne=attrs1.get("campagne", "unknown"),
            section1=attrs1.get("section_label", "unknown"),
            section2=ds_feature_section2.attrs.get("section_label", "unknown"),
            reference_receiver=attrs1.get("reference_receiver", "unknown"),
            obs_position=attrs1.get("obs_position", "unknown"),
            obs_position_filepath=attrs1.get("obs_position_filepath", "unknown"),
        ),
    )

    # Optional comparison with gamma centered along frequency
    for var, key, long_name, description in (
        (
            "d_theta_gamma_centered",
            "dist_theta_gamma_centered",
            r"$\theta_{\gamma}$ (centered)",
            "Euclidean angle between RTF gamma vectors centered along frequency",
        ),
        (
            "d_L2_gamma_centered",
            "dist_L2_gamma_centered",
            r"$L_{2_{\gamma}}$ (centered)",
            "Normalized euclidean distance (L2) between RTF gamma vectors centered along frequency",
        ),
    ):
        if key in distance_section12:
            ds_mfp[var] = (
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12[key],
            )
            ds_mfp[var].attrs = dict(
                long_name=long_name, description=description, minval=0, maxval=1
            )

    # Add attrs to variables
    ds_mfp.d_real.attrs = dict(
        units="m",
        long_name="Distance",
    )
    ds_mfp.d_theta.attrs = dict(
        long_name=r"$\theta_{\Pi}$",
        description="Hermitian angle between RTF vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_theta_gamma.attrs = dict(
        long_name=r"$\theta_{\gamma}$",
        description="Euclidean angle between RTF gamma vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_L2_gamma.attrs = dict(
        long_name=r"$L_{2_{\gamma}}$",
        description="Normalized euclidean distance (L2) between RTF gamma vectors",
        minval=0,
        maxval=1,
    )
    ds_mfp.d_dcf.attrs = dict(
        long_name=r"$C_{el}$",
        description="Normalized intercorrelation between intercorrelation vectors (DCF)",
        minval=0,
        maxval=1,
    )

    return ds_mfp


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


def apply_mfp_strategy_to_portion_intersections(
    portion,
    rcv_ref="obs1",
    fmin=5,
    fmax=50,
    verbose=False,
    compare_centered_gamma=False,
    intersections=None,
):
    """Compute the MFP ambiguity surfaces for meridional / zonal intersections.

    Step 1: the intersections to process are selected (see ``select_intersections``).
    Step 2: the RTF and DCF features of the sections involved in these
    intersections are derived (``all_features_single_section``) and stored in
    the ``feature`` group (variables ``pi_ij``, ``gamma_ij``, ``sij_scot``).
    Sections that are not involved are left untouched.
    Step 3: each selected intersection (meridional section, label starting with
    ``"sn"``, x zonal section, label starting with ``"oe"``) is processed and
    the result is stored in ``/mfp_intersection/<meridional>/<zonal>``.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree with a ``feature`` group (see ``derive_portion_cpsd``);
        modified in place.
    rcv_ref : str, optional
        Reference receiver.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    verbose : bool, optional
        If True, print the progress.
    compare_centered_gamma : bool, optional
        If True, the intersection datasets also contain ``d_theta_gamma_centered``
        and ``d_L2_gamma_centered`` (distances with ``gamma`` centered along
        frequency). The centered gamma itself is not stored.
    intersections : list of tuple of str or None, optional
        Intersections to process, as ``(meridional_label, zonal_label)`` pairs,
        e.g. ``[("sn0", "oe1"), ("sn0", "oe3")]``. If None (default), all
        intersections are processed.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the ``feature`` group is missing, if the portion has no meridional
        or no zonal section (no intersection can be built), or if
        ``intersections`` contains an invalid pair (see ``select_intersections``).
    """
    if "feature" not in portion.children:
        raise ValueError(
            "No 'feature' group in the portion: the CPSD has not been computed yet. "
            "Run with compute_cpsds = True first."
        )

    # Step 1 : list available sections and select the intersections to process
    sections = list(portion.feature.children)
    intersections_to_process = select_intersections(sections, intersections)
    sections_needed = {sec for pair in intersections_to_process for sec in pair}

    # Step 2 : derive the features of the needed sections only (avoids duplicated
    # calls to derive_section_features)
    for section_name, feature_node in portion.feature.children.items():
        if section_name not in sections_needed:
            continue
        if verbose:
            print(f"    Features of section '{section_name}'")
        ds_feature_section = feature_node.to_dataset()

        # Slice frequency band
        ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

        # Derive features
        features_section = all_features_single_section(
            ds_feature_section=ds_feature_section,
            rcv_ref=rcv_ref,
            fmin=fmin,
            fmax=fmax,
        )  # (nrcv, n_emission, nfreq)

        # Add coords
        ds_feature_section["rcv_id_feature"] = features_section["rcv_id"]

        # Store in dataset
        for var in ("pi_ij", "gamma_ij", "sij_scot"):
            ds_feature_section[var] = (
                ["rcv_id_feature", "src_datetime", "freq"],
                features_section[var],
            )

        # Add attributes
        ds_feature_section.attrs["reference_receiver"] = rcv_ref

        # Update portion
        feature_node.ds = ds_feature_section

    # Step 3 : intersections are between a meridional and a zonal section
    for i_inter, (section_name1, section_name2) in enumerate(intersections_to_process):
        if verbose:
            print(
                f"    Intersection '{section_name1}' x '{section_name2}' "
                f"({i_inter + 1}/{len(intersections_to_process)})"
            )
        ds_feature_section1 = portion.feature[section_name1].to_dataset()
        ds_feature_section2 = portion.feature[section_name2].to_dataset()

        # Compute distance (in real space and in features space)
        distance_section12 = distances_single_intersection(
            ds_feature_section1=ds_feature_section1,
            ds_feature_section2=ds_feature_section2,
            compare_centered_gamma=compare_centered_gamma,
        )

        # Build dataset to save
        ds_mfp = build_mfp_intersection_dataset(
            distance_section12=distance_section12,
            ds_feature_section1=ds_feature_section1,
            ds_feature_section2=ds_feature_section2,
        )

        # Add to portion
        new_path = f"/mfp_intersection/{section_name1}/{section_name2}"
        portion[new_path] = ds_mfp


# ======================================================================================================================
# Analysis of the results (position error of the estimated intersections)
# ======================================================================================================================

# Distance types compared in the results analysis, in display order
DISTANCE_TYPES = (
    "theta",
    "theta_gamma",
    "L2_gamma",
    "dcf",
    "theta_gamma_centered",
    "L2_gamma_centered",
)
DISTANCE_TYPE_LABELS = {
    "theta": r"$\theta_{\Pi}$",
    "theta_gamma": r"$\theta_{\gamma}$",
    "L2_gamma": r"$L_{2_{\gamma}}$",
    "dcf": r"$C_{el}$",
    "theta_gamma_centered": r"$\theta_{\gamma}$ (centered)",
    "L2_gamma_centered": r"$L_{2_{\gamma}}$ (centered)",
}
# Ways of computing the position error of an estimated intersection
POSITION_ERROR_MODES = ("section1", "section2", "both")


def get_distance_type_label(distance_type):
    """Return the display label of a distance type.

    Parameters
    ----------
    distance_type : str
        One of ``DISTANCE_TYPES`` (e.g. ``"theta"``, ``"dcf"``).

    Returns
    -------
    str
        Label (mathtext) used in the figures, or ``distance_type`` itself if it
        is unknown.
    """
    return DISTANCE_TYPE_LABELS.get(distance_type, distance_type)


def select_distance_types(available, requested=None):
    """Choose the distance types to analyse among the available ones.

    Parameters
    ----------
    available : iterable of str
        Distance types present in the results.
    requested : list of str or None, optional
        Distance types wished by the user. If None, all the available ones are
        returned.

    Returns
    -------
    list of str
        Selected distance types: in the order of ``requested`` (duplicates
        removed) or, if ``requested`` is None, in the canonical order of
        ``DISTANCE_TYPES``.

    Raises
    ------
    ValueError
        If a requested type is not available, or if the selection is empty.
    """
    ordered = [key for key in DISTANCE_TYPES if key in set(available)]
    if requested is None:
        selected = ordered
    else:
        selected = list(dict.fromkeys(requested))
        unknown = [key for key in selected if key not in ordered]
        if unknown:
            raise ValueError(
                f"Distance type(s) {unknown} not available in the results. "
                f"Available distance types: {ordered}."
            )
    if not selected:
        raise ValueError(
            f"No distance type to analyse (available distance types: {ordered})."
        )
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
    if "mfp_intersection" not in portion.children:
        return []
    return [
        (section1, section2)
        for section1, node in portion.mfp_intersection.children.items()
        for section2 in node.children
    ]


@functools.lru_cache(maxsize=None)
def _get_wgs84_geod():
    """Return a (cached) pyproj ``Geod`` object on the WGS84 ellipsoid."""
    from pyproj import Geod

    return Geod(ellps="WGS84")


def _geodesic_distance_m(lon1, lat1, lon2, lat2):
    """Geodesic distance between two points of the WGS84 ellipsoid.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Longitudes and latitudes in degrees.

    Returns
    -------
    float
        Distance in meters, NaN if it cannot be computed (invalid coordinates).
    """
    _, _, dist = _get_wgs84_geod().inv(lon1, lat1, lon2, lat2)
    dist = float(dist)
    return dist if np.isfinite(dist) else np.nan


def _argmin_2d(dist):
    """Index of the minimum of a 2D array, ignoring NaN and infinite values.

    Parameters
    ----------
    dist : numpy.ndarray
        2D distance matrix.

    Returns
    -------
    tuple of int or None
        ``(row, column)`` of the first minimum, or None if the matrix has no
        finite value.
    """
    valid = np.isfinite(dist)
    if not valid.any():
        return None
    flat_idx = np.argmin(np.where(valid, dist, np.inf))
    return np.unravel_index(flat_idx, dist.shape)


def localization_errors_from_distances(
    distance_section12, lon1, lat1, lon2, lat2, error_mode="section1"
):
    """Position error of the estimated intersection, for each distance type and receiver pair.

    The true intersection is the pair of shots ``(i1, i2)`` minimizing the
    real-space distance. For each distance type and receiver pair, the estimated
    intersection is the pair of shots ``(j1, j2)`` minimizing the feature
    distance (NaN values are ignored). The position error is the geodesic
    distance, on the WGS84 ellipsoid, between the true and the estimated shots:

    - ``"section1"``: shots ``i1`` and ``j1`` on section 1 (meridional section);
    - ``"section2"``: shots ``i2`` and ``j2`` on section 2 (zonal section);
    - ``"both"``: ``sqrt(error_section1**2 + error_section2**2)``.

    Section 1 being meridional, its error is essentially along the North-South
    axis, and the error of the zonal section 2 along the East-West axis: the
    ``"both"`` mode is thus the only one accounting for both directions.

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``: key
        ``dist_in_real_space`` (shape ``(n_emission1, n_emission2)``) and, for
        each available distance type, ``dist_<type>`` (shape
        ``(n_rcv, n_emission1, n_emission2)``).
    lon1, lat1 : array_like
        Shot positions of section 1, in degrees, shape ``(n_emission1,)``.
    lon2, lat2 : array_like
        Shot positions of section 2, in degrees, shape ``(n_emission2,)``.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed (see above).

    Returns
    -------
    dict of numpy.ndarray
        One array of shape ``(n_rcv,)`` per available distance type (keys of
        ``DISTANCE_TYPES``), position errors in meters. NaN when the feature
        distance has no valid value for the receiver pair.

    Raises
    ------
    ValueError
        If ``error_mode`` is unknown, if the shapes are inconsistent, or if the
        real-space distance has no valid value.
    """
    if error_mode not in POSITION_ERROR_MODES:
        raise ValueError(
            f"Unknown error_mode '{error_mode}'. Valid modes: {list(POSITION_ERROR_MODES)}."
        )

    d_real = np.asarray(distance_section12["dist_in_real_space"], dtype=float)
    lon1, lat1, lon2, lat2 = (
        np.asarray(arr, dtype=float) for arr in (lon1, lat1, lon2, lat2)
    )
    if (
        d_real.ndim != 2
        or lon1.shape != (d_real.shape[0],)
        or lat1.shape != (d_real.shape[0],)
        or lon2.shape != (d_real.shape[1],)
        or lat2.shape != (d_real.shape[1],)
    ):
        raise ValueError(
            f"Inconsistent shapes: real-space distance {d_real.shape}, section 1 positions "
            f"{lon1.shape}/{lat1.shape}, section 2 positions {lon2.shape}/{lat2.shape}."
        )

    target = _argmin_2d(d_real)
    if target is None:
        raise ValueError(
            "The real-space distance has no valid value: the true intersection "
            "cannot be determined."
        )
    i1_t, i2_t = target

    errors = {}
    for key in DISTANCE_TYPES:
        dist = distance_section12.get(f"dist_{key}")
        if dist is None:
            continue
        dist = np.asarray(dist, dtype=float)
        if dist.ndim != 3 or dist.shape[1:] != d_real.shape:
            raise ValueError(
                f"Distance 'dist_{key}' has shape {dist.shape}, expected "
                f"(n_rcv, {d_real.shape[0]}, {d_real.shape[1]})."
            )

        err = np.full(dist.shape[0], np.nan)
        for i_rcv in range(dist.shape[0]):
            guess = _argmin_2d(dist[i_rcv])
            if guess is None:
                continue
            j1, j2 = guess

            err1 = err2 = np.nan
            if error_mode in ("section1", "both"):
                err1 = _geodesic_distance_m(lon1[i1_t], lat1[i1_t], lon1[j1], lat1[j1])
            if error_mode in ("section2", "both"):
                err2 = _geodesic_distance_m(lon2[i2_t], lat2[i2_t], lon2[j2], lat2[j2])

            if error_mode == "section1":
                err[i_rcv] = err1
            elif error_mode == "section2":
                err[i_rcv] = err2
            else:
                err[i_rcv] = np.hypot(err1, err2)
        errors[key] = err

    return errors


def localization_errors_from_mfp_dataset(ds_mfp_intersection, error_mode="section1"):
    """Position errors of a stored intersection (dataset built by ``build_mfp_intersection_dataset``).

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with the variables ``d_real`` and
        ``d_<type>``, and the shot positions ``tir_src_lon_section1``,
        ``tir_src_lat_section1``, ``tir_src_lon_section2``,
        ``tir_src_lat_section2``.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed, see
        ``localization_errors_from_distances``.

    Returns
    -------
    errors : dict of numpy.ndarray
        Position errors in meters, one array of shape ``(n_rcv,)`` per
        distance type present in the dataset.
    rcv_ids : list of str
        Receiver of each error (``rcv_id_feature`` coordinate).
    """
    rcv_dim = "rcv_id_feature"
    dim1, dim2 = "src_datetime_section1", "src_datetime_section2"

    distance_section12 = {
        "dist_in_real_space": ds_mfp_intersection["d_real"].transpose(dim1, dim2).values
    }
    for key in DISTANCE_TYPES:
        var = f"d_{key}"
        if var in ds_mfp_intersection:
            distance_section12[f"dist_{key}"] = (
                ds_mfp_intersection[var].transpose(rcv_dim, dim1, dim2).values
            )

    errors = localization_errors_from_distances(
        distance_section12,
        lon1=ds_mfp_intersection["tir_src_lon_section1"].values,
        lat1=ds_mfp_intersection["tir_src_lat_section1"].values,
        lon2=ds_mfp_intersection["tir_src_lon_section2"].values,
        lat2=ds_mfp_intersection["tir_src_lat_section2"].values,
        error_mode=error_mode,
    )
    rcv_ids = [str(rcv_id) for rcv_id in ds_mfp_intersection[rcv_dim].values]

    return errors, rcv_ids


def errors_to_records(errors, rcv_ids, reference_rcv, **extra):
    """Convert position errors to a list of records (one per distance type and receiver pair).

    Parameters
    ----------
    errors : dict of numpy.ndarray
        Position errors per distance type, as returned by
        ``localization_errors_from_distances``.
    rcv_ids : sequence of str
        Receiver of each error.
    reference_rcv : str
        Reference receiver of the pairs ``(rcv_id, reference_rcv)``.
    **extra
        Additional fields copied in each record (e.g. ``intersection="sn2 x oe6"``).

    Returns
    -------
    list of dict
        Records with the keys of ``extra`` and ``distance_type``, ``rcv_id``,
        ``rcv_pair`` (``"(<rcv_id>, <reference_rcv>)"``) and ``error_m``. They
        can be concatenated and given to ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the number of errors does not match the number of receivers.
    """
    records = []
    for key, err in errors.items():
        if len(err) != len(rcv_ids):
            raise ValueError(
                f"{len(err)} errors for distance type '{key}' but {len(rcv_ids)} receivers."
            )
        for rcv_id, error in zip(rcv_ids, err):
            records.append(
                dict(
                    **extra,
                    distance_type=key,
                    rcv_id=str(rcv_id),
                    rcv_pair=f"({rcv_id}, {reference_rcv})",
                    error_m=float(error),
                )
            )
    return records


def build_frequency_bands(fmin, fmax, bandwidth):
    """Split ``[fmin, fmax]`` in frequency bands aligned on multiples of ``bandwidth``.

    The band limits are ``fmin``, the multiples of ``bandwidth`` strictly
    between ``fmin`` and ``fmax``, and ``fmax``. The first and last bands can
    therefore be narrower than ``bandwidth`` (e.g. ``[10, 400]`` and
    ``[400, 800]`` for ``fmin=10``, ``fmax=800`` and ``bandwidth=400``). Bands
    are contiguous and cover the whole interval.

    Parameters
    ----------
    fmin, fmax : float
        Limits of the interval, in Hz.
    bandwidth : float
        Nominal width of the bands, in Hz.

    Returns
    -------
    list of tuple of float
        ``(f_low, f_high)`` of each band, in increasing order.

    Raises
    ------
    ValueError
        If ``bandwidth`` is not positive or if ``fmax <= fmin``.
    """
    if bandwidth <= 0:
        raise ValueError(f"The frequency bandwidth must be positive (got {bandwidth}).")
    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must be greater than fmin ({fmin}).")

    k_first = int(np.floor(fmin / bandwidth)) + 1
    k_last = int(np.ceil(fmax / bandwidth))
    inner = [
        k * bandwidth for k in range(k_first, k_last) if fmin < k * bandwidth < fmax
    ]
    edges = [fmin, *inner, fmax]

    return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def frequency_band_label(f_low, f_high):
    """Label of a frequency band, e.g. ``"10-400"`` (Hz)."""
    return f"{f_low:g}-{f_high:g}"


def slice_frequency_band(ds_feature_section, f_low, f_high, include_upper=False):
    """Select the frequency bins of a band, without counting a bin in two adjacent bands.

    ``xarray`` label slicing (``sel(freq=slice(a, b))``) includes both limits,
    so a bin lying exactly on a limit would belong to two adjacent bands. Here
    the band is ``[f_low, f_high[`` unless ``include_upper`` is True.

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Dataset with a ``freq`` dimension.
    f_low, f_high : float
        Limits of the band, in Hz.
    include_upper : bool, optional
        If True, the bins equal to ``f_high`` are kept (use it for the last band
        of a series).

    Returns
    -------
    xarray.Dataset
        Dataset restricted to the band; its ``freq`` dimension can be empty.
    """
    freq = np.asarray(ds_feature_section["freq"].values)
    mask = freq >= f_low
    mask &= (freq <= f_high) if include_upper else (freq < f_high)
    return ds_feature_section.isel(freq=np.flatnonzero(mask))


def summarize_localization_errors(df_errors, success_threshold_m=100.0):
    """Statistics of the position errors, per distance type and receiver pair.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and
        ``error_m`` (one row per intersection), see ``errors_to_records``.
    success_threshold_m : float, optional
        An intersection is successfully localized if its error is not larger
        than this threshold, in meters.

    Returns
    -------
    pandas.DataFrame
        One row per ``(distance_type, rcv_pair)``, in order of appearance, with
        the columns ``n_intersections``, ``n_valid`` (errors that could be
        computed), ``mean_error_m``, ``median_error_m``, ``std_error_m`` (not
        corrected for the degrees of freedom), ``rms_error_m``, ``p90_error_m``
        and ``success_rate``. The statistics ignore the invalid errors (NaN),
        except ``success_rate`` where they count as failures
        (``success_rate = n_success / n_intersections``).

    Raises
    ------
    ValueError
        If ``df_errors`` is empty.
    """
    if len(df_errors) == 0:
        raise ValueError("No position error to summarize.")

    def _rms(x):
        x = x.dropna()
        return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan

    def _std(x):
        return x.std(ddof=0)

    def _p90(x):
        return x.quantile(0.9)

    def _success_rate(x):
        return float((x <= success_threshold_m).sum()) / len(x)

    summary = (
        df_errors.groupby(["distance_type", "rcv_pair"], sort=False)["error_m"]
        .agg(
            n_intersections="size",
            n_valid="count",
            mean_error_m="mean",
            median_error_m="median",
            std_error_m=_std,
            rms_error_m=_rms,
            p90_error_m=_p90,
            success_rate=_success_rate,
        )
        .reset_index()
    )
    return summary


def pivot_error_summary(summary, value="median_error_m"):
    """Table of one statistic: receiver pairs in rows, distance types in columns.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    value : str, optional
        Column of ``summary`` to tabulate (e.g. ``"median_error_m"`` or
        ``"success_rate"``).

    Returns
    -------
    pandas.DataFrame
        Pivot table, rows and columns in order of appearance in ``summary``.
    """
    rows = list(summary["rcv_pair"].drop_duplicates())
    cols = list(summary["distance_type"].drop_duplicates())
    table = summary.pivot(index="rcv_pair", columns="distance_type", values=value)
    return table.reindex(index=rows, columns=cols)


def best_rcv_pair_per_distance(summary, value="median_error_m", minimize=True):
    """Best receiver pair of each distance type.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    value : str, optional
        Statistic used to rank the pairs.
    minimize : bool, optional
        True if a lower value is better (errors), False otherwise (success rate).

    Returns
    -------
    dict
        ``{distance_type: (rcv_pair, value)}``; ``(None, nan)`` if the statistic
        is not available for any pair.
    """
    table = pivot_error_summary(summary, value)
    best = {}
    for distance_type in table.columns:
        col = table[distance_type].to_numpy(dtype=float)
        if not np.isfinite(col).any():
            best[distance_type] = (None, np.nan)
            continue
        idx = np.nanargmin(col) if minimize else np.nanargmax(col)
        best[distance_type] = (table.index[idx], float(col[idx]))
    return best


def select_and_order_distance_types(df_errors, requested=None):
    """Keep, and order, the distance types to analyse in a table of position errors.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with a ``distance_type`` column.
    requested : list of str or None, optional
        Distance types wished by the user (None: all the available ones), see
        ``select_distance_types``.

    Returns
    -------
    df_selected : pandas.DataFrame
        Records of the selected distance types, sorted by distance type in the
        order of ``distance_types`` (the order of the other rows is preserved).
    distance_types : list of str
        Selected distance types.

    Raises
    ------
    ValueError
        If a requested type is not available, or if the selection is empty.
    """
    distance_types = select_distance_types(
        df_errors["distance_type"].unique(), requested
    )
    order = {key: i for i, key in enumerate(distance_types)}
    df_selected = df_errors[df_errors["distance_type"].isin(distance_types)]
    df_selected = df_selected.sort_values(
        "distance_type", key=lambda col: col.map(order), kind="stable"
    ).reset_index(drop=True)
    return df_selected, distance_types


def _finalize_figure(fig, title=None):
    """Set the title of a figure (if any) and adjust its layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    title : str or None, optional
        Title of the figure.
    """
    if title:
        fig.suptitle(title)
    # A layout engine already set (e.g. constrained layout enabled in the
    # matplotlib rcParams) is kept: switching engine fails when a colorbar exists.
    if fig.get_layout_engine() is not None:
        return
    if title:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()


def _make_panel_grid(n_panels, max_cols=3, panel_size=(5.5, 4.5), **subplots_kwargs):
    """Create a grid of subplots with at least ``n_panels`` axes; the extra ones are hidden.

    Parameters
    ----------
    n_panels : int
        Number of panels needed.
    max_cols : int, optional
        Maximum number of columns.
    panel_size : tuple of float, optional
        Size of one panel ``(width, height)`` in inches.
    **subplots_kwargs
        Passed to ``matplotlib.pyplot.subplots`` (e.g. ``sharey=True``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of ``n_panels`` visible axes (followed by hidden ones).
    """
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        **subplots_kwargs,
    )
    axs = axs.flatten()
    for ax in axs[n_panels:]:
        ax.set_visible(False)
    return fig, axs


def plot_error_boxplots_per_rcv_pair(df_errors, distance_types, title=None):
    """Boxplots of the position error versus the distance type, one panel per receiver pair.

    Each box summarizes the errors of all the analysed intersections; the
    individual errors are drawn on top as points. Invalid errors (NaN) are not
    drawn.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and ``error_m``.
    distance_types : list of str
        Distance types to display, in the order of the x axis.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    rcv_pairs = list(df_errors["rcv_pair"].drop_duplicates())
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(rcv_pairs),
        figsize=(16, 8),
        sharey=True,
        squeeze=False,
    )
    axs = axs[0]
    # rng = np.random.default_rng(0)  # reproducible jitter

    for i_rcv, rcv_pair in enumerate(rcv_pairs):
        ax = axs[i_rcv]
        df_pair = df_errors[df_errors["rcv_pair"] == rcv_pair]
        for i_dist, distance_type in enumerate(distance_types, start=1):
            values = (
                df_pair.loc[df_pair["distance_type"] == distance_type, "error_m"]
                .dropna()
                .to_numpy()
            )
            if values.size == 0:
                continue
            bp = ax.boxplot(
                [values],
                positions=[i_dist],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="k"),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color(i_rcv))
                patch.set_alpha(0.6)
            # ax.scatter(
            #     i_dist + rng.uniform(-0.15, 0.15, values.size),
            #     values,
            #     color="k",
            #     s=15,
            #     zorder=3,
            # )

        ax.set_xlim(0.5, len(distance_types) + 0.5)
        ax.set_xticks(range(1, len(distance_types) + 1))
        ax.set_xticklabels(
            [get_distance_type_label(key) for key in distance_types],
            rotation=45,
            ha="right",
        )
        ax.set_title(rcv_pair, fontsize=16)

    fig.supxlabel("Distance type")
    fig.supylabel("Position error [m]")
    _finalize_figure(fig, title)

    return fig


def plot_rcv_pair_performance_heatmaps(summary, success_threshold_m, title=None):
    """Compare the receiver pairs: median error and success rate for each distance type.

    Two heatmaps (receiver pairs in rows, distance types in columns), annotated
    with the values. The best pair(s) of each distance type are framed in red
    (no frame if all the pairs have the same value).

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    success_threshold_m : float
        Threshold used to compute the success rate, only used in the title.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    n_pairs = summary["rcv_pair"].nunique()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, max(5, 0.7 * n_pairs + 3.5)),
        squeeze=False,
    )
    panels = [
        ("median_error_m", "Median position error [m]", "viridis_r", "{:.0f}", True),
        (
            "success_rate",
            f"Success rate (error <= {success_threshold_m:g} m)",
            "viridis",
            "{:.0%}",
            False,
        ),
    ]

    for ax, (value, title, cmap, fmt, minimize) in zip(axs[0], panels):
        table = pivot_error_summary(summary, value)
        data = table.to_numpy(dtype=float)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax)

        for (i, j), val in np.ndenumerate(data):
            if not np.isfinite(val):
                text, text_color = "-", "k"
            else:
                r, g, b, _ = im.cmap(im.norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text, text_color = fmt.format(val), ("k" if luminance > 0.5 else "w")
            ax.text(j, i, text, ha="center", va="center", color=text_color)

        # Frame the best pair(s) of each distance type; nothing if all the pairs are tied
        for j in range(data.shape[1]):
            col = data[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                continue
            best_value = np.nanmin(col) if minimize else np.nanmax(col)
            is_best = finite & (col == best_value)
            if is_best.sum() == finite.sum():
                continue
            for i_best in np.flatnonzero(is_best):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i_best - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="r",
                        linewidth=3,
                    )
                )

        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(
            [get_distance_type_label(key) for key in table.columns],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(list(table.index))
        ax.set_title(title, fontsize=16)

    _finalize_figure(fig, title)

    return fig


def plot_error_cdf(df_errors, distance_types, success_threshold_m, title=None):
    """Cumulative distribution of the position error, one panel per distance type.

    One curve per receiver pair. The fraction is computed over all the analysed
    intersections, so a curve that does not reach 1 means that some errors are
    larger than the x axis or invalid (NaN). The success threshold is drawn as a
    dashed vertical line.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and ``error_m``.
    distance_types : list of str
        Distance types to display (one panel each).
    success_threshold_m : float
        Success threshold, in meters.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    rcv_pairs = list(df_errors["rcv_pair"].drop_duplicates())
    all_errors = df_errors["error_m"].to_numpy(dtype=float)
    all_errors = all_errors[np.isfinite(all_errors)]
    x_max = 1.05 * all_errors.max() if all_errors.size else 1.0

    fig, axs = _make_panel_grid(len(distance_types), sharex=True, sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        df_dist = df_errors[df_errors["distance_type"] == distance_type]
        for i_rcv, rcv_pair in enumerate(rcv_pairs):
            errors = df_dist.loc[df_dist["rcv_pair"] == rcv_pair, "error_m"].to_numpy(
                dtype=float
            )
            valid = np.sort(errors[np.isfinite(errors)])
            if valid.size == 0:
                continue
            fraction = np.arange(1, valid.size + 1) / errors.size
            ax.step(
                np.r_[0.0, valid, x_max],
                np.r_[0.0, fraction, fraction[-1]],
                where="post",
                color=color(i_rcv),
                label=rcv_pair,
            )
        ax.axvline(success_threshold_m, color="k", linestyle="--", linewidth=1)
        ax.set_title(get_distance_type_label(distance_type), fontsize=16)
        ax.set_ylim(0, 1.02)

    axs[0].legend(loc="lower right", fontsize=9)
    fig.supxlabel("Position error [m]")
    fig.supylabel("Fraction of intersections")
    _finalize_figure(fig, title)

    return fig


def plot_error_vs_frequency_band(df_band_errors, distance_types, title=None):
    """Position error versus frequency band, one panel per distance type.

    One curve per receiver pair: mean of the position error over the analysed
    intersections, with the standard deviation as error bar. The bands are
    labelled by their limits.

    Parameters
    ----------
    df_band_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair``, ``error_m``,
        ``band_index``, ``band_label`` and ``band_center``.
    distance_types : list of str
        Distance types to display (one panel each).
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    bands = (
        df_band_errors[["band_index", "band_label", "band_center"]]
        .drop_duplicates()
        .sort_values("band_index")
    )
    stats = (
        df_band_errors.groupby(["distance_type", "rcv_pair", "band_index"], sort=False)
        .agg(
            band_center=("band_center", "first"),
            mean_error=("error_m", "mean"),
            std_error=("error_m", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )
    rcv_pairs = list(df_band_errors["rcv_pair"].drop_duplicates())

    fig, axs = _make_panel_grid(len(distance_types), sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        for i_rcv, rcv_pair in enumerate(rcv_pairs):
            sub = stats[
                (stats["distance_type"] == distance_type)
                & (stats["rcv_pair"] == rcv_pair)
            ].sort_values("band_index")
            if sub.empty:
                continue
            ax.errorbar(
                x=sub["band_center"],
                y=sub["mean_error"],
                yerr=sub["std_error"],
                color=color(i_rcv),
                marker="o",
                capsize=3,
                label=rcv_pair,
            )
        ax.set_xticks(bands["band_center"])
        ax.set_xticklabels(bands["band_label"], rotation=45, ha="right")
        ax.set_title(get_distance_type_label(distance_type), fontsize=16)

    axs[0].legend(loc="best", fontsize=9)
    fig.supxlabel("Frequency band [Hz]")
    fig.supylabel("Position error [m]")
    _finalize_figure(fig, title)

    return fig


if __name__ == "__main__":
    pass
