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
    dist_L2_gamma = dist_L2_gamma / np.nanmax(dist_L2_gamma)

    return dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf


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
    for start in range(0, nt1, block):
        sl = slice(start, min(start + block, nt1))
        dist_theta_gamma[sl], dist_L2_gamma[sl] = _gamma_distances_block(
            gamma_section1[sl, np.newaxis, :], gamma_b
        )

    return dist_theta_gamma, dist_L2_gamma / np.nanmax(dist_L2_gamma)


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

    # RTF
    pi_ij = sij.values / sjj.values
    gamma_ij = 20 * np.log10(np.abs(pi_ij))  # (nt, nf)

    # DCF
    sij_scot = sij.values / np.sqrt(sii.values * sjj.values)

    return pi_ij, gamma_ij, sij_scot


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
    along frequency (mean of each shot removed, ``np.mean`` so that a shot
    containing a NaN stays entirely NaN, as in the former implementation). The
    centered gamma is computed on the fly and is never stored. ``theta``
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
            gamma1_c = features_section1["gamma_ij"] - np.nanmean(
                features_section1["gamma_ij"], axis=1, keepdims=True
            )
            gamma2_c = features_section2["gamma_ij"] - np.nanmean(
                features_section2["gamma_ij"], axis=1, keepdims=True
            )
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


def apply_mfp_strategy_to_portion_intersections(
    portion,
    rcv_ref="obs1",
    fmin=5,
    fmax=50,
    verbose=False,
    compare_centered_gamma=False,
):
    """Compute the MFP ambiguity surfaces for every meridional / zonal intersection.

    Step 1: the RTF and DCF features of each section are derived
    (``all_features_single_section``) and stored in the ``feature`` group
    (variables ``pi_ij``, ``gamma_ij``, ``sij_scot``).
    Step 2: each meridional section (label starting with ``"sn"``) is compared
    with each zonal section (label starting with ``"oe"``) and the result is
    stored in ``/mfp_intersection/<meridional>/<zonal>``.

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

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the ``feature`` group is missing, or if the portion has no
        meridional or no zonal section (no intersection can be built).
    """
    if "feature" not in portion.children:
        raise ValueError(
            "No 'feature' group in the portion: the CPSD has not been computed yet. "
            "Run with compute_cpsds = True first."
        )

    # Step 1 : list available sections
    sections = list(portion.feature.children)

    # Step 2 : derive all features to avoid duplicated call to derive_section_features
    for section_name, feature_node in portion.feature.children.items():
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

    # Step 3
    meridional_sections = [sec for sec in sections if sec.startswith("sn")]
    zonal_sections = [sec for sec in sections if sec.startswith("oe")]
    if not meridional_sections or not zonal_sections:
        raise ValueError(
            "At least one meridional ('sn*') and one zonal ('oe*') section are required "
            f"to build intersections (sections found: {sections})."
        )

    # Intersection are between zonal and meridional sections
    for section_name1 in meridional_sections:
        ds_feature_section1 = portion.feature[section_name1].to_dataset()

        for section_name2 in zonal_sections:
            if verbose:
                print(f"    Intersection '{section_name1}' x '{section_name2}'")
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


if __name__ == "__main__":
    pass
