#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_utils.py
@Time    :   2026/09/28 21:54:27
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Functions of the acouplane project to read the raw data and build the portions.
             The functions of the analysis of a portion (CPSD, features, MFP, localization, combination of
             receivers, figures) are in the acouplane_*.py modules and re-exported here.
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

from publication.publication_figure import color
from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_DATA_CELERITY_SIPPICANS_DIR,
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
)
from get_data.obs_bin.obs_reader_acouplane import convert_data_raw_to_physical
from source.utils.utils_geo import geodetic_to_ecef, ecef_to_enu

# Functions moved to the thematic modules acouplane_*.py (re-exported: imports from acouplane_utils keep working)
from real_data_analysis.acouplane.src.acouplane_portion_io import (  # noqa: F401
    open_portion,
    save_portion_inplace,
    select_intersections,
    list_stored_intersections,
    restrict_dataset_to_receivers,
)
from real_data_analysis.acouplane.src.acouplane_preprocessing import (  # noqa: F401
    compute_src_rcv_geodesic_dist,
    apply_rcv_position_correction_to_portion,
    derive_portion_propagation_time,
    compute_cpsd_matrix,
    derive_portion_cpsd,
)
from real_data_analysis.acouplane.src.acouplane_features import (  # noqa: F401
    _MAX_BROADCAST_ELEMENTS,
    get_distance_between_sections,
    _gamma_distances_block,
    _feature_distances_block,
    derive_feature_distance_between_sections,
    _non_finite_to_nan,
    _normalize_by_max,
    _center_along_freq,
    derive_gamma_distance_between_sections,
    derive_section_features,
    all_features_single_section,
    distances_single_intersection,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (  # noqa: F401
    build_mfp_intersection_dataset,
    apply_mfp_strategy_to_portion_intersections,
)
from real_data_analysis.acouplane.src.acouplane_localization import (  # noqa: F401
    DISTANCE_TYPES,
    DISTANCE_TYPE_LABELS,
    POSITION_ERROR_MODES,
    get_distance_type_label,
    select_distance_types,
    _get_wgs84_geod,
    _geodesic_distance_m,
    _argmin_2d,
    localization_errors_from_distances,
    _distances_from_mfp_dataset,
    localization_errors_from_mfp_dataset,
    ambiguity_contrast_from_distances,
    ambiguity_contrast_from_mfp_dataset,
    errors_to_records,
    add_validity_column,
    summarize_localization_errors,
    pivot_error_summary,
    best_rcv_pair_per_distance,
    select_and_order_distance_types,
)
from real_data_analysis.acouplane.src.acouplane_frequency_bands import (  # noqa: F401
    build_frequency_bands,
    frequency_band_label,
    slice_frequency_band,
    build_cumulative_frequency_bands,
)
from real_data_analysis.acouplane.src.acouplane_combination import (  # noqa: F401
    RCV_COMBINATION_METHODS,
    format_rcv_combination,
    enumerate_rcv_combinations,
    combine_distances_over_receivers,
    rcv_combination_records,
)
from real_data_analysis.acouplane.src.acouplane_plot_portion import (  # noqa: F401
    plot_obs_pos,
    plot_portion_tir_pos,
    plot_pressure_section,
    plot_spectrogram_section,
)
from real_data_analysis.acouplane.src.acouplane_plot_analysis import (  # noqa: F401
    _finalize_figure,
    _make_panel_grid,
    plot_error_boxplots_per_rcv_pair,
    plot_rcv_pair_performance_heatmaps,
    plot_error_cdf,
    plot_error_vs_frequency_band,
    plot_error_vs_bandwidth,
    plot_contrast_vs_validity,
    plot_metric_vs_n_rcv,
)

# ======================================================================================================================
# Portion file handling
# ======================================================================================================================


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
    # obs_linear_time = {
    #     obs_id: obs_datetimes[obs_id] - obs_datetimes[obs_id][0]
    #     for obs_id in obs_datetimes.keys()
    # }
    # Ensure same date format is used
    # pd.to_datetime(recording_start).to_pydatetime()
    section_start = np.datetime64(section_start)
    section_end = np.datetime64(section_end)

    if obs_datetimes is None:

        for obs_id in ds_pressure.obs_id.values:
            # Step one : ensure data exists in the required period of time
            recording_start = ds_pressure.start_datetime.sel(obs_id=obs_id).values
            recording_stop = recording_start + np.timedelta64(
                int(1 / ds_pressure.fs * ds_pressure.sizes[f"time_{obs_id}"] * 1e6),
                "us",
            )

            section_ends_before_recording_start = recording_start >= section_end
            section_starts_after_recording_end = recording_stop <= section_start
            no_data_available = (
                section_ends_before_recording_start
                or section_starts_after_recording_end
            )
            if no_data_available:
                raise ValueError(
                    f"OBS '{obs_id}' has no data between {section_start} and {section_end} "
                    f"(recording spans {recording_start} -> {recording_stop})."
                )

            section_start_ = max(section_start, recording_start)
            section_end_ = min(section_end, recording_stop)

            tau_before_start_ns = section_start_ - recording_start
            tau_before_start_s = tau_before_start_ns / np.timedelta64(1, "s")
            idx_start = int(tau_before_start_s * ds_pressure.fs)
            # idx_start = max(idx_start, 0)

            section_duration_ns = section_end_ - section_start_
            section_duration_s = section_duration_ns / np.timedelta64(1, "s")
            idx_end = idx_start + int(section_duration_s * ds_pressure.fs)
            # idx_end = min(idx_end, ds_pressure.sizes[f"time_{obs_id}"])

            # print(
            #     f"{obs_id.upper()} : \t recording start {recording_start} / stop {recording_stop}\n\t section start {section_start} / stop {section_end} \n\t duration {section_duration_s}"
            # )

            obs_slice_idx.append((idx_start, idx_end))

    else:

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
    step_ns = int(round(1e9 / ds_pressure.fs))

    for i, obs_id in enumerate(ds_pressure.obs_id.values):

        if obs_datetimes is None:

            recording_start = ds_pressure.start_datetime.sel(obs_id=obs_id).values
            section_start_ = max(section_start, recording_start)
            time_vector = pd.date_range(
                start=section_start_,
                freq=f"{step_ns}ns",
                inclusive="left",
                periods=pressure_slice.sizes[f"time_{obs_id}"],
            )

            pressure_slice[f"time_{obs_id}"] = time_vector
        else:
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


# ======================================================================================================================
# Feature estimation (for RTF / DCF estimation)
# ======================================================================================================================


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


# ======================================================================================================================
# Analysis of the results (position error of the estimated intersections)
# ======================================================================================================================


# ======================================================================================================================
# Combination of receivers
# ======================================================================================================================


if __name__ == "__main__":
    pass
