import logging

import numpy as np
import pandas as pd

from interpolate_gps import interpolate_gps

# MMSI_JULES et progression_bar sont definis ailleurs dans ton code
# (a importer ici, ou a laisser dans le meme module que cette fonction).

logger = logging.getLogger(__name__)


def interpolate_ais(ais, time_step="10s", subset_idx=None, mmsi_to_include=MMSI_JULES):
    """
    Interpolate AIS data of each MMSI on a regular time grid.

    MMSI without enough valid data (fewer than 2 valid positions, no grid
    instant in their time span, or absent from `ais`) are skipped and reported
    in a single warning instead of crashing the whole run.

    Returns:
        ais_interp (pd.DataFrame): columns datetime, lon, lat, mmsi
        mmsi (np.ndarray): the MMSI that were requested (as before, skipped
            ones included)
    """
    mmsi = ais["mmsi"].unique()
    if subset_idx is not None:
        mmsi = mmsi[subset_idx]
        # Ensure mmsi_to_include is in mmsi in case we preprocess a subset
        if mmsi_to_include not in mmsi:
            mmsi = np.append(mmsi, mmsi_to_include)

    # One groupby instead of one boolean mask on the whole table per MMSI
    groups = {k: g for k, g in ais[ais["mmsi"].isin(mmsi)].groupby("mmsi", sort=False)}

    results, skipped = [], []

    index0 = 0
    indexf = mmsi.size - 1
    prev_progress = 0

    for i, mmsi_i in enumerate(mmsi):
        prev_progress = progression_bar(i, index0, indexf, prev_progress)

        ais_mmsi = groups.get(mmsi_i)
        if ais_mmsi is None:  # MMSI requested but absent from the data
            skipped.append(mmsi_i)
            continue

        ais_mmsi_interp = interpolate_gps(df_gps=ais_mmsi, time_step=time_step)
        if ais_mmsi_interp.empty:  # not enough valid data
            skipped.append(mmsi_i)
            continue

        ais_mmsi_interp["mmsi"] = mmsi_i
        results.append(ais_mmsi_interp)

    if skipped:
        logger.warning(
            "%d/%d MMSI skipped (absent or not enough valid data): %s",
            len(skipped),
            mmsi.size,
            skipped[:20] if len(skipped) > 20 else skipped,
        )

    # Single concatenation at the end (instead of one per MMSI)
    if results:
        ais_interp = pd.concat(results, ignore_index=True)
    else:
        ais_interp = pd.DataFrame(columns=["datetime", "lon", "lat", "mmsi"])

    return ais_interp, mmsi
