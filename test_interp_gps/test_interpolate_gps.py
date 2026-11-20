"""
Compare l'ancienne et la nouvelle implementation de interpolate_gps sur une
trajectoire dont la verite terrain est connue : navire a vitesse constante le
long d'une geodesique, echantillonne a des instants irreguliers non alignes sur
la grille. Dans ce cas, une interpolation geodesique correcte en temps doit
retrouver la verite a la precision numerique pres.

Usage : python test_interpolate_gps.py
"""

import numpy as np
import pandas as pd
from pyproj import Geod

from interpolate_gps import interpolate_gps as interpolate_gps_new

geod = Geod(ellps="WGS84")


# --------------------------------------------------------------------------
# Ancienne implementation (copie telle quelle)
# --------------------------------------------------------------------------
def interpolate_gps_old(df_gps, time_step="10s"):
    geod = Geod(ellps="WGS84")
    df_interp = pd.DataFrame()
    df_gps = df_gps.sort_values("datetime")
    high_res_pos = pd.DataFrame()

    t_start = df_gps["datetime"].iloc[0]
    t_start = t_start.ceil(time_step)
    t_end = df_gps["datetime"].iloc[-1]
    time_interp = pd.date_range(t_start, t_end, freq=time_step)
    high_res_time = pd.DataFrame({"datetime": time_interp})

    for i in range(1, len(df_gps)):
        t1 = df_gps["datetime"].iloc[i - 1]
        t2 = df_gps["datetime"].iloc[i]
        if i == len(df_gps) - 1:
            time_interp = high_res_time[
                (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] <= t2)
            ]
            include_last_point = True
        else:
            time_interp = high_res_time[
                (high_res_time["datetime"] >= t1) & (high_res_time["datetime"] < t2)
            ]
            include_last_point = False

        if time_interp.empty:
            continue

        lon1 = df_gps["lon"].iloc[i - 1]
        lat1 = df_gps["lat"].iloc[i - 1]
        lon2 = df_gps["lon"].iloc[i]
        lat2 = df_gps["lat"].iloc[i]

        terminus_idx = 0 if include_last_point else 1

        profile_coords = geod.inv_intermediate(
            lon1,
            lat1,
            lon2,
            lat2,
            npts=time_interp.size,
            return_back_azimuth=False,
            initial_idx=0,
            terminus_idx=terminus_idx,
        )

        new_data = pd.DataFrame({"lon": profile_coords.lons, "lat": profile_coords.lats})
        if high_res_pos.empty:
            high_res_pos = new_data
        else:
            high_res_pos = pd.concat([high_res_pos, new_data])

    high_res_time.reset_index(drop=True, inplace=True)
    high_res_pos.reset_index(drop=True, inplace=True)
    df_interp = pd.concat([high_res_time, high_res_pos], axis=1)
    df_interp.reset_index(drop=True, inplace=True)
    return df_interp


# --------------------------------------------------------------------------
# Verite terrain : vitesse constante le long d'une geodesique
# --------------------------------------------------------------------------
LON0, LAT0 = -4.50, 48.30  # rade de Brest
AZ = 60.0  # cap initial (deg)
SPEED = 6.2  # m/s (~12 noeuds)
T0 = pd.Timestamp("2026-11-20 12:00:03.400")


def truth(times):
    """Position vraie aux instants `times` (DatetimeIndex)."""
    s = (times - T0).total_seconds().to_numpy() * SPEED
    lon, lat, _ = geod.fwd(
        np.full(len(s), LON0), np.full(len(s), LAT0), np.full(len(s), AZ), s
    )
    return lon, lat


def err_m(df, label):
    lon_t, lat_t = truth(pd.DatetimeIndex(df["datetime"]))
    _, _, d = geod.inv(df["lon"].to_numpy(), df["lat"].to_numpy(), lon_t, lat_t)
    print(f"{label:>8} : erreur moyenne = {d.mean():9.3f} m | max = {d.max():9.3f} m")
    return d


def make_sparse_track(seed=0, duration_s=3600):
    """Mesures AIS-like : pas irreguliers 20-180 s, non alignes sur la grille."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(20, 180, size=200)
    offsets = np.cumsum(gaps)
    offsets = offsets[offsets < duration_s]
    times = T0 + pd.to_timedelta(np.r_[0, offsets], unit="s")
    lon, lat = truth(times)
    return pd.DataFrame({"datetime": times, "lon": lon, "lat": lat})


if __name__ == "__main__":
    # --- Cas 1 : trajectoire eparse, verite terrain connue -------------------
    df = make_sparse_track()
    print(f"{len(df)} mesures sur {(df.datetime.iloc[-1] - df.datetime.iloc[0])}\n")

    old = interpolate_gps_old(df, "10s")
    new = interpolate_gps_new(df, "10s")
    assert len(old) == len(new)
    assert (old["datetime"].to_numpy() == new["datetime"].to_numpy()).all()

    print("Cas 1 : erreur de position vs verite terrain (pas = 10 s)")
    d_old = err_m(old, "ancienne")
    d_new = err_m(new, "nouvelle")
    print(f"  (distance parcourue en 10 s a {SPEED} m/s = {10 * SPEED:.0f} m)\n")

    assert d_new.max() < 1e-3, "la nouvelle version doit etre exacte ici"
    assert d_old.max() > 1.0, "l'ancienne version doit montrer un biais"

    # --- Cas 2 : fractions de segment ----------------------------------------
    # 3 mesures (le 1er segment n'est donc pas le dernier) ; vitesse constante
    t = pd.Timestamp("2026-11-20 12:00:00")
    secs = np.array([3, 27, 60])
    p = pd.DataFrame(
        {
            "datetime": t + pd.to_timedelta(secs, unit="s"),
            "lon": LON0 + 0.01 * (secs - 3) / 24,
            "lat": LAT0,
        }
    )
    print("Cas 2 : mesures a t=3, 27 et 60 s, grille a 10 s")
    print("  fraction du 1er segment [3 s, 27 s] parcourue aux instants 10 s et 20 s")
    for name, f in (("ancienne", interpolate_gps_old), ("nouvelle", interpolate_gps_new)):
        out = f(p, "10s")
        frac = (out["lon"].to_numpy()[:2] - LON0) / 0.01
        print(f"  {name} : {np.round(frac, 3)}")
    print("  attendu  : [0.292 0.708]  ((10-3)/24 et (20-3)/24)")
