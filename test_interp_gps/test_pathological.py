"""
Cas pathologiques de interpolate_gps / interpolate_ais.
Usage : python test_pathological.py
"""

import builtins
import logging

import numpy as np
import pandas as pd

# Equivalents minimaux de tes globales (a remplacer par les vraies)
builtins.MMSI_JULES = 111
builtins.progression_bar = lambda i, i0, i1, prev: prev

from interpolate_ais import interpolate_ais  # noqa: E402
from interpolate_gps import interpolate_gps  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

T = pd.Timestamp("2026-11-20 12:00:00")


def mk(secs, lon=None, lat=None, mmsi=None):
    secs = np.asarray(secs, dtype=float)
    d = {
        "datetime": T + pd.to_timedelta(secs, unit="s"),
        "lon": -4.5 + 1e-4 * secs if lon is None else lon,
        "lat": 48.3 + 0 * secs if lat is None else lat,
    }
    if mmsi is not None:
        d["mmsi"] = mmsi
    return pd.DataFrame(d)


def check(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    assert cond, name


# --- interpolate_gps ---------------------------------------------------------
check("0 ligne -> vide, colonnes conservees",
      interpolate_gps(mk([]), "10s").empty
      and list(interpolate_gps(mk([]), "10s").columns) == ["datetime", "lon", "lat"])

check("1 seule mesure -> vide", interpolate_gps(mk([3]), "10s").empty)

check("2 mesures sans instant de grille entre elles -> vide",
      interpolate_gps(mk([1, 8]), "10s").empty)

# doublons de timestamp (y compris sur la derniere mesure) : pas de NaN/inf
dup = mk([3, 3, 27, 60, 60])
out = interpolate_gps(dup, "10s")
check("timestamps dupliques -> pas de NaN", out[["lon", "lat"]].notna().all().all())
check("timestamps dupliques -> grille complete", len(out) == 6)

# NaN et codes AIS 'non disponible' (lon=181, lat=91) ecartes
bad = mk([3, 10, 27, 60], lon=[-4.5, 181.0, np.nan, -4.4], lat=[48.3, 91.0, 48.3, 48.3])
out = interpolate_gps(bad, "10s")
check("codes AIS invalides / NaN ecartes",
      out[["lon", "lat"]].notna().all().all() and out["lon"].between(-4.5, -4.4).all())

# NaT dans datetime
nat = mk([3, 27, 60])
nat.loc[1, "datetime"] = pd.NaT
check("NaT ecarte", interpolate_gps(nat, "10s")["lon"].notna().all())

# donnees desordonnees
shuf = mk([60, 3, 27])
check("donnees desordonnees triees",
      interpolate_gps(shuf, "10s").equals(interpolate_gps(mk([3, 27, 60]), "10s")))

# timezone
tz = mk([3, 27, 60])
tz["datetime"] = tz["datetime"].dt.tz_localize("UTC")
check("datetimes tz-aware", len(interpolate_gps(tz, "10s")) == 6)

# --- interpolate_ais ---------------------------------------------------------
ais = pd.concat(
    [
        mk([3, 27, 60], mmsi=111),           # ok
        mk([5], mmsi=222),                    # 1 seule mesure
        mk([1, 8], mmsi=333),                 # aucun instant de grille
        mk([3, 3, 40], mmsi=444),             # doublon
        mk([3, 27], lon=[181.0, 181.0], lat=[91.0, 91.0], mmsi=555),  # tout invalide
    ],
    ignore_index=True,
)

res, mmsi = interpolate_ais(ais, "10s")
check("MMSI exploitables conserves", set(res["mmsi"].unique()) == {111, 444})
check("MMSI inexploitables ignores sans crash", len(mmsi) == 5)
check("pas de NaN dans le resultat", res[["lon", "lat"]].notna().all().all())

# MMSI a inclure absent des donnees (subset) -> pas de crash
res, mmsi = interpolate_ais(ais, "10s", subset_idx=[1, 2], mmsi_to_include=999)
check("mmsi_to_include absent -> pas de crash", res.empty and 999 in mmsi)

# Aucun MMSI exploitable -> DataFrame vide avec les bonnes colonnes
res, _ = interpolate_ais(ais[ais.mmsi.isin([222, 333])], "10s")
check("rien d'exploitable -> vide avec colonnes",
      res.empty and list(res.columns) == ["datetime", "lon", "lat", "mmsi"])

print("\nTous les tests passent.")
