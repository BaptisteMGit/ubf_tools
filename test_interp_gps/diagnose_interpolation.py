"""
Figures de diagnostic comparant l'ancienne et la nouvelle implementation de
interpolate_gps sur des donnees AIS reelles.

Usage :
    python diagnose_interpolation.py ais.csv --outdir figs
    python diagnose_interpolation.py ais.parquet --mmsi 227123456 227654321 --show
    python diagnose_interpolation.py ais.csv --time-col timestamp --epoch-unit s

Fichiers necessaires dans le meme dossier : interpolate_gps.py (nouvelle
version) et test_interpolate_gps.py (qui contient interpolate_gps_old).

Les deux versions recoivent exactement les memes donnees, pre-nettoyees de la
meme facon (NaN, codes AIS lon=181/lat=91, doublons de timestamp), de sorte que
les differences observees viennent uniquement de l'interpolation.
Il n'y a pas de verite terrain sur des donnees reelles : les figures montrent
l'ecart entre versions et la plausibilite physique (continuite de la vitesse).
"""

import argparse
import os
import time

import matplotlib
import numpy as np
import pandas as pd
from pyproj import Geod

geod = Geod(ellps="WGS84")
MS_TO_KN = 1.0 / 0.514444


# --------------------------------------------------------------------------
# Entrees / sorties
# --------------------------------------------------------------------------
def load_ais_(ais_spationav_fpath):
    # Chargement des données AIS détramées par le SHOM
    columns = ["mmsi", "lon", "lat", "datetime"]
    ais = pd.read_csv(
        ais_spationav_fpath, sep="\t", names=columns, header=None, usecols=[1, 2, 3, 4]
    )

    # Convert datetime column to datetime objects
    ais["datetime"] = pd.to_datetime(
        ais["datetime"], format="%Y-%m-%d %H:%M:%S", utc=True
    )

    # Remove AIS lines with unique positions (at least two points to define a trajectory)
    ais = ais.groupby("mmsi").filter(lambda x: len(x) > 1)

    # # MMSI du Jules
    # mmsi_jules = 226916000
    # df_gps_jules = ais.loc[ais["mmsi"] == mmsi_jules]

    return ais


def load_ais(path, cols, epoch_unit=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".feather", ".ft"):
        df = pd.read_feather(path)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Colonnes absentes : {missing}. Colonnes du fichier : {list(df.columns)}"
        )
    df = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})

    if epoch_unit:
        df["datetime"] = pd.to_datetime(
            df["datetime"], unit=epoch_unit, errors="coerce"
        )
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def clean(df):
    """Meme pre-nettoyage pour les deux versions."""
    df = df.dropna(subset=["datetime", "lon", "lat"])
    df = df[df["lon"].between(-180, 180) & df["lat"].between(-90, 90)]
    df = df.sort_values("datetime", kind="stable")
    return df.drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)


def choose_mmsi(ais, n, min_points):
    counts = ais.groupby("mmsi").size()
    counts = counts[counts >= min_points].sort_values(ascending=False)
    if counts.empty:
        raise SystemExit(f"Aucun MMSI avec au moins {min_points} points valides.")
    idx = np.unique(
        np.linspace(0, len(counts) - 1, min(n, len(counts))).round().astype(int)
    )
    return list(counts.index[idx])


# --------------------------------------------------------------------------
# Calculs
# --------------------------------------------------------------------------
def speed_kn(t, lon, lat):
    """Vitesse (noeuds) entre points successifs ; associee a l'instant d'arrivee."""
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    dt = t.diff().dt.total_seconds().to_numpy()[1:]
    _, _, d = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d / dt * MS_TO_KN
    return t.iloc[1:].to_numpy(), v


def run_both(df, step, old_fn, new_fn):
    t0 = time.perf_counter()
    try:
        old = old_fn(df, step)
    except Exception as e:  # l'ancienne version peut planter sur des cas degeneres
        print(f"   ! ancienne version en erreur : {e!r}")
        old = pd.DataFrame(columns=["datetime", "lon", "lat"])
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    new = new_fn(df, step)
    t_new = time.perf_counter() - t0
    return old, new, t_old, t_new


def compare(old, new):
    """Ecart (m) entre versions aux memes instants de grille."""
    m = new.merge(old, on="datetime", suffixes=("_new", "_old"), how="inner").dropna()
    if m.empty:
        return m.assign(diff_m=[])
    _, _, d = geod.inv(
        m["lon_old"].to_numpy(),
        m["lat_old"].to_numpy(),
        m["lon_new"].to_numpy(),
        m["lat_new"].to_numpy(),
    )
    return m.assign(diff_m=d)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def to_km(lon, lat, lon0, lat0):
    return (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * 111.32, (
        np.asarray(lat) - lat0
    ) * 111.32


def make_figure(mmsi, raw, old, new, cmp, step, outpath, show):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    c_raw, c_old, c_new = "black", "tab:red", "tab:blue"
    lon0, lat0 = raw["lon"].mean(), raw["lat"].mean()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"MMSI {mmsi} — {len(raw)} mesures, pas = {step}  "
        f"(rouge = ancienne version, bleu = nouvelle version)",
        fontsize=13,
    )

    # (a) Vue d'ensemble
    ax = axs[0, 0]
    ax.plot(*to_km(raw.lon, raw.lat, lon0, lat0), "-", color="0.8", lw=0.8, zorder=1)
    ax.plot(
        *to_km(old.lon, old.lat, lon0, lat0),
        ".",
        color=c_old,
        ms=2,
        label="ancienne",
        zorder=2,
    )
    ax.plot(
        *to_km(new.lon, new.lat, lon0, lat0),
        ".",
        color=c_new,
        ms=2,
        label="nouvelle",
        zorder=3,
    )
    ax.plot(
        *to_km(raw.lon, raw.lat, lon0, lat0),
        "o",
        color=c_raw,
        ms=3,
        label="mesures",
        zorder=4,
    )
    ax.set_title("Vue d'ensemble")
    ax.set_xlabel("Est (km)")
    ax.set_ylabel("Nord (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=3)

    # (b) Zoom sur le plus grand trou de mesures qui contient des points de grille
    ax = axs[0, 1]
    dts = raw["datetime"].diff().dt.total_seconds().to_numpy()
    k = int(np.nanargmax(dts)) if np.isfinite(dts).any() else 1
    lo, hi = max(k - 2, 0), min(k + 1, len(raw) - 1)
    win = raw.iloc[lo : hi + 1]
    gmask = lambda g: (g["datetime"] >= raw["datetime"].iloc[k - 1]) & (
        g["datetime"] <= raw["datetime"].iloc[k]
    )
    zl0, zt0 = win["lon"].mean(), win["lat"].mean()
    ax.plot(*to_km(win.lon, win.lat, zl0, zt0), "-", color="0.8", lw=0.8)
    for g, c, lab in ((old, c_old, "ancienne"), (new, c_new, "nouvelle")):
        gg = g[gmask(g)]
        ax.plot(
            *to_km(gg.lon, gg.lat, zl0, zt0), ".-", color=c, ms=4, lw=0.5, label=lab
        )
    ax.plot(*to_km(win.lon, win.lat, zl0, zt0), "o", color=c_raw, ms=6, label="mesures")
    ax.set_title(f"Zoom : plus grand trou entre mesures ({dts[k]:.0f} s)")
    ax.set_xlabel("Est (km)")
    ax.set_ylabel("Nord (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()

    # (c) Ecart entre versions au cours du temps
    ax = axs[0, 2]
    if len(cmp):
        ax.plot(cmp["datetime"], cmp["diff_m"], ".", ms=2, color="tab:purple")
    ymax = max(ax.get_ylim()[1], 1.0)
    ax.plot(
        raw["datetime"],
        np.zeros(len(raw)),
        "|",
        color=c_raw,
        ms=10,
        label="instants de mesure",
    )
    ax.set_ylim(-0.04 * ymax, ymax)
    ax.set_title("Ecart de position ancienne ↔ nouvelle (aux memes instants)")
    ax.set_ylabel("m")
    ax.legend(loc="upper right")

    # (d) Vitesse implicite
    ax = axs[1, 0]
    tr, vr = speed_kn(raw["datetime"], raw["lon"].to_numpy(), raw["lat"].to_numpy())
    ax.step(
        tr,
        vr,
        where="pre",
        color=c_raw,
        lw=1.5,
        label="entre mesures (reference)",
        zorder=3,
    )
    for g, c, lab in ((old, c_old, "ancienne"), (new, c_new, "nouvelle")):
        g = g.dropna()
        if len(g) > 1:
            tg, vg = speed_kn(g["datetime"], g["lon"].to_numpy(), g["lat"].to_numpy())
            ax.plot(tg, vg, ".-", color=c, ms=2, lw=0.6, label=lab, alpha=0.8)
    ax.set_title("Vitesse implicite (entre points successifs)")
    ax.set_ylabel("noeuds")
    ax.legend()
    for a in (axs[0, 2], axs[1, 0]):
        a.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

    # (e) Distribution des ecarts
    ax = axs[1, 1]
    if len(cmp):
        ax.hist(cmp["diff_m"], bins=50, color="tab:purple", alpha=0.8)
        med, p95, mx = (
            cmp["diff_m"].median(),
            cmp["diff_m"].quantile(0.95),
            cmp["diff_m"].max(),
        )
        ax.axvline(med, color="k", ls="--", label=f"mediane {med:.1f} m")
        ax.axvline(p95, color="k", ls=":", label=f"p95 {p95:.1f} m")
        ax.set_title(f"Distribution des ecarts (max {mx:.1f} m)")
        ax.legend()
    ax.set_xlabel("m")
    ax.set_ylabel("nb de points de grille")

    # (f) Echantillonnage des mesures
    ax = axs[1, 2]
    d = dts[1:][np.isfinite(dts[1:]) & (dts[1:] > 0)]
    bins = np.logspace(np.log10(max(d.min(), 0.5)), np.log10(d.max() + 1), 40)
    ax.hist(d, bins=bins, color="0.5")
    ax.axvline(
        pd.Timedelta(step).total_seconds(),
        color="g",
        ls="--",
        label=f"pas de grille ({step})",
    )
    ax.set_xscale("log")
    ax.set_title("Intervalles entre mesures")
    ax.set_xlabel("s")
    ax.set_ylabel("nb d'intervalles")
    ax.legend()

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("file", help="fichier AIS (csv, parquet, feather, pkl)")
    ap.add_argument(
        "--mmsi",
        nargs="+",
        type=int,
        help="MMSI a tracer (defaut : selection automatique)",
    )
    ap.add_argument(
        "--n", type=int, default=4, help="nb de MMSI choisis automatiquement (defaut 4)"
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="min. de points valides pour la selection auto",
    )
    ap.add_argument("--time-step", default="10s")
    ap.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="ne garder que les N premieres mesures par MMSI (l'ancienne version est lente) ; 0 = tout",
    )
    ap.add_argument("--outdir", default="diag_figs")
    ap.add_argument(
        "--show",
        action="store_true",
        help="afficher les figures (sinon sauvegarde seulement)",
    )
    ap.add_argument("--mmsi-col", default="mmsi")
    ap.add_argument("--time-col", default="datetime")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument(
        "--epoch-unit",
        default=None,
        help="si la colonne temps est un epoch : s, ms, ...",
    )
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    from interpolate_gps import interpolate_gps as new_fn
    from test_interpolate_gps import interpolate_gps_old as old_fn

    cols = {
        "mmsi": args.mmsi_col,
        "datetime": args.time_col,
        "lon": args.lon_col,
        "lat": args.lat_col,
    }
    # ais = load_ais(args.file, cols, args.epoch_unit)
    ais = load_ais_(args.file)
    print(f"{len(ais)} lignes, {ais['mmsi'].nunique()} MMSI")

    mmsi_list = args.mmsi or choose_mmsi(clean_all(ais), args.n, args.min_points)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for m in mmsi_list:
        raw = clean(ais[ais["mmsi"] == m])
        if len(raw) < 2:
            print(f"MMSI {m} : moins de 2 mesures valides, ignore")
            continue
        if args.max_points and len(raw) > args.max_points:
            print(
                f"MMSI {m} : {len(raw)} mesures, limite aux {args.max_points} premieres (--max-points)"
            )
            raw = raw.iloc[: args.max_points]

        print(f"MMSI {m} : {len(raw)} mesures")
        old, new, t_old, t_new = run_both(raw, args.time_step, old_fn, new_fn)
        cmp = compare(old, new)
        if new.empty:
            print("   aucun point de grille, ignore")
            continue

        out = os.path.join(args.outdir, f"diag_{m}.png")
        make_figure(m, raw, old, new, cmp, args.time_step, out, args.show)
        rows.append(
            {
                "mmsi": m,
                "n_mesures": len(raw),
                "dt_median_s": raw["datetime"].diff().dt.total_seconds().median(),
                "n_grille_new": len(new),
                "n_grille_old": len(old),
                "ecart_moy_m": cmp["diff_m"].mean() if len(cmp) else np.nan,
                "ecart_p95_m": cmp["diff_m"].quantile(0.95) if len(cmp) else np.nan,
                "ecart_max_m": cmp["diff_m"].max() if len(cmp) else np.nan,
                "t_old_s": t_old,
                "t_new_s": t_new,
            }
        )

    if rows:
        summary = pd.DataFrame(rows)
        path = os.path.join(args.outdir, "summary.csv")
        summary.to_csv(path, index=False)
        print("\n" + summary.round(2).to_string(index=False))
        print(f"\nFigures et summary.csv dans {args.outdir}/")


def clean_all(ais):
    """Nettoyage global rapide, uniquement pour compter les points valides par MMSI."""
    ais = ais.dropna(subset=["mmsi", "datetime", "lon", "lat"])
    return ais[ais["lon"].between(-180, 180) & ais["lat"].between(-90, 90)]


if __name__ == "__main__":
    main()
