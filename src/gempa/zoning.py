"""Zonasi seismik: K-Means spasial per lapisan kedalaman + statistik hazard per zona.

Jalankan dari root project:  python -m gempa.zoning
"""
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import ConvexHull
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from gempa import paths
from gempa.completeness import mc_maxc
from gempa.stats import annual_rate, b_value, complete_exposure_years, poisson_probability

EARTH_RADIUS_KM = 6371.0
# (kode, nama, batas atas kedalaman km): sumber dangkal, Benioff menengah, dan dalam
DEPTH_LAYERS = [("DK", "dangkal", 70), ("MN", "menengah", 300), ("DL", "dalam", np.inf)]
# Ukuran zona minimum agar b-value stabil (sigma_b ~ b / sqrt(n) ~ 0.07)
MIN_ZONE_EVENTS = 200
K_MAX = 40
RANDOM_STATE = 42
N_INIT = 10
SCAN_N_INIT = 4      # cukup untuk membandingkan k, jauh lebih cepat
SILHOUETTE_SAMPLE = 10_000
# Zona dengan event >= Mc zona kurang dari ini memakai b-value lapisannya
MIN_B_EVENTS = 50
HAZARD_MAGNITUDES = (5.0, 6.0)
HAZARD_YEARS = (10, 50)
HDBSCAN_PARAMS = {"min_cluster_size": 100, "min_samples": 10}
# Titik terjauh dari pusat zona yang diabaikan saat membuat poligon (outlier)
HULL_QUANTILE = 0.95


def zoning_events(clean):
    """Event untuk zonasi: gempa utama, Mw >= Mc tahunannya, bukan event semu."""
    mask = clean["is_mainshock"] & clean["above_mc"] & ~clean["suspect_event"]
    return clean[mask].reset_index(drop=True)


def depth_layer(depth):
    """Kode lapisan kedalaman (DK/MN/DL) untuk tiap kedalaman (km)."""
    depth = np.asarray(depth, dtype=float)
    conditions = []
    lower = -np.inf
    for _, _, upper in DEPTH_LAYERS:
        conditions.append((depth >= lower) & (depth < upper) if np.isfinite(upper) else depth >= lower)
        lower = upper
    return np.select(conditions, [code for code, _, _ in DEPTH_LAYERS], default="DL")


def to_ecef_km(lat, lon, depth):
    """Koordinat Kartesius (km) berpusat di inti bumi; jarak Euclidean = jarak 3D sebenarnya."""
    lat, lon = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    r = EARTH_RADIUS_KM - np.asarray(depth, dtype=float)
    return np.column_stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])


def _silhouette(X, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    size = SILHOUETTE_SAMPLE if len(X) > SILHOUETTE_SAMPLE else None
    return float(silhouette_score(X, labels, sample_size=size, random_state=RANDOM_STATE))


def choose_k(X, min_zone_events=MIN_ZONE_EVENTS, k_max=K_MAX):
    """k terbesar yang semua zonanya berisi >= min_zone_events event; juga tabel scan-nya."""
    rows = []
    upper = min(k_max, len(X) // min_zone_events)
    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, n_init=SCAN_N_INIT, random_state=RANDOM_STATE).fit(X)
        sizes = np.bincount(km.labels_, minlength=k)
        rows.append({"k": k, "min_size": int(sizes.min()), "silhouette": _silhouette(X, km.labels_),
                     "davies_bouldin": float(davies_bouldin_score(X, km.labels_))})
    scan = pd.DataFrame(rows, columns=["k", "min_size", "silhouette", "davies_bouldin"])
    valid = scan[scan["min_size"] >= min_zone_events]
    k = int(valid["k"].max()) if not valid.empty else 1
    return k, scan


@dataclass
class ZoningResult:
    models: dict          # kode lapisan -> {"kmeans": KMeans, "zone_ids": [...]}
    events: pd.DataFrame  # event zonasi + kolom layer & zone_id
    scan: pd.DataFrame
    comparison: pd.DataFrame
    zones: pd.DataFrame = None


def fit_zoning(events, min_zone_events=MIN_ZONE_EVENTS, k_max=K_MAX):
    """K-Means per lapisan kedalaman; ID zona diurutkan dari barat ke timur."""
    events = events.copy()
    events["layer"] = depth_layer(events["depth"])
    events["zone_id"] = None
    models, scans, comparisons = {}, [], []

    for code, name, _ in DEPTH_LAYERS:
        layer = events[events["layer"] == code]
        if layer.empty:
            continue
        X = to_ecef_km(layer["latitude"], layer["longitude"], layer["depth"])
        k, scan = choose_k(X, min_zone_events, k_max)
        scans.append(scan.assign(layer=code))

        if k > 1:
            km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
            labels = km.labels_
        else:
            km = KMeans(n_clusters=1, n_init=1, random_state=RANDOM_STATE).fit(X)
            labels = np.zeros(len(X), dtype=int)
        # Urutkan zona dari barat ke timur agar ID mudah dibaca
        mean_lon = pd.Series(layer["longitude"].to_numpy()).groupby(labels).mean()
        order = {label: rank for rank, label in enumerate(mean_lon.sort_values().index)}
        zone_ids = [None] * k
        for label, rank in order.items():
            zone_ids[label] = f"{code}-{rank + 1:02d}"
        events.loc[layer.index, "zone_id"] = [zone_ids[label] for label in labels]
        models[code] = {"kmeans": km, "zone_ids": zone_ids, "name": name}

        hd = HDBSCAN(copy=True, **HDBSCAN_PARAMS).fit(X)
        core = hd.labels_ >= 0
        comparisons.append({
            "layer": code,
            "n_events": len(X),
            "kmeans_k": k,
            "kmeans_silhouette": _silhouette(X, labels),
            "kmeans_davies_bouldin": float(davies_bouldin_score(X, labels)) if k > 1 else np.nan,
            "hdbscan_clusters": int(hd.labels_.max() + 1),
            "hdbscan_noise": float(1 - core.mean()),
            "hdbscan_silhouette": _silhouette(X[core], hd.labels_[core]),
            "hdbscan_davies_bouldin": (float(davies_bouldin_score(X[core], hd.labels_[core]))
                                       if hd.labels_.max() >= 1 else np.nan),
            "ari_on_hdbscan_core": (float(adjusted_rand_score(labels[core], hd.labels_[core]))
                                    if core.any() else np.nan),
        })

    return ZoningResult(models, events, pd.concat(scans, ignore_index=True), pd.DataFrame(comparisons))


def assign_zones(models, lat, lon, depth):
    """Zona untuk event apa pun: lapisan dari kedalaman, lalu centroid K-Means terdekat."""
    lat, lon, depth = (np.asarray(v, dtype=float) for v in (lat, lon, depth))
    layers = depth_layer(depth)
    zones = np.full(len(lat), None, dtype=object)
    for code, model in models.items():
        sel = layers == code
        if sel.any():
            labels = model["kmeans"].predict(to_ecef_km(lat[sel], lon[sel], depth[sel]))
            zones[sel] = np.asarray(model["zone_ids"], dtype=object)[labels]
    return zones


def _layer_name(code):
    return next(name for c, name, _ in DEPTH_LAYERS if c == code)


def _hazard(mags, times, mc_by_year, mc_floor, start, end):
    """Mc zona, b-value, dan laju tahunan pada Mc (hanya dari tahun yang lengkap)."""
    mc = max(mc_maxc(mags), mc_floor)
    years_ok = {year for year, value in mc_by_year.items() if value <= mc + 1e-9}
    in_complete = pd.DatetimeIndex(times).year.isin(list(years_ok))
    m = np.asarray(mags)[in_complete]
    b, sigma, n = b_value(m, mc)
    exposure = complete_exposure_years(mc_by_year, mc, start, end)
    return {"mc": mc, "b": b, "b_sigma": sigma, "n_b": n, "exposure_years": exposure,
            "rate_mc": n / exposure if exposure > 0 else np.nan}


def zone_statistics(events, mc_by_year, mc_floor, start, end):
    """Statistik hazard per zona. b-value lapisan dipakai bila event zona < MIN_B_EVENTS."""
    period_years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86400)
    layer_hazard = {
        code: _hazard(group["mw"].to_numpy(), group["time"], mc_by_year, mc_floor, start, end)
        for code, group in events.groupby("layer")
    }

    rows = []
    for zone_id, group in events.groupby("zone_id"):
        code = group["layer"].iloc[0]
        h = _hazard(group["mw"].to_numpy(), group["time"], mc_by_year, mc_floor, start, end)
        b, sigma, b_source = h["b"], h["b_sigma"], "zona"
        if h["n_b"] < MIN_B_EVENTS or not np.isfinite(b):
            b, sigma, b_source = layer_hazard[code]["b"], layer_hazard[code]["b_sigma"], "lapisan"
        row = {
            "zone_id": zone_id,
            "layer": code,
            "layer_name": _layer_name(code),
            "location": group["location"].mode().iloc[0],
            "n_events": len(group),
            "lat": group["latitude"].mean(),
            "lon": group["longitude"].mean(),
            "depth_median": group["depth"].median(),
            "mw_max": group["mw"].max(),
            "observed_m5": int((group["mw"] >= 5).sum()),
            "observed_m6": int((group["mw"] >= 6).sum()),
            "mc": h["mc"],
            "n_b": h["n_b"],
            "exposure_years": h["exposure_years"],
            "b": b,
            "b_sigma": sigma,
            "b_source": b_source,
            "a": np.log10(h["rate_mc"]) + b * h["mc"],
        }
        for m in HAZARD_MAGNITUDES:
            rate = float(annual_rate(h["rate_mc"], b, h["mc"], m))
            tag = f"m{m:.0f}"
            row[f"rate_{tag}"] = rate
            row[f"return_period_{tag}"] = 1 / rate if rate > 0 else np.inf
            row[f"expected_{tag}_catalog"] = rate * period_years
            for t in HAZARD_YEARS:
                row[f"p{t}_{tag}"] = float(poisson_probability(rate, t))
        rows.append(row)

    zones = pd.DataFrame(rows).sort_values(["layer", "zone_id"],
                                           key=lambda s: s.map({c: i for i, (c, _, _) in enumerate(DEPTH_LAYERS)})
                                           if s.name == "layer" else s)
    # Nama unik: lokasi dominan + lapisan, diberi nomor bila sama
    base = zones["location"] + " (" + zones["layer_name"] + ")"
    dup = base.groupby(base).cumcount() + 1
    counts = base.map(base.value_counts())
    zones["name"] = np.where(counts > 1, base + " #" + dup.astype(str), base)
    return zones.reset_index(drop=True)


def zone_polygons(events, quantile=HULL_QUANTILE):
    """Poligon convex hull (lon, lat) tiap zona, tanpa titik terjauh (outlier)."""
    polygons = {}
    for zone_id, group in events.groupby("zone_id"):
        pts = group[["longitude", "latitude"]].to_numpy()
        center = np.median(pts, axis=0)
        dist = np.hypot(*(pts - center).T)
        core = pts[dist <= np.quantile(dist, quantile)]
        if len(core) < 3:
            continue
        hull = core[ConvexHull(core).vertices]
        ring = np.vstack([hull, hull[:1]]).round(4).tolist()
        polygons[zone_id] = ring
    return polygons


def build_geojson(zones, polygons):
    features = []
    for _, z in zones.iterrows():
        if z["zone_id"] not in polygons:
            continue
        props = {k: (None if isinstance(v, float) and not np.isfinite(v) else
                     (round(v, 6) if isinstance(v, float) else v))
                 for k, v in z.items() if k not in ("lat", "lon")}
        props["centroid"] = [round(z["lon"], 4), round(z["lat"], 4)]
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [polygons[z["zone_id"]]]}})
    return {"type": "FeatureCollection", "features": features}


def run(clean):
    """Pipeline zonasi lengkap dari katalog bersih (output gempa.clean)."""
    events = zoning_events(clean)
    result = fit_zoning(events)
    mc_by_year = clean.groupby(clean["time"].dt.year)["mc"].first().to_dict()
    mc_floor = mc_maxc(clean["mw"])
    result.zones = zone_statistics(result.events, mc_by_year, mc_floor, clean["time"].min(), clean["time"].max())
    return result, mc_by_year, mc_floor


def save_zoning(result, clean, mc_floor, models_dir=paths.MODELS_DIR, processed_dir=paths.PROCESSED_DIR):
    models_dir, processed_dir = Path(models_dir), Path(processed_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.models, models_dir / paths.ZONING_MODEL.name)
    result.zones.to_csv(models_dir / paths.ZONE_STATS.name, index=False)
    geojson = build_geojson(result.zones, zone_polygons(result.events))
    with open(models_dir / paths.ZONES_GEOJSON.name, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    zone_of_event = pd.DataFrame({
        "eventID": clean["eventID"],
        "zone_id": assign_zones(result.models, clean["latitude"], clean["longitude"], clean["depth"]),
    })
    zone_of_event["in_zoning_set"] = zone_of_event["eventID"].isin(result.events["eventID"])
    zone_of_event.to_parquet(processed_dir / paths.ZONE_EVENTS.name, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scikit_learn": sklearn.__version__,
        "n_zoning_events": len(result.events),
        "selection": "gempa utama, Mw >= Mc tahunan, bukan event semu",
        "depth_layers": [{"code": c, "name": n, "max_depth_km": None if not np.isfinite(u) else u}
                         for c, n, u in DEPTH_LAYERS],
        "params": {"min_zone_events": MIN_ZONE_EVENTS, "k_max": K_MAX, "n_init": N_INIT, "scan_n_init": SCAN_N_INIT,
                   "random_state": RANDOM_STATE, "min_b_events": MIN_B_EVENTS, "mc_floor": mc_floor,
                   "hdbscan": HDBSCAN_PARAMS, "hazard_magnitudes": list(HAZARD_MAGNITUDES),
                   "hazard_years": list(HAZARD_YEARS)},
        "k_per_layer": {code: len(m["zone_ids"]) for code, m in result.models.items()},
        "k_scan": result.scan.to_dict("records"),
        "comparison": result.comparison.to_dict("records"),
    }
    with open(models_dir / paths.ZONING_METADATA.name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=float)
    return metadata


def load_clean_catalog(path=paths.CLEAN_CATALOG):
    """Katalog bersih dari disk; dibuat ulang (gempa.clean) bila belum ada."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    from gempa.clean import clean_catalog
    from gempa.data import load_catalog
    clean, _ = clean_catalog(load_catalog(paths.RAW_CATALOG))
    return clean


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zonasi seismik dan statistik hazard per zona.")
    parser.add_argument("--catalog", type=Path, default=paths.CLEAN_CATALOG, help="Katalog bersih (parquet)")
    parser.add_argument("--models-dir", type=Path, default=paths.MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=paths.PROCESSED_DIR)
    args = parser.parse_args(argv)

    clean = load_clean_catalog(args.catalog)
    result, _, mc_floor = run(clean)
    save_zoning(result, clean, mc_floor, args.models_dir, args.processed_dir)

    print(f"Event zonasi: {len(result.events):,}")
    print("k per lapisan:", {code: len(m["zone_ids"]) for code, m in result.models.items()})
    print(result.comparison.round(3).to_string(index=False))
    cols = ["zone_id", "name", "n_events", "mw_max", "mc", "b", "b_sigma", "b_source",
            "return_period_m6", "p50_m6", "observed_m6", "expected_m6_catalog"]
    print(result.zones[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
