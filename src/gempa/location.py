"""Hazard di sekitar satu lokasi: zona yang memuat titik dan laju gempa dalam radius tertentu."""
import numpy as np
import pandas as pd

from gempa.geo import haversine_km
from gempa.stats import poisson_probability

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32
HAZARD_MAGNITUDES = (5, 6)
HAZARD_YEARS = (10, 50)


def point_in_ring(lon, lat, ring):
    """Ray casting: apakah titik (lon, lat) ada di dalam poligon [[lon, lat], ...]."""
    ring = np.asarray(ring, dtype=float)
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    crosses = (y0 > lat) != (y1 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (lat - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.count_nonzero(crosses & (lon < x_cross)) % 2)


def distance_to_ring_km(lon, lat, ring):
    """Jarak terdekat (km) dari titik ke tepi poligon, proyeksi ekuirektangular lokal."""
    ring = np.asarray(ring, dtype=float)
    scale = np.array([KM_PER_DEG_LON * np.cos(np.radians(lat)), KM_PER_DEG_LAT])
    pts = (ring - [lon, lat]) * scale
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(-(a * ab).sum(axis=1) / np.maximum((ab ** 2).sum(axis=1), 1e-12), 0, 1)
    nearest = a + t[:, None] * ab
    return float(np.hypot(nearest[:, 0], nearest[:, 1]).min())


def zones_at_point(geojson, lat, lon):
    """Zona tiap lapisan di titik ini: yang poligonnya memuat titik, atau yang terdekat.

    Mengembalikan DataFrame zone_id, layer, inside, distance_km (0 bila di dalam), satu baris per lapisan.
    """
    rows = []
    for feature in geojson["features"]:
        ring = feature["geometry"]["coordinates"][0]
        inside = point_in_ring(lon, lat, ring)
        rows.append({"zone_id": feature["properties"]["zone_id"], "layer": feature["properties"]["layer"],
                     "inside": inside, "distance_km": 0.0 if inside else distance_to_ring_km(lon, lat, ring)})
    zones = pd.DataFrame(rows, columns=["zone_id", "layer", "inside", "distance_km"])
    order = {code: i for i, code in enumerate(dict.fromkeys(zones["layer"]))}
    best = zones.sort_values(["layer", "distance_km"]).groupby("layer", sort=False).head(1)
    return best.sort_values("layer", key=lambda s: s.map(order)).reset_index(drop=True)


def radius_hazard(events, zones, lat, lon, radius_km):
    """Laju & probabilitas gempa utama M >= 5 / M >= 6 dengan episentrum dalam radius dari titik.

    Laju tiap zona dibagi rata ke event zonasinya: kontribusi zona = laju zona x (event zona di dalam
    radius / total event zona). events: event zonasi dengan kolom zone_id, latitude, longitude.
    """
    distance = haversine_km(lat, lon, events["latitude"].to_numpy(), events["longitude"].to_numpy())
    inside = events[distance <= radius_km]
    stats = zones.set_index("zone_id")
    share = (inside.groupby("zone_id").size() / stats["n_events"]).dropna()
    share = share[share > 0]

    contributions = stats.loc[share.index, ["name", "layer"]].assign(share=share)
    for m in HAZARD_MAGNITUDES:
        contributions[f"rate_m{m}"] = stats.loc[share.index, f"rate_m{m}"] * share
    contributions = contributions.sort_values(f"rate_m{HAZARD_MAGNITUDES[0]}", ascending=False).reset_index()

    summary = {"n_events": len(inside)}
    for m in HAZARD_MAGNITUDES:
        rate = float(contributions[f"rate_m{m}"].sum())
        summary[f"rate_m{m}"] = rate
        summary[f"return_period_m{m}"] = 1 / rate if rate > 0 else np.inf
        for t in HAZARD_YEARS:
            summary[f"p{t}_m{m}"] = float(poisson_probability(rate, t))
    return summary, contributions
