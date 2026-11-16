import json

import joblib
import numpy as np
import pandas as pd
import pytest

from gempa.reference import has_reference_match
from gempa.stats import annual_rate, b_value, complete_exposure_years, poisson_probability
from gempa.zoning import (assign_zones, build_geojson, choose_k, depth_layer, fit_zoning, save_zoning,
                          to_ecef_km, zone_polygons, zone_statistics, zoning_events)


def test_b_value_recovers_true_b():
    rng = np.random.default_rng(1)
    # Gutenberg-Richter kontinu dengan b = 1: M - Mc ~ Exponential(scale = log10(e) / b)
    mags = 4.0 + rng.exponential(np.log10(np.e) / 1.0, size=5000)
    b, sigma, n = b_value(mags, 4.0)
    assert n == 5000
    assert b == pytest.approx(1.0, abs=3 * sigma)
    assert 0 < sigma < 0.05


def test_b_value_too_few_events():
    b, sigma, n = b_value([4.1], 4.0)
    assert np.isnan(b) and n == 1


def test_rates_and_poisson():
    assert annual_rate(10.0, 1.0, 4.0, 6.0) == pytest.approx(0.1)
    assert poisson_probability(0.1, 10) == pytest.approx(1 - np.exp(-1))


def test_complete_exposure_years_counts_partial_years():
    mc = {2020: 3.0, 2021: 5.0, 2022: 3.0}
    years = complete_exposure_years(mc, 4.0, pd.Timestamp("2020-07-02", tz="UTC"),
                                    pd.Timestamp("2022-07-02", tz="UTC"))
    # paruh kedua 2020 + paruh pertama 2022 (2021 belum lengkap pada M4)
    assert years == pytest.approx(1.0, abs=0.01)


def test_reference_match_time_and_distance():
    ref = pd.DataFrame({"time": pd.to_datetime(["2020-01-01 00:00:30"], utc=True),
                        "latitude": [0.0], "longitude": [120.0]})
    times = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00", "2020-01-01 00:05:00"], utc=True)
    matched = has_reference_match(times, [0.5, 3.0, 0.0], [120.0, 120.0, 120.0], ref)
    # 55 km & 30 s: cocok; 333 km: tidak; 270 s: tidak
    assert list(matched) == [True, False, False]


def test_depth_layer_boundaries():
    assert list(depth_layer([0, 69.9, 70, 299, 300, 650])) == ["DK", "DK", "MN", "MN", "DL", "DL"]


def test_ecef_distance_matches_geometry():
    a = to_ecef_km([0.0, 0.0], [120.0, 121.0], [0.0, 0.0])
    assert np.linalg.norm(a[0] - a[1]) == pytest.approx(111.2, abs=0.5)
    b = to_ecef_km([0.0, 0.0], [120.0, 120.0], [0.0, 100.0])
    assert np.linalg.norm(b[0] - b[1]) == pytest.approx(100.0)


def _blobs(rng, centers, n, depth, mw_scale=0.43):
    rows = []
    for lat, lon in centers:
        for _ in range(n):
            rows.append({"latitude": lat + rng.normal(0, 0.3), "longitude": lon + rng.normal(0, 0.3),
                         "depth": depth + rng.uniform(0, 20),
                         "mw": 4.0 + rng.exponential(mw_scale)})
    return rows


@pytest.fixture
def zoning_catalog():
    rng = np.random.default_rng(0)
    rows = (_blobs(rng, [(-7, 107), (-3, 128), (2, 125)], 120, depth=10)
            + _blobs(rng, [(-7, 120)], 120, depth=150))
    df = pd.DataFrame(rows)
    df["time"] = pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 3650, len(df)), unit="D")
    df["eventID"] = [f"e{i}" for i in range(len(df))]
    df["location"] = "Test"
    return df


def test_choose_k_respects_min_zone_size(zoning_catalog):
    shallow = zoning_catalog[zoning_catalog["depth"] < 70]
    X = to_ecef_km(shallow["latitude"], shallow["longitude"], shallow["depth"])
    k, scan = choose_k(X, min_zone_events=100, k_max=6)
    assert k == 3
    assert (scan.loc[scan["k"] <= k, "min_size"] >= 100).all()


def test_fit_zoning_assign_and_statistics(zoning_catalog):
    result = fit_zoning(zoning_catalog, min_zone_events=100, k_max=6)
    assert sorted(result.events["zone_id"].unique()) == ["DK-01", "DK-02", "DK-03", "MN-01"]
    # DK-01 = zona paling barat (Jawa, lon 107)
    west = result.events[result.events["zone_id"] == "DK-01"]["longitude"].mean()
    assert west == pytest.approx(107, abs=0.5)
    # assign_zones konsisten dengan label hasil fit
    assigned = assign_zones(result.models, zoning_catalog["latitude"], zoning_catalog["longitude"],
                            zoning_catalog["depth"])
    assert (assigned == result.events["zone_id"].to_numpy()).all()

    mc_by_year = {year: 4.0 for year in range(2015, 2026)}
    zones = zone_statistics(result.events, mc_by_year, 4.0,
                            pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
    assert len(zones) == 4
    # sampel dibuat dengan b = 1; tiap zona harus dalam 3 sigma (Shi & Bolt)
    assert ((zones["b"] - 1.0).abs() <= 3 * zones["b_sigma"]).all()
    assert (zones["p50_m6"] >= zones["p10_m6"]).all()
    assert (zones["rate_m5"] > zones["rate_m6"]).all()
    assert zones["name"].is_unique

    geojson = build_geojson(zones, zone_polygons(result.events))
    json.dumps(geojson)
    assert len(geojson["features"]) == 4
    ring = geojson["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) >= 4


def test_zoning_events_keeps_complete_confirmed_mainshocks():
    clean = pd.DataFrame({"eventID": list("abcd"),
                          "is_mainshock": [True, False, True, True],
                          "above_mc": [True, True, False, True],
                          "suspect_event": [False, False, False, True]})
    assert list(zoning_events(clean)["eventID"]) == ["a"]


def test_comparison_reports_both_methods(zoning_catalog):
    result = fit_zoning(zoning_catalog, min_zone_events=100, k_max=6)
    comp = result.comparison.set_index("layer")
    assert list(comp.index) == ["DK", "MN"]
    assert comp.loc["DK", "kmeans_k"] == 3 and comp.loc["MN", "kmeans_k"] == 1
    # tiga gugus terpisah jelas: HDBSCAN harus menemukannya juga dan sepakat dengan K-Means
    assert comp.loc["DK", "hdbscan_clusters"] == 3
    assert comp.loc["DK", "ari_on_hdbscan_core"] > 0.9
    # MN hanya 120 event < 2 x 100: tidak ada k >= 2 yang dicoba
    assert set(result.scan["layer"]) == {"DK"}


def test_save_zoning_writes_loadable_artifacts(zoning_catalog, tmp_path):
    result = fit_zoning(zoning_catalog, min_zone_events=100, k_max=6)
    result.zones = zone_statistics(result.events, {year: 4.0 for year in range(2015, 2026)}, 4.0,
                                   pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
    # event di luar set zonasi tetap diberi zona
    clean = pd.concat([zoning_catalog, zoning_catalog.head(1).assign(eventID="extra")], ignore_index=True)
    metadata = save_zoning(result, clean, 4.0, tmp_path / "models", tmp_path / "processed")

    models = joblib.load(tmp_path / "models" / "zoning_kmeans.joblib")
    assert set(models) == {"DK", "MN"}
    assert metadata["k_per_layer"] == {"DK": 3, "MN": 1}
    json.loads((tmp_path / "models" / "zoning_metadata.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(tmp_path / "models" / "zone_stats.csv")
    assert list(stats["zone_id"]) == ["DK-01", "DK-02", "DK-03", "MN-01"]
    geojson = json.loads((tmp_path / "models" / "zones.geojson").read_text(encoding="utf-8"))
    assert len(geojson["features"]) == 4

    zone_events = pd.read_parquet(tmp_path / "processed" / "zone_events.parquet")
    assert len(zone_events) == len(clean)
    assert zone_events["zone_id"].notna().all()
    assert zone_events.set_index("eventID").loc["extra", "in_zoning_set"] == False  # noqa: E712
    assert zone_events["in_zoning_set"].sum() == len(zoning_catalog)
