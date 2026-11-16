import numpy as np
import pandas as pd
import pytest

from gempa.location import distance_to_ring_km, point_in_ring, radius_hazard, zones_at_point
from gempa.stats import poisson_probability
from gempa.timezones import local_timezone, to_local_time

SQUARE = [[100, -5], [102, -5], [102, -3], [100, -3], [100, -5]]


def test_point_in_ring():
    assert point_in_ring(101, -4, SQUARE)
    assert not point_in_ring(103, -4, SQUARE)
    assert not point_in_ring(101, -2, SQUARE)


def test_distance_to_ring_km():
    # 1 derajat bujur di timur sisi x = 102, dekat ekuator
    assert distance_to_ring_km(103, -4, SQUARE) == pytest.approx(111.32 * np.cos(np.radians(4)), rel=1e-3)
    assert distance_to_ring_km(101, -2, SQUARE) == pytest.approx(110.57, rel=1e-3)


def _feature(zone_id, layer, ring):
    return {"type": "Feature", "properties": {"zone_id": zone_id, "layer": layer},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_zones_at_point_picks_containing_or_nearest_per_layer():
    east = [[104, -5], [106, -5], [106, -3], [104, -3], [104, -5]]
    geojson = {"features": [_feature("DK-01", "DK", SQUARE), _feature("DK-02", "DK", east),
                            _feature("MN-01", "MN", east)]}
    result = zones_at_point(geojson, -4, 101)
    assert list(result["zone_id"]) == ["DK-01", "MN-01"]
    assert result.loc[0, "inside"] and result.loc[0, "distance_km"] == 0
    assert not result.loc[1, "inside"] and result.loc[1, "distance_km"] == pytest.approx(333, abs=2)


def test_radius_hazard_splits_zone_rate_by_event_share():
    zones = pd.DataFrame({"zone_id": ["A", "B"], "name": ["Zona A", "Zona B"], "layer": ["DK", "DK"],
                          "n_events": [4, 2], "rate_m5": [2.0, 1.0], "rate_m6": [0.2, 0.1]})
    # A: 2 dari 4 event dekat titik (0, 100); B: semua event jauh
    events = pd.DataFrame({"zone_id": ["A", "A", "A", "A", "B", "B"],
                           "latitude": [0.0, 0.1, 5.0, 5.0, 5.0, 5.0],
                           "longitude": [100.0, 100.1, 100.0, 100.0, 100.0, 100.0]})
    summary, contributions = radius_hazard(events, zones, 0.0, 100.0, 50)
    assert summary["n_events"] == 2
    assert summary["rate_m5"] == pytest.approx(1.0)
    assert summary["return_period_m6"] == pytest.approx(10.0)
    assert summary["p50_m6"] == pytest.approx(float(poisson_probability(0.1, 50)))
    assert list(contributions["zone_id"]) == ["A"]
    assert contributions.loc[0, "share"] == pytest.approx(0.5)


def test_radius_hazard_empty_area():
    zones = pd.DataFrame({"zone_id": ["A"], "name": ["Zona A"], "layer": ["DK"], "n_events": [1],
                          "rate_m5": [1.0], "rate_m6": [0.1]})
    events = pd.DataFrame({"zone_id": ["A"], "latitude": [5.0], "longitude": [100.0]})
    summary, contributions = radius_hazard(events, zones, 0.0, 120.0, 100)
    assert summary["n_events"] == 0 and summary["rate_m6"] == 0 and np.isinf(summary["return_period_m6"])
    assert contributions.empty


@pytest.mark.parametrize("city, lat, lon, expected", [
    ("Banda Aceh", 5.55, 95.32, "WIB"), ("Jakarta", -6.2, 106.85, "WIB"), ("Banyuwangi", -8.22, 114.37, "WIB"),
    ("Pontianak", -0.03, 109.33, "WIB"), ("Palangka Raya", -2.21, 113.92, "WIB"), ("Denpasar", -8.65, 115.22, "WITA"),
    ("Banjarmasin", -3.32, 114.59, "WITA"), ("Balikpapan", -1.27, 116.83, "WITA"), ("Makassar", -5.15, 119.43, "WITA"),
    ("Kupang", -10.17, 123.6, "WITA"), ("Manado", 1.47, 124.84, "WITA"), ("Talaud", 4.0, 126.8, "WITA"),
    ("Ternate", 0.79, 127.38, "WIT"), ("Ambon", -3.7, 128.18, "WIT"), ("Jayapura", -2.53, 140.72, "WIT"),
])
def test_local_timezone_major_cities(city, lat, lon, expected):
    assert local_timezone([lat], [lon])[0] == expected, city


def test_to_local_time_applies_offset():
    times = pd.to_datetime(["2022-11-21 06:21:00", "2019-09-25 23:46:00"], utc=True)
    local, zone = to_local_time(times, [-6.8, -3.4], [107.1, 128.4])
    assert list(zone) == ["WIB", "WIT"]
    assert local[0] == pd.Timestamp("2022-11-21 13:21:00")
    assert local[1] == pd.Timestamp("2019-09-26 08:46:00")
