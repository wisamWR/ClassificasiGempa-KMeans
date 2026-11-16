import json
import urllib.error
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gempa import bmkg, paths
from gempa.location import zone_context

DATA = Path(__file__).parent / "data"


def load(name):
    return json.loads((DATA / f"bmkg_{name}.json").read_text(encoding="utf-8"))


def test_parse_single_latest_event():
    df = bmkg.parse_feed(load("autogempa"))
    assert len(df) == 1
    event = df.iloc[0]
    assert event["time"] == pd.Timestamp("2026-11-15T20:51:57", tz="UTC")
    assert (event["latitude"], event["longitude"]) == (-5.97, 128.82)
    assert event["mag"] == 5.4 and event["depth"] == 323
    assert event["tsunami"] == "Tidak berpotensi tsunami"
    assert event["felt"] is None  # "-" di feed
    assert event["shakemap_url"] == "https://static.bmkg.go.id/20261116035430.mmi.jpg"


@pytest.mark.parametrize("name, column", [("gempaterkini", "tsunami"), ("gempadirasakan", "felt")])
def test_parse_event_lists(name, column):
    df = bmkg.parse_feed(load(name))
    assert len(df) == 15
    assert df["time"].is_monotonic_decreasing
    assert df[column].notna().all()
    assert df["shakemap_url"].isna().all()
    assert df["latitude"].between(-12, 7).all() and df["longitude"].between(94, 142).all()


def test_parse_skips_incomplete_events_and_rejects_unknown_format():
    payload = {"Infogempa": {"gempa": [
        {"DateTime": "2026-01-01T00:00:00+00:00", "Coordinates": "-7.1,107.2", "Magnitude": "4.1",
         "Kedalaman": "10 km"},
        {"DateTime": "2026-01-02T00:00:00+00:00", "Coordinates": "", "Magnitude": "4.5", "Kedalaman": "10 km"},
    ]}}
    df = bmkg.parse_feed(payload)
    assert len(df) == 1 and df.loc[0, "depth"] == 10
    with pytest.raises(bmkg.FeedError):
        bmkg.parse_feed({"unexpected": []})


def test_fetch_feed_wraps_network_errors(monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(bmkg.urllib.request, "urlopen", offline)
    with pytest.raises(bmkg.FeedError):
        bmkg.fetch_feed("terkini")


def test_zone_context_matches_zone_statistics():
    models = joblib.load(paths.ZONING_MODEL)
    zones = pd.read_csv(paths.ZONE_STATS)
    geojson = json.loads(paths.ZONES_GEOJSON.read_text(encoding="utf-8"))
    # Cianjur dangkal M6 dan Laut Banda dalam M6
    context = zone_context(models, zones, geojson, [-6.86, -6.5], [107.01, 129.5], [10, 450], [6.0, 6.0])
    assert list(context["zone_id"].str[:2]) == ["DK", "DL"]
    assert context.loc[0, "distance_km"] == 0
    expected = zones.set_index("zone_id").loc[context["zone_id"], "return_period_m6"].to_numpy()
    assert np.allclose(context["recurrence_years"], expected)
