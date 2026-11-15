import json

import joblib
import numpy as np
import pandas as pd
import pytest

from gempa.data import load_catalog
from gempa.train import RISK_LEVELS, label_names, save_artifacts, summarize_clusters, train


@pytest.fixture
def synthetic_events():
    """Empat kelompok jelas: dangkal-besar, dangkal-kecil, menengah, dalam."""
    rng = np.random.default_rng(0)
    centers = [(5.5, 15), (2.5, 10), (4.0, 150), (4.5, 500)]
    rows = []
    for mag, depth in centers:
        for _ in range(60):
            rows.append({
                "mag": mag + rng.normal(0, 0.1),
                "depth": max(depth + rng.normal(0, 5), 0),
                "latitude": rng.uniform(-10, 5),
                "longitude": rng.uniform(95, 140),
            })
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2020-01-01", periods=len(df), freq="h", tz="UTC")
    return df


def test_label_names():
    assert label_names(4) == RISK_LEVELS
    assert label_names(3) == ["Level 1", "Level 2", "Level 3"]


def test_summarize_orders_labels_by_severity():
    events = pd.DataFrame({
        "cluster": [7, 7, 3, 3, 1, 1, 9, 9],
        "mag": [5.5, 5.5, 2.5, 2.5, 4.0, 4.0, 4.5, 4.5],
        "depth": [15, 15, 10, 10, 150, 150, 500, 500],
        "latitude": 0.0,
        "longitude": 0.0,
    })
    summary = summarize_clusters(events)
    assert list(summary["label"]) == RISK_LEVELS
    assert summary["severity_score"].is_monotonic_increasing
    # Dalam (500 km) = Low, dangkal-besar = Very High
    assert summary.loc[summary["label"] == "Low", "cluster"].item() == 9
    assert summary.loc[summary["label"] == "Very High", "cluster"].item() == 7


def test_train_selects_k_and_labels_events(synthetic_events):
    result = train(synthetic_events, k_range=range(2, 6), sample_size=None)
    assert result.k == 4
    assert set(result.events["label"]) == set(RISK_LEVELS)
    # Label event konsisten dengan prediksi pipeline + label_map
    pred = result.pipeline.predict(synthetic_events[["mag", "depth"]])
    mapped = pd.Series(pred).map(result.label_map)
    assert (mapped.to_numpy() == result.events["label"].to_numpy()).all()
    very_high = result.events.loc[result.events["label"] == "Very High", "mag"].mean()
    assert very_high == pytest.approx(5.5, abs=0.2)


def test_train_respects_fixed_k(synthetic_events):
    result = train(synthetic_events, k=3, k_range=range(2, 5), sample_size=None)
    assert result.k == 3
    assert result.events["label"].str.startswith("Level").all()


def test_save_artifacts_roundtrip(synthetic_events, tmp_path):
    result = train(synthetic_events, k=4, k_range=range(2, 5), sample_size=None)
    meta = save_artifacts(result, tmp_path / "models", tmp_path / "processed", source="synthetic")

    pipeline = joblib.load(tmp_path / "models" / "kmeans_pipeline.joblib")
    np.testing.assert_array_equal(pipeline.predict(synthetic_events[["mag", "depth"]]),
                                  result.pipeline.predict(synthetic_events[["mag", "depth"]]))

    saved_meta = json.loads((tmp_path / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert saved_meta == meta
    assert saved_meta["params"]["k"] == 4
    assert sorted(saved_meta["label_map"].values()) == sorted(RISK_LEVELS)

    events = pd.read_parquet(tmp_path / "processed" / "earthquakes_with_cluster.parquet")
    assert len(events) == len(synthetic_events)
    assert {"cluster", "label", "time"} <= set(events.columns)


def test_load_catalog_renames_and_drops_incomplete(tmp_path):
    tsv = tmp_path / "katalog.tsv"
    tsv.write_text(
        "eventID\tdatetime\tlatitude\tlongitude\tmagnitude\tdepth\n"
        "a\t2020-01-01 00:00:00+00:00\t-6.1\t106.8\t4.5\t10\n"
        "b\t2020-01-02 00:00:00+00:00\t-6.2\t106.9\t\t20\n"
        "c\tbukan-tanggal\t-6.3\t107.0\t3.1\t30\n",
        encoding="utf-8",
    )
    df = load_catalog(tsv)
    assert list(df["eventID"]) == ["a"]
    assert {"time", "mag"} <= set(df.columns)
    assert str(df["time"].dt.tz) == "UTC"
