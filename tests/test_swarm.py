import numpy as np
import pandas as pd

from gempa.swarm import run


def _burst(rng, prefix, lat, lon, start, mags, sequence_id=None):
    n = len(mags)
    ids = [f"{prefix}{i}" for i in range(n)]
    return pd.DataFrame({
        "eventID": ids,
        "time": pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 1, n)), unit="D"),
        "latitude": lat + rng.normal(0, 0.02, n), "longitude": lon + rng.normal(0, 0.02, n),
        "depth": np.full(n, 10.0), "mw": mags, "location": prefix,
        "sequence_id": sequence_id if sequence_id is not None else [ids[int(np.argmax(mags))]] * n,
    })


def test_st_dbscan_separates_in_space_and_time_and_classifies():
    rng = np.random.default_rng(0)
    parent = pd.DataFrame({"eventID": ["P"], "time": [pd.Timestamp("2019-12-31", tz="UTC")], "latitude": [5.0],
                           "longitude": [100.0], "depth": [10.0], "mw": [7.0], "location": ["P"], "sequence_id": ["P"]})
    events = pd.concat([
        parent,
        _burst(rng, "main", -7.0, 107.0, "2020-01-01", [6.0] + [4.0] * 20),        # gempa utama + susulan
        _burst(rng, "swarm", -2.0, 120.0, "2020-01-01", rng.uniform(4.0, 4.3, 20)),  # tanpa gempa dominan
        _burst(rng, "late", -7.0, 107.0, "2020-03-01", [4.0] * 5),                   # tempat sama, 2 bulan kemudian
        _burst(rng, "child", -8.0, 110.0, "2020-01-01", [4.5] + [4.0] * 15, sequence_id="P"),
    ], ignore_index=True)
    labels, clusters = run(events, eps_km=20, eps_days=2, min_samples=5)

    cluster_of = events.merge(labels, on="eventID").groupby("location")["st_cluster"].agg(set)
    assert set(cluster_of.index) == {"main", "swarm", "late", "child"}  # gempa utama P sendirian = noise
    assert all(len(ids) == 1 for ids in cluster_of)
    # Tempat sama tetapi 2 bulan kemudian: klaster terpisah
    assert cluster_of["late"] != cluster_of["main"]
    kinds = dict(zip(clusters["location"], clusters["kind"]))
    assert kinds == {"main": "mainshock", "swarm": "swarm", "late": "swarm", "child": "secondary"}
    main = clusters.set_index("location").loc["main"]
    assert main["n_events"] == 21 and main["mw_max"] == 6.0 and main["mw_gap"] == 2.0
