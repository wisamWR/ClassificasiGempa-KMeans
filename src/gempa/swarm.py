"""Clustering spatio-temporal ST-DBSCAN (Birant & Kut 2007) untuk deret gempa dan deteksi swarm.

Dua gempa bertetangga bila jarak 3D <= eps_km dan selisih waktunya <= eps_days. Jenis klaster:
- "secondary": sebagian besar gempanya susulan (Gardner-Knopoff) dari gempa utama lebih besar di luar klaster,
  mis. potongan deret susulan di sepanjang ruptur yang panjang;
- "swarm": tidak ada gempa yang jauh lebih besar dari lainnya (selisih Mw terbesar dan kedua < SWARM_MAX_GAP),
  yaitu swarm atau doublet;
- "mainshock": deret gempa utama-susulan.

Jalankan dari root project:  python -m gempa.swarm
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph

from gempa import paths
from gempa.zoning import to_ecef_km

EPS_KM = 20.0
EPS_DAYS = 2.0
MIN_SAMPLES = 10
SWARM_MAX_GAP = 0.5
KINDS = {"swarm": "Swarm / doublet", "mainshock": "Gempa utama–susulan", "secondary": "Bagian deret gempa lain"}


def st_dbscan(times, lat, lon, depth, eps_km=EPS_KM, eps_days=EPS_DAYS, min_samples=MIN_SAMPLES):
    """Label klaster ST-DBSCAN tiap gempa (-1 = noise)."""
    X = to_ecef_km(lat, lon, depth)
    days = (pd.DatetimeIndex(pd.to_datetime(times, utc=True)) - pd.Timestamp("1970-01-01", tz="UTC")) \
        / pd.Timedelta(days=1)
    days = np.asarray(days, dtype=float)
    graph = radius_neighbors_graph(X, eps_km, mode="distance", include_self=False).tocoo()
    keep = np.abs(days[graph.row] - days[graph.col]) <= eps_days
    # Jarak 0 (lokasi sama persis) dianggap "bukan tetangga" oleh matriks sparse, jadi diberi nilai kecil
    graph = coo_matrix((np.maximum(graph.data[keep], 1e-6), (graph.row[keep], graph.col[keep])),
                       shape=graph.shape).tocsr()
    return DBSCAN(eps=eps_km, min_samples=min_samples, metric="precomputed").fit_predict(graph)


def _kind(group, largest, gap, mw_by_event):
    sequence = group["sequence_id"].mode()
    if not sequence.empty and sequence.iloc[0] not in set(group["eventID"]):
        parent_mw = mw_by_event.get(sequence.iloc[0], np.nan)
        dependent = (group["sequence_id"] == sequence.iloc[0]).mean()
        if dependent > 0.5 and parent_mw > largest["mw"]:
            return "secondary"
    return "swarm" if gap < SWARM_MAX_GAP else "mainshock"


def summarize_clusters(events, labels):
    """Ringkasan tiap klaster: ukuran, rentang waktu, Mw terbesar, selisih Mw terbesar-kedua, dan jenisnya.

    events butuh kolom eventID, time, latitude, longitude, depth, mw, location, dan sequence_id.
    """
    mw_by_event = dict(zip(events["eventID"], events["mw"]))
    clustered = events.assign(st_cluster=labels)
    clustered = clustered[clustered["st_cluster"] >= 0]
    rows = []
    for cluster, group in clustered.groupby("st_cluster"):
        ordered = group.sort_values("mw", ascending=False)
        largest = ordered.iloc[0]
        gap = float(largest["mw"] - ordered.iloc[1]["mw"])
        start, end = group["time"].min(), group["time"].max()
        rows.append({
            "st_cluster": int(cluster),
            "n_events": len(group),
            "start": start,
            "end": end,
            "duration_days": (end - start).total_seconds() / 86400,
            "mw_max": float(largest["mw"]),
            "mw_gap": gap,
            "largest_event": largest["eventID"],
            "latitude": group["latitude"].mean(),
            "longitude": group["longitude"].mean(),
            "depth_median": float(group["depth"].median()),
            "location": group["location"].mode().iloc[0],
            "kind": _kind(group, largest, gap, mw_by_event),
        })
    return pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)


def run(events, eps_km=EPS_KM, eps_days=EPS_DAYS, min_samples=MIN_SAMPLES):
    """(label klaster per event, ringkasan klaster) untuk event yang diberikan."""
    labels = st_dbscan(events["time"], events["latitude"], events["longitude"], events["depth"],
                       eps_km, eps_days, min_samples)
    event_labels = pd.DataFrame({"eventID": events["eventID"].to_numpy(), "st_cluster": labels})
    return event_labels[event_labels["st_cluster"] >= 0].reset_index(drop=True), summarize_clusters(events, labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Klaster spatio-temporal (ST-DBSCAN) dan deteksi swarm.")
    parser.add_argument("--events", type=Path, default=paths.PROCESSED_EVENTS)
    parser.add_argument("--processed-dir", type=Path, default=paths.PROCESSED_DIR)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.events)
    # Hanya gempa di atas Mc tahunan dan lolos verifikasi: kepadatan tidak bias oleh perubahan jaringan
    events = df[df["above_mc"] & ~df["suspect_event"].fillna(False).astype(bool)].reset_index(drop=True)
    event_labels, clusters = run(events)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    event_labels.to_parquet(args.processed_dir / paths.ST_CLUSTER_EVENTS.name, index=False)
    clusters.to_parquet(args.processed_dir / paths.ST_CLUSTERS.name, index=False)
    print(f"{len(clusters)} klaster dari {len(events):,} gempa ({len(event_labels) / len(events):.1%} masuk klaster)")
    print(clusters["kind"].map(KINDS).value_counts().to_string())


if __name__ == "__main__":
    main()
