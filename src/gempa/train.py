"""Training K-Means klaster risiko gempa (fitur: magnitudo & kedalaman).

Jalankan dari root project:  python -m gempa.train
"""
import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from gempa import paths
from gempa.data import load_catalog
from gempa.intensity import epicentral_intensity

FEATURES = ["mag", "depth"]
RISK_LEVELS = ["Low", "Moderate", "High", "Very High"]
RANDOM_STATE = 42
K_RANGE = range(2, 8)
SCAN_N_INIT = 3      # cukup untuk membandingkan k, jauh lebih cepat
FINAL_N_INIT = 10
# Silhouette O(n^2): hitung pada sampel acak tetap (seed sama) agar cepat & konsisten
SILHOUETTE_SAMPLE = 20_000
# Kolom data olahan yang disimpan untuk aplikasi
PROCESSED_COLUMNS = ["eventID", "time", "latitude", "longitude", "mag", "mag_type",
                     "depth", "location", "cluster", "label"]


@dataclass
class TrainResult:
    pipeline: Pipeline
    events: pd.DataFrame
    summary: pd.DataFrame
    k_scan: pd.DataFrame
    metrics: dict
    params: dict

    @property
    def k(self):
        return self.params["k"]

    @property
    def label_map(self):
        return dict(zip(self.summary["cluster"], self.summary["label"]))


def build_pipeline(k, n_init=FINAL_N_INIT, random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=random_state)),
    ])


def silhouette(X_scaled, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE):
    if not sample_size or sample_size >= len(X_scaled):
        sample_size = None
    return float(silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=random_state))


def scan_k(X, k_range=K_RANGE, n_init=SCAN_N_INIT, sample_size=SILHOUETTE_SAMPLE,
           random_state=RANDOM_STATE):
    """Inertia (elbow) dan silhouette untuk tiap kandidat k."""
    X_scaled = MinMaxScaler().fit_transform(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X_scaled)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": silhouette(X_scaled, km.labels_, sample_size, random_state),
        })
    return pd.DataFrame(rows)


def label_names(k):
    if k == len(RISK_LEVELS):
        return list(RISK_LEVELS)
    return [f"Level {i + 1}" for i in range(k)]


def summarize_clusters(events):
    """Rata-rata fitur per klaster + severity I0 (Gutenberg-Richter) dari rata-rata tersebut.

    Label diberikan berurutan dari severity terendah ke tertinggi.
    """
    summary = (
        events.groupby("cluster")
        .agg(mag=("mag", "mean"), depth=("depth", "mean"),
             latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )
    summary["severity_score"] = epicentral_intensity(summary["mag"], summary["depth"])
    summary = summary.sort_values("severity_score").reset_index(drop=True)
    summary["label"] = label_names(len(summary))
    return summary


def evaluate(X_scaled, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE):
    return {
        "silhouette": silhouette(X_scaled, labels, sample_size, random_state),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }


def train(df, k=None, k_range=K_RANGE, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE,
          k_scan=None):
    """Pilih k (silhouette tertinggi bila k tidak diberikan), fit pipeline, beri label klaster.

    k_scan hasil scan_k() bisa diberikan agar pencarian k tidak dihitung ulang.
    """
    X = df[FEATURES]
    if k_scan is None:
        k_scan = scan_k(X, k_range, sample_size=sample_size, random_state=random_state)
    if k is None:
        k = int(k_scan.loc[k_scan["silhouette"].idxmax(), "k"])

    pipeline = build_pipeline(k, random_state=random_state).fit(X)
    events = df.copy()
    events["cluster"] = pipeline["kmeans"].labels_

    summary = summarize_clusters(events)
    events["label"] = events["cluster"].map(dict(zip(summary["cluster"], summary["label"])))

    X_scaled = pipeline["scaler"].transform(X)
    metrics = evaluate(X_scaled, events["cluster"].to_numpy(), sample_size, random_state)
    params = {
        "k": k,
        "k_range": [min(k_range), max(k_range)],
        "scan_n_init": SCAN_N_INIT,
        "final_n_init": FINAL_N_INIT,
        "random_state": random_state,
        "silhouette_sample_size": sample_size,
    }
    return TrainResult(pipeline, events, summary, k_scan, metrics, params)


def save_artifacts(result, models_dir=paths.MODELS_DIR, processed_dir=paths.PROCESSED_DIR,
                   source=paths.RAW_CATALOG.name):
    models_dir, processed_dir = Path(models_dir), Path(processed_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.pipeline, models_dir / paths.PIPELINE_FILE.name)
    result.summary.to_csv(models_dir / paths.CLUSTER_INFO_FILE.name, index=False)
    columns = [c for c in PROCESSED_COLUMNS if c in result.events.columns]
    result.events[columns].to_parquet(processed_dir / paths.PROCESSED_EVENTS.name, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "source": source,
        "n_events": len(result.events),
        "time_range": [result.events["time"].min().isoformat(), result.events["time"].max().isoformat()],
        "features": FEATURES,
        "scaler": "MinMaxScaler",
        "params": result.params,
        "k_scan": result.k_scan.to_dict("records"),
        "metrics": result.metrics,
        "label_map": {str(int(c)): label for c, label in result.label_map.items()},
    }
    with open(models_dir / paths.METADATA_FILE.name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Latih ulang model K-Means klaster risiko gempa.")
    parser.add_argument("--catalog", type=Path, default=paths.RAW_CATALOG, help="Path katalog TSV BMKG")
    parser.add_argument("--k", type=int, default=None, help="Jumlah klaster (default: pilih via silhouette)")
    parser.add_argument("--silhouette-sample", type=int, default=SILHOUETTE_SAMPLE,
                        help="Ukuran sampel silhouette (0 = seluruh data, lambat)")
    parser.add_argument("--models-dir", type=Path, default=paths.MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=paths.PROCESSED_DIR)
    args = parser.parse_args(argv)

    df = load_catalog(args.catalog)
    print(f"Loaded {len(df):,} events from {args.catalog}")

    result = train(df, k=args.k, sample_size=args.silhouette_sample)
    print("\nPencarian k:")
    print(result.k_scan.to_string(index=False))
    print(f"\nk terpilih: {result.k}")
    print(result.summary.to_string(index=False))
    print("\nMetrik:", {name: round(value, 4) for name, value in result.metrics.items()})

    save_artifacts(result, args.models_dir, args.processed_dir, source=args.catalog.name)
    print(f"\nArtifact tersimpan di {args.models_dir} dan {args.processed_dir}")


if __name__ == "__main__":
    main()
