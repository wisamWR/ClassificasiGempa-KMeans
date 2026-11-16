from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_CATALOG = ROOT_DIR / "data" / "raw" / "katalog_gempa_v2.tsv"
USGS_REFERENCE = ROOT_DIR / "data" / "raw" / "usgs_comcat_m4.0_2008_2025.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PROCESSED_EVENTS = PROCESSED_DIR / "earthquakes_with_cluster.parquet"
CLEAN_CATALOG = PROCESSED_DIR / "catalog_clean.parquet"
CLEAN_REPORT = PROCESSED_DIR / "catalog_report.json"
ZONE_EVENTS = PROCESSED_DIR / "zone_events.parquet"
FOCAL_MECHANISMS = PROCESSED_DIR / "focal_mechanisms.parquet"
ZONING_MODEL = MODELS_DIR / "zoning_kmeans.joblib"
ZONE_STATS = MODELS_DIR / "zone_stats.csv"
ZONES_GEOJSON = MODELS_DIR / "zones.geojson"
ZONING_METADATA = MODELS_DIR / "zoning_metadata.json"
PIPELINE_FILE = MODELS_DIR / "kmeans_pipeline.joblib"
CLUSTER_INFO_FILE = MODELS_DIR / "cluster_info.csv"
METADATA_FILE = MODELS_DIR / "metadata.json"
