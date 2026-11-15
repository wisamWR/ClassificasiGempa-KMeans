import pandas as pd

# Nama kolom di katalog BMKG -> nama yang dipakai di seluruh pipeline
COLUMN_MAP = {
    "datetime": "time",
    "magnitude": "mag",
}
NUMERIC_COLUMNS = ["latitude", "longitude", "depth", "mag"]
REQUIRED_COLUMNS = ["time", "latitude", "longitude", "mag", "depth"]


def load_catalog(path):
    """Baca katalog TSV BMKG, samakan nama kolom, konversi tipe, buang baris tidak lengkap."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns=COLUMN_MAP)

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
