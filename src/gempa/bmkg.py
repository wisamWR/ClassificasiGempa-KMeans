"""Feed gempa publik BMKG (data.bmkg.go.id).

Ketentuan BMKG: cantumkan BMKG sebagai sumber data di aplikasi; maksimal 60 permintaan per menit per IP.
"""
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE_URL = "https://data.bmkg.go.id/DataMKG/TEWS/"
SHAKEMAP_URL = "https://static.bmkg.go.id/"
SOURCE_URL = "https://data.bmkg.go.id/gempabumi/"
FEEDS = {
    "terbaru": "autogempa.json",        # 1 gempa terbaru
    "terkini": "gempaterkini.json",     # 15 gempa M >= 5,0 terbaru
    "dirasakan": "gempadirasakan.json",  # 15 gempa dirasakan terbaru
}
# Magnitudo publik BMKG setara tipe "M" di katalog, sehingga dikonversi ke Mw dengan relasi yang sama
MAG_TYPE = "M"
COLUMNS = ["time", "latitude", "longitude", "mag", "depth", "region", "tsunami", "felt", "shakemap_url"]


class FeedError(Exception):
    """Feed BMKG tidak bisa diambil atau formatnya tidak dikenal."""


def fetch_feed(name, timeout=10):
    """JSON mentah satu feed BMKG."""
    url = BASE_URL + FEEDS[name]
    request = urllib.request.Request(url, headers={"User-Agent": "klasifikasi-gempa-kmeans-viewer"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise FeedError(f"Gagal mengambil {url}: {error}") from error


def _number(text):
    """'323 km' -> 323.0; '5.4' -> 5.4."""
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return np.nan


def parse_feed(payload):
    """DataFrame gempa dari JSON feed BMKG (satu gempa atau daftar gempa), urut dari yang terbaru."""
    try:
        events = payload["Infogempa"]["gempa"]
    except (KeyError, TypeError) as error:
        raise FeedError("Format feed BMKG tidak dikenal") from error
    if isinstance(events, dict):
        events = [events]

    rows = []
    for event in events:
        lat, _, lon = str(event.get("Coordinates", "")).partition(",")
        shakemap = event.get("Shakemap")
        felt = event.get("Dirasakan")
        rows.append({
            "time": event.get("DateTime"),
            "latitude": _number(lat),
            "longitude": _number(lon),
            "mag": _number(event.get("Magnitude")),
            "depth": _number(event.get("Kedalaman")),
            "region": event.get("Wilayah", ""),
            "tsunami": event.get("Potensi"),
            "felt": None if felt in (None, "", "-") else felt,
            "shakemap_url": SHAKEMAP_URL + shakemap if shakemap else None,
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag", "depth"])
    return df.sort_values("time", ascending=False).reset_index(drop=True)
