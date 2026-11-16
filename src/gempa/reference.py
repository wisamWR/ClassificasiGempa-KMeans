"""Katalog pembanding independen USGS ComCat untuk verifikasi gempa besar.

Unduh (sekali, hasil disimpan di data/raw/):  python -m gempa.reference
Data USGS berstatus domain publik.
"""
import argparse
import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from gempa import paths
from gempa.geo import haversine_km

FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Wilayah katalog BMKG (sama dengan batas validasi di gempa.clean)
BOUNDS = {"minlatitude": -12, "maxlatitude": 7, "minlongitude": 94, "maxlongitude": 142}
MIN_MAGNITUDE = 4.0
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "id", "place"]
# Toleransi pencocokan event BMKG <-> USGS
MATCH_SECONDS = 60
MATCH_KM = 150


def fetch_usgs(start_year=2008, end_year=2025, min_magnitude=MIN_MAGNITUDE):
    """Unduh katalog USGS per tahun (batas API 20.000 event per permintaan)."""
    frames = []
    for year in range(start_year, end_year + 1):
        params = {"format": "csv", "starttime": f"{year}-01-01", "endtime": f"{year + 1}-01-01",
                  "minmagnitude": min_magnitude, "orderby": "time-asc", **BOUNDS}
        url = f"{FDSN_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as response:
            frames.append(pd.read_csv(io.StringIO(response.read().decode("utf-8")), usecols=COLUMNS))
    return pd.concat(frames, ignore_index=True)


def load_reference(path=paths.USGS_REFERENCE):
    ref = pd.read_csv(path)
    ref["time"] = pd.to_datetime(ref["time"], utc=True, format="ISO8601")
    return ref


def _epoch_seconds(times):
    # Satuan eksplisit: resolusi datetime pandas bisa ns atau us
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def has_reference_match(times, lat, lon, ref, max_seconds=MATCH_SECONDS, max_km=MATCH_KM):
    """True bila ada event USGS dalam +-max_seconds dan <= max_km dari tiap event."""
    t = _epoch_seconds(times)
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    ref = ref.sort_values("time")
    rt = _epoch_seconds(ref["time"])
    rlat, rlon = ref["latitude"].to_numpy(), ref["longitude"].to_numpy()

    lo = np.searchsorted(rt, t - max_seconds, side="left")
    hi = np.searchsorted(rt, t + max_seconds, side="right")
    matched = np.zeros(len(t), dtype=bool)
    for i in np.flatnonzero(hi > lo):
        d = haversine_km(lat[i], lon[i], rlat[lo[i]:hi[i]], rlon[lo[i]:hi[i]])
        matched[i] = bool((d <= max_km).any())
    return matched


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unduh katalog pembanding USGS ComCat.")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=paths.USGS_REFERENCE)
    args = parser.parse_args(argv)

    ref = fetch_usgs(args.start_year, args.end_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(args.output, index=False)
    print(f"{len(ref):,} event USGS M>={MIN_MAGNITUDE} tersimpan di {args.output}")


if __name__ == "__main__":
    main()
