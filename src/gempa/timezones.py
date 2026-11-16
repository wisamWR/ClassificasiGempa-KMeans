"""Perkiraan zona waktu Indonesia (WIB/WITA/WIT) dari koordinat.

Batas zona waktu mengikuti batas provinsi, sehingga aturan berbasis koordinat ini hanya perkiraan.
Untuk gempa di laut, zona waktu daratan terdekat juga tidak selalu jelas.
"""
import numpy as np
import pandas as pd

UTC_OFFSET_HOURS = {"WIB": 7, "WITA": 8, "WIT": 9}


def local_timezone(lat, lon):
    """WIB (UTC+7), WITA (UTC+8), atau WIT (UTC+9) untuk tiap koordinat."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    # Maluku & Papua; Sangihe-Talaud (Sulawesi Utara) di utara Halmahera tetap WITA
    wit = (lon >= 125.5) & ~((lat > 2.0) & (lon < 127.0))
    # Sumatra, Jawa (sampai Banyuwangi), Kalimantan Barat & Tengah
    wib = (lon < 114.4) | ((lon < 115.3) & (lat > -1.9) & (lat < 0.5))
    return np.select([wit, wib], ["WIT", "WIB"], default="WITA")


def to_local_time(times, lat, lon):
    """(waktu setempat tanpa zona waktu, label zona waktu) untuk waktu UTC."""
    zone = local_timezone(lat, lon)
    offset = pd.to_timedelta(pd.Series(zone).map(UTC_OFFSET_HOURS).to_numpy(), unit="h")
    utc = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return (utc + offset).tz_localize(None), zone
