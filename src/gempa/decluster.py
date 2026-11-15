"""Declustering katalog dengan metode window Gardner & Knopoff (1974).

Rumus window mengikuti implementasi OpenQuake HMTK (GardnerKnopoffWindow):
    jarak  L = 10^(0.1238*M + 0.983) km
    waktu  T = 10^(0.5409*M - 0.547) hari untuk M < 6.5
           T = 10^(0.032*M + 2.7389)  hari untuk M >= 6.5
Event diproses dari magnitudo terbesar. Event lain dalam window jarak & waktu
setelahnya dianggap aftershock (seperti metode asli). Window mundur untuk foreshock
bisa diaktifkan lewat foreshock_ratio (proporsi dari window waktu, 1.0 = simetris).
"""
import numpy as np
import pandas as pd

from gempa.geo import haversine_km


def gk_distance_km(mag):
    return np.power(10.0, 0.1238 * np.asarray(mag, dtype=float) + 0.983)


def gk_time_days(mag):
    mag = np.asarray(mag, dtype=float)
    return np.where(mag >= 6.5,
                    np.power(10.0, 0.032 * mag + 2.7389),
                    np.power(10.0, 0.5409 * mag - 0.547))


def gardner_knopoff(times, latitudes, longitudes, mags, foreshock_ratio=0.0):
    """Kembalikan (sequence_id, is_mainshock) per event, urutan sama dengan input.

    sequence_id = indeks (posisi input) mainshock pemilik sekuens; event independen
    menjadi mainshock bagi dirinya sendiri.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    mags = np.asarray(mags, dtype=float)
    n = len(mags)

    # Waktu dalam hari, diurutkan agar window waktu bisa dicari dengan searchsorted
    t_days = ((times - times.min()) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    order = np.argsort(t_days, kind="stable")
    t_sorted = t_days[order]

    sequence = np.full(n, -1, dtype=np.int64)
    win_t = gk_time_days(mags)
    win_d = gk_distance_km(mags)

    # Magnitudo terbesar dulu; seri diputus oleh waktu kejadian (lebih awal dulu)
    for i in np.lexsort((t_days, -mags)):
        if sequence[i] != -1:
            continue
        sequence[i] = i
        lo = np.searchsorted(t_sorted, t_days[i] - foreshock_ratio * win_t[i], side="left")
        hi = np.searchsorted(t_sorted, t_days[i] + win_t[i], side="right")
        candidates = order[lo:hi]
        candidates = candidates[sequence[candidates] == -1]
        if len(candidates) == 0:
            continue
        near = haversine_km(lat[i], lon[i], lat[candidates], lon[candidates]) <= win_d[i]
        sequence[candidates[near]] = i

    is_mainshock = sequence == np.arange(n)
    return sequence, is_mainshock
