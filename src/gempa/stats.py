"""Statistik Gutenberg-Richter: b-value, laju tahunan, periode ulang, probabilitas Poisson."""
import numpy as np
import pandas as pd

LOG10_E = np.log10(np.e)


def b_value(mags, mc):
    """b-value maximum likelihood Aki (1965) untuk magnitudo kontinu >= mc.

    Mengembalikan (b, sigma_b, n); sigma_b menurut Shi & Bolt (1982).
    """
    m = np.asarray(mags, dtype=float)
    m = m[m >= mc]
    n = len(m)
    if n < 2 or m.mean() <= mc:
        return np.nan, np.nan, n
    b = LOG10_E / (m.mean() - mc)
    sigma = 2.3 * b ** 2 * np.sqrt(np.sum((m - m.mean()) ** 2) / (n * (n - 1)))
    return float(b), float(sigma), n


def complete_exposure_years(mc_by_year, mc, start, end):
    """Total tahun (pecahan) dalam periode katalog yang sudah lengkap pada magnitudo mc.

    mc_by_year: dict {tahun: Mc tahun tersebut}. Tahun pertama/terakhir dihitung sebagian
    sesuai rentang katalog [start, end].
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total = 0.0
    for year, mc_year in mc_by_year.items():
        if mc_year > mc + 1e-9:
            continue
        lo = max(start, pd.Timestamp(year=year, month=1, day=1, tz=start.tz))
        hi = min(end, pd.Timestamp(year=year + 1, month=1, day=1, tz=start.tz))
        if hi > lo:
            total += (hi - lo).total_seconds() / (365.25 * 86400)
    return total


def annual_rate(rate_at_mc, b, mc, m):
    """Laju tahunan gempa >= m dari relasi Gutenberg-Richter log N = a - b M."""
    return rate_at_mc * np.power(10.0, -b * (np.asarray(m, dtype=float) - mc))


def poisson_probability(rate, years):
    """Probabilitas minimal satu kejadian dalam `years` tahun (proses Poisson)."""
    return 1.0 - np.exp(-np.asarray(rate, dtype=float) * years)
