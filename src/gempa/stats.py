"""Statistik kegempaan: Gutenberg-Richter (b-value, laju, periode ulang, Poisson) dan peluruhan Omori-Utsu."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOG10_E = np.log10(np.e)
# Rentang wajar parameter Omori-Utsu; fit yang menyentuh batas menandakan deret tidak meluruh (mis. swarm)
OMORI_C_BOUNDS = (1e-4, 10.0)
OMORI_P_BOUNDS = (0.2, 3.0)


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


def _omori_integral(c, p, start, end):
    """Integral (t + c)^-p dari start sampai end."""
    if abs(p - 1) < 1e-6:
        return np.log((end + c) / (start + c))
    return ((end + c) ** (1 - p) - (start + c) ** (1 - p)) / (1 - p)


def omori_mle(times, start, end):
    """Parameter Omori-Utsu n(t) = K / (t + c)^p dengan maximum likelihood (Ogata 1983).

    times: waktu gempa susulan (hari sejak gempa utama) di dalam [start, end].
    Mengembalikan (K, c, p, n). K per hari; NaN bila gempa susulan terlalu sedikit. c dan p dibatasi
    OMORI_C_BOUNDS dan OMORI_P_BOUNDS.
    """
    t = np.asarray(times, dtype=float)
    t = t[(t >= start) & (t <= end)]
    n = len(t)
    if n < 10:
        return np.nan, np.nan, np.nan, n

    def neg_log_likelihood(params):
        c, p = np.exp(params[0]), params[1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            integral = _omori_integral(c, p, start, end)
            if not np.isfinite(integral) or integral <= 0:
                return 1e300
            k = n / integral  # K MLE untuk c dan p tertentu
            return -(n * np.log(k) - p * np.log(t + c).sum() - n)

    bounds = [tuple(np.log(OMORI_C_BOUNDS)), OMORI_P_BOUNDS]
    best = min((minimize(neg_log_likelihood, [np.log(c0), p0], method="Powell", bounds=bounds,
                         options={"xtol": 1e-6, "ftol": 1e-9, "maxiter": 5000})
                for c0 in (0.01, 0.1, 1.0) for p0 in (0.8, 1.1)), key=lambda r: r.fun)
    c, p = float(np.exp(best.x[0])), float(best.x[1])
    return n / _omori_integral(c, p, start, end), c, p, n
