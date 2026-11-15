"""Estimasi magnitude of completeness (Mc)."""
import numpy as np
import pandas as pd

BIN_WIDTH = 0.1
# Koreksi +0.2 untuk metode maximum curvature yang cenderung underestimate
# (Woessner & Wiemer, 2005)
MAXC_CORRECTION = 0.2
MIN_EVENTS = 50


def mc_maxc(mags, bin_width=BIN_WIDTH, correction=MAXC_CORRECTION):
    """Maximum curvature (Wiemer & Wyss, 2000): bin magnitudo dengan frekuensi terbanyak + koreksi."""
    mags = np.asarray(mags, dtype=float)
    mags = mags[~np.isnan(mags)]
    if len(mags) == 0:
        return np.nan
    bins = np.round(mags / bin_width).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    return round(values[np.argmax(counts)] * bin_width + correction, 2)


def mc_by_year(times, mags, min_events=MIN_EVENTS, **kwargs):
    """Mc per tahun; tahun dengan event < min_events memakai Mc seluruh katalog."""
    df = pd.DataFrame({"year": pd.DatetimeIndex(times).year, "mag": np.asarray(mags, dtype=float)})
    overall = mc_maxc(df["mag"], **kwargs)
    rows = []
    for year, group in df.groupby("year"):
        mc = mc_maxc(group["mag"], **kwargs) if len(group) >= min_events else overall
        rows.append({"year": int(year), "n_events": len(group), "mc": mc})
    return pd.DataFrame(rows), overall
