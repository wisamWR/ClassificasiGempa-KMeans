"""Homogenisasi magnitudo katalog BMKG ke magnitudo momen (Mw).

Relasi konversi (regresi linear OLS) dari:
    Taruna, R. M. & Pratiwi, A. (2021). Konversi Empiris Summary Magnitude, Local Magnitude,
    Body-Wave Magnitude, Surface Magnitude, dan Moment Magnitude Menggunakan Data Gempabumi
    1922-2020 di Nusa Tenggara Barat. Jurnal Sains Teknologi & Lingkungan, 7(1), 1-12.
    https://doi.org/10.29303/jstl.v7i1.198

Relasi tersebut diuji pada pasangan magnitudo katalog ini yang punya Mw mekanisme fokal
(magTypeFM == "Mw"): bias rata-rata M -0.016, MLv -0.015, mb +0.020 (lihat notebook data_cleaning).
"""
import numpy as np
import pandas as pd

TARUNA_2021 = "Taruna & Pratiwi (2021)"

# mag_type -> (slope, intercept, rentang valid [min, max], sumber); Mw = slope * mag + intercept
CONVERSIONS = {
    "M": (1.0166, -0.2207, (2.93, 7.16), TARUNA_2021),
    "MLv": (0.7353, 1.0722, (2.93, 7.57), TARUNA_2021),
    "ML": (0.6424, 1.5595, (1.75, 7.82), TARUNA_2021),
    "mb": (1.1993, -1.2261, (3.67, 6.99), TARUNA_2021),
}
# Sudah berupa (turunan) magnitudo momen, dipakai apa adanya
MW_TYPES = {"Mw", "Mw(mB)", "Mwp"}
# Jumlah sangat sedikit (< 15 event) tanpa relasi yang teruji: dipakai apa adanya dan ditandai
APPROX_TYPES = {"mB", "Mjma"}


def to_mw(mag, mag_type):
    """Konversi magnitudo ke Mw per tipe.

    Mengembalikan DataFrame dengan kolom:
      mw              magnitudo momen hasil konversi
      mw_method       'conversion' | 'native' | 'approx' | 'unknown'
      mw_extrapolated True bila magnitudo di luar rentang valid relasi konversi
    """
    mag = pd.Series(mag, dtype=float).reset_index(drop=True)
    mag_type = pd.Series(mag_type, dtype="object").reset_index(drop=True)

    mw = pd.Series(np.nan, index=mag.index)
    method = pd.Series("unknown", index=mag.index, dtype="object")
    extrapolated = pd.Series(False, index=mag.index)

    for mtype, (slope, intercept, (lo, hi), _) in CONVERSIONS.items():
        sel = mag_type == mtype
        mw[sel] = slope * mag[sel] + intercept
        method[sel] = "conversion"
        extrapolated[sel] = (mag[sel] < lo) | (mag[sel] > hi)

    native = mag_type.isin(MW_TYPES)
    mw[native] = mag[native]
    method[native] = "native"

    approx = mag_type.isin(APPROX_TYPES)
    mw[approx] = mag[approx]
    method[approx] = "approx"

    return pd.DataFrame({"mw": mw, "mw_method": method, "mw_extrapolated": extrapolated})
