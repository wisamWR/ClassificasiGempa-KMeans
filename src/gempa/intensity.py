import numpy as np
import pandas as pd

# Kedalaman minimum agar log10(h) tidak singular di h = 0
MIN_DEPTH_KM = 5.0

# Kelas intensitas (skala Modified Mercalli) untuk I0 hasil estimasi, urut lemah -> kuat.
# Batas bawah memakai pembulatan: I0 < 2.5 dibaca sebagai intensitas I-II, dst.
MMI_CLASSES = ["I-II", "III-IV", "V-VI", "VII+"]
MMI_BOUNDS = [2.5, 4.5, 6.5]
MMI_DESCRIPTIONS = {
    "I-II": "Tidak terasa atau hanya dirasakan sedikit orang dalam keadaan diam.",
    "III-IV": "Terasa di dalam rumah, benda tergantung bergoyang, jendela bergetar.",
    "V-VI": "Dirasakan hampir semua orang, benda jatuh, kerusakan ringan.",
    "VII+": "Berpotensi merusak bangunan, terutama yang tidak tahan gempa.",
}


def epicentral_intensity(mag, depth_km, min_depth_km=MIN_DEPTH_KM):
    """Estimasi intensitas episentrum Gutenberg-Richter (1956).

    I0 = 1.5 * M - 3.5 * log10(h) + 3.0, dengan h = kedalaman (km).
    Menerima skalar maupun array.
    """
    depth = np.maximum(np.asarray(depth_km, dtype=float), min_depth_km)
    return 1.5 * np.asarray(mag, dtype=float) - 3.5 * np.log10(depth) + 3.0


def mmi_class(intensity):
    """Petakan I0 ke kelas MMI (lihat MMI_CLASSES); menerima skalar maupun array."""
    idx = np.digitize(np.asarray(intensity, dtype=float), MMI_BOUNDS)
    classes = np.asarray(MMI_CLASSES, dtype=object)[idx]
    if np.ndim(classes) == 0:
        return str(classes)
    return pd.Categorical(classes, categories=MMI_CLASSES, ordered=True)
