import numpy as np

# Kedalaman minimum agar log10(h) tidak singular di h = 0
MIN_DEPTH_KM = 5.0


def epicentral_intensity(mag, depth_km, min_depth_km=MIN_DEPTH_KM):
    """Estimasi intensitas episentrum Gutenberg-Richter (1956).

    I0 = 1.5 * M - 3.5 * log10(h) + 3.0, dengan h = kedalaman (km).
    Menerima skalar maupun array.
    """
    depth = np.maximum(np.asarray(depth_km, dtype=float), min_depth_km)
    return 1.5 * np.asarray(mag, dtype=float) - 3.5 * np.log10(depth) + 3.0
