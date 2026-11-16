"""Mekanisme fokal: jenis sesar dari strike/dip/rake dengan klasifikasi Frohlich (1992).

Jalankan dari root project:  python -m gempa.focal
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gempa import paths

# Frohlich (1992): sumbu yang hampir vertikal menentukan jenis sesar
T_PLUNGE_REVERSE = 50.0
P_PLUNGE_NORMAL = 60.0
B_PLUNGE_STRIKE_SLIP = 60.0
FAULT_TYPES = ["naik", "turun", "geser", "oblik"]
FAULT_LABELS = {"naik": "Sesar naik", "turun": "Sesar turun", "geser": "Sesar geser", "oblik": "Oblik"}
NP_COLUMNS = ["strikeNP1", "dipNP1", "rakeNP1", "strikeNP2", "dipNP2", "rakeNP2"]


def _plunge(v):
    """Kemiringan (derajat dari horizontal) vektor dengan sumbu x = utara, y = timur, z = bawah."""
    return np.degrees(np.arcsin(np.clip(np.abs(v[2]) / np.linalg.norm(v, axis=0), 0, 1)))


def pbt_plunges(strike, dip, rake):
    """Plunge sumbu P, T, dan B (derajat) dari satu bidang nodal (Aki & Richards 1980)."""
    phi, delta, lam = (np.radians(np.asarray(v, dtype=float)) for v in (strike, dip, rake))
    normal = np.array([-np.sin(delta) * np.sin(phi), np.sin(delta) * np.cos(phi), -np.cos(delta)])
    slip = np.array([np.cos(lam) * np.cos(phi) + np.cos(delta) * np.sin(lam) * np.sin(phi),
                     np.cos(lam) * np.sin(phi) - np.cos(delta) * np.sin(lam) * np.cos(phi),
                     -np.sin(lam) * np.sin(delta)])
    return _plunge(normal - slip), _plunge(normal + slip), _plunge(np.cross(normal, slip, axis=0))


def faulting_type(strike, dip, rake):
    """'naik' (reverse/thrust), 'turun' (normal), 'geser' (strike-slip), atau 'oblik'."""
    p, t, b = pbt_plunges(strike, dip, rake)
    return np.select([t > T_PLUNGE_REVERSE, p > P_PLUNGE_NORMAL, b > B_PLUNGE_STRIKE_SLIP],
                     ["naik", "turun", "geser"], default="oblik")


def focal_mechanisms(catalog):
    """Event dengan bidang nodal lengkap + plunge P/T/B dan jenis sesar."""
    fm = catalog.dropna(subset=NP_COLUMNS)[["eventID", "time", "latitude", "longitude", "depth", "mag", *NP_COLUMNS]]
    fm = fm.reset_index(drop=True)
    p, t, b = pbt_plunges(fm["strikeNP1"], fm["dipNP1"], fm["rakeNP1"])
    fm["p_plunge"], fm["t_plunge"], fm["b_plunge"] = p, t, b
    fm["fault_type"] = pd.Categorical(faulting_type(fm["strikeNP1"], fm["dipNP1"], fm["rakeNP1"]),
                                      categories=FAULT_TYPES)
    return fm


def main(argv=None):
    from gempa.data import load_catalog

    parser = argparse.ArgumentParser(description="Klasifikasi jenis sesar dari mekanisme fokal.")
    parser.add_argument("--catalog", type=Path, default=paths.RAW_CATALOG)
    parser.add_argument("--output", type=Path, default=paths.FOCAL_MECHANISMS)
    args = parser.parse_args(argv)

    fm = focal_mechanisms(load_catalog(args.catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fm.to_parquet(args.output, index=False)
    print(f"{len(fm):,} mekanisme fokal tersimpan di {args.output}")
    print(fm["fault_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
