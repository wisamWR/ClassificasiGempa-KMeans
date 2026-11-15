"""Pembersihan katalog: validasi, homogenisasi Mw, Mc, declustering, intensitas per event.

Jalankan dari root project:  python -m gempa.clean
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gempa import paths
from gempa.completeness import mc_by_year
from gempa.data import load_catalog
from gempa.decluster import gardner_knopoff
from gempa.intensity import epicentral_intensity, mmi_class
from gempa.magnitude import CONVERSIONS, to_mw

# Kedalaman default BMKG bila kedalaman tidak bisa ditentukan
FIXED_DEPTHS_KM = (0, 10)
# Indikator lokasi kurang baik: celah azimut besar atau sedikit fase
MAX_AZIMUTH_GAP = 180
MIN_PHASECOUNT = 5
# Event besar dengan fase sangat sedikit DAN celah azimut besar kemungkinan event semu
# (mis. bmg2019wikz "Seram M7.2": 4 fase, gap 208°, 8 menit setelah M7.1 Laut Maluku).
# Fase saja tidak cukup: median fase seluruh event M>=5 tahun 2019-2021 hanya ~10.
SUSPECT_MIN_MW = 5.0
SUSPECT_MAX_PHASECOUNT = 10
# Window mundur declustering (0 = aftershock saja, seperti Gardner & Knopoff asli).
# Window simetris membuat Cianjur 2022 (Mw 5.6) terbaca sebagai foreshock event Mw 5.97.
FORESHOCK_RATIO = 0.0
# Batas wilayah katalog
LAT_RANGE = (-12.0, 7.0)
LON_RANGE = (94.0, 142.0)


def clean_catalog(df):
    """Tambahkan kolom kualitas, Mw, Mc, declustering, dan intensitas. Mengembalikan (df, report)."""
    df = df.copy()
    n_raw = len(df)

    # Validasi dasar
    duplicated = df["eventID"].duplicated(keep="first")
    in_area = df["latitude"].between(*LAT_RANGE) & df["longitude"].between(*LON_RANGE)
    negative_depth = df["depth"] < 0
    valid = ~duplicated & in_area & ~negative_depth
    df = df[valid].reset_index(drop=True)

    df["depth_fixed"] = df["depth"].isin(FIXED_DEPTHS_KM)
    df["low_quality_location"] = (df["azimuth_gap"] > MAX_AZIMUTH_GAP) | (df["phasecount"] < MIN_PHASECOUNT)

    # Homogenisasi magnitudo
    df = pd.concat([df, to_mw(df["mag"], df["mag_type"])], axis=1)
    df = df[df["mw"].notna()].reset_index(drop=True)

    # Magnitude of completeness per tahun
    mc_table, mc_overall = mc_by_year(df["time"], df["mw"])
    mc_of_year = dict(zip(mc_table["year"], mc_table["mc"]))
    df["mc"] = df["time"].dt.year.map(mc_of_year)
    df["above_mc"] = df["mw"] >= df["mc"] - 1e-9

    df["suspect_event"] = (
        (df["mw"] >= SUSPECT_MIN_MW)
        & (df["phasecount"] < SUSPECT_MAX_PHASECOUNT)
        & (df["azimuth_gap"] > MAX_AZIMUTH_GAP)
    )

    # Declustering Gardner-Knopoff tanpa event semu (window besarnya bisa menyerap sekuens lain)
    real = df[~df["suspect_event"]]
    sequence, is_mainshock = gardner_knopoff(real["time"], real["latitude"], real["longitude"], real["mw"],
                                             foreshock_ratio=FORESHOCK_RATIO)
    _, is_mainshock_sym = gardner_knopoff(real["time"], real["latitude"], real["longitude"], real["mw"],
                                          foreshock_ratio=1.0)
    df["sequence_id"] = pd.Series(real["eventID"].to_numpy()[sequence], index=real.index)
    df["is_mainshock"] = pd.Series(is_mainshock, index=real.index).reindex(df.index, fill_value=False)

    # Estimasi intensitas episentrum per event (absolut, bukan relatif antar-klaster)
    df["intensity"] = epicentral_intensity(df["mw"], df["depth"])
    df["mmi_class"] = mmi_class(df["intensity"])

    sequences = df.groupby("sequence_id").size()
    largest = (
        df[df["is_mainshock"]].set_index("eventID")
        .assign(n_events=sequences)
        .sort_values("n_events", ascending=False)
        .head(10)
    )
    report = {
        "n_raw": n_raw,
        "n_clean": len(df),
        "removed": {
            "duplicated_eventID": int(duplicated.sum()),
            "outside_area": int((~in_area).sum()),
            "negative_depth": int(negative_depth.sum()),
        },
        "depth_fixed": int(df["depth_fixed"].sum()),
        "low_quality_location": int(df["low_quality_location"].sum()),
        "suspect_event": int(df["suspect_event"].sum()),
        "magnitude": {
            "relations": {t: {"slope": s, "intercept": i, "valid_range": list(r), "source": src}
                          for t, (s, i, r, src) in CONVERSIONS.items()},
            "method_counts": df["mw_method"].value_counts().to_dict(),
            "extrapolated": int(df["mw_extrapolated"].sum()),
        },
        "completeness": {
            "method": "maximum curvature + 0.2 (bin 0.1)",
            "mc_overall": mc_overall,
            "by_year": mc_table.to_dict("records"),
            "above_mc": int(df["above_mc"].sum()),
        },
        "declustering": {
            "method": "Gardner & Knopoff (1974), window aftershock (window HMTK)",
            "foreshock_ratio": FORESHOCK_RATIO,
            "mainshocks_symmetric_window": int(is_mainshock_sym.sum()),
            "mainshocks": int(df["is_mainshock"].sum()),
            "dependent": int((~df["is_mainshock"] & ~df["suspect_event"]).sum()),
            "excluded_suspect": int(df["suspect_event"].sum()),
            "largest_sequences": [
                {"eventID": eid, "time": row["time"].isoformat(), "mw": round(float(row["mw"]), 2),
                 "location": row["location"], "n_events": int(row["n_events"])}
                for eid, row in largest.iterrows()
            ],
        },
        "mmi_class_counts": df["mmi_class"].value_counts().sort_index().to_dict(),
    }
    return df, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bersihkan katalog gempa BMKG.")
    parser.add_argument("--catalog", type=Path, default=paths.RAW_CATALOG)
    parser.add_argument("--output", type=Path, default=paths.CLEAN_CATALOG)
    parser.add_argument("--report", type=Path, default=paths.CLEAN_REPORT)
    args = parser.parse_args(argv)

    df, report = clean_catalog(load_catalog(args.catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df.columns if df[c].isna().all()]).to_parquet(args.output, index=False)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))

    print(f"{report['n_raw']:,} event mentah -> {report['n_clean']:,} event bersih")
    print(f"Mc keseluruhan: {report['completeness']['mc_overall']}, event >= Mc: {report['completeness']['above_mc']:,}")
    print(f"Mainshock: {report['declustering']['mainshocks']:,}, dependent: {report['declustering']['dependent']:,}")
    print("Kelas MMI:", report["mmi_class_counts"])
    print(f"Tersimpan: {args.output} dan {args.report}")


if __name__ == "__main__":
    main()
