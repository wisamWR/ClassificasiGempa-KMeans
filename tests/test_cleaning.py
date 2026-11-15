import numpy as np
import pandas as pd
import pytest

from gempa.clean import clean_catalog
from gempa.completeness import mc_by_year, mc_maxc
from gempa.decluster import gardner_knopoff, gk_distance_km, gk_time_days
from gempa.intensity import mmi_class
from gempa.magnitude import to_mw


def test_to_mw_conversion_native_and_approx():
    out = to_mw([4.0, 4.0, 5.0, 5.5, 6.0, 4.0], ["M", "MLv", "mb", "Mw(mB)", "mB", "Xyz"])
    np.testing.assert_allclose(out["mw"][:3], [1.0166 * 4 - 0.2207, 0.7353 * 4 + 1.0722, 1.1993 * 5 - 1.2261])
    assert out["mw"][3] == 5.5 and out["mw_method"][3] == "native"
    assert out["mw"][4] == 6.0 and out["mw_method"][4] == "approx"
    assert np.isnan(out["mw"][5]) and out["mw_method"][5] == "unknown"


def test_to_mw_flags_extrapolation():
    out = to_mw([2.0, 3.5, 8.0], ["MLv", "MLv", "MLv"])
    assert list(out["mw_extrapolated"]) == [True, False, True]


def test_mmi_class_boundaries():
    assert mmi_class(2.49) == "I-II"
    assert mmi_class(2.5) == "III-IV"
    assert mmi_class(6.49) == "V-VI"
    assert mmi_class(6.5) == "VII+"
    classes = mmi_class(np.array([1.0, 7.0]))
    assert list(classes) == ["I-II", "VII+"] and classes.ordered


def test_mc_maxc_peak_plus_correction():
    mags = [2.8] * 5 + [3.0] * 20 + [3.1] * 12 + [3.5] * 3
    assert mc_maxc(mags) == pytest.approx(3.2)


def test_mc_by_year_falls_back_for_sparse_years():
    times = pd.to_datetime(["2020-01-01"] * 60 + ["2021-01-01"] * 3, utc=True)
    mags = [3.0] * 60 + [5.0] * 3
    table, overall = mc_by_year(times, mags, min_events=50)
    assert overall == pytest.approx(3.2)
    assert table.set_index("year")["mc"].to_dict() == {2020: pytest.approx(3.2), 2021: pytest.approx(3.2)}


def test_gk_windows():
    assert gk_distance_km(6.0) == pytest.approx(10 ** (0.1238 * 6 + 0.983))
    assert gk_time_days(6.0) == pytest.approx(10 ** (0.5409 * 6 - 0.547))
    assert gk_time_days(7.0) == pytest.approx(10 ** (0.032 * 7 + 2.7389))


@pytest.fixture
def sequence_catalog():
    t0 = pd.Timestamp("2020-06-01", tz="UTC")
    rows = [
        ("main", t0, 0.0, 120.0, 6.0),
        ("after1", t0 + pd.Timedelta(days=1), 0.05, 120.05, 4.0),
        ("after2", t0 + pd.Timedelta(days=30), -0.1, 120.0, 3.5),
        ("fore", t0 - pd.Timedelta(days=2), 0.02, 120.0, 4.5),
        ("far", t0 + pd.Timedelta(days=1), 5.0, 125.0, 3.0),          # ~780 km
        ("late", t0 + pd.Timedelta(days=600), 0.0, 120.0, 3.0),       # setelah window ~499 hari
    ]
    return pd.DataFrame(rows, columns=["id", "time", "lat", "lon", "mw"])


def test_gardner_knopoff_aftershocks_only(sequence_catalog):
    c = sequence_catalog
    seq, main = gardner_knopoff(c["time"], c["lat"], c["lon"], c["mw"])
    result = dict(zip(c["id"], main))
    assert result == {"main": True, "after1": False, "after2": False, "fore": True, "far": True, "late": True}
    assert seq[1] == seq[2] == 0


def test_gardner_knopoff_symmetric_catches_foreshock(sequence_catalog):
    c = sequence_catalog
    _, main = gardner_knopoff(c["time"], c["lat"], c["lon"], c["mw"], foreshock_ratio=1.0)
    assert dict(zip(c["id"], main))["fore"] is np.False_


def test_clean_catalog_columns_and_suspect_flag():
    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    n = 80
    df = pd.DataFrame({
        "eventID": [f"e{i}" for i in range(n)],
        "time": [t0 + pd.Timedelta(days=i * 20) for i in range(n)],
        "latitude": np.linspace(-8, 2, n),
        "longitude": np.linspace(100, 130, n),
        "mag": [3.5] * (n - 1) + [7.0],
        "mag_type": ["M"] * (n - 1) + ["M"],
        "depth": [10] * (n - 1) + [10],
        "phasecount": [40.0] * (n - 1) + [4.0],
        "azimuth_gap": [90.0] * (n - 1) + [250.0],
        "location": "Test",
    })
    clean, report = clean_catalog(df)
    expected = {"mw", "mw_method", "mw_extrapolated", "depth_fixed", "low_quality_location", "mc",
                "above_mc", "suspect_event", "sequence_id", "is_mainshock", "intensity", "mmi_class"}
    assert expected <= set(clean.columns)
    last = clean.iloc[-1]
    assert last["suspect_event"] and not last["is_mainshock"]
    assert report["suspect_event"] == 1
    assert clean["depth_fixed"].all()
    # Event lain berjauhan (20 hari, ratusan km): semuanya mainshock
    assert clean["is_mainshock"].sum() == n - 1
