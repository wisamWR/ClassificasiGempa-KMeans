import numpy as np
import pandas as pd
import pytest

from gempa.focal import faulting_type, focal_mechanisms, pbt_plunges


@pytest.mark.parametrize("strike, dip, rake, expected", [
    (0, 45, 90, "naik"),     # thrust murni
    (0, 45, -90, "turun"),   # sesar normal murni
    (0, 90, 0, "geser"),     # sesar geser mengiri vertikal
    (30, 85, 180, "geser"),  # sesar geser menganan
    (0, 60, 30, "oblik"),    # campuran naik-geser (tidak ada sumbu yang curam)
])
def test_faulting_type_end_members(strike, dip, rake, expected):
    assert faulting_type([strike], [dip], [rake])[0] == expected


def test_pure_thrust_axes():
    p, t, b = pbt_plunges([0], [45], [90])
    assert t[0] == pytest.approx(90) and p[0] == pytest.approx(0, abs=1e-6) and b[0] == pytest.approx(0, abs=1e-6)


def test_both_nodal_planes_give_same_type():
    # Solusi BMKG di katalog: Cianjur 2022 (bmg2022wulo) dan Palu 2018 (bmg2018tabs, dip 55° -> oblik)
    planes = np.array([[347.19, 82.77, -167.17, 255.55, 77.27, -7.41],
                       [352.01, 55.41, -9.10, 87.21, 82.52, -145.07]])
    first = faulting_type(planes[:, 0], planes[:, 1], planes[:, 2])
    second = faulting_type(planes[:, 3], planes[:, 4], planes[:, 5])
    assert list(first) == list(second) == ["geser", "oblik"]


def test_focal_mechanisms_keeps_complete_planes():
    catalog = pd.DataFrame({
        "eventID": ["a", "b"], "time": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "latitude": [-7.0, -8.0], "longitude": [107.0, 116.0], "depth": [10, 15], "mag": [5.0, 6.0],
        "strikeNP1": [0, np.nan], "dipNP1": [45, 30], "rakeNP1": [90, 90],
        "strikeNP2": [180, 0], "dipNP2": [45, 60], "rakeNP2": [90, 90],
    })
    fm = focal_mechanisms(catalog)
    assert list(fm["eventID"]) == ["a"]
    assert fm.loc[0, "fault_type"] == "naik"
    assert {"p_plunge", "t_plunge", "b_plunge"} <= set(fm.columns)
