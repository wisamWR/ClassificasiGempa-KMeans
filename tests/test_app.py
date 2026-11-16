"""Uji aplikasi Streamlit (semua halaman) dengan AppTest; geocoder Nominatim di-mock."""
from types import SimpleNamespace

import pandas as pd
import pytest
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from streamlit.testing.v1 import AppTest

from gempa import paths
from gempa.geo import haversine_km
from gempa.location import radius_hazard

APP = str(paths.ROOT_DIR / "app" / "streamlit_app.py")
PLACES = {
    "Cianjur": (-6.82, 107.14, "Cianjur, Jawa Barat, Indonesia"),
    "Pontianak": (-0.03, 109.33, "Pontianak, Kalimantan Barat, Indonesia"),
}


def fake_geocode(self, query, **kwargs):
    if query == "Timeout":
        raise GeocoderTimedOut("mock timeout")
    if query not in PLACES:
        return None
    lat, lon, address = PLACES[query]
    return SimpleNamespace(latitude=lat, longitude=lon, address=address)


@pytest.fixture(autouse=True)
def mock_geocoder(monkeypatch):
    monkeypatch.setattr(Nominatim, "geocode", fake_geocode)


@pytest.fixture(scope="module")
def events():
    df = pd.read_parquet(paths.PROCESSED_EVENTS).merge(pd.read_parquet(paths.ZONE_EVENTS), on="eventID")
    df["suspect_event"] = df["suspect_event"].fillna(False).astype(bool)
    return df


def run(at):
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def open_page(page=None):
    at = run(AppTest.from_file(APP, default_timeout=300))
    if page:
        at.switch_page(f"views/{page}.py")
        run(at)
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader])


def test_default_page_is_cek_lokasi_with_disclaimer():
    at = open_page()
    assert at.title[0].value == "📍 Cek Lokasi"
    assert any("Ketik nama kota" in i.value for i in at.info)
    assert "tidak memprediksi gempa" in " ".join(c.value for c in at.sidebar.caption)


def test_cek_lokasi_cianjur(events):
    at = open_page()
    at.text_input(key="cek_query").set_value("Cianjur")
    run(at)
    lat, lon, _ = PLACES["Cianjur"]
    radius = 100
    real = events[~events["suspect_event"]]
    near = real[haversine_km(lat, lon, real["latitude"].to_numpy(), real["longitude"].to_numpy()) <= radius]
    assert metric(at, "Total Kejadian") == f"{len(near):,}"
    assert metric(at, "Mw Tertinggi") == f"{near['mw'].max():.1f}"

    zones = pd.read_csv(paths.ZONE_STATS)
    summary, _ = radius_hazard(events[events["in_zoning_set"]], zones, lat, lon, radius)
    p = summary["p50_m6"]
    expected = "> 99%" if p >= 0.995 else ("< 1%" if p < 0.005 else f"{p:.0%}")
    assert metric(at, "M ≥ 6 dalam 50 tahun") == expected

    zone_table, top5 = at.dataframe[-2].value, at.dataframe[-1].value
    assert list(zone_table["Lapisan"].str[:4]) == ["Dang", "Mene", "Dala"]
    assert zone_table.iloc[0]["Zona"].startswith("DK-")
    assert len(top5) == 5 and top5["Waktu (setempat)"].str.endswith("WIB").all()
    assert "Intensitas tertinggi tercatat" in text(at)


def test_cek_lokasi_radius_and_low_seismicity_area():
    at = open_page()
    at.text_input(key="cek_query").set_value("Pontianak")
    run(at)
    assert any("lebih dari 100 km dari zona gempa dangkal" in w.value for w in at.warning)
    at.slider(key="cek_radius").set_value(25)
    run(at)
    assert any("tidak dapat diperkirakan" in i.value or "Tidak ada catatan" in i.value for i in at.info)


@pytest.mark.parametrize("query, message", [("Antah Berantah", "tidak ditemukan"), ("Timeout", "Gagal koneksi")])
def test_cek_lokasi_geocoding_errors(query, message):
    at = open_page()
    at.text_input(key="cek_query").set_value(query)
    run(at)
    assert any(message in w.value for w in at.warning)
    assert any("Ketik nama kota" in i.value for i in at.info)


def test_peta_gempa_filters(events):
    at = open_page("peta_gempa")
    real = events[~events["suspect_event"]]
    assert metric(at, "Total Kejadian") == f"{len(real):,}"
    assert all(f"MMI {c}" in text(at) for c in ["I-II", "III-IV", "V-VI", "VII+"])

    at.checkbox(key="peta_mainshock").check()
    at.multiselect(key="peta_layer").set_value(["DK"])
    run(at)
    expected = real[real["is_mainshock"] & (real["depth"] < 70)]
    assert metric(at, "Total Kejadian") == f"{len(expected):,}"

    at.button[0].click()  # Reset Filter
    run(at)
    assert metric(at, "Total Kejadian") == f"{len(real):,}"


def test_peta_gempa_swaps_reversed_dates():
    at = open_page("peta_gempa")
    at.date_input(key="peta_start_date").set_value(pd.Timestamp("2020-01-01").date())
    at.date_input(key="peta_end_date").set_value(pd.Timestamp("2019-01-01").date())
    run(at)
    assert any("ditukar" in w.value for w in at.sidebar.warning)


def test_zona_all_layers_and_detail():
    zones = pd.read_csv(paths.ZONE_STATS)
    at = open_page("zona")
    for layer in ["DK", "MN", "DL"]:
        at.radio(key="zona_layer").set_value(layer)
        at.toggle(key="zona_heatmap").set_value(layer == "MN")
        run(at)
        assert len(at.dataframe[0].value) == (zones["layer"] == layer).sum()
        assert at.selectbox(key="zona_selected").value.startswith(layer)
    at.radio(key="zona_layer").set_value("DK")
    run(at)
    at.selectbox(key="zona_selected").set_value("DK-05")
    at.selectbox(key="zona_metric").set_value("b-value")
    run(at)
    assert at.subheader[0].value.startswith("DK-05")
    assert any(m.value.startswith("**Jenis sesar** (") for m in at.markdown)


def test_kategorikan_default_cianjur():
    at = open_page("kategorikan")
    assert any("MMI VII+" in m.value for m in at.markdown)
    assert "Zona DK-" in text(at)
    assert "sekali tiap" in text(at)
    at.selectbox[0].set_value("mb")
    at.number_input[1].set_value(128.4)  # bujur Laut Banda
    at.number_input[2].set_value(150.0)
    at.button[0].click()
    run(at)
    assert "dari mb 5.6" in text(at)
    assert "Zona MN-" in text(at)


def test_eksplorasi_all_tabs():
    at = open_page("eksplorasi")
    assert "Frohlich (1992)" in text(at)
    for section in at.selectbox(key="eks_section").options:
        at.selectbox(key="eks_section").set_value(section)
        run(at)
    for region in at.selectbox(key="eks_region").options:
        at.selectbox(key="eks_region").set_value(region)
        run(at)


@pytest.mark.parametrize("page, phrase", [("metodologi", "Keterbatasan"), ("keselamatan", "InaRISK")])
def test_static_pages(page, phrase):
    at = open_page(page)
    assert phrase in text(at)


@pytest.fixture
def bmkg_feeds(monkeypatch):
    """Feed BMKG dari tests/data; tsunami=True mengubah gempa terbaru menjadi berpotensi tsunami."""
    import json
    from pathlib import Path

    import streamlit as st

    from gempa import bmkg

    files = {"terbaru": "autogempa", "terkini": "gempaterkini", "dirasakan": "gempadirasakan"}
    state = {"offline": False, "tsunami": False}

    def fake_fetch(name, timeout=10):
        if state["offline"]:
            raise bmkg.FeedError("offline")
        payload = json.loads((Path(__file__).parent / "data" / f"bmkg_{files[name]}.json").read_text(encoding="utf-8"))
        if state["tsunami"] and name == "terbaru":
            payload["Infogempa"]["gempa"]["Potensi"] = "Berpotensi tsunami"
        return payload

    monkeypatch.setattr(bmkg, "fetch_feed", fake_fetch)
    st.cache_data.clear()
    yield state
    st.cache_data.clear()


def test_gempa_terkini_page(bmkg_feeds):
    at = open_page("gempa_terkini")
    assert "BMKG" in text(at)
    assert any("M 5.4" in m.value and "BURUSELATAN" in m.value for m in at.markdown)
    assert any(m.value.startswith("Zona **DL-01**") for m in at.markdown)
    assert not at.error
    big, felt = at.dataframe[0].value, at.dataframe[1].value
    assert len(big) == 15 and len(felt) == 15
    assert big["Gempa sebesar ini di zona"].str.startswith("sekali tiap").all()
    assert felt["Waktu (setempat)"].str.contains("WIB|WITA|WIT").all()


def test_gempa_terkini_tsunami_alert(bmkg_feeds):
    bmkg_feeds["tsunami"] = True
    at = open_page("gempa_terkini")
    assert any("Berpotensi tsunami" in e.value for e in at.error)


def test_gempa_terkini_offline(bmkg_feeds):
    bmkg_feeds["offline"] = True
    at = open_page("gempa_terkini")
    assert any("tidak bisa diambil" in w.value for w in at.warning)
    assert not at.dataframe


def test_susulan_sequences(events):
    at = open_page("susulan")
    assert at.metric[0].label == "Gempa utama"
    assert "Ogata 1983" in text(at) and "p = " in text(at)
    assert len(at.selectbox(key="susulan_sequence").options) == 20
    # Deret terbesar dengan gempa utama < Mw 5: swarm yang tidak meluruh seperti Omori
    real = events[~events["suspect_event"]]
    largest = real.groupby("sequence_id").size().nlargest(20).index
    mains = real.set_index("eventID").loc[largest, "mw"]
    at.selectbox(key="susulan_sequence").set_value(mains[mains < 5].index[0])
    run(at)
    assert any("tidak meluruh" in w.value for w in at.warning)
