"""Eksplorasi Data: magnitudo & Gutenberg-Richter, deret waktu, penampang kedalaman, dan tampilan 3D."""
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from common import (FAULT_COLORS, LAYER_LABELS, fault_type_chart, gr_chart, load_events, load_focal, load_report,
                    load_zoning, mc_by_year)
from gempa.focal import FAULT_LABELS, FAULT_TYPES
from gempa.stats import b_value, complete_exposure_years

# Penampang: (sumbu horizontal, kolom pita, batas pita, batas sumbu)
SECTIONS = {
    "Jawa (bujur 106°–112°)": ("latitude", "longitude", (106, 112), (-11.5, -4)),
    "Sumatra Utara (lintang 0°–4°)": ("longitude", "latitude", (0, 4), (93, 101)),
    "Laut Maluku (lintang 0°–3°)": ("longitude", "latitude", (0, 3), (122, 130)),
    "Laut Banda (bujur 124°–130°)": ("latitude", "longitude", (124, 130), (-9, -3)),
}
# Wilayah 3D: (bujur min, bujur max, lintang min, lintang max)
REGIONS = {
    "Jawa–Nusa Tenggara": (104, 125, -12, -5),
    "Sulawesi–Maluku": (118, 130, -6, 5),
    "Laut Banda": (122, 134, -9, -3),
    "Sumatra": (94, 107, -7, 6),
    "Indonesia": (94, 142, -12, 7),
}
LAYER_COLORS = {"DK": [214, 39, 40], "MN": [255, 127, 14], "DL": [31, 119, 180]}

df = load_events()
real = df[~df["suspect_event"]]

st.title("📊 Eksplorasi Data")
st.caption(f"{len(real):,} gempa (tanpa event semu) dari katalog BMKG {df['time'].min():%Y}–{df['time'].max():%Y}.")

tab_mag, tab_time, tab_depth, tab_3d, tab_fm = st.tabs(
    ["Magnitudo", "Waktu", "Penampang kedalaman", "3D", "Mekanisme fokal"])

with tab_mag:
    left, right = st.columns(2)
    with left:
        hist = real.assign(jenis=np.where(real["is_mainshock"], "gempa utama", "gempa susulan / sekuens"))
        chart = alt.Chart(hist[["mw", "jenis"]]).mark_bar().encode(
            # Bin 0,2: Mw hasil konversi linear dari magnitudo berbin 0,1 menimbulkan pola sisir pada bin 0,1
            x=alt.X("mw:Q", bin=alt.Bin(step=0.2), title="Mw"),
            y=alt.Y("count():Q", stack=True, title="Jumlah gempa"),
            color=alt.Color("jenis:N", title=None, legend=alt.Legend(orient="top")),
        ).properties(title="Distribusi magnitudo (Mw)", height=320)
        st.altair_chart(chart, width="stretch")
    with right:
        # Kelengkapan berbeda antarwilayah: Mc zona tertinggi menjamin semua zona lengkap. Dengan Mc katalog
        # keseluruhan (lebih rendah) b-value bias rendah karena zona yang belum lengkap ikut terhitung.
        mc = float(load_zoning()[1]["mc"].max())
        main = real[real["is_mainshock"]]
        years_ok = [year for year, value in mc_by_year().items() if value <= mc + 1e-9]
        mags = main.loc[main["time"].dt.year.isin(years_ok), "mw"]
        b, sigma, n = b_value(mags, mc)
        exposure = complete_exposure_years(mc_by_year(), mc, df["time"].min(), df["time"].max())
        st.altair_chart(gr_chart(mags, exposure, mc, b, n / exposure,
                                 f"Gutenberg–Richter gempa utama, b = {b:.2f} ± {sigma:.2f}"),
                        width="stretch")
    st.caption(f"Garis putus-putus = magnitude of completeness (Mc = {mc:.1f}, Mc zona tertinggi): di bawahnya "
               "sebagian gempa kecil tidak terekam, sehingga kurva melandai. Kelengkapan katalog berbeda antarwilayah, "
               f"jadi Mc keseluruhan ({load_report()['completeness']['mc_overall']:.1f}) memberi b-value yang bias "
               "rendah. Statistik hazard tiap zona memakai Mc zonanya sendiri.")

with tab_time:
    monthly = (real.set_index("time").resample("MS")
               .agg(semua=("mw", "size"), gempa_utama=("is_mainshock", "sum")).reset_index()
               .melt("time", var_name="seri", value_name="jumlah"))
    chart = alt.Chart(monthly).mark_line().encode(
        x=alt.X("time:T", title="Bulan"), y=alt.Y("jumlah:Q", title="Jumlah gempa per bulan"),
        color=alt.Color("seri:N", title=None, legend=alt.Legend(orient="top")),
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    st.caption("Lonjakan seluruh gempa yang tidak diikuti gempa utama adalah sekuens gempa susulan "
               "(mis. Lombok 2018, Ambon & Laut Maluku 2019).")

    report = load_report()
    yearly = pd.DataFrame(report["completeness"]["by_year"])
    match = pd.Series(report["verification"]["match_rate_by_year"], name="cocok").rename_axis("year").reset_index()
    match["year"] = match["year"].astype(int)
    yearly = yearly.merge(match, on="year", how="left")
    left, right = st.columns(2)
    with left:
        st.altair_chart(alt.Chart(yearly).mark_line(point=True).encode(
            x=alt.X("year:O", title="Tahun"), y=alt.Y("mc:Q", title="Mc", scale=alt.Scale(zero=False)),
        ).properties(title="Magnitude of completeness per tahun", height=260), width="stretch")
    with right:
        st.altair_chart(alt.Chart(yearly).mark_bar().encode(
            x=alt.X("year:O", title="Tahun"),
            y=alt.Y("cocok:Q", title="Tercatat juga di USGS", axis=alt.Axis(format="%")),
            color=alt.condition(alt.datum.cocok < 0.8, alt.value("#C44E52"), alt.value("#4C72B0")),
        ).properties(title="Verifikasi gempa Mw ≥ 5 dengan katalog USGS", height=260), width="stretch")
    st.caption("Mc turun seiring bertambahnya stasiun BMKG. Pada 2019–2021 banyak gempa Mw ≥ 5 di katalog tidak "
               "tercatat jaringan lain; gempa tersebut ditandai sebagai event semu.")

with tab_depth:
    section = st.selectbox("Penampang", list(SECTIONS), key="eks_section")
    axis, band, (lo, hi), limits = SECTIONS[section]
    part = real[real[band].between(lo, hi) & real[axis].between(*limits) & (real["mw"] >= 3.5)]
    part = part.assign(lapisan=part["layer"].map(LAYER_LABELS))
    chart = alt.Chart(part[[axis, "depth", "mw", "lapisan"]]).mark_circle(opacity=0.5).encode(
        x=alt.X(f"{axis}:Q", title="Lintang" if axis == "latitude" else "Bujur", scale=alt.Scale(domain=limits)),
        y=alt.Y("depth:Q", title="Kedalaman (km)", scale=alt.Scale(reverse=True)),
        size=alt.Size("mw:Q", legend=None, scale=alt.Scale(range=[4, 120])),
        color=alt.Color("lapisan:N", title="Lapisan", sort=list(LAYER_LABELS.values()),
                        scale=alt.Scale(domain=list(LAYER_LABELS.values()),
                                        range=[f"rgb{tuple(LAYER_COLORS[c])}" for c in LAYER_LABELS])),
        tooltip=["mw", "depth", axis],
    ).properties(height=420)
    st.altair_chart(chart, width="stretch")
    st.caption(f"{len(part):,} gempa Mw ≥ 3,5. Gempa yang makin dalam menjauhi palung memperlihatkan lempeng yang "
               "menunjam (zona Benioff).")

with tab_3d:
    c1, c2 = st.columns(2)
    region = c1.selectbox("Wilayah", list(REGIONS), key="eks_region")
    exaggeration = c2.slider("Pembesaran vertikal (×)", 1, 5, 1, key="eks_exaggeration",
                             help="1 = skala sebenarnya: kedalaman 600 km digambar sejauh 600 km.")
    lon_min, lon_max, lat_min, lat_max = REGIONS[region]
    pts = real[real["longitude"].between(lon_min, lon_max) & real["latitude"].between(lat_min, lat_max)
               & (real["mw"] >= 4.0)][["longitude", "latitude", "depth", "mw", "layer"]]
    pts = pts.assign(z=-pts["depth"] * 1000.0 * exaggeration, color=pts["layer"].map(LAYER_COLORS))
    deck = pdk.Deck(
        layers=[pdk.Layer("PointCloudLayer", data=pts, get_position=["longitude", "latitude", "z"],
                          get_color="color", point_size=3, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=(lat_min + lat_max) / 2, longitude=(lon_min + lon_max) / 2,
                                         zoom=4.6 if region != "Indonesia" else 3.6, pitch=55, bearing=-15),
        tooltip={"text": "Mw {mw}\nKedalaman {depth} km"},
    )
    st.pydeck_chart(deck, height=560)
    st.caption(f"{len(pts):,} gempa Mw ≥ 4. Merah = dangkal, oranye = menengah, biru = dalam. Seret untuk memutar, "
               "Ctrl/Shift + seret untuk mengubah sudut pandang.")

with tab_fm:
    fm = load_focal()
    labels = [FAULT_LABELS[t] for t in FAULT_TYPES]
    st.caption(f"{len(fm):,} mekanisme fokal BMKG ({fm['time'].min():%Y}–{fm['time'].max():%Y}). Jenis sesar "
               "ditentukan dari kemiringan sumbu tekanan (P), tarikan (T), dan netral (B) menurut Frohlich (1992): "
               "T curam = sesar naik, P curam = sesar turun, B curam = sesar geser, selain itu oblik.")
    # Skala proyeksi eksplisit: Vega-Lite tidak otomatis menyesuaikan peta ke sebaran data
    chart = alt.Chart(fm[["longitude", "latitude", "jenis", "mw", "depth", "location"]]).mark_circle(
        opacity=0.7).encode(
        longitude="longitude:Q", latitude="latitude:Q",
        size=alt.Size("mw:Q", legend=None, scale=alt.Scale(range=[4, 90])),
        color=alt.Color("jenis:N", title="Jenis sesar", sort=labels,
                        scale=alt.Scale(domain=labels, range=[FAULT_COLORS[t] for t in FAULT_TYPES])),
        tooltip=["jenis", alt.Tooltip("mw:Q", format=".1f"), "depth", "location"],
    ).project(type="mercator", center=[118, -2.5], scale=950).properties(height=440, title="Sebaran jenis sesar")
    st.altair_chart(chart, width="stretch")
    by_layer = fm.assign(lapisan=fm["layer"].map(LAYER_LABELS))
    st.altair_chart(fault_type_chart(by_layer, by="lapisan", sort=list(LAYER_LABELS.values()))
                    .properties(title="Proporsi per lapisan kedalaman"), width="stretch")
    st.caption("Gempa dangkal didominasi sesar naik (zona subduksi dan sesar anjak) dan sesar geser, sedangkan "
               "gempa dalam lebih banyak sesar turun dan oblik. Jenis sesar per zona ada di **Zona Seismik**.")
