"""Kategorikan Gempa: lokasi + kedalaman + magnitudo -> intensitas, zona, dan konteks historis zona."""
import pandas as pd
import streamlit as st

from common import (LAYER_LABELS, MMI_ADVICE, MMI_COLORS, fmt_years, load_events, load_kmeans, load_zoning,
                    result_box)
from gempa.intensity import MMI_DESCRIPTIONS, epicentral_intensity, mmi_class
from gempa.location import zone_context
from gempa.magnitude import to_mw

MAG_TYPES = ["Mw", "M", "MLv", "ML", "mb"]
FAR_FROM_ZONE_KM = 150

df = load_events()
models, zones, geojson, _ = load_zoning()
pipeline, id_to_label, _ = load_kmeans()

st.title("⚡ Kategorikan Gempa")
st.caption("Masukkan data gempa (misalnya dari info gempa BMKG) untuk melihat perkiraan intensitas di episentrum, "
           "zona sumbernya, dan seberapa sering gempa sebesar itu terjadi di zona tersebut.")

# Nilai awal: gempa Cianjur 21 November 2022
with st.form("kategorikan"):
    c1, c2, c3, c4, c5 = st.columns(5)
    lat = c1.number_input("Lintang", min_value=-12.0, max_value=7.0, value=-6.86, step=0.01, format="%.2f")
    lon = c2.number_input("Bujur", min_value=94.0, max_value=142.0, value=107.01, step=0.01, format="%.2f")
    depth = c3.number_input("Kedalaman (km)", min_value=0.0, max_value=750.0, value=10.0, step=5.0)
    mag = c4.number_input("Magnitudo", min_value=1.0, max_value=9.5, value=5.6, step=0.1)
    mag_type = c5.selectbox("Tipe magnitudo", MAG_TYPES, help="Selain Mw, dikonversi ke Mw (Taruna & Pratiwi 2021).")
    st.form_submit_button("Kategorikan", type="primary")

converted = to_mw(pd.Series([mag]), pd.Series([mag_type])).iloc[0]
mw = float(converted["mw"])
i0 = float(epicentral_intensity(mw, depth))
mmi = mmi_class(i0)

left, right = st.columns(2)
with left:
    note = f" (dari {mag_type} {mag:.1f})" if mag_type != "Mw" else ""
    result_box(f"MMI {mmi}",
               f"<p><b>Mw {mw:.1f}</b>{note}, kedalaman {depth:.0f} km → I₀ ≈ {i0:.1f}</p>"
               f"<p><b>Efek:</b> {MMI_DESCRIPTIONS[mmi]}</p><p><b>Saran:</b> {MMI_ADVICE[mmi]}</p>",
               MMI_COLORS[mmi])
    if converted["mw_extrapolated"]:
        st.caption("⚠️ Magnitudo di luar rentang data relasi konversi; nilai Mw adalah ekstrapolasi.")
    st.caption("Intensitas adalah perkiraan guncangan tepat di atas pusat gempa. Makin jauh dari episentrum, "
               "guncangan makin lemah.")

# Zona: lapisan dari kedalaman, lalu centroid K-Means terdekat
context = zone_context(models, zones, geojson, lat, lon, depth, mw).iloc[0]
zone_id, distance = context["zone_id"], context["distance_km"]
z = zones.set_index("zone_id").loc[zone_id]
zone_events = df[df["in_zoning_set"] & (df["zone_id"] == zone_id)]
smaller = (zone_events["mw"] < mw).mean()

with right:
    position = "di dalam zona" if distance == 0 else f"{distance:.0f} km dari tepi zona"
    st.markdown(f"#### 🧭 Zona {zone_id} · {z['name']}")
    st.markdown(
        f"- Lapisan **{LAYER_LABELS[z['layer']]}**, episentrum {position}.\n"
        f"- Gempa utama **M ≥ {mw:.1f}** di zona ini rata-rata terjadi sekali tiap **{fmt_years(context['recurrence_years'])}**.\n"
        f"- Lebih besar dari **{smaller:.0%}** gempa utama yang tercatat di zona ini "
        f"(terbesar: Mw {z['mw_max']:.1f}).\n"
        f"- b-value zona {z['b']:.2f} ± {z['b_sigma']:.2f}."
    )
    if distance > FAR_FROM_ZONE_KM:
        st.warning(f"Lokasi lebih dari {FAR_FROM_ZONE_KM} km dari zona mana pun pada kedalaman ini, sehingga "
                   "statistik zona terdekat kurang mewakili.")

    # Model K-Means dilatih dengan magnitudo asli katalog (mag) dan kedalaman
    profile = id_to_label[pipeline.predict(pd.DataFrame([[mag, depth]], columns=["mag", "depth"]))[0]]
    st.caption(f"Profil klaster K-Means (magnitudo & kedalaman): **{profile}**. Label ini relatif antar-klaster, "
               "bukan tingkat risiko.")
