"""Gempa Susulan: animasi deret gempa susulan dan peluruhan Omori-Utsu untuk sekuens terbesar."""
import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium

from common import MMI_CLASSES, MMI_COLORS, fmt_local, load_events
from gempa.stats import OMORI_C_BOUNDS, omori_mle

N_SEQUENCES = 20
MIN_SEQUENCE_EVENTS = 30
# Beberapa jam pertama katalog belum lengkap (gempa susulan kecil tertutup gempa besar)
OMORI_START_DAYS = 0.1
# Jendela waktu Gardner-Knopoff bisa bertahun-tahun dan ikut memuat kegempaan latar yang menekan nilai p
OMORI_END_DAYS = 100
# Rentang p yang lazim untuk deret gempa susulan; di luar ini peluruhannya tidak khas
TYPICAL_P = (0.5, 2.0)

df = load_events()
real = df[~df["suspect_event"]]


@st.cache_data
def largest_sequences():
    sizes = real.groupby("sequence_id").size()
    sizes = sizes[sizes >= MIN_SEQUENCE_EVENTS].nlargest(N_SEQUENCES)
    mains = real.set_index("eventID").loc[sizes.index]
    return pd.DataFrame({"sequence_id": sizes.index, "n_events": sizes.to_numpy(), "mw": mains["mw"].to_numpy(),
                         "location": mains["location"].to_numpy(), "local_time": mains["local_time"].to_numpy(),
                         "tz": mains["tz"].to_numpy()})


sequences = largest_sequences()
labels = {row.sequence_id: f"{row.local_time:%d %b %Y} · Mw {row.mw:.1f} · {row.location} ({row.n_events:,} gempa)"
          for row in sequences.itertuples()}

st.title("🔁 Gempa Susulan")
st.caption(f"{N_SEQUENCES} deret gempa terbesar hasil declustering Gardner–Knopoff (1974): gempa utama beserta gempa "
           "susulan dalam jendela jarak dan waktunya. Tekan ▶ pada peta untuk memutar urutan kejadian.")
sequence_id = st.selectbox("Deret gempa", list(labels), format_func=labels.get, key="susulan_sequence")

events = real[real["sequence_id"] == sequence_id].sort_values("time")
main = events.loc[events["eventID"] == sequence_id].iloc[0]
after = events[events["eventID"] != sequence_id].assign(
    days=lambda d: (d["time"] - main["time"]).dt.total_seconds() / 86400)

cols = st.columns(4)
cols[0].metric("Gempa utama", f"Mw {main['mw']:.1f}")
cols[1].metric("Gempa susulan", f"{len(after):,}")
cols[2].metric("Susulan terbesar", f"Mw {after['mw'].max():.1f}" if not after.empty else "–")
cols[3].metric("Rentang", f"{after['days'].max():.0f} hari" if not after.empty else "–")
st.caption(f"Gempa utama: {fmt_local(main['local_time'], main['tz'])}, kedalaman {main['depth']} km, "
           f"{main['location']}.")

# --- Peta animasi ---
features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [row.longitude, row.latitude]},
    "properties": {
        "time": row.time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "popup": f"Mw {row.mw:.1f}, {row.depth} km<br>{fmt_local(row.local_time, row.tz)}",
        "icon": "circle",
        "iconstyle": {"fillColor": MMI_COLORS[row.mmi_class], "fillOpacity": 0.8, "stroke": True,
                      "color": "#333333", "weight": 0.5, "radius": 2 + 1.5 * row.mw},
    },
} for row in events.itertuples()]
m = folium.Map(location=[main["latitude"], main["longitude"]], zoom_start=8)
folium.Marker([main["latitude"], main["longitude"]], icon=folium.Icon(color="red", icon="star"),
              tooltip=f"Gempa utama Mw {main['mw']:.1f}").add_to(m)
TimestampedGeoJson({"type": "FeatureCollection", "features": features}, period="P1D", duration=None,
                   add_last_point=False, auto_play=False, loop=False, max_speed=20, transition_time=150,
                   date_options="YYYY-MM-DD").add_to(m)
st_folium(m, height=480, use_container_width=True, returned_objects=[], key=f"susulan_map_{sequence_id}")
st.caption("Warna = kelas intensitas MMI di episentrum; titik bertambah per hari sejak gempa utama.")

if after.empty:
    st.stop()

# --- Peluruhan Omori-Utsu ---
left, right = st.columns(2)
with left:
    complete = after[after["above_mc"]]
    end = float(min(after["days"].max(), OMORI_END_DAYS))
    k, c, p, n = omori_mle(complete["days"], OMORI_START_DAYS, end)
    atypical = np.isfinite(p) and (not TYPICAL_P[0] <= p <= TYPICAL_P[1] or np.isclose(c, OMORI_C_BOUNDS[1]))
    edges = np.logspace(np.log10(OMORI_START_DAYS), np.log10(max(end, OMORI_START_DAYS * 10)), 25)
    counts, _ = np.histogram(complete["days"], bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    observed = pd.DataFrame({"hari": centers, "laju": counts / np.diff(edges), "seri": "observasi"})
    fitted_label = f"Omori-Utsu, p = {p:.2f}"
    color = alt.Color("seri:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["observasi", fitted_label], range=["#7f7f7f", "#d62728"]))
    x = alt.X("hari:Q", scale=alt.Scale(type="log"), title="Hari sejak gempa utama")
    y = alt.Y("laju:Q", scale=alt.Scale(type="log"), title="Gempa susulan per hari")
    layers = [alt.Chart(observed[observed["laju"] > 0]).mark_circle(size=50).encode(x=x, y=y, color=color)]
    if np.isfinite(p):
        fitted = pd.DataFrame({"hari": centers, "laju": k / (centers + c) ** p, "seri": fitted_label})
        layers.append(alt.Chart(fitted).mark_line(strokeWidth=2).encode(x=x, y=y, color=color))
    st.altair_chart(alt.layer(*layers).properties(title="Peluruhan gempa susulan", height=320), width="stretch")
    if np.isfinite(p):
        st.caption(f"Omori-Utsu n(t) = K/(t + c)^p dengan maximum likelihood (Ogata 1983) pada {n} gempa susulan ≥ Mc "
                   f"antara {OMORI_START_DAYS} dan {end:.0f} hari: K = {k:.1f}/hari, c = {c:.3f} hari, p = {p:.2f}. "
                   "Nilai p sekitar 1 adalah peluruhan khas; p lebih besar berarti susulan mereda lebih cepat.")
        if atypical:
            st.warning("Deret ini tidak meluruh seperti gempa susulan biasa (kemungkinan swarm, atau gempa susulan "
                       "bercampur kegempaan latar), sehingga parameter Omori-Utsu-nya kurang bermakna.")
    else:
        st.caption("Gempa susulan di atas Mc terlalu sedikit untuk fit Omori-Utsu.")
with right:
    chart = alt.Chart(after[["days", "mw", "depth", "mmi_class"]]).mark_circle(opacity=0.6).encode(
        x=alt.X("days:Q", title="Hari sejak gempa utama"),
        y=alt.Y("mw:Q", title="Mw", scale=alt.Scale(zero=False)),
        color=alt.Color("mmi_class:N", title="MMI", scale=alt.Scale(domain=MMI_CLASSES,
                                                                    range=[MMI_COLORS[c] for c in MMI_CLASSES])),
        tooltip=[alt.Tooltip("days:Q", format=".1f"), "mw", "depth"],
    ).properties(title="Magnitudo gempa susulan terhadap waktu", height=320)
    st.altair_chart(chart, width="stretch")
