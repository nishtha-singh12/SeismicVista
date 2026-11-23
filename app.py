from pathlib import Path

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title="SeismicVista", layout="wide")

RISK_ORDER = ["Low Risk", "Moderate Risk", "High Risk", "Critical"]
RISK_WEIGHT = {"Low Risk": 0, "Moderate Risk": 33, "High Risk": 67, "Critical": 100}
RISK_COLORS = {"Low Risk": "#2fd1a0", "Moderate Risk": "#f0b429",
               "High Risk": "#f08a3c", "Critical": "#e5484d"}
RISK_NOTES = {
    "Low Risk": "Most likely a small event, magnitude below 5.0. Usually felt weakly or not at all.",
    "Moderate Risk": "Most likely magnitude 5.0 to 5.9. Can damage poorly built structures near the epicentre.",
    "High Risk": "Most likely magnitude 6.0 to 6.9. Can cause serious damage in populated areas.",
    "Critical": "Most likely magnitude 7.0 or above. Can cause severe damage over large areas.",
}
MAG_LABELS = {"md": "Duration (tiny local quakes)", "ml": "Local (small-moderate)",
              "mb": "Body wave (mid-size, global)", "ms_20": "Surface wave (shallow, larger)",
              "mwr": "Regional moment", "mwb": "Body-wave moment", "mw": "Moment (large)",
              "mwc": "Centroid moment (large)", "mww": "W-phase moment (large)"}
RISK_RANGE = {"Low Risk": "Below 5.0", "Moderate Risk": "5.0 – 5.9",
              "High Risk": "6.0 – 6.9", "Critical": "7.0 and above"}
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');
.stApp { background:#0a0e17; font-family:'DM Sans',sans-serif; }
h1,h2,h3,.disp { font-family:'Playfair Display',serif !important; color:#f3ece0; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#111827; border:1px solid #1f2a3d; border-radius:16px; }
.sub { color:#8b97ad; font-size:14px; margin:-8px 0 14px 0; }
.res { text-align:center; }
.res .lvl { font-family:'Playfair Display',serif; font-size:44px; font-weight:700; line-height:1.1; }
.res .cap { color:#8b97ad; font-size:13px; }
.pr { display:flex; align-items:center; gap:10px; margin:7px 0; font-size:13px; color:#c9d1e0; }
.pr .n { width:105px; } .pr .t { flex:1; background:#1a2336; border-radius:6px; height:10px; }
.pr .b { height:10px; border-radius:6px; } .pr .v { width:48px; text-align:right; }
.refrow { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-top:8px; }
.refcard { background:#0f1626; border:1px solid #1f2a3d; border-left-width:4px; border-radius:10px; padding:10px 14px; opacity:.65; }
.refcard.on { opacity:1; }
.refcard .rn { font-size:12px; font-weight:700; } .refcard .rr { font-family:'Playfair Display',serif; font-size:20px; margin-top:2px; }
.reft { color:#8b97ad; font-size:13px; margin-top:18px; }
.note { background:#16213a; border-radius:10px; padding:10px 14px; color:#c9d1e0; font-size:14px; margin-top:10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    return (joblib.load(BASE_DIR / "Risk_Predictor_model.pkl"),
            joblib.load(BASE_DIR / "le_magType.pkl"),
            joblib.load(BASE_DIR / "le_risk_level.pkl"))


def predict(model, le_mag, le_risk, depth, lat, lon, mag_type):
    vals = {"depth": depth, "latitude": lat, "longitude": lon,
            "magType": le_mag.transform([mag_type])[0]}
    row = pd.DataFrame([vals])[list(model.feature_names_in_)]
    pred = le_risk.inverse_transform(model.predict(row))[0]
    proba = pd.Series(model.predict_proba(row)[0],
                      index=le_risk.inverse_transform(model.classes_)
                      ).reindex(RISK_ORDER).fillna(0.0)
    return pred, proba


def gauge_html(index, color):
    angle = 180 * index / 100
    return (
        '<svg viewBox="0 0 200 115" width="100%" style="max-width:300px">'
        '<defs><linearGradient id="g"><stop offset="0" stop-color="#4c8dff"/>'
        '<stop offset="0.5" stop-color="#f0b429"/><stop offset="1" stop-color="#e5484d"/>'
        '</linearGradient></defs>'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#g)" '
        'stroke-width="12" stroke-linecap="round"/>'
        f'<line x1="100" y1="100" x2="34" y2="100" stroke="{color}" stroke-width="3" '
        f'stroke-linecap="round" transform="rotate({angle:.1f} 100 100)"/>'
        f'<circle cx="100" cy="100" r="6" fill="{color}"/></svg>')


try:
    model, le_mag, le_risk = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()
if "mag" in model.feature_names_in_:
    st.error("This model was trained with magnitude as an input. Replace the .pkl files with the retrained ones.")
    st.stop()

st.session_state.setdefault("lat", 28.61)
st.session_state.setdefault("lon", 77.21)


st.title("SeismicVista")
st.markdown('<div class="sub">Earthquake severity estimate trained on USGS data, 2017 to 2026. '
            'Click the map to set the location.</div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

with left:
    with st.container(border=True):
        # Map first, so a click can update lat/lon before the sliders are drawn
        m = folium.Map(location=[20, 10], zoom_start=2, min_zoom=2, tiles=None,
                       world_copy_jump=True)
        folium.TileLayer(ESRI + "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                         attr="Tiles © Esri", max_zoom=16).add_to(m)
        folium.TileLayer(ESRI + "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                         attr="Tiles © Esri", max_zoom=16).add_to(m)
        folium.CircleMarker([st.session_state.lat, st.session_state.lon], radius=9,
                            color="#f0b429", fill=True, fill_opacity=0.85).add_to(m)
        out = st_folium(m, height=340, use_container_width=True,
                        returned_objects=["last_clicked"], key="map")
        click = out.get("last_clicked") if out else None
        if click:
            key = (round(click["lat"], 4), round(click["lng"], 4))
            if key != st.session_state.get("_last_click"):
                st.session_state["_last_click"] = key
                st.session_state.lat = max(-90.0, min(90.0, round(click["lat"], 2)))
                st.session_state.lon = round(((click["lng"] + 180) % 360) - 180, 2)
                st.rerun()

        c1, c2 = st.columns(2)
        lat = c1.slider("Latitude", -90.0, 90.0, step=0.1, key="lat")
        lon = c2.slider("Longitude", -180.0, 180.0, step=0.1, key="lon")
        c3, c4 = st.columns(2)
        depth = c3.slider("Depth (km)", 0, 700, 30)
        mag_types = [t for t in MAG_LABELS if t in set(le_mag.classes_)]
        mag_type = c4.selectbox("Magnitude type", mag_types,
                                index=mag_types.index("mb") if "mb" in mag_types else 0,
                                format_func=lambda t: f"{t} — {MAG_LABELS[t.lower()]}"
                                if t.lower() in MAG_LABELS else t,
                                help="Method used to measure magnitude. Large earthquakes are usually measured with "
                                     "mww/mwc and small ones with md/ml, so this input strongly "
                                     "pushes the result toward Critical or Low.")

pred, proba = predict(model, le_mag, le_risk, depth, lat, lon, mag_type)
color = RISK_COLORS[pred]
index = sum(proba[c] * RISK_WEIGHT[c] for c in RISK_ORDER)

with right:
    with st.container(border=True):
        st.subheader("Risk assessment")
        bars = "".join(
            f'<div class="pr"><span class="n">{c}</span><span class="t"><div class="b" '
            f'style="width:{proba[c]*100:.1f}%;background:{RISK_COLORS[c]}"></div></span>'
            f'<span class="v">{proba[c]*100:.1f}%</span></div>' for c in RISK_ORDER)
        cards = "".join(
            f'<div class="refcard{" on" if c == pred else ""}" style="border-left-color:{RISK_COLORS[c]}">'
            f'<div class="rn" style="color:{RISK_COLORS[c]}">{c}</div>'
            f'<div class="rr">{RISK_RANGE[c]}</div></div>' for c in RISK_ORDER)
        st.markdown(
            f'<div class="res">{gauge_html(index, color)}'
            f'<div class="lvl" style="color:{color}">{pred}</div></div>'
            f'<div style="margin-top:16px">{bars}</div>'
            f'<div class="reft">Magnitude bands</div><div class="refrow">{cards}</div>',
            unsafe_allow_html=True)

with st.expander("How does this work?"):
    st.markdown(
        "Risk levels are magnitude bands: **Low** below 5.0, **Moderate** 5.0 to 5.9, "
        "**High** 6.0 to 6.9, **Critical** 7.0 and above. From the location, depth and magnitude "
        "type, a gradient boosting model estimates which band a past-style earthquake most likely "
        "falls in. It never sees the actual magnitude, and it does **not** predict when or where "
        "the next earthquake will happen.")
