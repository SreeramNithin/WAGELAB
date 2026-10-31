from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -------------------------------------------------------------------
# WAGE//LAB — FIFA Player Wage Intelligence
# Model feature order is preserved from the original training data.
# -------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = APP_DIR / "wage_predictor_model"
FEATURES = [
    "Overall",
    "Potential",
    "Age",
    "Nationality",
    "Attacking_WorkRate",
    "Defensive_WorkRate",
]
WORK_RATES = {"Low": 0, "Medium": 1, "High": 2}

st.set_page_config(
    page_title="WAGE//LAB — Player Valuation",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------- Design -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#f4f5f7; --muted:#9ba3b4; --lime:#c8f169; --panel:#151a22; --line:rgba(255,255,255,.09); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(ellipse at 82% 0%, rgba(111,145,67,.13), transparent 34%), #0b0e13; color:var(--ink); }
.block-container { max-width:1320px; padding-top:2rem; padding-bottom:4rem; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { right:1rem; }
.hero { border:1px solid var(--line); border-radius:24px; padding:30px 34px; background:linear-gradient(115deg,rgba(200,241,105,.10),rgba(21,26,34,.92) 48%,rgba(21,26,34,.65)); position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; width:270px;height:270px;border:1px solid rgba(200,241,105,.15);border-radius:50%;right:7%;top:-145px;box-shadow:0 0 0 28px rgba(200,241,105,.025),0 0 0 58px rgba(200,241,105,.02); }
.eyebrow { color:var(--lime); font:500 11px 'DM Mono',monospace; letter-spacing:2.4px; text-transform:uppercase; }
.hero h1 { font-size:clamp(34px,5vw,58px); line-height:1.02; letter-spacing:-2.8px; margin:14px 0 10px; font-weight:800; color:#f7f8f9; }
.hero p { color:#aab2c0; max-width:640px; font-size:14px; line-height:1.7; margin:0; }
.hero-tag { display:inline-block; margin-top:22px; border:1px solid rgba(200,241,105,.32); color:var(--lime); padding:7px 11px; border-radius:999px; font:11px 'DM Mono',monospace; letter-spacing:.7px; }
.section-kicker { color:var(--lime);font:11px 'DM Mono',monospace;letter-spacing:1.7px;text-transform:uppercase;margin:24px 0 8px; }
.section-title { color:#f2f4f7;font-size:24px;font-weight:800;letter-spacing:-.8px;margin:0 0 5px; }
.section-sub { color:#929bab;font-size:13px;margin-bottom:18px; }
.panel { background:linear-gradient(145deg,rgba(25,31,41,.98),rgba(17,21,29,.98)); border:1px solid var(--line); border-radius:18px; padding:20px 22px; }
.metric-label { color:#9ba3b4;font:10px 'DM Mono',monospace;letter-spacing:1.3px;text-transform:uppercase; }
.metric-value { color:#f4f6f8;font-size:27px;font-weight:800;letter-spacing:-1px;margin-top:8px; }
.metric-note { color:#8f99a9;font-size:11px;margin-top:3px; }
div[data-testid="stMetric"] { background:#151a22;border:1px solid var(--line);padding:16px 18px;border-radius:15px; }
div[data-testid="stMetricLabel"] { color:#a2aaba; }
div[data-testid="stMetricValue"] { color:#f4f6f8; }
.stTabs [data-baseweb="tab-list"] { gap:8px;border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:#9ba3b4;padding:12px 18px; }
.stTabs [aria-selected="true"] { color:var(--lime)!important; }
.stButton>button { background:var(--lime);color:#10140b;border:0;border-radius:10px;font-weight:800;padding:.72rem 1.2rem; }
.stButton>button:hover { background:#d8ff83;color:#10140b; }
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div { background:#111720;border-color:#303947;color:#f4f5f7; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--lime); }
div[data-testid="stAlert"] { border-radius:12px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:12px;overflow:hidden; }
.footer { border-top:1px solid var(--line); margin-top:42px;padding-top:18px;color:#778192;font:10px 'DM Mono',monospace;letter-spacing:1px;display:flex;justify-content:space-between; }
.small-mono { color:#9ba3b4;font:10px 'DM Mono',monospace;letter-spacing:1px; }
div[data-testid="stForm"] { border:1px solid var(--line);border-radius:18px;padding:20px;background:#151a22; }
</style>
""", unsafe_allow_html=True)

# ------------------------------- Data -------------------------------
@st.cache_data
def load_data():
    wages = pd.read_csv(DATA_DIR / "mean_wages.csv")
    mapping = pd.read_csv(DATA_DIR / "nationality_mapping.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    wages["Wage(in Euro)"] = pd.to_numeric(wages["Wage(in Euro)"], errors="coerce")
    players["Wage"] = pd.to_numeric(players["Wage"], errors="coerce")
    players["Overall"] = pd.to_numeric(players["Overall"], errors="coerce")
    players["Potential"] = pd.to_numeric(players["Potential"], errors="coerce")
    players["Age"] = pd.to_numeric(players["Age"], errors="coerce")
    mapping["Model_Code"] = pd.to_numeric(mapping["Model_Code"], errors="raise").astype(int)
    return wages.dropna(), mapping, players.dropna(subset=["Wage", "Overall", "Age"])

@st.cache_resource(show_spinner="Loading the valuation engine…")
def load_model():
    # Streamlit Cloud: set MODEL_URL in App settings → Secrets.
    # Local development: place wage_predictor_model beside app.py.
    model_url = os.getenv("MODEL_URL", "").strip()
    try:
        model_url = model_url or st.secrets.get("MODEL_URL", "").strip()
    except Exception:
        pass

    if MODEL_PATH.exists():
        path = MODEL_PATH
    elif model_url:
        response = requests.get(model_url, timeout=(20, 300), stream=True)
        response.raise_for_status()
        downloaded = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
                    downloaded += len(chunk)
            path = Path(tmp.name)
        if downloaded < 1_000_000:
            path.unlink(missing_ok=True)
            raise RuntimeError("The model download looks incomplete. Check the MODEL_URL release asset.")
    else:
        raise FileNotFoundError(
            "Model file not found. Add wage_predictor_model beside app.py for local use, "
            "or configure MODEL_URL in Streamlit Cloud secrets."
        )

    with path.open("rb") as f:
        return pickle.load(f)

wages, mapping, players = load_data()
nationality_to_code = dict(zip(mapping["Nationality"], mapping["Model_Code"]))
model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

# ------------------------------- Header -------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Football economics / player valuation engine</div>
  <h1>WAGE<span style="color:#c8f169">//</span>LAB</h1>
  <p>A data-driven lens on football's wage market. Explore how player attributes, potential, age and nationality shape estimated weekly earnings.</p>
  <span class="hero-tag">FIFA PLAYER WAGE PREDICTOR &nbsp;·&nbsp; MODEL SERIES 01</span>
</div>
""", unsafe_allow_html=True)

# Small status strip
c1, c2, c3, c4 = st.columns(4)
c1.metric("PLAYER RECORDS", f"{len(players):,}", "Training reference dataset")
c2.metric("NATIONALITIES", f"{mapping['Nationality'].nunique():,}", "Original model encoding")
c3.metric("MODEL INPUTS", "06", "Original feature sequence")
c4.metric("CURRENCY", "EUR / WEEK", "Estimated wage")

if model is None:
    st.error("The prediction engine is not connected yet.")
    st.code(model_error or "Unknown model loading error")
    st.info("Deployment setup: upload the original model as a GitHub Release asset, then add MODEL_URL to Streamlit Cloud → App settings → Secrets. See README.md.")

st.markdown('<div class="section-kicker">01 / The workspace</div>', unsafe_allow_html=True)
tab_predict, tab_market, tab_method = st.tabs(["⚡  Player lab", "◈  Wage landscape", "◎  Model notes"])

# ------------------------------- Prediction Lab -------------------------------
with tab_predict:
    left, right = st.columns([1.02, .98], gap="large")
    with left:
        st.markdown('<div class="section-title">Build a player profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Adjust the six attributes used by the original trained model.</div>', unsafe_allow_html=True)
        with st.form("player_profile"):
            p1, p2 = st.columns(2)
            with p1:
                overall = st.slider("Overall rating", 1, 99, 75, help="Player's current overall rating.")
                age = st.slider("Age", 16, 45, 25)
                attacking = st.selectbox("Attacking work rate", list(WORK_RATES), index=1)
            with p2:
                potential = st.slider("Potential rating", 1, 99, 82)
                nationality = st.selectbox("Nationality", sorted(nationality_to_code), index=sorted(nationality_to_code).index("Argentina") if "Argentina" in nationality_to_code else 0)
                defensive = st.selectbox("Defensive work rate", list(WORK_RATES), index=1)
            submitted = st.form_submit_button("GENERATE WAGE ESTIMATE  ↗", use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Valuation output</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Your estimated weekly wage and market context.</div>', unsafe_allow_html=True)
        if submitted:
            if model is None:
                st.warning("Connect the model to enable predictions. Deployment instructions are in README.md.")
            else:
                input_row = [[
                    overall, potential, age, int(nationality_to_code[nationality]),
                    WORK_RATES[attacking], WORK_RATES[defensive]
                ]]
                try:
                    prediction = float(np.asarray(model.predict(input_row)).ravel()[0])
                    if not np.isfinite(prediction):
                        raise ValueError("The model returned a non-finite prediction.")
                    nat_avg = float(wages.loc[wages["Nationality"] == nationality, "Wage(in Euro)"].mean())
                    overall_avg = float(players.loc[players["Overall"].between(overall-2, overall+2), "Wage"].median())
                    st.markdown(f"""
                    <div class="panel">
                      <div class="eyebrow">Predicted weekly wage</div>
                      <div style="font-size:48px;font-weight:800;letter-spacing:-2px;color:#c8f169;margin:10px 0 4px">€{prediction:,.0f}</div>
                      <div style="color:#9ba3b4;font-size:12px">EUR / week · Random Forest estimate</div>
                      <hr style="border-color:rgba(255,255,255,.10);margin:20px 0">
                      <div style="display:flex;justify-content:space-between;gap:15px">
                        <div><div class="metric-label">Nationality average</div><div style="font-size:19px;font-weight:800;margin-top:6px">€{nat_avg:,.0f}</div></div>
                        <div><div class="metric-label">Similar overall median</div><div style="font-size:19px;font-weight:800;margin-top:6px">{"€{:,.0f}".format(overall_avg) if pd.notna(overall_avg) else "N/A"}</div></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    delta = prediction - nat_avg
                    st.metric("Compared with nationality average", f"€{prediction:,.0f}", f"{delta:+,.0f} EUR / week vs average")
                    st.caption("This is a model estimate, not an official contract value. Actual wages depend on club, league, contract terms and other factors not represented in the model.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        else:
            st.markdown("""
            <div class="panel" style="min-height:300px;display:flex;flex-direction:column;justify-content:center">
              <div class="small-mono">AWAITING PLAYER INPUT</div>
              <div style="font-size:30px;font-weight:800;letter-spacing:-1px;margin:12px 0;color:#f4f6f8">The next valuation<br>starts with a profile.</div>
              <div style="font-size:13px;color:#929bab;line-height:1.7">Set the player attributes on the left and generate an estimated weekly wage. Your result will appear here with a market benchmark.</div>
              <div style="margin-top:25px;color:#c8f169;font:11px 'DM Mono',monospace">6 FEATURES · 1 PREDICTION · EUR/WEEK</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------- Market landscape -------------------------------
with tab_market:
    st.markdown('<div class="section-title">The global wage landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore average wages by nationality and the distribution of wages across the original FIFA player dataset.</div>', unsafe_allow_html=True)
    map_data = wages.copy()
    try:
        import pycountry
        iso_map = {c.name: c.alpha_3 for c in pycountry.countries}
        # Common naming aliases found in football datasets.
        iso_map.update({"England":"GBR", "Scotland":"GBR", "Wales":"GBR", "Northern Ireland":"GBR", "Korea Republic":"KOR", "Côte d'Ivoire":"CIV", "DR Congo":"COD"})
        map_data["iso"] = map_data["Nationality"].map(iso_map)
        fig_map = px.choropleth(
            map_data.dropna(subset=["iso"]), locations="iso", locationmode="ISO-3",
            color="Wage(in Euro)", hover_name="Nationality",
            color_continuous_scale=["#17202a", "#55713a", "#c8f169"],
            labels={"Wage(in Euro)": "Mean wage (€ / week)"},
            template="plotly_dark"
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=430,
            coloraxis_colorbar=dict(title="EUR / week")
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as exc:
        st.warning(f"Map unavailable: {exc}")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        top = wages.sort_values("Wage(in Euro)", ascending=False).head(15).sort_values("Wage(in Euro)")
        fig = px.bar(top, x="Wage(in Euro)", y="Nationality", orientation="h",
                     color="Wage(in Euro)", color_continuous_scale=["#55713a","#c8f169"],
                     template="plotly_dark", labels={"Wage(in Euro)":"Mean wage (€ / week)"})
        fig.update_layout(title="Top 15 nationality averages", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=500, coloraxis_showscale=False,
                          margin=dict(l=5,r=5,t=50,b=5))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.scatter(players.sample(min(2500, len(players)), random_state=7),
                          x="Overall", y="Wage", color="Age", size="Potential",
                          hover_data=["Nationality"], color_continuous_scale=["#55713a","#c8f169"],
                          template="plotly_dark", labels={"Wage":"Weekly wage (€)"})
        fig2.update_layout(title="Overall rating vs weekly wage", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=500,
                           margin=dict(l=5,r=5,t=50,b=5))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Explore nationality averages"):
        st.dataframe(wages.sort_values("Wage(in Euro)", ascending=False).rename(
            columns={"Wage(in Euro)":"Mean wage (€ / week)"}
        ), use_container_width=True, hide_index=True)

# ------------------------------- Methodology -------------------------------
with tab_method:
    st.markdown('<div class="section-title">Inside the engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A transparent view of the model inputs, training reference data and important limitations.</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><div class="eyebrow">Model architecture</div><h3 style="margin:12px 0 6px;color:#f4f6f8">Random Forest Regressor</h3><p style="color:#9ba3b4;font-size:13px;line-height:1.7">The original trained model is retained. The application passes the six features in their original training order and uses the original nationality-to-code mapping.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="eyebrow">Prediction inputs</div><h3 style="margin:12px 0 6px;color:#f4f6f8">Six model features</h3><p style="color:#9ba3b4;font-size:13px;line-height:1.9">01 &nbsp; Overall rating<br>02 &nbsp; Potential rating<br>03 &nbsp; Age<br>04 &nbsp; Nationality code<br>05 &nbsp; Attacking work rate<br>06 &nbsp; Defensive work rate</p></div>', unsafe_allow_html=True)
    st.markdown("#### Feature importance")
    if model is not None and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
        importance = importance.sort_values("Importance", ascending=True)
        fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#55713a","#c8f169"],
                         template="plotly_dark")
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False, height=330, margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.caption("Feature importance will appear once the model is connected.")
    st.info("Model limitation: the predictor only knows the six features used during training. It does not account for club, league, contract length, agent fees, transfer status, bonuses or current market inflation. Treat outputs as indicative estimates.")

st.markdown("""
<div class="footer">
  <span>WAGE//LAB &nbsp;·&nbsp; FOOTBALL ECONOMICS</span>
  <span>BUILT WITH DATA / NOT A CONTRACT VALUATION</span>
</div>
""", unsafe_allow_html=True)
