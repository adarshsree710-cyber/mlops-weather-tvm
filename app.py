import json, os, pickle
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="TVM Weather Forecast", page_icon="🌤️", layout="wide")

LOCATIONS = {
    "technopark": {"label": "Technopark", "lat": 8.5574, "lon": 76.8800},
    "thampanoor": {"label": "Thampanoor", "lat": 8.4875, "lon": 76.9525},
}
LOOKBACK = 48
HORIZON  = 24
FEATURE_COLS = ["temperature_2m","relative_humidity_2m","precipitation","wind_speed_10m","hour_of_day","day_of_week"]

@st.cache_data(ttl=1800)
def fetch_recent(lat, lon):
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=4)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "Asia/Kolkata"
    }, timeout=20)
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time":"datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df.ffill(inplace=True); df.fillna(0, inplace=True)
    return df.tail(LOOKBACK).reset_index(drop=True)

@st.cache_resource
def load_model(loc):
    path = os.path.join("models", f"{loc}_model.pkl")
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return pickle.load(f)

@st.cache_resource
def load_scaler(loc):
    path = os.path.join("models", f"{loc}_scaler.pkl")
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return pickle.load(f)

def predict(loc, df):
    model = load_model(loc); scaler = load_scaler(loc)
    if model is None or scaler is None or df.empty: return None
    features = scaler.transform(df[FEATURE_COLS].values)
    X = features.flatten().reshape(1, -1)
    return model.predict(X)[0] if hasattr(model.predict(X)[0], '__len__') else np.full(HORIZON, model.predict(X)[0])

version_data = json.load(open("version.json")) if os.path.exists("version.json") else {}

st.title("🌤️ Thiruvananthapuram Weather Forecast")
st.markdown("24-hour forecasts powered by a Random Forest model, retrained daily.")
if version_data:
    st.caption(f"Model {version_data.get('version')} | Trained: {version_data.get('trained_on')} | RMSE Technopark: {version_data.get('rmse_technopark',0):.2f}°C | RMSE Thampanoor: {version_data.get('rmse_thampanoor',0):.2f}°C")
st.divider()

tabs = st.tabs([v["label"] for v in LOCATIONS.values()])
for tab, (loc_key, loc_meta) in zip(tabs, LOCATIONS.items()):
    with tab:
        with st.spinner(f"Loading {loc_meta['label']}..."):
            df = fetch_recent(loc_meta["lat"], loc_meta["lon"])
            forecast = predict(loc_key, df)

        col1, col2 = st.columns([4,1])
        with col2:
            st.subheader("📊 Quick Stats")
            if forecast is not None:
                st.metric("Min Forecast", f"{float(np.min(forecast)):.1f}°C")
                st.metric("Max Forecast", f"{float(np.max(forecast)):.1f}°C")
            if not df.empty:
                st.metric("Precip Chance", f"{(df['precipitation']>0).mean()*100:.1f}%")
                st.metric("Avg Wind", f"{df['wind_speed_10m'].mean():.1f} km/h")

        with col1:
            st.subheader(f"🌡️ Temperature — {loc_meta['label']}")
            fig = go.Figure()
            if not df.empty:
                fig.add_trace(go.Scatter(x=df["datetime"], y=df["temperature_2m"], mode="lines", name="Observed (last 48h)", line=dict(color="#4A90D9", width=2)))
                last_time = df["datetime"].iloc[-1]
            else:
                last_time = datetime.utcnow()
            if forecast is not None:
                times = [last_time + timedelta(hours=i+1) for i in range(HORIZON)]
                vals  = [float(forecast)] * HORIZON if not hasattr(forecast, '__len__') else [float(x) for x in forecast]
                fig.add_trace(go.Scatter(x=times, y=vals, mode="lines+markers", name="Forecast (next 24h)", line=dict(color="#E07B39", width=2, dash="dash")))
            else:
                st.warning("Model not found. Run python src/train.py first.")
            fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)", hovermode="x unified", height=380)
            st.plotly_chart(fig, use_container_width=True)
