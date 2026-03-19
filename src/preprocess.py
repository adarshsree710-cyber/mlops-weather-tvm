"""
preprocess.py — Cleans raw CSVs, engineers features, normalises data,
and builds sliding-window arrays ready for LSTM training.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"

LOCATIONS = ["technopark", "thampanoor"]

FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "hour_of_day",
    "day_of_week",
]
TARGET_COL = "temperature_2m"

# Window sizes (can be overridden via params.yaml / env)
LOOKBACK  = int(os.getenv("LOOKBACK",  48))   # hours of history fed to LSTM
HORIZON   = int(os.getenv("HORIZON",   24))   # hours ahead to predict
TRAIN_RATIO = 0.80


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    before = len(df)
    df.dropna(subset=["temperature_2m"], inplace=True)
    after = len(df)
    if before != after:
        log.info("Dropped %d rows with null temperature_2m", before - after)

    # Fill remaining NaN in non-target columns with forward fill then 0
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"]  = df["datetime"].dt.hour.astype(int)
    df["day_of_week"]  = df["datetime"].dt.dayofweek.astype(int)
    return df


def build_windows(data: np.ndarray, lookback: int, horizon: int):
    """Slide over time-series to build (X, y) pairs.

    X shape: (samples, lookback, features)
    y shape: (samples, horizon)  — temperature only
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])           # all features
        y.append(data[i + lookback : i + lookback + horizon, 0])  # temp only (col 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def chronological_split(X: np.ndarray, y: np.ndarray, ratio: float):
    split = int(len(X) * ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    for loc in LOCATIONS:
        log.info("Processing %s …", loc)
        csv_path = os.path.join(RAW_DIR, f"{loc}.csv")

        if not os.path.exists(csv_path):
            log.error("Raw CSV not found: %s — run collect.py first.", csv_path)
            raise FileNotFoundError(csv_path)

        df = load_and_clean(csv_path)
        df = add_time_features(df)
        df.sort_values("datetime", inplace=True)

        # Select & order feature columns
        feature_data = df[FEATURE_COLS].values

        # Fit scaler on training portion only to prevent data leakage
        train_end = int(len(feature_data) * TRAIN_RATIO)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(feature_data[:train_end])
        scaled = scaler.transform(feature_data)

        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f"{loc}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info("Scaler saved → %s", scaler_path)

        # Build sliding windows
        X, y = build_windows(scaled, LOOKBACK, HORIZON)
        log.info("Windows built: X=%s  y=%s", X.shape, y.shape)

        # Chronological train/test split
        X_train, y_train, X_test, y_test = chronological_split(X, y, TRAIN_RATIO)
        log.info("Split: train=%d  test=%d", len(X_train), len(X_test))

        # Save arrays
        prefix = os.path.join(PROCESSED_DIR, loc)
        np.save(f"{prefix}_X_train.npy", X_train)
        np.save(f"{prefix}_y_train.npy", y_train)
        np.save(f"{prefix}_X_test.npy",  X_test)
        np.save(f"{prefix}_y_test.npy",  y_test)
        log.info("Saved processed arrays for %s", loc)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
