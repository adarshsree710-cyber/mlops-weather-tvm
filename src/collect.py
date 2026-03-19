"""
collect.py — Downloads hourly weather data from Open-Meteo API.
Fetches 6 months of history on first run, then appends yesterday's 24 rows on each subsequent run.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, date

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LOCATIONS = {
    "technopark":  {"lat": 8.5574, "lon": 76.8800},
    "thampanoor":  {"lat": 8.4875, "lon": 76.9525},
}

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
RAW_DIR       = os.path.join("data", "raw")
INITIAL_MONTHS = 6


def fetch_weather(lat: float, lon: float, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo archive endpoint."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(VARIABLES),
        "start_date": start,
        "end_date":   end,
        "timezone":   "Asia/Kolkata",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            df = pd.DataFrame(hourly)
            df.rename(columns={"time": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except requests.exceptions.RequestException as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                log.error("All %d attempts failed. Exiting.", max_retries)
                sys.exit(1)


def get_date_range(csv_path: str):
    """Return (start_date, end_date) strings for the data fetch."""
    yesterday = date.today() - timedelta(days=1)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, usecols=["datetime"], parse_dates=["datetime"])
        last_dt = existing["datetime"].max()
        start = (last_dt + timedelta(hours=1)).date()
        log.info("Existing CSV found. Appending from %s to %s", start, yesterday)
    else:
        start = date.today() - timedelta(days=INITIAL_MONTHS * 30)
        log.info("No existing CSV. Fetching initial %d months: %s → %s",
                 INITIAL_MONTHS, start, yesterday)

    if start > yesterday:
        log.info("CSV already up to date (last date: %s). Nothing to fetch.", yesterday)
        return None, None

    return start.isoformat(), yesterday.isoformat()


def save_or_append(df: pd.DataFrame, csv_path: str):
    """Append new rows to existing CSV, or create it if absent."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, parse_dates=["datetime"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset="datetime", keep="last", inplace=True)
        combined.sort_values("datetime", inplace=True)
        combined.to_csv(csv_path, index=False)
        log.info("Appended %d rows → %s (total: %d rows)", len(df), csv_path, len(combined))
    else:
        df.to_csv(csv_path, index=False)
        log.info("Created %s with %d rows", csv_path, len(df))


def main():
    for name, coords in LOCATIONS.items():
        csv_path = os.path.join(RAW_DIR, f"{name}.csv")
        start, end = get_date_range(csv_path)

        if start is None:
            continue

        log.info("Fetching %s  (%s → %s) …", name, start, end)
        df = fetch_weather(coords["lat"], coords["lon"], start, end)

        if df.empty:
            log.warning("Empty response for %s — skipping.", name)
            continue

        save_or_append(df, csv_path)

    log.info("Data collection complete.")


if __name__ == "__main__":
    main()
