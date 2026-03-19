import os
import json
import logging
import pickle
import subprocess
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
LOCATIONS = ["technopark", "thampanoor"]
VERSION_COUNTER_FILE = os.path.join(MODELS_DIR, ".version_counter")

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except:
        return "unknown"

def next_version():
    os.makedirs(MODELS_DIR, exist_ok=True)
    counter = 1
    if os.path.exists(VERSION_COUNTER_FILE):
        with open(VERSION_COUNTER_FILE) as f:
            counter = int(f.read().strip()) + 1
    with open(VERSION_COUNTER_FILE, "w") as f:
        f.write(str(counter))
    return f"v{counter}.0"

def train_location(loc):
    prefix = os.path.join(PROCESSED_DIR, loc)
    X_train = np.load(f"{prefix}_X_train.npy").reshape(len(np.load(f"{prefix}_X_train.npy")), -1)
    y_train = np.load(f"{prefix}_y_train.npy")
    X_test  = np.load(f"{prefix}_X_test.npy").reshape(len(np.load(f"{prefix}_X_test.npy")), -1)
    y_test  = np.load(f"{prefix}_y_test.npy")

    log.info("[%s] Training Random Forest...", loc)
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    log.info("[%s] MAE=%.4f  RMSE=%.4f", loc, mae, rmse)

    model_path = os.path.join(MODELS_DIR, f"{loc}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("[%s] Model saved → %s", loc, model_path)
    return mae, rmse

def main():
    results = {}
    for loc in LOCATIONS:
        mae, rmse = train_location(loc)
        results[loc] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    version = next_version()
    version_data = {
        "version": version,
        "trained_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": get_git_sha(),
        "rmse_technopark": results["technopark"]["rmse"],
        "mae_technopark":  results["technopark"]["mae"],
        "rmse_thampanoor": results["thampanoor"]["rmse"],
        "mae_thampanoor":  results["thampanoor"]["mae"],
    }
    with open("version.json", "w") as f:
        json.dump(version_data, f, indent=2)
    log.info("Done! %s", json.dumps(version_data, indent=2))

if __name__ == "__main__":
    main()