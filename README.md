# 🌤️ MLOps Weather Forecasting — Technopark & Thampanoor

> **1-Week Sprint · 100% Free Tools · End-to-End MLOps**
>
> A Streamlit app that shows daily weather forecasts for two Thiruvananthapuram locations,
> backed by an LSTM model that retrains itself every day via a GitHub Actions + DVC pipeline.

---

## 📐 Architecture

```
Open-Meteo API
     │  (hourly weather data)
     ▼
src/collect.py  ──► data/raw/*.csv  ──► DVC (Google Drive remote)
     │
src/preprocess.py ──► data/processed/*.npy + models/*_scaler.pkl
     │
src/train.py ──► models/*_model.keras + metrics.json + version.json
     │
app.py ──► Streamlit Community Cloud  (auto-deploy on git push)
     ▲
GitHub Actions  (runs daily at 01:00 UTC)
```

---

## 🗂️ Repository Structure

```
.
├── src/
│   ├── collect.py          # Downloads data from Open-Meteo API
│   ├── preprocess.py       # Cleans, engineers features, builds windows
│   └── train.py            # Trains LSTM, saves model, writes metrics & version
├── data/
│   ├── raw/                # Raw CSVs from Open-Meteo (DVC-tracked)
│   └── processed/          # NumPy arrays for training (DVC-tracked)
├── models/                 # Trained .keras models + scaler.pkl (DVC-tracked)
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml   # Daily CI/CD workflow
├── app.py                  # Streamlit application
├── dvc.yaml                # Pipeline stage definitions
├── params.yaml             # All hyperparameters and config
├── metrics.json            # Latest MAE / RMSE per region
├── version.json            # Model version, training date, git SHA
└── requirements.txt        # Python dependencies
```

---

## 🚀 Day-by-Day Setup Guide

### Day 1 — Setup & Data

**1. Create GitHub repo & clone it**
```bash
git clone https://github.com/YOUR_USERNAME/mlops-weather-tvm.git
cd mlops-weather-tvm
```

**2. Copy all project files into the repo**, then:
```bash
pip install -r requirements.txt
```

**3. Initialise DVC and set up Google Drive remote**

First, create a Google Cloud service account:
- Go to https://console.cloud.google.com → IAM → Service Accounts
- Create a service account, grant it "Editor" role on Google Drive
- Download the JSON key file

Then configure DVC:
```bash
dvc init
dvc remote add -d myremote gdrive://YOUR_GDRIVE_FOLDER_ID
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path /path/to/key.json
git add .dvc/config
git commit -m "init: DVC with Google Drive remote"
```

**4. Fetch initial 6 months of data**
```bash
python src/collect.py
dvc add data/raw/technopark.csv data/raw/thampanoor.csv
dvc push
git add data/raw/.gitignore data/raw/*.csv.dvc
git commit -m "data: initial 6-month fetch"
git push
```

---

### Day 2 — Preprocessing

```bash
python src/preprocess.py
dvc repro preprocess    # or full: dvc repro
git add dvc.lock
git commit -m "pipeline: add preprocess stage"
git push
```

Inspect outputs:
```python
import numpy as np
X = np.load("data/processed/technopark_X_train.npy")
print(X.shape)   # should be (samples, 48, 6)
```

---

### Day 3 — Model Training

```bash
dvc repro train          # runs preprocess + train if needed
cat metrics.json         # check MAE / RMSE
cat version.json         # check version metadata
git add dvc.lock metrics.json version.json
git commit -m "model: first LSTM training run"
git push
```

To tune hyperparameters, edit `params.yaml` and re-run `dvc repro`.
DVC only re-runs stages whose inputs changed.

---

### Day 4 — Streamlit App

**Test locally:**
```bash
streamlit run app.py
```
Open http://localhost:8501 — you should see the forecast chart.

**Deploy to Streamlit Community Cloud:**
1. Go to https://share.streamlit.io
2. Click "New app" → connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **Deploy**

---

### Day 5 — GitHub Actions

**Add secrets to your GitHub repo** (Settings → Secrets → Actions):

| Secret name | Value |
|---|---|
| `GDRIVE_CREDENTIALS_DATA` | `base64 -i service_account.json` output |
| `GIT_TOKEN` | GitHub Personal Access Token (repo scope) |

**Trigger manually to test:**
- Go to Actions tab → "Daily Weather Pipeline" → "Run workflow"

Watch the logs. After it succeeds, check that `version.json` in your repo has been updated.

---

### Day 6 — Polish & Error Handling

- `src/collect.py` already includes retry logic (3 attempts with backoff)
- Run the full pipeline for both regions: `dvc repro`
- Improve UI in `app.py` if desired (labels, colours, etc.)
- Update README with your live Streamlit URL

---

### Day 7 — Review & Buffer

- Wait for 01:00 UTC automated run
- Check that the Streamlit app shows a new model version & training date
- Record a screen demo
- Submit your GitHub repo URL to Paatshala

---

## 🔑 GitHub Secrets

| Secret | How to create |
|--------|---------------|
| `GDRIVE_CREDENTIALS_DATA` | Base64-encode your Google service account JSON key: `base64 -i key.json \| tr -d '\n'` |
| `GIT_TOKEN` | GitHub → Settings → Developer settings → Personal access tokens → Classic → `repo` scope |

---

## ⚙️ Hyperparameter Tuning

All parameters live in `params.yaml`. Change any value and run `dvc repro` —
only affected stages re-execute.

| Parameter | Default | Effect |
|---|---|---|
| `model.lstm_units` | 64 | LSTM capacity |
| `model.dropout` | 0.2 | Regularisation |
| `model.lookback` | 48 | Hours of history input |
| `model.horizon` | 24 | Hours ahead to forecast |
| `model.epochs` | 30 | Max training epochs |
| `model.patience` | 5 | Early stopping patience |
| `model.batch_size` | 32 | Mini-batch size |
| `model.learning_rate` | 0.001 | Adam LR |

---

## 🎓 MLOps Concepts Covered

- ✅ Data versioning with DVC
- ✅ Reproducible ML pipelines (DAG stages)
- ✅ Parameterised training (`params.yaml`)
- ✅ Time-series model training & evaluation
- ✅ Model artifact management
- ✅ Automated daily retraining (GitHub Actions)
- ✅ CI/CD for ML (code push → auto-deploy)
- ✅ Model metadata tracking (`version.json`)
- ✅ Free-tier cloud deployment (Streamlit Cloud)

---

## 🛠️ Free Tools Stack

| What | Tool |
|---|---|
| Weather data | Open-Meteo API (no key needed) |
| ML pipeline | DVC |
| Data storage | Google Drive (15 GB free) |
| Deep learning | TensorFlow / Keras |
| Web app | Streamlit |
| App hosting | Streamlit Community Cloud |
| CI/CD | GitHub Actions |
| Code hosting | GitHub |
