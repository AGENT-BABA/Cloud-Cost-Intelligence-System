"""
train.py
--------
Trains the Isolation Forest model on historical data and saves it to disk.

Run this ONCE (or whenever you have new labelled historical data):
    python train.py

Outputs saved to Isolation_Forest/model/:
    model.joblib   — the trained IsolationForest
    scaler.joblib  — the fitted RobustScaler
    metadata.json  — feature list, contamination rate, training timestamp
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_STORE = os.path.join(THIS_DIR, "..", "Data_Collector")
MODEL_DIR  = os.path.join(THIS_DIR, "model")

CSV_PATH = os.path.join(DATA_STORE, "Processed", "final_data_pipeline.csv")
GT_PATH  = os.path.join(DATA_STORE, "Raw",       "anomaly_ground_truth.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Features (must stay identical in predict.py) ───────────────────────────────
FEATURES = [
    "cpu_utilization",
    "network_in",
    "network_out",
    "memory_usage",
    "requests",
    "error_rate",
    "storage_free",
    "billing_rate",
    "cost_per_hour",
]

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("[train] Loading training data...")
df = pd.read_csv(CSV_PATH)
print(f"[train] Loaded {len(df)} rows.")

with open(GT_PATH) as f:
    gt = json.load(f)

df["label"] = df["timestamp"].map(gt["anomaly_labels"]).fillna(1).astype(int)

anom_count   = (df["label"] == -1).sum()
contamination = anom_count / len(df)
print(f"[train] Anomaly rate: {contamination:.4f} ({anom_count}/{len(df)})")

# ── 2. Scale ───────────────────────────────────────────────────────────────────
X = df[FEATURES].values
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Train ───────────────────────────────────────────────────────────────────
print("[train] Training Isolation Forest...")
model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=contamination,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_scaled)
print("[train] Training complete.")

# ── 4. Save model + scaler ─────────────────────────────────────────────────────
model_path  = os.path.join(MODEL_DIR, "model.joblib")
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

joblib.dump(model,  model_path)
joblib.dump(scaler, scaler_path)
print(f"[train] Model  saved → {model_path}")
print(f"[train] Scaler saved → {scaler_path}")

# ── 5. Save metadata ───────────────────────────────────────────────────────────
metadata = {
    "trained_at":    datetime.now(timezone.utc).isoformat(),
    "n_rows":        len(df),
    "n_anomalies":   int(anom_count),
    "contamination": round(float(contamination), 6),
    "features":      FEATURES,
    "model_params": {
        "n_estimators": 200,
        "max_samples":  "auto",
        "random_state": 42,
    },
}
meta_path = os.path.join(MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[train] Metadata saved → {meta_path}")
print("\n[train] Done. Run predict.py to use the model on new data.")
