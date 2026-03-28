"""
anomaly_model.py  (lives in: CLOUD COST INTEL/Isolation_Forest/)
Trains Isolation Forest on pipeline output, evaluates against ground-truth.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ → Isolation_Forest/anomaly_model.py
# go up one level → CLOUD COST INTEL/  then into DataStorage/
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_STORE = os.path.join(THIS_DIR, "..", "Data_Collector")

CSV_PATH = os.path.join(DATA_STORE, "Processed", "final_data_pipeline.csv")
GT_PATH  = os.path.join(DATA_STORE, "Raw",       "anomaly_ground_truth.json")
OUT_PATH = os.path.join(DATA_STORE, "Processed", "final_data_labelled_predicted.csv")

# ── 1. Load pipeline CSV ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

# ── 2. Load ground-truth labels ───────────────────────────────────────────────
with open(GT_PATH) as f:
    gt = json.load(f)

df["label"] = df["timestamp"].map(gt["anomaly_labels"]).fillna(1).astype(int)

anom_count = (df["label"] == -1).sum()
norm_count = (df["label"] ==  1).sum()
print(f"Anomalies: {anom_count} ({anom_count/len(df)*100:.1f}%)  |  Normal: {norm_count}")

# ── 3. Features ────────────────────────────────────────────────────────────────
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

X = df[FEATURES].values
y = df["label"].values

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train Isolation Forest ─────────────────────────────────────────────────
contamination = anom_count / len(df)
model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=contamination,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_scaled)

# ── 5. Predict ────────────────────────────────────────────────────────────────
preds          = model.predict(X_scaled)
scores         = model.decision_function(X_scaled)
scores_for_auc = -scores   # higher = more anomalous

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ANOMALY DETECTION — EVALUATION REPORT")
print("="*60)

print("\n── Classification Report ──")
print(classification_report(y, preds,
      target_names=["Anomaly (-1)", "Normal (1)"], digits=4))

print("── Confusion Matrix ──")
cm = confusion_matrix(y, preds, labels=[-1, 1])
print(f"            Pred:-1   Pred:+1")
print(f"True:-1  {cm[0,0]:8d}  {cm[0,1]:8d}   ← anomalies")
print(f"True:+1  {cm[1,0]:8d}  {cm[1,1]:8d}   ← normal")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

auroc = roc_auc_score((y == -1).astype(int), scores_for_auc)
auprc = average_precision_score((y == -1).astype(int), scores_for_auc)

print(f"\n── Summary Metrics (Anomaly = Positive Class) ──")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  AUROC       : {auroc:.4f}")
print(f"  AUPRC       : {auprc:.4f}")
print(f"  False Alarms: {fp}  (normal rows flagged as anomaly)")
print(f"  Missed Anom : {fn}  (anomaly rows missed)")

# ── 7. Save labelled output ───────────────────────────────────────────────────
df["predicted_label"] = preds
df["anomaly_score"]   = scores_for_auc.round(6)
df.to_csv(OUT_PATH, index=False)
print(f"\nLabelled predictions saved → {os.path.abspath(OUT_PATH)}")
print("\n" + "="*60)
print("DONE")
print("="*60)