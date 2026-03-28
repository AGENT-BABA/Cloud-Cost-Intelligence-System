"""
anomaly_model_eval.py  (lives in: CLOUD COST INTEL/Isolation_Forest/)
Realistic evaluation: train on first 5 days, test on last 2.
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
# __file__ → Isolation_Forest/anomaly_model_eval.py
# go up one level → CLOUD COST INTEL/  then into DataStorage/
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_STORE = os.path.join(THIS_DIR, "..", "Data_Collector")

CSV_PATH = os.path.join(DATA_STORE, "Processed", "final_data_pipeline.csv")
GT_PATH  = os.path.join(DATA_STORE, "Raw",       "anomaly_ground_truth.json")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

with open(GT_PATH) as f:
    gt = json.load(f)

df["label"] = df["timestamp"].map(gt["anomaly_labels"]).fillna(1).astype(int)

FEATURES = [
    "cpu_utilization", "network_in", "network_out",
    "memory_usage", "requests", "error_rate",
    "storage_free", "billing_rate", "cost_per_hour",
]

# ── Temporal split: train days 1-5, test days 6-7 ────────────────────────────
df["date"] = df["timestamp"].str[:10]
train_days = ["2026-03-21", "2026-03-22", "2026-03-23", "2026-03-24", "2026-03-25"]
test_days  = ["2026-03-26", "2026-03-27"]

train = df[df["date"].isin(train_days)]
test  = df[df["date"].isin(test_days)]

# ── 5% label noise on train (simulate real-world uncertainty) ─────────────────
rng = np.random.default_rng(7)
noise_mask = rng.random(len(train)) < 0.05
train_labels_noisy = train["label"].values.copy()
train_labels_noisy[noise_mask] *= -1

X_train = train[FEATURES].values
X_test  = test[FEATURES].values
y_test  = test["label"].values

scaler  = RobustScaler().fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_te_sc = scaler.transform(X_test)

cont = float(np.clip((train_labels_noisy == -1).mean(), 0.05, 0.4))

model = IsolationForest(
    n_estimators=300,
    max_samples=min(256, len(X_train)),
    contamination=cont,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_tr_sc)

preds  = model.predict(X_te_sc)
scores = -model.decision_function(X_te_sc)

# ── Results ───────────────────────────────────────────────────────────────────
print("="*60)
print("REALISTIC EVAL — Train: days 1-5 | Test: days 6-7")
print(f"Train size: {len(train)} | Test size: {len(test)}")
print(f"Test anomalies: {(y_test==-1).sum()} / {len(y_test)}")
print("="*60)

print("\n── Classification Report ──")
print(classification_report(y_test, preds,
      target_names=["Anomaly (-1)", "Normal (1)"], digits=4))

print("── Confusion Matrix ──")
cm = confusion_matrix(y_test, preds, labels=[-1, 1])
print(f"            Pred:-1   Pred:+1")
print(f"True:-1  {cm[0,0]:8d}  {cm[0,1]:8d}   ← anomalies")
print(f"True:+1  {cm[1,0]:8d}  {cm[1,1]:8d}   ← normal")

y_bin = (y_test == -1).astype(int)
auroc = roc_auc_score(y_bin, scores)
auprc = average_precision_score(y_bin, scores)

tp = cm[0,0]; fp = cm[1,0]; fn = cm[0,1]
precision = tp/(tp+fp) if (tp+fp) else 0
recall    = tp/(tp+fn) if (tp+fn) else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

print(f"\n── Summary Metrics ──")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  AUROC       : {auroc:.4f}")
print(f"  AUPRC       : {auprc:.4f}")
print(f"  False Alarms: {fp}")
print(f"  Missed Anom : {fn}")

# ── Feature contribution ──────────────────────────────────────────────────────
print("\n── Feature Contribution (mean diff: anomaly vs normal rows) ──")
test_df = test.copy().reset_index(drop=True)
test_df["anomaly_score"] = scores
test_df["predicted"]     = preds
anom_rows = test_df[test_df["predicted"] == -1]
norm_rows = test_df[test_df["predicted"] ==  1]

feat_importance = {f: abs(anom_rows[f].mean() - norm_rows[f].mean()) for f in FEATURES}
max_val = max(feat_importance.values())
for f, v in sorted(feat_importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(v / max_val * 30)
    print(f"  {f:20s}: {bar} ({v:.2f})")