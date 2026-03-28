"""
predict.py
----------
Loads the saved Isolation Forest model and runs predictions on new data.

Can be used in two ways:

  1. As a CLI — pass a CSV of new metric data:
         python predict.py --input smoke_latest.csv

  2. As a module — called by the full pipeline:
         from Isolation_Forest.predict import detect_anomalies
         anomalies = detect_anomalies(df)

Returns a list of anomaly dicts, each containing all metric fields
plus 'anomaly' (-1) and 'score' (float). These are passed directly to
rag_pipeline.pipeline.run_pipeline().
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(THIS_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH   = os.path.join(MODEL_DIR, "metadata.json")


def _load_model():
    """Load the saved model and scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            "Run: python Isolation_Forest/train.py"
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"[predict] Model loaded. Trained at: {meta['trained_at']}")
    print(f"[predict] Features : {meta['features']}")
    return model, scaler, meta


def detect_anomalies(df: pd.DataFrame, threshold: float = 0.0) -> list[dict]:
    """
    Run anomaly detection on a DataFrame of new AWS metrics.

    Args:
        df:         DataFrame with columns matching the training feature list.
        threshold:  Anomaly score threshold. Rows with score below this are
                    flagged. Default 0.0 catches all model-flagged anomalies.

    Returns:
        List of dicts — one per anomalous row — containing all original columns
        plus 'anomaly' (-1) and 'score' (the raw decision_function value).
        Returns an empty list if no anomalies are found.
    """
    model, scaler, meta = _load_model()
    features = meta["features"]

    # Validate all required features are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"[predict] Missing columns in input data: {missing}")

    X        = df[features].values
    X_scaled = scaler.transform(X)       # use transform, NOT fit_transform

    preds  = model.predict(X_scaled)            # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)  # lower = more anomalous

    df = df.copy()
    df["anomaly"] = preds
    df["score"]   = scores.round(6)

    anomaly_df = df[df["anomaly"] == -1]
    print(f"[predict] Rows evaluated: {len(df)} | Anomalies detected: {len(anomaly_df)}")

    return anomaly_df.to_dict(orient="records")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isolation Forest anomaly detection on new metric data.")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a CSV file of new AWS metric data (e.g. smoke_latest.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional: path to save detected anomalies as a CSV file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[predict] ERROR: Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"[predict] Loaded {len(df)} rows from {args.input}")

    anomalies = detect_anomalies(df)

    if not anomalies:
        print("[predict] No anomalies detected.")
    else:
        print(f"\n[predict] {len(anomalies)} anomaly(s) found:")
        for a in anomalies:
            print(f"  Timestamp: {a.get('timestamp', 'N/A')} | Score: {a['score']:.4f}")

        if args.output:
            pd.DataFrame(anomalies).to_csv(args.output, index=False)
            print(f"[predict] Anomalies saved → {args.output}")
