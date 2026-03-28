"""
api.py
------
FastAPI wrapper around the Cloud Cost Intelligence ML pipeline.

Exposes one main endpoint:
  POST /analyze
    Body: { "metrics": { ...all AWS metric fields... } }
    Returns: { "stage1": {...}, "stage2": {...}, "execution": {...} }

Run locally:
  uvicorn api:app --reload --port 8000

On Render:
  Set start command to: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import sys

# Make sure sibling packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from rag_pipeline.pipeline import run_pipeline, DEFAULT_SMOKE_PATH
from Isolation_Forest.predict import detect_anomalies


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cloud Cost Intelligence API",
    description="Isolation Forest + RAG pipeline for AWS anomaly detection and remediation.",
    version="1.0.0",
)

# Allow your Render frontend / backend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class MetricsPayload(BaseModel):
    """One row of AWS metric data."""
    timestamp:       str
    cpu_utilization: float
    network_in:      float
    network_out:     float
    memory_usage:    float
    requests:        float
    error_rate:      float
    storage_free:    float
    billing_rate:    float
    cost_per_hour:   float


class AnalyzeRequest(BaseModel):
    """Request body for /analyze."""
    metrics: MetricsPayload
    smoke_data_path: str | None = Field(
        default=None,
        description="Optional override path to smoke_latest.json. Defaults to the bundled path.",
    )


class AnalyzeResponse(BaseModel):
    """Response from /analyze."""
    anomaly_detected: bool
    anomaly_score:    float
    stage1:           dict
    stage2:           dict
    execution:        dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Cloud Cost Intelligence API",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    """
    Run the full pipeline on a single metric snapshot.

    1. Runs Isolation Forest to confirm it is anomalous.
    2. Passes it through the two-stage RAG pipeline.
    3. Dispatches the resulting action (dry_run by default).
    """
    row = body.metrics.model_dump()

    # ── Step 1: Confirm anomaly via Isolation Forest ──────────────────────────
    df = pd.DataFrame([row])
    try:
        anomalies = detect_anomalies(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not anomalies:
        return AnalyzeResponse(
            anomaly_detected=False,
            anomaly_score=0.0,
            stage1={},
            stage2={"decision": "NO_ACTION", "reason": "Isolation Forest did not flag this row as anomalous."},
            execution={"action_taken": "NO_ACTION", "success": True, "result": {}, "dry_run": True},
        )

    anomaly_row = anomalies[0]   # Take the first (and likely only) flagged row

    # ── Step 2: Run RAG pipeline ──────────────────────────────────────────────
    smoke_path = body.smoke_data_path or DEFAULT_SMOKE_PATH
    try:
        result = run_pipeline(anomaly_row, smoke_data_path=smoke_path, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return AnalyzeResponse(
        anomaly_detected=True,
        anomaly_score=float(anomaly_row.get("score", 0.0)),
        stage1=result["stage1"],
        stage2=result["stage2"],
        execution=result["execution"],
    )


@app.post("/analyze/batch")
def analyze_batch(rows: list[MetricsPayload]):
    """
    Run the pipeline on multiple metric rows at once.
    Returns a list of results, one per anomaly detected.
    """
    df = pd.DataFrame([r.model_dump() for r in rows])
    try:
        anomalies = detect_anomalies(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for anomaly_row in anomalies:
        try:
            result = run_pipeline(anomaly_row, verbose=False)
            results.append({
                "timestamp":       anomaly_row.get("timestamp"),
                "anomaly_score":   anomaly_row.get("score"),
                "stage1":          result["stage1"],
                "stage2":          result["stage2"],
                "execution":       result["execution"],
            })
        except Exception as e:
            results.append({
                "timestamp": anomaly_row.get("timestamp"),
                "error":     str(e),
            })

    return {"anomalies_found": len(anomalies), "results": results}
