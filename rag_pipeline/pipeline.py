"""
pipeline.py
-----------
Master orchestrator for the Cloud Cost Intelligence RAG pipeline.

This is the ONLY file you need to call from the Isolation Forest module.
It wires together:
  1. Loading smoke_latest.json (last 5 timestamps)
  2. Stage 1 — Technical Verdict   (verdict1.py)
  3. Stage 2 — Business Judgment   (verdict2.py)

Usage (as a module — called by Isolation Forest):
    from rag_pipeline.pipeline import run_pipeline
    final = run_pipeline(anomaly_result)

    # final["decision"] → action to execute (e.g. "stop_instance" or "NO_ACTION")
    # final["reason"]   → explanation of the final judgment

Usage (standalone test):
    python pipeline.py
"""

import os
import json
import sys

# Allow running directly from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verdict1 import run_verdict1, DEFAULT_SMOKE_PATH
from verdict2 import run_verdict2
from actions import dispatch


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(
    anomaly_result: dict,
    smoke_data_path: str = DEFAULT_SMOKE_PATH,
    verbose: bool = True,
) -> dict:
    """
    Run the full two-stage RAG pipeline for a detected anomaly.

    Args:
        anomaly_result:   Dict of the anomalous row from Isolation Forest.
                          Must include: timestamp, cpu_utilization, network_in,
                          network_out, memory_usage, requests, error_rate,
                          storage_free, billing_rate, cost_per_hour,
                          anomaly (-1), score (float).

        smoke_data_path:  Path to smoke_latest.json.
                          Defaults to Data_Collector/Processed/smoke_latest.json.

        verbose:          If True, prints a formatted summary to stdout.

    Returns:
        Dict with the final verdict:
        {
            "stage1": { ...verdict1 fields... },
            "stage2": {
                "decision": "<action_function_name or NO_ACTION>",
                "reason":   "<explanation paragraph>"
            }
        }
    """
    if verbose:
        _banner("CLOUD COST INTELLIGENCE — RAG PIPELINE STARTED")
        print(f"  Anomaly timestamp : {anomaly_result.get('timestamp', 'N/A')}")
        print(f"  Anomaly score     : {anomaly_result.get('score', 'N/A')}")
        print(f"  smoke data path   : {smoke_data_path}\n")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if verbose:
        _banner("STAGE 1 — TECHNICAL VERDICT")

    v1 = run_verdict1(anomaly_result, smoke_data_path)

    # Separate injected timestamps from the verdict fields
    recent_timestamps = v1.get("_recent_timestamps", [])
    v1_display = {k: v for k, v in v1.items() if k != "_recent_timestamps"}

    if verbose:
        print("\n[Stage 1 Result]")
        print(json.dumps(v1_display, indent=2))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if verbose:
        _banner("STAGE 2 — BUSINESS-AWARE JUDGMENT")

    v2 = run_verdict2(v1, anomaly_result)

    if verbose:
        print("\n[Stage 2 Result]")
        print(json.dumps(v2, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────────
    if verbose:
        _banner("FINAL DECISION")
        decision = v2.get("decision", "NO_ACTION")
        reason   = v2.get("reason", "")
        risk     = v1_display.get("risk_level", "UNKNOWN")
        print(f"  Final Action : {decision}")
        print(f"  Risk Level   : {risk}")
        print(f"  Reason       : {reason}\n")

    # ── Execute action ────────────────────────────────────────────────────────
    execution_result = dispatch(v2, anomaly_result, v1_display)

    if verbose:
        print(f"\n[executor] Action dispatched: {execution_result['action_taken']}")
        if execution_result.get("dry_run"):
            print("[executor] Running in DRY_RUN mode — configure agent_config.json to enable real actions.")

    return {
        "stage1": v1_display,
        "stage2": v2,
        "execution": execution_result,
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _banner(title: str):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated anomaly row — replace with real Isolation Forest output
    sample_anomaly = {
        "timestamp": "2026-03-27T23:55:00+00:00",
        "cpu_utilization": 98.7,
        "network_in": 10.56,
        "network_out": 115.26,
        "memory_usage": 87.39,
        "requests": 501.0,
        "error_rate": 4.5043,
        "storage_free": 14.7,
        "billing_rate": 1.026,
        "cost_per_hour": 0.06125,
        "anomaly": -1,
        "score": -0.85,
    }

    result = run_pipeline(sample_anomaly)
    print("\n[pipeline] Full result dict:")
    print(json.dumps(result, indent=2))
