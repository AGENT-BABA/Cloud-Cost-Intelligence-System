"""
executor.py
-----------
The bridge between the RAG pipeline verdict and the actual AWS action functions.

Called by pipeline.py after Stage 2 returns its final decision:
    from rag_pipeline.actions.executor import dispatch
    result = dispatch(stage2_verdict, anomaly_row)

The dispatcher:
  1. Reads the "decision" field from the Stage 2 verdict
  2. Extracts parameters from Stage 1's "parameters" field (resource IDs, etc.)
  3. Calls the appropriate action function
  4. Returns a result dict with success status and detail

All actions default to DRY_RUN mode until agent_config.json is configured by the web UI.
"""

import json
import logging

from .alerts import send_alert
from .ec2 import stop_instance
from .ebs import delete_volume, snapshot_then_delete, tag_resource
from .lambda_fn import set_lambda_concurrency
from .config import is_dry_run

logger = logging.getLogger(__name__)

# Maps action names (from actions.txt) to their function implementations
ACTION_REGISTRY = {
    "send_alert": send_alert,
    "stop_instance": stop_instance,
    "set_lambda_concurrency": set_lambda_concurrency,
    "delete_volume": delete_volume,
    "tag_resource": tag_resource,
    "snapshot_then_delete": snapshot_then_delete,
}


def dispatch(stage2_verdict: dict, anomaly_row: dict, stage1_verdict: dict = None) -> dict:
    """
    Dispatch the pipeline verdict to the appropriate AWS action function.

    Args:
        stage2_verdict:  The Stage 2 output dict. Must have "decision" and "reason".
        anomaly_row:     The original anomaly dict from Isolation Forest (for resource IDs, etc.)
        stage1_verdict:  The Stage 1 output dict (optional, used to extract parameters).

    Returns:
        Dict with keys:
          action_taken (str), success (bool), result (dict), dry_run (bool)
    """
    decision = stage2_verdict.get("decision", "NO_ACTION").strip()
    reason = stage2_verdict.get("reason", "")
    dry = is_dry_run()

    print(f"\n[executor] Decision received: {decision}")
    print(f"[executor] Dry-run mode     : {dry}")

    # ── NO_ACTION: log and return ─────────────────────────────────────────────
    if decision == "NO_ACTION":
        print(f"[executor] NO_ACTION — no AWS call will be made.")
        print(f"[executor] Reason: {reason}")
        return {
            "action_taken": "NO_ACTION",
            "success": True,
            "result": {"detail": reason},
            "dry_run": dry,
        }

    # ── Unknown action: fall back to alert ───────────────────────────────────
    if decision not in ACTION_REGISTRY:
        logger.warning(f"[executor] Unknown action '{decision}' — falling back to send_alert.")
        result = send_alert(
            message=f"Unknown action '{decision}' returned by pipeline. Manual review required. Reason: {reason}",
            severity="warning",
            resource_id=anomaly_row.get("timestamp", "unknown"),
        )
        return {"action_taken": "send_alert (fallback)", "success": False, "result": result, "dry_run": dry}

    # ── Build kwargs from pipeline parameters ─────────────────────────────────
    params = (stage1_verdict or {}).get("parameters", {})
    fn = ACTION_REGISTRY[decision]
    kwargs = _build_kwargs(decision, params, anomaly_row, reason)

    print(f"[executor] Calling {decision}({', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())})")

    try:
        result = fn(**kwargs)
        print(f"[executor] Result: {json.dumps(result, indent=2)}")
        return {
            "action_taken": decision,
            "success": result.get("success", False),
            "result": result,
            "dry_run": dry,
        }
    except Exception as e:
        logger.error(f"[executor] Action {decision} raised an exception: {e}")
        return {
            "action_taken": decision,
            "success": False,
            "result": {"detail": str(e)},
            "dry_run": dry,
        }


def _build_kwargs(action: str, params: dict, anomaly: dict, reason: str) -> dict:
    """
    Map pipeline parameters to the correct function argument names.
    Falls back to safe defaults if specific resource IDs are missing.
    """
    resource_id = (
        params.get("resource_id")
        or params.get("instance_id")
        or params.get("function_name")
        or params.get("volume_id")
        or anomaly.get("resource_id", "unknown")
    )

    if action == "send_alert":
        return {
            "message": params.get("message", reason),
            "severity": params.get("severity", params.get("urgency", "warning")),
            "resource_id": resource_id,
        }

    elif action == "stop_instance":
        return {"instance_id": params.get("instance_id", resource_id)}

    elif action == "set_lambda_concurrency":
        return {
            "function_name": params.get("function_name", resource_id),
            "limit": int(params.get("limit", 0)),
        }

    elif action == "delete_volume":
        return {"volume_id": params.get("volume_id", resource_id)}

    elif action == "tag_resource":
        default_tags = {"review-required": "true", "flagged-by": "cloud-cost-agent"}
        return {
            "resource_id": resource_id,
            "tags": params.get("tags", default_tags),
        }

    elif action == "snapshot_then_delete":
        return {"volume_id": params.get("volume_id", resource_id)}

    return {}
