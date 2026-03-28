"""
alerts.py
---------
Implements send_alert() — notifies the team when an anomaly is detected
and automatic action is not safe to execute.

Alert channel is controlled by agent_config.json:
  "channel": "log"    → writes to alerts.log only (default / testing)
  "channel": "sns"    → publishes to an AWS SNS topic
  "channel": "email"  → sends an email via SMTP (e.g. Gmail)

The web UI will populate the config; no changes needed in this file.
"""

import json
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .config import get_alert_config, get_boto3_session, is_dry_run

# ── Alert log file (always written regardless of channel) ─────────────────────
_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # actions/
    "..",                                         # rag_pipeline/
    "..",                                         # project root
    "alerts.log",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Public function ───────────────────────────────────────────────────────────

def send_alert(message: str, severity: str = "warning", resource_id: str = "unknown") -> dict:
    """
    Send a notification alert for the detected anomaly.

    Args:
        message:     Human-readable description of the anomaly and recommended action.
        severity:    One of: "info", "warning", "critical"
        resource_id: AWS resource ID or name that triggered the anomaly.

    Returns:
        Dict with keys: success (bool), channel (str), detail (str)
    """
    alert_cfg = get_alert_config()
    channel = alert_cfg.get("channel", "log").lower()
    dry = is_dry_run()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "resource_id": resource_id,
        "message": message,
    }

    # Always write to the local log file
    _write_log(payload)

    if dry:
        print(f"[send_alert] DRY_RUN — alert NOT sent externally. Logged to alerts.log.")
        print(f"  Severity    : {severity}")
        print(f"  Resource    : {resource_id}")
        print(f"  Message     : {message}")
        return {"success": True, "channel": "log (dry_run)", "detail": "Logged only — dry_run=True"}

    if channel == "sns":
        return _send_via_sns(payload, alert_cfg)
    elif channel == "email":
        return _send_via_email(payload, alert_cfg)
    else:
        # Default: log only
        print(f"[send_alert] Alert logged to alerts.log (channel=log).")
        return {"success": True, "channel": "log", "detail": "Written to alerts.log"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _write_log(payload: dict):
    """Always appends the alert payload to alerts.log as JSON."""
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        logger.warning(f"[send_alert] Could not write to alerts.log: {e}")


def _send_via_sns(payload: dict, cfg: dict) -> dict:
    """Publish alert to AWS SNS topic."""
    topic_arn = cfg.get("sns_topic_arn", "").strip()
    if not topic_arn:
        return {"success": False, "channel": "sns", "detail": "sns_topic_arn not set in agent_config.json"}

    try:
        session = get_boto3_session()
        sns = session.client("sns")
        subject = f"[{payload['severity'].upper()}] Cloud Cost Alert — {payload['resource_id']}"
        body = (
            f"Timestamp  : {payload['timestamp']}\n"
            f"Severity   : {payload['severity']}\n"
            f"Resource   : {payload['resource_id']}\n"
            f"Message    : {payload['message']}\n"
        )
        response = sns.publish(TopicArn=topic_arn, Subject=subject, Message=body)
        msg_id = response.get("MessageId", "unknown")
        print(f"[send_alert] SNS message published. MessageId: {msg_id}")
        return {"success": True, "channel": "sns", "detail": f"MessageId={msg_id}"}
    except Exception as e:
        logger.error(f"[send_alert] SNS publish failed: {e}")
        return {"success": False, "channel": "sns", "detail": str(e)}


def _send_via_email(payload: dict, cfg: dict) -> dict:
    """Send alert email via SMTP."""
    required = ["smtp_host", "smtp_user", "smtp_password", "email_to", "email_from"]
    missing = [k for k in required if not cfg.get(k, "").strip()]
    if missing:
        return {"success": False, "channel": "email", "detail": f"Missing config keys: {missing}"}

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{payload['severity'].upper()}] Cloud Cost Alert — {payload['resource_id']}"
        msg["From"] = cfg["email_from"]
        msg["To"] = cfg["email_to"]

        body = (
            f"<h3>Cloud Cost Intelligence Alert</h3>"
            f"<p><b>Timestamp:</b> {payload['timestamp']}<br>"
            f"<b>Severity:</b> {payload['severity']}<br>"
            f"<b>Resource:</b> {payload['resource_id']}<br>"
            f"<b>Message:</b> {payload['message']}</p>"
        )
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port", 587))) as server:
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(cfg["email_from"], cfg["email_to"], msg.as_string())

        print(f"[send_alert] Email sent to {cfg['email_to']}")
        return {"success": True, "channel": "email", "detail": f"Sent to {cfg['email_to']}"}
    except Exception as e:
        logger.error(f"[send_alert] Email send failed: {e}")
        return {"success": False, "channel": "email", "detail": str(e)}
