"""
config.py
---------
Loads agent_config.json (written by the web UI during user onboarding).
All action functions import from here — they never hardcode credentials.

If the config file is missing or credentials are empty, the system
automatically operates in DRY_RUN mode (safe — logs actions, no AWS calls).
"""

import json
import os

# The config file lives at the project root, two levels above this file:
# rag_pipeline/actions/config.py → ../../agent_config.json
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # actions/
    "..",                                         # rag_pipeline/
    "..",                                         # Cloud Cost Intelligence System/
    "agent_config.json",
)


def load_config() -> dict:
    """
    Load and return the full agent_config.json contents.
    Returns a safe default (dry_run=True) if the file is missing.
    """
    if not os.path.exists(_CONFIG_PATH):
        print(f"[config] WARNING: agent_config.json not found at {_CONFIG_PATH}. Running in DRY_RUN mode.")
        return _default_config()

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


def get_boto3_session():
    """
    Build and return a boto3 Session using credentials from agent_config.json.
    Falls back to boto3's default credential chain (env vars, ~/.aws/credentials,
    IAM role) if access_key_id is left empty in the config.
    """
    import boto3

    cfg = load_config()
    aws = cfg.get("aws", {})

    access_key = aws.get("access_key_id", "").strip()
    secret_key = aws.get("secret_access_key", "").strip()
    region = aws.get("region", "us-east-1").strip()

    if access_key and secret_key:
        # Explicit credentials provided by web UI
        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    else:
        # Fall back to default boto3 credential chain
        print("[config] No explicit credentials in agent_config.json. Using default boto3 credential chain.")
        return boto3.Session(region_name=region)


def is_dry_run() -> bool:
    """Returns True if dry_run mode is active (no real AWS calls will be made)."""
    cfg = load_config()
    return cfg.get("dry_run", True)


def get_alert_config() -> dict:
    """Returns the alerts section of the config."""
    cfg = load_config()
    return cfg.get("alerts", {"channel": "log"})


def _default_config() -> dict:
    return {
        "aws": {"access_key_id": "", "secret_access_key": "", "region": "us-east-1"},
        "alerts": {"channel": "log"},
        "dry_run": True,
    }
