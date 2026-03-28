"""
ec2.py
------
EC2 action: stop_instance()

Stops an EC2 instance (NOT terminate — data is preserved, can be restarted).
Only used when: CPU < 3% sustained for 4+ hours AND no active connections.

Safety:
  - DRY_RUN mode: logs the action but never calls AWS
  - Checks instance state before stopping (won't stop an already-stopped instance)
  - Will NOT stop instances tagged critical=true or env=production
"""

import logging

from .config import get_boto3_session, is_dry_run

logger = logging.getLogger(__name__)


def stop_instance(instance_id: str) -> dict:
    """
    Stop an EC2 instance.

    Args:
        instance_id: The EC2 instance ID (e.g. "i-0a1b2c3d4e5f67890")

    Returns:
        Dict with keys: success (bool), action (str), instance_id (str),
                        previous_state (str), current_state (str), detail (str)
    """
    dry = is_dry_run()

    if dry:
        print(f"[stop_instance] DRY_RUN — would stop EC2 instance: {instance_id}")
        return {
            "success": True,
            "action": "stop_instance",
            "instance_id": instance_id,
            "previous_state": "running",
            "current_state": "stopping (simulated)",
            "detail": "DRY_RUN — no real AWS call made",
        }

    try:
        session = get_boto3_session()
        ec2 = session.client("ec2")

        # --- Safety check: verify instance exists and is running ---
        desc = ec2.describe_instances(InstanceIds=[instance_id])
        reservations = desc.get("Reservations", [])
        if not reservations:
            return _error(instance_id, f"Instance {instance_id} not found.")

        instance = reservations[0]["Instances"][0]
        state = instance["State"]["Name"]
        tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}

        # --- Safety check: never touch production / critical resources ---
        if tags.get("env", "").lower() == "production":
            return _error(instance_id, "Refused: instance is tagged env=production. Manual action required.")
        if tags.get("critical", "").lower() == "true":
            return _error(instance_id, "Refused: instance is tagged critical=true. Manual action required.")

        if state != "running":
            return {
                "success": False,
                "action": "stop_instance",
                "instance_id": instance_id,
                "previous_state": state,
                "current_state": state,
                "detail": f"Instance is already in state '{state}' — stop skipped.",
            }

        # --- Execute stop ---
        response = ec2.stop_instances(InstanceIds=[instance_id])
        new_state = response["StoppingInstances"][0]["CurrentState"]["Name"]
        prev_state = response["StoppingInstances"][0]["PreviousState"]["Name"]

        print(f"[stop_instance] Instance {instance_id} is now {new_state} (was {prev_state})")
        return {
            "success": True,
            "action": "stop_instance",
            "instance_id": instance_id,
            "previous_state": prev_state,
            "current_state": new_state,
            "detail": "Stop command sent successfully.",
        }

    except Exception as e:
        logger.error(f"[stop_instance] Error: {e}")
        return _error(instance_id, str(e))


def _error(instance_id: str, detail: str) -> dict:
    return {
        "success": False,
        "action": "stop_instance",
        "instance_id": instance_id,
        "previous_state": "unknown",
        "current_state": "unknown",
        "detail": detail,
    }
