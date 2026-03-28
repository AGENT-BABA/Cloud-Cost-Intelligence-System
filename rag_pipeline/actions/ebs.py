"""
ebs.py
------
EBS volume actions:
  - tag_resource()         — tags any AWS resource (safe, no operational impact)
  - delete_volume()        — permanently deletes an unattached volume (irreversible)
  - snapshot_then_delete() — takes snapshot first, then deletes (safer than delete_volume)

Safety:
  - DRY_RUN mode for all destructive operations
  - delete_volume() refuses to act on attached volumes
  - snapshot_then_delete() waits for snapshot completion before deleting
"""

import logging
import time

from .config import get_boto3_session, is_dry_run

logger = logging.getLogger(__name__)


# ── tag_resource ──────────────────────────────────────────────────────────────

def tag_resource(resource_id: str, tags: dict) -> dict:
    """
    Add tags to any AWS resource.

    Args:
        resource_id: Any AWS resource ID (EC2, EBS, Lambda, RDS, etc.)
        tags:        Dict of tag key-value pairs, e.g. {"review-required": "true"}

    Returns:
        Dict with keys: success (bool), action (str), resource_id (str),
                        tags_applied (dict), detail (str)
    """
    dry = is_dry_run()

    if dry:
        print(f"[tag_resource] DRY_RUN — would apply tags {tags} to resource: {resource_id}")
        return {
            "success": True,
            "action": "tag_resource",
            "resource_id": resource_id,
            "tags_applied": tags,
            "detail": "DRY_RUN — no real AWS call made",
        }

    try:
        session = get_boto3_session()
        ec2 = session.client("ec2")
        boto_tags = [{"Key": k, "Value": v} for k, v in tags.items()]
        ec2.create_tags(Resources=[resource_id], Tags=boto_tags)
        print(f"[tag_resource] Applied tags {tags} to {resource_id}")
        return {
            "success": True,
            "action": "tag_resource",
            "resource_id": resource_id,
            "tags_applied": tags,
            "detail": "Tags applied successfully.",
        }
    except Exception as e:
        logger.error(f"[tag_resource] Error: {e}")
        return {
            "success": False,
            "action": "tag_resource",
            "resource_id": resource_id,
            "tags_applied": {},
            "detail": str(e),
        }


# ── delete_volume ─────────────────────────────────────────────────────────────

def delete_volume(volume_id: str) -> dict:
    """
    Permanently delete an unattached EBS volume.
    IRREVERSIBLE — only use when volume state is 'available' (not attached).

    Args:
        volume_id: The EBS volume ID (e.g. "vol-0abc123def456789a")

    Returns:
        Dict with keys: success (bool), action (str), volume_id (str),
                        volume_state (str), detail (str)
    """
    dry = is_dry_run()

    if dry:
        print(f"[delete_volume] DRY_RUN — would permanently delete EBS volume: {volume_id}")
        return {
            "success": True,
            "action": "delete_volume",
            "volume_id": volume_id,
            "volume_state": "available (simulated)",
            "detail": "DRY_RUN — no real AWS call made",
        }

    try:
        session = get_boto3_session()
        ec2 = session.client("ec2")

        # --- Safety check: must be unattached ---
        desc = ec2.describe_volumes(VolumeIds=[volume_id])
        volume = desc["Volumes"][0]
        state = volume["State"]

        if state != "available":
            return {
                "success": False,
                "action": "delete_volume",
                "volume_id": volume_id,
                "volume_state": state,
                "detail": f"Refused: volume state is '{state}', not 'available'. Cannot safely delete.",
            }

        # Check it has no attachments
        attachments = volume.get("Attachments", [])
        if attachments:
            attached_to = attachments[0].get("InstanceId", "unknown")
            return {
                "success": False,
                "action": "delete_volume",
                "volume_id": volume_id,
                "volume_state": state,
                "detail": f"Refused: volume is still attached to instance {attached_to}.",
            }

        ec2.delete_volume(VolumeId=volume_id)
        print(f"[delete_volume] Volume {volume_id} deleted permanently.")
        return {
            "success": True,
            "action": "delete_volume",
            "volume_id": volume_id,
            "volume_state": "deleted",
            "detail": "Volume deleted permanently.",
        }

    except Exception as e:
        logger.error(f"[delete_volume] Error: {e}")
        return {
            "success": False,
            "action": "delete_volume",
            "volume_id": volume_id,
            "volume_state": "unknown",
            "detail": str(e),
        }


# ── snapshot_then_delete ──────────────────────────────────────────────────────

def snapshot_then_delete(volume_id: str) -> dict:
    """
    Take a snapshot of an EBS volume and then delete it.
    Safer than delete_volume — data is preserved in the snapshot.

    Args:
        volume_id: The EBS volume ID (e.g. "vol-0abc123def456789a")

    Returns:
        Dict with keys: success (bool), action (str), volume_id (str),
                        snapshot_id (str), detail (str)
    """
    dry = is_dry_run()

    if dry:
        print(f"[snapshot_then_delete] DRY_RUN — would snapshot then delete EBS volume: {volume_id}")
        return {
            "success": True,
            "action": "snapshot_then_delete",
            "volume_id": volume_id,
            "snapshot_id": "snap-DRYRUN",
            "detail": "DRY_RUN — no real AWS call made",
        }

    try:
        session = get_boto3_session()
        ec2 = session.client("ec2")

        # --- Step 1: Create snapshot ---
        print(f"[snapshot_then_delete] Creating snapshot of volume {volume_id}...")
        snap_response = ec2.create_snapshot(
            VolumeId=volume_id,
            Description=f"Auto-snapshot before deletion by Cloud Cost Intelligence Agent",
            TagSpecifications=[{
                "ResourceType": "snapshot",
                "Tags": [
                    {"Key": "created-by", "Value": "cloud-cost-agent"},
                    {"Key": "source-volume", "Value": volume_id},
                ]
            }]
        )
        snapshot_id = snap_response["SnapshotId"]
        print(f"[snapshot_then_delete] Snapshot {snapshot_id} created. Waiting for completion...")

        # --- Step 2: Wait for snapshot to complete (max 10 minutes) ---
        waiter = ec2.get_waiter("snapshot_completed")
        waiter.wait(
            SnapshotIds=[snapshot_id],
            WaiterConfig={"Delay": 15, "MaxAttempts": 40},
        )
        print(f"[snapshot_then_delete] Snapshot {snapshot_id} completed.")

        # --- Step 3: Delete volume ---
        ec2.delete_volume(VolumeId=volume_id)
        print(f"[snapshot_then_delete] Volume {volume_id} deleted. Snapshot preserved as {snapshot_id}.")
        return {
            "success": True,
            "action": "snapshot_then_delete",
            "volume_id": volume_id,
            "snapshot_id": snapshot_id,
            "detail": f"Snapshot {snapshot_id} created and volume deleted successfully.",
        }

    except Exception as e:
        logger.error(f"[snapshot_then_delete] Error: {e}")
        return {
            "success": False,
            "action": "snapshot_then_delete",
            "volume_id": volume_id,
            "snapshot_id": "unknown",
            "detail": str(e),
        }
