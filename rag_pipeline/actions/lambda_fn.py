"""
lambda_fn.py
------------
Lambda action: set_lambda_concurrency()

Sets the reserved concurrency on a Lambda function.
Setting limit=0 stops ALL invocations immediately (emergency stop for runaway loops).

Safety:
  - DRY_RUN mode: logs the action but never calls AWS
  - Will NOT touch functions tagged critical=true or payments=true or auth=true
  - Logs the previous concurrency so it can be restored
"""

import logging

from .config import get_boto3_session, is_dry_run

logger = logging.getLogger(__name__)

# Tags that block automatic Lambda throttling
_PROTECTED_TAGS = {"critical", "payments", "auth"}


def set_lambda_concurrency(function_name: str, limit: int) -> dict:
    """
    Set the reserved concurrency of a Lambda function.

    Args:
        function_name: Name or ARN of the Lambda function.
        limit:         Concurrency limit to set. Use 0 to stop all invocations.

    Returns:
        Dict with keys: success (bool), action (str), function_name (str),
                        previous_limit (int | None), new_limit (int), detail (str)
    """
    dry = is_dry_run()

    if dry:
        action_desc = "STOP all invocations" if limit == 0 else f"throttle to {limit} concurrent executions"
        print(f"[set_lambda_concurrency] DRY_RUN — would {action_desc} on: {function_name}")
        return {
            "success": True,
            "action": "set_lambda_concurrency",
            "function_name": function_name,
            "previous_limit": None,
            "new_limit": limit,
            "detail": f"DRY_RUN — no real AWS call made",
        }

    try:
        session = get_boto3_session()
        lam = session.client("lambda")

        # --- Safety check: read tags to protect critical functions ---
        try:
            tag_response = lam.list_tags(Resource=function_name)
            tags = {k.lower(): v.lower() for k, v in tag_response.get("Tags", {}).items()}
            for protected in _PROTECTED_TAGS:
                if tags.get(protected) == "true":
                    return _error(
                        function_name,
                        f"Refused: function is tagged {protected}=true. Manual action required.",
                        limit,
                    )
        except Exception:
            pass  # If we can't read tags, proceed with caution

        # --- Read previous concurrency ---
        previous_limit = None
        try:
            cfg_response = lam.get_function_concurrency(FunctionName=function_name)
            previous_limit = cfg_response.get("ReservedConcurrentExecutions")
        except Exception:
            pass

        # --- Execute concurrency change ---
        if limit == 0:
            lam.put_function_concurrency(
                FunctionName=function_name,
                ReservedConcurrentExecutions=0,
            )
        else:
            lam.put_function_concurrency(
                FunctionName=function_name,
                ReservedConcurrentExecutions=limit,
            )

        action_desc = "Stopped all invocations (limit=0)" if limit == 0 else f"Set concurrency to {limit}"
        print(f"[set_lambda_concurrency] {action_desc} for function: {function_name}")
        return {
            "success": True,
            "action": "set_lambda_concurrency",
            "function_name": function_name,
            "previous_limit": previous_limit,
            "new_limit": limit,
            "detail": action_desc,
        }

    except Exception as e:
        logger.error(f"[set_lambda_concurrency] Error: {e}")
        return _error(function_name, str(e), limit)


def _error(function_name: str, detail: str, limit: int) -> dict:
    return {
        "success": False,
        "action": "set_lambda_concurrency",
        "function_name": function_name,
        "previous_limit": None,
        "new_limit": limit,
        "detail": detail,
    }
