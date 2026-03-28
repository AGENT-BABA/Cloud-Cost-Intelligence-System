"""
actions package
---------------
Exports the main dispatch function for use by the pipeline.

Usage:
    from rag_pipeline.actions import dispatch
    result = dispatch(stage2_verdict, anomaly_row, stage1_verdict)
"""

from .executor import dispatch, ACTION_REGISTRY

__all__ = ["dispatch", "ACTION_REGISTRY"]
