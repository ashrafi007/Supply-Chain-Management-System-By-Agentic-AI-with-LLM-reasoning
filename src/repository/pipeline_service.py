"""
The single public entrypoint. Everything else in src.repository is private to
this module's use.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.pipeline_state import PipelineExecutor

from .run_repository import PLACEHOLDER_MANIFEST_VERSION, create_pending_run, persist_failure, persist_success
from .state_builder import build_initial_state


def run_pipeline_for_sku(session: Session, sku_id: str, executor: PipelineExecutor) -> str:
    """Returns run_id. Raises nothing — all outcomes are recorded in pipeline_runs."""
    state = build_initial_state(session, sku_id)
    if state is None:
        run_id = create_pending_run(session, sku_id, manifest_version=PLACEHOLDER_MANIFEST_VERSION)
        persist_failure(session, run_id, error="sku_not_found")
        return run_id

    run_id = create_pending_run(session, sku_id, manifest_version=PLACEHOLDER_MANIFEST_VERSION)

    try:
        result = executor.invoke(state)
    except Exception as exc:
        persist_failure(session, run_id, error=str(exc))
        return run_id

    try:
        persist_success(session, run_id, sku_id, result)
    except Exception as exc:
        persist_failure(session, run_id, error=f"constraint violation: {exc}")

    return run_id


def run_pipeline_for_skus(
    session: Session, sku_ids: list[str], executor: PipelineExecutor
) -> list[str]:
    return [run_pipeline_for_sku(session, sku_id, executor) for sku_id in sku_ids]
