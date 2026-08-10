"""
Batch entrypoint for order_queue. Not detailed in queue_migration_spec.md (it assumes
this already existed) — designed here from the "row lifecycle" rule: a sweep may flip
status (pending -> due_today, or auto-flag expired once past due_date), but the row
itself stays until a human removes it via deletion_service.remove_from_queue.

run_sweep contains no session.delete(...) and never imports deletion_service — that
is what makes "sweeps never delete" true by construction, not just by convention.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from src.db.models import PipelineRun
from src.llm import explainer_service
from src.llm.client import OpenRouterClient
from src.pipeline_state import PipelineExecutor
from src.repository.pipeline_service import run_pipeline_for_skus

from . import queue_repository


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_sweep(session: Session, as_of: date, executor: PipelineExecutor) -> dict:
    # Must run before get_due_skus: a row with due_date < as_of must become 'expired',
    # not get caught by get_due_skus's due_date <= as_of and become 'due_today'.
    expired_count = queue_repository.mark_expired(session, as_of)

    due_sku_ids = queue_repository.get_due_skus(session, as_of)
    run_ids = run_pipeline_for_skus(session, due_sku_ids, executor) if due_sku_ids else []

    # mark_evaluated runs even for a failed run: run_pipeline_for_sku never raises, and
    # "evaluated" means "an attempt was recorded", not "succeeded".
    evaluated_at = _utcnow()
    for sku_id, run_id in zip(due_sku_ids, run_ids):
        queue_repository.mark_evaluated(session, sku_id, run_id, evaluated_at)

    return {
        "as_of": as_of,
        "expired_count": expired_count,
        "evaluated_sku_ids": list(due_sku_ids),
        "run_ids": list(run_ids),
    }


def run_sweep_and_explain(
    session: Session, as_of: date, executor: PipelineExecutor, llm_client: OpenRouterClient
) -> dict:
    """run_sweep(), then a whole-run LLM explanation for every SKU that was actually
    evaluated AND succeeded (llm_insertion_spec.md SS9 raises RunNotFoundError for a
    failed run -- no predictions row exists to explain, so those are skipped, not
    retried or errored).

    Kept as a separate function rather than folding this into run_sweep() itself:
    llm_insertion_spec.md SS2 documents LLM generation as on-demand, deliberately never
    part of a sweep, specifically so a bare sweep never depends on network/API
    availability. This is the opt-in entrypoint for a caller (a scheduler, a CLI) that
    *does* want "every SKU gets explained automatically" -- run_sweep() itself, and its
    "sweep never touches the network" test guarantee, are unchanged.
    """
    result = run_sweep(session, as_of, executor)

    explanations: dict[str, dict] = {}
    for run_id in result["run_ids"]:
        run = session.get(PipelineRun, run_id)
        if run is None or run.status != "success":
            continue
        explanations[run_id] = explainer_service.explain(session, llm_client, run_id, agent_name=None)

    result["explanations"] = explanations
    return result
