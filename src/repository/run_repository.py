"""
Persistence for pipeline_runs, predictions, agent_traces. Single responsibility:
writes only — no decisions about what the pipeline should do, only recording
what it did.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.db.models import AgentTrace, PipelineRun, Prediction
from src.pipeline_state import PREDICTION_FIELDS, PipelineResult

PLACEHOLDER_MANIFEST_VERSION = "unknown"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """SQLite round-trips DateTime columns as naive datetimes; _utcnow() writes
    tz-aware UTC values, so treat a naive read-back as UTC rather than local time."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def create_pending_run(session: Session, sku_id: str, manifest_version: str) -> str:
    run_id = str(uuid.uuid4())
    session.add(
        PipelineRun(
            run_id=run_id,
            sku_id=sku_id,
            started_at=_utcnow(),
            status="pending",
            manifest_version=manifest_version,
        )
    )
    session.commit()
    return run_id


def persist_success(
    session: Session, run_id: str, sku_id: str, result: PipelineResult
) -> None:
    final_state = result["final_state"]

    try:
        prediction_kwargs = {field: final_state[field] for field in PREDICTION_FIELDS}
        session.add(Prediction(run_id=run_id, sku_id=sku_id, **prediction_kwargs))

        for event in result["agent_events"]:
            session.add(
                AgentTrace(
                    run_id=run_id,
                    agent_name=event["agent_name"],
                    sequence=event["sequence"],
                    status=event["status"],
                    latency_ms=event["latency_ms"],
                    output=event["output"],
                    note=event["note"],
                )
            )

        run = session.get(PipelineRun, run_id)
        completed_at = _utcnow()
        started_at = _as_utc(run.started_at)
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        run.status = "success"
        run.completed_at = completed_at
        run.latency_ms = latency_ms
        run.manifest_version = result["manifest_version"]

        session.flush()  # CHECK constraints fire here; IntegrityError propagates uncaught
        session.commit()
    except Exception:
        session.rollback()
        raise


def persist_failure(session: Session, run_id: str, error: str) -> None:
    run = session.get(PipelineRun, run_id)
    run.status = "failed"
    run.completed_at = _utcnow()
    run.error = error
    session.commit()
