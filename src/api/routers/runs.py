"""Read-only endpoints over pipeline_runs/predictions/agent_traces -- the
frontend's window into pipeline history. No endpoint here triggers a run or
mutates anything; that's a separate concern for later.

    GET /runs               -- list recent runs, optionally filtered by sku_id
    GET /runs/{run_id}      -- full detail: run + prediction + agent traces
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas import AgentTraceOut, PredictionOut, RunDetail, RunSummary
from src.db.models import AgentTrace, PipelineRun, Prediction

router = APIRouter()


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    sku_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> list[PipelineRun]:
    stmt = select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
    if sku_id is not None:
        stmt = stmt.where(PipelineRun.sku_id == sku_id)
    return list(session.execute(stmt).scalars().all())


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run_detail(run_id: str, session: Session = Depends(get_db)) -> RunDetail:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")

    prediction = session.get(Prediction, run_id)

    traces = session.execute(
        select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.sequence)
    ).scalars().all()

    return RunDetail(
        run_id=run.run_id,
        sku_id=run.sku_id,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        latency_ms=run.latency_ms,
        manifest_version=run.manifest_version,
        error=run.error,
        prediction=PredictionOut.model_validate(prediction) if prediction is not None else None,
        traces=[AgentTraceOut.model_validate(t) for t in traces],
    )
