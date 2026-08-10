"""order_queue read + write surface, plus the sweep trigger.

    GET  /queue                 -- list queue rows, optional ?status= filter
    GET  /queue/{sku_id}        -- one queue row
    POST /queue                 -- enqueue an existing SKU (ingestion_service)
    POST /queue/sweep           -- run the sweep now (sweep_service.run_sweep_and_explain)

No business logic lives here -- every route is a thin adapter over the existing
queue/repository/sweep service layer, same discipline as runs.py and explanations.py.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_llm_client
from src.api.schemas import EnqueueRequest, QueueEntryOut, SweepResult
from src.llm.client import OpenRouterClient
from src.queue import ingestion_service, sweep_service
from src.queue.models import OrderQueue

router = APIRouter()

_VALID_STATUSES = ("pending", "due_today", "expired")


@router.get("/queue", response_model=list[QueueEntryOut])
def list_queue(
    status: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> list[OrderQueue]:
    if status is not None and status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid status {status!r}, expected one of {_VALID_STATUSES}")
    stmt = select(OrderQueue).order_by(OrderQueue.due_date, OrderQueue.sku_id)
    if status is not None:
        stmt = stmt.where(OrderQueue.status == status)
    return list(session.execute(stmt).scalars().all())


@router.get("/queue/{sku_id}", response_model=QueueEntryOut)
def get_queue_entry(sku_id: str, session: Session = Depends(get_db)) -> OrderQueue:
    row = session.get(OrderQueue, sku_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"sku_id {sku_id!r} is not in order_queue")
    return row


@router.post("/queue", response_model=QueueEntryOut, status_code=201)
def enqueue_sku(body: EnqueueRequest, session: Session = Depends(get_db)) -> OrderQueue:
    try:
        ingestion_service.enqueue_new_sku(session, body.sku_id, due_date=body.due_date, source=body.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return session.get(OrderQueue, body.sku_id)


@router.post("/queue/sweep", response_model=SweepResult)
def trigger_sweep(
    as_of: date | None = None,
    explain: bool = True,
    session: Session = Depends(get_db),
    client: OpenRouterClient = Depends(get_llm_client),
) -> dict:
    from src.orchestrator.executor import LangGraphExecutor

    executor = LangGraphExecutor()
    effective_as_of = as_of or date.today()
    if explain:
        result = sweep_service.run_sweep_and_explain(session, effective_as_of, executor, client)
    else:
        result = sweep_service.run_sweep(session, effective_as_of, executor)
        result = {**result, "explanations": {}}
    return result
