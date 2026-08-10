"""SKU onboarding + read surface -- the HTTP equivalent of scripts/add_new_sku.py.

    GET  /skus            -- list
    GET  /skus/{sku_id}   -- detail (skus row + its raw inventory_current features)
    POST /skus            -- add a new SKU (skus + inventory_current), enqueue it into
                              order_queue, optionally run the pipeline immediately

No business logic lives here -- delegates to sku_ingestion.add_new_sku and
ingestion_service.enqueue_new_sku, the same functions the CLI script uses, so the two
entrypoints can never drift in behavior.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas import NewSkuRequest, NewSkuResult, SkuDetail, SkuOut
from src.db.models import InventoryCurrent, Sku
from src.queue import ingestion_service
from src.queue.models import OrderQueue
from src.repository.sku_ingestion import RAW_FEATURE_COLUMNS, add_new_sku

router = APIRouter()


@router.get("/skus", response_model=list[SkuOut])
def list_skus(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_db),
) -> list[Sku]:
    stmt = select(Sku).order_by(Sku.sku_id).limit(limit)
    return list(session.execute(stmt).scalars().all())


@router.get("/skus/{sku_id}", response_model=SkuDetail)
def get_sku(sku_id: str, session: Session = Depends(get_db)) -> dict:
    sku = session.get(Sku, sku_id)
    if sku is None:
        raise HTTPException(status_code=404, detail=f"sku_id {sku_id!r} not found")

    inv = session.get(InventoryCurrent, sku_id)
    raw_features = {col: getattr(inv, col) for col in RAW_FEATURE_COLUMNS} if inv is not None else {}

    return {
        "sku_id": sku.sku_id,
        "supplier_id": sku.supplier_id,
        "description": sku.description,
        "created_at": sku.created_at,
        "raw_features": raw_features,
    }


@router.post("/skus", response_model=NewSkuResult, status_code=201)
def create_sku(
    body: NewSkuRequest,
    session: Session = Depends(get_db),
) -> dict:
    try:
        add_new_sku(
            session, body.sku_id, body.raw_features,
            supplier_id=body.supplier_id, description=body.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    due_date = date.today() + timedelta(days=body.due_in_days)
    try:
        ingestion_service.enqueue_new_sku(session, body.sku_id, due_date=due_date, source="manual_add")
    except ValueError as exc:
        # SKU row was already created above (matches scripts/add_new_sku.py's own
        # partial-success behavior on this same failure path) -- surface it clearly
        # rather than silently losing the queue failure.
        raise HTTPException(status_code=400, detail=f"sku created but enqueue failed: {exc}")

    run_id = None
    if body.run_now:
        from src.orchestrator.executor import LangGraphExecutor
        from src.repository.pipeline_service import run_pipeline_for_sku

        run_id = run_pipeline_for_sku(session, body.sku_id, LangGraphExecutor())

    queued = session.get(OrderQueue, body.sku_id)
    return {"sku_id": body.sku_id, "queued": queued, "run_id": run_id}
