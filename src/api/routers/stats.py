"""Dashboard aggregate numbers -- the one summary endpoint a frontend landing page
needs so it isn't stuck computing counts client-side over /queue and /runs.

    GET /stats

"Current" state per SKU is read via order_queue.last_run_id (the most recent
evaluated run for each queued SKU), not every historical predictions row -- a SKU
re-evaluated five times should count once, as its latest result, not five times.
SKUs never queued (no order_queue row at all) are excluded from the grade/risk
counts for the same reason -- there's no "current" prediction to attribute to them.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas import StatsOut
from src.db.models import Prediction, Sku
from src.queue.models import OrderQueue

router = APIRouter()


@router.get("/stats", response_model=StatsOut)
def get_stats(session: Session = Depends(get_db)) -> dict:
    total_skus = session.execute(select(func.count()).select_from(Sku)).scalar_one()

    queue_rows = session.execute(
        select(OrderQueue.status, func.count()).group_by(OrderQueue.status)
    ).all()
    queue_counts_by_status = {status: count for status, count in queue_rows}

    latest_run_ids = session.execute(
        select(OrderQueue.last_run_id).where(OrderQueue.last_run_id.is_not(None))
    ).scalars().all()

    supplier_grade_distribution: Counter[str] = Counter()
    high_risk_sku_count = 0
    if latest_run_ids:
        predictions = session.execute(
            select(Prediction).where(Prediction.run_id.in_(latest_run_ids))
        ).scalars().all()
        for p in predictions:
            if p.supplier_risk:
                supplier_grade_distribution[p.supplier_risk] += 1
            if p.alarm_triggered:
                high_risk_sku_count += 1

    return {
        "total_skus": total_skus,
        "queue_counts_by_status": queue_counts_by_status,
        "supplier_grade_distribution": dict(supplier_grade_distribution),
        "high_risk_sku_count": high_risk_sku_count,
    }
