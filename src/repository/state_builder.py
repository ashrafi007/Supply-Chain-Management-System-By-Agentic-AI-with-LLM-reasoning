"""
DB row -> PipelineState. Single responsibility: reads only, no writes, no cleaning,
no feature engineering (that lives inside each agent wrapper via features.py).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.db.models import InventoryCurrent
from src.pipeline_state import PipelineState

_METADATA_COLS = {"sku_id", "snapshot_at", "raw_extra", "updated_at"}


def build_initial_state(session: Session, sku_id: str) -> PipelineState | None:
    row = session.get(InventoryCurrent, sku_id)
    if row is None:
        return None

    raw_features = {
        col.name: getattr(row, col.name)
        for col in InventoryCurrent.__table__.columns
        if col.name not in _METADATA_COLS
    }

    return PipelineState(
        sku_id=sku_id,
        raw_features=raw_features,
        demand_forecast=None,
        demand_velocity_band=None,
        stockout_risk=None,
        backorder_prob=None,
        alarm_triggered=None,
        urgency_score=None,
        correction_factor=None,
        replenishment_qty=None,
        route=None,
        supplier_risk=None,
        recommendation=None,
        trace=[],
        errors=[],
    )
