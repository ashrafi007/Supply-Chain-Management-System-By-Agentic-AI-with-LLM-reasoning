"""
New-SKU onboarding: insert a Sku + InventoryCurrent row from raw ("mother")
feature values supplied by a caller.

This module never takes engineered features (safety_gap, perf_gap, inv_velocity,
sales_trend, ...) as input, and never stores them — inventory_current is deliberately
raw-only (src/db/columns.py's ENGINEERED_EXCLUDE). Those are computed automatically,
fresh on every pipeline run, by features.fill_engineered_columns() inside
LangGraphExecutor.invoke() (src/orchestrator/executor.py). This module only prepares
the raw inputs that step depends on.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

import features as F
from src.db.models import InventoryCurrent, Sku, Supplier

_FIXED_INVENTORY_COLS = {"sku_id", "snapshot_at", "raw_extra", "updated_at"}

# Every raw column a new SKU can supply, in the same order inventory_current's
# schema declares them (which itself follows the raw CSV header order — see
# src/db/columns.py). Derived from the live ORM model, not hand-typed, so this
# never drifts from the actual table.
RAW_FEATURE_COLUMNS: list[str] = [
    c.name for c in InventoryCurrent.__table__.columns if c.name not in _FIXED_INVENTORY_COLS
]


def add_new_sku(
    session: Session,
    sku_id: str,
    raw_features: dict,
    supplier_id: str | None = None,
    description: str | None = None,
) -> None:
    """Insert one new Sku + InventoryCurrent row from raw feature values.

    - Raises ValueError if sku_id already exists in skus.
    - Raises ValueError if supplier_id is given but doesn't exist in suppliers.
    - Any RAW_FEATURE_COLUMNS key missing from raw_features is stored as NULL —
      inventory_current's raw columns are nullable by design (models.py); a
      caller doesn't have to supply every one of them.
    - Flag columns (potential_issue, deck_risk, oe_constraint, ppap_risk,
      stop_auto_buy, rev_stop) accept "Yes"/"No"/0/1 and are cleaned to 0/1 via
      the same features.clean_raw_row() the seeder uses — no new formula here.
    """
    if session.get(Sku, sku_id) is not None:
        raise ValueError(f"sku_id {sku_id!r} already exists in skus")
    if supplier_id is not None and session.get(Supplier, supplier_id) is None:
        raise ValueError(f"supplier_id {supplier_id!r} does not exist in suppliers")

    row = {col: raw_features.get(col) for col in RAW_FEATURE_COLUMNS}
    cleaned = F.clean_raw_row(pd.DataFrame([row])).iloc[0]

    kwargs: dict = {}
    for col in RAW_FEATURE_COLUMNS:
        val = cleaned[col]
        if pd.isna(val):
            kwargs[col] = None
        elif hasattr(val, "item"):
            kwargs[col] = val.item()
        else:
            kwargs[col] = val

    # Flush the Sku insert before adding InventoryCurrent: there's no relationship()
    # between the two mappers, so the unit of work has no dependency processor to
    # order the inserts on its own (same reason src/db/seed.py flushes skus before
    # inventory rows) -- without this, inventory_current's FK insert can race ahead
    # of the skus insert and fail.
    session.add(Sku(sku_id=sku_id, supplier_id=supplier_id, description=description))
    session.flush()
    session.add(InventoryCurrent(sku_id=sku_id, **kwargs))
    session.commit()
