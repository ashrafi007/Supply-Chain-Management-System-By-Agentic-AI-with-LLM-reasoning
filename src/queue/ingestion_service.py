"""
New-SKU flow: the one validated entrypoint for adding rows to order_queue.

Not detailed in queue_migration_spec.md (it assumes this already existed) — designed
here as a thin validation layer in front of queue_repository.enqueue. Both a future
"manual add" frontend action and any later scheduled reorder-trigger job call this
same function with different `source` values; deciding *when* to call it with
source='scheduled' is out of scope here (queue_migration_spec.md §8).

sweep_service.py never calls this or queue_repository.enqueue directly — a sweep only
transitions/evaluates rows that already exist, it never creates them.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from sqlalchemy.orm import Session

from src.db.models import Sku

from . import queue_repository
from .models import OrderQueue

_VALID_SOURCES = ("scheduled", "manual_add")


def enqueue_new_sku(
    session: Session,
    sku_id: str,
    due_date: date,
    source: Literal["scheduled", "manual_add"],
) -> None:
    if source not in _VALID_SOURCES:
        raise ValueError(f"invalid source {source!r}")
    if session.get(Sku, sku_id) is None:
        raise ValueError(f"sku_id {sku_id!r} does not exist in skus")
    if session.get(OrderQueue, sku_id) is not None:
        raise ValueError(f"sku_id {sku_id!r} is already active in order_queue")

    queue_repository.enqueue(session, sku_id, due_date, source)
