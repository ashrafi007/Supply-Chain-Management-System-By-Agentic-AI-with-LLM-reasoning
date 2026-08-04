"""
Reads/writes for order_queue and order_queue_log (queue_migration_spec.md §5 Step 4).

Design decision — get_due_skus: this function performs the pending -> due_today
transition itself (a bulk UPDATE), then reads back and returns the resulting sku_ids.
It is not a read-only query despite its name. This keeps every order_queue mutation
inside this module — sweep_service.py stays a pure orchestrator with no ORM writes of
its own — and makes the sweep idempotent: a crashed sweep can re-run get_due_skus and
pick the same due_today rows back up rather than losing them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .models import OrderQueue, OrderQueueLog

_VALID_REASONS = ("expired_manual", "fulfilled")


def get_due_skus(session: Session, as_of: date) -> list[str]:
    session.execute(
        update(OrderQueue)
        .where(OrderQueue.status == "pending")
        .where(OrderQueue.due_date <= as_of)
        .values(status="due_today")
    )
    session.commit()

    rows = (
        session.execute(
            select(OrderQueue.sku_id)
            .where(OrderQueue.status == "due_today")
            .where(OrderQueue.due_date <= as_of)
            .order_by(OrderQueue.due_date, OrderQueue.sku_id)
        )
        .scalars()
        .all()
    )
    return list(rows)


def mark_expired(session: Session, as_of: date) -> int:
    """Status-only. Never deletes — rows past due_date with no order placed simply
    become 'expired' and stay in order_queue until a human removes them."""
    stmt = (
        update(OrderQueue)
        .where(OrderQueue.status.in_(("pending", "due_today")))
        .where(OrderQueue.due_date < as_of)
        .values(status="expired")
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount


def mark_evaluated(
    session: Session, sku_id: str, run_id: str, evaluated_at: datetime
) -> None:
    row = session.get(OrderQueue, sku_id)
    if row is None:
        raise ValueError(f"sku_id {sku_id!r} not found in order_queue")
    row.last_run_id = run_id
    row.last_evaluated_at = evaluated_at
    session.commit()


def enqueue(session: Session, sku_id: str, due_date: date, source: str) -> None:
    """No existence/duplicate validation here — that belongs to ingestion_service.
    This stays a plain insert since deletion_service's re-enqueue path calls it directly."""
    session.add(OrderQueue(sku_id=sku_id, due_date=due_date, source=source))
    session.commit()


def log_and_delete(
    session: Session,
    sku_id: str,
    reason: Literal["expired_manual", "fulfilled"],
    deleted_by: str | None,
) -> None:
    """Reads the order_queue row, writes it to order_queue_log, then deletes it.
    One transaction. Raises before any write if sku_id isn't currently queued."""
    if reason not in _VALID_REASONS:
        raise ValueError(f"invalid reason {reason!r}")

    row = session.get(OrderQueue, sku_id)
    if row is None:
        raise ValueError(f"sku_id {sku_id!r} not found in order_queue")

    try:
        session.add(
            OrderQueueLog(
                sku_id=row.sku_id,
                queued_at=row.queued_at,
                due_date=row.due_date,
                last_run_id=row.last_run_id,
                reason=reason,
                deleted_by=deleted_by,
            )
        )
        session.delete(row)
        session.flush()
        session.commit()
    except Exception:
        session.rollback()
        raise
