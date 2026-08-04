"""
SQLAlchemy models for the order queue feature (queue_migration_spec.md §3).

Registers OrderQueue and OrderQueueLog on the existing Base from src.db.base — additive
only, never a second declarative_base() or a second engine.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OrderQueue(Base):
    __tablename__ = "order_queue"

    sku_id: Mapped[str] = mapped_column(Text, ForeignKey("skus.sku_id"), primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    queued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_run_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("pipeline_runs.run_id"), nullable=True
    )
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'due_today', 'expired')", name="ck_queue_status"
        ),
        CheckConstraint(
            "source IN ('scheduled', 'manual_add')", name="ck_queue_source"
        ),
        Index("idx_queue_status", "status"),
        Index("idx_queue_due_date", "due_date"),
    )


class OrderQueueLog(Base):
    """Append-only. No FK on sku_id/last_run_id — the rows they'd reference are
    already gone by the time a log entry is written (queue_migration_spec.md §3.2)."""

    __tablename__ = "order_queue_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku_id: Mapped[str] = mapped_column(Text, nullable=False)
    queued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    deleted_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "reason IN ('expired_manual', 'fulfilled')", name="ck_queue_log_reason"
        ),
        Index("idx_queue_log_sku", "sku_id"),
    )
