"""The new manual deletion entrypoint (queue_migration_spec.md §5 Step 5)."""

from __future__ import annotations

from typing import Literal

from sqlalchemy.orm import Session

from . import queue_repository


def remove_from_queue(
    session: Session,
    sku_id: str,
    reason: Literal["expired_manual", "fulfilled"],
    deleted_by: str | None = None,
) -> None:
    """The only way a row leaves order_queue. Never called automatically by a sweep
    or a scheduler — this is the function a future "Mark Complete"/"Remove Expired"
    frontend action calls. Raises if sku_id is not currently in order_queue."""
    queue_repository.log_and_delete(session, sku_id, reason, deleted_by)
