"""Fixtures shared by the order_queue test suite."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.queue import queue_repository


@pytest.fixture()
def queued_sku_id(db_session, known_sku_id) -> str:
    """known_sku_id enqueued today, due in 7 days, source='manual_add' — a sku_id
    already active in order_queue, for tests that need a preexisting row."""
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=date.today() + timedelta(days=7), source="manual_add"
    )
    return known_sku_id
