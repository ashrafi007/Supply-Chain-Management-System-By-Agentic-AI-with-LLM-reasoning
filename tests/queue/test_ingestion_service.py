from __future__ import annotations

from datetime import date, timedelta

from src.queue import ingestion_service
from src.queue.models import OrderQueue
from tests.conftest import UNKNOWN_SKU_ID

TODAY = date.today()


def test_enqueue_new_sku_succeeds_for_valid_sku(db_session, known_sku_id):
    ingestion_service.enqueue_new_sku(
        db_session, known_sku_id, due_date=TODAY + timedelta(days=5), source="manual_add"
    )

    row = db_session.get(OrderQueue, known_sku_id)
    assert row is not None
    assert row.status == "pending"


def test_enqueue_new_sku_raises_for_unknown_sku(db_session):
    try:
        ingestion_service.enqueue_new_sku(
            db_session, UNKNOWN_SKU_ID, due_date=TODAY, source="manual_add"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass

    assert db_session.get(OrderQueue, UNKNOWN_SKU_ID) is None


def test_enqueue_new_sku_raises_for_already_active_sku(db_session, queued_sku_id):
    original = db_session.get(OrderQueue, queued_sku_id)
    original_due_date = original.due_date

    try:
        ingestion_service.enqueue_new_sku(
            db_session, queued_sku_id, due_date=TODAY + timedelta(days=99), source="manual_add"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass

    row = db_session.get(OrderQueue, queued_sku_id)
    assert row.due_date == original_due_date


def test_enqueue_new_sku_rejects_invalid_source(db_session, known_sku_id):
    try:
        ingestion_service.enqueue_new_sku(db_session, known_sku_id, due_date=TODAY, source="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass

    assert db_session.get(OrderQueue, known_sku_id) is None
