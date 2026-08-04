from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from src.queue import queue_repository
from src.queue.models import OrderQueue, OrderQueueLog
from src.repository.pipeline_service import run_pipeline_for_sku
from tests.fixtures.stub_executor import StubExecutor

TODAY = date.today()


def test_enqueue_creates_pending_row(db_session, known_sku_id):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY + timedelta(days=3), source="manual_add"
    )

    row = db_session.get(OrderQueue, known_sku_id)
    assert row is not None
    assert row.status == "pending"
    assert row.due_date == TODAY + timedelta(days=3)
    assert row.source == "manual_add"
    assert row.queued_at is not None


def test_get_due_skus_transitions_and_returns_due_rows(db_session, known_sku_id, orphan_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")
    queue_repository.enqueue(
        db_session, orphan_sku_id, due_date=TODAY + timedelta(days=1), source="scheduled"
    )

    due = queue_repository.get_due_skus(db_session, TODAY)

    assert due == [known_sku_id]
    assert db_session.get(OrderQueue, known_sku_id).status == "due_today"
    assert db_session.get(OrderQueue, orphan_sku_id).status == "pending"


def test_mark_expired_flips_only_past_due_rows_and_returns_count(
    db_session, known_sku_id, orphan_sku_id
):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY - timedelta(days=1), source="scheduled"
    )
    queue_repository.enqueue(
        db_session, orphan_sku_id, due_date=TODAY + timedelta(days=1), source="scheduled"
    )

    count = queue_repository.mark_expired(db_session, TODAY)

    assert count == 1
    assert db_session.get(OrderQueue, known_sku_id).status == "expired"
    assert db_session.get(OrderQueue, orphan_sku_id).status == "pending"
    # never deletes
    assert db_session.get(OrderQueue, known_sku_id) is not None


def test_mark_expired_does_not_touch_due_today(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")

    count = queue_repository.mark_expired(db_session, TODAY)

    assert count == 0
    assert db_session.get(OrderQueue, known_sku_id).status == "pending"


def test_mark_evaluated_updates_run_and_timestamp_without_changing_status(
    db_session, known_sku_id
):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")
    run_id = run_pipeline_for_sku(db_session, known_sku_id, StubExecutor())
    evaluated_at = datetime.now(timezone.utc)

    queue_repository.mark_evaluated(db_session, known_sku_id, run_id, evaluated_at)

    row = db_session.get(OrderQueue, known_sku_id)
    assert row.last_run_id == run_id
    # SQLite's DateTime column drops tzinfo on round-trip; compare naive values.
    assert row.last_evaluated_at == evaluated_at.replace(tzinfo=None)
    assert row.status == "pending"


def test_mark_evaluated_raises_for_missing_row(db_session, known_sku_id):
    try:
        queue_repository.mark_evaluated(db_session, known_sku_id, "fake-run-id", datetime.now(timezone.utc))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_log_and_delete_moves_row_to_log_with_correct_reason(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="manual_add")

    queue_repository.log_and_delete(db_session, known_sku_id, reason="fulfilled", deleted_by="tester")

    assert db_session.get(OrderQueue, known_sku_id) is None
    logs = db_session.execute(
        select(OrderQueueLog).where(OrderQueueLog.sku_id == known_sku_id)
    ).scalars().all()
    assert len(logs) == 1
    assert logs[0].reason == "fulfilled"
    assert logs[0].deleted_by == "tester"


def test_log_and_delete_nonexistent_sku_raises_and_writes_nothing(db_session, known_sku_id):
    try:
        queue_repository.log_and_delete(db_session, known_sku_id, reason="fulfilled", deleted_by=None)
        assert False, "expected ValueError"
    except ValueError:
        pass

    logs = db_session.execute(
        select(OrderQueueLog).where(OrderQueueLog.sku_id == known_sku_id)
    ).scalars().all()
    assert logs == []
