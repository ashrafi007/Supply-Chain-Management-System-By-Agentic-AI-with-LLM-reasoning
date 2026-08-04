from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select

from src.db.models import AgentTrace, PipelineRun, Prediction, Sku
from src.queue import deletion_service, queue_repository
from src.queue.models import OrderQueue, OrderQueueLog
from src.repository.pipeline_service import run_pipeline_for_sku
from tests.fixtures.stub_executor import StubExecutor

TODAY = date.today()


def test_remove_from_queue_fulfilled_success(db_session, queued_sku_id):
    deletion_service.remove_from_queue(db_session, queued_sku_id, reason="fulfilled")

    assert db_session.get(OrderQueue, queued_sku_id) is None
    logs = db_session.execute(
        select(OrderQueueLog).where(OrderQueueLog.sku_id == queued_sku_id)
    ).scalars().all()
    assert len(logs) == 1
    assert logs[0].reason == "fulfilled"


def test_remove_from_queue_expired_manual_success(db_session, known_sku_id):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY - timedelta(days=2), source="scheduled"
    )

    deletion_service.remove_from_queue(db_session, known_sku_id, reason="expired_manual")

    assert db_session.get(OrderQueue, known_sku_id) is None
    logs = db_session.execute(
        select(OrderQueueLog).where(OrderQueueLog.sku_id == known_sku_id)
    ).scalars().all()
    assert len(logs) == 1
    assert logs[0].reason == "expired_manual"


def test_remove_from_queue_raises_for_not_queued_sku(db_session, known_sku_id):
    try:
        deletion_service.remove_from_queue(db_session, known_sku_id, reason="fulfilled")
        assert False, "expected ValueError"
    except ValueError:
        pass

    logs = db_session.execute(
        select(OrderQueueLog).where(OrderQueueLog.sku_id == known_sku_id)
    ).scalars().all()
    assert logs == []


def test_remove_from_queue_preserves_unrelated_history(db_session, known_sku_id):
    run_id = run_pipeline_for_sku(db_session, known_sku_id, StubExecutor())
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")

    deletion_service.remove_from_queue(db_session, known_sku_id, reason="fulfilled")

    assert db_session.get(Sku, known_sku_id) is not None
    run = db_session.get(PipelineRun, run_id)
    assert run is not None
    assert run.status == "success"
    assert db_session.get(Prediction, run_id) is not None
    traces = db_session.execute(
        select(AgentTrace).where(AgentTrace.run_id == run_id)
    ).scalars().all()
    assert len(traces) > 0


def test_remove_from_queue_allows_reenqueue(db_session, queued_sku_id):
    deletion_service.remove_from_queue(db_session, queued_sku_id, reason="fulfilled")

    queue_repository.enqueue(
        db_session, queued_sku_id, due_date=TODAY + timedelta(days=1), source="manual_add"
    )

    row = db_session.get(OrderQueue, queued_sku_id)
    assert row is not None
    assert row.status == "pending"
