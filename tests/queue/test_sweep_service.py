from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select

from src.db.models import PipelineRun
from src.queue import queue_repository, sweep_service
from src.queue.models import OrderQueue
from tests.fixtures.stub_executor import StubExecutor

TODAY = date.today()


def test_run_sweep_flips_expired_without_invoking_pipeline(db_session, known_sku_id):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY - timedelta(days=1), source="scheduled"
    )

    result = sweep_service.run_sweep(db_session, TODAY, StubExecutor())

    row = db_session.get(OrderQueue, known_sku_id)
    assert row is not None
    assert row.status == "expired"
    assert row.last_run_id is None
    assert result["expired_count"] == 1
    assert result["evaluated_sku_ids"] == []


def test_run_sweep_transitions_due_today_invokes_pipeline_and_marks_evaluated(
    db_session, known_sku_id
):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")

    result = sweep_service.run_sweep(db_session, TODAY, StubExecutor())

    row = db_session.get(OrderQueue, known_sku_id)
    assert row.status == "due_today"
    assert row.last_run_id is not None
    assert row.last_evaluated_at is not None
    assert result["evaluated_sku_ids"] == [known_sku_id]

    run = db_session.get(PipelineRun, row.last_run_id)
    assert run.status == "success"


def test_run_sweep_ignores_not_yet_due_rows(db_session, known_sku_id):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY + timedelta(days=1), source="scheduled"
    )

    result = sweep_service.run_sweep(db_session, TODAY, StubExecutor())

    row = db_session.get(OrderQueue, known_sku_id)
    assert row.status == "pending"
    assert row.last_run_id is None
    assert result["evaluated_sku_ids"] == []
    assert result["expired_count"] == 0


def test_run_sweep_never_deletes_any_row(db_session, known_sku_id, orphan_sku_id):
    queue_repository.enqueue(
        db_session, known_sku_id, due_date=TODAY - timedelta(days=1), source="scheduled"
    )
    queue_repository.enqueue(db_session, orphan_sku_id, due_date=TODAY, source="scheduled")

    before = db_session.execute(select(OrderQueue.sku_id)).scalars().all()
    sweep_service.run_sweep(db_session, TODAY, StubExecutor())
    after = db_session.execute(select(OrderQueue.sku_id)).scalars().all()

    assert set(before) == set(after)


def test_run_sweep_marks_evaluated_even_on_pipeline_failure(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")

    result = sweep_service.run_sweep(db_session, TODAY, StubExecutor(raises=RuntimeError("boom")))

    row = db_session.get(OrderQueue, known_sku_id)
    assert row.last_run_id is not None
    assert row.last_evaluated_at is not None
    run = db_session.get(PipelineRun, row.last_run_id)
    assert run.status == "failed"
    assert result["evaluated_sku_ids"] == [known_sku_id]
