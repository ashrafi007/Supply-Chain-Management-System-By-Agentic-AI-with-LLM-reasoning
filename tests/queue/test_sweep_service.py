from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select

from src.db.models import PipelineRun
from src.llm.models import LLMExplanation
from src.queue import queue_repository, sweep_service
from src.queue.models import OrderQueue
from tests.fixtures.stub_executor import StubExecutor
from tests.llm.conftest import StubOpenRouterClient

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


class TestRunSweepAndExplain:
    """run_sweep_and_explain: opt-in automation layered on top of the untouched
    run_sweep(). No test here duplicates TestRunSweep's assertions about queue-row
    transitions -- those still go through the exact same run_sweep() call underneath."""

    def test_generates_and_caches_explanation_for_each_successful_run(
        self, db_session, known_sku_id
    ):
        queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")
        client = StubOpenRouterClient(response="Polished sweep explanation.")

        result = sweep_service.run_sweep_and_explain(db_session, TODAY, StubExecutor(), client)

        run_id = result["run_ids"][0]
        assert client.call_count == 1
        assert run_id in result["explanations"]
        assert result["explanations"][run_id]["explanation"] == "Polished sweep explanation."
        assert result["explanations"][run_id]["was_polished"] is True

        # Actually persisted, not just returned -- a second call must hit the cache,
        # not the (stub) network, per explainer_service.explain()'s _get_cached path.
        cached = db_session.execute(
            select(LLMExplanation).where(LLMExplanation.run_id == run_id)
        ).scalar_one_or_none()
        assert cached is not None
        assert cached.was_polished == 1

    def test_skips_explanation_for_failed_pipeline_run(self, db_session, known_sku_id):
        queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")
        client = StubOpenRouterClient()

        result = sweep_service.run_sweep_and_explain(
            db_session, TODAY, StubExecutor(raises=RuntimeError("boom")), client
        )

        assert client.call_count == 0
        assert result["explanations"] == {}

    def test_skips_network_call_entirely_when_nothing_is_due(self, db_session, known_sku_id):
        queue_repository.enqueue(
            db_session, known_sku_id, due_date=TODAY + timedelta(days=1), source="scheduled"
        )
        client = StubOpenRouterClient()

        result = sweep_service.run_sweep_and_explain(db_session, TODAY, StubExecutor(), client)

        assert client.call_count == 0
        assert result["explanations"] == {}

    def test_run_sweep_itself_is_unaffected_by_this_wrapper_existing(self, db_session, known_sku_id):
        """Regression guard for the design intent in llm_insertion_spec.md SS2: calling
        run_sweep() directly must still never touch the network, no matter what
        run_sweep_and_explain() does elsewhere in this module."""
        queue_repository.enqueue(db_session, known_sku_id, due_date=TODAY, source="scheduled")

        result = sweep_service.run_sweep(db_session, TODAY, StubExecutor())

        assert "explanations" not in result
