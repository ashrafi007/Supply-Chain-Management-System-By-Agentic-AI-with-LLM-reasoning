"""Real seeded DB + repository layer (untouched) + real LangGraphExecutor, full stack.

Replaces orchestrator_spec.md's 11.7 (swap StubExecutor for real in test_repository.py):
that file's tests are inherently StubExecutor-shaped (forcing raises/CHECK-violations
a real executor can't be told to reproduce), so the real "does the real graph work
against the real repository layer" proof lives here instead.
"""

from __future__ import annotations

from sqlalchemy import select

from src.db.models import AgentTrace, PipelineRun, Prediction
from src.orchestrator.executor import LangGraphExecutor
from src.repository.pipeline_service import run_pipeline_for_sku


def test_real_executor_against_real_seeded_db_succeeds(db_session, known_sku_id):
    executor = LangGraphExecutor()
    run_id = run_pipeline_for_sku(db_session, known_sku_id, executor)

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "success", getattr(run, "error", None)

    prediction = db_session.execute(
        select(Prediction).where(Prediction.run_id == run_id)
    ).scalar_one()

    traces = db_session.execute(
        select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.sequence)
    ).scalars().all()
    assert len(traces) == 5
    assert [t.sequence for t in traces] == [1, 2, 3, 4, 5]

    # Routing (Agent 4) is permanently cancelled -- these stay None by design,
    # not by accident. A future contributor should not "fix" this as a bug.
    assert prediction.route is None
    assert prediction.replenishment_qty is None
    # recommendation is reserved for a future LLM narration node, out of scope here.
    assert prediction.recommendation is None
