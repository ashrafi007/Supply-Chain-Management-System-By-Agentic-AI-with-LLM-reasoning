from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.db.models import AgentTrace, PipelineRun, Prediction, Sku
from src.pipeline_state import AgentEvent
from src.repository.pipeline_service import run_pipeline_for_sku, run_pipeline_for_skus
from src.repository.state_builder import build_initial_state
from tests.conftest import UNKNOWN_SKU_ID
from tests.fixtures.stub_executor import KeyedStubExecutor, StubExecutor


# ── infra sanity check ───────────────────────────────────────────────────────


def test_savepoint_rollback_discards_committed_rows(db_session):
    """Confirms the conftest fixture actually rolls back commits made by code
    under test, before anything is built on top of it."""
    run_id = "sanity-check-run-id"
    sku_id = db_session.execute(select(Sku.sku_id).limit(1)).scalar_one()
    db_session.add(
        PipelineRun(
            run_id=run_id,
            sku_id=sku_id,
            started_at=datetime.now(timezone.utc),
            status="pending",
            manifest_version="sanity",
        )
    )
    db_session.commit()
    assert db_session.get(PipelineRun, run_id) is not None


def test_savepoint_rollback_confirms_previous_test_did_not_persist(db_session):
    assert db_session.get(PipelineRun, "sanity-check-run-id") is None


# ── state_builder ─────────────────────────────────────────────────────────────


def test_build_initial_state_known_sku_returns_populated_state(db_session, known_sku_id):
    state = build_initial_state(db_session, known_sku_id)

    assert state is not None
    assert state["sku_id"] == known_sku_id
    assert isinstance(state["raw_features"], dict)
    assert "national_inv" in state["raw_features"]
    assert state["demand_forecast"] is None
    assert state["trace"] == []
    assert state["errors"] == []


def test_build_initial_state_unknown_sku_returns_none(db_session):
    assert build_initial_state(db_session, UNKNOWN_SKU_ID) is None


# ── run_pipeline_for_sku: success path ───────────────────────────────────────


def test_run_pipeline_for_sku_success(db_session, known_sku_id):
    run_id = run_pipeline_for_sku(db_session, known_sku_id, StubExecutor())

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "success"
    assert run.manifest_version == "test-manifest-v1"

    predictions = db_session.execute(
        select(Prediction).where(Prediction.run_id == run_id)
    ).scalars().all()
    assert len(predictions) == 1

    traces = db_session.execute(
        select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.sequence)
    ).scalars().all()
    assert len(traces) == 5
    assert [t.sequence for t in traces] == [1, 2, 3, 4, 5]


# ── run_pipeline_for_sku: failure paths ──────────────────────────────────────


def test_run_pipeline_for_sku_unknown_sku(db_session, orphan_sku_id):
    run_id = run_pipeline_for_sku(db_session, orphan_sku_id, StubExecutor())

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "failed"
    assert run.error == "sku_not_found"

    predictions = db_session.execute(
        select(Prediction).where(Prediction.run_id == run_id)
    ).scalars().all()
    assert predictions == []


def test_run_pipeline_for_sku_executor_raises(db_session, known_sku_id):
    executor = StubExecutor(raises=RuntimeError("model file not found"))
    run_id = run_pipeline_for_sku(db_session, known_sku_id, executor)

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "failed"
    assert "model file not found" in run.error

    predictions = db_session.execute(
        select(Prediction).where(Prediction.run_id == run_id)
    ).scalars().all()
    assert predictions == []


def test_run_pipeline_for_sku_check_constraint_violation(db_session, known_sku_id):
    executor = StubExecutor(field_overrides={"urgency_score": 1.4})
    run_id = run_pipeline_for_sku(db_session, known_sku_id, executor)

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "failed"
    assert "constraint" in run.error.lower()

    predictions = db_session.execute(
        select(Prediction).where(Prediction.run_id == run_id)
    ).scalars().all()
    assert predictions == []


def test_run_pipeline_for_sku_suppression_case(db_session, known_sku_id):
    events = [
        AgentEvent(agent_name="demand_predictor", sequence=1, status="success",
                   latency_ms=10, output={}, note=None),
        AgentEvent(agent_name="risk_detector", sequence=2, status="success",
                   latency_ms=10, output={}, note=None),
        AgentEvent(agent_name="forecast_optimizer", sequence=3, status="skipped",
                   latency_ms=None, output={}, note="suppressed: alarm_triggered=1"),
        AgentEvent(agent_name="inventory_rebalancer", sequence=4, status="success",
                   latency_ms=10, output={}, note=None),
        AgentEvent(agent_name="supplier_auditor", sequence=5, status="success",
                   latency_ms=10, output={}, note=None),
    ]
    executor = StubExecutor(
        field_overrides={"alarm_triggered": 1, "correction_factor": 1.0},
        agent_events=events,
    )

    run_id = run_pipeline_for_sku(db_session, known_sku_id, executor)

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "success"

    skipped = db_session.execute(
        select(AgentTrace).where(
            AgentTrace.run_id == run_id, AgentTrace.status == "skipped"
        )
    ).scalars().all()
    assert len(skipped) == 1
    assert skipped[0].note == "suppressed: alarm_triggered=1"
    assert skipped[0].agent_name == "forecast_optimizer"


# ── batch ─────────────────────────────────────────────────────────────────────


def test_run_pipeline_for_skus_isolates_failures(db_session):
    sku_ids = db_session.execute(select(Sku.sku_id).limit(3)).scalars().all()
    assert len(sku_ids) == 3
    good_a, good_b, bad = sku_ids

    executor = KeyedStubExecutor({
        good_a: StubExecutor(),
        good_b: StubExecutor(),
        bad: StubExecutor(raises=RuntimeError("boom")),
    })

    run_ids = run_pipeline_for_skus(db_session, [good_a, good_b, bad], executor)

    assert len(run_ids) == 3
    assert len(set(run_ids)) == 3

    statuses = {
        sku_id: db_session.get(PipelineRun, run_id).status
        for sku_id, run_id in zip([good_a, good_b, bad], run_ids)
    }
    assert statuses[good_a] == "success"
    assert statuses[good_b] == "success"
    assert statuses[bad] == "failed"
