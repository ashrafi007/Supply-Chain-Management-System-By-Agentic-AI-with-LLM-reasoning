"""Fixtures shared by the llm_explanations test suite."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.db.models import AgentTrace, PipelineRun, Prediction
from src.llm.client import LLMUnavailableError


@pytest.fixture(autouse=True)
def _openrouter_key_env(monkeypatch):
    """Router/lifespan construction needs OPENROUTER_API_KEY present; the real
    network is never hit -- every test stubs client.polish() or takes the
    short-draft path."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy-key")


@pytest.fixture()
def seeded_run(db_session, known_sku_id) -> str:
    """A completed pipeline run with a full predictions row and one agent_traces
    row per agent, matching the real shapes written by src/orchestrator/nodes/*.
    alarm_triggered=1, so forecast_optimizer is realistically suppressed
    (src/orchestrator/edges.py should_suppress) -- its trace is 'skipped', not 'ok'.
    """
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Committed separately first -- matches src/repository/run_repository.py's
    # create_pending_run/persist_success split, which this mirrors for test data.
    db_session.add(PipelineRun(
        run_id=run_id, sku_id=known_sku_id, started_at=now, completed_at=now,
        status="success", latency_ms=120, manifest_version="test",
    ))
    db_session.commit()

    db_session.add(Prediction(
        run_id=run_id, sku_id=known_sku_id,
        demand_forecast=120.5, demand_velocity_band=None, stockout_risk=None,
        backorder_prob=0.97, alarm_triggered=1,
        urgency_score=0.82,
        correction_factor=1.0,
        replenishment_qty=None, route=None,
        supplier_risk="D", recommendation=None,
    ))
    traces = [
        AgentTrace(
            run_id=run_id, agent_name="demand_predictor", sequence=1, status="success",
            latency_ms=5, output={"tool": "demand_predictor", "status": "ok"}, note=None,
        ),
        AgentTrace(
            run_id=run_id, agent_name="risk_detector", sequence=2, status="success",
            latency_ms=5, output={
                "tool": "risk_detector", "status": "ok",
                "model_version": "stacking_ensemble_v1",
                "threshold_used": 0.945, "predicted_label": True,
            }, note=None,
        ),
        AgentTrace(
            run_id=run_id, agent_name="inventory_rebalancer", sequence=3, status="success",
            latency_ms=5, output={
                "tool": "inventory_rebalancer", "status": "ok",
                "model_version": "xgboost_urgency_v1",
                "recommended_qty": 42.0, "batch_rank": 1,
            }, note=None,
        ),
        AgentTrace(
            run_id=run_id, agent_name="forecast_optimizer", sequence=4, status="skipped",
            latency_ms=0, output={
                "tool": "forecast_optimizer", "status": "skipped", "reason": "alarm_triggered=1",
            }, note="suppressed: alarm_triggered=1",
        ),
        AgentTrace(
            run_id=run_id, agent_name="supplier_auditor", sequence=5, status="success",
            latency_ms=5, output={
                "tool": "supplier_auditor", "status": "ok",
                "model_version": "xgboost_supplier_auditor_v1",
                "risk_probability": 0.81, "stop_auto_buy_triggered": True,
                "trigger_reason": "MODEL", "delivery_stress": 0.6,
            }, note=None,
        ),
    ]
    for trace in traces:
        db_session.add(trace)
    db_session.commit()
    return run_id


class StubOpenRouterClient:
    """No network call -- polish() behavior is fully configurable per test."""

    model = "stub-model"

    def __init__(self, response: str = "Polished explanation.", raise_unavailable: bool = False):
        self._response = response
        self._raise_unavailable = raise_unavailable
        self.call_count = 0

    def polish(self, draft: str) -> str:
        self.call_count += 1
        if self._raise_unavailable:
            raise LLMUnavailableError("stub: simulated failure")
        return self._response
