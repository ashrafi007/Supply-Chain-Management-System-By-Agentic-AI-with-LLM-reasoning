"""Fixtures shared by the read-API test suite."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.db.models import AgentTrace, PipelineRun, Prediction


@pytest.fixture(autouse=True)
def _openrouter_key_env(monkeypatch):
    """main.py's lifespan constructs an OpenRouterClient on startup regardless of
    which router is under test; the real network is never hit by these tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy-key")


def _make_run(db_session, sku_id: str, *, status: str = "success", with_prediction: bool = True) -> str:
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    db_session.add(PipelineRun(
        run_id=run_id, sku_id=sku_id, started_at=now, completed_at=now,
        status=status, latency_ms=100, manifest_version="test",
    ))
    db_session.commit()

    if with_prediction:
        db_session.add(Prediction(
            run_id=run_id, sku_id=sku_id,
            demand_forecast=10.0, backorder_prob=0.1, alarm_triggered=0,
            urgency_score=0.2, correction_factor=1.0, supplier_risk="A",
        ))
        db_session.add(AgentTrace(
            run_id=run_id, agent_name="demand_predictor", sequence=1, status="success",
            latency_ms=5, output={"tool": "demand_predictor", "status": "ok"}, note=None,
        ))
        db_session.commit()

    return run_id


@pytest.fixture()
def seeded_run(db_session, known_sku_id) -> str:
    return _make_run(db_session, known_sku_id)


@pytest.fixture()
def seeded_failed_run(db_session, known_sku_id) -> str:
    return _make_run(db_session, known_sku_id, status="failed", with_prediction=False)
