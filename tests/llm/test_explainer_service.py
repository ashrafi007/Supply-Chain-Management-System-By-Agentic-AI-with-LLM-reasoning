"""tests/llm/test_explainer_service.py (llm_insertion_spec.md SS12) -- stub
OpenRouterClient, no real network call ever happens in this file."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.db.models import PipelineRun, Prediction
from src.llm import explainer_service
from src.llm.draft_builder import UnknownAgentError
from src.llm.explainer_service import RunNotFoundError
from tests.llm.conftest import StubOpenRouterClient


def test_first_call_stores_row_second_call_uses_cache(db_session, seeded_run):
    client = StubOpenRouterClient()

    # Whole-run (agent_name=None), not a single agent: seeded_run's rich field set
    # (demand + risk + urgency + correction + suppression + grade) reliably produces
    # a draft over SHORT_DRAFT_WORD_THRESHOLD so this test actually exercises the
    # polish path -- individual agents' humanized drafts (e.g. risk_detector's) are
    # now intentionally short one-sentence summaries and would skip polish entirely.
    first = explainer_service.explain(db_session, client, seeded_run, None)
    assert first["cached"] is False
    assert client.call_count == 1

    second = explainer_service.explain(db_session, client, seeded_run, None)
    assert second["cached"] is True
    assert client.call_count == 1  # stub not called again
    assert second["explanation"] == first["explanation"]


def test_api_failure_falls_back_to_draft_no_exception(db_session, seeded_run):
    client = StubOpenRouterClient(raise_unavailable=True)

    result = explainer_service.explain(db_session, client, seeded_run, None)

    assert result["was_polished"] is False
    assert result["fallback_reason"] == "api_error"
    assert result["model_used"] == explainer_service.TEMPLATE_ONLY
    assert result["explanation"]  # the draft itself -- a complete, correct answer


def test_short_draft_skips_network_call_entirely(db_session, known_sku_id):
    client = StubOpenRouterClient()
    run_id = "short-draft-run"
    now = datetime.now(timezone.utc)
    db_session.add(PipelineRun(
        run_id=run_id, sku_id=known_sku_id, started_at=now, completed_at=now,
        status="success", latency_ms=10, manifest_version="test",
    ))
    db_session.commit()
    # demand_forecast=None -> "did not produce a forecast" draft, well under 40 words.
    db_session.add(Prediction(run_id=run_id, sku_id=known_sku_id, demand_forecast=None))
    db_session.commit()

    result = explainer_service.explain(db_session, client, run_id, "demand_predictor")

    assert client.call_count == 0
    assert result["fallback_reason"] == "short_draft"
    assert result["was_polished"] is False


def test_unknown_agent_raises_before_client_called(db_session, seeded_run):
    client = StubOpenRouterClient()

    with pytest.raises(UnknownAgentError):
        explainer_service.explain(db_session, client, seeded_run, "agent_99_nonexistent")

    assert client.call_count == 0


def test_unknown_run_id_raises(db_session):
    client = StubOpenRouterClient()

    with pytest.raises(RunNotFoundError):
        explainer_service.explain(db_session, client, "no-such-run-id", None)

    assert client.call_count == 0
