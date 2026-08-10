"""tests/llm/test_explanations_router.py (llm_insertion_spec.md SS12)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.deps import get_db, get_llm_client
from src.api.main import app
from tests.llm.conftest import StubOpenRouterClient


def _client_with_overrides(db_session, stub_client) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_llm_client] = lambda: stub_client
    return TestClient(app)


def test_whole_run_and_per_agent_requests_return_200(db_session, seeded_run):
    client = _client_with_overrides(db_session, StubOpenRouterClient())
    try:
        with client:
            whole_run = client.post(f"/predictions/{seeded_run}/explain")
            assert whole_run.status_code == 200

            per_agent = client.post(
                f"/predictions/{seeded_run}/explain", params={"agent_name": "risk_detector"}
            )
            assert per_agent.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_unknown_run_id_returns_404(db_session):
    client = _client_with_overrides(db_session, StubOpenRouterClient())
    try:
        with client:
            response = client.post("/predictions/no-such-run/explain")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_unknown_agent_name_returns_400(db_session, seeded_run):
    client = _client_with_overrides(db_session, StubOpenRouterClient())
    try:
        with client:
            response = client.post(
                f"/predictions/{seeded_run}/explain", params={"agent_name": "not_a_real_agent"}
            )
            assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_api_failure_still_returns_200_with_fallback_reason(db_session, seeded_run):
    client = _client_with_overrides(db_session, StubOpenRouterClient(raise_unavailable=True))
    try:
        with client:
            # Whole-run, not a single agent: seeded_run's rich field set reliably
            # exceeds SHORT_DRAFT_WORD_THRESHOLD, so this actually exercises the
            # "API call attempted and failed" path rather than "skipped as short."
            response = client.post(f"/predictions/{seeded_run}/explain")
            assert response.status_code == 200
            assert response.json()["fallback_reason"] == "api_error"
    finally:
        app.dependency_overrides.clear()
