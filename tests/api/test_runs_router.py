"""tests/api/test_runs_router.py -- GET /runs, GET /runs/{run_id}."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from tests.api.conftest import _make_run


def _client_with_db_override(db_session) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def test_list_runs_returns_the_seeded_run(db_session, seeded_run, known_sku_id):
    client = _client_with_db_override(db_session)
    try:
        with client:
            response = client.get("/runs", params={"sku_id": known_sku_id})
            assert response.status_code == 200
            run_ids = [row["run_id"] for row in response.json()]
            assert seeded_run in run_ids
    finally:
        app.dependency_overrides.clear()


def test_list_runs_respects_limit(db_session, known_sku_id):
    client = _client_with_db_override(db_session)
    for _ in range(3):
        _make_run(db_session, known_sku_id)
    try:
        with client:
            response = client.get("/runs", params={"sku_id": known_sku_id, "limit": 2})
            assert response.status_code == 200
            assert len(response.json()) == 2
    finally:
        app.dependency_overrides.clear()


def test_get_run_detail_includes_prediction_and_traces(db_session, seeded_run):
    client = _client_with_db_override(db_session)
    try:
        with client:
            response = client.get(f"/runs/{seeded_run}")
            assert response.status_code == 200
            body = response.json()
            assert body["run_id"] == seeded_run
            assert body["prediction"]["supplier_risk"] == "A"
            assert len(body["traces"]) == 1
            assert body["traces"][0]["agent_name"] == "demand_predictor"
    finally:
        app.dependency_overrides.clear()


def test_get_run_detail_handles_failed_run_with_no_prediction(db_session, seeded_failed_run):
    client = _client_with_db_override(db_session)
    try:
        with client:
            response = client.get(f"/runs/{seeded_failed_run}")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "failed"
            assert body["prediction"] is None
            assert body["traces"] == []
    finally:
        app.dependency_overrides.clear()


def test_get_run_detail_404_for_unknown_run_id(db_session):
    client = _client_with_db_override(db_session)
    try:
        with client:
            response = client.get("/runs/no-such-run")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
