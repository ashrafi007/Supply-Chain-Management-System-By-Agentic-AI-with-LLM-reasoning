"""tests/api/test_queue_router.py -- GET/POST /queue, GET /queue/{sku_id}, POST /queue/sweep."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from src.api.deps import get_db, get_llm_client
from src.api.main import app
from src.queue import queue_repository
from tests.llm.conftest import StubOpenRouterClient


def _client(db_session, llm_client=None) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    if llm_client is not None:
        app.dependency_overrides[get_llm_client] = lambda: llm_client
    return TestClient(app)


def test_list_queue_empty_by_default(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/queue")
            assert response.status_code == 200
            assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_list_queue_returns_enqueued_row_and_filters_by_status(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=date.today(), source="manual_add")
    client = _client(db_session)
    try:
        with client:
            response = client.get("/queue")
            assert response.status_code == 200
            assert [r["sku_id"] for r in response.json()] == [known_sku_id]

            response = client.get("/queue", params={"status": "pending"})
            assert len(response.json()) == 1

            response = client.get("/queue", params={"status": "expired"})
            assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_list_queue_rejects_invalid_status(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/queue", params={"status": "bogus"})
            assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_get_queue_entry_404_for_unqueued_sku(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.get(f"/queue/{known_sku_id}")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_post_queue_enqueues_a_known_sku(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            due = (date.today() + timedelta(days=3)).isoformat()
            response = client.post("/queue", json={"sku_id": known_sku_id, "due_date": due, "source": "manual_add"})
            assert response.status_code == 201
            body = response.json()
            assert body["sku_id"] == known_sku_id
            assert body["status"] == "pending"
    finally:
        app.dependency_overrides.clear()


def test_post_queue_rejects_unknown_sku(db_session):
    client = _client(db_session)
    try:
        with client:
            due = date.today().isoformat()
            response = client.post("/queue", json={"sku_id": "SKU-DOES-NOT-EXIST-999999", "due_date": due})
            assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_post_queue_sweep_runs_due_sku_and_explains_it(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=date.today(), source="manual_add")
    stub_client = StubOpenRouterClient(response="Polished sweep explanation.")
    client = _client(db_session, llm_client=stub_client)
    try:
        with client:
            response = client.post("/queue/sweep")
            assert response.status_code == 200
            body = response.json()
            assert body["evaluated_sku_ids"] == [known_sku_id]
            assert len(body["run_ids"]) == 1
            assert body["run_ids"][0] in body["explanations"]
    finally:
        app.dependency_overrides.clear()


def test_post_queue_sweep_no_explain_flag_skips_network(db_session, known_sku_id):
    queue_repository.enqueue(db_session, known_sku_id, due_date=date.today(), source="manual_add")
    stub_client = StubOpenRouterClient()
    client = _client(db_session, llm_client=stub_client)
    try:
        with client:
            response = client.post("/queue/sweep", params={"explain": False})
            assert response.status_code == 200
            assert response.json()["explanations"] == {}
            assert stub_client.call_count == 0
    finally:
        app.dependency_overrides.clear()
