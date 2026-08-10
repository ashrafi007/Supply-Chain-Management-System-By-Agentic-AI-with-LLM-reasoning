"""tests/api/test_stats_router.py -- GET /stats."""

from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from src.queue import queue_repository
from tests.api.conftest import _make_run


def _client(db_session) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def test_stats_counts_total_skus_and_empty_queue(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/stats")
            assert response.status_code == 200
            body = response.json()
            assert body["total_skus"] >= 1
            assert body["queue_counts_by_status"] == {}
            assert body["supplier_grade_distribution"] == {}
            assert body["high_risk_sku_count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_stats_reflects_queued_skus_latest_run(db_session, known_sku_id):
    run_id = _make_run(db_session, known_sku_id)  # supplier_risk="A", alarm_triggered=0
    queue_repository.enqueue(db_session, known_sku_id, due_date=date.today(), source="manual_add")
    queue_repository.mark_evaluated(db_session, known_sku_id, run_id, datetime.now(timezone.utc))

    client = _client(db_session)
    try:
        with client:
            response = client.get("/stats")
            assert response.status_code == 200
            body = response.json()
            assert body["queue_counts_by_status"] == {"pending": 1}
            assert body["supplier_grade_distribution"] == {"A": 1}
            assert body["high_risk_sku_count"] == 0
    finally:
        app.dependency_overrides.clear()
