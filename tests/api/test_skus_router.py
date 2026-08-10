"""tests/api/test_skus_router.py -- GET /skus, GET /skus/{sku_id}, POST /skus."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app

_FULL_RAW_FEATURES = {
    "national_inv": 100.0, "lead_time": 8.0, "in_transit_qty": 40.0,
    "forecast_3_month": 70.0, "forecast_6_month": 140.0, "forecast_9_month": 210.0,
    "sales_1_month": 20.0, "sales_3_month": 60.0, "sales_6_month": 120.0, "sales_9_month": 180.0,
    "min_bank": 25.0, "potential_issue": 0, "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.9, "perf_12_month_avg": 0.88, "local_bo_qty": 0.0,
    "deck_risk": 0, "oe_constraint": 0, "ppap_risk": 0, "stop_auto_buy": 0, "rev_stop": 0,
}


def _client(db_session) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def test_list_skus_includes_known_sku(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/skus")
            assert response.status_code == 200
            ids = [s["sku_id"] for s in response.json()]
            assert known_sku_id in ids
    finally:
        app.dependency_overrides.clear()


def test_get_sku_detail_includes_raw_features(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.get(f"/skus/{known_sku_id}")
            assert response.status_code == 200
            body = response.json()
            assert body["sku_id"] == known_sku_id
            assert "national_inv" in body["raw_features"]
    finally:
        app.dependency_overrides.clear()


def test_get_sku_404_for_unknown_sku(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/skus/SKU-DOES-NOT-EXIST-999999")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_post_skus_creates_and_enqueues_new_sku(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.post("/skus", json={
                "sku_id": "TEST-NEW-SKU-API-1",
                "raw_features": _FULL_RAW_FEATURES,
                "due_in_days": 5,
            })
            assert response.status_code == 201
            body = response.json()
            assert body["sku_id"] == "TEST-NEW-SKU-API-1"
            assert body["queued"]["status"] == "pending"
            assert body["run_id"] is None

            detail = client.get("/skus/TEST-NEW-SKU-API-1")
            assert detail.status_code == 200
            assert detail.json()["raw_features"]["national_inv"] == 100.0
    finally:
        app.dependency_overrides.clear()


def test_post_skus_rejects_duplicate_sku_id(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.post("/skus", json={"sku_id": known_sku_id, "raw_features": {}})
            assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_post_skus_run_now_executes_the_real_pipeline(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.post("/skus", json={
                "sku_id": "TEST-NEW-SKU-API-RUNNOW",
                "raw_features": _FULL_RAW_FEATURES,
                "run_now": True,
            })
            assert response.status_code == 201
            body = response.json()
            assert body["run_id"] is not None
    finally:
        app.dependency_overrides.clear()
