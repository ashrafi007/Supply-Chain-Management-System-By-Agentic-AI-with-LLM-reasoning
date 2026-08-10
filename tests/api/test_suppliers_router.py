"""tests/api/test_suppliers_router.py -- GET /suppliers, GET /suppliers/{supplier_id}."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.deps import get_db
from src.api.main import app
from src.db.models import Supplier


def _client(db_session) -> TestClient:
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def _make_supplier(db_session, supplier_id: str = "TEST-SUPPLIER-1") -> str:
    db_session.add(Supplier(supplier_id=supplier_id, name="Test Supplier Co", country="BD", lead_time_avg_days=5.0))
    db_session.commit()
    return supplier_id


def test_list_suppliers_includes_seeded_supplier(db_session):
    supplier_id = _make_supplier(db_session)
    client = _client(db_session)
    try:
        with client:
            response = client.get("/suppliers")
            assert response.status_code == 200
            ids = [s["supplier_id"] for s in response.json()]
            assert supplier_id in ids
    finally:
        app.dependency_overrides.clear()


def test_get_supplier_detail(db_session):
    supplier_id = _make_supplier(db_session)
    client = _client(db_session)
    try:
        with client:
            response = client.get(f"/suppliers/{supplier_id}")
            assert response.status_code == 200
            body = response.json()
            assert body["name"] == "Test Supplier Co"
            assert body["country"] == "BD"
    finally:
        app.dependency_overrides.clear()


def test_get_supplier_404_for_unknown_id(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/suppliers/NO-SUCH-SUPPLIER")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
