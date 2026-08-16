"""tests/api/test_db_router.py -- generic reflection-based table browser + CRUD
(GET/POST/PUT/DELETE /db/tables...). Uses `suppliers` and `skus` as the exercise
tables since they're simple, real, and safe to mutate; the router itself is
table-name-agnostic so these two stand in for all ten."""

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


def test_list_tables_includes_every_mapped_table(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/db/tables")
            assert response.status_code == 200
            names = {t["name"] for t in response.json()}
            assert names == {
                "suppliers", "skus", "inventory_current", "pipeline_runs",
                "predictions", "agent_traces", "forecast_actuals",
                "order_queue", "order_queue_log", "llm_explanations",
            }
    finally:
        app.dependency_overrides.clear()


def test_list_tables_row_counts_reflect_real_data(db_session, known_sku_id):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/db/tables")
            skus_row = next(t for t in response.json() if t["name"] == "skus")
            assert skus_row["row_count"] >= 1
            assert skus_row["primary_key"] == "sku_id"
    finally:
        app.dependency_overrides.clear()


def test_get_table_404_for_unknown_table(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.get("/db/tables/not_a_real_table")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_table_returns_column_metadata_and_rows(db_session):
    db_session.add(Supplier(supplier_id="SUP-META-TEST", name="Meta Test Co"))
    db_session.commit()

    client = _client(db_session)
    try:
        with client:
            response = client.get("/db/tables/suppliers")
            assert response.status_code == 200
            body = response.json()
            assert body["primary_key"] == "supplier_id"
            col_names = {c["name"] for c in body["columns"]}
            assert col_names == {"supplier_id", "name", "country", "lead_time_avg_days", "created_at"}
            pk_col = next(c for c in body["columns"] if c["name"] == "supplier_id")
            assert pk_col["primary_key"] is True
            sku_ids = [r["supplier_id"] for r in body["rows"]]
            assert "SUP-META-TEST" in sku_ids
    finally:
        app.dependency_overrides.clear()


def test_create_row_then_read_it_back(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.post(
                "/db/tables/suppliers",
                json={"supplier_id": "SUP-CRUD-1", "name": "CRUD Co", "country": "US"},
            )
            assert response.status_code == 201
            body = response.json()
            assert body["supplier_id"] == "SUP-CRUD-1"
            assert body["name"] == "CRUD Co"
            assert body["created_at"]  # server-side default applied even via Core insert
    finally:
        app.dependency_overrides.clear()


def test_create_row_missing_required_field_returns_400(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.post("/db/tables/suppliers", json={"supplier_id": "SUP-BAD"})
            assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_update_row_changes_fields_but_not_primary_key(db_session):
    db_session.add(Supplier(supplier_id="SUP-UPDATE-1", name="Original Name", country="US"))
    db_session.commit()

    client = _client(db_session)
    try:
        with client:
            response = client.put(
                "/db/tables/suppliers/SUP-UPDATE-1",
                json={"name": "Renamed Co", "supplier_id": "SUP-SHOULD-BE-IGNORED"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["supplier_id"] == "SUP-UPDATE-1"  # PK immutable via this endpoint
            assert body["name"] == "Renamed Co"
    finally:
        app.dependency_overrides.clear()


def test_update_row_404_for_missing_row(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.put("/db/tables/suppliers/NO-SUCH-SUPPLIER", json={"name": "x"})
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_row_succeeds_and_row_is_gone(db_session):
    db_session.add(Supplier(supplier_id="SUP-DELETE-1", name="Delete Me"))
    db_session.commit()

    client = _client(db_session)
    try:
        with client:
            response = client.delete("/db/tables/suppliers/SUP-DELETE-1")
            assert response.status_code == 204

            follow_up = client.get("/db/tables/suppliers")
            ids = [r["supplier_id"] for r in follow_up.json()["rows"]]
            assert "SUP-DELETE-1" not in ids
    finally:
        app.dependency_overrides.clear()


def test_delete_row_404_for_missing_row(db_session):
    client = _client(db_session)
    try:
        with client:
            response = client.delete("/db/tables/suppliers/NO-SUCH-SUPPLIER")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_row_blocked_by_foreign_key_returns_409(db_session, known_sku_id):
    # known_sku_id's skus row has supplier_id="UNKNOWN" per seed data -- deleting that
    # supplier while a real sku references it must be blocked, not silently succeed
    # or 500.
    client = _client(db_session)
    try:
        with client:
            response = client.delete("/db/tables/suppliers/UNKNOWN")
            assert response.status_code == 409
    finally:
        app.dependency_overrides.clear()
