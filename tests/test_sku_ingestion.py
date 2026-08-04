from __future__ import annotations

import features as F
from src.db.models import InventoryCurrent, Sku
from src.queue import ingestion_service
from src.queue.models import OrderQueue
from src.repository.pipeline_service import run_pipeline_for_sku
from src.repository.sku_ingestion import RAW_FEATURE_COLUMNS, add_new_sku
from tests.fixtures.stub_executor import StubExecutor

NEW_SKU_ID = "TEST-NEW-SKU-0001"

BASE_RAW_FEATURES = {
    "national_inv": 500,
    "lead_time": 7,
    "in_transit_qty": 20,
    "forecast_3_month": 90,
    "forecast_6_month": 180,
    "forecast_9_month": 270,
    "sales_1_month": 30,
    "sales_3_month": 90,
    "sales_6_month": 180,
    "sales_9_month": 270,
    "min_bank": 50,
    "pieces_past_due": 0,
    "perf_6_month_avg": 0.95,
    "perf_12_month_avg": 0.90,
    "local_bo_qty": 0,
    "potential_issue": "No",
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
}


def test_add_new_sku_creates_sku_and_inventory_row(db_session):
    add_new_sku(db_session, NEW_SKU_ID, BASE_RAW_FEATURES)

    sku = db_session.get(Sku, NEW_SKU_ID)
    inv = db_session.get(InventoryCurrent, NEW_SKU_ID)
    assert sku is not None
    assert inv is not None
    assert inv.national_inv == 500
    assert inv.min_bank == 50


def test_add_new_sku_cleans_yes_no_flags_to_binary(db_session):
    add_new_sku(db_session, NEW_SKU_ID, BASE_RAW_FEATURES)

    inv = db_session.get(InventoryCurrent, NEW_SKU_ID)
    assert inv.deck_risk == 1
    assert inv.potential_issue == 0
    assert inv.oe_constraint == 0


def test_add_new_sku_missing_raw_fields_become_null(db_session):
    sparse = {"national_inv": 100, "min_bank": 10}
    add_new_sku(db_session, NEW_SKU_ID, sparse)

    inv = db_session.get(InventoryCurrent, NEW_SKU_ID)
    assert inv.national_inv == 100
    untouched = [c for c in RAW_FEATURE_COLUMNS if c not in sparse]
    assert any(getattr(inv, c) is None for c in untouched)


def test_add_new_sku_raises_for_duplicate_sku_id(db_session, known_sku_id):
    try:
        add_new_sku(db_session, known_sku_id, BASE_RAW_FEATURES)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_add_new_sku_raises_for_unknown_supplier(db_session):
    try:
        add_new_sku(db_session, NEW_SKU_ID, BASE_RAW_FEATURES, supplier_id="NOT-A-SUPPLIER")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert db_session.get(Sku, NEW_SKU_ID) is None


def test_engineered_features_computed_correctly_from_stored_raw_values(db_session):
    add_new_sku(db_session, NEW_SKU_ID, BASE_RAW_FEATURES)
    inv = db_session.get(InventoryCurrent, NEW_SKU_ID)

    raw = {c: getattr(inv, c) for c in RAW_FEATURE_COLUMNS}
    engineered = F.fill_engineered_columns(raw)

    assert engineered["safety_gap"] == inv.national_inv - inv.min_bank
    assert engineered["perf_gap"] == abs(inv.perf_6_month_avg - inv.perf_12_month_avg)
    assert engineered["went_on_backorder"] == 0


def test_new_sku_can_be_enqueued_and_run_through_pipeline(db_session):
    from datetime import date

    add_new_sku(db_session, NEW_SKU_ID, BASE_RAW_FEATURES)
    ingestion_service.enqueue_new_sku(db_session, NEW_SKU_ID, due_date=date.today(), source="manual_add")

    assert db_session.get(OrderQueue, NEW_SKU_ID) is not None

    run_id = run_pipeline_for_sku(db_session, NEW_SKU_ID, StubExecutor())

    from src.db.models import PipelineRun

    run = db_session.get(PipelineRun, run_id)
    assert run.status == "success"
