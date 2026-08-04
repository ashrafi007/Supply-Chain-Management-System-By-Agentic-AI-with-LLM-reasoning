"""
Verification script (spec §7 step-5). Proves the schema's invariants are *enforced by the DB*,
not merely documented. Run after create_db:

    python -m src.db.verify_db      # prints SCHEMA OK and exits 0, or the failure and exits 1

Every mutating check runs inside a transaction that is rolled back, so the database stays empty.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from .base import SessionLocal, engine
from .models import (
    InventoryCurrent,
    PipelineRun,
    Prediction,
    Sku,
    Supplier,
    TARGET_COLUMN_NAMES,
)

EXPECTED_TABLES = {
    "suppliers", "skus", "inventory_current", "pipeline_runs",
    "predictions", "agent_traces", "forecast_actuals",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_fixture(session) -> None:
    """Create a valid supplier -> sku -> run inside the current (uncommitted) transaction,
    so downstream CHECK-violation inserts fail on the CHECK, not on a missing FK.

    Each parent is flushed before its child: the models declare column-level ForeignKeys but
    no ORM relationship(), so the unit-of-work does not reorder inserts by FK dependency on
    its own. Explicit ordered flushes guarantee the parent row exists first."""
    session.add(Supplier(supplier_id="SUP-1", name="Acme"))
    session.flush()
    session.add(Sku(sku_id="SKU-1", supplier_id="SUP-1"))
    session.flush()
    session.add(
        PipelineRun(
            run_id="RUN-1", sku_id="SKU-1", started_at=_now(),
            status="pending", manifest_version="test-manifest-v1",
        )
    )
    session.flush()


# ── individual checks: each returns None on success or raises AssertionError ──

def check_1_tables_exist() -> None:
    found = set(inspect(engine).get_table_names())
    missing = EXPECTED_TABLES - found
    extra = found - EXPECTED_TABLES
    assert not missing, f"missing tables: {sorted(missing)}"
    assert not extra, f"unexpected extra tables: {sorted(extra)}"


def check_2_foreign_keys_on() -> None:
    with engine.connect() as conn:
        val = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
    assert val == 1, f"PRAGMA foreign_keys returned {val!r}, expected 1"


def check_3_valid_chain_inserts() -> None:
    with SessionLocal() as s:
        _seed_fixture(s)
        s.add(InventoryCurrent(sku_id="SKU-1", national_inv=42.0, stop_auto_buy=0))
        s.flush()  # must not raise
        s.rollback()


def check_4_urgency_out_of_range_rejected() -> None:
    try:
        with SessionLocal() as s:
            _seed_fixture(s)
            s.add(Prediction(run_id="RUN-1", sku_id="SKU-1", urgency_score=1.4))
            s.flush()
        raise AssertionError("urgency_score=1.4 was accepted; CHECK not enforced")
    except IntegrityError:
        pass


def check_5_suppression_invariant_rejected() -> None:
    try:
        with SessionLocal() as s:
            _seed_fixture(s)
            s.add(
                Prediction(
                    run_id="RUN-1", sku_id="SKU-1",
                    alarm_triggered=1, correction_factor=0.8,
                )
            )
            s.flush()
        raise AssertionError(
            "alarm_triggered=1 with correction_factor=0.8 was accepted; "
            "suppression CHECK not enforced"
        )
    except IntegrityError:
        pass


def check_6_bad_supplier_fk_rejected() -> None:
    try:
        with SessionLocal() as s:
            s.add(Sku(sku_id="SKU-ORPHAN", supplier_id="NOPE-DOES-NOT-EXIST"))
            s.flush()
        raise AssertionError("SKU with non-existent supplier_id was accepted; FK not enforced")
    except IntegrityError:
        pass


def check_7_duplicate_sku_rejected() -> None:
    try:
        with SessionLocal() as s:
            _seed_fixture(s)
            s.add(InventoryCurrent(sku_id="SKU-1", national_inv=1.0))
            s.flush()
            s.add(InventoryCurrent(sku_id="SKU-1", national_inv=2.0))
            s.flush()
        raise AssertionError("duplicate sku_id in inventory_current was accepted; PK not enforced")
    except IntegrityError:
        pass


def check_8_no_target_column() -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("inventory_current")}
    leaked = cols & set(TARGET_COLUMN_NAMES)
    assert not leaked, f"inventory_current contains target column(s): {sorted(leaked)}"


def check_9_invalid_velocity_band_rejected() -> None:
    try:
        with SessionLocal() as s:
            _seed_fixture(s)
            s.add(Prediction(run_id="RUN-1", sku_id="SKU-1", demand_velocity_band="Zoom"))
            s.flush()
        raise AssertionError("invalid demand_velocity_band 'Zoom' accepted; CHECK not enforced")
    except IntegrityError:
        pass


def check_10_bad_stockout_flag_rejected() -> None:
    try:
        with SessionLocal() as s:
            _seed_fixture(s)
            s.add(Prediction(run_id="RUN-1", sku_id="SKU-1", stockout_risk=2))
            s.flush()
        raise AssertionError("stockout_risk=2 accepted; binary CHECK not enforced")
    except IntegrityError:
        pass


CHECKS = [
    ("1. all seven tables exist", check_1_tables_exist),
    ("2. PRAGMA foreign_keys == 1", check_2_foreign_keys_on),
    ("3. supplier->sku->inventory insert + rollback", check_3_valid_chain_inserts),
    ("4. urgency_score=1.4 rejected", check_4_urgency_out_of_range_rejected),
    ("5. alarm=1 & correction_factor=0.8 rejected", check_5_suppression_invariant_rejected),
    ("6. SKU with bad supplier_id rejected", check_6_bad_supplier_fk_rejected),
    ("7. duplicate inventory_current sku_id rejected", check_7_duplicate_sku_rejected),
    ("8. no target column in inventory_current", check_8_no_target_column),
    ("9. invalid demand_velocity_band rejected", check_9_invalid_velocity_band_rejected),
    ("10. stockout_risk=2 rejected", check_10_bad_stockout_flag_rejected),
]


def main() -> int:
    failures: list[str] = []
    for label, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {label}")
        except Exception as exc:  # noqa: BLE001 — report any failure and continue
            print(f"  FAIL  {label}  --  {exc}")
            failures.append(label)

    print()
    if failures:
        print(f"SCHEMA FAILED — {len(failures)} check(s) failed: {failures}")
        return 1
    print("SCHEMA OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
