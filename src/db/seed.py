"""
src/db/seed.py — populate suppliers / skus / inventory_current from the stratified sample
(.CLAUDE/seed_data_spec.md, entrypoint for §4).

Usage:
    python -m scripts.sample_skus          # optional: inspect the sample first
    python -m src.db.seed                   # seed (refuses if the three tables aren't empty)
    python -m src.db.seed --force           # truncate the three tables and reseed
    python -m src.db.seed --verify          # run post-seed verification only (spec §5)

Populates only ``suppliers``, ``skus``, ``inventory_current``. Never touches the four
execution tables (``pipeline_runs``, ``predictions``, ``agent_traces``, ``forecast_actuals``)
— those are written only by the running pipeline (spec §0).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

import features as F
from scripts.sample_skus import SKU_COL, TARGET_COL, sample_skus
from .base import SessionLocal, engine
from .columns import ENGINEERED_EXCLUDE, TARGET_COLS, raw_column_specs
from .models import InventoryCurrent, Sku, Supplier

UNKNOWN_SUPPLIER_ID = "UNKNOWN"

RAW_SPECS = raw_column_specs()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _table_counts() -> dict[str, int]:
    with engine.connect() as conn:
        return {
            t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            for t in ("suppliers", "skus", "inventory_current")
        }


# ── Step 2 — supplier extraction ────────────────────────────────────────────

def _extract_suppliers(sample: pd.DataFrame) -> tuple[list[Supplier], pd.Series]:
    """
    Returns (supplier rows to insert, a Series mapping each sampled sku -> supplier_id).

    This dataset carries no supplier column at all (checked against the raw CSV header),
    so per spec §4 Step 2 we insert a single synthetic 'UNKNOWN' supplier so the skus.supplier_id
    FK target exists, and note it as a data limitation rather than leaving supplier_id NULL.
    """
    supplier_col = next((c for c in ("supplier", "supplier_id", "supplier_name") if c in sample.columns), None)

    if supplier_col is None:
        print(
            "  [data limitation] source has no supplier column — inserting a single "
            f"synthetic supplier_id={UNKNOWN_SUPPLIER_ID!r}; every seeded SKU links to it."
        )
        suppliers = [Supplier(supplier_id=UNKNOWN_SUPPLIER_ID, name="Unknown (no supplier column in source data)")]
        sku_to_supplier = pd.Series(UNKNOWN_SUPPLIER_ID, index=sample.index)
        return suppliers, sku_to_supplier

    distinct = sample[supplier_col].dropna().unique().tolist()
    suppliers = [Supplier(supplier_id=str(v), name=str(v)) for v in distinct]
    sku_to_supplier = sample[supplier_col].map(lambda v: str(v) if pd.notna(v) else None)
    return suppliers, sku_to_supplier


# ── Step 3 — SKU insertion ──────────────────────────────────────────────────

def _build_skus(sample: pd.DataFrame, sku_to_supplier: pd.Series) -> list[Sku]:
    return [
        Sku(sku_id=str(int(row[SKU_COL])), supplier_id=sku_to_supplier.loc[idx])
        for idx, row in sample.iterrows()
    ]


# ── Step 4 — inventory insertion ────────────────────────────────────────────

def _to_cell(value, sqltype) -> object:
    if pd.isna(value):
        return None
    from sqlalchemy import Float, Integer
    if sqltype is Integer:
        return int(value)
    if sqltype is Float:
        return float(value)
    return str(value)


def _build_inventory_rows(sample: pd.DataFrame) -> list[InventoryCurrent]:
    # Assert the target column is not among the columns we're about to store (spec §4 Step 4.3).
    raw_names = {spec.name for spec in RAW_SPECS}
    assert TARGET_COLS.isdisjoint(raw_names), (
        f"target column(s) {TARGET_COLS & raw_names} present in inventory_current raw columns"
    )

    cleaned = F.clean_raw_row(sample)

    now = _now()
    rows = []
    for idx, row in cleaned.iterrows():
        kwargs = {
            "sku_id": str(int(row[SKU_COL])),
            "snapshot_at": now,
            "updated_at": now,
        }
        for spec in RAW_SPECS:
            kwargs[spec.name] = _to_cell(row[spec.name], spec.sqltype)
        rows.append(InventoryCurrent(**kwargs))
    return rows


# ── Step 5 — force handling ──────────────────────────────────────────────────

def _refuse_if_nonempty() -> None:
    counts = _table_counts()
    nonempty = {t: n for t, n in counts.items() if n > 0}
    if nonempty:
        raise SystemExit(
            f"Refusing to seed: tables already contain data {nonempty}. Use --force to "
            f"truncate suppliers/skus/inventory_current and reseed."
        )


def _force_truncate(session) -> None:
    print("--force: truncating inventory_current, skus, suppliers (FK-safe order)")
    session.execute(text("DELETE FROM inventory_current"))
    session.execute(text("DELETE FROM skus"))
    session.execute(text("DELETE FROM suppliers"))
    session.commit()


# ── Seeding orchestration ───────────────────────────────────────────────────

def seed(force: bool = False, target_size: int = 5000) -> pd.DataFrame:
    with SessionLocal() as session:
        if force:
            _force_truncate(session)
        else:
            _refuse_if_nonempty()

        sample = sample_skus(target_size=target_size)

        assert not sample[SKU_COL].duplicated().any(), "sampler produced duplicate sku_id — sampling bug"

        suppliers, sku_to_supplier = _extract_suppliers(sample)
        session.add_all(suppliers)
        session.flush()

        skus = _build_skus(sample, sku_to_supplier)
        session.add_all(skus)
        session.flush()

        inventory_rows = _build_inventory_rows(sample)
        session.add_all(inventory_rows)
        session.flush()  # a duplicate sku_id here raises IntegrityError, never silently upserts

        session.commit()

    print(f"\nSeeded {len(suppliers)} supplier(s), {len(skus)} SKU(s), {len(inventory_rows)} inventory row(s).")
    return sample


# ── §5 — Post-seed verification ─────────────────────────────────────────────

def verify(target_size: int = 5000) -> bool:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {label}" + (f"  -- {detail}" if detail else ""))

    with engine.connect() as conn:
        counts = {
            t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            for t in ("suppliers", "skus", "inventory_current")
        }
        check("1. suppliers/skus/inventory_current row counts > 0", all(n > 0 for n in counts.values()), str(counts))

        orphans = conn.execute(text(
            "SELECT COUNT(*) FROM inventory_current i "
            "LEFT JOIN skus s ON i.sku_id = s.sku_id WHERE s.sku_id IS NULL"
        )).scalar()
        check("2. every inventory_current.sku_id has a matching skus row", orphans == 0, f"orphans={orphans}")

        total = conn.execute(text("SELECT COUNT(*) FROM inventory_current")).scalar()
        distinct = conn.execute(text("SELECT COUNT(DISTINCT sku_id) FROM inventory_current")).scalar()
        check("3. exactly one inventory_current row per sku_id", total == distinct, f"total={total} distinct={distinct}")

        db_cols = {c["name"] for c in inspect(engine).get_columns("inventory_current")}
        check("4. no target column in inventory_current", db_cols.isdisjoint(TARGET_COLS), str(db_cols & TARGET_COLS))
        check("5. no engineered-feature column in inventory_current", db_cols.isdisjoint(ENGINEERED_EXCLUDE), str(db_cols & ENGINEERED_EXCLUDE))

        # 6. alarm-positive cross-check: the target isn't stored in the DB, so recover it by
        # regenerating the sample deterministically (same source + SEED=42) and joining sku_ids.
        sample = sample_skus(target_size=target_size)
        alarm_skus = set(sample.loc[sample[TARGET_COL] == "Yes", SKU_COL].astype(int).astype(str))
        if alarm_skus:
            placeholders = ",".join(f"'{s}'" for s in alarm_skus)
            n_alarm_in_db = conn.execute(text(
                f"SELECT COUNT(*) FROM inventory_current WHERE sku_id IN ({placeholders})"
            )).scalar()
        else:
            n_alarm_in_db = 0
        check("6. alarm-positive SKUs present in the seed (suppression demo requires > 0)", n_alarm_in_db > 0, f"count={n_alarm_in_db}")

        # 7. Spot-check seeded rows through features.clean_raw_row -- the only generic
        # value-level transform available on inventory_current's raw columns.
        # clean_raw_row is deliberately non-imputing (inventory_current is nullable by
        # design -- see models.py), so "no NaN/inf survives" would be the wrong bar: real
        # missingness is *supposed* to survive. The meaningful assertions are (a) no inf
        # survives -- that would be a real bug, (b) flag columns land in {0, 1, NaN}, never
        # a stray "Yes"/"No" token, and (c) a row actually drawn from the missing-values
        # stratum still carries real NaN afterward -- proving nothing was silently
        # fabricated to paper over missingness.
        col_names = ", ".join(spec.name for spec in RAW_SPECS)
        missing_mask = sample.isna().any(axis=1)
        missing_skus = list(sample.loc[missing_mask, SKU_COL].astype(int).astype(str).head(2))
        clean_skus = list(sample.loc[~missing_mask, SKU_COL].astype(int).astype(str).head(3))
        spot_ids = missing_skus + clean_skus
        placeholders = ",".join(f"'{s}'" for s in spot_ids)
        spot_rows = conn.execute(text(
            f"SELECT sku_id, {col_names} FROM inventory_current WHERE sku_id IN ({placeholders})"
        )).mappings().all()
        spot_df = pd.DataFrame([dict(r) for r in spot_rows])
        transformed = F.clean_raw_row(spot_df)
        numeric_block = transformed[[spec.name for spec in RAW_SPECS]]

        no_inf = not np.isinf(numeric_block.to_numpy(dtype=float)).any()
        flag_block = transformed[F.FLAG_COLS]
        flags_valid = bool((flag_block.isin([0.0, 1.0]) | flag_block.isna()).to_numpy().all())
        missing_rows_mask = transformed["sku_id"].isin(missing_skus)
        preserves_real_missingness = (
            not missing_skus or bool(numeric_block.loc[missing_rows_mask].isna().to_numpy().any())
        )

        check("7a. spot-check: no inf survives features.clean_raw_row", no_inf)
        check("7b. spot-check: flag columns land in {0, 1, NaN}", flags_valid)
        check(
            "7c. spot-check: genuine missingness preserved, not fabricated",
            preserves_real_missingness,
            f"missing-stratum rows sampled={len(missing_skus)}",
        )

    print("\nRealized stratum composition (8):")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed suppliers/skus/inventory_current from a stratified sample.")
    parser.add_argument("--force", action="store_true", help="truncate and reseed")
    parser.add_argument("--verify", action="store_true", help="run post-seed verification only, no seeding")
    parser.add_argument("--target-size", type=int, default=5000)
    args = parser.parse_args(argv)

    if args.verify:
        print("Post-seed verification:")
        return 0 if verify(target_size=args.target_size) else 1

    seed(force=args.force, target_size=args.target_size)

    print("\nPost-seed verification:")
    return 0 if verify(target_size=args.target_size) else 1


if __name__ == "__main__":
    sys.exit(main())
