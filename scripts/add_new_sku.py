"""
scripts/add_new_sku.py -- interactive onboarding for a brand-new SKU.

Serially prompts for the SKU's identity and its raw ("mother") feature values,
inserts it into `skus`/`inventory_current`, adds it to `order_queue`, then shows
the engineered features (safety_gap, perf_gap, inv_velocity, sales_trend) the
system computes on its own -- automatically, at pipeline-run time -- and
optionally runs the real pipeline immediately so you can see a full prediction
come out the other end.

    python -m scripts.add_new_sku
    python -m scripts.add_new_sku --sku-id SKU-DEMO-001 --due-in-days 3 --run-now

Non-interactive raw feature values can be supplied via --set col=value (repeatable);
anything not supplied is prompted for interactively (blank = NULL, nullable by design).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import features as F

_FLAG_PROMPT_HINT = "Yes/No"
_NUMERIC_PROMPT_HINT = "number, blank = unknown"


def _prompt_raw_features(preset: dict) -> dict:
    from src.repository.sku_ingestion import RAW_FEATURE_COLUMNS

    values: dict = {}
    print("\n-- Raw ('mother') features -- press Enter to leave a value unset (NULL) --\n")
    for col in RAW_FEATURE_COLUMNS:
        if col in preset:
            values[col] = preset[col]
            continue
        hint = _FLAG_PROMPT_HINT if col in F.FLAG_COLS else _NUMERIC_PROMPT_HINT
        raw = input(f"  {col} [{hint}]: ").strip()
        values[col] = raw if raw != "" else None
    return values


def _parse_set_args(pairs: list[str]) -> dict:
    preset = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects col=value, got {pair!r}")
        col, val = pair.split("=", 1)
        preset[col.strip()] = val.strip()
    return preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sku-id", help="new SKU identifier (prompted if omitted)")
    parser.add_argument("--supplier-id", default=None, help="existing supplier_id, optional")
    parser.add_argument("--description", default=None, help="free-text description, optional")
    parser.add_argument("--due-in-days", type=int, default=7, help="queue due_date, N days from today (default 7)")
    parser.add_argument("--set", action="append", default=[], metavar="col=value", help="preset a raw feature value non-interactively; repeatable")
    parser.add_argument("--run-now", action="store_true", help="run the real pipeline immediately after onboarding")
    args = parser.parse_args(argv)

    from src.db.base import SessionLocal
    from src.queue import ingestion_service
    from src.repository.sku_ingestion import add_new_sku

    sku_id = args.sku_id or input("New sku_id: ").strip()
    if not sku_id:
        print("sku_id is required.")
        return 1

    preset = _parse_set_args(args.set)
    raw_features = _prompt_raw_features(preset)

    session = SessionLocal()
    try:
        print(f"\n-- Adding {sku_id!r} to skus + inventory_current --")
        add_new_sku(
            session,
            sku_id,
            raw_features,
            supplier_id=args.supplier_id,
            description=args.description,
        )
        print("  done.")

        due_date = date.today() + timedelta(days=args.due_in_days)
        print(f"\n-- Adding {sku_id!r} to order_queue (due {due_date}, source='manual_add') --")
        ingestion_service.enqueue_new_sku(session, sku_id, due_date=due_date, source="manual_add")
        print("  done.")

        print("\n-- Engineered features (computed automatically -- never stored, recomputed every pipeline run) --")
        from src.db.models import InventoryCurrent
        from src.repository.sku_ingestion import RAW_FEATURE_COLUMNS

        inv = session.get(InventoryCurrent, sku_id)
        cleaned_raw = {col: getattr(inv, col) for col in RAW_FEATURE_COLUMNS}
        engineered = F.fill_engineered_columns(cleaned_raw)
        for key in ("safety_gap", "perf_gap", "inv_velocity", "sales_trend", "went_on_backorder"):
            print(f"  {key}: {engineered[key]!r}")

        if args.run_now:
            from src.orchestrator.executor import LangGraphExecutor
            from src.repository.pipeline_service import run_pipeline_for_sku

            print(f"\n-- Running the real pipeline for {sku_id!r} now --")
            run_id = run_pipeline_for_sku(session, sku_id, LangGraphExecutor())
            print(f"  run_id: {run_id}")
            print(f"  (see: python -m scripts.run_single_prediction --sku-id {sku_id})")
    except ValueError as exc:
        print(f"\nFailed: {exc}")
        return 1
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
