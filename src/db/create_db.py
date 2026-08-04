"""
Entrypoint: create ``data/app.db`` with the seven-table schema (spec §7 step-4).

Usage:
    python -m src.db.create_db            # create; refuses if data/app.db already exists
    python -m src.db.create_db --force    # drop everything and recreate

Prints each created table with its column count and the derived ``inventory_current`` column
list annotated raw / excluded-engineered / excluded-target, so it can be reviewed against the
notebooks (spec §3.3 step-6).
"""

from __future__ import annotations

import argparse
import sys

from .base import DB_PATH, Base, engine
# Importing from .models executes the module, registering all seven models on Base.metadata.
from .models import INVENTORY_COLUMN_SPECS


def _print_tables() -> None:
    print("\nCreated tables:")
    for name in sorted(Base.metadata.tables.keys()):
        table = Base.metadata.tables[name]
        print(f"  {name:20s} {len(table.columns):>3d} columns")


def _print_inventory_columns() -> None:
    print("\ninventory_current — derived column classification (from the raw CSV header):")
    kind_order = {"raw": 0, "excluded-engineered": 1, "excluded-target": 2}
    for spec in sorted(INVENTORY_COLUMN_SPECS, key=lambda s: (kind_order.get(s.kind, 9), s.name)):
        sqltype = spec.sqltype.__name__.upper() if spec.sqltype else "-"
        typ = {"INTEGER": "INTEGER", "FLOAT": "REAL", "TEXT": "TEXT"}.get(sqltype, sqltype)
        print(f"  {spec.name:22s} {typ:8s} {spec.kind}")
    raw = [s for s in INVENTORY_COLUMN_SPECS if s.kind == "raw"]
    print(f"\n  -> {len(raw)} raw feature columns stored (+ 4 fixed: "
          f"sku_id, snapshot_at, raw_extra, updated_at)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the runtime SQLite database.")
    parser.add_argument(
        "--force", action="store_true",
        help="drop all tables and recreate (default: refuse if data/app.db exists)",
    )
    args = parser.parse_args(argv)

    if DB_PATH.exists() and not args.force:
        print(f"Refusing to run: {DB_PATH} already exists. Use --force to drop and recreate.")
        return 1

    if args.force:
        print(f"--force: dropping all tables in {DB_PATH}")
        Base.metadata.drop_all(engine)

    Base.metadata.create_all(engine)
    print(f"Schema created at {DB_PATH}")

    _print_tables()
    _print_inventory_columns()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
