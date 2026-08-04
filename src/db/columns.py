"""
Derive the ``inventory_current`` feature columns from the raw CSV — never hand-typed.

This module satisfies the spec's core requirement (.CLAUDE/database_spec.md §3.3): the schema
of ``inventory_current`` is *generated* from the raw CSV header, cross-checked against the
known raw/engineered/target classification, rather than being typed by hand or copied from
``processed_dataset.pkl``.

Classification (spec §3.3, steps 1–5):
  * ``sku``                -> the ``sku_id`` primary key / FK (handled in models.py, not a feature).
  * ``went_on_backorder``  -> excluded-target (leakage guard, spec §3.3.4).
  * engineered features    -> excluded-engineered: computed at serve time by features.py, never
                              stored (they are not present in the raw CSV, so this is defensive).
  * everything else        -> a typed raw column: INTEGER binary flags, REAL numeric, TEXT other.

Types are inferred from the *data* (a sampled slice of the CSV), not a hardcoded list, so the
schema tracks the CSV. The six Yes/No flag columns resolve to INTEGER because the seeder /
features.py encode Yes/No -> 0/1 as a value-level cleaning step (spec §5); the DB stores the
cleaned 0/1 form.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Float, Integer, Text
from sqlalchemy.types import TypeEngine


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# The raw CSV is the source of truth for inputs (spec §1). The spec's canonical path is
# data/raw/rsm.csv; in this repo the uploaded dataset lives at Dataset/RSM_Dataset.csv.
# Overridable via the RSM_RAW_CSV env var.
RAW_CSV_PATH: Path = Path(
    os.environ.get("RSM_RAW_CSV", PROJECT_ROOT / "Dataset" / "RSM_Dataset.csv")
)

# The SKU identifier column -> becomes the sku_id PK/FK on inventory_current.
ID_COL: str = "sku"

# The prediction target. Excluded unconditionally — a target column in the serving table is a
# leakage accident waiting to happen (spec §3.3.4).
TARGET_COLS: frozenset[str] = frozenset({"went_on_backorder"})

# Engineered features computed from other columns at serve time by features.py — never stored
# (spec §3.3.3). Not present in the raw CSV header, so this set is defensive: if a future CSV
# ever carries a precomputed engineered column, we still refuse to store it.
ENGINEERED_EXCLUDE: frozenset[str] = frozenset(
    {
        # level-1 engineered (were precomputed in the old Processed_Dataset.pkl)
        "inv_velocity", "safety_gap", "sales_trend", "perf_gap",
        # higher-order engineered features derived by the inference tools
        "coverage_ratio", "transit_coverage", "inventory_health_net", "sales_spike_ratio",
        "months_of_stock_left", "backorder_pressure", "supplier_risk_score",
        "forecast_accuracy_gap", "inv_depletion_rate", "lead_time_volatility",
        "safety_stock_urgency", "performance_trend", "demand_forecast_ratio",
        "replenishment_gap", "critical_state",
    }
)

# How many data rows to sample when inferring a column's type.
_SAMPLE_ROWS: int = 1000

# Textual tokens (case-insensitive, stripped) that mark a Yes/No-style binary flag column.
_BOOL_TOKENS: frozenset[str] = frozenset({"yes", "no", "true", "false"})
_BLANK_TOKENS: frozenset[str] = frozenset({"", "na", "nan", "null", "none"})


@dataclass(frozen=True)
class ColumnSpec:
    """A derived column: its name, the SQLAlchemy type to emit, and its classification."""

    name: str
    sqltype: type[TypeEngine] | None  # None for excluded columns
    kind: str  # "raw" | "excluded-target" | "excluded-engineered"


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _classify_type(values: list[str]) -> type[TypeEngine]:
    """Infer a SQLAlchemy type from sampled non-blank string values."""
    non_blank = [v for v in values if v.strip().lower() not in _BLANK_TOKENS]
    if not non_blank:
        # No signal at all -> safest is TEXT; will hold the raw_extra escape values.
        return Text

    if all(v.strip().lower() in _BOOL_TOKENS for v in non_blank):
        return Integer  # Yes/No flag, stored cleaned as 0/1

    if all(_looks_numeric(v) for v in non_blank):
        # A genuinely numeric-encoded 0/1 flag (both values seen) is INTEGER; anything else
        # (including a count column that happens to be all-zero in the sample) is REAL.
        distinct = {float(v) for v in non_blank}
        if distinct <= {0.0, 1.0} and len(distinct) >= 2:
            return Integer
        return Float  # REAL numeric

    return Text  # categorical / free text


def derive_inventory_columns(csv_path: Path | None = None) -> list[ColumnSpec]:
    """
    Read the raw CSV header + a sample of rows and return the classification for every
    non-identifier column, in header order.

    The identifier column (``sku``) is intentionally omitted — it is the ``sku_id`` PK/FK
    added separately in models.py. Raises FileNotFoundError if the CSV is missing, because
    the schema must be derived, not guessed.
    """
    path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {path}. inventory_current columns are derived from the raw "
            f"CSV header (spec §3.3); set the RSM_RAW_CSV env var to point at it."
        )

    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples: dict[str, list[str]] = {col: [] for col in header}
        for i, row in enumerate(reader):
            if i >= _SAMPLE_ROWS:
                break
            for col, val in zip(header, row):
                samples[col].append(val)

    specs: list[ColumnSpec] = []
    for col in header:
        if col == ID_COL:
            continue  # -> sku_id PK/FK, not a feature column
        if col in TARGET_COLS:
            specs.append(ColumnSpec(col, None, "excluded-target"))
        elif col in ENGINEERED_EXCLUDE:
            specs.append(ColumnSpec(col, None, "excluded-engineered"))
        else:
            specs.append(ColumnSpec(col, _classify_type(samples[col]), "raw"))
    return specs


def raw_column_specs(csv_path: Path | None = None) -> list[ColumnSpec]:
    """Just the columns that get emitted as typed storage on inventory_current."""
    return [s for s in derive_inventory_columns(csv_path) if s.kind == "raw"]
