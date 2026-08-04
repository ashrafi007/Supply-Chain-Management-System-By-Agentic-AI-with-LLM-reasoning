"""
SQLAlchemy 2.0 declarative models for the seven-table runtime schema (spec §3).

Six tables are typed statically. ``inventory_current`` is built dynamically from
:func:`src.db.columns.derive_inventory_columns` so its feature columns are *derived from the
raw CSV*, never hand-typed (spec §3.3 / acceptance criterion §8).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base
from .columns import ColumnSpec, TARGET_COLS, derive_inventory_columns


def _utcnow() -> datetime:
    """UTC now — spec §1 (never store local time), §7 step-3 (not datetime.utcnow)."""
    return datetime.now(timezone.utc)


# Agent 1 velocity-band taxonomy (Demand Predictor notebook §17). Ordered low -> high demand.
DEMAND_VELOCITY_BANDS: tuple[str, ...] = (
    "Dead", "Trickle", "Slow", "Steady", "Fast", "Surge",
)
_BANDS_SQL = ", ".join(f"'{b}'" for b in DEMAND_VELOCITY_BANDS)


# ── 3.1 suppliers ────────────────────────────────────────────────────────────
class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str | None] = mapped_column(Text, nullable=True)
    lead_time_avg_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


# ── 3.2 skus ─────────────────────────────────────────────────────────────────
class Sku(Base):
    __tablename__ = "skus"

    sku_id: Mapped[str] = mapped_column(Text, primary_key=True)
    # Nullable FK: a SKU with an unknown supplier must still be predictable (spec §3.2).
    supplier_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("suppliers.supplier_id"), nullable=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (Index("idx_skus_supplier", "supplier_id"),)


# ── 3.4 pipeline_runs ────────────────────────────────────────────────────────
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)  # UUID4 string
    sku_id: Mapped[str] = mapped_column(Text, ForeignKey("skus.sku_id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manifest_version: Mapped[str] = mapped_column(Text, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'success', 'failed')", name="ck_runs_status"
        ),
        Index("idx_runs_sku", "sku_id"),
        Index("idx_runs_started", "started_at"),
    )


# ── 3.5 predictions ──────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    run_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pipeline_runs.run_id"), primary_key=True
    )
    # Deliberate denormalization for the dashboard's hot "top N by urgency" query (spec §3.5).
    sku_id: Mapped[str] = mapped_column(Text, ForeignKey("skus.sku_id"), nullable=False)
    # Agent 1 — Demand Predictor (Models/Demand_predictor/agent1_h6_model.joblib).
    # demand_forecast = point forecast (predicted 6-month demand). demand_velocity_band and
    # stockout_risk are the operational hand-offs to Agent 3 (see notebook §17).
    demand_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)         # Agent 1
    demand_velocity_band: Mapped[str | None] = mapped_column(Text, nullable=True)       # Agent 1
    stockout_risk: Mapped[int | None] = mapped_column(Integer, nullable=True)           # Agent 1
    backorder_prob: Mapped[float | None] = mapped_column(Float, nullable=True)     # Agent 2
    alarm_triggered: Mapped[int | None] = mapped_column(Integer, nullable=True)    # Agent 2
    urgency_score: Mapped[float | None] = mapped_column(Float, nullable=True)      # Agent 3
    correction_factor: Mapped[float | None] = mapped_column(Float, nullable=True)  # Agent 5
    replenishment_qty: Mapped[float | None] = mapped_column(Float, nullable=True)  # Agent 4
    route: Mapped[str | None] = mapped_column(Text, nullable=True)                 # Agent 4
    supplier_risk: Mapped[str | None] = mapped_column(Text, nullable=True)         # Agent 6
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)        # LLM node
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        # Agent 1: velocity band must be a valid taxonomy value (notebook §17).
        CheckConstraint(
            f"demand_velocity_band IS NULL OR demand_velocity_band IN ({_BANDS_SQL})",
            name="ck_pred_velocity_band",
        ),
        # Agent 1: stockout_risk is a 0/1 flag.
        CheckConstraint(
            "stockout_risk IS NULL OR stockout_risk IN (0, 1)",
            name="ck_pred_stockout_binary",
        ),
        CheckConstraint(
            "backorder_prob IS NULL OR (backorder_prob >= 0 AND backorder_prob <= 1)",
            name="ck_pred_backorder_prob_range",
        ),
        CheckConstraint(
            "alarm_triggered IS NULL OR alarm_triggered IN (0, 1)",
            name="ck_pred_alarm_binary",
        ),
        CheckConstraint(
            "correction_factor IS NULL OR correction_factor > 0",
            name="ck_pred_correction_positive",
        ),
        # Load-bearing: enforces the Agent 3 clip at the storage layer (spec §3.5).
        CheckConstraint(
            "urgency_score IS NULL OR (urgency_score >= 0 AND urgency_score <= 1)",
            name="ck_pred_urgency_range",
        ),
        # Load-bearing: the risk-suppression invariant — when Agent 2 raises an alarm,
        # Agent 5 must not reduce the forecast (correction_factor must be 1.0). Spec §3.5.
        CheckConstraint(
            "alarm_triggered IS NULL OR alarm_triggered = 0 OR correction_factor = 1.0",
            name="ck_pred_suppression_invariant",
        ),
        Index("idx_pred_sku", "sku_id"),
        Index("idx_pred_alarm", "alarm_triggered"),
        # "Show all Surge / high-velocity SKUs" — the Agent-1 analogue of idx_pred_alarm.
        Index("idx_pred_velocity_band", "demand_velocity_band"),
    )


# Descending index on urgency_score — defined against the column so DESC is emitted (spec §3.5).
Index("idx_pred_urgency", Prediction.urgency_score.desc())


# ── 3.6 agent_traces ─────────────────────────────────────────────────────────
class AgentTrace(Base):
    __tablename__ = "agent_traces"

    trace_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, ForeignKey("pipeline_runs.run_id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        CheckConstraint(
            "status IN ('success', 'skipped', 'failed')", name="ck_traces_status"
        ),
        Index("idx_traces_run", "run_id"),
    )


# ── 3.7 forecast_actuals ─────────────────────────────────────────────────────
class ForecastActual(Base):
    __tablename__ = "forecast_actuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku_id: Mapped[str] = mapped_column(Text, ForeignKey("skus.sku_id"), nullable=False)
    period: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. '2026-08'
    human_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    agent_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_demand: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("sku_id", "period", name="uq_actuals_sku_period"),
        Index("idx_actuals_sku_period", "sku_id", "period"),
    )


# ── 3.3 inventory_current (derived) ──────────────────────────────────────────
# The target column name, exported so verify_db can assert it never becomes a column.
TARGET_COLUMN_NAMES: frozenset[str] = TARGET_COLS

# Full derived classification (raw / excluded-*), exported for create_db's annotated printout.
INVENTORY_COLUMN_SPECS: list[ColumnSpec] = derive_inventory_columns()

_SQLTYPE_MAP = {Integer: Integer, Float: Float, Text: Text}


def _build_inventory_current() -> type[Base]:
    """Assemble the InventoryCurrent mapped class from the derived raw column specs.

    Fixed columns (spec §3.3): sku_id (PK + FK -> skus.sku_id, enforcing one-row-per-SKU),
    snapshot_at, raw_extra (JSON escape hatch), updated_at. Every derived raw column is
    emitted with its inferred type and is nullable (value-level cleaning happens in
    features.py at serve time, not via NOT NULL — spec §5).
    """
    namespace: dict = {
        "__tablename__": "inventory_current",
        "sku_id": mapped_column(
            Text, ForeignKey("skus.sku_id"), primary_key=True
        ),
        "snapshot_at": mapped_column(DateTime, nullable=False, default=_utcnow),
        "raw_extra": mapped_column(JSON, nullable=True),
        "updated_at": mapped_column(
            DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
        ),
        "__table_args__": (Index("idx_inventory_snapshot", "snapshot_at"),),
    }

    for spec in INVENTORY_COLUMN_SPECS:
        if spec.kind != "raw":
            continue  # excluded-target / excluded-engineered are never stored
        if spec.name in namespace:
            raise ValueError(f"Derived column {spec.name!r} collides with a fixed column")
        namespace[spec.name] = mapped_column(spec.sqltype, nullable=True)

    return type("InventoryCurrent", (Base,), namespace)


InventoryCurrent = _build_inventory_current()
