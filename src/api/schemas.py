"""Pydantic response models for the read API. Separate from inference_tools/schemas.py
(agent I/O contracts) -- these describe what the frontend receives over HTTP."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    run_id: str
    sku_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    latency_ms: int | None

    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    demand_forecast: float | None
    demand_velocity_band: str | None
    stockout_risk: int | None
    backorder_prob: float | None
    alarm_triggered: int | None
    urgency_score: float | None
    correction_factor: float | None
    supplier_risk: str | None
    recommendation: str | None

    model_config = {"from_attributes": True}


class AgentTraceOut(BaseModel):
    agent_name: str
    sequence: int
    status: str
    latency_ms: int | None
    output: dict | None
    note: str | None

    model_config = {"from_attributes": True}


class RunDetail(BaseModel):
    run_id: str
    sku_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    latency_ms: int | None
    manifest_version: str
    error: str | None
    prediction: PredictionOut | None
    traces: list[AgentTraceOut]


# ── order_queue ──────────────────────────────────────────────────────────────

class QueueEntryOut(BaseModel):
    sku_id: str
    status: str
    queued_at: datetime
    due_date: date
    last_run_id: str | None
    last_evaluated_at: datetime | None
    source: str

    model_config = {"from_attributes": True}


class EnqueueRequest(BaseModel):
    sku_id: str
    due_date: date
    source: Literal["scheduled", "manual_add"] = "manual_add"


class SweepResult(BaseModel):
    as_of: date
    expired_count: int
    evaluated_sku_ids: list[str]
    run_ids: list[str]
    explanations: dict[str, dict] = Field(default_factory=dict)


# ── suppliers ────────────────────────────────────────────────────────────────

class SupplierOut(BaseModel):
    supplier_id: str
    name: str
    country: str | None
    lead_time_avg_days: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── skus ─────────────────────────────────────────────────────────────────────

class SkuOut(BaseModel):
    sku_id: str
    supplier_id: str | None
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SkuDetail(SkuOut):
    raw_features: dict


class NewSkuRequest(BaseModel):
    sku_id: str
    raw_features: dict = Field(default_factory=dict)
    supplier_id: str | None = None
    description: str | None = None
    due_in_days: int = 7
    run_now: bool = False


class NewSkuResult(BaseModel):
    sku_id: str
    queued: QueueEntryOut
    run_id: str | None = None


# ── dashboard stats ──────────────────────────────────────────────────────────

class StatsOut(BaseModel):
    """Counts are computed from each queued SKU's most recent pipeline_runs/predictions
    row (via order_queue.last_run_id), not every historical run."""

    total_skus: int
    queue_counts_by_status: dict[str, int]
    supplier_grade_distribution: dict[str, int]
    high_risk_sku_count: int
