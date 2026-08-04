"""Pydantic response models for the read API. Separate from inference_tools/schemas.py
(agent I/O contracts) -- these describe what the frontend receives over HTTP."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


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
