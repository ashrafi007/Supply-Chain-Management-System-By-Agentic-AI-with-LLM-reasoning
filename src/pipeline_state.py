"""
Shared contract between the repository layer and the (future) LangGraph orchestrator.

Neither ``src.db`` nor ``src.repository`` may be imported here — this module must stay
a leaf so both sides can depend on it without depending on each other.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

# Fields that map 1:1 onto `predictions` columns, by name, with zero drift.
# Single source of truth: if a field is added to PipelineState, add it here and to
# the `predictions` schema in the same change (repository_layer_spec.md §5.2).
PREDICTION_FIELDS: tuple[str, ...] = (
    "demand_forecast",
    "demand_velocity_band",
    "stockout_risk",
    "backorder_prob",
    "alarm_triggered",
    "urgency_score",
    "correction_factor",
    "replenishment_qty",
    "route",
    "supplier_risk",
    "recommendation",
)


class PipelineState(TypedDict):
    sku_id: str
    raw_features: dict

    demand_forecast: float | None          # Agent 1
    demand_velocity_band: str | None       # Agent 1
    stockout_risk: int | None              # Agent 1
    backorder_prob: float | None           # Agent 2
    alarm_triggered: int | None            # Agent 2
    urgency_score: float | None            # Agent 3
    correction_factor: float | None        # Agent 5
    replenishment_qty: float | None        # Agent 4
    route: str | None                      # Agent 4
    supplier_risk: str | None              # Agent 6
    recommendation: str | None             # LLM node

    trace: list
    errors: list


class AgentEvent(TypedDict):
    agent_name: str
    sequence: int
    status: Literal["success", "skipped", "failed"]
    latency_ms: int | None
    output: dict
    note: str | None


class PipelineResult(TypedDict):
    final_state: PipelineState
    agent_events: list[AgentEvent]
    manifest_version: str


class PipelineExecutor(Protocol):
    def invoke(self, state: PipelineState) -> PipelineResult: ...
