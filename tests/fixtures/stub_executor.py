"""
Configurable PipelineExecutor double so the repository layer's tests don't depend
on the (not-yet-built) LangGraph orchestrator.
"""

from __future__ import annotations

from src.pipeline_state import AgentEvent, PipelineResult, PipelineState

_AGENT_NAMES = (
    "demand_predictor",
    "risk_detector",
    "inventory_rebalancer",
    "forecast_optimizer",
    "supplier_auditor",
)


def default_success_fields() -> dict:
    """A fully valid, CHECK-satisfying set of prediction fields — the baseline
    every scenario starts from before field_overrides is applied."""
    return dict(
        demand_forecast=120.5,
        demand_velocity_band="Steady",
        stockout_risk=0,
        backorder_prob=0.12,
        alarm_triggered=0,
        urgency_score=0.3,
        correction_factor=1.0,
        replenishment_qty=50.0,
        route="standard",
        supplier_risk="low",
        recommendation="reorder 50 units",
    )


def default_events() -> list[AgentEvent]:
    return [
        AgentEvent(
            agent_name=name,
            sequence=i,
            status="success",
            latency_ms=10,
            output={},
            note=None,
        )
        for i, name in enumerate(_AGENT_NAMES, start=1)
    ]


class StubExecutor:
    """Configurable PipelineExecutor. Construction-time knobs cover every
    scenario in repository_layer_spec.md §8 without per-test subclasses."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        field_overrides: dict | None = None,
        agent_events: list[AgentEvent] | None = None,
        manifest_version: str = "test-manifest-v1",
    ):
        self._raises = raises
        self._field_overrides = field_overrides or {}
        self._agent_events = agent_events if agent_events is not None else default_events()
        self._manifest_version = manifest_version

    def invoke(self, state: PipelineState) -> PipelineResult:
        if self._raises is not None:
            raise self._raises

        final_state = {**state, **default_success_fields(), **self._field_overrides}
        return PipelineResult(
            final_state=final_state,
            agent_events=self._agent_events,
            manifest_version=self._manifest_version,
        )


class KeyedStubExecutor:
    """Dispatches by sku_id so run_pipeline_for_skus (one executor, many SKUs)
    can exercise per-SKU behavior in a single test."""

    def __init__(self, per_sku: dict[str, StubExecutor]):
        self._per_sku = per_sku

    def invoke(self, state: PipelineState) -> PipelineResult:
        return self._per_sku[state["sku_id"]].invoke(state)
