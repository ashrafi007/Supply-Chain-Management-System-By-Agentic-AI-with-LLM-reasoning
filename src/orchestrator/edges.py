"""Pure routing/suppression predicates -- no I/O, no LangGraph dependency, unit-testable standalone.

Routing (Agent 4) is permanently out of scope (cancelled), so the spec's
route_after_risk/route_before_forecast_opt helpers are intentionally omitted --
the suppression branch is implemented in-node in agent_5_forecast_opt.py.
"""

from __future__ import annotations

from src.orchestrator.state import PipelineState


def should_suppress(state: PipelineState) -> bool:
    """Returns True if Agent 5's corrector must be bypassed."""
    return state.get("alarm_triggered") == 1
