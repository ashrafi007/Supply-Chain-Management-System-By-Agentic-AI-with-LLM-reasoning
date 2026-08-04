"""Re-exports the frozen PipelineState contract from src.pipeline_state; defines the graph-internal-only GraphState extension for the three trace convention keys."""

from __future__ import annotations

from typing import NotRequired

from src.pipeline_state import (  # noqa: F401
    AgentEvent,
    PipelineExecutor,
    PipelineResult,
    PipelineState,
    PREDICTION_FIELDS,
)


class GraphState(PipelineState):
    """
    LangGraph-internal extension of PipelineState. NEVER used as the public
    contract type (repository layer, tests, PipelineResult all use the plain
    PipelineState from src.pipeline_state) -- only graph.py's StateGraph is
    built against this, so the three node-convention keys (_latency_ms,
    _skipped, _note) have a declared channel. tracing.py strips them back out
    before anything becomes AgentEvent.output; they never leak into
    PipelineState/predictions.
    """

    _latency_ms: NotRequired[int | None]
    _skipped: NotRequired[bool]
    _note: NotRequired[str | None]
