"""LangGraph node adapter for the Demand Predictor agent. No model logic lives here -- calls the already-verified DemandPredictorTool.run()."""

from __future__ import annotations

import time

from src.orchestrator import manifest
from src.orchestrator.state import GraphState

_tool = manifest.DEMAND_PREDICTOR_TOOL


def node(state: GraphState) -> dict:
    start = time.monotonic()
    delta = _tool.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
