"""LangGraphExecutor -- implements PipelineExecutor, wraps graph.py's compiled app."""

from __future__ import annotations

import features as F
from src.orchestrator import graph, manifest, tracing
from src.orchestrator.state import PipelineExecutor, PipelineResult, PipelineState

_CONVENTION_KEYS = ("_latency_ms", "_skipped", "_note")


class LangGraphExecutor(PipelineExecutor):
    def __init__(self):
        self._graph = graph.build_graph()

    def invoke(self, state: PipelineState) -> PipelineResult:
        working_state = dict(state)
        working_state["raw_features"] = F.fill_engineered_columns(working_state["raw_features"])

        updates: list[tuple[str, dict]] = []
        final_state = dict(working_state)
        for chunk in self._graph.stream(working_state, stream_mode="updates"):
            (node_key, delta), = chunk.items()
            updates.append((node_key, delta))
            final_state.update({k: v for k, v in delta.items() if k not in _CONVENTION_KEYS})

        return PipelineResult(
            final_state=final_state,
            agent_events=tracing.build_agent_events(updates),
            manifest_version=manifest.MANIFEST_VERSION,
        )
