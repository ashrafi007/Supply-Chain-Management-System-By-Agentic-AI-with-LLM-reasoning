"""Converts the graph's per-node execution deltas into list[AgentEvent] -- the only place this translation happens."""

from __future__ import annotations

from src.orchestrator.graph import NODE_TO_AGENT_NAME
from src.orchestrator.state import AgentEvent

_CONVENTION_KEYS = ("_latency_ms", "_skipped", "_note")


def build_agent_events(updates: list[tuple[str, dict]]) -> list[AgentEvent]:
    """
    ``updates`` is [(node_key, delta), ...] in true execution order, as
    collected from graph.stream(state, stream_mode="updates").

    output is this node's own trace-entry dict (the last element it appended
    to state["trace"]) -- not the raw field delta, not the full accumulated
    trace list. That entry already carries the rich per-tool detail (recommended_qty,
    supplier_grade context, bias_severity, etc.) that never becomes a top-level
    PipelineState key.
    """
    events: list[AgentEvent] = []
    for i, (node_key, delta) in enumerate(updates, start=1):
        agent_name = NODE_TO_AGENT_NAME[node_key]
        trace_list = delta.get("trace") or []
        own_entry = trace_list[-1] if trace_list else {}

        if delta.get("_skipped"):
            status = "skipped"
        elif own_entry.get("status") == "failed":
            status = "failed"
        else:
            status = "success"

        events.append(AgentEvent(
            agent_name=agent_name,
            sequence=i,
            status=status,
            latency_ms=delta.get("_latency_ms"),
            output=own_entry,
            note=delta.get("_note"),
        ))
    return events
