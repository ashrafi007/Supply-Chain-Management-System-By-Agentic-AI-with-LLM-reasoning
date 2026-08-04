"""StateGraph assembly: 5 linear agent nodes (Routing/Agent 4 cancelled), add_edge, compile()."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.orchestrator.nodes import (
    agent_1_demand,
    agent_2_risk,
    agent_3_rebalancer,
    agent_5_forecast_opt,
    agent_6_auditor,
)
from src.orchestrator.state import GraphState

# Graph-node-key -> canonical AgentEvent.agent_name. Values must match
# tests/fixtures/stub_executor.py's _AGENT_NAMES exactly.
NODE_TO_AGENT_NAME: dict[str, str] = {
    "demand": "demand_predictor",
    "risk": "risk_detector",
    "rebalancer": "inventory_rebalancer",
    "forecast_opt": "forecast_optimizer",
    "auditor": "supplier_auditor",
}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("demand", agent_1_demand.node)
    g.add_node("risk", agent_2_risk.node)
    g.add_node("rebalancer", agent_3_rebalancer.node)
    g.add_node("forecast_opt", agent_5_forecast_opt.node)
    g.add_node("auditor", agent_6_auditor.node)

    g.add_edge(START, "demand")
    g.add_edge("demand", "risk")
    g.add_edge("risk", "rebalancer")
    g.add_edge("rebalancer", "forecast_opt")
    g.add_edge("forecast_opt", "auditor")  # routing cancelled -- direct edge
    g.add_edge("auditor", END)

    return g.compile()
