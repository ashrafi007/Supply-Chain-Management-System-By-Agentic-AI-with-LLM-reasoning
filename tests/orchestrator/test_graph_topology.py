"""Graph topology: 5 nodes + START/END, edge order, no routing anywhere; should_suppress unit-tested standalone."""

from __future__ import annotations

from src.orchestrator import edges
from src.orchestrator.graph import NODE_TO_AGENT_NAME, build_graph


def test_should_suppress_is_pure_and_needs_no_langgraph():
    assert edges.should_suppress({"alarm_triggered": 1}) is True
    assert edges.should_suppress({"alarm_triggered": 0}) is False
    assert edges.should_suppress({}) is False
    assert edges.should_suppress({"alarm_triggered": None}) is False


def test_graph_has_exactly_five_agent_nodes_no_routing():
    g = build_graph()
    nodes = set(g.get_graph().nodes)
    expected = {"__start__", "__end__", "demand", "risk", "rebalancer", "forecast_opt", "auditor"}
    assert nodes == expected
    assert "routing" not in nodes


def test_graph_edges_match_spec_order():
    g = build_graph()
    edge_pairs = {(e.source, e.target) for e in g.get_graph().edges}
    assert edge_pairs == {
        ("__start__", "demand"),
        ("demand", "risk"),
        ("risk", "rebalancer"),
        ("rebalancer", "forecast_opt"),
        ("forecast_opt", "auditor"),
        ("auditor", "__end__"),
    }


def test_a3_rebalancer_strictly_after_a2_risk():
    g = build_graph()
    edge_pairs = {(e.source, e.target) for e in g.get_graph().edges}
    assert ("risk", "rebalancer") in edge_pairs


def test_node_to_agent_name_matches_stub_executor_names():
    from tests.fixtures.stub_executor import _AGENT_NAMES
    assert set(NODE_TO_AGENT_NAME.values()) == set(_AGENT_NAMES)
