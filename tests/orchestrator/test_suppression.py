"""THE test: alarm_triggered=1 forces correction_factor=1.0, agent_5 skipped -- the paper's novelty claim."""

from __future__ import annotations

from src.orchestrator.nodes import agent_5_forecast_opt


def test_node_level_suppression_is_isolated_from_the_rest_of_the_graph():
    """agent_5_forecast_opt.node() short-circuits on alarm_triggered=1 without
    touching the real model at all -- provable with zero LangGraph machinery."""
    state = {"sku_id": "X", "alarm_triggered": 1, "trace": []}
    delta = agent_5_forecast_opt.node(state)

    assert delta["correction_factor"] == 1.0
    assert delta["_skipped"] is True
    assert "suppressed" in delta["_note"]
    assert delta["trace"][-1]["status"] == "skipped"


def test_full_graph_suppression_end_to_end(monkeypatch, sample_state):
    """Feeding alarm_triggered=1 through the compiled graph yields
    correction_factor==1.0 and an AgentEvent for forecast_optimizer with
    status=skipped and a note containing 'suppressed'. The risk node is
    monkeypatched to force alarm_triggered=1, since the real model's high
    threshold (0.945) won't reliably trigger on synthetic data -- this test's
    job is proving the graph-level suppression wiring, not risk model accuracy.
    """
    import src.orchestrator.nodes.agent_2_risk as agent_2_risk

    def fake_risk_node(state):
        trace = list(state.get("trace", []))
        trace.append({"tool": "risk_detector", "status": "ok", "forced_for_test": True})
        return {"backorder_prob": 0.99, "alarm_triggered": 1, "trace": trace}

    monkeypatch.setattr(agent_2_risk, "node", fake_risk_node)

    from src.orchestrator.executor import LangGraphExecutor
    executor = LangGraphExecutor()
    result = executor.invoke(sample_state)

    assert result["final_state"]["correction_factor"] == 1.0

    events_by_agent = {e["agent_name"]: e for e in result["agent_events"]}
    forecast_event = events_by_agent["forecast_optimizer"]
    assert forecast_event["status"] == "skipped"
    assert "suppressed" in forecast_event["note"]
