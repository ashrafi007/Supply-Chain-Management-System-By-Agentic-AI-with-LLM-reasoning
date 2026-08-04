"""LangGraphExecutor().invoke() matches the PipelineExecutor/PipelineResult/AgentEvent shape exactly."""

from __future__ import annotations

from tests.fixtures.stub_executor import _AGENT_NAMES


def test_invoke_returns_exact_pipeline_result_shape(sample_state):
    from src.orchestrator.executor import LangGraphExecutor

    result = LangGraphExecutor().invoke(sample_state)

    assert set(result.keys()) == {"final_state", "agent_events", "manifest_version"}
    assert isinstance(result["manifest_version"], str) and result["manifest_version"]
    assert isinstance(result["final_state"], dict)


def test_agent_events_shape_and_sequence(sample_state):
    from src.orchestrator.executor import LangGraphExecutor

    result = LangGraphExecutor().invoke(sample_state)
    events = result["agent_events"]

    assert len(events) == 5
    for i, event in enumerate(events, start=1):
        assert set(event.keys()) == {"agent_name", "sequence", "status", "latency_ms", "output", "note"}
        assert event["sequence"] == i
        assert event["status"] in ("success", "skipped", "failed")

    assert [e["agent_name"] for e in events] == list(_AGENT_NAMES)


def test_manifest_version_is_stable_across_invocations(sample_state):
    from src.orchestrator.executor import LangGraphExecutor

    executor = LangGraphExecutor()
    r1 = executor.invoke(sample_state)
    r2 = executor.invoke(dict(sample_state, sku_id="TEST-SKU-0002"))
    assert r1["manifest_version"] == r2["manifest_version"]
