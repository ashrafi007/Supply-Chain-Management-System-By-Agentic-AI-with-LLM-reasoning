"""tests/llm/test_grounding.py (llm_insertion_spec.md SS12)."""

from __future__ import annotations

from src.llm.grounding import AGENT_GROUNDING
from src.orchestrator.graph import NODE_TO_AGENT_NAME


def test_every_real_agent_name_has_a_grounding_entry():
    real_agent_names = set(NODE_TO_AGENT_NAME.values())
    assert real_agent_names <= AGENT_GROUNDING.keys()


def test_no_agent_4_entry_exists():
    assert "agent_4_routing" not in AGENT_GROUNDING
    assert not any("4" in key for key in AGENT_GROUNDING)


def test_forecast_optimizer_categorical_value_sets_match_documented_meanings():
    bias_severity = AGENT_GROUNDING["forecast_optimizer"]["bias_severity"]["values"]
    assert set(bias_severity) == {"NONE", "MILD", "SEVERE"}

    recommendation = AGENT_GROUNDING["forecast_optimizer"]["recommendation"]["values"]
    assert set(recommendation) == {"REDUCE_PLAN", "INCREASE_PLAN", "HOLD"}


def test_supplier_auditor_categorical_value_sets_match_documented_meanings():
    supplier_risk = AGENT_GROUNDING["supplier_auditor"]["supplier_risk"]["values"]
    assert set(supplier_risk) == {"A", "B", "C", "D"}

    trigger_reason = AGENT_GROUNDING["supplier_auditor"]["trigger_reason"]["values"]
    assert set(trigger_reason) == {"MODEL+RULE", "MODEL", "RULE", "NONE"}
