"""tests/llm/test_draft_builder.py -- the most important file (llm_insertion_spec.md SS12).

Asserts facts, not fluency: every draft is checked against the literal grounding
text and the literal input values it was built from.
"""

from __future__ import annotations

import re

import pytest

from src.llm.draft_builder import (
    SHORT_DRAFT_WORD_THRESHOLD,
    UnknownAgentError,
    build_agent_draft,
    build_run_draft,
    is_short_draft,
)
from src.llm.grounding import AGENT_GROUNDING

# Only "meaningful" numbers -- decimals or 2+ digit integers -- count as facts to
# trace. Single stray digits (e.g. the "2" in "Agent 2" or "F2-optimized") are not
# claims and would make this check unworkably noisy.
_NUMBER_RE = re.compile(r"\d+\.\d+|\d{2,}")


def test_supplier_grade_d_mentions_critical_and_threshold():
    draft = build_agent_draft("supplier_auditor", {"supplier_risk": "D"})
    assert "Critical" in draft
    assert "[0.70, 1.00]" in draft


def test_risk_detector_alarm_mentions_0945_not_generic_midpoint():
    draft = build_agent_draft("risk_detector", {"backorder_prob": 0.97, "alarm_triggered": True})
    assert "0.945" in draft


def test_forecast_optimizer_risk_override_states_reason_explicitly():
    draft = build_agent_draft(
        "forecast_optimizer",
        {"correction_factor": 1.0, "recommendation": "HOLD", "risk_override_applied": True},
    )
    assert "never shrink the forecast" in draft


def test_every_number_in_draft_traces_to_field_values_or_grounding():
    field_values = {"backorder_prob": 0.87, "alarm_triggered": True}
    draft = build_agent_draft("risk_detector", field_values)

    grounding_text = str(AGENT_GROUNDING["risk_detector"])
    allowed_numbers = {f"{field_values['backorder_prob']:.1%}".rstrip("%")}

    for number in _NUMBER_RE.findall(draft):
        assert number in grounding_text or number in allowed_numbers, (
            f"Number {number!r} in draft is not traceable to field_values or "
            f"AGENT_GROUNDING: {draft!r}"
        )


def test_unknown_agent_raises():
    with pytest.raises(UnknownAgentError):
        build_agent_draft("agent_4_routing", {})


def test_short_draft_is_flagged_for_skip():
    short = "Agent 1 did not run."
    assert len(short.split()) < SHORT_DRAFT_WORD_THRESHOLD
    assert is_short_draft(short) is True

    # risk_detector's own draft is intentionally short now (humanized single-sentence
    # form -- see draft_builder.py's module docstring); use the multi-sentence
    # forecast_optimizer override scenario to exercise the "genuinely long" branch.
    long_draft = build_agent_draft(
        "forecast_optimizer",
        {"correction_factor": 1.0, "recommendation": "HOLD", "risk_override_applied": True},
    )
    assert is_short_draft(long_draft) is False


def test_build_run_draft_includes_suppression_note():
    predictions_row = {
        "demand_forecast": 100.0, "backorder_prob": 0.97, "alarm_triggered": 1,
        "urgency_score": 0.8, "correction_factor": 1.0, "supplier_risk": "D",
    }
    draft = build_run_draft("SKU-TEST-1", predictions_row, suppression_note="suppressed: alarm_triggered=1")
    assert "suppressed: alarm_triggered=1" in draft
    assert "never shrink the forecast" in draft
