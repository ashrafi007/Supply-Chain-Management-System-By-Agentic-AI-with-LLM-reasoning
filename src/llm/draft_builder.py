"""
Deterministic, fully-grounded draft assembly (llm_insertion_spec.md SS6).

This module IS the explanation. The LLM in client.py only rewords its output --
get this right and the rest is safe by construction. Rule enforced here, checked
in tests/llm/test_draft_builder.py: every number or category that appears in a
draft must come from the caller-supplied field_values (real run data) or from
AGENT_GROUNDING (the real, documented meaning) -- never free-text commentary that
isn't traceable to one of those two sources.
"""

from __future__ import annotations

from src.llm.grounding import AGENT_GROUNDING

# A draft under this many words gains little from LLM rephrasing -- skip the
# network round-trip entirely (llm_insertion_spec.md SS2 "Skip-polish rule").
SHORT_DRAFT_WORD_THRESHOLD = 40


class UnknownAgentError(ValueError):
    """Raised when an agent_name has no entry in AGENT_GROUNDING. Callers (the
    explainer service / API router) treat this as a request-shape error (HTTP 400),
    not a fallback case."""


def is_short_draft(draft: str) -> bool:
    return len(draft.split()) < SHORT_DRAFT_WORD_THRESHOLD


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _fmt_num(value: float) -> str:
    return f"{value:.2f}"


def _build_demand_predictor_draft(field_values: dict) -> str:
    forecast = field_values.get("demand_forecast")
    if forecast is None:
        return (
            "Agent 1 (Demand Predictor) did not produce a forecast for this SKU "
            "(demand_forecast is null)."
        )
    meaning = AGENT_GROUNDING["demand_predictor"]["demand_forecast"]["description"]
    return (
        f"Agent 1 (Demand Predictor) forecasts {_fmt_num(forecast)} units of demand "
        f"for this SKU. {meaning}"
    )


def _build_risk_detector_draft(field_values: dict) -> str:
    prob = field_values.get("backorder_prob")
    alarm = field_values.get("alarm_triggered")
    if prob is None:
        return "Agent 2 (Risk Detector) did not produce a risk assessment for this SKU."
    g = AGENT_GROUNDING["risk_detector"]
    threshold = g["alarm_triggered"]["threshold"]
    if alarm:
        status = f"flagged as high risk (probability >= {threshold}, the operational threshold)"
    else:
        status = f"not flagged as high risk (below the {threshold} operational threshold)"
    return (
        f"Agent 2 (Risk Detector) estimates a {_fmt_pct(prob)} probability of backorder "
        f"for this SKU and it is {status}. {g['alarm_triggered']['description']}"
    )


def _build_inventory_rebalancer_draft(field_values: dict) -> str:
    urgency = field_values.get("urgency_score")
    if urgency is None:
        return "Agent 3 (Inventory Rebalancer) did not produce an urgency score for this SKU."
    g = AGENT_GROUNDING["inventory_rebalancer"]
    parts = [
        f"Agent 3 (Inventory Rebalancer) computed a restock urgency score of "
        f"{_fmt_num(urgency)} for this SKU. {g['urgency_score']['description']}"
    ]
    qty = field_values.get("recommended_qty")
    rank = field_values.get("batch_rank")
    if qty is not None:
        qty_sentence = f"It recommends ordering {qty:.0f} units. {g['recommended_qty']['description']}"
        parts.append(qty_sentence)
    if rank is not None:
        parts.append(f"Within the current batch, this SKU ranks #{rank} by urgency. {g['batch_rank']['description']}")
    return " ".join(parts)


def _build_forecast_optimizer_draft(field_values: dict) -> str:
    factor = field_values.get("correction_factor")
    if factor is None:
        return "Agent 5 (Forecast Optimizer) did not run for this SKU."
    g = AGENT_GROUNDING["forecast_optimizer"]
    parts = [
        f"Agent 5 (Forecast Optimizer) applies a correction factor of {_fmt_num(factor)} "
        f"to the human forecast for this SKU. {g['correction_factor']['description']}"
    ]
    recommendation = field_values.get("recommendation")
    if recommendation and recommendation in g["recommendation"]["values"]:
        parts.append(f"Recommendation: {recommendation}. {g['recommendation']['values'][recommendation]}")
    bias_severity = field_values.get("bias_severity")
    if bias_severity and bias_severity in g["bias_severity"]["values"]:
        parts.append(f"Bias severity: {bias_severity}. {g['bias_severity']['values'][bias_severity]}")
    if field_values.get("risk_override_applied"):
        parts.append(g["risk_override_applied"]["description"])
    return " ".join(parts)


def _build_supplier_auditor_draft(field_values: dict) -> str:
    grade = field_values.get("supplier_risk")
    if grade is None:
        return "Agent 6 (Supplier Auditor) did not produce a grade for this supplier."
    g = AGENT_GROUNDING["supplier_auditor"]
    meaning = g["supplier_risk"]["values"][grade]
    parts = [f"This supplier received grade {grade}. {meaning}"]
    if field_values.get("stop_auto_buy_triggered"):
        parts.append(g["stop_auto_buy_triggered"]["description"])
        reason = field_values.get("trigger_reason")
        if reason and reason in g["trigger_reason"]["values"]:
            parts.append(f"Trigger reason: {reason}. {g['trigger_reason']['values'][reason]}")
    return " ".join(parts)


_AGENT_DRAFT_BUILDERS = {
    "demand_predictor": _build_demand_predictor_draft,
    "risk_detector": _build_risk_detector_draft,
    "inventory_rebalancer": _build_inventory_rebalancer_draft,
    "forecast_optimizer": _build_forecast_optimizer_draft,
    "supplier_auditor": _build_supplier_auditor_draft,
}


def build_agent_draft(agent_name: str, field_values: dict) -> str:
    """Assembles a plain-English draft from AGENT_GROUNDING[agent_name] + the actual
    values for this run. Pure string templating -- no model call, no randomness."""
    if agent_name not in AGENT_GROUNDING:
        raise UnknownAgentError(agent_name)
    return _AGENT_DRAFT_BUILDERS[agent_name](field_values)


# Order the whole-run draft walks the reduced PipelineState fields in.
_RUN_FIELD_ORDER = (
    "demand_forecast", "backorder_prob", "alarm_triggered",
    "urgency_score", "correction_factor", "supplier_risk",
)


def build_run_draft(sku_id: str, predictions_row: dict, suppression_note: str | None = None) -> str:
    """
    Whole-run version. Walks the reduced PipelineState fields (demand_forecast,
    backorder_prob, alarm_triggered, urgency_score, correction_factor, supplier_risk),
    looks up each one's grounding, and assembles a short paragraph. If
    suppression_note is present (from agent_traces), explicitly states the override
    using forecast_optimizer's risk_override_applied grounding text.
    """
    parts = [f"Summary for SKU {sku_id}:"]

    forecast = predictions_row.get("demand_forecast")
    if forecast is not None:
        parts.append(f"Agent 1 forecasts {_fmt_num(forecast)} units of demand over the next 6 months.")

    prob = predictions_row.get("backorder_prob")
    alarm = predictions_row.get("alarm_triggered")
    if prob is not None:
        threshold = AGENT_GROUNDING["risk_detector"]["alarm_triggered"]["threshold"]
        status = "flagged as high risk" if alarm else "not flagged as high risk"
        parts.append(f"Agent 2 estimates a {_fmt_pct(prob)} backorder probability and the SKU is {status} (threshold {threshold}).")

    urgency = predictions_row.get("urgency_score")
    if urgency is not None:
        parts.append(f"Agent 3 assigns a restock urgency score of {_fmt_num(urgency)}.")

    factor = predictions_row.get("correction_factor")
    if factor is not None:
        rec_values = AGENT_GROUNDING["forecast_optimizer"]["recommendation"]["values"]
        if factor < 0.90:
            label = "REDUCE_PLAN"
        elif factor > 1.10:
            label = "INCREASE_PLAN"
        else:
            label = "HOLD"
        parts.append(f"Agent 5 applies a correction factor of {_fmt_num(factor)} to the human forecast ({label}: {rec_values[label]}).")

    if suppression_note:
        override_desc = AGENT_GROUNDING["forecast_optimizer"]["risk_override_applied"]["description"]
        parts.append(f"Note: {suppression_note}. {override_desc}")

    grade = predictions_row.get("supplier_risk")
    if grade is not None:
        g = AGENT_GROUNDING["supplier_auditor"]["supplier_risk"]
        parts.append(f"Agent 6 grades the supplier {grade}: {g['values'][grade]}")

    return " ".join(parts)
