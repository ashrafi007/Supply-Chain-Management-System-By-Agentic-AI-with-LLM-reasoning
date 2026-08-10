"""
Deterministic, fully-grounded draft assembly (llm_insertion_spec.md SS6).

This module IS the explanation. The LLM in client.py only rewords its output --
get this right and the rest is safe by construction. Rule enforced here, checked
in tests/llm/test_draft_builder.py: every number or category that appears in a
draft must come from the caller-supplied field_values (real run data) or from
AGENT_GROUNDING (the real, documented meaning) -- never free-text commentary that
isn't traceable to one of those two sources.

Audience: this draft is what a non-technical reader sees whenever polishing is
skipped or unavailable (short draft, API error, rate limit) -- not just a fallback
for developers. So it is written directly in plain business language throughout,
not "plain sentence + a raw jargon aside" -- earlier versions of this module
appended a "(Technical basis: ...)" clause with the literal grounding text (e.g.
"correction_factor < 0.90 -- cut the production/purchasing plan.") after every
plain sentence; that defeated the point the moment polish wasn't available, which
is exactly when a non-technical reader is most likely to see the raw draft
unmodified. AGENT_GROUNDING's exact field names and internal thresholds still
live in AGENT_OUTPUT_MEANINGS.md / grounding.py for anyone who needs the technical
detail -- this module no longer repeats them verbatim in the human-facing text
except where a real number (like the 0.945 alert threshold) is itself part of
the plain sentence, not decoration bolted onto it.
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


# Plain-language framing for each supplier grade -- business outcome, not the
# underlying probability band.
_PLAIN_SUPPLIER_GRADE = {
    "A": "This supplier looks solid -- approved, no action needed.",
    "B": "This supplier is worth keeping an eye on -- schedule a contract review.",
    "C": "This supplier is at risk -- escalate to procurement for a closer look.",
    "D": "This supplier is in the critical zone -- may need to be cut off from automatic ordering.",
}

# Plain-language framing for a forecast-optimizer recommendation.
_PLAIN_FORECAST_RECOMMENDATION = {
    "REDUCE_PLAN": "the original human forecast looks too high, so we recommend cutting the purchasing/production plan",
    "INCREASE_PLAN": "the original human forecast looks too low, so we recommend raising the purchasing/production plan",
    "HOLD": "the original human forecast looks accurate, so no change is recommended",
}


def _build_demand_predictor_draft(field_values: dict) -> str:
    forecast = field_values.get("demand_forecast")
    if forecast is None:
        return "We don't have a sales forecast for this product right now -- the demand model didn't return a result."
    return f"Expected demand: this product is projected to sell about {_fmt_num(forecast)} units over the next 6 months."


def _build_risk_detector_draft(field_values: dict) -> str:
    prob = field_values.get("backorder_prob")
    alarm = field_values.get("alarm_triggered")
    if prob is None:
        return "We don't have a stockout-risk read on this product yet."
    threshold = AGENT_GROUNDING["risk_detector"]["alarm_triggered"]["threshold"]
    if alarm:
        return (
            f"Stockout risk: HIGH. There's a {_fmt_pct(prob)} chance this product runs out of "
            f"stock, which is high enough that the system has raised an alert (our alert line is "
            f"{threshold}), so this needs attention now."
        )
    return (
        f"Stockout risk: low. There's a {_fmt_pct(prob)} chance this product runs out of "
        f"stock, below our {threshold} alert line, so no urgent action is needed here."
    )


def _build_inventory_rebalancer_draft(field_values: dict) -> str:
    urgency = field_values.get("urgency_score")
    if urgency is None:
        return "We don't have a restocking-urgency read on this product yet."
    parts = [f"Restock urgency: {_fmt_num(urgency)} out of 1.00 (higher means more urgent)."]
    qty = field_values.get("recommended_qty")
    rank = field_values.get("batch_rank")
    if qty is not None:
        parts.append(f"Recommended action: order {qty:.0f} more units to bring this product back to a safe stock level.")
    if rank is not None:
        parts.append(f"This is currently the #{rank} most urgent product in this batch.")
    return " ".join(parts)


def _build_forecast_optimizer_draft(field_values: dict) -> str:
    factor = field_values.get("correction_factor")
    if factor is None:
        return "The forecast-correction check didn't run for this product."
    recommendation = field_values.get("recommendation")
    plain = _PLAIN_FORECAST_RECOMMENDATION.get(
        recommendation, "here's how the forecast compares to the original human estimate"
    )
    parts = [f"Forecast check: {plain} (adjustment factor {_fmt_num(factor)})."]
    bias_severity = field_values.get("bias_severity")
    if bias_severity == "SEVERE":
        parts.append("The human forecast has been significantly too high for a while now, worth a closer look.")
    elif bias_severity == "MILD":
        parts.append("The human forecast has been a bit high lately, worth noting.")
    if field_values.get("risk_override_applied"):
        parts.append(
            "Note: this forecast was deliberately left unchanged even though the numbers might "
            "suggest otherwise, because this product is already flagged as at risk of running "
            "out -- the policy is to never shrink the forecast for a product that's already at "
            "risk of stocking out."
        )
    return " ".join(parts)


def _build_supplier_auditor_draft(field_values: dict) -> str:
    grade = field_values.get("supplier_risk")
    if grade is None:
        return "We don't have a supplier risk grade for this yet."
    g = AGENT_GROUNDING["supplier_auditor"]
    plain = _PLAIN_SUPPLIER_GRADE.get(grade, "")
    parts = [f"Supplier check: grade {grade}. {plain}"]
    if grade == "D":
        # The only place this module still states a raw probability band verbatim --
        # kept because supplier grade is a regulatory/audit-relevant number, and this
        # exact phrasing ("Critical", "[0.70, 1.00]") is asserted by
        # tests/llm/test_draft_builder.py as a deliberate traceability floor for the
        # single highest-stakes category in the system (stop_auto_buy eligibility).
        parts.append(f"({g['supplier_risk']['values']['D']})")
    if field_values.get("stop_auto_buy_triggered"):
        parts.append("Automatic purchase orders from this supplier have been paused until a human reviews it.")
        reason = field_values.get("trigger_reason")
        if reason == "MODEL+RULE":
            parts.append("Both our model and a hard business rule agree on this -- a strong signal.")
        elif reason == "RULE":
            parts.append("This was caught by a safety-net rule (severe overdue deliveries) even though the model alone didn't flag it.")
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


def build_run_draft(sku_id: str, predictions_row: dict, suppression_note: str | None = None) -> str:
    """
    Whole-run version. Walks the reduced PipelineState fields (demand_forecast,
    backorder_prob, alarm_triggered, urgency_score, correction_factor, supplier_risk)
    and assembles a short paragraph in plain business language -- what this
    product's numbers mean and what (if anything) to do about it. This is the
    version shown by default (agent_name=None) -- the one a non-technical reader
    is most likely to see, including in the raw-draft fallback path when polishing
    isn't available -- so it stays fully plain-English with no trailing jargon
    aside, except the one literal token (suppression_note, e.g.
    "suppressed: alarm_triggered=1") that a caller supplied and this function is
    contractually required to surface verbatim for audit purposes.
    """
    parts = [f"Here's a plain-English summary for product {sku_id}:"]

    forecast = predictions_row.get("demand_forecast")
    if forecast is not None:
        parts.append(f"We expect to sell about {_fmt_num(forecast)} units of this product over the next 6 months.")

    prob = predictions_row.get("backorder_prob")
    alarm = predictions_row.get("alarm_triggered")
    if prob is not None:
        threshold = AGENT_GROUNDING["risk_detector"]["alarm_triggered"]["threshold"]
        if alarm:
            parts.append(
                f"Stockout risk is HIGH -- a {_fmt_pct(prob)} chance of running out, which is high "
                f"enough to trip our alert line (set at {threshold}), so this needs attention."
            )
        else:
            parts.append(
                f"Stockout risk is low -- a {_fmt_pct(prob)} chance of running out, below our "
                f"{threshold} alert line, so no urgent action is needed here."
            )

    urgency = predictions_row.get("urgency_score")
    if urgency is not None:
        parts.append(
            f"On a 0-to-1 scale of how urgently it needs restocking, this product scores "
            f"{_fmt_num(urgency)} (higher means more urgent)."
        )

    factor = predictions_row.get("correction_factor")
    if factor is not None:
        if factor < 0.90:
            plain = "the original human forecast looks too high, so we recommend cutting the purchasing/production plan"
        elif factor > 1.10:
            plain = "the original human forecast looks too low, so we recommend raising the purchasing/production plan"
        else:
            plain = "the original human forecast looks accurate, so no change is recommended"
        parts.append(f"On the purchasing plan: {plain} (adjustment factor {_fmt_num(factor)}).")

    if suppression_note:
        parts.append(
            f"Note ({suppression_note}): this forecast was deliberately left unchanged even "
            f"though the numbers might suggest otherwise, because this product is already "
            f"flagged as at risk of running out -- the policy is to never shrink the forecast "
            f"for a product that's already at risk of stocking out."
        )

    grade = predictions_row.get("supplier_risk")
    if grade is not None:
        plain_grade = _PLAIN_SUPPLIER_GRADE.get(grade, "")
        parts.append(f"Supplier check: grade {grade}. {plain_grade}")

    return " ".join(parts)
