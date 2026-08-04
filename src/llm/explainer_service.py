"""
Public entrypoint for on-demand explanations (llm_insertion_spec.md SS9).

explain() always returns a complete, correct result -- either polished or the raw
draft -- and only ever raises for a genuine request-shape problem (unknown run_id,
unknown agent_name). A network/API problem never propagates past this function;
it degrades to the deterministic draft instead.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import AgentTrace, PipelineRun, Prediction
from src.llm.client import LLMUnavailableError, OpenRouterClient
from src.llm.draft_builder import (
    UnknownAgentError,
    build_agent_draft,
    build_run_draft,
    is_short_draft,
)
from src.llm.models import LLMExplanation

TEMPLATE_ONLY = "template-only"


class RunNotFoundError(ValueError):
    """No pipeline_runs row for the given run_id, or the run never produced a
    predictions row (e.g. it failed) -- either way, there is nothing to explain."""


def _get_cached(session: Session, run_id: str, agent_name: str | None) -> LLMExplanation | None:
    stmt = select(LLMExplanation).where(
        LLMExplanation.run_id == run_id, LLMExplanation.agent_name == agent_name
    )
    return session.execute(stmt).scalar_one_or_none()


def _latest_trace(session: Session, run_id: str, agent_name: str) -> AgentTrace | None:
    stmt = (
        select(AgentTrace)
        .where(AgentTrace.run_id == run_id, AgentTrace.agent_name == agent_name)
        .order_by(AgentTrace.sequence.desc())
    )
    return session.execute(stmt).scalars().first()


def _suppression_note(session: Session, run_id: str) -> str | None:
    trace = _latest_trace(session, run_id, "forecast_optimizer")
    if trace is not None and trace.status == "skipped":
        return trace.note
    return None


# Canonical per-agent fields sourced from `predictions` (the reduced PipelineState).
# Richer fields (recommended_qty, bias_severity, trigger_reason, ...) are layered on
# top from agent_traces.output, which is where the tools record them (SS9 step 2).
_CANONICAL_FIELDS = {
    "demand_predictor": lambda p: {"demand_forecast": p.demand_forecast},
    "risk_detector": lambda p: {
        "backorder_prob": p.backorder_prob,
        "alarm_triggered": p.alarm_triggered,
    },
    "inventory_rebalancer": lambda p: {"urgency_score": p.urgency_score},
    "forecast_optimizer": lambda p: {"correction_factor": p.correction_factor},
    "supplier_auditor": lambda p: {"supplier_risk": p.supplier_risk},
}


def _build_draft(session: Session, run_id: str, agent_name: str | None, prediction: Prediction) -> str:
    if agent_name is None:
        note = _suppression_note(session, run_id)
        predictions_row = {
            "demand_forecast": prediction.demand_forecast,
            "backorder_prob": prediction.backorder_prob,
            "alarm_triggered": prediction.alarm_triggered,
            "urgency_score": prediction.urgency_score,
            "correction_factor": prediction.correction_factor,
            "supplier_risk": prediction.supplier_risk,
        }
        return build_run_draft(prediction.sku_id, predictions_row, suppression_note=note)

    if agent_name not in _CANONICAL_FIELDS:
        raise UnknownAgentError(agent_name)

    field_values = _CANONICAL_FIELDS[agent_name](prediction)
    trace = _latest_trace(session, run_id, agent_name)
    if trace is not None and trace.output:
        field_values.update(trace.output)

    return build_agent_draft(agent_name, field_values)


def explain(
    session: Session, client: OpenRouterClient, run_id: str, agent_name: str | None = None
) -> dict:
    cached = _get_cached(session, run_id, agent_name)
    if cached is not None:
        return {
            "explanation": cached.final_text,
            "cached": True,
            "was_polished": bool(cached.was_polished),
            "fallback_reason": cached.fallback_reason,
            "model_used": cached.model_used,
        }

    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)

    prediction = session.get(Prediction, run_id)
    if prediction is None:
        raise RunNotFoundError(run_id)

    # Must succeed before any network call (SS9 step 3) -- raises UnknownAgentError,
    # a request-shape error, never a fallback case.
    draft = _build_draft(session, run_id, agent_name, prediction)

    if is_short_draft(draft):
        final_text, was_polished, fallback_reason, model_used = draft, False, "short_draft", TEMPLATE_ONLY
    else:
        try:
            final_text = client.polish(draft)
            was_polished, fallback_reason, model_used = True, None, client.model
        except LLMUnavailableError:
            final_text, was_polished, fallback_reason, model_used = draft, False, "api_error", TEMPLATE_ONLY

    session.add(
        LLMExplanation(
            run_id=run_id,
            agent_name=agent_name,
            draft_text=draft,
            final_text=final_text,
            was_polished=int(was_polished),
            fallback_reason=fallback_reason,
            model_used=model_used,
        )
    )
    session.commit()

    return {
        "explanation": final_text,
        "cached": False,
        "was_polished": was_polished,
        "fallback_reason": fallback_reason,
        "model_used": model_used,
    }
