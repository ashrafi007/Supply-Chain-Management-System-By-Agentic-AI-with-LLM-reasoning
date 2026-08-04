"""LangGraph node adapter for the Forecast Optimizer agent. No model logic lives here -- calls the already-verified optimize_forecast(). The suppression branch (novelty claim) lives in node()."""

from __future__ import annotations

import time
from pathlib import Path

from inference_tools.forecast_optimizer_tool import load_forecast_optimizer_artifacts, optimize_forecast
from inference_tools.schemas import ForecastOptimizerInput
from src.orchestrator import edges, manifest
from src.orchestrator.state import GraphState


class ForecastOptimizerAgent:
    """Thin orchestrator-side adapter: state in, state delta out. Owns correction_factor.

    recommendation (REDUCE_PLAN/HOLD/INCREASE_PLAN) stays out of the delta --
    PipelineState.recommendation is reserved for a future LLM narration node
    (out of scope here); the categorical label is preserved at trace
    granularity only.
    """

    NAME = "forecast_optimizer"

    def __init__(self, models_dir: str | Path | None = None):
        load_forecast_optimizer_artifacts(str(models_dir) if models_dir else None)

    def run(self, state: GraphState) -> dict:
        trace = list(state.get("trace", []))
        try:
            sku = state["sku_id"]
            raw = state["raw_features"]
            bo = state["backorder_prob"]
            alarm = state["alarm_triggered"]
            if bo is None or alarm is None:
                raise ValueError(
                    "backorder_prob/alarm_triggered is None -- upstream risk detector did not run or failed"
                )
            payload = ForecastOptimizerInput(
                skus=[sku], data={sku: raw},
                backorder_probability={sku: bo}, alarm_triggered={sku: bool(alarm)},
            )
            result = optimize_forecast(payload)
            rec = result.recommendations[0]
            trace.append({
                "tool": self.NAME,
                "status": "ok",
                "model_version_a": result.model_version_a,
                "model_version_b": result.model_version_b,
                "recommendation": rec.recommendation,
                "bias_detected": rec.bias_detected,
                "bias_severity": rec.bias_severity,
                "risk_override_applied": rec.risk_override_applied,
            })
            return {"correction_factor": rec.correction_factor, "trace": trace}
        except Exception as exc:
            trace.append({
                "tool": self.NAME,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return {"correction_factor": None, "trace": trace}


_agent = ForecastOptimizerAgent(models_dir=manifest.FORECAST_OPTIMIZER_MODELS_DIR)


def node(state: GraphState) -> dict:
    if edges.should_suppress(state):
        trace = list(state.get("trace", []))
        trace.append({
            "tool": ForecastOptimizerAgent.NAME,
            "status": "skipped",
            "reason": f"alarm_triggered={state['alarm_triggered']}",
        })
        return {
            "correction_factor": 1.0,
            "trace": trace,
            "_skipped": True,
            "_note": f"suppressed: alarm_triggered={state['alarm_triggered']}",
        }

    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
