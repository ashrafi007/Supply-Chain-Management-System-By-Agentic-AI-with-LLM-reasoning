"""LangGraph node adapter for the Risk Detector agent. No model logic lives here -- calls the already-verified predict_risk()."""

from __future__ import annotations

import time
from pathlib import Path

from inference_tools.risk_detector_tool import load_risk_detector_model, predict_risk
from inference_tools.schemas import RiskDetectorInput
from src.orchestrator import manifest
from src.orchestrator.state import GraphState


class RiskDetectorAgent:
    """Thin orchestrator-side adapter: state in, state delta out. Owns backorder_prob and alarm_triggered."""

    NAME = "risk_detector"

    def __init__(self, model_path: str | Path | None = None):
        load_risk_detector_model(str(model_path) if model_path else None)

    def run(self, state: GraphState) -> dict:
        trace = list(state.get("trace", []))
        try:
            sku = state["sku_id"]
            raw = state["raw_features"]
            payload = RiskDetectorInput(skus=[sku], data={sku: raw})
            result = predict_risk(payload)
            pred = result.predictions[0]
            trace.append({
                "tool": self.NAME,
                "status": "ok",
                "model_version": result.model_version,
                "threshold_used": result.threshold_used,
                "predicted_label": pred.predicted_label,
            })
            return {
                "backorder_prob": pred.backorder_probability,
                "alarm_triggered": int(pred.is_high_risk),
                "trace": trace,
            }
        except Exception as exc:
            trace.append({
                "tool": self.NAME,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return {"backorder_prob": None, "alarm_triggered": None, "trace": trace}


_agent = RiskDetectorAgent(model_path=manifest.RISK_DETECTOR_MODEL_PATH)


def node(state: GraphState) -> dict:
    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
