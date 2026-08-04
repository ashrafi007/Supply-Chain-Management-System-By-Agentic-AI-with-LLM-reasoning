"""LangGraph node adapter for the Supplier Auditor agent. No model logic lives here -- calls the already-verified audit_suppliers()."""

from __future__ import annotations

import time
from pathlib import Path

from inference_tools.supplier_auditor_tool import audit_suppliers, load_supplier_auditor_artifact
from inference_tools.schemas import SupplierAuditorInput
from src.orchestrator import manifest
from src.orchestrator.state import GraphState


class SupplierAuditorAgent:
    """Thin orchestrator-side adapter: state in, state delta out. Owns supplier_risk (= supplier_grade)."""

    NAME = "supplier_auditor"

    def __init__(self, model_path: str | Path | None = None):
        load_supplier_auditor_artifact(str(model_path) if model_path else None)

    def run(self, state: GraphState) -> dict:
        trace = list(state.get("trace", []))
        try:
            sku = state["sku_id"]
            raw = state["raw_features"]
            payload = SupplierAuditorInput(skus=[sku], data={sku: raw})
            result = audit_suppliers(payload)
            res = result.results[0]
            trace.append({
                "tool": self.NAME,
                "status": "ok",
                "model_version": result.model_version,
                "risk_probability": res.risk_probability,
                "stop_auto_buy_triggered": res.stop_auto_buy_triggered,
                "trigger_reason": res.trigger_reason,
                "delivery_stress": res.delivery_stress,
            })
            return {"supplier_risk": res.supplier_grade, "trace": trace}
        except Exception as exc:
            trace.append({
                "tool": self.NAME,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return {"supplier_risk": None, "trace": trace}


_agent = SupplierAuditorAgent(model_path=manifest.SUPPLIER_AUDITOR_MODEL_PATH)


def node(state: GraphState) -> dict:
    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
