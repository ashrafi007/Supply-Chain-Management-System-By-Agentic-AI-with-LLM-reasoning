"""LangGraph node adapter for the Inventory Rebalancer agent. No model logic lives here -- calls the already-verified rebalance_inventory()."""

from __future__ import annotations

import time
from pathlib import Path

from inference_tools.inventory_rebalancer_tool import load_rebalancer_artifacts, rebalance_inventory
from inference_tools.schemas import InventoryRebalancerInput
from src.orchestrator import manifest
from src.orchestrator.state import GraphState


class RebalancerAgent:
    """Thin orchestrator-side adapter: state in, state delta out. Owns urgency_score.

    replenishment_qty stays out of the delta -- Agent 4 (Routing), the field's
    documented owner, is permanently cancelled; recommended_qty is preserved
    at trace granularity only (agent_traces.output), not in predictions.
    """

    NAME = "inventory_rebalancer"

    def __init__(self, models_dir: str | Path | None = None):
        load_rebalancer_artifacts(str(models_dir) if models_dir else None)

    def run(self, state: GraphState) -> dict:
        trace = list(state.get("trace", []))
        try:
            sku = state["sku_id"]
            raw = state["raw_features"]
            bo = state["backorder_prob"]
            if bo is None:
                raise ValueError("backorder_prob is None -- upstream risk detector did not run or failed")
            payload = InventoryRebalancerInput(
                skus=[sku], data={sku: raw}, backorder_probability={sku: bo}
            )
            result = rebalance_inventory(payload)
            rec = result.recommendations[0]
            trace.append({
                "tool": self.NAME,
                "status": "ok",
                "model_version": result.model_version,
                "recommended_qty": rec.recommended_qty,
                "batch_rank": rec.batch_rank,
            })
            return {"urgency_score": rec.urgency_score_predicted, "trace": trace}
        except Exception as exc:
            trace.append({
                "tool": self.NAME,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return {"urgency_score": None, "trace": trace}


_agent = RebalancerAgent(models_dir=manifest.REBALANCER_MODEL_PATH.parent)


def node(state: GraphState) -> dict:
    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
