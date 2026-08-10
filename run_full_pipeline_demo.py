"""run_full_pipeline_demo.py -- end-to-end demo: runs real SKUs through the full
5-agent orchestrator (Demand Predictor -> Risk Detector -> Inventory Rebalancer ->
Forecast Optimizer -> Supplier Auditor), then generates a plain-English LLM
explanation for each one. Built to be shown to a non-technical audience -- clean,
narrated output, not a raw data dump.

    python run_full_pipeline_demo.py
    python run_full_pipeline_demo.py --sku-id 1111949 --sku-id 1113049
    python run_full_pipeline_demo.py --n 5

If OPENROUTER_API_KEY isn't set (no .env, no env var), this still runs end to end --
every explanation falls back to the deterministic, plain-English draft instead of the
LLM-polished version. That fallback is itself part of the design (never a hard
error), so the demo works either way.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

_WIDTH = 78
_AGENT_ROWS = (
    ("Agent 1", "Demand Predictor", "demand_forecast", "{:.2f} units over next 6 months"),
    ("Agent 2", "Risk Detector", "backorder_prob", "{:.1%} backorder probability"),
    ("Agent 3", "Inventory Rebalancer", "urgency_score", "{:.2f} restock urgency (0-1)"),
    ("Agent 5", "Forecast Optimizer", "correction_factor", "{:.2f}x adjustment to human forecast"),
    ("Agent 6", "Supplier Auditor", "supplier_risk", "grade {}"),
)


def _rule(char: str = "-") -> None:
    print(char * _WIDTH)


def _pick_default_sku_ids(session, n: int) -> list[str]:
    from src.db.models import InventoryCurrent
    rows = session.execute(select(InventoryCurrent.sku_id).limit(n)).scalars().all()
    if len(rows) < n:
        raise RuntimeError(f"data/app.db has fewer than {n} seeded SKUs -- run scripts/sample_skus.py first")
    return list(rows)


def _print_agent_row(label_1: str, label_2: str, field: str, fmt: str, prediction) -> None:
    value = getattr(prediction, field, None)
    rendered = "n/a (did not run)" if value is None else fmt.format(value)
    print(f"  {label_1:<8} {label_2:<22} -> {rendered}")


def _print_explanation_box(text: str) -> None:
    for line in textwrap.wrap(text, width=_WIDTH - 4):
        print(f"  | {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run real SKUs through the full orchestrator and print a plain-English explanation for each."
    )
    parser.add_argument(
        "--sku-id", action="append", dest="sku_ids",
        help="repeatable; run a specific SKU. Defaults to the first --n seeded SKUs.",
    )
    parser.add_argument("--n", type=int, default=2, help="how many default SKUs to run if --sku-id is omitted (default 2)")
    parser.add_argument(
        "--model", default="nvidia/nemotron-nano-9b-v2:free",
        help="OpenRouter model for polishing (default: nvidia/nemotron-nano-9b-v2:free -- "
             "chosen for this demo script specifically because it had free-tier quota "
             "available when client.py's own default, openai/gpt-oss-20b:free, was "
             "rate-limited from heavy testing. Swap freely; see .CLAUDE/llm_activation_spec.md.",
    )
    args = parser.parse_args(argv)

    from src.db.base import SessionLocal
    from src.db.models import PipelineRun, Prediction
    from src.llm import explainer_service
    from src.llm.client import LLMUnavailableError, OpenRouterClient
    from src.orchestrator.executor import LangGraphExecutor
    from src.repository.pipeline_service import run_pipeline_for_sku

    session = SessionLocal()

    print("=" * _WIDTH)
    print("  SUPPLY CHAIN AGENTIC AI -- END-TO-END PIPELINE DEMO")
    print("  5-agent orchestrator + LLM-generated plain-English explanation")
    print("=" * _WIDTH)

    if "OPENROUTER_API_KEY" in os.environ:
        client = OpenRouterClient(model=args.model)
        print(f"\nLLM model: {client.model} (OpenRouter)")
    else:
        print("\nOPENROUTER_API_KEY not set -- explanations below use the deterministic")
        print("plain-English draft instead of an LLM-polished version (graceful fallback).")

        class _NoKeyClient:
            model = "no-key-configured"

            def polish(self, draft: str) -> str:
                raise LLMUnavailableError("OPENROUTER_API_KEY not set")

        client = _NoKeyClient()

    summary_rows: list[tuple] = []

    try:
        sku_ids = args.sku_ids or _pick_default_sku_ids(session, args.n)
        executor = LangGraphExecutor()

        for sku_id in sku_ids:
            print()
            _rule()
            print(f"SKU {sku_id}")
            _rule()

            run_id = run_pipeline_for_sku(session, sku_id, executor)
            run = session.get(PipelineRun, run_id)
            status_symbol = "OK " if run.status == "success" else "FAIL"
            print(f"Pipeline run: [{status_symbol}] {run.status}  (run_id {run_id}, {run.latency_ms}ms)")

            if run.status != "success":
                print(f"  error: {run.error}")
                summary_rows.append((sku_id, run.status, None, None, None, None))
                continue

            prediction = session.execute(
                select(Prediction).where(Prediction.run_id == run_id)
            ).scalar_one_or_none()

            print()
            for label_1, label_2, field, fmt in _AGENT_ROWS:
                _print_agent_row(label_1, label_2, field, fmt, prediction)

            result = explainer_service.explain(session, client, run_id, agent_name=None)
            print()
            print("  AI-generated summary (plain English, for a non-technical reader):")
            print(f"  {'+' + '-' * (_WIDTH - 4) + '+'}")
            _print_explanation_box(result["explanation"])
            print(f"  {'+' + '-' * (_WIDTH - 4) + '+'}")
            tag = "LLM-polished" if result["was_polished"] else f"fallback: {result['fallback_reason']}"
            print(f"  ({tag}, model: {result['model_used']})")

            summary_rows.append((
                sku_id, run.status,
                prediction.demand_forecast, prediction.backorder_prob,
                prediction.urgency_score, prediction.supplier_risk,
            ))

        print()
        print("=" * _WIDTH)
        print("  SUMMARY")
        print("=" * _WIDTH)
        header = f"{'SKU':<14}{'Status':<10}{'Demand':<10}{'Backorder%':<12}{'Urgency':<10}{'Supplier'}"
        print(header)
        print("-" * _WIDTH)
        for sku_id, status, demand, backorder, urgency, grade in summary_rows:
            demand_s = f"{demand:.2f}" if demand is not None else "-"
            backorder_s = f"{backorder:.1%}" if backorder is not None else "-"
            urgency_s = f"{urgency:.2f}" if urgency is not None else "-"
            grade_s = grade or "-"
            print(f"{sku_id:<14}{status:<10}{demand_s:<10}{backorder_s:<12}{urgency_s:<10}{grade_s}")
        print("=" * _WIDTH)
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
