"""run_llm_explanation_example.py -- runs two real SKUs through the full 5-agent
pipeline, then has the LLM explanation layer (llm_insertion_spec.md) summarize
each run's result in plain English.

    python run_llm_explanation_example.py
    python run_llm_explanation_example.py --sku-id 1111949 --sku-id 1113049

If OPENROUTER_API_KEY isn't set (no .env, no env var), this still runs end to
end -- it demonstrates the graceful-fallback path instead of the polish path,
which is itself the point of the draft-first design (SS0 of the spec).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _print_row(title: str, obj) -> None:
    print(f"\n--- {title} ---")
    if obj is None:
        print("  (none)")
        return
    for col in obj.__table__.columns:
        print(f"  {col.name}: {getattr(obj, col.name)!r}")


def _pick_default_sku_ids(session, n: int) -> list[str]:
    from src.db.models import InventoryCurrent
    rows = session.execute(select(InventoryCurrent.sku_id).limit(n)).scalars().all()
    if len(rows) < n:
        raise RuntimeError(f"data/app.db has fewer than {n} seeded SKUs -- run scripts/sample_skus.py first")
    return list(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run two SKUs through the real pipeline and print the LLM explanation for each."
    )
    parser.add_argument(
        "--sku-id", action="append", dest="sku_ids",
        help="repeatable; give twice for two SKUs. Defaults to the first two seeded SKUs.",
    )
    args = parser.parse_args(argv)

    from src.db.base import SessionLocal
    from src.db.models import PipelineRun, Prediction
    from src.llm import explainer_service
    from src.llm.client import LLMUnavailableError, OpenRouterClient
    from src.orchestrator.executor import LangGraphExecutor
    from src.repository.pipeline_service import run_pipeline_for_sku

    session = SessionLocal()

    if "OPENROUTER_API_KEY" in os.environ:
        client = OpenRouterClient()
        print(f"Using OpenRouter model: {client.model}")
    else:
        print(
            "OPENROUTER_API_KEY not set (no .env found) -- running in fallback-only mode.\n"
            "Every explanation below will be the deterministic draft, not an LLM-polished version.\n"
            "Set OPENROUTER_API_KEY (see .env.example) to see real polishing."
        )

        class _NoKeyClient:
            model = "no-key-configured"

            def polish(self, draft: str) -> str:
                raise LLMUnavailableError("OPENROUTER_API_KEY not set")

        client = _NoKeyClient()

    try:
        sku_ids = args.sku_ids or _pick_default_sku_ids(session, 2)
        executor = LangGraphExecutor()

        for sku_id in sku_ids:
            print(f"\n{'=' * 70}\nSKU {sku_id}\n{'=' * 70}")

            run_id = run_pipeline_for_sku(session, sku_id, executor)
            print(f"run_id: {run_id}")

            run = session.get(PipelineRun, run_id)
            _print_row("pipeline_runs", run)

            prediction = session.execute(
                select(Prediction).where(Prediction.run_id == run_id)
            ).scalar_one_or_none()
            _print_row("predictions", prediction)

            if run.status != "success" or prediction is None:
                print("\n--- llm explanation ---\n  (skipped: run did not succeed)")
                continue

            result = explainer_service.explain(session, client, run_id, agent_name=None)
            print("\n--- llm explanation (whole run) ---")
            print(f"  was_polished:    {result['was_polished']}")
            print(f"  fallback_reason: {result['fallback_reason']}")
            print(f"  model_used:      {result['model_used']}")
            print(f"  explanation:\n    {result['explanation']}")
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
