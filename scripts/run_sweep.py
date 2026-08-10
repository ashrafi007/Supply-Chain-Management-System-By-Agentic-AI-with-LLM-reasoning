"""scripts/run_sweep.py -- CLI entrypoint for order_queue: runs every due SKU
through the real orchestrator, then generates a whole-run LLM explanation for
each one that succeeded (src.queue.sweep_service.run_sweep_and_explain).

    python -m scripts.run_sweep
    python -m scripts.run_sweep --no-explain      # sweep only, original behavior
    python -m scripts.run_sweep --as-of 2026-08-15

If OPENROUTER_API_KEY isn't set, this still runs end to end -- every
explanation falls back to the deterministic draft (same graceful-fallback
design as run_llm_explanation_example.py), it does not error.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every due order_queue SKU through the pipeline, then explain each result."
    )
    parser.add_argument(
        "--as-of", type=date.fromisoformat, default=None,
        help="ISO date to sweep as of (default: today). Mainly for testing due-date logic.",
    )
    parser.add_argument(
        "--no-explain", action="store_true",
        help="Run the sweep only -- skip LLM explanations entirely (no network calls at all).",
    )
    args = parser.parse_args(argv)
    as_of = args.as_of or date.today()

    from src.db.base import SessionLocal
    from src.llm.client import LLMUnavailableError, OpenRouterClient
    from src.orchestrator.executor import LangGraphExecutor
    from src.queue import sweep_service

    session = SessionLocal()
    try:
        executor = LangGraphExecutor()

        if args.no_explain:
            result = sweep_service.run_sweep(session, as_of, executor)
            explanations = {}
        else:
            if "OPENROUTER_API_KEY" in os.environ:
                client = OpenRouterClient()
                print(f"Using OpenRouter model: {client.model}")
            else:
                print(
                    "OPENROUTER_API_KEY not set -- every explanation below will be the "
                    "deterministic draft, not LLM-polished. See .CLAUDE/llm_activation_spec.md."
                )

                class _NoKeyClient:
                    model = "no-key-configured"

                    def polish(self, draft: str) -> str:
                        raise LLMUnavailableError("OPENROUTER_API_KEY not set")

                client = _NoKeyClient()

            result = sweep_service.run_sweep_and_explain(session, as_of, executor, client)
            explanations = result["explanations"]

        session.commit()

        print(f"\nas_of:            {result['as_of']}")
        print(f"expired_count:     {result['expired_count']}")
        print(f"evaluated_sku_ids: {result['evaluated_sku_ids']}")
        print(f"run_ids:           {result['run_ids']}")

        if not args.no_explain:
            print(f"\n{len(explanations)} explanation(s) generated:")
            for run_id, exp in explanations.items():
                print(f"\n--- run {run_id} ---")
                print(f"  was_polished:    {exp['was_polished']}")
                print(f"  fallback_reason: {exp['fallback_reason']}")
                print(f"  model_used:      {exp['model_used']}")
                print(f"  explanation:\n    {exp['explanation']}")

            skipped = set(result["run_ids"]) - set(explanations)
            if skipped:
                print(f"\n{len(skipped)} run(s) skipped (pipeline failed, nothing to explain): {sorted(skipped)}")
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
