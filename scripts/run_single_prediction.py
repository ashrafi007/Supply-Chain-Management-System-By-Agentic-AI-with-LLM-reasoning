"""scripts/run_single_prediction.py -- CLI demo: runs one SKU through the real LangGraphExecutor, pretty-prints pipeline_runs/predictions/agent_traces.

    python -m scripts.run_single_prediction --sku-id SKU00123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _print_row(title: str, obj) -> None:
    print(f"\n--- {title} ---")
    if obj is None:
        print("  (none)")
        return
    for col in obj.__table__.columns:
        print(f"  {col.name}: {getattr(obj, col.name)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SKU through the real orchestrator and print the persisted rows.")
    parser.add_argument("--sku-id", required=True)
    args = parser.parse_args(argv)

    from src.db.base import SessionLocal
    from src.db.models import AgentTrace, PipelineRun, Prediction
    from src.orchestrator.executor import LangGraphExecutor
    from src.repository.pipeline_service import run_pipeline_for_sku

    session = SessionLocal()
    try:
        executor = LangGraphExecutor()
        run_id = run_pipeline_for_sku(session, args.sku_id, executor)
        print(f"run_id: {run_id}")

        run = session.get(PipelineRun, run_id)
        _print_row("pipeline_runs", run)

        prediction = session.execute(
            select(Prediction).where(Prediction.run_id == run_id)
        ).scalar_one_or_none()
        _print_row("predictions", prediction)

        traces = session.execute(
            select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.sequence)
        ).scalars().all()
        print(f"\n--- agent_traces ({len(traces)} rows) ---")
        for t in traces:
            print(f"  [{t.sequence}] {t.agent_name}: status={t.status} latency_ms={t.latency_ms} note={t.note!r}")
            print(f"       output={t.output!r}")
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
