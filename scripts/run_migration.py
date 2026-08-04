"""
One-time additive migration: registers every feature's tables on the existing Base
and creates only what's missing (queue_migration_spec.md §5 Step 2;
llm_insertion_spec.md §3 uses the same procedure).

Usage:
    python -m scripts.run_migration

Safe to run against a live, populated data/app.db — Base.metadata.create_all only
creates tables that don't yet exist; it never touches, alters, or drops an existing one.
"""

from __future__ import annotations

from src.db.base import Base, engine
import src.db.models  # noqa: F401 -- side effect: registers the 7 existing tables on Base
import src.queue.models  # noqa: F401 -- side effect: registers OrderQueue/OrderQueueLog on Base
import src.llm.models  # noqa: F401 -- side effect: registers LLMExplanation on Base


def run() -> None:
    before = set(Base.metadata.tables.keys())
    Base.metadata.create_all(engine)
    after = set(Base.metadata.tables.keys())
    print(f"Tables known to Base before create_all: {sorted(before)}")
    print(f"Tables known to Base after create_all:  {sorted(after)}")
    print("Migration complete — no existing table was dropped or altered.")


if __name__ == "__main__":
    run()
