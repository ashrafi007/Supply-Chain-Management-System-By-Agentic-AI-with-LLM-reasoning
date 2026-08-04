"""
SQLAlchemy model for the llm_explanations feature (llm_insertion_spec.md SS3).

Registers LLMExplanation on the existing Base from src.db.base -- additive only,
never a second declarative_base() or a second engine (same discipline as
src.queue.models / queue_migration_spec.md).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMExplanation(Base):
    """Cache of on-demand explanations. draft_text is always stored alongside
    final_text so every fact can be traced to a deterministic source regardless
    of whether the polish step ran (llm_insertion_spec.md SS3).

    Note: SQLite treats NULLs as distinct under UNIQUE, so the (run_id, agent_name)
    constraint does not by itself prevent two whole-run (agent_name=NULL) rows for
    the same run_id -- explainer_service's cache-check-before-insert (SS9 step 1)
    is what actually enforces one row per (run_id, agent_name) in practice.
    """

    __tablename__ = "llm_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, ForeignKey("pipeline_runs.run_id"), nullable=False)
    agent_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    draft_text: Mapped[str] = mapped_column(Text, nullable=False)
    final_text: Mapped[str] = mapped_column(Text, nullable=False)
    was_polished: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_used: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        CheckConstraint("was_polished IN (0, 1)", name="ck_llm_explanations_was_polished"),
        CheckConstraint(
            "fallback_reason IS NULL OR fallback_reason IN ('short_draft', 'api_error')",
            name="ck_llm_explanations_fallback_reason",
        ),
        UniqueConstraint("run_id", "agent_name", name="uq_llm_explanations_run_agent"),
    )
