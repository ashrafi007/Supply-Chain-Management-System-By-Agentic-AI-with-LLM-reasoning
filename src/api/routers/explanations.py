"""POST /predictions/{run_id}/explain (llm_insertion_spec.md SS10).

No 503 case: per explainer_service, an API/network failure degrades to the draft
rather than erroring, so the only non-200 outcomes here are request-shape errors --
unknown run_id (404) or unknown agent_name (400).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, get_llm_client
from src.llm import explainer_service
from src.llm.client import OpenRouterClient
from src.llm.draft_builder import UnknownAgentError
from src.llm.explainer_service import RunNotFoundError

router = APIRouter()


@router.post("/predictions/{run_id}/explain")
def explain_run(
    run_id: str,
    agent_name: str | None = None,
    session: Session = Depends(get_db),
    client: OpenRouterClient = Depends(get_llm_client),
) -> dict:
    try:
        return explainer_service.explain(session, client, run_id, agent_name)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    except UnknownAgentError:
        raise HTTPException(status_code=400, detail=f"unknown agent_name {agent_name!r}")
