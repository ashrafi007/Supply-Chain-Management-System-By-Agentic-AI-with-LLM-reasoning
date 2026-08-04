"""Shared FastAPI dependencies: a per-request DB session and the app-scoped LLM client."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from src.db.base import SessionLocal
from src.llm.client import OpenRouterClient


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_llm_client(request: Request) -> OpenRouterClient:
    """Client is constructed once in main.py's lifespan and stored on app.state --
    never re-constructed per request."""
    return request.app.state.llm_client
