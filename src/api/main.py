"""
Minimal FastAPI application, scaffolded solely to host the on-demand explanation
endpoint (.CLAUDE/llm_insertion_spec.md SS10-11). No other API surface exists yet --
this is intentionally thin, not a general web layer.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.routers import explanations, runs
from src.llm.client import OpenRouterClient

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast and loudly if the key is missing, rather than discovering it on the
    # first user click (llm_insertion_spec.md SS2, SS11). No model is loaded here --
    # OpenRouter is a cloud call, so this has no memory/startup-time cost.
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY not set -- see .env.example")
    app.state.llm_client = OpenRouterClient()
    yield


app = FastAPI(title="Supply Chain Agentic AI", lifespan=lifespan)
app.include_router(explanations.router)
app.include_router(runs.router)
