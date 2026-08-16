"""
FastAPI application backing the frontend: pipeline runs/predictions, LLM
explanations, the order_queue (list/enqueue/sweep), suppliers, SKU onboarding,
and dashboard stats.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import db, explanations, queue, runs, skus, stats, suppliers
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

# CORS_ORIGINS: comma-separated allowlist, e.g. "http://localhost:3000,https://app.example.com".
# Defaults to "*" (any origin) for local dev -- without this, every browser request from a
# frontend on a different origin (which is the normal case) is blocked before it ever reaches
# a route. allow_credentials is forced off when the origin list is "*": browsers reject the
# combination of a wildcard origin with credentialed requests outright, so there is nothing to
# gain by setting it True here -- set explicit origins in CORS_ORIGINS if cookies/auth headers
# with credentials are ever needed.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = ["*"] if _cors_origins_env == "*" else [o.strip() for o in _cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(explanations.router)
app.include_router(runs.router)
app.include_router(queue.router)
app.include_router(suppliers.router)
app.include_router(skus.router)
app.include_router(stats.router)
app.include_router(db.router)
