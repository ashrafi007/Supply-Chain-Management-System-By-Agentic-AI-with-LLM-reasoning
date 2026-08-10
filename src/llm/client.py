"""OpenRouter wrapper -- cloud call, no local model, no download (llm_insertion_spec.md SS8)."""

from __future__ import annotations

import os

import httpx

from src.llm.prompts import POLISH_PROMPT

# meta-llama/llama-3.1-8b-instruct:free was retired by OpenRouter (now 404s -- paid-only
# slug replaced it). Swapped 2026-08-10 to a currently-live free-tier model; verified
# directly against OpenRouter's /models endpoint before pinning. Re-verify if this
# ever starts 404ing too -- free-tier model availability changes on OpenRouter's side,
# not this codebase's.
DEFAULT_MODEL = "openai/gpt-oss-20b:free"
_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT_SECONDS = 15.0


class LLMUnavailableError(RuntimeError):
    """Any network failure, non-200 status, or rate limit (HTTP 429). The caller
    (explainer_service) catches this and falls back to the raw draft -- this class
    itself does not swallow the error, it just raises a clean, specific exception."""


class OpenRouterClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]  # fail at construction if missing
        self.base_url = _BASE_URL

    def polish(self, draft: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": POLISH_PROMPT.format(draft=draft)}],
        }
        try:
            response = httpx.post(self.base_url, headers=headers, json=body, timeout=_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code == 429:
            raise LLMUnavailableError("OpenRouter rate limited (HTTP 429)")
        if response.status_code != 200:
            raise LLMUnavailableError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMUnavailableError(f"OpenRouter response malformed: {exc}") from exc
