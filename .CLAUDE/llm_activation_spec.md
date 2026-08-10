# LLM Activation Spec — Turning On Real OpenRouter Polishing

**Scope:** Activate the already-built LLM explanation layer (`.CLAUDE/llm_insertion_spec.md`) end to end — obtain a real `OPENROUTER_API_KEY`, run on the free tier to prove the wiring, then move to a paid model when ready. Also covers automating explanation generation across an entire `order_queue` sweep (Phase D), since on-demand-only stops being enough once you want every SKU explained without a manual per-run_id call. **Phase A/B/C needed no missing code** — `src/llm/*`, `src/api/*`, and `llm_explanations` already existed and passed their tests; only `.env` + a retired free-model slug needed fixing. **Phase D added one new function** (`sweep_service.run_sweep_and_explain`) plus a CLI script, both additive.
**Owner:** whoever owns the `.env` / deployment secrets for this project.
**Prerequisites:** `.CLAUDE/llm_insertion_spec.md` fully implemented (it is — see `SESSION_SUMMARY.md`). `python -m pytest -v` green on current `main`.
**Supersedes:** nothing — this is additive throughout. Phase A/B are config-only, Phase C is one optional file edit, Phase D is one new function + one new script, none of which touch `run_sweep()`'s existing behavior or its test guarantees.

---

## 0. The one rule this entire spec exists to enforce

Run the full suite before touching anything, and again after every phase. LLM tests are mocked (`tests/llm/`, `tests/api/`), so they must stay green regardless of whether a real key is present — if a phase breaks them, stop.

```bash
source venv/bin/activate
python -m pytest -v
```

Record the pass count now (179 baseline; 183 after Phase D's new tests — see §6). It must not go down at any phase gate below.

---

## 1. What must NOT change

- `src/llm/client.py`, `draft_builder.py`, `grounding.py`, `explainer_service.py`, `prompts.py` — the draft-first, fact-grounded, graceful-fallback design is correct as built. This spec does not touch model logic.
- `src/db/models.py` / the `llm_explanations` table — already migrated, do not re-run `create_db.py --force`.
- No test file is edited to make activation "pass." A real key either works or it doesn't; the mocked tests are a separate concern and stay as-is.

---

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Key provider | OpenRouter (one account, one key) | Already the wired provider — `client.py` speaks OpenRouter's REST shape specifically, not a generic OpenAI-compatible client |
| Free → paid transition | **Same key**, just add billing credits later | OpenRouter does not issue separate free/paid keys — access to paid models is gated by account balance, not key type |
| Where the key lives | `.env` at repo root (gitignored), loaded via `python-dotenv` in `src/api/main.py` and `run_llm_explanation_example.py` | Already implemented — nothing to build |
| Model selection | Phase A/B: keep hardcoded `DEFAULT_MODEL` in `client.py`. Phase C (optional): promote to `OPENROUTER_MODEL` env var | Lets you flip free→paid with a config change, no redeploy-with-code-edit, once you're ready |
| Verification order | free key → confirm wiring → add credits → swap model string → confirm paid polish | Never debug "is my key valid" and "is the paid model better" at the same time |

---

## 3. Phase A — Get a free key and prove the wiring works

### 3.1 How to find / create an OpenRouter API key

1. Go to **https://openrouter.ai** and sign in (GitHub, Google, or email — no payment info required for this step).
2. Click your account avatar (top right) → **Keys** — or go directly to **https://openrouter.ai/settings/keys**.
3. Click **Create Key**.
4. Give it a name (e.g. `supply-chain-agentic-ai-dev`) — naming it per-project makes it easy to revoke later without affecting other integrations.
5. Copy the key immediately — it's shown once, in the form `sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`. If you navigate away before copying, you must delete that key and create a new one (OpenRouter never re-displays a key).
6. Optional but recommended: click into the new key's row and set a **credit limit** (even `$0`/free-only while testing) — a hard ceiling that prevents surprise spend if the key ever leaks, independent of what your account balance later becomes.

No credit card is required to use the free-tier models (any model id suffixed `:free`, e.g. the one already configured in `client.py`).

### 3.2 Install the key locally

```bash
cd /Users/home/Documents/Projects/Updated-Supply-Chain-Agentic_AI
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Confirm it's ignored by git (it already is, per `.gitignore:4`):
```bash
git check-ignore -v .env
```

### 3.3 Verify — standalone script

```bash
python run_llm_explanation_example.py
```

**Pass criteria:**
- Printed line `Using OpenRouter model: meta-llama/llama-3.1-8b-instruct:free` (not the "running in fallback-only mode" message).
- For each SKU, `was_polished: True` and `fallback_reason: None`.
- If `was_polished: False` with `fallback_reason: api_error` — the key is present but the call is failing; the surrounding exception text (from `LLMUnavailableError` in `client.py`) states why (bad key, 429 rate limit, etc.).

### 3.4 Verify — through the API

```bash
uvicorn src.api.main:app --reload
```
Startup must **not** raise `RuntimeError("OPENROUTER_API_KEY not set ...")` (`main.py:26-27`). Then, with a real `run_id` from `pipeline_runs`:
```bash
curl -X POST http://localhost:8000/explanations/<run_id>
```
Inspect the response for `"was_polished": true`.

### 3.5 Verify — database

```bash
sqlite3 data/app.db "SELECT run_id, agent_name, was_polished, fallback_reason, model_used FROM llm_explanations ORDER BY id DESC LIMIT 5;"
```
Newly created rows should show `was_polished=1`, `fallback_reason` NULL, `model_used` matching whatever `DEFAULT_MODEL` currently resolves to (see §3's status note below if the originally-planned slug 404s).

### 3.6 Regression gate

```bash
python -m pytest -v
```
Still all green (mocked LLM tests are unaffected by a real key existing).

**Phase A is complete when 3.3–3.5 all pass.** This is enough for a demo — do not skip to Phase B unless you actually need paid-model prose quality.

> **Status: done (2026-08-10).** The key was valid on the first attempt (auth succeeded immediately), but `DEFAULT_MODEL`'s original slug (`meta-llama/llama-3.1-8b-instruct:free`) had been retired by OpenRouter — a 404 with `"This model is unavailable for free"`, not an auth failure. Queried OpenRouter's live `/models` endpoint, tested 3 currently-free candidates against the real key, and repointed `DEFAULT_MODEL` to `openai/gpt-oss-20b:free` (cleanest instruction-following of the three tried; `nvidia/nemotron-nano-9b-v2:free` also worked). §3.3–3.6 all passed after the swap. **Takeaway: if this model ever 404s again, it's OpenRouter retiring the slug, not a key/code problem — repeat the same query-and-swap process, don't assume the key broke.**

---

## 4. Phase B — Move to a paid model (when ready)

No new key. No code change required for the minimum version of this phase.

1. Add billing credits to the **same** OpenRouter account: https://openrouter.ai/settings/credits.
2. Pick a paid model. Check current pricing at https://openrouter.ai/models before committing — reasonable low-cost options for this use case (short polish-only prompts, `POLISH_PROMPT` already constrains it to not invent facts):
   - `anthropic/claude-3.5-haiku`
   - `openai/gpt-4o-mini`
3. Change `DEFAULT_MODEL` in `src/llm/client.py:11` to the chosen model id.
4. Re-run §3.3–3.6 verification. Same pass criteria, plus visually compare a few `final_text` values against Phase A's free-model output for prose quality.

---

## 5. Phase C (optional) — Env-driven model selection

Only do this if you expect to flip between free/paid models across environments (e.g. free in CI/dev, paid in a demo/prod) without a code edit + redeploy each time.

**Change, `src/llm/client.py`:**
```python
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
```

**Add to `.env.example`:**
```
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

No other file changes — `OpenRouterClient.__init__` already accepts `model` as an optional override, this only changes what the *default* resolves to. Re-run the full §3 verification (both with `OPENROUTER_MODEL` unset — must still default to the free model — and set to a paid id) plus `pytest -v` before considering this phase done.

---

## 6. Phase D — Automate: every SKU in a sweep gets explained, not just on-demand

**Problem this solves:** `llm_insertion_spec.md` §2 deliberately made LLM generation on-demand-only, one endpoint, never part of a sweep — so a bare `order_queue` sweep never depends on network/API availability. That's still the right default. But it means "run the queue, then look at every SKU's explanation" required a manual `POST /explanations/{run_id}` call per run_id, with no batch entrypoint. Phase D adds that batch entrypoint as an **opt-in layer on top of**, not a change to, the existing sweep.

### 6.1 Design decision

| Decision | Choice | Reason |
|---|---|---|
| Where the automation lives | New function `run_sweep_and_explain()` in `src/queue/sweep_service.py`, alongside the untouched `run_sweep()` | `run_sweep()`'s existing test suite asserts (structurally, via `StubOpenRouterClient` never being involved) that a bare sweep touches no network. A new function preserves that guarantee for any caller who still wants it, rather than adding an `explain: bool` flag that could accidentally default the wrong way later |
| Which runs get explained | Only `run_id`s where `pipeline_runs.status == "success"` | A failed run has no `predictions` row — `explainer_service.explain()` would raise `RunNotFoundError` for nothing gained; failed runs are silently skipped, not retried or surfaced as an explanation error |
| Scope per run | Whole-run explanation only (`agent_name=None`), not per-agent | Matches the existing example script's behavior; per-agent explanations remain available on-demand via the existing endpoint if ever needed |
| Caching | Unchanged — `explainer_service.explain()`'s own `(run_id, agent_name)` cache applies automatically | Re-running a sweep for the same day/run_id never re-calls the network for a run already explained |
| CLI entrypoint | New `scripts/run_sweep.py`, `--no-explain` flag to opt back out per-invocation | Mirrors `run_llm_explanation_example.py`'s graceful no-key fallback; `--no-explain` gives a network-free escape hatch without needing a second script |

### 6.2 What changed

- **`src/queue/sweep_service.py`** — added `run_sweep_and_explain(session, as_of, executor, llm_client) -> dict`. Calls `run_sweep()` verbatim, then loops its `run_ids`, explaining each successful one; returns the same dict as `run_sweep()` plus an `"explanations"` key (`{run_id: explain()-result}`).
- **`tests/queue/test_sweep_service.py`** — added `TestRunSweepAndExplain` (4 tests): explains + caches successful runs, skips failed runs (zero network calls), skips entirely when nothing is due (zero network calls), and a regression guard proving `run_sweep()` called directly still returns no `"explanations"` key at all.
- **`tests/fixtures/stub_executor.py`** — fixed a latent bug surfaced by the new tests: `default_success_fields()` set `supplier_risk="low"`, not a real `A`/`B`/`C`/`D` grade. Harmless everywhere it was previously used (nothing asserted that field), but `draft_builder.py` looks the grade up in `AGENT_GROUNDING` and `KeyError`s on anything else — the first test path to actually build a whole-run draft off this stub. Changed to `"A"`.
- **`scripts/run_sweep.py`** (new) — CLI: `python -m scripts.run_sweep [--as-of YYYY-MM-DD] [--no-explain]`. No script previously existed to invoke `run_sweep()` at all outside of tests.

### 6.3 Verify

```bash
python -m pytest tests/queue/test_sweep_service.py -v   # 9 passed (5 pre-existing + 4 new)
python -m pytest -v                                       # 183 passed, up from 179, zero regressions
```

Live smoke test (enqueue a real SKU due today, run the CLI, confirm explanation, clean the queue row back out):
```bash
python -m scripts.run_sweep
```
Verified 2026-08-10 against `data/app.db`: 1 SKU enqueued → swept → explained via `openai/gpt-oss-20b:free`, `was_polished: True`. Queue row removed afterward to leave `order_queue` at its pre-test state (0 rows).

### 6.4 Still a manual step: actually scheduling the sweep

Phase D makes *explanation* automatic once a sweep runs — it does not make the *sweep itself* run on a timer. Nothing in this repo currently invokes `run_sweep`/`run_sweep_and_explain` outside of tests and this new script; there is no cron job, systemd timer, or scheduler wired up. If "fully automatic, no human runs a command" is the actual goal, that's a separate, later decision (a daily cron calling `python -m scripts.run_sweep`, or an OS-level scheduler) — deliberately out of scope here so it doesn't get decided as a side effect of an LLM-activation spec.

---

## 7. Rollback

Deleting or blanking `OPENROUTER_API_KEY` in `.env` and restarting the API server immediately returns the app to the fail-fast startup check (`main.py:26-27`) — if you want graceful degradation instead of a hard failure during rollback, remove `.env` entirely and use `run_llm_explanation_example.py`'s script path, which tolerates a missing key (`run_llm_explanation_example.py:63-79`) by design; the API's `main.py` intentionally does not.

At the OpenRouter dashboard, revoking a key (Settings → Keys → delete) takes effect immediately — any in-flight `polish()` call after that point raises `LLMUnavailableError` and falls back to the raw draft per §0 of `llm_insertion_spec.md`, never a hard error to the caller.
