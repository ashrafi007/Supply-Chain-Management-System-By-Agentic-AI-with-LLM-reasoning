# Daily Report — 2026-07-27

## TL;DR

Built and shipped the full on-demand LLM explanation layer (grounded drafts + OpenRouter polishing + graceful fallback), stood up the first API layer for the project (FastAPI, previously nonexistent), scaffolded read endpoints for the upcoming frontend, and root-caused a Supplier Auditor grading concern down to a genuine 96.3% base rate in the training data rather than a bug. Test suite grew from **154 → 179 passing**, zero regressions. Nothing is committed to git yet (still zero commits repo-wide).

---

## 1. Project status audit

Reviewed the full repo end to end (~140 tracked files, none yet committed to git) and produced two reference documents:

- **`PROJECT_FILE_PATHS.md`** — every tracked file, grouped by directory.
- **`AGENT_OUTPUT_MEANINGS.md`** — for all 5 shipped agents (Demand Predictor, Risk Detector, Inventory Rebalancer, Forecast Optimizer, Supplier Auditor), every output field and every possible value/band/class it can take, in plain business language, cross-checked against the training notebooks and the actual `inference_tools/` code (not just the specs). Built via 5 parallel research agents, one per notebook.
  - Flagged a known production gap while doing this: `demand_velocity_band` and `stockout_risk` are fully defined in the Demand Predictor notebook but were never ported into the production wrapper — they return `None` today.

Also answered a question about the 7-day order queue table: confirmed `order_queue` / `order_queue_log` (`src/queue/`) is the real table, built the day before, with the 7-day window logic in `queue_migration_spec.md`.

---

## 2. LLM explanation layer (`llm_insertion_spec.md`) — fully implemented

Read the spec, confirmed a real prerequisite gap (it assumed an API layer that didn't exist), and built that first per your direction, then the full spec on top.

### API layer (new — `src/api/`)
- `main.py` — minimal FastAPI app; `lifespan` validates `OPENROUTER_API_KEY` at startup and refuses to start if missing (verified live).
- `deps.py` — `get_db` (per-request session), `get_llm_client`.
- `routers/explanations.py` — `POST /predictions/{run_id}/explain`.

### LLM module (new — `src/llm/`)
- `grounding.py` — the documented meaning of every agent output field/value, transcribed from `AGENT_OUTPUT_MEANINGS.md`. **One deliberate correction from the spec text**: used the real `agent_traces.agent_name` values (`demand_predictor`, `risk_detector`, etc.) as dict keys instead of the spec's `agent_1_demand`/etc, since that's what the code and the spec's own acceptance test require.
- `draft_builder.py` — deterministic, fully fact-traceable draft assembly per agent + whole-run summaries, with suppression-note handling.
- `prompts.py` — the "polish, never invent a fact" instruction template.
- `client.py` — `OpenRouterClient` (cloud call via `httpx`, no local model, no download).
- `models.py` — `LLMExplanation` SQLAlchemy model.
- `explainer_service.py` — the public `explain()` entrypoint: cache check → pull real values from `predictions` + `agent_traces.output` → build draft → skip-polish for short drafts → polish with graceful fallback to the draft on any API failure → cache.

### Database
- `llm_explanations` table added via the same additive-migration discipline as the queue feature (baseline pytest → register on `Base` → `create_all` → verify via `sqlite3` → regression gate). No existing table touched; verified schema directly.

### Tests — 25 new, all passing
- `tests/llm/` (20): grounding consistency, draft fact-traceability, explainer service (caching, API-failure fallback, short-draft skip, unknown-agent/unknown-run errors — all with a stub client, no real network), and the explain endpoint (200/404/400, fallback surfaced in body).
- `tests/api/` (5, added later — see §4).

### Verified live, not just by test
- App genuinely refuses to start without `OPENROUTER_API_KEY`.
- Ran two real SKUs (1111949, 1113049) through the actual 5-agent pipeline and had the explanation layer summarize both — output matched the underlying `predictions`/`agent_traces` rows exactly. New file: `run_llm_explanation_example.py`.

---

## 3. Security incident

You pasted a live Hugging Face API token in chat along with `facebook/bart-large-cnn` loading code. Flagged it immediately as compromised (recommended revoking at huggingface.co/settings/tokens), did not write it to any file, and did not use it — confirmed with you that the OpenRouter-based design stays as-is and BART is not being reintroduced (the spec explicitly dropped local BART for this exact hardware-constraint reason).

---

## 4. Read API endpoints (new — for the upcoming frontend)

Discussed whether starting frontend work now would be risky given a model retrain planned for tomorrow. Conclusion: not risky as long as the frontend consumes the API/DB contract rather than model internals — but there were no read endpoints yet at all (only the explain endpoint existed). Scaffolded:

- **`GET /runs`** — list recent runs, optional `?sku_id=` filter, `?limit=` (default 20, max 100).
- **`GET /runs/{run_id}`** — full detail: run status/timing + its prediction row (`null` if the run failed before producing one) + all agent traces in order.

New files: `src/api/schemas.py` (response models, separate from `inference_tools/schemas.py`'s agent I/O contracts), `src/api/routers/runs.py`. 5 new tests in `tests/api/`. Smoke-tested live against the real `data/app.db` — correctly returned the actual runs from this session.

---

## 5. Supplier Auditor investigation

You noticed both demo SKUs graded "D" (critical) and asked to check the output. Investigation across 300 real SKUs (using the actual preprocessing path, not a shortcut) showed **91% grade D, 100% flagged for `stop_auto_buy`** — looked like a bug at first.

Root-caused properly, with two false starts corrected along the way:
1. First hypothesis (wrong): `went_on_backorder` defaults to 0 for every SKU. Checked feature importance — it's 0.0041, third-from-last of 35 features. Not the cause.
2. Second check confirmed the real cause: the model's own training label, `stop_auto_buy`, is **96.34% positive across the entire 1.93M-row source dataset** — confirmed directly from the training notebook's own printed output, matching an independent check of the raw CSV. The model's held-out test-set report (precision 0.81 / recall 0.72 on the rare 3.7% minority class, ROC-AUC 0.98, MCC 0.76) shows it's a legitimately well-calibrated model given that imbalance, not degenerate.

**Conclusion: not a bug.** The grading engine, feature engineering, and model are all working as designed. The near-universal D grade faithfully reflects the training data. Flagged as a possible semantic concern for tomorrow: `stop_auto_buy` may reflect "auto-reorder is disabled" (which naturally applies to slow/dead SKUs) more than "this supplier is untrustworthy" — worth understanding before the retrain, not something to silently patch.

Also discussed the mechanics of swapping in tomorrow's retrained model: same pickle shape (`model`/`preprocessor`/`threshold`/`feature_cols`) dropped at the same path requires zero code changes (the loader is `functools.lru_cache`d per-process, so a running server needs a restart to pick it up); a changed feature set would require updating `REQUIRED_RAW_COLS`/`ENGINEERED_COLS`/`engineer_features()` in `inference_tools/supplier_auditor_tool.py` to match.

---

## Test suite growth

| Point in the day | Passing |
|---|---|
| Start of session (baseline) | 154 |
| After `llm_insertion_spec.md` implementation | 174 |
| After read API endpoints | **179** |

Zero regressions at any point.

---

## Files created today

```
AGENT_OUTPUT_MEANINGS.md
PROJECT_FILE_PATHS.md
.env.example
run_llm_explanation_example.py

src/api/__init__.py
src/api/main.py
src/api/deps.py
src/api/schemas.py
src/api/routers/__init__.py
src/api/routers/explanations.py
src/api/routers/runs.py

src/llm/__init__.py
src/llm/grounding.py
src/llm/draft_builder.py
src/llm/prompts.py
src/llm/client.py
src/llm/models.py
src/llm/explainer_service.py

tests/llm/__init__.py
tests/llm/conftest.py
tests/llm/test_grounding.py
tests/llm/test_draft_builder.py
tests/llm/test_explainer_service.py
tests/llm/test_explanations_router.py

tests/api/__init__.py
tests/api/conftest.py
tests/api/test_runs_router.py
```

**Modified:** `requirements.txt` (fastapi, uvicorn, httpx, python-dotenv), `scripts/run_migration.py` (registers `LLMExplanation`), `data/app.db` (new `llm_explanations` table).

---

## Open items for tomorrow

- [ ] Rotate the Hugging Face token that was pasted in chat, if not already done.
- [ ] Retrain/update the Supplier Auditor notebook and model. When ready, artifact swap is a same-path pickle drop (see §5) unless the feature set changes.
- [ ] Set a real `OPENROUTER_API_KEY` in `.env` to see actual LLM-polished explanations (everything so far has run in graceful-fallback/draft-only mode).
- [ ] Port `demand_velocity_band`/`stockout_risk` computation into `inference_tools/demand_predictor_tool.py` if those fields are needed (currently notebook-only, flagged earlier).
- [ ] First git commit is still outstanding — nothing in the repo is version-controlled yet.
