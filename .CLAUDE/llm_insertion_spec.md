# LLM Insertion Spec — On-Demand Explanation via OpenRouter (Cloud, No Local Download)

**Scope:** On-demand explanation of a prediction result (whole-run or single-agent), using a **deterministic, fully-grounded draft** built from `AGENT_OUTPUT_MEANINGS.md`, optionally polished by a free OpenRouter model. **No model is downloaded or run locally.** Not automatic, not a pipeline node, never runs during a sweep.
**Supersedes:** the local-BART version of this spec entirely.
**Owner:** whoever picks this up.
**Prerequisites:** API spec built. `AGENT_OUTPUT_MEANINGS.md` (provided) is the grounding source of truth, transcribed below. An OpenRouter account and free-tier API key.

---

## 0. Why the draft-first design is kept even though the model is now cloud-hosted

Going cloud removes the reason to *require* the draft-then-summarize workaround BART needed (an instruction-following chat model can genuinely read a grounding definition and explain it, unlike a summarization-only model). But the draft is kept anyway, for a better reason: it turns the LLM call into an optional **polish** step instead of the sole source of truth.

```
grounding.py (real field/value meanings, from AGENT_OUTPUT_MEANINGS.md)
      +
actual run values (from predictions / agent_traces)
      ↓
draft_builder.py — plain Python string templating, deterministic, no LLM, ALWAYS correct
      ↓
client.py — OpenRouter is asked to rephrase the draft into smoother prose,
             explicitly instructed not to add any fact not already in the draft
      ↓
returned explanation (cached) — falls back to the raw draft if the API call fails for any reason
```

This gets you three things at once: an instruction-following model that can genuinely produce better prose than a bare template, a hard guarantee that no fact in the final output can be invented (the model is polishing, not originating), and **graceful degradation** — if OpenRouter is down, rate-limited, or the venue's internet drops mid-demo, the endpoint still returns a fully correct explanation instead of erroring.

---

## 1. Correction from earlier specs — Agent 4 does not exist

Per your doc: **Agent 4 "Routing" was cancelled.** Pipeline order is A1 → A2 → A3 → A5 → A6, five agents. This matches `src/orchestrator/nodes/` (no `agent_4_routing.py`) — no code change needed, just confirming `grounding.py` carries no stale Agent 4 entry.

---

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | OpenRouter free-tier model (e.g. `meta-llama/llama-3.1-8b-instruct:free`) | No download, no local compute burden — the actual reason for this update |
| Role of the model | **Polisher**, not source of truth | The draft is always built first and is always correct on its own; the model only improves phrasing |
| Grounding source | `AGENT_OUTPUT_MEANINGS.md`, transcribed into `grounding.py`, unchanged from the prior version | Nothing about your real definitions changes based on which model reads them |
| Trigger | On-demand only, one endpoint | Unchanged |
| Scope | Whole-run or single named agent | Unchanged — matches your original Agent 6 grade example |
| Caching | Stored on first request per `(run_id, agent_name)` | Unchanged |
| Skip-polish rule | If the draft is under ~40 words, return it unpolished | A short draft gains little from rephrasing and it saves a network round-trip |
| **Failure handling** | **Fall back to the raw draft**, not an HTTP error | Network failure, rate limit, or bad key should degrade gracefully — the draft alone is already a correct, complete answer |
| API key | `OPENROUTER_API_KEY`, env var / `.env`, validated at **startup** | Fail fast and loudly if missing, rather than discovering it on the first user click |

---

## 3. New table: `llm_explanations` (additive migration — same discipline as `queue_migration_spec.md`)

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK autoincrement |
| `run_id` | TEXT | NOT NULL, FK → `pipeline_runs.run_id` |
| `agent_name` | TEXT | NULL — `NULL` means whole-run explanation |
| `draft_text` | TEXT | NOT NULL — the deterministic, fully-grounded draft, always stored |
| `final_text` | TEXT | NOT NULL — what's actually returned: the polished text, or the draft itself on skip/fallback |
| `was_polished` | INTEGER | NOT NULL, CHECK in (0, 1) |
| `fallback_reason` | TEXT | NULL — `'short_draft'`, `'api_error'`, or `NULL` if genuinely polished |
| `model_used` | TEXT | NOT NULL — e.g. `'meta-llama/llama-3.1-8b-instruct:free'`, or `'template-only'` |
| `created_at` | DATETIME | NOT NULL, default UTC now |

UNIQUE on (`run_id`, `agent_name`). Storing `draft_text` alongside `final_text` is what lets you demonstrate, concretely, that every fact traces to a deterministic source — regardless of whether the polish step ran.

Migration procedure: identical to `queue_migration_spec.md` §5 — baseline `pytest -v`, register on the existing `Base`, `create_all`, verify directly via `sqlite3`, regression gate again.

---

## 4. Files to create

```
src/
└── llm/
    ├── __init__.py
    ├── grounding.py           # transcribed AGENT_OUTPUT_MEANINGS.md — unchanged, see §5
    ├── draft_builder.py       # deterministic template assembly — unchanged, see §6
    ├── prompts.py             # NEW — the "polish this draft, don't add facts" instruction template
    ├── client.py              # OpenRouterClient — HTTP call, no local model
    ├── models.py              # LLMExplanation SQLAlchemy model, joins existing Base
    └── explainer_service.py   # public entrypoint: explain(run_id, agent_name=None)

src/api/routers/
    └── explanations.py         # POST /predictions/{run_id}/explain

tests/
└── llm/
    ├── __init__.py
    ├── test_grounding.py
    ├── test_draft_builder.py       # the important one — asserts facts, not fluency
    ├── test_explainer_service.py   # stub client, no real network call
    └── test_explanations_router.py
```

---

## 5. `grounding.py` — transcribed from `AGENT_OUTPUT_MEANINGS.md` (unchanged from prior version)

```python
"""
Grounding data for LLM explanations. Source of truth: AGENT_OUTPUT_MEANINGS.md.
Do not add any meaning here that isn't traceable to that document.
"""

AGENT_GROUNDING = {

    "agent_1_demand": {
        "shipped_fields": ["demand_forecast"],
        "not_shipped": ["demand_velocity_band", "stockout_risk"],
        "demand_forecast": {
            "description": (
                "Point forecast of units expected to sell over the next 6 months "
                "(sales_6_month), rounded to 2 decimals, never negative."
            ),
        },
    },

    "agent_2_risk": {
        "shipped_fields": ["backorder_prob", "alarm_triggered"],
        "backorder_prob": {
            "description": "Probability this SKU will go on backorder / stock out. Higher = riskier.",
        },
        "alarm_triggered": {
            "description": (
                "Operational high-risk flag. True when the underlying probability >= 0.945 "
                "(the F2-optimized threshold, tuned to weight missed stockouts as costlier than false alarms). "
                "This is NOT the generic 0.5 midpoint — there is a real band between 0.5 and 0.945 where the "
                "model leans toward backorder but is not confident enough to trigger action."
            ),
            "threshold": 0.945,
        },
    },

    "agent_3_rebalancer": {
        "shipped_fields": ["urgency_score", "recommended_qty", "batch_rank", "top_priority_skus"],
        "urgency_score": {
            "description": (
                "Composite urgency to restock, 0.0-1.0. Composed of: inventory gap below safety floor (40%), "
                "Agent 2's backorder probability (35%), depletion rate (15%), safety-stock urgency (10%). "
                "There are no discrete tiers in the model itself — any critical/high/medium/low banding is "
                "an interpretation layer, not a model output."
            ),
        },
        "recommended_qty": {
            "description": (
                "Units needed to bring stock to the safety floor: max(min_bank - national_inv, 0). "
                "0 means already at or above the safety floor; NaN is an intentional passthrough from "
                "NaN raw inputs, not an error."
            ),
        },
        "batch_rank": {
            "description": "This SKU's urgency rank within the current batch only, not a global rank.",
        },
        "manufacture_rank": {
            "description": (
                "Always null today — the global reference-distribution artifact needed to compute it "
                "does not exist yet. Permanently unusable in the current system, not '1' or 'no ranking needed'."
            ),
        },
        "top_priority_skus_membership": {
            "description": (
                "A SKU appears here when recommended_qty > 0 AND it is in a critical state: months of stock "
                "remaining is less than lead time, it is already below safety floor, and forecasted demand "
                "exceeds available plus incoming inventory. Means this SKU will run out before a new order "
                "could even arrive — restock immediately."
            ),
        },
    },

    "agent_5_forecast_opt": {
        "shipped_fields": ["correction_factor", "adjusted_forecast_3m", "bias_detected",
                           "risk_override_applied", "recommendation"],
        "correction_factor": {
            "description": (
                "Multiplier applied to the human 3-month forecast, range 0.3-1.5. Below 1.0 means the human "
                "forecast was too high and is corrected downward; above 1.0 means it was too low and is "
                "corrected upward; exactly 1.0 means no correction, OR that Agent 2's risk override forced it."
            ),
        },
        "bias_severity": {
            "values": {
                "NONE": "correction_factor >= 0.90 — no material bias in the human forecast.",
                "MILD": "0.75 <= correction_factor < 0.90 — moderate overestimate, worth noting.",
                "SEVERE": "correction_factor < 0.75 — large overestimate, human forecast significantly too high.",
            },
        },
        "bias_detected": {
            "description": (
                "True when bias_probability >= 0.765 — classified as a chronic over-forecaster "
                "(training rule: forecast >15% above actual sales historically)."
            ),
            "threshold": 0.765,
        },
        "risk_override_applied": {
            "description": (
                "True means Agent 2 flagged this SKU as high backorder risk (alarm_triggered), which forces "
                "correction_factor back to 1.0 regardless of what the corrector model predicted — a deliberate "
                "rule: never shrink the forecast for a SKU already at risk of stocking out. If Agent 2 was "
                "skipped for this SKU, Agent 5 is skipped too and correction_factor defaults to 1.0 with no "
                "bias check performed at all."
            ),
        },
        "recommendation": {
            "values": {
                "REDUCE_PLAN": "correction_factor < 0.90 — cut the production/purchasing plan.",
                "INCREASE_PLAN": "correction_factor > 1.10 — raise the production/purchasing plan.",
                "HOLD": "0.90-1.10 — forecast is roughly accurate, no action needed.",
            },
        },
    },

    "agent_6_auditor": {
        "shipped_fields": ["supplier_risk"],
        "supplier_risk": {
            "description": "Supplier risk grade, letter A-D, based on the model's predicted stop_auto_buy probability.",
            "values": {
                "A": "[0.00, 0.20) — Approved, no action needed.",
                "B": "[0.20, 0.45) — Watch, schedule a contract review.",
                "C": "[0.45, 0.70) — At Risk, escalate to procurement.",
                "D": "[0.70, 1.00] — Critical, the band where auto-buy can actually be stopped.",
            },
        },
        "stop_auto_buy_triggered": {
            "description": (
                "True if EITHER the model probability crosses the decision threshold, OR "
                "(supplier_grade == D AND delivery_stress > 0.5) — a rule-based safety net for severe "
                "overdue-delivery backlogs the model alone might miss. Means: halt automatic purchase orders "
                "from this supplier, human review required."
            ),
        },
        "trigger_reason": {
            "values": {
                "MODEL+RULE": "Both the ML model and the business rule agree — strongest stop signal.",
                "MODEL": "Only the ML probability crossed threshold.",
                "RULE": "Only the business rule fired (grade D + high delivery stress) despite the model "
                        "probability being below threshold — catches a failure pattern the model alone missed.",
                "NONE": "Neither condition fired — not flagged.",
            },
        },
    },
}
```

---

## 6. `draft_builder.py` — the deterministic explanation (unchanged from prior version)

**This function IS the explanation.** The LLM only rewords it. Get this right and the rest is safe by construction.

```python
def build_agent_draft(agent_name: str, field_values: dict) -> str:
    """
    Assembles a plain-English draft from AGENT_GROUNDING[agent_name] + the actual values
    for this run. Pure string templating — no model call, no randomness.
    """
```

Example for the exact case you raised — Agent 6, grade:

```python
grade = field_values["supplier_risk"]
grounding = AGENT_GROUNDING["agent_6_auditor"]["supplier_risk"]
meaning = grounding["values"][grade]
draft = f"This supplier received grade {grade}. {meaning}"
```

```python
def build_run_draft(sku_id: str, predictions_row: dict, suppression_note: str | None) -> str:
    """
    Whole-run version. Walks the reduced PipelineState fields (demand_forecast, backorder_prob,
    alarm_triggered, urgency_score, correction_factor, supplier_risk), looks up each one's
    grounding, and assembles a short paragraph. If suppression_note is present (from
    agent_traces), explicitly states the override using agent_5_forecast_opt's
    risk_override_applied grounding text.
    """
```

**Rule enforced by this file, checked in tests:** every number or category that appears in the draft must have come from `field_values` (real run data) or `AGENT_GROUNDING` (the real doc). No free-text commentary that isn't traceable to one of those two sources.

---

## 7. `prompts.py` — NEW, the polish instruction

Unlike a summarization model, an instruction-following model can be told explicitly not to invent anything. Use that directly:

```python
POLISH_PROMPT = """You are rephrasing an already-correct explanation for a supply chain planner. \
Rewrite the following draft in clear, natural prose. \

RULES — follow exactly:
- Do NOT add any fact, number, category, or claim that is not already present in the draft below.
- Do NOT change any number.
- You may reorder sentences, improve flow, and adjust tone for a professional but approachable reader.
- Keep it to 2-4 sentences.

Draft:
{draft}

Rewritten explanation:"""
```

The instruction to never add a fact is the load-bearing line — it's what makes this safe to use even though the model is instruction-following and could otherwise "helpfully" elaborate with invented specifics.

---

## 8. `client.py` — OpenRouter wrapper (cloud, no download)

```python
class OpenRouterClient:
    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct:free", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]   # fail at construction if missing
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def polish(self, draft: str) -> str:
        """
        POST to base_url:
          headers: {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
          body: {"model": self.model, "messages": [{"role": "user", "content": POLISH_PROMPT.format(draft=draft)}]}
        Returns response["choices"][0]["message"]["content"].
        Set a request timeout (e.g. 15s).
        Raises LLMUnavailableError on any network failure, non-200 status, or HTTP 429 (rate limit) —
        the CALLER (explainer_service) catches this and falls back to the draft; this class itself
        does not swallow the error, it just raises a clean, specific exception.
        """
```

**Config, not code:** `OPENROUTER_API_KEY` in a `.env` file (gitignored) or environment variable. Add `OPENROUTER_API_KEY=` to a `.env.example` so teammates know it's required, without the real key ever being committed.

No model download, no local weights, no GPU/CPU device concerns — this is the entire benefit of this version over the BART one.

---

## 9. `explainer_service.py` — the public entrypoint, with graceful fallback

```python
def explain(session: Session, client: OpenRouterClient, run_id: str, agent_name: str | None = None) -> dict:
```

Sequence:
1. Check cache (`llm_explanations` by `(run_id, agent_name)`) — return immediately if found.
2. Pull real values from `predictions` (+ `agent_traces.output` for agent-specific fields not in the reduced `PipelineState`, e.g. `bias_severity`, `trigger_reason`).
3. Build the draft via `draft_builder.py`. **This must succeed before any network call is attempted** — if the draft can't be built (unknown agent, missing grounding), raise here; this is a request-shape error (400), not a fallback case.
4. If the draft is short (§2 rule): skip the network call entirely. `was_polished=False`, `fallback_reason='short_draft'`, `model_used='template-only'`.
5. Otherwise, call `client.polish(draft)`.
   - **On success:** `was_polished=True`, `fallback_reason=None`, `model_used='meta-llama/llama-3.1-8b-instruct:free'`.
   - **On `LLMUnavailableError` (network failure, rate limit, bad key):** catch it here, fall back to the raw draft. `was_polished=False`, `fallback_reason='api_error'`, `model_used='template-only'`. **Do not raise past this point** — the draft is a complete, correct answer on its own.
6. Store `draft_text`, `final_text`, and all flags in `llm_explanations`.
7. Return `{"explanation": final_text, "cached": False, "was_polished": ..., "fallback_reason": ..., "model_used": ...}`.

This is the key behavioral difference from a pure-cloud design with no draft: **a network problem during your demo produces a slightly-less-polished but still fully correct explanation, not an error message.**

---

## 10. Endpoint — `explanations.py`

**`POST /predictions/{run_id}/explain`**, optional `agent_name` query param.
- 404 if `run_id` unknown.
- 400 if `agent_name` given but not in `AGENT_GROUNDING`, or draft-building otherwise fails.
- **No 503 case** — per §9, an API/network failure degrades to the draft rather than erroring. If you want the frontend to be able to show a subtle "showing basic explanation, network unavailable" note, surface `fallback_reason` in the response body for it to check — but the HTTP status stays 200 either way.

---

## 11. Startup — validate the key, don't load a model

In `src/api/main.py`'s `lifespan`:

```python
if "OPENROUTER_API_KEY" not in os.environ:
    raise RuntimeError("OPENROUTER_API_KEY not set — see .env.example")
app.state.llm_client = OpenRouterClient()
```

No model loading step, no memory footprint from this component, no download step to run ahead of time. This entire section is lighter than the equivalent step in the local-BART version.

---

## 12. Tests

### `test_grounding.py`
1. Every agent name in `agent_traces` has a top-level key in `AGENT_GROUNDING`.
2. No `agent_4` key exists anywhere.
3. Every categorical field's real value set (Agent 5's `bias_severity`/`recommendation`, Agent 6's `supplier_risk`/`trigger_reason`) is fully represented, cross-checked against `AGENT_OUTPUT_MEANINGS.md`'s tables.

### `test_draft_builder.py` — unchanged, still the most important file
1. Agent 6 grade `D` → draft contains `"Critical"` and the literal `"[0.70, 1.00]"` threshold text.
2. Agent 2 alarm case → draft mentions `0.945` explicitly, not the generic `0.5`.
3. Agent 5 `risk_override_applied=True` → draft explicitly states the override reason.
4. Every number in a draft for a fixed test input matches either `field_values` or a literal in `AGENT_GROUNDING` — explicit assertion.
5. A draft under the word-count threshold is correctly flagged for skip.

### `test_explainer_service.py` (stub `OpenRouterClient` — no real network call)
1. First call stores a row; second identical call returns cached, stub not called again.
2. Stub configured to raise `LLMUnavailableError` → `explain()` returns the **draft** as `final_text`, `was_polished=False`, `fallback_reason='api_error'`, **no exception propagates**.
3. Short draft → stub's `polish` is never called at all (assert call count 0), `fallback_reason='short_draft'`.
4. Unknown agent name → raises before the stub is constructed or called.

### `test_explanations_router.py`
1. Whole-run and per-agent requests both return 200.
2. Unknown `run_id` → 404. Unknown `agent_name` → 400.
3. Simulate the client raising → endpoint still returns 200 with `fallback_reason='api_error'` in the body.

---

## 13. Acceptance criteria

- [ ] No LLM/network call occurs inside `sweep_service`, `pipeline_service`, or any orchestrator node.
- [ ] `grounding.py` matches §5 exactly, no Agent 4 entries.
- [ ] `draft_builder.py`'s output is provably fact-traceable per `test_draft_builder.py` assertion 4.
- [ ] `OPENROUTER_API_KEY` validated at startup; app fails to start if missing.
- [ ] `.env` gitignored; `.env.example` present with the key name only.
- [ ] Short drafts skip the network call entirely (verified by call-count assertion, not just documented).
- [ ] An API/network failure produces a 200 with the draft as fallback — never a 500 or 503.
- [ ] `llm_explanations` added via additive migration; existing tables/data verified unaffected.
- [ ] `pytest -v` full repo ≥ pre-spec baseline.
- [ ] No model download step anywhere in the setup instructions.

---

## 14. Explicitly out of scope

- Any local model (BART, Ollama, etc.) — deliberately dropped per your hardware constraint.
- Automatic/pipeline-integrated summarization.
- Streaming responses.

---

## 15. Notes for the paper

Worth a paragraph: rather than relying on an LLM's internal reasoning for factual accuracy, the system generates every explanatory fact deterministically from a documented, code-verified grounding table, and uses a cloud-hosted instruction-following model solely to improve prose quality — explicitly constrained by prompt instruction from adding new facts. This preserves the same reproducibility guarantee as the rest of the pipeline (the same run produces the same draft, always) while adding a genuine reliability property worth naming directly: the explanation layer degrades gracefully under network failure, since the deterministic draft is itself a complete, correct answer and is served automatically if the cloud call fails. This is a stronger operational claim than most LLM-augmented systems make, which typically fail outright when their model dependency is unavailable.
