# Repository Layer Spec — Database ↔ Pipeline Glue

**Scope:** The module that reads a SKU into a `PipelineState`, hands it to the orchestrator, and persists the result. **No model logic. No graph logic.** Pure assembly and persistence.
**Owner:** Person A or B (whoever finishes first — this module has no ML dependency)
**Prerequisites:** Database seeded. All six agent wrappers verified (Layers 1–2 of `verification_spec.md`).
**Explicitly not a prerequisite:** the LangGraph orchestrator does not need to exist yet. This spec defines the *interface* the orchestrator must satisfy, so the two can be built independently and meet in the middle.

---

## 1. Why this exists as its own layer

Two things must never know about each other directly: the database, and the graph of agents. If the orchestrator wrote to SQLite itself, every agent would need database access, which contradicts the architecture — agents run purely in memory. If the API layer built `PipelineState` by hand per request, every caller would duplicate this logic.

This module is the only thing that touches both. Everything else either talks to the database (repository, API) or talks to the graph (orchestrator, API) — never both.

---

## 2. The interface boundary with the orchestrator

The orchestrator doesn't exist yet. This spec defines the contract it must implement, so this layer can be built and tested now with a stub, and the real orchestrator drops in later without changing a line here.

```python
class PipelineExecutor(Protocol):
    def invoke(self, state: PipelineState) -> PipelineResult:
        ...
```

Where `PipelineResult` is:

```python
class PipelineResult(TypedDict):
    final_state: PipelineState          # the state after all agents ran
    agent_events: list[AgentEvent]      # one per agent, in execution order
    manifest_version: str               # which model manifest produced this
```

```python
class AgentEvent(TypedDict):
    agent_name: str
    sequence: int
    status: Literal["success", "skipped", "failed"]
    latency_ms: int | None
    output: dict            # the state delta this agent returned
    note: str | None        # e.g. "suppressed: alarm_triggered=1"
```

This shape maps directly onto `agent_traces` — the orchestrator spec (next) must produce exactly this. Build a `StubExecutor` now that returns a hardcoded `PipelineResult` so this layer's tests don't block on the orchestrator existing.

---

## 3. Files to create

```
src/repository/
    __init__.py
    state_builder.py     # inventory_current row -> PipelineState
    run_repository.py    # persistence: pipeline_runs, predictions, agent_traces
    pipeline_service.py  # orchestrates the two above + calls the executor
tests/
    test_repository.py
    fixtures/
        stub_executor.py
```

---

## 4. `state_builder.py`

**Single responsibility: read one row, produce one `PipelineState`. No writes.**

```python
def build_initial_state(session: Session, sku_id: str) -> PipelineState | None:
```

- Query `inventory_current` for `sku_id`.
- If no row exists, return `None`. Do not raise — "SKU not found" is a normal, expected outcome the caller must handle, not an exceptional one.
- If found, map the row's columns into `PipelineState["raw_features"]` as a plain dict (not an ORM object — the graph must not hold a live database reference).
- Initialize the rest of `PipelineState`: all agent-output fields `None`, `trace = []`, `errors = []`.
- **Do not** run any cleaning or feature engineering here. That happens inside each agent wrapper via `features.py`. This function's only job is DB row → raw dict.

---

## 5. `run_repository.py`

**Single responsibility: persistence. No decisions about what the pipeline should do — only recording what it did.**

Three functions, used in sequence by `pipeline_service.py`:

### 5.1 `create_pending_run`
```python
def create_pending_run(session: Session, sku_id: str, manifest_version: str) -> str:
```
- Generate a UUID4 `run_id`.
- Insert into `pipeline_runs`: `status='pending'`, `started_at=now()`, `manifest_version`.
- Commit immediately — a pending row must exist before the graph runs, so a crash mid-execution still leaves a record rather than silently vanishing.
- Return `run_id`.

### 5.2 `persist_success`
```python
def persist_success(session: Session, run_id: str, sku_id: str, result: PipelineResult) -> None:
```
- Insert one `predictions` row from `result["final_state"]`, mapping every `PipelineState` field to its matching column **by name** — this mapping must never drift; if a field is added to `PipelineState`, it must be added here and to the `predictions` schema in the same change.
- Insert one `agent_traces` row per entry in `result["agent_events"]`, preserving `sequence`.
- Update `pipeline_runs`: `status='success'`, `completed_at=now()`, `latency_ms` computed from `started_at`.
- **Single transaction.** All of `predictions` + `agent_traces` + the `pipeline_runs` update commit together, or none do. A partially-written run is worse than a failed one — it looks successful in the runs table while missing its predictions.
- **Let CHECK-constraint violations propagate as a specific, catchable exception**, not a generic one. A violation here (e.g. `urgency_score` outside [0,1], or the suppression invariant broken) means an **agent produced an invalid output** — that's not a persistence failure, it's a correctness bug, and the caller must record it as a **failed run with a specific reason**, not silently drop the insert.

### 5.3 `persist_failure`
```python
def persist_failure(session: Session, run_id: str, error: str) -> None:
```
- Update `pipeline_runs`: `status='failed'`, `completed_at=now()`, `error=error`.
- Write **no** `predictions` row and **no** `agent_traces` rows for this run. A failed run has no result to record — write partial trace data only if the orchestrator explicitly separates "agents that ran before the failure" as their own event list; if so, those still get written, but `predictions` stays empty.
- This must be reachable from three distinct causes, each producing a distinguishable `error` string:
  1. SKU not found (`state_builder` returned `None`)
  2. The executor raised an unexpected exception
  3. A CHECK-constraint violation surfaced from `persist_success` (§5.2)

---

## 6. `pipeline_service.py`

**The single public entrypoint. Everything above is private to this module's use.**

```python
def run_pipeline_for_sku(session: Session, sku_id: str, executor: PipelineExecutor) -> str:
    """Returns run_id. Raises nothing — all outcomes are recorded in pipeline_runs."""
```

Sequence:

1. `state = build_initial_state(session, sku_id)`
2. If `state is None` → create a run row directly in `failed` status with `error="sku_not_found"` (skip the pending step; there's nothing to attempt). Return `run_id`.
3. Otherwise: `run_id = create_pending_run(...)`.
4. `try: result = executor.invoke(state)`
5. On success → `persist_success(session, run_id, sku_id, result)`. If this raises (constraint violation) → catch it, call `persist_failure` with the specific reason, and **re-raise nothing further** — the caller gets a valid `run_id` pointing at a `failed` row either way.
6. On any exception from `executor.invoke` → `persist_failure(session, run_id, error=str(exception))`.
7. Always return `run_id`. **This function must never raise.** A caller (the API layer, later) always gets a `run_id` back and can look up what happened — the failure is data, not an exception the caller has to handle specially.

This "never raises, always returns a run_id" contract is what makes the API layer trivial later: `POST /predict` calls this, gets a `run_id`, and returns it regardless of outcome; the client checks `status` via `GET /results/{run_id}`.

---

## 7. Batch variant

Add `run_pipeline_for_skus(session, sku_ids: list[str], executor) -> list[str]` — loops §6 per SKU. **Each SKU gets its own transaction.** One SKU's constraint violation must not roll back or block the others. This is what the batch-prediction endpoint (Person B's track) will call.

---

## 8. Tests (`tests/test_repository.py`)

Use the `StubExecutor` from §2 — these tests do not depend on the orchestrator existing.

1. `build_initial_state` on a known-seeded `sku_id` returns a populated `PipelineState` with `raw_features` matching the DB row.
2. `build_initial_state` on an unknown `sku_id` returns `None`.
3. `run_pipeline_for_sku` with a stub that returns a valid successful `PipelineResult` → assert `pipeline_runs.status='success'`, exactly one `predictions` row, `agent_traces` row count matches the stub's event list, `sequence` is contiguous.
4. `run_pipeline_for_sku` on an unknown SKU → assert `pipeline_runs.status='failed'`, `error='sku_not_found'`, **zero** `predictions` rows.
5. `run_pipeline_for_sku` with a stub whose `invoke` raises → assert `status='failed'`, error message captured, zero `predictions` rows.
6. `run_pipeline_for_sku` with a stub that returns `urgency_score=1.4` → assert the CHECK constraint fires, the run ends `status='failed'` with a reason mentioning the constraint, and **no partial `predictions` row exists** (transaction rolled back cleanly).
7. **The suppression case, using the seeded alarm-positive SKUs:** configure the stub to mimic a real alarm-positive result (`alarm_triggered=1`, `correction_factor=1.0`, Agent 5 event `status='skipped'` with the suppression note) → assert it persists successfully and the trace is queryable and shows the skip.
8. Batch: three SKUs, one deliberately failing → assert the other two still succeed and each has its own `run_id`.

---

## 9. Acceptance criteria

- [ ] `state_builder.py` performs reads only — no writes, no cleaning, no feature engineering.
- [ ] `run_repository.py` performs writes only — no decisions about pipeline outcomes beyond what it's told.
- [ ] `pipeline_service.run_pipeline_for_sku` never raises; every call returns a `run_id`.
- [ ] Success path is one atomic transaction across `predictions` + `agent_traces` + the `pipeline_runs` update.
- [ ] A CHECK-constraint violation results in a `failed` run with zero `predictions` rows — never a partial write.
- [ ] SKU-not-found, executor exception, and constraint violation are each distinguishable in `pipeline_runs.error`.
- [ ] Batch runs isolate failures per SKU.
- [ ] All tests in §8 pass using `StubExecutor` — none depend on the real orchestrator.
- [ ] The `PipelineExecutor` protocol in §2 is exactly what the orchestrator spec is written against next.

---

## 10. Explicitly out of scope

- The LangGraph graph itself — next spec, must implement `PipelineExecutor`.
- The FastAPI endpoints — later spec, will call `pipeline_service.run_pipeline_for_sku` / `_for_skus`.
- Any model or `features.py` logic — that lives entirely behind the executor.

---

## 11. Notes for the paper

The "never raises, always records" discipline is why every prediction in this system is auditable — a failure is a row in `pipeline_runs` with a reason, not a stack trace that vanished. Combined with `agent_traces`, this is the basis for the reproducibility and explainability claims: any result, success or failure, can be traced to a specific manifest version and a specific sequence of agent outcomes.
