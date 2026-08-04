# Orchestrator Spec — LangGraph Agent Linking

**Scope:** Wire the six verified agent wrappers into one LangGraph `StateGraph` that satisfies the `PipelineExecutor` protocol from `repository_layer_spec.md`. No new model logic. No new persistence logic.
**Owner:** Person A (Agents + Orchestration track)
**Prerequisites (all complete):** Database seeded. All six agents wrapped and parity-verified. Repository layer built and passing tests against `StubExecutor`.
**Deliverable:** A real `PipelineExecutor` implementation that drops into the repository layer with zero changes elsewhere.

---

## 1. What this module must satisfy (recap of the contract it's built against)

```python
class PipelineExecutor(Protocol):
    def invoke(self, state: PipelineState) -> PipelineResult: ...
```

```python
class PipelineResult(TypedDict):
    final_state: PipelineState
    agent_events: list[AgentEvent]
    manifest_version: str

class AgentEvent(TypedDict):
    agent_name: str
    sequence: int
    status: Literal["success", "skipped", "failed"]
    latency_ms: int | None
    output: dict
    note: str | None
```

The repository layer already tests against this shape via `StubExecutor`. This spec's only job is to make the **real** graph produce it. If the shape here drifts from the repository spec, nothing downstream needs to change — only this file needs to match.

---

## 2. Folder structure

Create exactly this layout. Every file below gets a one-line docstring at the top of its own file stating its single responsibility — no file should need a comment to explain what it's for.

```
src/
└── orchestrator/
    ├── __init__.py                 # exports: build_executor(), LangGraphExecutor
    ├── state.py                    # PipelineState TypedDict — the frozen contract, single source of truth
    ├── manifest.py                 # loads models/manifest.json, resolves artifact paths, returns manifest_version
    ├── nodes/
    │   ├── __init__.py
    │   ├── agent_1_demand.py       # thin LangGraph node wrapper around DemandPredictorAgent.run
    │   ├── agent_2_risk.py         # thin LangGraph node wrapper around RiskDetectorAgent.run
    │   ├── agent_3_rebalancer.py   # thin LangGraph node wrapper around RebalancerAgent.run
    │   ├── agent_5_forecast_opt.py # thin LangGraph node wrapper + suppression check
    │   ├── agent_4_routing.py      # thin LangGraph node wrapper around RoutingAgent.run
    │   └── agent_6_auditor.py      # thin LangGraph node wrapper around AuditorAgent.run
    ├── edges.py                    # conditional routing logic (the suppression branch), pure function, no I/O
    ├── graph.py                    # StateGraph assembly: add_node, add_edge, add_conditional_edges, compile()
    ├── executor.py                 # LangGraphExecutor — implements PipelineExecutor, wraps graph.py's compiled app
    └── tracing.py                  # converts LangGraph's internal run trace into list[AgentEvent] — timing, status, notes

tests/
└── orchestrator/
    ├── __init__.py
    ├── test_graph_topology.py      # asserts node order, edge structure, conditional edge exists
    ├── test_executor_contract.py   # asserts LangGraphExecutor.invoke() matches PipelineExecutor shape
    ├── test_suppression.py         # THE test — alarm_triggered=1 forces correction_factor=1.0, agent_5 skipped
    └── test_end_to_end.py          # repository_layer + real executor + seeded DB, full stack

scripts/
└── run_single_prediction.py        # CLI demo: takes a --sku-id, runs it, pretty-prints the three DB tables
```

Rationale for the split: `nodes/` holds only thin adapters (state in, state delta out) — the actual ML logic stays in the agent wrapper classes you already built and verified, untouched. `edges.py` is a pure function with no I/O so it's trivially unit-testable in isolation from LangGraph itself. `tracing.py` exists because LangGraph's native execution trace and your `AgentEvent` shape are not the same thing — this file is the only place that translation happens.

---

## 3. `state.py`

Single source of truth for `PipelineState`. Import this everywhere else — never redefine it.

```python
class PipelineState(TypedDict):
    sku_id: str
    raw_features: dict
    demand_forecast: Optional[float]
    backorder_prob: Optional[float]
    alarm_triggered: Optional[int]
    urgency_score: Optional[float]
    correction_factor: Optional[float]
    replenishment_qty: Optional[float]
    route: Optional[str]
    supplier_risk: Optional[str]
    recommendation: Optional[str]
    trace: list
    errors: list
```

This must be byte-for-byte the same shape the repository layer's `state_builder.py` produces and the `predictions` table expects. If it isn't already shared as one importable module between the two, fix that now — two independently maintained copies of this TypedDict is how the two layers drift apart silently.

---

## 4. `manifest.py`

- Load `models/manifest.json` (or your chosen format) once.
- Expose `MANIFEST_VERSION: str` — a hash or version string identifying this exact set of artifact files. Used to populate `pipeline_runs.manifest_version` and `PipelineResult["manifest_version"]`.
- Expose per-agent artifact paths for the six wrapper classes to load from at construction.
- **Load nothing lazily.** Every path here should be resolved and validated to exist at import time — the same discipline as Layer 1 of `verification_spec.md`. If an artifact is missing, fail at startup, not mid-request.

---

## 5. Nodes (`nodes/*.py`)

Each node file is a thin function, not a class — the class-based agent wrappers already exist and are already verified. A node's only job is: construct the wrapper once at module load (or receive it via dependency injection — pick one and be consistent across all six), call `.run(state)`, and return the delta.

```python
# nodes/agent_2_risk.py
"""LangGraph node adapter for the Risk Detector agent. No model logic lives here."""

_agent = RiskDetectorAgent.from_manifest(manifest.RISK_DETECTOR_PATHS)

def node(state: PipelineState) -> dict:
    return _agent.run(state)
```

Each node's `run()` already returns a state delta (built when you wrote the wrappers) — this file adds nothing except registering that function as a graph node. If any node needs timing captured for `agent_events`, wrap the call, don't modify the agent class:

```python
def node(state: PipelineState) -> dict:
    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
```

(`tracing.py` strips the `_latency_ms` convention key back out when building `AgentEvent` — don't let it leak into `PipelineState` itself; keep it out of the `predictions` mapping.)

---

## 6. `edges.py` — the suppression logic, as a pure function

```python
def should_suppress(state: PipelineState) -> bool:
    """Returns True if Agent 5's corrector must be bypassed."""
    return state.get("alarm_triggered") == 1
```

```python
def route_after_risk(state: PipelineState) -> Literal["rebalancer"]:
    return "rebalancer"   # A3 always follows A2; branching happens after A3, at A5

def route_before_forecast_opt(state: PipelineState) -> Literal["suppress", "correct"]:
    return "suppress" if should_suppress(state) else "correct"
```

This is the single most important file in the module for your paper's novelty claim, and it must be **testable with zero LangGraph machinery** — `test_suppression.py`'s core assertions should be callable against `should_suppress()` directly, independent of whether the graph runs correctly.

The "suppress" branch does not skip Agent 5 silently — it routes to a small suppression node that sets `correction_factor = 1.0` explicitly and records the skip:

```python
# nodes/agent_5_forecast_opt.py
def node(state: PipelineState) -> dict:
    if edges.should_suppress(state):
        return {
            "correction_factor": 1.0,
            "_skipped": True,
            "_note": f"suppressed: alarm_triggered={state['alarm_triggered']}",
        }
    start = time.monotonic()
    delta = _agent.run(state)
    delta["_latency_ms"] = int((time.monotonic() - start) * 1000)
    return delta
```

---

## 7. `graph.py` — assembly

```python
def build_graph() -> CompiledGraph:
    g = StateGraph(PipelineState)
    g.add_node("demand", agent_1_demand.node)
    g.add_node("risk", agent_2_risk.node)
    g.add_node("rebalancer", agent_3_rebalancer.node)
    g.add_node("forecast_opt", agent_5_forecast_opt.node)
    g.add_node("routing", agent_4_routing.node)
    g.add_node("auditor", agent_6_auditor.node)

    g.add_edge(START, "demand")
    g.add_edge("demand", "risk")
    g.add_edge("risk", "rebalancer")
    g.add_edge("rebalancer", "forecast_opt")   # branching happens INSIDE this node, per §6
    g.add_edge("forecast_opt", "routing")
    g.add_edge("routing", "auditor")
    g.add_edge("auditor", END)

    return g.compile()
```

Note: the suppression branch in this build lives inside the `forecast_opt` node's own logic (§6), not as a LangGraph `add_conditional_edges` fork to two different downstream nodes — because both branches converge to the same next node (`routing`) with only the *output*, not the *path*, differing. If you want the branch visible in the compiled graph topology itself (stronger for the architecture diagram in the paper), use `add_conditional_edges("rebalancer", edges.route_before_forecast_opt, {"suppress": "suppress_node", "correct": "forecast_opt"})` with two small nodes that both set state and both proceed to `"routing"`. Either is correct; **pick one and state the choice in the paper's methods section**, since reviewers may ask to see the branch in the diagram.

---

## 8. `tracing.py`

Converts LangGraph's execution into `list[AgentEvent]`. Since nodes return `_latency_ms`, `_skipped`, `_note` as convention keys in their delta dicts, this module:

1. Strips those convention keys out of the delta before it's used as `output` in `AgentEvent` (they don't belong in `PipelineState`/`predictions`).
2. Assigns `sequence` based on actual invocation order (LangGraph's stream/trace gives you this).
3. Sets `status="skipped"` when `_skipped` is present, `"success"` otherwise, `"failed"` if the node raised (caught at the executor level — see §9).

---

## 9. `executor.py` — the real `PipelineExecutor`

```python
class LangGraphExecutor:
    def __init__(self):
        self._graph = graph.build_graph()

    def invoke(self, state: PipelineState) -> PipelineResult:
        events = []
        try:
            final_state = self._graph.invoke(state)
        except Exception as e:
            # a node raised — still return a PipelineResult, don't propagate;
            # repository_layer's persist_failure expects to catch this from executor.invoke itself,
            # so re-raise here IS correct per the repository spec's contract (§6 step 4/6 there).
            raise
        events = tracing.build_agent_events(self._graph, final_state)
        return {
            "final_state": final_state,
            "agent_events": events,
            "manifest_version": manifest.MANIFEST_VERSION,
        }
```

Confirm against `repository_layer_spec.md` §6: the repository layer expects `executor.invoke()` to either return a valid `PipelineResult` or raise — it does its own catching. Do not swallow exceptions here.

---

## 10. Folder tree, decorated (for reference / commit message)

```
src/orchestrator/
├── __init__.py                  # public exports only
├── state.py                     # ★ the frozen PipelineState contract
├── manifest.py                  # model artifact registry + manifest_version
├── nodes/
│   ├── agent_1_demand.py        # A1 — entry point, no upstream dependency
│   ├── agent_2_risk.py          # A2 — writes backorder_prob, alarm_triggered
│   ├── agent_3_rebalancer.py    # A3 — REQUIRES A2's output (35% of urgency formula)
│   ├── agent_5_forecast_opt.py  # A5 — ★ suppression branch lives here
│   ├── agent_4_routing.py       # A4
│   └── agent_6_auditor.py       # A6
├── edges.py                     # ★ should_suppress() — the novelty claim, as pure code
├── graph.py                     # StateGraph wiring, compile()
├── executor.py                  # ★ PipelineExecutor implementation — the deliverable
└── tracing.py                   # LangGraph trace -> AgentEvent
```

---

## 11. Commands

Run these from the project root, in this order.

### 11.1 Environment
```bash
pip install -r requirements.txt --break-system-packages
```

### 11.2 Validate the manifest loads before touching the graph
```bash
python -m src.orchestrator.manifest
```
Expect: prints `MANIFEST_VERSION` and every resolved artifact path. Any missing file fails here, loudly, before anything else runs.

### 11.3 Unit tests — pure logic, no LangGraph, no DB
```bash
pytest tests/orchestrator/test_graph_topology.py -v
```
Expect: asserts the compiled graph has exactly six nodes plus START/END, edges match §7, and `should_suppress()` is unit-testable in isolation.

### 11.4 Suppression test — the core novelty proof, in isolation
```bash
pytest tests/orchestrator/test_suppression.py -v
```
Expect: feeding a state with `alarm_triggered=1` through the compiled graph yields `correction_factor == 1.0` and an `AgentEvent` for `agent_5_forecast_opt` with `status="skipped"` and a note containing `"suppressed"`.

### 11.5 Executor contract test
```bash
pytest tests/orchestrator/test_executor_contract.py -v
```
Expect: `LangGraphExecutor().invoke(sample_state)` returns a dict with exactly the keys `final_state`, `agent_events`, `manifest_version`, and `agent_events` is a non-empty list of dicts each containing `agent_name`, `sequence`, `status`, `latency_ms`, `output`, `note`.

### 11.6 Full orchestrator suite
```bash
pytest tests/orchestrator/ -v
```

### 11.7 Swap the stub for the real executor in the repository tests
```bash
pytest tests/test_repository.py -v --executor=real
```
(Or, if `--executor` isn't wired as a CLI flag, edit the fixture in `tests/test_repository.py` to import `LangGraphExecutor` instead of `StubExecutor` and re-run plainly with `pytest tests/test_repository.py -v`.) Expect: the same 8 tests that passed against the stub now pass against the real graph — this is Layer 3 of `verification_spec.md`, now unblocked.

### 11.8 End-to-end, single seeded SKU
```bash
python -m scripts.run_single_prediction --sku-id SKU00123
```
Expect: prints the `run_id`, then the row from `pipeline_runs`, the row from `predictions`, and all six rows from `agent_traces` in `sequence` order.

### 11.9 End-to-end, a known alarm-positive seeded SKU — run this one for your supervisor
```bash
python -m scripts.run_single_prediction --sku-id <an alarm-positive SKU from your seed composition log>
```
Expect: `predictions.alarm_triggered = 1`, `predictions.correction_factor = 1.0`, and in the printed `agent_traces`, the `agent_5_forecast_opt` row shows `status = skipped` with the suppression note. **This single command's output is your end-to-end proof of the system's core contribution.**

### 11.10 Full test suite, whole repo
```bash
pytest -v
```
Expect: every test from every earlier spec (artifacts, wrapper parity, repository, orchestrator) green in one run. This is the command to run right before showing your supervisor anything.

---

## 12. Acceptance criteria

- [ ] Folder structure matches §2 exactly; every file has a one-line docstring stating its single responsibility.
- [ ] `PipelineState` in `orchestrator/state.py` is imported by the repository layer too — not redefined.
- [ ] `manifest.py` fails at import time if any artifact path is missing (11.2 passes).
- [ ] Node files contain no model logic — they call the already-verified wrapper classes only.
- [ ] `should_suppress()` in `edges.py` is a pure function, unit-testable with no LangGraph or DB dependency.
- [ ] Graph topology test (11.3) passes: six nodes, correct edge order, A3 strictly after A2.
- [ ] Suppression test (11.4) passes in isolation.
- [ ] Executor contract test (11.5) passes — `LangGraphExecutor` produces exactly the `PipelineResult` shape.
- [ ] Repository tests (11.7) pass against the real executor with zero code changes to the repository layer itself.
- [ ] Command 11.9, run against a real seeded alarm-positive SKU, shows the suppression in the printed `agent_traces`.
- [ ] `pytest -v` (11.10) is fully green.

---

## 13. Explicitly out of scope

- FastAPI endpoints — next spec, will call `pipeline_service.run_pipeline_for_sku` with `LangGraphExecutor()` injected.
- The optional LLM narration node — can be added as a seventh node after `auditor` later; not required for this spec to be complete.
- Frontend — separate track, should already be underway against mocked API responses.

---

## 14. Notes for the paper

Section 7's note about the two valid ways to express the suppression branch (in-node logic vs. `add_conditional_edges` with two convergent nodes) is worth resolving deliberately rather than by default — whichever is chosen should match what the architecture diagram shows. The suppression test passing in isolation, independent of the full graph, is what lets the methods section claim the invariant is verified at the unit level, not just observed anecdotally in a demo run.
