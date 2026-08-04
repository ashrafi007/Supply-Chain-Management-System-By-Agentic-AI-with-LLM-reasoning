# Migration Spec — Integrating `order_queue` Into the Existing Codebase

**Scope:** Add the `order_queue` table and its service layer to the **existing, already-seeded** database and codebase, additively. Add manual (not automatic) deletion. Nothing already built may be modified, renamed, or regress.
**Owner:** whoever owns the DB/queue track.
**Prerequisites:** Everything already built — `database_spec.md`, `seed_data_spec.md`, `repository_layer_spec.md`, `orchestrator_spec.md` — is committed and its tests pass.
**Supersedes:** the deletion behavior in `order_queue_spec.md` §5/§10. Rows are **never deleted automatically**. Deletion is a manual, explicit action — either after the 7-day window closes with no order placed, or after the order is confirmed complete.

---

## 0. The one rule this entire spec exists to enforce

**Run your full existing test suite before touching anything, and again after every step.** If a "before" run isn't green, stop and fix that first — you must not attribute a pre-existing failure to this migration, and you must not let this migration hide inside an already-broken suite.

```bash
pytest -v
```

Record the pass count now. Every later gate in this spec compares against this number — it must never go down, only up.

---

## 1. What must NOT change (read this before writing anything)

Explicitly, none of the following are touched by this spec. If Claude Code's plan for any step here shows a diff to one of these files, stop and reconsider before applying it:

- `src/db/models.py` — the seven existing table definitions are not edited. `order_queue` is a **new** class in a **new** file, registered on the **same** `Base`.
- `src/db/base.py` — the engine, session factory, and `PRAGMA` setup are reused as-is, not duplicated.
- `src/orchestrator/*` — the graph, nodes, edges, executor are untouched. The queue's sweep function calls `run_pipeline_for_skus` as a black box.
- `src/repository/*` — `state_builder.py`, `run_repository.py`, `pipeline_service.py` are untouched.
- `data/app.db`'s existing data — `suppliers`, `skus`, `inventory_current`, and any `pipeline_runs`/`predictions`/`agent_traces` rows from testing so far are preserved. This migration must not require deleting and recreating the database file.
- Any existing test file — no existing test is edited to make this migration pass. New behavior gets new tests.

---

## 2. Design decisions (final, supersedes the earlier draft)

| Decision | Choice | Reason |
|---|---|---|
| Migration mechanism | `Base.metadata.create_all(engine)`, run again | It is additive by construction — creates only tables that don't yet exist, never touches or drops existing ones. Safe to run against a live, populated `app.db` |
| Row lifecycle | Persists across sweeps, **status changes automatically, row deletion never does** | A sweep may flip `status` (`pending` → `due_today`, or auto-flag `expired` once past `due_date`), but the row itself stays until a human removes it |
| Deletion | **Manual only**, one explicit function, two valid reasons | Matches your instruction directly — no background job silently purging rows |
| Deletion reasons | `'expired_manual'` (past due_date, no order placed) or `'fulfilled'` (order completed) | Both are real business outcomes worth distinguishing in a log, even though the row itself is removed |
| Audit of deletions | Deleted rows are logged to a small append-only table before removal | Deleting from `order_queue` should not mean losing all record that the SKU was once queued — useful for the paper's operational narrative and for debugging "where did this SKU go" |

---

## 3. Schema change: `order_queue` (unchanged from prior spec) + new `order_queue_log`

### 3.1 `order_queue` — as specified previously, reproduced here for completeness

| Column | Type | Constraints |
|---|---|---|
| `sku_id` | TEXT | PK, FK → `skus.sku_id` |
| `status` | TEXT | NOT NULL, CHECK in (`'pending'`, `'due_today'`, `'expired'`), default `'pending'` |
| `queued_at` | DATETIME | NOT NULL, default UTC now |
| `due_date` | DATE | NOT NULL |
| `last_run_id` | TEXT | NULL, FK → `pipeline_runs.run_id` |
| `last_evaluated_at` | DATETIME | NULL |
| `source` | TEXT | NOT NULL, CHECK in (`'scheduled'`, `'manual_add'`) |

**Change from the prior draft:** `'fulfilled'` is removed from `status`'s CHECK. Fulfillment now means "deleted with reason `fulfilled`," not a status value — since fulfilled rows are meant to leave the active queue via the manual deletion path, not sit in it. `status` now only describes states of an *active* row.

### 3.2 `order_queue_log` — new, append-only

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK autoincrement |
| `sku_id` | TEXT | NOT NULL — no FK to `order_queue`, since the row it refers to is gone by the time this is written |
| `queued_at` | DATETIME | NOT NULL — copied from the deleted row |
| `due_date` | DATE | NOT NULL — copied |
| `last_run_id` | TEXT | NULL — copied |
| `reason` | TEXT | NOT NULL, CHECK in (`'expired_manual'`, `'fulfilled'`) |
| `deleted_at` | DATETIME | NOT NULL, default UTC now |
| `deleted_by` | TEXT | NULL — free text, who/what triggered it; NULL acceptable until the frontend supplies a user identity |

This is the only place "this SKU was once due" survives after deletion. Cheap to add now, expensive to reconstruct later.

---

## 4. Files to create (additive only — no existing file is edited except where explicitly noted in §5)

```
src/
└── queue/
    ├── __init__.py
    ├── models.py              # order_queue + order_queue_log, imported into Base's metadata
    ├── queue_repository.py    # reads/writes both tables
    ├── sweep_service.py       # batch entrypoint, as before
    ├── ingestion_service.py   # new-SKU flow, as before
    └── deletion_service.py    # NEW — the manual deletion function

tests/
└── queue/
    ├── __init__.py
    ├── test_queue_repository.py
    ├── test_sweep_service.py
    ├── test_ingestion_service.py
    └── test_deletion_service.py   # NEW

scripts/
└── run_migration.py          # NEW — the one-time additive migration runner
```

---

## 5. Step-by-step migration (in order — do not skip the gates)

### Step 0 — baseline gate
```bash
pytest -v
```
Confirm green. Record pass count.

### Step 1 — `src/queue/models.py`
Define `OrderQueue` and `OrderQueueLog` as SQLAlchemy models against the **existing** `Base` imported from `src/db/base.py`. Do not create a second `Base` or a second engine — that would create a second, disconnected database file.

```python
from src.db.base import Base   # the existing one — not a new declarative_base()
```

### Step 2 — `scripts/run_migration.py`
```python
from src.db.base import engine, Base
import src.db.models          # existing tables register on Base
import src.queue.models        # new tables register on Base

def run():
    before = set(Base.metadata.tables.keys())
    Base.metadata.create_all(engine)   # additive: creates only what's missing
    print(f"Tables now present: {sorted(before)}")
    print("Migration complete — no existing table was dropped or altered.")
```

Run it:
```bash
python -m scripts.run_migration
```

**Verify additivity directly**, don't just trust the description:
```bash
sqlite3 data/app.db "SELECT sku_id FROM inventory_current LIMIT 3;"
```
Your previously seeded rows must still be there, unchanged. If this returns nothing or errors, stop — do not proceed to Step 3.

### Step 3 — regression gate
```bash
pytest -v
```
Pass count must be **≥** the Step 0 baseline, and every test that passed before must still pass. New failures here mean the migration touched something it shouldn't have — go back to §1.

### Step 4 — `src/queue/queue_repository.py`
As in the prior spec, plus new functions for the log:

```python
def get_due_skus(session, as_of: date) -> list[str]: ...
def mark_expired(session, as_of: date) -> int: ...          # status only, no deletion
def mark_evaluated(session, sku_id, run_id, evaluated_at) -> None: ...
def enqueue(session, sku_id, due_date, source) -> None: ...
def log_and_delete(session, sku_id: str, reason: Literal["expired_manual", "fulfilled"], deleted_by: str | None) -> None:
    """Reads the order_queue row, writes it to order_queue_log, then deletes it. One transaction."""
```

### Step 5 — `src/queue/deletion_service.py` — the new manual deletion entrypoint

```python
def remove_from_queue(session: Session, sku_id: str, reason: Literal["expired_manual", "fulfilled"], deleted_by: str | None = None) -> None:
    """The only way a row leaves order_queue. Never called automatically by a sweep or a scheduler."""
```

- Raises if `sku_id` is not currently in `order_queue` — nothing to delete.
- Calls `queue_repository.log_and_delete(...)`.
- **This function is explicitly the one a future "Mark Complete" or "Remove Expired" button in the frontend will call.** No other code path deletes from `order_queue`. `sweep_service.py`'s expiry handling continues to only set `status='expired'` — it never deletes; a human (or a later scheduled cleanup, if you choose to add one explicitly and separately) decides when an expired row is actually removed.

### Step 6 — regression gate again
```bash
pytest -v
```
Still must be ≥ baseline.

### Step 7 — new tests, `tests/queue/`
Write and run the tests in §6 below.

### Step 8 — final full-suite gate
```bash
pytest -v
```
This is the number you report. Every test from every prior spec plus every new queue test, all green, in one run.

---

## 6. Tests

### `test_queue_repository.py` (as before, plus:)
6. `log_and_delete` removes the row from `order_queue` and creates exactly one matching row in `order_queue_log` with the correct `reason`.
7. `log_and_delete` on a nonexistent `sku_id` raises and writes nothing.

### `test_sweep_service.py` (as before)
No changes — sweeps still never delete.

### `test_deletion_service.py` (new)
1. `remove_from_queue(..., reason='fulfilled')` on an active row succeeds; row gone from `order_queue`, present in `order_queue_log` with `reason='fulfilled'`.
2. `remove_from_queue(..., reason='expired_manual')` on a row past `due_date` succeeds identically with the other reason.
3. Calling it on a `sku_id` not currently queued raises, and `order_queue_log` gains no row.
4. Deleting a row does not touch `skus`, `inventory_current`, or any `pipeline_runs`/`predictions`/`agent_traces` history for that SKU — the SKU's identity and prediction history are untouched; only its queue membership is gone.
5. After deletion, the `sku_id` **can be re-enqueued** via `enqueue(...)` — deletion is not permanent exclusion, just queue removal.

---

## 7. Acceptance criteria

- [ ] Step 0 baseline recorded, green.
- [ ] `run_migration.py` runs against the live `data/app.db` with zero data loss in any existing table (verified directly via `sqlite3`, not assumed).
- [ ] `order_queue` and `order_queue_log` both exist after migration.
- [ ] `status` CHECK on `order_queue` no longer includes `'fulfilled'`.
- [ ] No deletion of an `order_queue` row happens anywhere except through `remove_from_queue`.
- [ ] `sweep_service.py` still never deletes — only marks `status`.
- [ ] Every deletion produces exactly one `order_queue_log` row first.
- [ ] Final full-suite pass count ≥ the Step 0 baseline, with every previously-passing test still passing.
- [ ] No file listed in §1 shows an unexpected diff.

---

## 8. Explicitly out of scope

- Any scheduled/automatic cleanup job for expired rows — you asked for manual only; if you want an optional scheduled purge later, that is a distinct, separately-decided spec, not a default behavior here.
- Frontend buttons that call `remove_from_queue` — later spec.
- Changing what `mark_fulfilled`-equivalent business logic decides "the order is complete" — `remove_from_queue(reason='fulfilled')` is the mechanism; the decision of *when* to call it with that reason belongs to whatever business logic determines an order is done, specified later.

---

## 9. Notes for the paper

The append-only `order_queue_log` is a small addition worth naming: it means the system's operational history — which SKUs were queued, when, and how they exited the queue — is fully reconstructable even though the active queue itself is a mutable, human-curated view. This is the same audit discipline as `agent_traces` and `pipeline_runs`, applied one layer up, to the scheduling decision rather than the prediction decision.
