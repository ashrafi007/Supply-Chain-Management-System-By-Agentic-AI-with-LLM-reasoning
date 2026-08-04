# Database Spec — Agentic Supply Chain Management System

**Scope:** Create the SQLite database and schema. **No data loading.** Seeding is a separate spec.
**Owner:** Person B (Backend + Database track)
**Target:** End of Day 2 — must be complete before Checkpoint 1 (contracts frozen).
**Prerequisite:** The `PipelineState` schema must be frozen before this spec is executed.

---

## 1. Design decisions (do not deviate without discussion)

| Decision | Choice | Reason |
|---|---|---|
| Engine | **SQLite** | Embedded, zero-config, ships inside the desktop app as one file. Workload is transactional (point lookups by SKU, single-row inserts per run) — a row store, not a column store |
| Not DuckDB | DuckDB is analytical | DuckDB is used **only** in notebooks to query the training parquet. It never ships and is not part of this spec |
| Access layer | SQLAlchemy 2.0 declarative ORM | Lets us swap to Postgres later without rewriting queries |
| Migrations | None — `create_all()` | 20-day timeline, small schema. Add Alembic only if churn becomes painful |
| DB file | `data/app.db` | Gitignored. Regenerable from this spec at any time |
| Training data | **Never enters this database** | The 1.9M-row RSM dataset stays as parquet/CSV in `data/raw/`. It is training data; the runtime never reads it |
| Source of truth for inputs | **The raw CSV**, not `processed_dataset.pkl` | The DB stores raw inputs; features are derived at request time by `features.py`. Storing engineered features guarantees staleness and train/serve skew |
| Row scale | `inventory_current` holds **one row per SKU** — current state only | Not one row per historical transaction |
| Timestamps | UTC, `DateTime` type | Never store local time |

---

## 2. Step 0 — Pre-flight diagnostic (BLOCKING)

**Run this before writing any model class. Its result determines the primary key of `inventory_current`.**

```python
import pandas as pd
raw = pd.read_csv("data/raw/rsm.csv")

print("rows:", len(raw))
print("unique skus:", raw['sku'].nunique())
print("fully identical rows:", raw.duplicated().sum())
print("rows with repeated sku:", raw.duplicated(subset=['sku']).sum())

# If sku repeats, inspect what varies across the repeats:
if raw['sku'].duplicated().any():
    d = raw['sku'].value_counts().index[0]
    print(raw[raw['sku'] == d].head(10).to_string())
```

Classify into exactly one case and record the answer in the PR description:

### Case A — `nunique(sku) == len(raw)`, or repeats are fully identical rows
Every row is a distinct SKU (or duplicates are a data-quality artifact).
→ **`inventory_current` PK = `sku_id`.** Proceed with this spec as written.
→ Duplicates are dropped in the training-only bucket (see §5), never at serve time.

### Case B — repeats differ by a date/period column
The dataset is a time series of snapshots.
→ **`inventory_current` PK = `sku_id`**, seeded with the **latest row per SKU**. Older rows remain in the parquet as training history.
→ **Sub-check, blocking:** does any feature require *other rows* to compute (lags, rolling windows, "sales over last N months" computed across rows)? Inspect the preprocessing notebook for `.shift()`, `.rolling()`, `.groupby().transform()`.
  - **No** (all aggregates are already columns on the row) → latest-row-per-SKU is sufficient. Proceed as written.
  - **Yes** → this spec is insufficient. `inventory_current` becomes `inventory_snapshots` with composite PK `(sku_id, snapshot_at)` holding a bounded history window. **Stop and amend the spec before continuing.**

### Case C — repeats differ by location / plant / warehouse
The entity is not the SKU; it is `(sku, location)`.
→ **Stop.** PK becomes composite `(sku_id, location_id)`, and `sku_id` alone stops being a valid identifier across the API, the `PipelineState`, and every table below. **Amend the spec before continuing.**

Default assumption for the rest of this document: **Case A or Case B without cross-row features.**

---

## 3. Tables

Seven tables. Three are populated by seed data (later spec), four are written by the running pipeline.

| Table | Written by | Purpose |
|---|---|---|
| `suppliers` | Seed | Supplier identity — Agent 6 subject |
| `skus` | Seed | SKU identity, links to supplier |
| `inventory_current` | Seed / updates | Current-state **raw** columns the agents consume |
| `pipeline_runs` | Pipeline | One row per invocation — metadata, status, latency |
| `predictions` | Pipeline | One row per run — all six agents' outputs, 1:1 with `PipelineState` |
| `agent_traces` | Pipeline | One row per agent per run — explainability + audit trail |
| `forecast_actuals` | Later | Realized demand, for Agent 5 MAPE-vs-human-baseline tracking |

### 3.1 `suppliers`

| Column | Type | Constraints |
|---|---|---|
| `supplier_id` | TEXT | PK |
| `name` | TEXT | NOT NULL |
| `country` | TEXT | NULL |
| `lead_time_avg_days` | REAL | NULL |
| `created_at` | DATETIME | NOT NULL, default UTC now |

### 3.2 `skus`

| Column | Type | Constraints |
|---|---|---|
| `sku_id` | TEXT | PK |
| `supplier_id` | TEXT | FK → `suppliers.supplier_id`, NULL allowed |
| `description` | TEXT | NULL |
| `created_at` | DATETIME | NOT NULL, default UTC now |

FK is nullable: a SKU with an unknown supplier must still be predictable.
Index: `idx_skus_supplier` on `supplier_id`.

### 3.3 `inventory_current`

One row per SKU. **The only table the pipeline reads at prediction time.**

Fixed columns:

| Column | Type | Constraints |
|---|---|---|
| `sku_id` | TEXT | PK, FK → `skus.sku_id` |
| `snapshot_at` | DATETIME | NOT NULL, default UTC now |
| `raw_extra` | JSON | NULL — escape hatch for columns not yet typed |
| `updated_at` | DATETIME | NOT NULL, default UTC now |

`sku_id` as both PK and FK enforces one-row-per-SKU at the storage layer. This is the constraint that structurally prevents the 1.9M historical rows from creeping in.

**REQUIRED — derive the feature columns. Do not invent them, and do not copy them from `processed_dataset.pkl`.**

1. Read the header of `data/raw/rsm.csv`. These are the candidate columns.
2. Load every saved feature-column artifact (`models/feature_columns.pkl` for Agent 2, the Agent 5 equivalent, and the artifacts for Agents 1, 3, 4, 6). Take their union — this is the set of features the models actually consume.
3. For each feature in that union, trace it back to the preprocessing notebook:
   - **Raw** (read directly from the CSV) → **emit a typed column here**.
   - **Engineered** (computed from other columns — e.g. `coverage_ratio`, `transit_coverage`, depletion rates) → **do not store**. Store its *inputs* instead. Engineered features are derived at runtime by `features.py`.
4. **Exclude the target column** (`went_on_backorder` or equivalent) unconditionally. A target column in the serving table is a leakage accident waiting to happen. If per-row ground truth is needed for evaluation, it belongs in a separate table or stays in the parquet.
5. Emit each retained raw column with an appropriate type: `REAL` numeric, `INTEGER` binary flags, `TEXT` categorical.
6. **Print the final column list to stdout** during creation, annotated `raw` / `excluded-engineered` / `excluded-target`, so it can be reviewed against the notebooks.

Rationale: if this table's columns drift from what the models were trained on, every prediction is silently wrong. The schema is generated from the artifacts and the CSV, never hand-typed.

Index: `idx_inventory_snapshot` on `snapshot_at`.

### 3.4 `pipeline_runs`

| Column | Type | Constraints |
|---|---|---|
| `run_id` | TEXT | PK — UUID4 string |
| `sku_id` | TEXT | NOT NULL, FK → `skus.sku_id` |
| `started_at` | DATETIME | NOT NULL |
| `completed_at` | DATETIME | NULL |
| `status` | TEXT | NOT NULL, CHECK in (`'pending'`, `'success'`, `'failed'`) |
| `latency_ms` | INTEGER | NULL |
| `manifest_version` | TEXT | NOT NULL — which model manifest produced this run |
| `error` | TEXT | NULL |

Indexes: `idx_runs_sku` on `sku_id`; `idx_runs_started` on `started_at`.

`manifest_version` is what makes runs reproducible — it pins which model artifacts were in memory. Do not omit it.

A row that cannot be cleaned at serve time gets `status='failed'` and a reason in `error`. It is **never** silently dropped — see §5.

### 3.5 `predictions`

One row per run. Mirrors the frozen `PipelineState` field for field.

| Column | Type | Constraints |
|---|---|---|
| `run_id` | TEXT | PK, FK → `pipeline_runs.run_id` |
| `sku_id` | TEXT | NOT NULL, FK → `skus.sku_id` |
| `demand_forecast` | REAL | NULL — Agent 1 |
| `backorder_prob` | REAL | NULL — Agent 2, CHECK between 0 and 1 |
| `alarm_triggered` | INTEGER | NULL — Agent 2, CHECK in (0, 1) |
| `urgency_score` | REAL | NULL — Agent 3, CHECK between 0 and 1 |
| `correction_factor` | REAL | NULL — Agent 5, CHECK > 0 |
| `replenishment_qty` | REAL | NULL — Agent 4 |
| `route` | TEXT | NULL — Agent 4 |
| `supplier_risk` | TEXT | NULL — Agent 6 |
| `recommendation` | TEXT | NULL — optional LLM node output |
| `created_at` | DATETIME | NOT NULL, default UTC now |

**Table-level CHECK constraints — both are load-bearing:**

```
CHECK (urgency_score IS NULL OR (urgency_score >= 0 AND urgency_score <= 1))
```
Enforces the Agent 3 clipping fix at the storage layer. If the clip is ever removed upstream, the insert fails loudly instead of corrupting results silently.

```
CHECK (alarm_triggered IS NULL OR alarm_triggered = 0 OR correction_factor = 1.0)
```
Enforces the **risk-suppression invariant**: when Agent 2 raises an alarm, Agent 5 must not reduce the forecast. This is the system's core novelty claim, guaranteed by the database. A rejected insert means the suppression logic broke.

`sku_id` here is a deliberate denormalization — it is reachable via `predictions → pipeline_runs → skus`, but the dashboard's hottest query is "top N SKUs by urgency" and carrying it directly saves a join on every load.

Indexes: `idx_pred_sku` on `sku_id`; `idx_pred_urgency` on `urgency_score` (descending); `idx_pred_alarm` on `alarm_triggered`.

### 3.6 `agent_traces`

| Column | Type | Constraints |
|---|---|---|
| `trace_id` | INTEGER | PK autoincrement |
| `run_id` | TEXT | NOT NULL, FK → `pipeline_runs.run_id` |
| `agent_name` | TEXT | NOT NULL — e.g. `'agent_2_risk_detector'` |
| `sequence` | INTEGER | NOT NULL — execution order within the run |
| `status` | TEXT | NOT NULL, CHECK in (`'success'`, `'skipped'`, `'failed'`) |
| `latency_ms` | INTEGER | NULL |
| `output` | JSON | NULL — the state delta this agent returned |
| `note` | TEXT | NULL — e.g. `'suppressed: alarm_triggered=1'` |
| `created_at` | DATETIME | NOT NULL, default UTC now |

Index: `idx_traces_run` on `run_id`.

This table is the explainability and reproducibility evidence for the paper. The `'skipped'` status plus a suppression note is how the conditional-edge behaviour becomes auditable.

### 3.7 `forecast_actuals`

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PK autoincrement |
| `sku_id` | TEXT | NOT NULL, FK → `skus.sku_id` |
| `period` | TEXT | NOT NULL — e.g. `'2026-08'` |
| `human_forecast` | REAL | NULL — baseline |
| `agent_forecast` | REAL | NULL |
| `actual_demand` | REAL | NULL |
| `recorded_at` | DATETIME | NOT NULL, default UTC now |

UNIQUE on (`sku_id`, `period`). Index: `idx_actuals_sku_period` on (`sku_id`, `period`).

The only table that makes the Agent 5 headline metric — MAPE improvement over human baseline — computable. Create it now even though it stays empty for a while.

---

## 4. Relationships

```
suppliers  1 ──< skus                     (supplier_id, nullable)
skus       1 ──1 inventory_current        (sku_id is PK and FK — one row per SKU)
skus       1 ──< pipeline_runs            (sku_id)
skus       1 ──< forecast_actuals         (sku_id, unique per period)
pipeline_runs 1 ──1 predictions           (run_id is PK and FK)
pipeline_runs 1 ──< agent_traces          (run_id)
```

`sku_id` is the spine of the identity side; `run_id` is the spine of the execution side; `pipeline_runs` is the bridge.

---

## 5. Cleaning contract (informs the schema — implement in `features.py`, not here)

Preprocessing operations split into two buckets. Sorting them correctly is what makes the schema's NOT NULL constraints meaningful.

**Value-level fixes — move into `features.py`, run at serve time:**
`inf`/`-inf` → NaN → median, clipping, type coercion, encoding, null filling.
These operate on a single row in isolation. Imputation values must be the **saved, train-fitted** statistics loaded from disk — never recomputed from whatever rows are present.

**Row-level filters — stay in the training notebook, never at serve time:**
`drop_duplicates()`, `dropna()`, outlier removal.
At serve time a planner asks about one specific SKU; the system cannot respond by deleting it. An unsalvageable row produces `pipeline_runs.status='failed'` with a reason.

The seeder (later spec) reads the dirty CSV, applies the value-level bucket, and writes **cleaned raw columns** into `inventory_current`. The agents apply the same function at request time — a no-op on seeded rows, real work on live data. One cleaning implementation, used in both places.

---

## 6. Files to create

```
src/db/
    __init__.py
    base.py          # engine, session factory, PRAGMA setup
    models.py        # SQLAlchemy declarative models
    create_db.py     # entrypoint — creates data/app.db
    verify_db.py     # verification script
data/
    app.db           # generated, gitignored
```

---

## 7. Step-by-step build

### Step 1 — Dependencies
Add to `requirements.txt`: `sqlalchemy>=2.0`. Nothing else; `sqlite3` is stdlib.

### Step 2 — `src/db/base.py`
- Create the engine against `data/app.db`, resolved from the **project root**, not the CWD.
- Create the directory if missing.
- **Register a `connect` event listener that runs `PRAGMA foreign_keys=ON` on every connection.** SQLite does not enforce foreign keys by default and the setting is per-connection. Omitting this silently disables every FK in this spec.
- Also set `PRAGMA journal_mode=WAL` — better concurrent-read behaviour for a desktop app.
- Expose `engine`, `SessionLocal`, `Base`.

### Step 3 — `src/db/models.py`
- Define all seven models using SQLAlchemy 2.0 `Mapped` / `mapped_column` style.
- Apply every CHECK constraint from §3 via `__table_args__` with `CheckConstraint`.
- Apply every index from §3 via `__table_args__` with `Index`.
- Use `default=lambda: datetime.now(timezone.utc)`. Do not use `datetime.utcnow` (deprecated).
- For `inventory_current`, perform the derivation in §3.3 before finalizing the class.

### Step 4 — `src/db/create_db.py`
- Import all models so they register on `Base.metadata`.
- Call `Base.metadata.create_all(engine)`.
- Print each created table with its column count.
- Print the derived `inventory_current` column list with `raw` / `excluded-engineered` / `excluded-target` annotations.
- Accept `--force` to drop and recreate. Without it, refuse to run if `data/app.db` exists.

### Step 5 — `src/db/verify_db.py`
Standalone script asserting the schema is correct. It must:
1. Confirm all seven tables exist.
2. Confirm `PRAGMA foreign_keys` returns 1.
3. Insert supplier → SKU → inventory row, then roll back.
4. **Insert a `predictions` row with `urgency_score = 1.4` and assert it is rejected.**
5. **Insert a `predictions` row with `alarm_triggered = 1` and `correction_factor = 0.8` and assert it is rejected.**
6. Insert a SKU with a non-existent `supplier_id` and assert it is rejected.
7. Insert two `inventory_current` rows with the same `sku_id` and assert the second is rejected.
8. Assert no column in `inventory_current` matches the target column name.
9. Print `SCHEMA OK` and exit 0, or print the failure and exit 1.

Steps 4, 5, 7 and 8 are the point of this script — they prove the invariants are enforced rather than merely documented.

### Step 6 — Gitignore
Add `data/app.db`, `data/app.db-wal`, `data/app.db-shm`. The database is generated, never committed.

### Step 7 — Run
```
python -m src.db.create_db
python -m src.db.verify_db
```

---

## 8. Acceptance criteria

- [ ] Step 0 diagnostic was run and the case (A / B / C) is recorded.
- [ ] `data/app.db` exists and contains exactly the seven tables in §3.
- [ ] `python -m src.db.verify_db` prints `SCHEMA OK` and exits 0.
- [ ] `PRAGMA foreign_keys` returns 1 on a fresh connection.
- [ ] An out-of-range `urgency_score` is rejected by the database.
- [ ] A suppression-violating row (`alarm_triggered=1`, `correction_factor != 1.0`) is rejected by the database.
- [ ] A duplicate `sku_id` in `inventory_current` is rejected.
- [ ] `inventory_current` contains **no engineered features and no target column**.
- [ ] The `inventory_current` columns were derived from the raw CSV header cross-checked against the saved feature-column artifacts — not hand-typed, not copied from `processed_dataset.pkl`.
- [ ] `predictions` matches the frozen `PipelineState` field for field.
- [ ] `data/app.db` is gitignored; `src/db/` is committed.
- [ ] Deleting `data/app.db` and re-running `create_db.py` reproduces the schema identically.

---

## 9. Explicitly out of scope

- **Loading any data.** No seeding, no CSV import, no parquet reads at runtime.
- Seed data plan (later spec): stratified sample of ~5,000–10,000 SKUs from the raw CSV with `SEED=42`, deliberately including ~250 alarm-positive, ~250 high-urgency, ~50 rows with missing values, remainder random. Cherry-picking only clean rows is explicitly rejected — it would remove the alarm-positive cases the suppression path depends on and leave the messy-input path untested.
- Alembic migrations.
- A `supplier_audits` table. Agent 6's verdict lands in `predictions.supplier_risk`. If Agent 6 later runs standalone per-supplier rather than per-SKU, add the table then.

---

## 10. Notes for the paper

`agent_traces` plus `pipeline_runs.manifest_version` is the reproducibility story: every prediction traces to the exact agent sequence and the exact model artifacts that produced it.

The suppression CHECK constraint is worth a sentence in the methods section — the invariant is not merely asserted in code, it is enforced by the storage layer and cannot be violated by a passing test suite.
