# Seed Data Spec — Populating the Database

**Status:** Finalized. SKU-cardinality audit confirmed **Case A** — every row in the raw CSV is a unique SKU. No composite keys, no time-series reduction. This spec reflects that.

**Scope:** Load a stratified sample of SKUs from the raw CSV into `suppliers`, `skus`, and `inventory_current`. Nothing else.
**Owner:** Person B (Backend + Database track)
**Prerequisites (both complete):**
- Database schema exists (`database_spec.md`)
- `features.py` value-level cleaning function exists and is the same function used by the agent wrappers (confirmed via wrapper parity checks)

**Runs after:** Database creation, `features.py`. Independent of the LangGraph orchestrator — can run in parallel with agent wiring.

---

## 0. What this is and is not

Populates the three tables describing *what exists in the world*:

- `suppliers` — supplier identities
- `skus` — SKU identities
- `inventory_current` — one raw current-state row per SKU

Does **not** touch the four execution tables (`pipeline_runs`, `predictions`, `agent_traces`, `forecast_actuals`) — those are written only by the running pipeline.

Does **not** load the 1.9M-row dataset. Loads a deliberate sample of a few thousand SKUs.

---

## 1. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Source | The **raw CSV**, not `processed_dataset.pkl` | The pkl is a batch artifact with no `sku` column and no path to single-row serving |
| SKU cardinality | **Case A — confirmed.** One row = one SKU | `inventory_current` PK is plain `sku_id`, no composite key |
| Sample size | ~5,000 SKUs (tunable) | Populates dashboards; ships inside the desktop app at ~2–3 MB |
| Sampling | **Stratified**, not random, not cherry-picked | Random loses the rare alarm cases at 137:1 imbalance; cherry-picking clean rows leaves the messy-input path untested |
| Seed | `SEED=42` | Reproducible, documentable in the paper |
| Cleaning | Value-level only, via `features.py` | Same function the agents call at request time — one implementation, two callers |
| Engineered features | **Not stored** | Derived at runtime by the agent wrappers |
| Target column | **Not stored** | Leakage risk with no serving purpose |
| Idempotency | `--force` truncates the three tables, then reseeds | Reproducible re-runs |

---

## 2. Files to create

```
src/db/
    seed.py            # entrypoint
scripts/
    sample_skus.py     # produces the stratified sample (importable + standalone)
```

---

## 3. Stratified sampling plan

For a target of 5,000 rows:

| Stratum | Count | Why it must be present |
|---|---|---|
| Alarm-positive (backorder-positive) | ~250 | Without these the Agent 2 → Agent 5 suppression path can never be demonstrated. At 137:1, random sampling yields almost none |
| High urgency | ~250 | Populates the rebalancer's "top N by urgency" view with real signal |
| Contains missing values | ~50 | Proves the live cleaning path works on genuinely dirty input |
| Random remainder | ~4,450 | Representative baseline |

Rules:

- Draw each stratum with `random_state=42`.
- A row may satisfy more than one stratum — deduplicate by `sku_id` after combining, top up the random remainder to reach the target.
- Alarm-positive and missing-value strata are **non-negotiable**. If the source has fewer than target, take all available and log the shortfall — never substitute clean rows to hit a round number.
- Print the realized composition at the end of the run.

---

## 4. Step-by-step build

### Step 1 — `scripts/sample_skus.py`
- Read the raw CSV.
- Build each stratum per §3 using the label column to identify alarm-positive rows.
- Combine, deduplicate on `sku_id`, top up the remainder.
- Return a DataFrame of **raw, uncleaned** rows (cleaning happens in the seeder, not here).
- Expose both an importable function and a `__main__` that writes `data/seed_sample.parquet` for inspection before committing to the DB.

### Step 2 — `src/db/seed.py`, supplier extraction
- Extract distinct suppliers from the sample. Insert into `suppliers`.
- If the raw data has no supplier column at all, insert a single synthetic `'UNKNOWN'` supplier so FK targets exist, and note this as a data limitation.

### Step 3 — `src/db/seed.py`, SKU insertion
- Insert one row per `sku_id` into `skus`, linking `supplier_id` (nullable).

### Step 4 — `src/db/seed.py`, inventory insertion
For each sampled row:
1. Apply the **value-level cleaning** function from `features.py`.
2. Select only the raw columns designated for `inventory_current` (per `database_spec.md` §3.3). Drop everything engineered.
3. **Assert the target column is not among them** — fail loudly if it is.
4. Set `snapshot_at` and `updated_at` to now (UTC).

Bulk insert into `inventory_current`. `sku_id` is the plain PK — a duplicate insert is a sampling bug and should raise, not silently upsert.

### Step 5 — `--force` handling
- Without `--force`: refuse to run if any of the three tables is non-empty.
- With `--force`: delete from `inventory_current`, `skus`, `suppliers` in FK-safe order, then reseed.
- Never touch the four execution tables.

### Step 6 — Run
```
python -m scripts.sample_skus          # optional: inspect the sample first
python -m src.db.seed                   # seed
python -m src.db.seed --force           # rebuild from scratch
```

---

## 5. Post-seed verification (`src/db/seed.py --verify`)

After seeding, assert and print:

1. `suppliers`, `skus`, `inventory_current` row counts are all > 0.
2. Every `inventory_current.sku_id` has a matching `skus` row.
3. `inventory_current` has exactly one row per `sku_id`.
4. No column in `inventory_current` matches the target column name.
5. No engineered-feature column (from a denylist) is present.
6. Count of alarm-positive SKUs in the seed is > 0, cross-checked against the sample's known labels. Zero means the suppression demo is impossible and the seed must be redrawn.
7. Spot-check: 5 seeded rows through `features.py`'s full transform, confirm no NaN/inf survives into the feature matrix.
8. Print the realized stratum composition.

---

## 6. Acceptance criteria

- [ ] `suppliers`, `skus`, `inventory_current` populated; the four execution tables remain empty.
- [ ] `inventory_current` holds exactly one row per SKU.
- [ ] Seed contains ≥ 1 alarm-positive SKU (target ~250), verified by label join.
- [ ] `inventory_current` contains no target column and no engineered features.
- [ ] Seeded rows pass through `features.py` with no surviving NaN/inf.
- [ ] `--force` cleanly rebuilds without touching execution tables.
- [ ] Realized stratum composition printed and recorded.
- [ ] Re-running with the same seed produces the same sample.

---

## 7. Out of scope

- Seeding `forecast_actuals` — populated later from realized demand.
- Writing `pipeline_runs` / `predictions` / `agent_traces` — only the running pipeline writes these.
- Loading the full 1.9M-row dataset — stays as parquet in `data/raw/`, training only.

---

## 8. Notes for the paper

Document the seed as a stratified demonstration sample, `SEED=42`, with its realized composition. State plainly that it over-samples the minority (backorder-positive) class relative to the true 137:1 population rate as a deliberate demonstration-coverage choice, not a claim about production prevalence — and that this has no effect on any trained model, since all six models were fit on the full imbalanced distribution.
