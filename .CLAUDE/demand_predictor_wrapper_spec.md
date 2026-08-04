# Inference Wrapper Spec — Agent 1: Demand Predictor

**Scope:** Wrap the trained Demand Predictor for single-row and batch inference behind the uniform agent interface. No retraining.
**Owner:** Person A (Agents + Orchestration track)
**Prerequisite:** The trained model bundle(s) exist on disk. `features.py` exists.
**Parity target:** The wrapper must reproduce the notebook's test-set predictions to within `rtol=1e-5` (see `verification_spec.md`).

---

## 1. What this model actually is (read before writing anything)

The Demand Predictor is **not** a plain `model.predict(X) → units` regressor. It is a LightGBM model with a **log-ratio correction head**. Getting the wrapper wrong here produces predictions that are silently, badly off — no exception, just nonsense units.

The chain, from the notebook:

1. The model does **not** predict demand directly. It predicts a **log-space delta** against a naive baseline `b`:
   ```
   delta_hat = model.predict(X, num_iteration=best_iteration)
   ```
   where the training target was `to_delta(y, b) = log1p(max(y,0)) - log1p(max(b,0))`.

2. The prediction in units is recovered by inverting that delta, applying a **Duan smearing factor** `s` (a multiplicative bias correction tuned on the validation set), and clipping at zero:
   ```
   demand = from_delta(delta_hat, b, s)      # ≈ expm1( delta_hat + log1p(max(b,0)) ) * s
   demand = max(demand, 0)
   ```

Three quantities in that chain are **learned or tuned and must be loaded from disk — never recomputed at inference:**

| Quantity | Notebook name | If recomputed at serve time |
|---|---|---|
| Boosting rounds | `best_iteration` / `BEST_ROUND` | Uses all rounds → overfit predictions |
| Smearing factor | `OPT_SMEAR` (val-tuned) | Systematic bias returns; the whole correction head is defeated |
| Baseline `b` | naive forecast per stage | Delta is measured against the wrong anchor → wrong units |

The fitted **imputer + RobustScaler** pipeline (`SimpleImputer(median)` → `RobustScaler`) is a fourth. It must be the train-fitted object loaded from disk. Refitting it on live rows is the classic train/serve skew bug — the same class as the unsaved `URG_NORM_PARAMS` issue on Agent 3.

---

## 2. Multi-stage decision (BLOCKING — resolve first)

The notebook trains multiple horizons: stages `h3`, `h6`, `h9` predicting `sales_3_month`, `sales_6_month`, `sales_9_month` (and a 1-month window as input). The frozen `PipelineState` has a single `demand_forecast` field.

Decide, and record the decision in the wrapper's docstring:

- **Option A (recommended):** one horizon is the canonical demand signal that populates `demand_forecast` (6-month is the usual choice for replenishment). The wrapper loads that stage's bundle by default and exposes the others via an optional `horizon` argument.
- **Option B:** the pipeline needs all horizons. Extend `PipelineState` to `demand_forecast: dict[str, float]` **before** wiring this agent, and update the `predictions` table to match.

Do not proceed until this is chosen — it changes the wrapper's return shape and the database schema.

---

## 3. Required artifacts (audit the notebook's export cell FIRST)

Section 18 of the notebook saves a `bundle` dict via `joblib.dump`. Open it and confirm it contains **all** of the following. If any is missing, add it to the export cell and re-export before writing the wrapper — the wrapper cannot be correct without them.

| Key | Purpose | Confirmed present? |
|---|---|---|
| `model` | The trained LightGBM regressor | Listed |
| `features` (`FEATURE_COLS`) | Exact feature names **and order** the model expects | Listed |
| `best_iteration` | Boosting rounds for `predict` | Listed |
| `smearing_factor` (`OPT_SMEAR`) | Val-tuned Duan factor | Listed |
| `target` / `stage` | Which horizon this bundle predicts | Listed |
| **imputer + scaler pipeline** | The **fitted** `SimpleImputer`→`RobustScaler` | **VERIFY — likely missing** |
| **baseline spec** | How to construct `b` from raw columns for this stage | **VERIFY — must be explicit** |

The last two are the ones most likely absent. The bundle listing in the notebook names "model, features, smearing factor, best_iteration, inverse-transform spec" — confirm the *fitted transformer object* and the *baseline construction rule* are actually inside, not just implied. If the imputer/scaler was fit inline and never saved, **re-run the export with it included.**

---

## 4. Feature handling — use the saved list, verbatim

The notebook's ablation found the engineered features (`velocity`, `accel`, `forecast_bias`, the `log_*` transforms) to be **within noise or slightly harmful**, and recommends the parsimonious raw+baseline feature set (stage "D"). This means **you cannot assume which features the final model uses.**

Rule: the wrapper computes exactly the columns in the saved `FEATURE_COLS`, in that order, and nothing else.

- If `FEATURE_COLS` contains an engineered column (e.g. `log_national_inv`), compute it via `features.py` using the notebook's exact formula (`log1p(national_inv.clip(lower=0))`).
- If it does not, do not compute it. Extra columns, or columns in the wrong order, silently corrupt LightGBM input.
- The **leakage guard** carries into serving: each stage only consumes sales windows strictly shorter than its target. The wrapper's required raw inputs are therefore the shorter-window sales columns plus whatever else `FEATURE_COLS` references. Document the required raw input columns explicitly.

---

## 5. The inference sequence (the heart of the wrapper)

For a raw input row (or batch), in this exact order:

1. **Validate inputs.** Assert every raw column that `FEATURE_COLS` and the baseline depend on is present. If any required column is missing, return a failed result (see §7) — do not fabricate.
2. **Clean (value-level).** Apply the same value-level cleaning as every other agent via `features.py`. No row dropping.
3. **Construct the baseline `b`** for this stage from the raw columns, using the saved baseline spec. This is the anchor the delta is measured against — getting it wrong is the most common way this specific wrapper fails.
4. **Build the feature matrix** `X` = the saved `FEATURE_COLS`, in order, including `log_base_pred = log1p(max(b,0))` if it is one of them.
5. **Transform** `X` with the **loaded fitted** imputer→scaler pipeline. Do not `fit`. Only `transform`.
6. **Predict the delta:** `delta_hat = model.predict(X, num_iteration=best_iteration)`.
7. **Invert:** `demand = from_delta(delta_hat, b, smearing_factor)`, then `max(demand, 0)`.
8. **Return** the demand value(s), rounded consistently with the notebook (`round(pred, 2)`).

Steps 3, 5, 6, 7 are where parity breaks. When the parity test fails, check them in that order.

---

## 6. Uniform agent interface

Wrap the sequence above so this agent is interchangeable with the other five and drops into LangGraph as a node.

- Expose a class (e.g. `DemandPredictorAgent`) that loads its bundle **once** at construction, from the manifest — not per call.
- Expose a `run(state) -> dict` method that reads `state["raw_features"]`, executes §5, and returns a **state delta**: `{"demand_forecast": <value>}` (or the dict form under Option B), plus an appended `trace` entry.
- On failure, return a delta that records the failure in `trace` and leaves `demand_forecast` as `None` — the graph continues; downstream agents that don't depend on demand still run.
- The agent must not read or write the database. It operates purely on the in-memory state.

This agent is an **entry-point node** (no upstream agent dependency), so it typically runs first in the graph.

---

## 7. Failure handling

- Missing required raw column → failed result, reason recorded, `demand_forecast=None`. Never impute a missing *input structural* column into existence.
- `inf`/NaN surviving into `X` after cleaning → this indicates a cleaning gap; fail loudly in tests, and in production record the failed run rather than feeding NaN to LightGBM.
- Serving runs on CPU (Apple Silicon M2). LightGBM `predict` is CPU by default; if the saved params carry `device='gpu'`, strip or ignore it at load — it must not force a GPU path at inference.

---

## 8. Parity hook (ties to `verification_spec.md`)

Before this wrapper is considered done, it must pass the parity test:

1. In the notebook, save `agent_1_reference.parquet`: the sampled raw rows' `_raw_idx` and the notebook's final `predicted_demand` for the chosen stage.
2. The wrapper runs on those same raw rows.
3. Assert `np.allclose(wrapper_demand, notebook_demand, rtol=1e-5, atol=1e-8)`.

Add the **single-row-vs-batch** assertion specifically here: run one row alone, compare to that row's value from the batch. If they differ, the imputer/scaler is being refit per call — the exact bug §1 and §3 warn about.

---

## 9. Acceptance criteria

- [ ] Multi-stage decision (§2) is made and recorded; `PipelineState` / `predictions` updated if Option B.
- [ ] The bundle contains the fitted imputer→scaler pipeline and an explicit baseline spec (re-exported if they were missing).
- [ ] The wrapper loads all artifacts once at construction, from the manifest.
- [ ] `FEATURE_COLS` is used verbatim, in order; no extra or reordered columns.
- [ ] `best_iteration`, `smearing_factor`, and baseline `b` are all loaded, never recomputed.
- [ ] The imputer→scaler is only `transform`ed, never `fit`.
- [ ] `run(state)` returns a clean `{"demand_forecast": ...}` delta plus a trace entry.
- [ ] Parity test passes at `rtol=1e-5` against the notebook on the reference rows.
- [ ] Single-row-vs-batch assertion passes (no per-call refitting).
- [ ] Missing-column and NaN cases fail cleanly, not silently.
- [ ] Inference runs on CPU with no GPU dependency.

---

## 10. Notes for the paper

The log-ratio parameterisation with a val-tuned Duan smearing factor is the notebook's stated single defensible contribution, and the ablation's null result on the engineered features is part of that story. The wrapper is where that contribution either survives deployment or quietly dies: if the smearing factor is recomputed, or the baseline is reconstructed differently at serve time, the deployed model no longer matches the reported results. Passing the `rtol=1e-5` parity check is the evidence that the deployed forecast is the same estimator described in the paper — worth one sentence in the deployment/reproducibility section.
