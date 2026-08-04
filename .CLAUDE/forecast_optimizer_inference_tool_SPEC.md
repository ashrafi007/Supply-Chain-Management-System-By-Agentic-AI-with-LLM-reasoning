# Spec: Forecast Optimizer (Agent 3) Inference Tool

## Objective
Extract the trained Forecast Optimizer models from `Forcast_opt.ipynb` into a
standalone, importable Python module exposing a single deterministic
inference function. This module will later be registered as a tool inside a
LangGraph multi-agent pipeline, so it must have **zero LangChain / LangGraph
/ LLM dependencies** — pure model inference only.

This is Phase 2 of the larger roadmap (`RSM_Agentic_Roadmap.pdf`). Do not
touch Phase 3+ concerns (agent reasoning, prompts, orchestration) here.

## Predictor vs. reasoner separation (non-negotiable)
This module contains only feature engineering, thresholding, and model
inference. No `@tool` decorators, no LangChain/LangGraph imports, no
prompts. A later, separate file (Phase 3) will thinly wrap this module's
function with LangChain's tool interface.

## Ground truth from the notebook (read this before writing any code)

**Two-task design.** This agent solves two problems from the same feature
set:
- **Task A (classification):** `is_overestimating` — detects whether the
  human-made forecast is a chronic overestimate. Target defined as
  `bias_ratio_3m > 1.15` where `bias_ratio_3m = forecast_3_month / (sales_3_month + 1e-5)`.
- **Task B (regression):** `correction_factor` — a multiplier to correct the
  human forecast, defined as `(sales_3_month / (forecast_3_month + 1e-5))`
  clipped to `[0.3, 1.5]`.

**Final selected models** (chosen by validation performance, NOT
necessarily the best-sounding algorithm — verify by loading
`agent3_thresholds.pkl['selected_model_a']` at runtime rather than
hardcoding):
- Task A candidates trained: LightGBM, CatBoost, BalancedRandomForest —
  selected by highest F1 on validation.
- Task B candidates trained: XGBoost, LightGBM, CatBoost — selected by
  lowest MAE on validation.

**⚠️ Known gap — Task B's selected model is not persisted.** The notebook
saves `'selected_model_a'` inside `agent3_thresholds.pkl`, but there is no
equivalent `'selected_model_b'` key saved anywhere. All three Task B models
are saved to disk (`agent3_corrector_xgb.pkl`, `agent3_corrector_lgbm.pkl`,
`agent3_corrector_catboost.pkl`), but nothing on disk records which one was
actually `final_model_b`. **Resolve this before implementation** — check the
notebook's saved output/logs for the printed `"Task B final: {best_b}"`
line, or re-run the validation comparison cell to regenerate the answer, and
hardcode/document the answer in this module. Do not guess; do not silently
default to XGBoost just because it's listed first.

**Dependency on Agent 2 (Risk Detector).** The notebook loads
`stacking_proposed.pkl` and `optimal_thresholds.pkl` directly to compute:
```
backorder_probability = agent2_model.predict_proba(X_agent2)[:, 1]
alarm_triggered        = (backorder_probability >= agent2_threshold)
risk_level_encoded     = encode_risk_level(backorder_probability)
    # >= 0.80 -> 3 (CRITICAL), >= 0.60 -> 2 (HIGH),
    # >= 0.35 -> 1 (MEDIUM), else -> 0 (LOW)
```
**This inference tool must NOT replicate that coupling** (same principle as
the Inventory Rebalancer tool). `backorder_probability` and `alarm_triggered`
must be **required input parameters** supplied by the caller (the
orchestrator, from the Risk Detector tool's output). `risk_level_encoded`
can be derived internally from the supplied `backorder_probability` using
`encode_risk_level()` above — that function has no model dependency, just
copy it verbatim. Document this deviation from the notebook in the module
docstring.

## Feature engineering pipeline (must run in this exact order)

**Step 1 — `create_features()`** (base features, same function as used in
the Inventory Rebalancer notebook — verify it is byte-identical between the
two notebooks before assuming so; if it has diverged, use *this* notebook's
version):
```
inventory_health_net  = (national_inv + in_transit_qty) - sales_1_month
sales_spike_ratio      = sales_1_month / ((sales_9_month / 9) + 0.001)
months_of_stock_left   = national_inv / (sales_1_month + 0.001)
backorder_pressure     = (pieces_past_due * (deck_risk + 1)) / (national_inv + 1)
supplier_risk_score    = deck_risk + oe_constraint + ppap_risk
forecast_accuracy_gap  = abs(forecast_3_month - sales_3_month) / (sales_3_month + 1)
inv_depletion_rate     = (sales_9_month - sales_1_month * 9) / (sales_9_month + 1)
lead_time_volatility   = lead_time * perf_gap
safety_stock_urgency   = (min_bank - national_inv) / (min_bank + 1)
performance_trend      = perf_6_month_avg - perf_12_month_avg
demand_forecast_ratio  = forecast_3_month / ((sales_9_month / 9) + 0.001)
transit_coverage       = in_transit_qty / (sales_1_month + 0.001)
coverage_ratio         = (national_inv + in_transit_qty) / (forecast_3_month + 1e-5)
replenishment_gap      = forecast_3_month - national_inv - in_transit_qty
critical_state         = (months_of_stock_left < lead_time)
                          AND (safety_stock_urgency > 0)
                          AND (replenishment_gap > 0)          # -> 0/1
# then: replace all +inf/-inf with 0 (NOT NaN — this notebook uses 0,
# unlike the Inventory Rebalancer notebook which used median-fill; do not
# unify these behaviors, replicate each notebook's actual choice)
```
⚠️ Same undefined-column issue as the Inventory Rebalancer notebook:
`perf_gap` is used in `lead_time_volatility` but never defined in this
notebook either — confirm it exists in `Processed_Dataset.pkl` before
implementing. If you already resolved this for the Inventory Rebalancer
tool, reuse that resolution here for consistency.

**Step 2 — integrate Agent 2 signals** (replace with passed-in values per
the dependency section above):
```
backorder_probability, alarm_triggered   <- supplied by caller
risk_level_encoded                        <- derived via encode_risk_level()
```

**Step 3 — `create_agent3_features()`** (Task-specific features):
```
bias_ratio_3m           = forecast_3_month / (sales_3_month + 1e-5)
bias_ratio_6m           = forecast_6_month / (sales_6_month + 1e-5)
bias_ratio_9m           = forecast_9_month / (sales_9_month + 1e-5)
error_3m                = forecast_3_month - sales_3_month
error_6m                = forecast_6_month - sales_6_month
error_9m                = forecast_9_month - sales_9_month
bias_trend_3_to_9       = bias_ratio_9m - bias_ratio_3m
chronic_overestimator   = (bias_ratio_3m > 1.10) AND (bias_ratio_6m > 1.10)
                           AND (bias_ratio_9m > 1.10)                # -> 0/1
forecast_consistency    = std(forecast_3_month, forecast_6_month, forecast_9_month)
                           / (forecast_9_month + 1e-5)
sales_velocity_ratio    = sales_1_month / ((sales_9_month / 9) + 1e-5)
demand_volatility       = std(sales_1_month, sales_3_month, sales_6_month, sales_9_month)
                           / (sales_9_month + 1e-5)
sales_momentum          = (sales_1_month - (sales_3_month / 3)) / (sales_3_month / 3 + 1e-5)
risk_adjusted_bias      = bias_ratio_3m * (1 - backorder_probability)
correction_suppressed   = int(alarm_triggered)
inv_vs_forecast         = (national_inv + in_transit_qty) / (forecast_3_month + 1e-5)
supplier_forecast_stress = (1 - clip(perf_6_month_avg, 0, 1)) * bias_ratio_3m
# then: replace inf with NaN, then fillna with per-column median
```
⚠️ **Known gap, same class of issue as before:** the median used for
`fillna` in this step is computed live from whatever DataFrame is passed
in (`d.fillna(d.median(numeric_only=True))`), and is **not a persisted
artifact**. For a single-row or small-batch inference call, computing a
median from 3–5 rows is meaningless/unstable. Resolve explicitly — do not
silently reuse a batch-local median for production inference. Preferred
option: precompute and persist column medians from the full historical
dataset once (via the DuckDB data-access layer) and load them here, exactly
as recommended for the Inventory Rebalancer tool's equivalent gap. Document
the choice in the module docstring.

**Step 4 — feature selection.** `FEATURE_COLS` = every numeric column in
the Step-3 output **except**:
```python
EXCLUDE_COLS = [
    'went_on_backorder', 'is_overestimating', 'correction_factor',
    'adjusted_forecast_3m', 'error_3m', 'error_6m', 'error_9m',
    'bias_ratio_3m', 'bias_ratio_6m', 'bias_ratio_9m',
    'chronic_overestimator', 'bias_trend_3_to_9',
    'risk_adjusted_bias', 'supplier_forecast_stress',
    'sales_3_month', 'sales_6_month', 'sales_9_month',
    'demand_volatility', 'sales_momentum', 'sales_velocity_ratio',
]
```
Load the authoritative, exact list from the saved `agent3_feature_cols.pkl`
artifact rather than re-deriving it from this spec — this list is provided
here only so you can sanity-check the loaded artifact matches expectations.

⚠️ **Flag this, do not silently fix it:** `EXCLUDE_COLS` removes raw
`sales_3_month/6_month/9_month` and a few derived columns to prevent target
leakage, but **`forecast_accuracy_gap`** (from Step 1) is *not* in
`EXCLUDE_COLS`, even though its formula directly uses `sales_3_month` — the
same variable both targets are built from. This is a potential leakage path
the original notebook's "LEAKAGE FIX" comment block did not close. Similarly
`inv_depletion_rate` uses `sales_9_month` directly and is also not excluded.
**Do not silently exclude these yourself either** — implement the wrapper
faithful to what the saved model was actually trained on (whatever is in
`agent3_feature_cols.pkl`), but surface this finding prominently to the team
in the module docstring and in your task summary, since it affects whether
the reported Task A/B metrics are trustworthy. This is a modeling
methodology question for the team/paper, not something to fix inside an
inference wrapper.

## Inference logic — reference implementation exists, adapt it
The notebook already contains a single-row reference inference function,
`agent3_predict(product_row, model_a, model_b, threshold_a, feature_cols)`.
Use it as the ground truth for exact business logic, but adapt it to:
1. Accept batches of rows, not just one (vectorize where practical; a loop
   over rows calling the same per-row logic is acceptable for a first
   version, but note if it becomes a performance concern for large batches).
2. Not access a plain `product_id` field unless confirmed present in the
   real dataset — **check whether `Processed_Dataset.pkl` actually has a
   `product_id` column, or whether this should be `sku`** (the rest of the
   system keys everything by `sku`). Do not assume; verify against the
   dataset schema and use whichever key actually exists.

Exact logic to replicate:
```python
bias_probability   = model_a.predict_proba(X_row)[:, 1]
bias_detected       = bias_probability >= final_threshold_a
correction_factor   = model_b.predict(X_row)

# Business override: if Agent 2's alarm already fired for this SKU,
# force correction_factor to 1.0 (i.e. trust the human forecast, don't
# also apply a bias correction on top of an active risk alarm)
if alarm_triggered:
    correction_factor = 1.0

adjusted_forecast_3m = forecast_3_month * correction_factor

if correction_factor < 0.90:   recommendation = 'REDUCE_PLAN'
elif correction_factor > 1.10: recommendation = 'INCREASE_PLAN'
else:                           recommendation = 'HOLD'

bias_severity = 'SEVERE' if correction_factor < 0.75 else \
                'MILD'   if correction_factor < 0.90 else 'NONE'
```
Preserve the override-to-1.0 behavior and the threshold cutoffs
(0.75/0.90/1.10) exactly — these encode real business logic from the team,
not arbitrary constants.

## Artifacts to load (confirm actual path via config, do not hardcode the
notebook's absolute Windows path)
```
agent3_bias_detector_lgbm.pkl        # Task A candidate
agent3_bias_detector_catboost.pkl    # Task A candidate
agent3_bias_detector_brf.pkl         # Task A candidate
agent3_corrector_xgb.pkl             # Task B candidate
agent3_corrector_lgbm.pkl            # Task B candidate
agent3_corrector_catboost.pkl        # Task B candidate
agent3_feature_cols.pkl              # authoritative FEATURE_COLS — required
agent3_thresholds.pkl                # contains selected_model_a + final_threshold_a
                                      # — required; does NOT contain
                                      #   selected_model_b (see gap above)
agent3_business_metrics.pkl          # informational only, not needed at inference
```
Only load the **two selected** models (one Task A, one Task B) at runtime —
do not load all six candidates into memory for every inference call.

## Non-goals
- No training code, no grid search, no plotting, no SHAP, no CV.
- No LangChain/LangGraph tool decorators.
- No changes to the DuckDB data-access layer.
- Do not load `stacking_proposed.pkl` or `optimal_thresholds.pkl` from
  within this module (see dependency section above).
- Do not modify the original notebook.
- Do not silently resolve the Task B model-selection gap or the
  `product_id`/`sku` question without checking actual artifacts/schema.

## Deliverable: file structure
```
inference_tools/
├── schemas.py                       # add to this shared file, do not fork it
└── forecast_optimizer_tool.py       # new file, this task's main deliverable
tests/
└── test_forecast_optimizer_tool.py  # new file
```

## Schema contract (`schemas.py`)

```python
class ForecastOptimizerInput(BaseModel):
    skus: list[str]
    data: dict
        # raw feature rows for these SKUs — must include every raw column
        # needed for create_features() and create_agent3_features().
        # Document exact expected shape in a docstring.
    backorder_probability: dict[str, float]   # sku -> probability, REQUIRED
    alarm_triggered: dict[str, bool]           # sku -> bool, REQUIRED
        # both must come from the Risk Detector tool's output via the
        # orchestrator — see dependency note above

class ForecastOptimizerRecommendation(BaseModel):
    sku: str
    human_forecast_3m: float
    adjusted_forecast_3m: float
    correction_factor: float
    bias_detected: bool
    bias_probability: float
    bias_severity: str          # "SEVERE" | "MILD" | "NONE"
    risk_override_applied: bool
    recommendation: str          # "REDUCE_PLAN" | "INCREASE_PLAN" | "HOLD"

class ForecastOptimizerOutput(BaseModel):
    recommendations: list[ForecastOptimizerRecommendation]
    flagged_for_reduction: list[str]   # SKUs with recommendation == REDUCE_PLAN
    flagged_for_increase: list[str]    # SKUs with recommendation == INCREASE_PLAN
    model_version_a: str               # e.g. "lightgbm_bias_detector_v1"
    model_version_b: str               # e.g. "xgboost_corrector_v1" — must
                                        # reflect whichever Task B model was
                                        # actually resolved (see gap above)
```

## Function contract (`forecast_optimizer_tool.py`)

```python
def load_forecast_optimizer_artifacts(models_dir: str | None = None):
    """
    Load only the two SELECTED models (Task A per agent3_thresholds.pkl's
    'selected_model_a'; Task B per the resolved gap above), plus
    agent3_feature_cols.pkl and the Task A threshold. Cache
    (module-level singleton or functools.lru_cache).
    Raise a clear, typed exception if a required artifact is missing.
    """

def optimize_forecast(input_data: ForecastOptimizerInput) -> ForecastOptimizerOutput:
    """
    Pure inference. Given SKUs + raw feature rows + backorder_probability +
    alarm_triggered:
      1. Run create_features() (Step 1 above), resolving perf_gap per the
         team's earlier decision for the Inventory Rebalancer tool.
      2. Attach backorder_probability, alarm_triggered, and derive
         risk_level_encoded via encode_risk_level().
      3. Run create_agent3_features() (Step 3 above), resolving the
         unpersisted-median gap per the documented decision.
      4. Select FEATURE_COLS from the loaded artifact.
      5. Run Task A model -> bias_probability, bias_detected.
      6. Run Task B model -> correction_factor, then apply the
         alarm_triggered override to 1.0 exactly as in agent3_predict().
      7. Compute adjusted_forecast_3m, recommendation, bias_severity exactly
         as in agent3_predict().
      8. Return ForecastOptimizerOutput as defined above.
    Must be deterministic: same input -> same output, every time.
    Must raise a clear, typed exception (not a bare Exception) if:
      - a requested SKU has no matching row in the input data
      - a requested SKU has no entry in backorder_probability or alarm_triggered
      - required raw columns for feature engineering are missing
      - any loaded artifact is malformed/incompatible
    """
```

## Determine and document during implementation
State these explicitly in a docstring at the top of
`forecast_optimizer_tool.py`:
- Which Task B model was actually `final_model_b` (resolve the persistence
  gap above) and how you determined it.
- Resolved definition/source of `perf_gap` (reuse the Inventory Rebalancer
  tool's resolution if already decided, for consistency).
- Resolved strategy for the unpersisted per-column medians in
  `create_agent3_features()`.
- Whether the dataset uses `product_id` or `sku` as the row key, confirmed
  against the actual schema.
- A note flagging the potential leakage via `forecast_accuracy_gap` and
  `inv_depletion_rate` remaining in `FEATURE_COLS` (see above) — for the
  team to evaluate, not for this module to silently fix.

## Acceptance criteria
- [ ] `forecast_optimizer_tool.py` has no imports from `langchain`,
      `langgraph`, or any LLM client library, and does not import or load
      `stacking_proposed.pkl` / `optimal_thresholds.pkl`.
- [ ] Docstring documents all five "determine and document" items above.
- [ ] Only two models loaded into memory (one Task A, one Task B), not all six.
- [ ] Artifacts load exactly once per process (cached).
- [ ] `optimize_forecast()` runs on a small hand-built sample (3–5 rows)
      covering: a normal row, a row with `alarm_triggered=True` (verify
      `correction_factor` is forced to 1.0 and `risk_override_applied=True`),
      a row with `-99.0` in perf averages, a row with NaN `lead_time`.
- [ ] Output matches `ForecastOptimizerOutput` exactly — validate with
      Pydantic, not just visually.
- [ ] `correction_factor` stays within the trained range behavior (clipped
      target was `[0.3, 1.5]` at training time — if raw model output falls
      outside a sane range, decide whether to clip and document it).
- [ ] `flagged_for_reduction` / `flagged_for_increase` are derived views
      consistent with `recommendations`, not independently computed.
- [ ] Unit tests cover: normal input, missing SKU, missing
      `backorder_probability`/`alarm_triggered` entries, malformed input,
      the alarm-override behavior, and determinism.
- [ ] No metrics are computed anywhere in this file.
- [ ] Module runs on both CPU-only and CUDA-available machines without code
      changes (only relevant if the selected models use GPU-capable
      libraries — LightGBM/BalancedRandomForest stay CPU regardless).

## Out of scope / do not do
- Do not build the LangGraph node or agent prompt for the Forecast
  Optimizer — Phase 3/4, separate task.
- Do not touch the DuckDB data-access layer.
- Do not call the Risk Detector's model or load its artifacts from within
  this module.
- Do not retrain or regenerate any model artifact.
- Do not silently resolve the Task B model-selection gap without checking
  the actual notebook output/logs first.
