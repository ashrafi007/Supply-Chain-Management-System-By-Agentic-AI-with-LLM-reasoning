# Spec: Inventory Rebalancer (Agent 3) Inference Tool

## Objective
Extract the trained Inventory Rebalancer model from
`Inventory_Rebalencer.ipynb` into a standalone, importable Python module
exposing a single deterministic inference function. This module will later
be registered as a tool inside a LangGraph multi-agent pipeline, so it must
have **zero LangChain / LangGraph / LLM dependencies** — pure model
inference only.

This is Phase 2 of the larger roadmap (`RSM_Agentic_Roadmap.pdf`). Do not
touch Phase 3+ concerns (agent reasoning, prompts, orchestration) here.

## Predictor vs. reasoner separation (non-negotiable)
This module contains only feature engineering, imputation, and model
inference. No `@tool` decorators, no LangChain/LangGraph imports, no
prompts. A later, separate file (Phase 3) will thinly wrap this module's
function with LangChain's tool interface.

## Ground truth from the notebook (read this before writing any code)

**Model:** `xgb.XGBRegressor`, target = `urgency_score` (continuous, 0–1).
Reported test metrics: R² = 0.9976, RMSE = 0.005455, MAE = 0.002674,
Spearman ρ = 0.9764. Trained with GPU (`device='cuda'`, `tree_method='hist'`)
when available, CPU fallback via `torch.cuda.is_available()` — reuse this
exact detection pattern, do not invent a different one.

**Dependency on Agent 2 (Risk Detector):** the notebook loads
`stacking_proposed.pkl` directly and runs `predict_proba()` to generate
`backorder_probability`, which is then used both as (a) a training feature
and (b) part of the `urgency_score` formula (35% weight).
**This inference tool must NOT replicate that coupling.** Per the
predictor/reasoner separation principle, `backorder_probability` must be a
**required input parameter** supplied by the caller (the orchestrator, which
gets it from the Risk Detector tool's output) — this module must not load
`stacking_proposed.pkl` itself. This is a deliberate architectural change
from how the notebook works, not a bug — document it clearly in the
module's top-of-file docstring so it isn't mistaken for an oversight.

**Feature engineering required before the model can run.** The notebook
computes 15 engineered columns from raw dataset fields before selecting a
13-feature subset. These must be replicated exactly, in this order of
dependency:

```
inventory_health_net  = (national_inv + in_transit_qty) - sales_1_month
sales_spike_ratio     = sales_1_month / (sales_9_month / 9 + 0.001)
months_of_stock_left  = national_inv / (sales_1_month + 0.001)
backorder_pressure    = (pieces_past_due * (deck_risk + 1)) / (national_inv + 1)
supplier_risk_score   = deck_risk + oe_constraint + ppap_risk
forecast_accuracy_gap = abs(forecast_3_month - sales_3_month) / (sales_3_month + 1)
inv_depletion_rate    = (sales_9_month - sales_1_month * 9) / (sales_9_month + 1)
lead_time_volatility  = lead_time * perf_gap
safety_stock_urgency  = (min_bank - national_inv) / (min_bank + 1)
performance_trend     = perf_6_month_avg - perf_12_month_avg
demand_forecast_ratio = forecast_3_month / (sales_9_month / 9 + 0.001)
transit_coverage      = in_transit_qty / (sales_1_month + 0.001)
coverage_ratio        = (national_inv + in_transit_qty) / (forecast_3_month + 1e-5)
replenishment_gap     = forecast_3_month - national_inv - in_transit_qty
critical_state        = (months_of_stock_left < lead_time)
                         AND (safety_stock_urgency > 0)
                         AND (replenishment_gap > 0)        # -> 0/1
```

⚠️ **`perf_gap` is referenced (`lead_time_volatility`) but its definition is
not visible in this notebook** — it must already exist in
`Processed_Dataset.pkl` as a precomputed column. **Confirm its definition
before implementing** (check the dataset schema / an earlier preprocessing
notebook). Do not guess a formula for it.

**The 13 model input features** (`REBALANCER_FEATURES`, exact order matters
for some model formats — persist the order from `rebalancer_feature_columns.pkl`
if that artifact exists, otherwise use this list in this order):

```
national_inv, min_bank, oe_constraint, months_of_stock_left,
inv_depletion_rate, safety_stock_urgency, backorder_pressure,
inventory_health_net, in_transit_qty, lead_time, coverage_ratio,
transit_coverage, backorder_probability
```

Note: `safety_gap` and `replenishment_gap` are **deliberately excluded**
from the model's features (target leakage — they're near-identical to
components of the urgency formula). Do not add them back in.

**Imputation:** a `SimpleImputer(strategy='median')` was fit on the
**training split only** and saved as `rebalancer_imputer.pkl`. This tool
must load and reuse that saved imputer's `.transform()` — never refit it.

**⚠️ Known gap to resolve before implementation:** separately from the
imputer, the notebook also does an inf → NaN → median-fill cleaning pass on
the 15 *engineered* columns (`ENGINEERED_COLS`), computed globally across
the full dataset, **before** train/test split. Those median values were
**never saved as an artifact** — only the post-cleaning, train-fit
`rebalancer_imputer.pkl` was persisted. This means a naive port of the
notebook's logic can't exactly reproduce that first cleaning step at
inference time for a new small batch. Resolve this explicitly — do not
silently skip it. Reasonable options, in order of preference:
  1. Recompute those medians once from the full historical dataset (via the
     DuckDB data-access layer) and store them as a new artifact
     (`rebalancer_engineered_col_medians.json` or similar) that this module
     loads at startup.
  2. If (1) isn't feasible, replace inf values with NaN and let the
     already-saved `rebalancer_imputer.pkl` handle NaNs during the final
     imputation step for the subset of engineered columns that overlap with
     `REBALANCER_FEATURES` — but flag that this changes behavior on the
     non-overlapping engineered columns, and document it.
  Pick one, document the choice and why in the module docstring, do not
  invent silently.

**Output fields the notebook produces** (Step 25):
```
urgency_score_predicted = xgb_model.predict(...)
manufacture_rank         = rank of urgency_score_predicted, descending, across the FULL dataset
recommended_qty          = clip(min_bank - national_inv, lower=0)
```

**⚠️ Design decision needed on `manufacture_rank`:** the notebook computes
rank via `.rank(ascending=False)` across the *entire* 1.9M-row dataset. For
an inference tool called with an arbitrary small batch of SKUs, a rank
computed only within that batch is **not the same thing** and would be
misleading if presented as "manufacture_rank" without qualification. Do not
silently rename or fudge this. Implement it as:
  - `urgency_score_predicted` — always returned, per-SKU, unambiguous.
  - `batch_rank` — rank within the current call's batch only, clearly named
    as such.
  - `manufacture_rank` (global) — only computable if the tool has access to
    a precomputed reference distribution/percentile table from the full
    dataset (e.g. stored during a periodic batch scoring job). If that
    doesn't exist yet, **omit this field** rather than mislabeling a
    batch-local rank as a global one, and note it as a follow-up need.

## Artifacts to load (from `models/` — confirm actual path via config, do
not hardcode the notebook's absolute Windows path)
```
rebalancer_xgboost.pkl           # the XGBoost model — required
rebalancer_feature_columns.pkl   # REBALANCER_FEATURES list/order — required
rebalancer_imputer.pkl           # fitted SimpleImputer — required
rebalancer_urgency_weights.pkl   # {'gap':0.40,'bo':0.35,'dep':0.15,'urg':0.10}
                                  # only needed if this tool also recomputes
                                  # urgency_score for validation — not
                                  # required for prediction itself
rebalancer_best_params.pkl       # informational only, not needed at inference
rebalancer_metrics.json          # informational only, not needed at inference
```
`rebalancer_linear_baseline.pkl` is the baseline model used for comparison
in the notebook — **do not load or serve it**, it's not the production model.

## Non-goals
- No training code, no grid search, no plotting, no SHAP, no ablation study.
- No LangChain/LangGraph tool decorators.
- No changes to the DuckDB data-access layer — this module receives
  already-fetched rows and an already-supplied `backorder_probability` per
  SKU; it does not query the database or call the Risk Detector itself.
- Do not load `stacking_proposed.pkl` from within this module (see
  dependency section above).
- Do not modify the original notebook.
- Do not silently invent values for `perf_gap` or the missing engineered-
  column medians — resolve both explicitly per the instructions above.

## Deliverable: file structure
```
inference_tools/
├── schemas.py                       # add to this shared file, do not fork it
└── inventory_rebalancer_tool.py     # new file, this task's main deliverable
tests/
└── test_inventory_rebalancer_tool.py  # new file
```

## Schema contract (`schemas.py`)

```python
class InventoryRebalancerInput(BaseModel):
    skus: list[str]
    data: dict
        # raw feature rows for these SKUs — must include every raw column
        # needed to compute the 15 engineered features (national_inv,
        # in_transit_qty, sales_1_month, sales_3_month, sales_9_month,
        # pieces_past_due, deck_risk, oe_constraint, ppap_risk,
        # forecast_3_month, lead_time, perf_gap, min_bank,
        # perf_6_month_avg, perf_12_month_avg). Document exact expected
        # shape in a docstring.
    backorder_probability: dict[str, float]
        # sku -> probability, REQUIRED (see dependency note above — this
        # must come from the Risk Detector tool's output via the orchestrator,
        # not be computed internally)

class RebalancerRecommendation(BaseModel):
    sku: str
    urgency_score_predicted: float      # 0.0-1.0
    recommended_qty: float              # units to reach safety floor, >= 0
    batch_rank: int                     # rank within this call's batch only
    manufacture_rank: int | None = None # global rank, only if a reference
                                         # distribution is available (see spec)

class InventoryRebalancerOutput(BaseModel):
    recommendations: list[RebalancerRecommendation]  # sorted by batch_rank
    top_priority_skus: list[str]        # derived from recommendations, not
                                         # independently computed
    model_version: str                  # e.g. "xgboost_urgency_v1"
```

## Function contract (`inventory_rebalancer_tool.py`)

```python
def load_rebalancer_artifacts(models_dir: str | None = None):
    """
    Load rebalancer_xgboost.pkl, rebalancer_feature_columns.pkl, and
    rebalancer_imputer.pkl. Cache (module-level singleton or
    functools.lru_cache) so repeated calls do not reload from disk.
    Reuse the notebook's CUDA-detection pattern (torch.cuda.is_available())
    for inference device selection, with CPU fallback.
    Raise a clear, typed exception if any required artifact is missing —
    do not fall back to the linear baseline model.
    """

def rebalance_inventory(input_data: InventoryRebalancerInput) -> InventoryRebalancerOutput:
    """
    Pure inference. Given SKUs + raw feature rows + backorder_probability:
      1. Compute the 15 engineered columns exactly as in the notebook
         (see formulas above), resolving perf_gap and the inf/NaN cleaning
         gap per the documented decision.
      2. Select the 13 REBALANCER_FEATURES (in the saved artifact's order),
         merging in the caller-supplied backorder_probability per SKU.
      3. Apply the loaded imputer's .transform() (never refit).
      4. Run xgb_model.predict() to get urgency_score_predicted.
      5. Compute recommended_qty = clip(min_bank - national_inv, 0).
      6. Compute batch_rank from urgency_score_predicted within this call.
      7. Sort by batch_rank and return InventoryRebalancerOutput.
    Must be deterministic: same input -> same output, every time.
    Must raise a clear, typed exception (not a bare Exception) if:
      - a requested SKU has no matching row in the input data
      - a requested SKU has no entry in backorder_probability
      - required raw columns for feature engineering are missing
      - any loaded artifact is malformed/incompatible
    """
```

## Determine and document during implementation
State these explicitly in a docstring at the top of
`inventory_rebalancer_tool.py`:
- Resolved definition/source of `perf_gap`.
- Resolved strategy for the inf/NaN engineered-column cleaning gap (see
  above), and why.
- Confirmation that `rebalancer_feature_columns.pkl` exists and matches the
  13-feature list in this spec — if it differs, use the artifact's list and
  flag the discrepancy rather than silently trusting either source blindly.
- Whether `manufacture_rank` (global) is implemented or omitted for this
  iteration, and what would be needed to add it later.

## Acceptance criteria
- [ ] `inventory_rebalancer_tool.py` has no imports from `langchain`,
      `langgraph`, or any LLM client library, and does not import or load
      `stacking_proposed.pkl`.
- [ ] Docstring documents the `perf_gap` resolution and the inf/NaN
      cleaning-gap resolution.
- [ ] Artifacts load exactly once per process (cached).
- [ ] `rebalance_inventory()` runs on a small hand-built sample (3–5 rows)
      covering: a normal row, a row with `-99.0` in perf averages, a row
      with NaN `lead_time`, a row with negative `national_inv`, and a row
      where `min_bank > national_inv` (should produce `recommended_qty > 0`).
- [ ] Output matches `InventoryRebalancerOutput` exactly — validate with
      Pydantic, not just visually.
- [ ] `recommended_qty` is never negative (clipped at 0) and
      `urgency_score_predicted` is within [0, 1] (or documented if the model
      can exceed that range).
- [ ] `top_priority_skus` is a derived view consistent with `recommendations`.
- [ ] Unit tests cover: normal input, missing SKU, missing
      `backorder_probability` entry for a requested SKU, malformed input
      (missing required raw columns), and determinism.
- [ ] No metrics are computed anywhere in this file.
- [ ] Module runs on both CPU-only and CUDA-available machines without code
      changes.

## Out of scope / do not do
- Do not build the LangGraph node or agent prompt for the Inventory
  Rebalancer — Phase 3/4, separate task.
- Do not touch the DuckDB data-access layer.
- Do not call the Risk Detector's model or load `stacking_proposed.pkl`
  from within this module.
- Do not retrain or regenerate any model artifact.
- Do not implement `manufacture_rank` as a global rank without an actual
  reference distribution available — omit rather than fake it.
