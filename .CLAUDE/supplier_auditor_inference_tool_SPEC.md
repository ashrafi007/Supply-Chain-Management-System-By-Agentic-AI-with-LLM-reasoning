# Spec: Supplier Auditor Inference Tool

## Objective
Extract the trained Supplier Auditor model from
`supplier_auditor_model_train.ipynb` into a standalone, importable Python
module exposing a single deterministic inference function. This module will
later be registered as a tool inside a LangGraph multi-agent pipeline, so it
must have **zero LangChain / LangGraph / LLM dependencies** — pure model
inference only.

This is Phase 2 of the larger roadmap (`RSM_Agentic_Roadmap.pdf`). Do not
touch Phase 3+ concerns (agent reasoning, prompts, orchestration) here.

## Predictor vs. reasoner separation (non-negotiable)
This module contains only feature engineering, preprocessing, model
inference, and the grading/flagging rules defined in the notebook. No
`@tool` decorators, no LangChain/LangGraph imports, no prompts. A later,
separate file (Phase 3) will thinly wrap this module's function with
LangChain's tool interface.

## Ground truth from the notebook (read this before writing any code)

**Good news first: this notebook is self-contained.** Unlike the Inventory
Rebalancer and Forecast Optimizer notebooks, this one does **not** depend on
loading another agent's model (no `stacking_proposed.pkl` coupling here).
It also saves **one single artifact file** containing everything needed for
inference — this is the easiest of the four models to wrap correctly.

**Model:** `XGBClassifier`, target = `stop_auto_buy` (binary). Trained with
`device = "cuda" if torch.cuda.is_available() else "cpu"`, `tree_method =
"hist"` — reuse this exact detection pattern.

**Single saved artifact — `models/supplier_auditor_model.pkl`** — is a
pickled Python dict with exactly these four keys:
```python
{
    "model":         <trained XGBClassifier>,
    "preprocessor":  <fitted sklearn Pipeline: SimpleImputer(median) -> RobustScaler>,
    "threshold":     <float, OPT_THRESH>,   # decision threshold selected by
                                             # maximizing MCC on the validation set
    "feature_cols":  <list[str], FEATURE_COLS>,  # exact column order the
                                             # model and preprocessor expect
}
```
Load this single file — there is no separate imputer/scaler artifact to
hunt for, and no ambiguity about which model was "selected" (there was only
ever one candidate trained and evaluated as the proposed model; baselines
in Section 11 are for comparison in the paper only, not alternate
candidates to choose between).

**Preprocessing — reuse the saved `Pipeline` object as-is.** It was fit on
the training split only (median imputation, then `RobustScaler`). Call
`preprocessor.transform(X)` — never refit it, never rebuild it from scratch.

## Feature engineering (`engineer_features()`, must run exactly as written)

```python
EPS = 1e-6   # note: this is a smaller epsilon than the 0.001/1e-5 used in
             # the Inventory Rebalancer / Forecast Optimizer notebooks —
             # do NOT reuse those notebooks' epsilon values here, this
             # model was trained with 1e-6 specifically.

perf_trend              = perf_6_month_avg - perf_12_month_avg
perf_decay_rate          = perf_trend / (perf_12_month_avg + EPS)
delivery_stress          = pieces_past_due / (national_inv + EPS)
backlog_transit_ratio    = local_bo_qty / (in_transit_qty + EPS)
forecast_deviation_3m    = forecast_3_month - sales_3_month
forecast_deviation_6m    = forecast_6_month - sales_6_month
forecast_accuracy_3m     = sales_3_month / (forecast_3_month + EPS)
daily_demand             = forecast_3_month / 90.0
inventory_coverage_days  = national_inv / (daily_demand + EPS)
transit_exposure_ratio   = in_transit_qty / (national_inv + EPS)

risk_flags               = [potential_issue, deck_risk, oe_constraint,
                             ppap_risk, rev_stop]
risk_flag_count           = sum(risk_flags, axis=1)
compound_risk_index       = (1 - perf_6_month_avg) * (risk_flag_count + 1)
```

⚠️ **Verify before implementing:** `risk_flag_count` sums the five
`risk_flags` columns directly (`df[risk_flags].sum(axis=1)`), which only
works if those columns are already numeric (0/1) in `Processed_Dataset.pkl`.
The raw RSM dataset has these as Yes/No object columns
(`potential_issue`, `deck_risk`, `oe_constraint`, `ppap_risk`, `rev_stop`
are all listed as object dtype in the known dataset schema). **Confirm
`Processed_Dataset.pkl` has already encoded these as 0/1 before this tool
assumes it can sum them directly** — if not, this tool must encode them
(Yes→1/No→0, matching whatever encoding the notebook's preprocessing
actually used) before calling `engineer_features()`, and that encoding step
must be documented, not guessed.

No inf/NaN cleanup step appears in `engineer_features()` itself — division
guards (`+ EPS`) are the only protection against divide-by-zero in this
notebook. If a raw input value is `NaN` going in, it will still be `NaN`
coming out of these formulas; the median imputation happens **downstream**
in the `preprocessor` pipeline (`SimpleImputer(strategy="median")`, fit on
train), which is fine — do not add an extra ad hoc NaN-cleaning step of
your own on top of it.

## Feature selection
```python
DROP_COLS    = ["sku", "stop_auto_buy"]
FEATURE_COLS = every column of engineer_features(df) output EXCEPT DROP_COLS
```
Use the `feature_cols` list from the loaded artifact as authoritative (it
should match this, but the artifact is the source of truth — if it
diverges, use the artifact's list and flag the discrepancy).

⚠️ Note: unlike the Inventory Rebalancer notebook (where `sku` was the
DataFrame index), **this notebook expects `sku` as a regular column** in
the input data. Confirm which convention your DuckDB data-access layer
actually returns and adapt accordingly — do not assume it matches the other
tools' convention.

## Inference logic — replicate the notebook's audit-report generation exactly

The notebook doesn't have a single named `predict()` function the way the
Forecast Optimizer notebook does — the equivalent logic lives in the
"Supplier Grade Engine" section (Section 17) and must be extracted into a
proper function. Replicate it exactly:

```python
# 1. Get raw model probability
risk_probability = model.predict_proba(preprocessor.transform(X))[:, 1]
risk_probability = round(risk_probability, 4)

# 2. Assign a letter grade from probability bands
GRADE_THRESHOLDS = {
    "A": (0.00, 0.20),
    "B": (0.20, 0.45),
    "C": (0.45, 0.70),
    "D": (0.70, 1.01),   # note the upper bound is 1.01, not 1.00 —
                          # intentional, so a probability of exactly 1.0
                          # still falls in D. Preserve this exact bound.
}
# assign_grade(p): first matching [lo, hi) band, else "D" as fallback

# 3. Combine model decision with a rule-based override
model_flag = risk_probability >= OPT_THRESH
rule_flag  = (supplier_grade == "D") and (delivery_stress > 0.5)
triggered  = model_flag or rule_flag
reason     = "MODEL+RULE" if (model_flag and rule_flag) else \
             "MODEL"      if model_flag else \
             "RULE"       if rule_flag  else "NONE"
```

Preserve the exact grade boundaries, the `0.5` delivery-stress rule
threshold, and the four-way `trigger_reason` categorization — these encode
real business rules, not arbitrary constants. Do not simplify the
model-flag-OR-rule-flag logic to just the model flag.

## Non-goals
- No training code, no grid search, no plotting, no SHAP, no ablation study,
  no McNemar's test, no cross-validation.
- No LangChain/LangGraph tool decorators.
- No changes to the DuckDB data-access layer.
- Do not modify the original notebook.
- Do not silently assume the risk-flag columns are pre-encoded — verify.

## Deliverable: file structure
```
inference_tools/
├── schemas.py                    # add to this shared file, do not fork it
└── supplier_auditor_tool.py      # new file, this task's main deliverable
tests/
└── test_supplier_auditor_tool.py # new file
```

## Schema contract (`schemas.py`)

```python
class SupplierAuditorInput(BaseModel):
    skus: list[str]
    data: dict
        # raw feature rows for these SKUs — must include every raw column
        # needed for engineer_features(): perf_6_month_avg,
        # perf_12_month_avg, pieces_past_due, national_inv, local_bo_qty,
        # in_transit_qty, forecast_3_month, sales_3_month, forecast_6_month,
        # sales_6_month, potential_issue, deck_risk, oe_constraint,
        # ppap_risk, rev_stop. Document exact expected shape (including
        # whether risk-flag columns arrive pre-encoded as 0/1) in a
        # docstring.

class SupplierAuditResult(BaseModel):
    sku: str
    risk_probability: float        # 0.0-1.0, rounded to 4 decimals
    supplier_grade: str            # "A" | "B" | "C" | "D"
    stop_auto_buy_triggered: bool
    trigger_reason: str            # "MODEL+RULE" | "MODEL" | "RULE" | "NONE"
    delivery_stress: float         # exposed since it drives the rule flag

class SupplierAuditorOutput(BaseModel):
    results: list[SupplierAuditResult]   # sorted by risk_probability, descending
    flagged_suppliers: list[str]         # SKUs where stop_auto_buy_triggered
                                          # is True — derived, not independently
                                          # computed
    grade_distribution: dict[str, int]   # count per grade, e.g. {"A": 12, "B": 4, ...}
    model_version: str                   # e.g. "xgboost_supplier_auditor_v1"
    threshold_used: float                # OPT_THRESH from the loaded artifact
```

## Function contract (`supplier_auditor_tool.py`)

```python
def load_supplier_auditor_artifact(model_path: str | None = None):
    """
    Load the single models/supplier_auditor_model.pkl dict (model,
    preprocessor, threshold, feature_cols). Cache (module-level singleton
    or functools.lru_cache) so repeated calls do not reload from disk.
    Raise a clear, typed exception if the file is missing or any of the
    four expected keys is absent.
    """

def audit_suppliers(input_data: SupplierAuditorInput) -> SupplierAuditorOutput:
    """
    Pure inference. Given SKUs + raw feature rows:
      1. Confirm/encode risk-flag columns as 0/1 per the resolved decision
         above.
      2. Run engineer_features() exactly as in the notebook (EPS = 1e-6).
      3. Select feature_cols (from the loaded artifact) in the correct order.
      4. Apply preprocessor.transform() (never refit).
      5. Run model.predict_proba() to get risk_probability.
      6. Assign supplier_grade via the GRADE_THRESHOLDS bands.
      7. Apply the model-flag / rule-flag combination logic to get
         stop_auto_buy_triggered and trigger_reason.
      8. Sort by risk_probability descending and return SupplierAuditorOutput.
    Must be deterministic: same input -> same output, every time.
    Must raise a clear, typed exception (not a bare Exception) if:
      - a requested SKU has no matching row in the input data
      - required raw columns are missing
      - the loaded artifact is malformed/incompatible
    """
```

## Determine and document during implementation
State these explicitly in a docstring at the top of
`supplier_auditor_tool.py`:
- Confirmation of whether `potential_issue`, `deck_risk`, `oe_constraint`,
  `ppap_risk`, `rev_stop` arrive as 0/1 numeric in the actual data source,
  or need encoding — and if encoding is needed, exactly what mapping was
  used (verify against how `Processed_Dataset.pkl` was actually built,
  don't assume Yes→1/No→0 without checking).
- Confirmation that `sku` is a column (not an index) in the data this tool
  receives, and how that's reconciled with the DuckDB data-access layer's
  actual output shape.

## Acceptance criteria
- [ ] `supplier_auditor_tool.py` has no imports from `langchain`,
      `langgraph`, or any LLM client library.
- [ ] Loads all four keys from the single artifact file; no separate
      imputer/scaler/threshold files are hunted for.
- [ ] Artifact loads exactly once per process (cached).
- [ ] `audit_suppliers()` runs on a small hand-built sample (3–5 rows)
      covering: a normal row, a row with `-99.0` in perf averages, a row
      with NaN `lead_time`-equivalent fields, a row engineered to land in
      each grade band (at least verify A and D are reachable), and a row
      where `rule_flag` fires independently of `model_flag` (grade D +
      delivery_stress > 0.5, but probability below threshold) to confirm
      `trigger_reason == "RULE"` works.
- [ ] Output matches `SupplierAuditorOutput` exactly — validate with
      Pydantic, not just visually.
- [ ] `flagged_suppliers` and `grade_distribution` are derived views
      consistent with `results`, not independently computed.
- [ ] The `D` grade band's upper bound of `1.01` (not `1.00`) is preserved
      exactly, and a probability of exactly `1.0` is verified to land in D.
- [ ] Unit tests cover: normal input, missing SKU, malformed input (missing
      required columns), the MODEL/RULE/MODEL+RULE/NONE trigger-reason
      branches, and determinism.
- [ ] No metrics are computed anywhere in this file.
- [ ] Module runs on both CPU-only and CUDA-available machines without code
      changes.

## Out of scope / do not do
- Do not build the LangGraph node or agent prompt for the Supplier Auditor
  — Phase 3/4, separate task.
- Do not touch the DuckDB data-access layer.
- Do not retrain or regenerate the model artifact.
- Do not silently assume the risk-flag encoding — verify it.
- Do not replace the `RobustScaler`/`SimpleImputer` pipeline with a
  reimplementation — always use the saved, fitted `preprocessor` object.
