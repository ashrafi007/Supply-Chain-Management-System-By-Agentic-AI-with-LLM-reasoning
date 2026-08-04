# Spec: Risk Detector Inference Tool

## Objective
Extract the trained Risk Detector model(s) from the existing `classification`
notebook into a standalone, importable Python module that exposes a single
deterministic inference function. This module will later be registered as a
tool inside a LangGraph multi-agent pipeline, so it must have **zero LangChain
/ LangGraph / LLM dependencies** — it is pure model inference.

This is Phase 2 of a larger roadmap (`RSM_Agentic_Roadmap.pdf`). Do not touch
Phase 3+ concerns (agent reasoning, prompts, orchestration) in this task.

## Context (read before starting)
- Source notebook: the ported `classification` notebook (Windows + RTX 4070,
  originally built on Mac M2). Locate it in the repo and treat it as the
  reference implementation — do not retrain, only extract inference logic.
- Known model details from the notebook:
  - XGBoost and CatBoost are routed to CUDA with CPU fallback via
    `torch.cuda.is_available()` for cross-platform GPU detection.
  - LightGBM is CPU-only (the pip wheel used lacks GPU support) — do not
    attempt to route it to CUDA.
  - Logistic Regression is wrapped in
    `make_pipeline(SimpleImputer, StandardScaler, LogisticRegression)` to
    handle NaNs — preserve this pipeline wrapping exactly, do not simplify it.
  - If SHAP explanation output is needed later, note that the beeswarm plot
    required forcing the booster to CPU via
    `model.get_booster().set_param({'device': 'cpu'})` — SHAP is NOT required
    for this task, just flagging it so it isn't silently broken.
- The model predicts `went_on_backorder` (binary classification).
- Target is imbalanced — do not use plain accuracy anywhere in this module;
  no metrics computation belongs in this file at all (metrics were already
  computed in the training notebook).
- Known dataset quirks to respect if this module does its own feature prep:
  - `lead_time` may contain NaN.
  - `perf_6_month_avg` / `perf_12_month_avg` use `-99.0` as a missing-value
    sentinel — must be treated as missing, not as a real value, before it
    reaches the model.
  - `national_inv` can be legitimately negative — do not clip or "fix" it.

## Non-goals
- No training code, no hyperparameter tuning, no plotting, no SHAP.
- No LangChain/LangGraph tool decorators — that wrapping happens in a later
  phase, in a different file.
- No changes to the DuckDB data-access layer — this module receives already-
  fetched rows (as a DataFrame) and returns predictions; it does not query
  the database itself.
- Do not modify the original `classification` notebook.

## Deliverable: file structure
Create the following, matching the team's existing `inference_tools/`
package layout:

```
inference_tools/
├── schemas.py                  # add to this shared file, do not fork it
└── risk_detector_tool.py       # new file, this task's main deliverable
tests/
└── test_risk_detector_tool.py  # new file
```

If `inference_tools/schemas.py` does not exist yet, create it — later model
wrappers (demand predictor, inventory rebalancer, supplier auditor) will
also add their schemas here.

## Schema contract (`schemas.py`)
Define with Pydantic (v2 style):

```python
class RiskDetectorInput(BaseModel):
    skus: list[str]                 # SKUs to score
    data: dict                      # or a DataFrame-serializable structure —
                                     # decide the cleanest interchange format
                                     # and document the choice in a docstring

class RiskDetectorPrediction(BaseModel):
    sku: str
    backorder_probability: float    # 0.0–1.0
    predicted_label: bool           # thresholded prediction
    is_high_risk: bool              # flagged per the chosen risk threshold

class RiskDetectorOutput(BaseModel):
    predictions: list[RiskDetectorPrediction]
    high_risk_skus: list[str]       # convenience subset, derived from predictions
    model_version: str              # e.g. "xgboost_v2" or "stacking_ensemble_v1"
    threshold_used: float
```

Adjust field names only if the notebook's actual output shape requires it —
but keep the principle: **output must be structured and interpretable, never
a raw array or raw model object.**

## Function contract (`risk_detector_tool.py`)

```python
def load_risk_detector_model(model_path: str | None = None):
    """
    Load the trained model from disk. Must be cached (module-level singleton
    or functools.lru_cache) so repeated calls do not reload from disk.
    Auto-detect GPU availability the same way the source notebook does
    (torch.cuda.is_available()), with CPU fallback. Respect the per-model
    GPU rules above (LightGBM stays CPU-only if it's part of the final model).
    """

def predict_risk(input_data: RiskDetectorInput) -> RiskDetectorOutput:
    """
    Pure inference. Given SKUs + their feature rows:
      1. Apply the same missing-value handling used in training
         (-99.0 sentinel -> NaN, lead_time NaN handling matching the
         notebook's approach).
      2. Run the loaded model's .predict_proba() (or equivalent).
      3. Apply the risk threshold (make this a named constant or config
         value, not a magic number buried in logic).
      4. Return RiskDetectorOutput as defined above.
    Must be deterministic: same input -> same output, every time.
    Must raise a clear, typed exception (not a bare Exception) if:
      - a requested SKU has no matching row in the input data
      - the model file cannot be loaded
      - required feature columns are missing from the input
    """
```

## Determine and document during implementation
These are things Claude Code should figure out by reading the notebook, and
should state explicitly at the top of `risk_detector_tool.py` as a docstring:
- Which trained model is the final one to wrap: single model (XGBoost?) or
  the stacking ensemble? If the notebook has multiple candidate models,
  ask before picking one — do not guess.
- Exact list and order of feature columns the model expects.
- The risk threshold value used to set `is_high_risk` — pull this from the
  notebook if one was already chosen; otherwise flag it as needing a
  decision rather than inventing one.
- File path/format the trained model was saved in (`.json`, `.cbm`, `.pkl`,
  `.txt`) and load it with the matching library.

## Acceptance criteria
- [ ] `risk_detector_tool.py` has no imports from `langchain`, `langgraph`,
      or any LLM client library.
- [ ] Model loads exactly once per process (verify with a print/log
      statement or a test asserting the loader is cached).
- [ ] `predict_risk()` runs successfully on a small hand-built sample of
      3–5 rows covering: a normal row, a row with `-99.0` in perf averages,
      a row with NaN `lead_time`, a row with negative `national_inv`.
- [ ] Output matches the `RiskDetectorOutput` schema exactly — validate with
      Pydantic, not just visually.
- [ ] `high_risk_skus` is consistent with `predictions` (i.e., it's a
      derived view, not independently computed / able to drift out of sync).
- [ ] Unit tests in `tests/test_risk_detector_tool.py` cover: normal input,
      missing SKU, malformed input (missing required columns), and confirm
      determinism (same input called twice returns identical output).
- [ ] No metrics (ROC-AUC, F2, etc.) are computed anywhere in this file —
      that belongs to the training notebook / evaluation phase, not here.
- [ ] Module runs on both CPU-only and CUDA-available machines without code
      changes (GPU detection is automatic, not hardcoded).

## Out of scope / do not do
- Do not build the LangGraph node or agent prompt for the Risk Detector —
  that's Phase 3/4, a separate task.
- Do not touch the DuckDB data-access layer.
- Do not retrain or re-tune the model.
- Do not add SHAP/explainability output unless explicitly asked in a
  follow-up task.
