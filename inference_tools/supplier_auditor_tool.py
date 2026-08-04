"""
Supplier Auditor Inference Tool — Pure model inference, no LangChain/LangGraph/LLM dependencies.

**Final model:** XGBClassifier, target = stop_auto_buy (binary).
  Trained with device = "cuda" if torch.cuda.is_available() else "cpu", tree_method = "hist"
  (Notebooks/supplier_auditor_model_train.ipynb Cell 3/13). This module reuses the same
  torch-based GPU-detection pattern and defensively forces the booster to CPU at load time
  when no GPU is present, matching inventory_rebalancer_tool.py / forecast_optimizer_tool.py.

**Single saved artifact:** models/supplier_auditor_model.pkl (repo path:
  Models/supply_auditor/supplier_auditor_model.pkl), a pickle.dump'd dict (not joblib) with
  exactly four keys: model, preprocessor, threshold, feature_cols. Confirmed by direct
  pickle.load in this repo — no separate imputer/scaler/threshold artifacts exist for this
  model, unlike the Inventory Rebalancer / Forecast Optimizer tools. Loaded here via
  pickle.load (not joblib.load) to match how it was written (Notebook Cell 37:
  pickle.dump(..., protocol=pickle.HIGHEST_PROTOCOL)).

**Risk-flag encoding — resolved, no encoding needed (required item from spec):**
  The spec warned that potential_issue/deck_risk/oe_constraint/ppap_risk/rev_stop might
  arrive as Yes/No object columns in the "raw RSM dataset" and require encoding before
  risk_flag_count = df[risk_flags].sum(axis=1) would work. Confirmed false for this
  pipeline's actual input: Notebook Cell 5's captured df_raw.head() output shows deck_risk,
  oe_constraint, ppap_risk, rev_stop, stop_auto_buy, went_on_backorder all as bare 0/1
  integers already in Processed_Dataset.pkl (the object-dtype Yes/No schema describes some
  other, unprocessed raw dataset, not this one). This is corroborated by
  Risk_Detector_Classification_Model.ipynb Cell 16, which lists potential_issue, deck_risk,
  oe_constraint, ppap_risk, stop_auto_buy, rev_stop, went_on_backorder as binary_cols and
  fillna(0)'s them directly as numerics — no .map({'Yes': 1, 'No': 0}) or get_dummies call
  appears anywhere across any of the four training notebooks in this repo. This module
  therefore performs NO Yes/No -> 0/1 mapping; risk flags are required to already be 0/1
  numeric in SupplierAuditorInput.data, matching the trained model's actual input contract.

**`sku` column vs. index — resolved, contradicts this spec file's own claim:**
  The spec asserted "this notebook expects sku as a regular column" (unlike the Inventory
  Rebalancer notebook, where sku is the index). Directly inspecting Notebook Cell 5's
  captured df_raw.head() output shows `sku` rendered as the DataFrame's index (e.g. row
  labels 3285085, 3285131, ...) and absent from the printed `df_raw.columns` list (26
  entries, none named "sku"). So sku IS the index here too — the same convention as the
  Inventory Rebalancer notebook, not what this spec assumed. (Separately, Notebook Cell 35's
  audit-report code does `audit.reset_index(drop=True)` and then synthesizes fake
  "SUP-00000"-style SKUs when "sku" is absent from columns — i.e. the notebook's own report
  generation silently discards the real sku identity. That is notebook-side report-building
  behavior, not something this inference module replicates.) Consequently,
  SupplierAuditorInput.data is keyed by sku exactly like RiskDetectorInput /
  InventoryRebalancerInput / ForecastOptimizerInput (dict-of-dicts, sku is the dict key,
  never an expected raw column in the per-SKU feature dict).

**FEATURE_COLS scope wider than the spec's schema-contract docstring (verify-before-coding
  finding):** The spec's SupplierAuditorInput docstring lists ~15 raw columns (only the ones
  engineer_features() reads directly). Directly loading the artifact shows feature_cols has
  36 entries: all 25 raw columns of Processed_Dataset.pkl's schema except stop_auto_buy
  (including went_on_backorder, lead_time, forecast_9_month, sales_1_month, sales_9_month,
  min_bank, inv_velocity, safety_gap, sales_trend, perf_gap — none of which engineer_features()
  touches, but all of which are required raw passthrough columns the model was trained on)
  plus the 11 engineered columns. REQUIRED_RAW_COLS below reflects the full 25, not the
  spec's narrower ~15-column list.

**Preprocessing:** the saved Pipeline (SimpleImputer(median) -> RobustScaler), fit on the
  training split only, is reused as-is via preprocessor.transform(X) — never refit.

**Grade bands / rule override:** GRADE_THRESHOLDS, assign_grade(), and the
  model-flag-OR-rule-flag combination logic (Notebook Cell 35, Section 17 "Supplier Grade
  Engine") are reproduced exactly, including the D band's [0.70, 1.01) upper bound (so a
  probability of exactly 1.0 still lands in D) and the delivery_stress > 0.5 rule threshold.

**Predictor/reasoner separation:**
  This module contains only feature engineering, preprocessing, model inference, and the
  grading/flagging rules defined in the notebook. No @tool decorators, no LangChain/
  LangGraph imports, no prompts, no model chaining to any other agent's artifact (this
  notebook is self-contained — no stacking_proposed.pkl coupling).

**Non-goals:**
  No LangChain/LangGraph imports or @tool decorators. No training, grid search, plotting,
  SHAP, ablation, McNemar's test, cross-validation. No changes to any DuckDB data-access
  layer. No retrain/regeneration of the model artifact. No modification of the original
  notebook.
"""

from __future__ import annotations

import functools
import pickle
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import (
    SupplierAuditorError,
    SupplierAuditorInput,
    SupplierAuditorOutput,
    SupplierAuditResult,
    SupplierAuditorModelLoadError,
    SupplierAuditorSkuNotFoundError,
    SupplierAuditorMissingFeatureColumnsError,
)

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants (documentation/comparison; loaded artifact is runtime truth)
# ──────────────────────────────────────────────────────────────────────────

# All 25 raw columns of Processed_Dataset.pkl's 26-column schema needed by this pipeline
# (every raw column except `stop_auto_buy`, the target). `sku` is the DataFrame's index in
# Processed_Dataset.pkl, never a raw column — see module docstring — so it is not listed
# here; it is the SupplierAuditorInput.data dict key instead.
REQUIRED_RAW_COLS: list[str] = [
    'national_inv', 'lead_time', 'in_transit_qty',
    'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
    'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
    'min_bank', 'potential_issue', 'pieces_past_due',
    'perf_6_month_avg', 'perf_12_month_avg',
    'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk',
    'rev_stop', 'went_on_backorder', 'inv_velocity', 'safety_gap',
    'sales_trend', 'perf_gap',
]

# The five 0/1 columns summed into risk_flag_count. Confirmed already numeric in
# Processed_Dataset.pkl (see module docstring) — no encoding applied here.
RISK_FLAG_COLS: list[str] = [
    'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'rev_stop',
]

# The 11 columns computed by engineer_features(), reproduced from
# Notebooks/supplier_auditor_model_train.ipynb Cell 9.
ENGINEERED_COLS: list[str] = [
    'perf_trend', 'perf_decay_rate', 'delivery_stress', 'backlog_transit_ratio',
    'forecast_deviation_3m', 'forecast_deviation_6m', 'forecast_accuracy_3m',
    'inventory_coverage_days', 'transit_exposure_ratio',
    'risk_flag_count', 'compound_risk_index',
]

# Documentation constant: the real 36-item feature_cols contents of
# models/supplier_auditor_model.pkl, confirmed via direct pickle.load. The loaded artifact
# is the actual runtime source of truth (see load_supplier_auditor_artifact()); this
# constant exists for comparison/documentation.
FEATURE_COLS: list[str] = REQUIRED_RAW_COLS + ENGINEERED_COLS

EPS = 1e-6   # Notebook Cell 9's epsilon. Smaller than the 0.001/1e-5 used by the Inventory
             # Rebalancer / Forecast Optimizer notebooks — intentionally not reused here.

# Grade probability bands — Notebook Cell 35. Note the D band's upper bound is 1.01, not
# 1.00: intentional, so a probability of exactly 1.0 still falls in D. Preserve exactly.
GRADE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "A": (0.00, 0.20),
    "B": (0.20, 0.45),
    "C": (0.45, 0.70),
    "D": (0.70, 1.01),
}

DEFAULT_MODEL_VERSION = "xgboost_supplier_auditor_v2"


@dataclass(frozen=True)
class LoadedSupplierAuditorModel:
    """Immutable container for the loaded supplier auditor model + metadata."""
    model: Any
    preprocessor: Any
    threshold: float
    feature_cols: list[str]
    model_version: str


def _detect_gpu_available() -> bool:
    """
    Auto-detect GPU via torch, with nvidia-smi subprocess fallback.
    Mirrors the notebook's device = "cuda" if torch.cuda.is_available() else "cpu" pattern
    (Cell 3), extended with the same nvidia-smi fallback used by the sibling tools.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        pass

    if not shutil.which('nvidia-smi'):
        return False

    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        return bool(out)
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def load_supplier_auditor_artifact(model_path: str | None = None) -> LoadedSupplierAuditorModel:
    """
    Load the single models/supplier_auditor_model.pkl dict (model, preprocessor, threshold,
    feature_cols). Cached (module-level lru_cache singleton per process) so repeated calls
    do not reload from disk.

    Args:
        model_path: Optional absolute path to the artifact file.
                    If None, defaults to <repo_root>/Models/supply_auditor/supplier_auditor_model.pkl.

    Returns:
        LoadedSupplierAuditorModel(model, preprocessor, threshold, feature_cols, model_version).

    Raises:
        SupplierAuditorModelLoadError: If the file is missing, unreadable, or any of the
            four expected keys (model, preprocessor, threshold, feature_cols) is absent
            or malformed.
    """
    if model_path is None:
        repo_root = Path(__file__).parent.parent
        model_path = repo_root / 'Models' / 'supply_auditor' / 'supplier_auditor_model.pkl'
    else:
        model_path = Path(model_path)

    try:
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)
    except Exception as e:
        raise SupplierAuditorModelLoadError(
            f"Failed to load supplier auditor artifact from {model_path}: {e}"
        ) from e

    if not isinstance(artifact, dict):
        raise SupplierAuditorModelLoadError(
            f"Expected {model_path} to contain a dict, got {type(artifact)}"
        )

    required_keys = {'model', 'preprocessor', 'threshold', 'feature_cols'}
    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        raise SupplierAuditorModelLoadError(
            f"{model_path} is missing required key(s): {missing_keys}"
        )

    model = artifact['model']
    preprocessor = artifact['preprocessor']
    threshold = artifact['threshold']
    feature_cols = artifact['feature_cols']

    if not hasattr(model, 'predict_proba'):
        raise SupplierAuditorModelLoadError(
            f"{model_path}['model'] does not implement predict_proba: {type(model)}"
        )
    if not hasattr(preprocessor, 'transform'):
        raise SupplierAuditorModelLoadError(
            f"{model_path}['preprocessor'] does not implement transform: {type(preprocessor)}"
        )
    if not isinstance(feature_cols, list):
        raise SupplierAuditorModelLoadError(
            f"Expected {model_path}['feature_cols'] to contain a list, got {type(feature_cols)}"
        )
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as e:
        raise SupplierAuditorModelLoadError(
            f"Expected {model_path}['threshold'] to be a float, got {threshold!r}"
        ) from e

    if feature_cols != FEATURE_COLS:
        warnings.warn(
            f"Loaded supplier_auditor_model.pkl's feature_cols differs from the "
            f"FEATURE_COLS module constant: artifact={feature_cols!r} vs "
            f"constant={FEATURE_COLS!r}. Using the artifact's list.",
            stacklevel=2
        )

    # Defensive fix for XGBoost GPU-trained boosters: force CPU if no GPU present.
    gpu_available = _detect_gpu_available()
    if not gpu_available:
        try:
            model.get_booster().set_param({'device': 'cpu'})
        except Exception:
            pass  # Silently skip if not applicable.

    return LoadedSupplierAuditorModel(
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
        feature_cols=feature_cols,
        model_version=DEFAULT_MODEL_VERSION,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate Notebooks/supplier_auditor_model_train.ipynb Cell 9's engineer_features()
    exactly, EPS = 1e-6. Assumes RISK_FLAG_COLS already arrive as 0/1 numeric (see module
    docstring) — no encoding performed here.

    No inf/NaN cleanup step here, matching the notebook: division guards (+ EPS) are the
    only protection against divide-by-zero. NaN inputs propagate through as NaN; median
    imputation happens downstream in the loaded preprocessor Pipeline.
    """
    df = df.copy()

    df["perf_trend"] = df["perf_6_month_avg"] - df["perf_12_month_avg"]
    df["perf_decay_rate"] = df["perf_trend"] / (df["perf_12_month_avg"] + EPS)
    df["delivery_stress"] = df["pieces_past_due"] / (df["national_inv"] + EPS)
    df["backlog_transit_ratio"] = df["local_bo_qty"] / (df["in_transit_qty"] + EPS)
    df["forecast_deviation_3m"] = df["forecast_3_month"] - df["sales_3_month"]
    df["forecast_deviation_6m"] = df["forecast_6_month"] - df["sales_6_month"]
    df["forecast_accuracy_3m"] = df["sales_3_month"] / (df["forecast_3_month"] + EPS)
    daily_demand = df["forecast_3_month"] / 90.0
    df["inventory_coverage_days"] = df["national_inv"] / (daily_demand + EPS)
    df["transit_exposure_ratio"] = df["in_transit_qty"] / (df["national_inv"] + EPS)
    df["risk_flag_count"] = df[RISK_FLAG_COLS].sum(axis=1)
    df["compound_risk_index"] = (1 - df["perf_6_month_avg"]) * (df["risk_flag_count"] + 1)

    return df


def assign_grade(p: float) -> str:
    """
    Notebook Cell 35's assign_grade(), verbatim. First matching [lo, hi) band in
    GRADE_THRESHOLDS, else "D" as fallback. The D band's 1.01 upper bound (not 1.00) is
    intentional — a probability of exactly 1.0 must still land in D.
    """
    for grade, (lo, hi) in GRADE_THRESHOLDS.items():
        if lo <= p < hi:
            return grade
    return "D"


def combine_trigger_flags(
    risk_probability: float,
    supplier_grade: str,
    delivery_stress: float,
    threshold: float,
) -> tuple[bool, str]:
    """
    Notebook Cell 35's flag_logic(), verbatim: combine the model decision with the
    rule-based override. Kept as a standalone pure function (rather than inlined in
    audit_suppliers()) so the RULE-only branch — unreachable via the real loaded model's
    threshold, since GRADE_THRESHOLDS["D"] starts at 0.70 which is always >= the actual
    trained OPT_THRESH (v2: ~0.36; v1 was ~0.08), making rule_flag-without-model_flag
    structurally impossible at the real threshold — can still be exercised directly in
    tests with a synthetic threshold.

    Returns:
        (triggered, reason) where reason is one of
        "MODEL+RULE" | "MODEL" | "RULE" | "NONE".
    """
    model_flag = risk_probability >= threshold
    rule_flag = (supplier_grade == "D") and (delivery_stress > 0.5)
    triggered = model_flag or rule_flag
    reason = (
        "MODEL+RULE" if (model_flag and rule_flag) else
        "MODEL" if model_flag else
        "RULE" if rule_flag else
        "NONE"
    )
    return triggered, reason


def audit_suppliers(input_data: SupplierAuditorInput) -> SupplierAuditorOutput:
    """
    Pure inference. Given SKUs + raw feature rows:
      1. Risk-flag columns are assumed already 0/1 numeric (confirmed, see module
         docstring) — no encoding step is performed.
      2. Run engineer_features() exactly as in the notebook (EPS = 1e-6).
      3. Select feature_cols (from the loaded artifact) in the correct order.
      4. Apply preprocessor.transform() (never refit).
      5. Run model.predict_proba() to get risk_probability, rounded to 4 decimals.
      6. Assign supplier_grade via assign_grade() / GRADE_THRESHOLDS.
      7. Apply combine_trigger_flags() to get stop_auto_buy_triggered and trigger_reason.
      8. Sort by risk_probability descending and return SupplierAuditorOutput.
    Deterministic: same input -> same output, every time.

    Raises:
        SupplierAuditorSkuNotFoundError: a requested SKU has no matching row in input data
            (defensive re-check; Pydantic's validate_sku_coverage already guarantees this
            at construction time).
        SupplierAuditorMissingFeatureColumnsError: required raw columns are missing, or
            feature engineering failed to produce a column required by feature_cols.
        SupplierAuditorModelLoadError: the loaded artifact is malformed/incompatible.
    """
    loaded = load_supplier_auditor_artifact()

    # Defensive re-check (belt-and-suspenders; Pydantic's validate_sku_coverage already
    # guarantees this at construction time).
    missing_skus = set(input_data.skus) - set(input_data.data.keys())
    if missing_skus:
        raise SupplierAuditorSkuNotFoundError(
            f"Requested SKUs not found in data dict: {missing_skus}"
        )

    rows = []
    for sku in input_data.skus:
        row = input_data.data[sku]
        missing_cols = set(REQUIRED_RAW_COLS) - set(row.keys())
        if missing_cols:
            raise SupplierAuditorMissingFeatureColumnsError(
                f"SKU {sku}: missing required raw columns {missing_cols}"
            )
        rows.append(row)

    df = pd.DataFrame(rows, index=list(input_data.skus))

    df_eng = engineer_features(df)

    try:
        X = df_eng[loaded.feature_cols]
    except KeyError as e:
        missing_engineered = set(loaded.feature_cols) - set(df_eng.columns)
        raise SupplierAuditorMissingFeatureColumnsError(
            f"Feature engineering failed to produce required columns: {missing_engineered}"
        ) from e

    risk_probability = loaded.model.predict_proba(loaded.preprocessor.transform(X))[:, 1]
    risk_probability = np.round(risk_probability, 4)

    delivery_stress = df_eng['delivery_stress'].to_numpy()

    results = []
    for i, sku in enumerate(input_data.skus):
        prob = float(risk_probability[i])
        grade = assign_grade(prob)
        ds = float(delivery_stress[i])
        triggered, reason = combine_trigger_flags(prob, grade, ds, loaded.threshold)

        results.append(SupplierAuditResult(
            sku=sku,
            risk_probability=prob,
            supplier_grade=grade,
            stop_auto_buy_triggered=triggered,
            trigger_reason=reason,
            delivery_stress=ds,
        ))

    results.sort(key=lambda r: r.risk_probability, reverse=True)

    flagged_suppliers = [r.sku for r in results if r.stop_auto_buy_triggered]

    grade_distribution: dict[str, int] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for r in results:
        grade_distribution[r.supplier_grade] += 1

    return SupplierAuditorOutput(
        results=results,
        flagged_suppliers=flagged_suppliers,
        grade_distribution=grade_distribution,
        model_version=loaded.model_version,
        threshold_used=loaded.threshold,
    )
