"""
Forecast Optimizer Inference Tool — Pure model inference, no LangChain/LangGraph/LLM dependencies.

**Two-task design:**
  Task A (classification, `is_overestimating`): detects whether the human forecast is a
    chronic overestimate. Target: bias_ratio_3m > 1.15.
  Task B (regression, `correction_factor`): a multiplier to correct the human forecast.
    Target: (sales_3_month / (forecast_3_month + 1e-5)).clip(0.3, 1.5).

**Final model B — resolved persistence gap (item #1 from spec):**
  `agent3_thresholds.pkl` saves `selected_model_a` but has NO `selected_model_b` key —
  confirmed by directly loading the artifact (keys: lgbm_a, catboost_a, brf_a,
  overestimate_threshold, selected_model_a, selected_threshold_a). The winner is only
  recoverable from Notebooks/Forcast_opt.ipynb's saved cell 26 output:
      "TASK B — CORRECTION REGRESSOR: VALIDATION COMPARISON"
      XGBoost   MAE 0.1618  RMSE 0.2840  R2 0.7384
      LightGBM  MAE 0.1664  RMSE 0.2884  R2 0.7302
      CatBoost  MAE 0.1692  RMSE 0.2887  R2 0.7295
      "-> Selected for final test: XGBoost (lowest MAE on val set)"
      "Task B final: XGBoost"
  corroborated by cell 36's held-out test section header
  "TASK B - CORRECTION REGRESSOR (XGBoost) - TEST RESULTS". This is hardcoded here as
  SELECTED_MODEL_B = "XGBoost" (-> agent3_corrector_xgb.pkl), not guessed and not defaulted
  to the first-listed candidate. Task A's winner (LightGBM, threshold 0.765) IS persisted
  in agent3_thresholds.pkl and is loaded dynamically at runtime, not hardcoded.

**`perf_gap` resolution (same as inventory_rebalancer_tool.py):**
  Confirmed present as raw column #26 in Processed_Dataset.pkl's schema (notebook cell 10's
  captured df.columns output). It is never computed in this notebook either. Treated as a
  required, opaque raw passthrough input field with no derivation — consistent with how
  inventory_rebalancer_tool.py and risk_detector_tool.py treat the same column.

**`sku` vs `product_id` (resolved item from spec):**
  Neither column exists in Processed_Dataset.pkl's actual 26-column schema (confirmed via
  the notebook's captured df.columns output). The notebook's agent3_predict() reads
  product_row.get('product_id', 'unknown'), which would silently return 'unknown' for
  every real row — a notebook bug, not behavior to replicate. This module uses `sku` purely
  as the caller-supplied dict key (ForecastOptimizerInput.data: dict[sku, feature_dict]),
  matching RiskDetectorInput / InventoryRebalancerInput, never as an expected raw column.

**Leakage flag on forecast_accuracy_gap / inv_depletion_rate — resolved, does NOT apply:**
  The spec flagged that these two create_features() columns use sales_3_month/sales_9_month
  directly (the same variables both targets are built from) and are not in EXCLUDE_COLS.
  Tracing the notebook precisely: `df_full = create_features(df)` is used ONLY to
  reconstruct Agent 2's 40-feature input for calling its predict_proba() directly (the
  forbidden coupling this module must not replicate) and is discarded afterward.
  `df_agent3 = create_agent3_features(df)` is built from the RAW `df`, never merged with
  `df_full`. Directly loading agent3_feature_cols.pkl confirms zero overlap: none of
  create_features()'s 15 engineered columns (including forecast_accuracy_gap and
  inv_depletion_rate) appear in the real 28-item FEATURE_COLS. So this specific leakage
  risk does not materialize in the actually-shipped model. create_features() is still
  implemented below for spec fidelity/parity, but by design its output is NOT merged into
  the Agent-3 pipeline — it is computed and discarded, exactly matching the notebook's own
  (likely accidental) behavior. Do not "fix" this by merging it in; that would change the
  model's expected input distribution.

**Unpersisted per-column median gap (create_agent3_features()'s `d.fillna(d.median())`):**
  The notebook's fillna(median) runs over the WHOLE working frame (raw + engineered columns
  together), not just the engineered columns, and those medians were never persisted as an
  artifact. Recomputing real historical medians is infeasible here: Processed_Dataset.pkl
  and any DuckDB data-access layer are both absent from this repo (same infeasibility
  documented in inventory_rebalancer_tool.py). Resolution: replicate fillna(median) computed
  from the CURRENT CALL's batch (not a persisted historical value), then apply a documented
  neutral 0.0 fallback for any column whose batch median is itself NaN (e.g. single-row
  calls where that column's only value is NaN). This is a known limitation: single-row/
  small-batch predictions do not benefit from a true historical median. A future
  improvement would persist real historical per-column medians (e.g. via a DuckDB
  data-access layer, once one exists) as a proper artifact.

**Predictor/reasoner separation:**
  This module contains only feature engineering, thresholding, and model inference.
  `backorder_probability` and `alarm_triggered` are required caller-supplied inputs (from
  the Risk Detector tool's output via the orchestrator), never computed internally. This
  module does NOT load stacking_proposed.pkl or optimal_thresholds.pkl — a deliberate
  architectural deviation from the notebook's inline computation, matching
  inventory_rebalancer_tool.py's precedent.

**Artifact paths:**
  Models/forecast/agent3_bias_detector_lgbm.pkl      — Task A candidate (selected here)
  Models/forecast/agent3_bias_detector_catboost.pkl  — Task A candidate (not loaded)
  Models/forecast/agent3_bias_detector_brf.pkl       — Task A candidate (not loaded; ~3.7GB)
  Models/forecast/agent3_corrector_xgb.pkl           — Task B candidate (selected here)
  Models/forecast/agent3_corrector_lgbm.pkl          — Task B candidate (not loaded)
  Models/forecast/agent3_corrector_catboost.pkl      — Task B candidate (not loaded)
  Models/forecast/agent3_feature_cols.pkl            — authoritative FEATURE_COLS (required)
  Models/forecast/agent3_thresholds.pkl              — selected_model_a + threshold (required)
  Models/forecast/agent3_business_metrics.pkl        — informational only, never loaded

**Non-goals:**
  No LangChain/LangGraph imports or @tool decorators. No training, grid search, plotting,
  SHAP, ablation. No changes to any DuckDB data-access layer (none exists). No retrain/
  regeneration of model artifacts. No loading of stacking_proposed.pkl/optimal_thresholds.pkl.
  No modification of the original notebook.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .schemas import (
    ForecastOptimizerError,
    ForecastOptimizerInput,
    ForecastOptimizerOutput,
    ForecastOptimizerRecommendation,
    ForecastModelLoadError,
    ForecastSkuNotFoundError,
    ForecastMissingFeatureColumnsError,
    MissingRiskSignalError,
)

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants (documentation/comparison; loaded artifact is runtime truth)
# ──────────────────────────────────────────────────────────────────────────

# All raw columns of Processed_Dataset.pkl's real 26-column schema needed anywhere in
# this pipeline (create_features(), create_agent3_features(), or as a direct FEATURE_COLS
# passthrough) — i.e. every raw column except `went_on_backorder`, which is never needed
# at inference (it's the historical label, excluded from FEATURE_COLS entirely).
REQUIRED_RAW_COLS: list[str] = [
    'national_inv', 'lead_time', 'in_transit_qty',
    'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
    'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
    'min_bank', 'potential_issue', 'pieces_past_due',
    'perf_6_month_avg', 'perf_12_month_avg',
    'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk',
    'stop_auto_buy', 'rev_stop', 'inv_velocity', 'safety_gap',
    'sales_trend', 'perf_gap',
]

# Step 1 — the 15 columns computed by create_features(), reproduced from
# Notebooks/Forcast_opt.ipynb Cell 12 for spec fidelity. NONE of these reach
# FEATURE_COLS (see module docstring's leakage-flag resolution) — computed but discarded.
ENGINEERED_COLS_STEP1: list[str] = [
    'inventory_health_net', 'sales_spike_ratio', 'months_of_stock_left',
    'backorder_pressure', 'supplier_risk_score', 'forecast_accuracy_gap',
    'inv_depletion_rate', 'lead_time_volatility', 'safety_stock_urgency',
    'performance_trend', 'demand_forecast_ratio', 'transit_coverage',
    'coverage_ratio', 'replenishment_gap', 'critical_state',
]

# Documentation constant: the real 28-item agent3_feature_cols.pkl contents, confirmed via
# direct joblib.load. The loaded artifact is the actual runtime source of truth (see
# load_forecast_optimizer_artifacts()); this constant exists for comparison/documentation.
AGENT3_FEATURES: list[str] = [
    'national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
    'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'min_bank',
    'potential_issue', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg',
    'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy',
    'rev_stop', 'inv_velocity', 'safety_gap', 'sales_trend', 'perf_gap',
    'backorder_probability', 'alarm_triggered', 'risk_level_encoded',
    'forecast_consistency', 'correction_suppressed', 'inv_vs_forecast',
]

# Task A candidates -> artifact filename. Only the one named by
# agent3_thresholds.pkl['selected_model_a'] is ever loaded.
MODEL_A_FILE_MAP: dict[str, str] = {
    'LightGBM': 'agent3_bias_detector_lgbm.pkl',
    'CatBoost': 'agent3_bias_detector_catboost.pkl',
    'BalancedRF': 'agent3_bias_detector_brf.pkl',
}

# Task B — resolved via notebook cell 26/36 evidence (see module docstring). Not persisted
# in any artifact, so hardcoded here rather than guessed.
SELECTED_MODEL_B: str = 'XGBoost'
MODEL_B_FILE: str = 'agent3_corrector_xgb.pkl'


@dataclass(frozen=True)
class LoadedForecastOptimizerModel:
    """Immutable container for the loaded forecast optimizer models + metadata."""
    model_a: Any
    model_b: Any
    feature_cols: list[str]
    threshold_a: float
    model_version_a: str
    model_version_b: str


def _detect_gpu_available() -> bool:
    """
    Auto-detect GPU via torch, with nvidia-smi subprocess fallback.
    Mirrors inventory_rebalancer_tool.py / risk_detector_tool.py's Cell 4 logic.
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
def load_forecast_optimizer_artifacts(models_dir: str | None = None) -> LoadedForecastOptimizerModel:
    """
    Load only the two SELECTED models (Task A per agent3_thresholds.pkl's
    'selected_model_a'; Task B hardcoded to XGBoost per the resolved persistence gap —
    see module docstring), plus agent3_feature_cols.pkl and the Task A threshold.
    Cached (module-level lru_cache singleton per process).

    Never loads the non-selected Task A/B candidates or agent3_business_metrics.pkl.

    Args:
        models_dir: Optional absolute path to the artifact directory.
                    If None, defaults to <repo_root>/Models/forecast.

    Returns:
        LoadedForecastOptimizerModel(model_a, model_b, feature_cols, threshold_a,
        model_version_a, model_version_b).

    Raises:
        ForecastModelLoadError: If any required artifact cannot be loaded or is malformed.
    """
    if models_dir is None:
        repo_root = Path(__file__).parent.parent
        models_dir = repo_root / 'Models' / 'forecast'
    else:
        models_dir = Path(models_dir)

    thresholds_path = models_dir / 'agent3_thresholds.pkl'
    feature_cols_path = models_dir / 'agent3_feature_cols.pkl'

    try:
        thresholds = joblib.load(thresholds_path)
    except Exception as e:
        raise ForecastModelLoadError(
            f"Failed to load forecast optimizer thresholds from {thresholds_path}: {e}"
        ) from e

    if not isinstance(thresholds, dict) or 'selected_model_a' not in thresholds:
        raise ForecastModelLoadError(
            f"Expected {thresholds_path} to contain a dict with 'selected_model_a', "
            f"got {thresholds!r}"
        )

    selected_model_a = thresholds['selected_model_a']
    threshold_a = thresholds.get('selected_threshold_a')
    if threshold_a is None:
        raise ForecastModelLoadError(
            f"{thresholds_path} is missing 'selected_threshold_a'"
        )

    if selected_model_a not in MODEL_A_FILE_MAP:
        raise ForecastModelLoadError(
            f"Unknown selected_model_a {selected_model_a!r} in {thresholds_path}; "
            f"expected one of {list(MODEL_A_FILE_MAP)}"
        )

    model_a_path = models_dir / MODEL_A_FILE_MAP[selected_model_a]
    try:
        model_a = joblib.load(model_a_path)
    except Exception as e:
        raise ForecastModelLoadError(
            f"Failed to load Task A model ({selected_model_a}) from {model_a_path}: {e}"
        ) from e

    model_b_path = models_dir / MODEL_B_FILE
    try:
        model_b = joblib.load(model_b_path)
    except Exception as e:
        raise ForecastModelLoadError(
            f"Failed to load Task B model ({SELECTED_MODEL_B}) from {model_b_path}: {e}"
        ) from e

    try:
        feature_cols = joblib.load(feature_cols_path)
    except Exception as e:
        raise ForecastModelLoadError(
            f"Failed to load forecast optimizer feature columns from {feature_cols_path}: {e}"
        ) from e

    if not isinstance(feature_cols, list):
        raise ForecastModelLoadError(
            f"Expected {feature_cols_path} to contain a list, got {type(feature_cols)}"
        )

    if feature_cols != AGENT3_FEATURES:
        warnings.warn(
            f"Loaded agent3_feature_cols.pkl differs from the AGENT3_FEATURES module "
            f"constant: artifact={feature_cols!r} vs constant={AGENT3_FEATURES!r}. "
            f"Using the artifact's list.",
            stacklevel=2
        )

    # Defensive fix for XGBoost GPU-trained boosters: force CPU if no GPU present.
    gpu_available = _detect_gpu_available()
    if not gpu_available:
        try:
            model_b.get_booster().set_param({'device': 'cpu'})
        except Exception:
            pass  # Silently skip if not applicable.

    return LoadedForecastOptimizerModel(
        model_a=model_a,
        model_b=model_b,
        feature_cols=feature_cols,
        threshold_a=float(threshold_a),
        model_version_a=f"{selected_model_a.lower()}_bias_detector_v1",
        model_version_b=f"{SELECTED_MODEL_B.lower()}_corrector_v1",
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — reproduced verbatim from Notebooks/Forcast_opt.ipynb Cell 12 for spec
    fidelity. Computes the 15 ENGINEERED_COLS_STEP1 columns; inf/-inf replaced with 0.

    NOTE: unlike a typical pipeline stage, this function's output is intentionally NOT
    merged into the Agent-3 feature pipeline downstream (see module docstring's
    leakage-flag resolution) — in the notebook it is used only to reconstruct Agent 2's
    feature space for a model-chaining call this module must not replicate. It is
    reproduced here so the Step-1 formulas are available/testable, but callers of
    optimize_forecast() must not expect its columns to influence the model output.
    """
    d = df.copy()
    d['inventory_health_net'] = (d['national_inv'] + d['in_transit_qty']) - d['sales_1_month']
    d['sales_spike_ratio'] = d['sales_1_month'] / ((d['sales_9_month'] / 9) + 0.001)
    d['months_of_stock_left'] = d['national_inv'] / (d['sales_1_month'] + 0.001)
    d['backorder_pressure'] = (d['pieces_past_due'] * (d['deck_risk'] + 1)) / (d['national_inv'] + 1)
    d['supplier_risk_score'] = d['deck_risk'] + d['oe_constraint'] + d['ppap_risk']
    d['forecast_accuracy_gap'] = abs(d['forecast_3_month'] - d['sales_3_month']) / (d['sales_3_month'] + 1)
    d['inv_depletion_rate'] = (d['sales_9_month'] - d['sales_1_month'] * 9) / (d['sales_9_month'] + 1)
    d['lead_time_volatility'] = d['lead_time'] * d['perf_gap']
    d['safety_stock_urgency'] = (d['min_bank'] - d['national_inv']) / (d['min_bank'] + 1)
    d['performance_trend'] = d['perf_6_month_avg'] - d['perf_12_month_avg']
    d['demand_forecast_ratio'] = d['forecast_3_month'] / ((d['sales_9_month'] / 9) + 0.001)
    d['transit_coverage'] = d['in_transit_qty'] / (d['sales_1_month'] + 0.001)
    d['coverage_ratio'] = (d['national_inv'] + d['in_transit_qty']) / (d['forecast_3_month'] + 1e-5)
    d['replenishment_gap'] = d['forecast_3_month'] - d['national_inv'] - d['in_transit_qty']
    d['critical_state'] = (
        (d['months_of_stock_left'] < d['lead_time'])
        & (d['safety_stock_urgency'] > 0)
        & (d['replenishment_gap'] > 0)
    ).astype(int)
    d = d.replace([np.inf, -np.inf], 0)
    return d


def encode_risk_level(prob: float) -> int:
    """
    Verbatim from the spec/notebook. No model dependency — pure thresholding.
    >= 0.80 -> 3 (CRITICAL), >= 0.60 -> 2 (HIGH), >= 0.35 -> 1 (MEDIUM), else -> 0 (LOW).
    """
    if prob >= 0.80:
        return 3
    elif prob >= 0.60:
        return 2
    elif prob >= 0.35:
        return 1
    else:
        return 0


def create_agent3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 — reproduced verbatim (F1-F12) from Notebooks/Forcast_opt.ipynb Cell 16.
    Requires `backorder_probability` and `alarm_triggered` columns already attached
    (Step 2 — see optimize_forecast()).

    Median-fill gap resolution (see module docstring): the notebook's
    `d.replace([inf,-inf], nan); d.fillna(d.median(numeric_only=True))` runs over the
    whole frame and its medians were never persisted. Here the median is computed from
    the CURRENT CALL's batch (no persisted historical value is recoverable — see
    docstring), then a documented neutral 0.0 fallback covers any column whose batch
    median is itself NaN (e.g. single-row calls with a NaN input in that column).
    """
    d = df.copy()

    d['bias_ratio_3m'] = d['forecast_3_month'] / (d['sales_3_month'] + 1e-5)
    d['bias_ratio_6m'] = d['forecast_6_month'] / (d['sales_6_month'] + 1e-5)
    d['bias_ratio_9m'] = d['forecast_9_month'] / (d['sales_9_month'] + 1e-5)

    d['error_3m'] = d['forecast_3_month'] - d['sales_3_month']
    d['error_6m'] = d['forecast_6_month'] - d['sales_6_month']
    d['error_9m'] = d['forecast_9_month'] - d['sales_9_month']

    d['bias_trend_3_to_9'] = d['bias_ratio_9m'] - d['bias_ratio_3m']

    d['chronic_overestimator'] = (
        (d['bias_ratio_3m'] > 1.10)
        & (d['bias_ratio_6m'] > 1.10)
        & (d['bias_ratio_9m'] > 1.10)
    ).astype(int)

    d['forecast_consistency'] = d[
        ['forecast_3_month', 'forecast_6_month', 'forecast_9_month']
    ].std(axis=1) / (d['forecast_9_month'] + 1e-5)

    d['sales_velocity_ratio'] = d['sales_1_month'] / ((d['sales_9_month'] / 9) + 1e-5)

    d['demand_volatility'] = d[
        ['sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month']
    ].std(axis=1) / (d['sales_9_month'] + 1e-5)

    d['sales_momentum'] = (
        d['sales_1_month'] - (d['sales_3_month'] / 3)
    ) / (d['sales_3_month'] / 3 + 1e-5)

    d['risk_adjusted_bias'] = d['bias_ratio_3m'] * (1 - d['backorder_probability'])

    d['correction_suppressed'] = d['alarm_triggered'].astype(int)

    d['inv_vs_forecast'] = (
        d['national_inv'] + d['in_transit_qty']
    ) / (d['forecast_3_month'] + 1e-5)

    d['supplier_forecast_stress'] = (
        (1 - d['perf_6_month_avg'].clip(0, 1)) * d['bias_ratio_3m']
    )

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.fillna(d.median(numeric_only=True))
    d = d.fillna(0.0)  # neutral last-resort fallback when the batch median is itself NaN

    return d


def optimize_forecast(input_data: ForecastOptimizerInput) -> ForecastOptimizerOutput:
    """
    Pure inference. Given SKUs + raw feature rows + backorder_probability +
    alarm_triggered:
      1. Run create_features() (Step 1) for spec fidelity — its output is computed but
         intentionally discarded, matching the notebook's actual behavior (see module
         docstring's leakage-flag resolution).
      2. Attach backorder_probability, alarm_triggered, and derive risk_level_encoded
         via encode_risk_level().
      3. Run create_agent3_features() (Step 3), resolving the unpersisted-median gap
         per the documented batch-median + neutral-fallback strategy.
      4. Select FEATURE_COLS from the loaded artifact (exact column order).
      5. Run Task A model -> bias_probability, bias_detected.
      6. Run Task B model -> correction_factor (clipped to [0.3, 1.5], matching the
         training target's own clip), then apply the alarm_triggered override to 1.0.
      7. Compute adjusted_forecast_3m, recommendation, bias_severity exactly as in the
         notebook's agent3_predict() reference implementation.
      8. Return ForecastOptimizerOutput.
    Deterministic: same input -> same output, every time (same batch composition; see
    module docstring re: batch-local medians and the tradeoff this implies for a SKU
    called alongside different co-batched SKUs).

    Raises:
        ForecastSkuNotFoundError: a requested SKU has no matching row in input data.
        MissingRiskSignalError: a requested SKU has no entry in backorder_probability
            or alarm_triggered.
        ForecastMissingFeatureColumnsError: required raw columns are missing, or feature
            engineering failed to produce a column required by the loaded feature_cols.
        ForecastModelLoadError: a loaded artifact is malformed/incompatible.
    """
    loaded = load_forecast_optimizer_artifacts()

    # Defensive re-checks (belt-and-suspenders; Pydantic's validate_sku_coverage already
    # guarantees these at construction time).
    missing_skus = set(input_data.skus) - set(input_data.data.keys())
    if missing_skus:
        raise ForecastSkuNotFoundError(
            f"Requested SKUs not found in data dict: {missing_skus}"
        )

    missing_bo = set(input_data.skus) - set(input_data.backorder_probability.keys())
    if missing_bo:
        raise MissingRiskSignalError(
            f"Requested SKUs not found in backorder_probability dict: {missing_bo}"
        )

    missing_alarm = set(input_data.skus) - set(input_data.alarm_triggered.keys())
    if missing_alarm:
        raise MissingRiskSignalError(
            f"Requested SKUs not found in alarm_triggered dict: {missing_alarm}"
        )

    # Build the raw feature frame, preserving input order.
    rows = []
    for sku in input_data.skus:
        row = input_data.data[sku]
        missing_cols = set(REQUIRED_RAW_COLS) - set(row.keys())
        if missing_cols:
            raise ForecastMissingFeatureColumnsError(
                f"SKU {sku}: missing required raw columns {missing_cols}"
            )
        rows.append(row)

    df = pd.DataFrame(rows, index=list(input_data.skus))

    # Step 1 — computed for spec fidelity, intentionally discarded (see docstring).
    _ = create_features(df)

    # Step 2 — integrate Agent 2 signals (caller-supplied, never computed here).
    df = df.copy()
    df['backorder_probability'] = [input_data.backorder_probability[sku] for sku in input_data.skus]
    df['alarm_triggered'] = [bool(input_data.alarm_triggered[sku]) for sku in input_data.skus]
    df['risk_level_encoded'] = df['backorder_probability'].apply(encode_risk_level)

    # Step 3 — Agent-3-specific feature engineering.
    df_agent3 = create_agent3_features(df)

    # Step 4 — feature selection from the loaded artifact's exact column order.
    try:
        X = df_agent3[loaded.feature_cols]
    except KeyError as e:
        missing_engineered = set(loaded.feature_cols) - set(df_agent3.columns)
        raise ForecastMissingFeatureColumnsError(
            f"Feature engineering failed to produce required columns: {missing_engineered}"
        ) from e

    # Task A model matrix needs alarm_triggered as int (matches training: agent2_preds.astype(int)).
    X = X.copy()
    if 'alarm_triggered' in X.columns:
        X['alarm_triggered'] = X['alarm_triggered'].astype(int)

    # Step 5 — Task A inference.
    bias_probability = loaded.model_a.predict_proba(X)[:, 1]
    bias_detected = bias_probability >= loaded.threshold_a

    # Step 6 — Task B inference + range clip (matches training target's own .clip(0.3, 1.5)).
    raw_correction_factor = loaded.model_b.predict(X)
    correction_factor = np.clip(raw_correction_factor, 0.3, 1.5)

    # Business override: an active Agent-2 alarm forces correction_factor to 1.0.
    alarm_array = df['alarm_triggered'].to_numpy()
    risk_override_applied = alarm_array
    correction_factor = np.where(alarm_array, 1.0, correction_factor)

    # Step 7 — adjusted forecast, recommendation, bias severity.
    human_forecast_3m = df['forecast_3_month'].to_numpy(dtype=float)
    adjusted_forecast_3m = human_forecast_3m * correction_factor

    recommendation = np.select(
        [correction_factor < 0.90, correction_factor > 1.10],
        ['REDUCE_PLAN', 'INCREASE_PLAN'],
        default='HOLD',
    )
    bias_severity = np.select(
        [correction_factor < 0.75, correction_factor < 0.90],
        ['SEVERE', 'MILD'],
        default='NONE',
    )

    recommendations = []
    for i, sku in enumerate(input_data.skus):
        recommendations.append(ForecastOptimizerRecommendation(
            sku=sku,
            human_forecast_3m=float(human_forecast_3m[i]),
            adjusted_forecast_3m=float(adjusted_forecast_3m[i]),
            correction_factor=float(correction_factor[i]),
            bias_detected=bool(bias_detected[i]),
            bias_probability=float(bias_probability[i]),
            bias_severity=str(bias_severity[i]),
            risk_override_applied=bool(risk_override_applied[i]),
            recommendation=str(recommendation[i]),
        ))

    flagged_for_reduction = [r.sku for r in recommendations if r.recommendation == 'REDUCE_PLAN']
    flagged_for_increase = [r.sku for r in recommendations if r.recommendation == 'INCREASE_PLAN']

    return ForecastOptimizerOutput(
        recommendations=recommendations,
        flagged_for_reduction=flagged_for_reduction,
        flagged_for_increase=flagged_for_increase,
        model_version_a=loaded.model_version_a,
        model_version_b=loaded.model_version_b,
    )
