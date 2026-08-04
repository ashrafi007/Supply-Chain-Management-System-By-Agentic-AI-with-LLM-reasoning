"""
Risk Detector Inference Tool — Pure model inference, no LangChain/LangGraph/LLM dependencies.

**Final model:** Stacking Ensemble (sklearn StackingClassifier).
  - Base learners: LightGBM, CatBoost, BalancedRandomForest.
  - Meta-learner: Logistic Regression (C=0.1, cv=5, stack_method='predict_proba').
  - Best F2-score on validation: 0.4432 (recall-weighted, β=2).
  - Best PR-AUC: 0.3336.
  - Trained on ~1.9M SKUs with ~0.7% positive class (went_on_backorder=1).

**Feature columns (exact order, 40 total):**
  Columns 1–25 are raw input features (loaded from Processed_Dataset.pkl):
    national_inv, lead_time, in_transit_qty, forecast_3_month, forecast_6_month,
    forecast_9_month, sales_1_month, sales_3_month, sales_6_month, sales_9_month,
    min_bank, potential_issue, pieces_past_due, perf_6_month_avg, perf_12_month_avg,
    local_bo_qty, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop,
    inv_velocity, safety_gap, sales_trend, perf_gap.

  Columns 26–40 are engineered features (computed via create_features()):
    inventory_health_net, sales_spike_ratio, months_of_stock_left,
    backorder_pressure, supplier_risk_score, forecast_accuracy_gap,
    inv_depletion_rate, lead_time_volatility, safety_stock_urgency,
    performance_trend, demand_forecast_ratio, transit_coverage,
    coverage_ratio, replenishment_gap, critical_state.

**Risk threshold (high-risk classification):** 0.945
  F2-optimized on validation set, argmax over threshold grid ∈ [0.01, 0.99).
  Saved to Models/risk_detector&Inventory_rebalencer/optimal_threshold.pkl.
  Secondary threshold (standard classification): 0.5 (predicted_label).

**Model file path:** Models/risk_detector&Inventory_rebalencer/stacking_proposed.pkl
  Format: joblib pickle (.pkl).
  Loaded via joblib.load(), wrapped in a cached singleton.

**NaN handling decision:**
  The training notebook (Risk_Detector_Classification_Model.ipynb Cell 14) builds
  the feature matrix BEFORE the imputation step (Cell 16), meaning the stacking
  model was actually trained on data with NaN present. To avoid train/serve skew,
  this module passes NaN straight through after feature engineering, never inventing
  an imputation step at inference time.

  Exception: perf_6_month_avg and perf_12_month_avg receive defensive -99.0 → NaN
  conversion, safe no-op if the upstream data already has them cleaned.

**Upstream data quirks (per spec):**
  - lead_time may contain NaN (passed through, not imputed).
  - perf_6_month_avg / perf_12_month_avg may have -99.0 sentinel (converted to NaN).
  - national_inv can be legitimately negative (not clipped).
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
    RiskDetectorError,
    RiskDetectorInput,
    RiskDetectorOutput,
    RiskDetectorPrediction,
    MissingFeatureColumnsError,
    ModelLoadError,
    SkuNotFoundError,
)

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class LoadedModel:
    """Immutable container for a loaded model + metadata."""
    model: Any
    feature_columns: list[str]
    threshold: float
    model_version: str


def _detect_gpu_available() -> bool:
    """
    Auto-detect GPU via torch, with nvidia-smi subprocess fallback.
    Mirrors the notebook's Cell 4 logic.
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
def load_risk_detector_model(model_path: str | None = None) -> LoadedModel:
    """
    Load the trained stacking ensemble model from disk (cached, singleton).

    Args:
        model_path: Optional absolute path to stacking_proposed.pkl.
                   If None, defaults to the repo's Models/ directory.

    Returns:
        LoadedModel(model, feature_columns, threshold, model_version).

    Raises:
        ModelLoadError: If any artifact cannot be loaded.
    """
    if model_path is None:
        repo_root = Path(__file__).parent.parent
        model_path = (
            repo_root / 'Models' / 'risk_detector&Inventory_rebalencer'
            / 'stacking_proposed.pkl'
        )
    else:
        model_path = Path(model_path)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {model_path}: {e}") from e

    # Load feature columns and threshold from adjacent .pkl files in the same directory.
    model_dir = model_path.parent
    feature_cols_path = model_dir / 'feature_columns.pkl'
    threshold_path = model_dir / 'optimal_threshold.pkl'

    try:
        feature_columns = joblib.load(feature_cols_path)
        threshold = float(joblib.load(threshold_path))
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load feature columns or threshold from {model_dir}: {e}"
        ) from e

    if not isinstance(feature_columns, list):
        raise ModelLoadError(
            f"Expected feature_columns.pkl to contain a list, got {type(feature_columns)}"
        )

    if not (0.0 <= threshold <= 1.0):
        raise ModelLoadError(
            f"Expected threshold in [0.0, 1.0], got {threshold}"
        )

    # Defensive fix for XGBoost GPU-trained boosters: force CPU if no GPU present.
    # (The stacking ensemble doesn't contain XGBoost, but this is defensive for
    # robustness if the artifact architecture ever changes.)
    gpu_available = _detect_gpu_available()
    if not gpu_available:
        try:
            if hasattr(model, 'estimators_'):
                for name, est in model.estimators_:
                    if hasattr(est, 'get_booster'):
                        est.get_booster().set_param({'device': 'cpu'})
        except Exception:
            pass  # Silently skip if not applicable (this model doesn't have boosters).

    return LoadedModel(
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        model_version='stacking_ensemble_v1'
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact feature engineering from the training notebook
    (Risk_Detector_Classification_Model.ipynb, Cell 12).

    Input df must have all 25 raw columns (1–25 in the spec).
    Output adds 15 engineered columns (26–40 in the spec).

    Does NOT impute NaN or handle -99.0 sentinel — callers must do that before
    passing to this function (see _prepare_feature_row).

    Replaces infinite values with 0 (matching the notebook exactly).
    """
    d = df.copy()

    d['inventory_health_net'] = (
        d['national_inv'] + d['in_transit_qty']
    ) - d['sales_1_month']

    d['sales_spike_ratio'] = (
        d['sales_1_month'] / ((d['sales_9_month'] / 9) + 0.001)
    )

    d['months_of_stock_left'] = (
        d['national_inv'] / (d['sales_1_month'] + 0.001)
    )

    d['backorder_pressure'] = (
        (d['pieces_past_due'] * (d['deck_risk'] + 1)) / (d['national_inv'] + 1)
    )

    d['supplier_risk_score'] = (
        d['deck_risk'] + d['oe_constraint'] + d['ppap_risk']
    )

    d['forecast_accuracy_gap'] = (
        abs(d['forecast_3_month'] - d['sales_3_month']) / (d['sales_3_month'] + 1)
    )

    d['inv_depletion_rate'] = (
        (d['sales_9_month'] - d['sales_1_month'] * 9) / (d['sales_9_month'] + 1)
    )

    d['lead_time_volatility'] = d['lead_time'] * d['perf_gap']

    d['safety_stock_urgency'] = (
        (d['min_bank'] - d['national_inv']) / (d['min_bank'] + 1)
    )

    d['performance_trend'] = d['perf_6_month_avg'] - d['perf_12_month_avg']

    d['demand_forecast_ratio'] = (
        d['forecast_3_month'] / ((d['sales_9_month'] / 9) + 0.001)
    )

    d['transit_coverage'] = d['in_transit_qty'] / (d['sales_1_month'] + 0.001)

    d['coverage_ratio'] = (
        (d['national_inv'] + d['in_transit_qty']) / (d['forecast_3_month'] + 1e-5)
    )

    d['replenishment_gap'] = (
        d['forecast_3_month'] - d['national_inv'] - d['in_transit_qty']
    )

    d['critical_state'] = (
        (d['months_of_stock_left'] < d['lead_time'])
        & (d['safety_stock_urgency'] > 0)
        & (d['replenishment_gap'] > 0)
    ).astype(int)

    # Replace infinite values with 0 (per notebook).
    d = d.replace([np.inf, -np.inf], 0)

    return d


def _prepare_feature_row(
    sku: str,
    row_dict: dict[str, Any],
    feature_columns: list[str]
) -> pd.Series:
    """
    Prepare a single SKU's feature row for inference.

    1. Converts -99.0 → NaN in perf_6_month_avg and perf_12_month_avg.
    2. Checks that all required raw columns (1–25) are present.
    3. Runs create_features() to derive columns 26–40.
    4. Reindexes to match the persisted feature_columns order.
    5. Returns a pd.Series with NaN intact (no imputation).

    Args:
        sku: SKU identifier (for error messages).
        row_dict: Feature dict, e.g. {'national_inv': 100, 'lead_time': 5, ...}
        feature_columns: Persisted 40-column list from the model.

    Returns:
        pd.Series with columns matching feature_columns, in order.

    Raises:
        MissingFeatureColumnsError: If any required raw column is absent.
    """
    # Raw columns that must be present (columns 1–25 in the spec).
    required_raw_cols = [
        'national_inv', 'lead_time', 'in_transit_qty',
        'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
        'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
        'min_bank', 'potential_issue', 'pieces_past_due',
        'perf_6_month_avg', 'perf_12_month_avg',
        'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk',
        'stop_auto_buy', 'rev_stop',
        'inv_velocity', 'safety_gap', 'sales_trend', 'perf_gap'
    ]

    missing = set(required_raw_cols) - set(row_dict.keys())
    if missing:
        raise MissingFeatureColumnsError(
            f"SKU {sku}: missing required columns {missing}"
        )

    # Start with raw columns.
    row_series = pd.Series(row_dict)

    # Defensive -99.0 → NaN sentinel conversion for perf columns.
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        if col in row_series.index:
            val = row_series[col]
            if isinstance(val, (int, float)) and val == -99.0:
                row_series[col] = np.nan

    # Apply feature engineering (adds columns 26–40).
    df_temp = pd.DataFrame([row_series])
    df_engineered = create_features(df_temp)
    engineered_series = df_engineered.iloc[0]

    # Reindex to the persisted feature order, raising KeyError if any column is missing.
    try:
        final_row = engineered_series[feature_columns]
    except KeyError as e:
        missing_engineered = set(feature_columns) - set(engineered_series.index)
        raise MissingFeatureColumnsError(
            f"SKU {sku}: feature engineering failed to produce {missing_engineered}"
        ) from e

    return final_row


def predict_risk(input_data: RiskDetectorInput) -> RiskDetectorOutput:
    """
    Predict backorder risk for the given SKUs.

    Args:
        input_data: RiskDetectorInput with skus list and data dict.

    Returns:
        RiskDetectorOutput with predictions, high_risk_skus, metadata.

    Raises:
        SkuNotFoundError: If any SKU in input_data.skus is missing from data dict.
        MissingFeatureColumnsError: If required features are absent.
        ModelLoadError: If the model cannot be loaded (on first call only).
    """
    loaded = load_risk_detector_model()

    # Validate that all requested SKUs are present.
    missing_skus = set(input_data.skus) - set(input_data.data.keys())
    if missing_skus:
        raise SkuNotFoundError(
            f"Requested SKUs not found in data dict: {missing_skus}"
        )

    # Prepare feature matrix for all SKUs.
    feature_rows = []
    for sku in input_data.skus:
        row = _prepare_feature_row(sku, input_data.data[sku], loaded.feature_columns)
        feature_rows.append(row)

    X = pd.DataFrame(feature_rows)

    # Inference: get probability of went_on_backorder = 1 (positive class).
    proba = loaded.model.predict_proba(X)[:, 1]

    # Apply thresholds.
    predictions = []
    for sku, prob in zip(input_data.skus, proba):
        pred = RiskDetectorPrediction(
            sku=sku,
            backorder_probability=float(prob),
            predicted_label=bool(prob >= 0.5),
            is_high_risk=bool(prob >= loaded.threshold)
        )
        predictions.append(pred)

    # Build high_risk_skus from predictions.
    high_risk_skus = [p.sku for p in predictions if p.is_high_risk]

    return RiskDetectorOutput(
        predictions=predictions,
        high_risk_skus=high_risk_skus,
        model_version=loaded.model_version,
        threshold_used=loaded.threshold
    )
