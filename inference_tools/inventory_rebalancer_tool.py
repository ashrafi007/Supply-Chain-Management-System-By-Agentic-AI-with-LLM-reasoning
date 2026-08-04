"""
Inventory Rebalancer Inference Tool — Pure model inference, no LangChain/LangGraph/LLM dependencies.

**Final model:** XGBoost Regressor (xgb.XGBRegressor).
  - Target: urgency_score (continuous, clipped to [0, 1] during training).
  - Reported test metrics: R² = 0.9976, RMSE = 0.005455, MAE = 0.002674, Spearman ρ = 0.9764.
  - Trained with GPU (`device='cuda'`, `tree_method='hist'`) when available, CPU fallback.
  - Feature count: 13 (after selecting a subset from 15 engineered columns).

**Predictor/reasoner separation:**
  This module contains only feature engineering, imputation, and model inference.
  `backorder_probability` is a required caller-supplied input (from the Risk Detector tool's
  output via the orchestrator), never computed internally. This module does NOT load
  `stacking_proposed.pkl` — a deliberate architectural deviation from the notebook's inline
  computation, not an oversight. The separation ensures this tool has zero model-chaining
  dependencies and can be tested/deployed independently.

**Feature engineering (15 engineered columns computed, 13 selected for the model):**
  All 15 are replicated exactly from Inventory_Rebalencer.ipynb Cell 9:
    inventory_health_net, sales_spike_ratio, months_of_stock_left,
    backorder_pressure, supplier_risk_score, forecast_accuracy_gap,
    inv_depletion_rate, lead_time_volatility, safety_stock_urgency,
    performance_trend, demand_forecast_ratio, transit_coverage,
    coverage_ratio, replenishment_gap, critical_state.
  Only the subset {months_of_stock_left, backorder_pressure, inv_depletion_rate,
  safety_stock_urgency, inventory_health_net, coverage_ratio, transit_coverage}
  plus the caller-supplied backorder_probability are selected into the model's
  X matrix (REBALANCER_FEATURES). The remaining 8 engineered columns are computed
  for notebook fidelity and for critical_state's internal use (top_priority_skus filtering),
  but never reach the model.

**`perf_gap` resolution (required item #1 from spec):**
  `perf_gap` is NOT computed in any notebook in this repo. It is raw column data
  pre-existing in the external Processed_Dataset.pkl (which does not exist in this repo).
  Its source formula is unrecoverable from this codebase. This module treats it as a
  required raw input field, passed through untouched with no imputation or derivation,
  matching how risk_detector_tool.py treats the same column.

**inf/NaN engineered-column cleaning gap resolution (required item #2 from spec):**
  The notebook's Cell 9 does a global inf→NaN→median-fill pass on all 15 ENGINEERED_COLS
  on the full dataset before train/test split. Those medians were never saved as an artifact.
  The spec considered two options:
    (1) Recompute medians via a DuckDB data-access layer and persist as a new artifact.
    (2) Replace inf→NaN across ENGINEERED_COLS but skip median-fill; let the already-fit
        rebalancer_imputer.pkl handle NaN during the final imputation step for columns
        that are selected into REBALANCER_FEATURES; leave NaN untouched in the other 8
        non-selected engineered columns (they never reach the model anyway).

  Option (1) is INFEASIBLE here — confirmed via repo-wide grep that no DuckDB layer exists
  anywhere in this repo. This module uses option (2). Specifically:
  - _compute_engineered_features() replaces [inf, -inf] with NaN across all 15 ENGINEERED_COLS
    (matching the notebook's cleaning-pass scope).
  - Of those 15, only 7 overlap with REBALANCER_FEATURES. NaN in those 7 is handled by the
    loaded rebalancer_imputer.pkl during the final imputation step (post feature selection).
  - NaN in the remaining 8 non-selected engineered columns is left as-is and never reaches
    the model — harmless, intentional, not a bug. Document this as matching the notebook's
    behavior as closely as feasible given the missing-artifact constraint.
  - This is deliberately different from risk_detector_tool.py's create_features(), which
    replaces inf with 0 (not NaN) — the difference is intentional, driven by each tool's
    own resolved gap analysis, not a bug or inconsistency to "fix."

**`manufacture_rank` (global) omission (required item #4 from spec):**
  RebalancerRecommendation.manufacture_rank is always None in this iteration.
  Global rank (percentile across the full dataset) is not implementable without a
  periodically-refreshed reference distribution/percentile table artifact computed
  over the entire dataset from a periodic batch-scoring job. Such an artifact does not
  exist yet. To implement this later:
    - A batch-scoring job would compute urgency_scores over the full dataset,
      generate percentile bins/quantiles, and save as a reference artifact
      (e.g., rebalancer_percentile_reference.pkl or similar).
    - load_rebalancer_artifacts() would load this reference.
    - rebalance_inventory() would map each SKU's urgency_score to its percentile
      using the reference distribution.
  Until then, manufacture_rank must remain None to avoid silently returning misleading
  batch-local ranks as global ranks.

**`rebalancer_feature_columns.pkl` confirmation (required item #3 from spec):**
  Direct inspection via joblib.load confirms the artifact contains exactly the
  13-item list in this spec's order:
    ['national_inv', 'min_bank', 'oe_constraint', 'months_of_stock_left',
     'inv_depletion_rate', 'safety_stock_urgency', 'backorder_pressure',
     'inventory_health_net', 'in_transit_qty', 'lead_time', 'coverage_ratio',
     'transit_coverage', 'backorder_probability']
  No discrepancy between artifact and spec. At runtime, the loaded artifact's list
  is the authoritative source of truth (not the module constant REBALANCER_FEATURES),
  with a warnings.warn() if they ever diverge in a future artifact refresh.

**`urgency_score_predicted` clipping:**
  The training target was explicitly .clip(0, 1) before fitting (see Inventory_Rebalencer.ipynb
  Cell 12). Since XGBRegressor predictions on OOD inputs are not range-guaranteed,
  np.clip(raw_prediction, 0.0, 1.0) is applied at inference time to mirror the target's
  own definition. This ensures urgency_score_predicted always satisfies [0.0, 1.0].

**`recommended_qty` behavior:**
  Computed as clip(min_bank - national_inv, lower=0) from the raw (non-imputed, non-engineered)
  input row values. Matches Inventory_Rebalencer.ipynb Step 25 exactly. If either min_bank or
  national_inv is NaN in the raw input, recommended_qty will be NaN — faithful notebook
  replication, not a bug. Use np.clip, not Python's max() — max(nan, 0) vs max(0, nan) are
  asymmetric/order-dependent in Python, while np.clip(nan, 0, None) consistently propagates NaN.

**`batch_rank` (ties in urgency_score):**
  Rank within the current call's batch only (1 = most urgent), computed from
  urgency_score_predicted descending, ties broken by original position in input_data.skus
  (stable sort via np.argsort(..., kind='stable')). Produces unique consecutive integers 1..N.
  Does not use pandas .rank() (would produce non-integer ranks on ties); uses numpy's stable
  argsort instead.

**`top_priority_skus` definition:**
  SKUs where recommended_qty > 0 AND critical_state == 1. The critical_state flag is computed
  internally during _compute_engineered_features (it's one of the 15 engineered columns) but
  is deliberately NOT exposed as a field on RebalancerRecommendation. It exists solely to
  support this filtering. See InventoryRebalancerOutput's docstring in schemas.py for why
  top_priority_skus has no strict Pydantic consistency validator (unlike high_risk_skus).

**Artifact paths and formats:**
  Models/risk_detector&Inventory_rebalencer/rebalancer_xgboost.pkl — joblib pickle of xgb.XGBRegressor.
  Models/risk_detector&Inventory_rebalencer/rebalancer_feature_columns.pkl — joblib pickle of list[str], 13 items.
  Models/risk_detector&Inventory_rebalencer/rebalancer_imputer.pkl — joblib pickle of fitted sklearn.impute.SimpleImputer(strategy='median').
  Do not load: rebalancer_urgency_weights.pkl (informational only), rebalancer_best_params.pkl (informational),
  rebalancer_metrics.json (informational), rebalancer_linear_baseline.pkl (baseline, not production),
  stacking_proposed.pkl (that's the Risk Detector's model, never loaded by this module).

**Non-goals:**
  No LangChain/LangGraph imports or @tool decorators. No training, grid search, plotting, SHAP, ablation.
  No changes to any DuckDB data-access layer (none exists). No retrain/regeneration of model artifacts.
  No loading of stacking_proposed.pkl. No modification of the original notebook.
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
    InventoryRebalancerError,
    InventoryRebalancerInput,
    InventoryRebalancerOutput,
    RebalancerRecommendation,
    RebalancerModelLoadError,
    RebalancerSkuNotFoundError,
    RebalancerMissingFeatureColumnsError,
    MissingBackorderProbabilityError,
)

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants (documentation/comparison; loaded artifact is runtime truth)
# ──────────────────────────────────────────────────────────────────────────

# Source-of-truth documentation constant. The loaded artifact's list is the
# actual runtime source of truth — see load_rebalancer_artifacts(). This
# constant exists for comparison/documentation and must match the artifact
# as of the last confirmed inspection (see module docstring item #3).
REBALANCER_FEATURES: list[str] = [
    'national_inv', 'min_bank', 'oe_constraint', 'months_of_stock_left',
    'inv_depletion_rate', 'safety_stock_urgency', 'backorder_pressure',
    'inventory_health_net', 'in_transit_qty', 'lead_time', 'coverage_ratio',
    'transit_coverage', 'backorder_probability',
]

# All 15 engineered columns computed by _compute_engineered_features(), in
# dependency order. Only the 7 that also appear in REBALANCER_FEATURES are
# ever selected into the model's X matrix; the remaining 8 are computed for
# notebook fidelity / critical_state's internal use only.
ENGINEERED_COLS: list[str] = [
    'inventory_health_net', 'sales_spike_ratio', 'months_of_stock_left',
    'backorder_pressure', 'supplier_risk_score', 'forecast_accuracy_gap',
    'inv_depletion_rate', 'lead_time_volatility', 'safety_stock_urgency',
    'performance_trend', 'demand_forecast_ratio', 'transit_coverage',
    'coverage_ratio', 'replenishment_gap', 'critical_state',
]

# All 15 raw input columns required to compute the engineered features.
REQUIRED_RAW_COLS: list[str] = [
    'national_inv', 'in_transit_qty', 'sales_1_month', 'sales_3_month',
    'sales_9_month', 'pieces_past_due', 'deck_risk', 'oe_constraint',
    'ppap_risk', 'forecast_3_month', 'lead_time', 'perf_gap', 'min_bank',
    'perf_6_month_avg', 'perf_12_month_avg',
]


@dataclass(frozen=True)
class LoadedRebalancerModel:
    """Immutable container for the loaded rebalancer model + metadata."""
    model: Any
    feature_columns: list[str]
    imputer: Any
    model_version: str


def _detect_gpu_available() -> bool:
    """
    Auto-detect GPU via torch, with nvidia-smi subprocess fallback.
    Mirrors the notebook's Cell 4 logic (same as risk_detector_tool.py).
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
def load_rebalancer_artifacts(models_dir: str | None = None) -> LoadedRebalancerModel:
    """
    Load rebalancer_xgboost.pkl, rebalancer_feature_columns.pkl, and
    rebalancer_imputer.pkl (cached, singleton per process).

    Never loads rebalancer_urgency_weights.pkl, rebalancer_best_params.pkl,
    rebalancer_metrics.json, rebalancer_linear_baseline.pkl, or
    stacking_proposed.pkl.

    Args:
        models_dir: Optional absolute path to the artifact directory.
                    If None, defaults to <repo_root>/Models/risk_detector&Inventory_rebalencer.

    Returns:
        LoadedRebalancerModel(model, feature_columns, imputer, model_version).

    Raises:
        RebalancerModelLoadError: If any required artifact cannot be loaded or is malformed.
    """
    if models_dir is None:
        repo_root = Path(__file__).parent.parent
        models_dir = (
            repo_root / 'Models' / 'risk_detector&Inventory_rebalencer'
        )
    else:
        models_dir = Path(models_dir)

    model_path = models_dir / 'rebalancer_xgboost.pkl'
    feature_cols_path = models_dir / 'rebalancer_feature_columns.pkl'
    imputer_path = models_dir / 'rebalancer_imputer.pkl'

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RebalancerModelLoadError(
            f"Failed to load rebalancer model from {model_path}: {e}"
        ) from e

    try:
        feature_columns = joblib.load(feature_cols_path)
    except Exception as e:
        raise RebalancerModelLoadError(
            f"Failed to load rebalancer feature columns from {feature_cols_path}: {e}"
        ) from e

    try:
        imputer = joblib.load(imputer_path)
    except Exception as e:
        raise RebalancerModelLoadError(
            f"Failed to load rebalancer imputer from {imputer_path}: {e}"
        ) from e

    if not isinstance(feature_columns, list):
        raise RebalancerModelLoadError(
            f"Expected rebalancer_feature_columns.pkl to contain a list, got {type(feature_columns)}"
        )

    # Compare loaded feature_columns against module constant; warn if diverged (but don't raise).
    if feature_columns != REBALANCER_FEATURES:
        warnings.warn(
            f"Loaded rebalancer_feature_columns.pkl differs from the REBALANCER_FEATURES "
            f"module constant: artifact={feature_columns!r} vs constant={REBALANCER_FEATURES!r}. "
            f"Using the artifact's list.",
            stacklevel=2
        )

    # Defensive fix for XGBoost GPU-trained boosters: force CPU if no GPU present.
    gpu_available = _detect_gpu_available()
    if not gpu_available:
        try:
            model.get_booster().set_param({'device': 'cpu'})
        except Exception:
            pass  # Silently skip if not applicable (model doesn't have a booster method).

    return LoadedRebalancerModel(
        model=model,
        feature_columns=feature_columns,
        imputer=imputer,
        model_version='xgboost_urgency_v1'
    )


def _compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact 15-column feature engineering from
    Inventory_Rebalencer.ipynb Cell 9.

    Input df must have all 15 raw columns in REQUIRED_RAW_COLS.
    Output adds all 15 ENGINEERED_COLS.

    Unlike risk_detector_tool.py's create_features() (which replaces inf with 0),
    this function replaces inf with NaN, matching the notebook's cleaning-pass
    scope on ENGINEERED_COLS but NOT its subsequent global median-fill step
    (that step's medians were never persisted as an artifact — see module
    docstring's resolved-gap section). NaN in the 7 engineered columns that
    overlap with REBALANCER_FEATURES is handled downstream by the loaded
    rebalancer_imputer.pkl; NaN in the remaining 8 is left untouched and never
    reaches the model.
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

    # Replace infinite values with NaN (per the notebook's cleaning-pass scope,
    # but stop short of the global median-fill step whose medians were never persisted).
    d[ENGINEERED_COLS] = d[ENGINEERED_COLS].replace([np.inf, -np.inf], np.nan)

    return d


def _prepare_feature_row(
    sku: str,
    row_dict: dict[str, Any],
    backorder_probability: float,
    feature_columns: list[str],
) -> tuple[pd.Series, int]:
    """
    Prepare a single SKU's model-input feature row.

    1. Checks all REQUIRED_RAW_COLS are present in row_dict.
    2. Runs _compute_engineered_features() to derive the 15 engineered cols.
    3. Merges in the caller-supplied backorder_probability for this SKU.
    4. Reindexes to feature_columns order (the loaded artifact's list).
    5. Returns a tuple of (reindexed_feature_row, critical_state_scalar).

    Returns:
        Tuple[pd.Series, int]: (model_input_row_with_NaN_intact, critical_state_flag).
        The critical_state flag is returned here to avoid re-computing it in
        rebalance_inventory() — it's used downstream for top_priority_skus filtering.

    Raises:
        RebalancerMissingFeatureColumnsError: if any required raw column is absent,
            or if feature engineering fails to produce a column needed by feature_columns.
    """
    missing = set(REQUIRED_RAW_COLS) - set(row_dict.keys())
    if missing:
        raise RebalancerMissingFeatureColumnsError(
            f"SKU {sku}: missing required raw columns {missing}"
        )

    # Start with raw columns.
    row_series = pd.Series(row_dict)

    # Apply feature engineering (adds all 15 ENGINEERED_COLS).
    df_temp = pd.DataFrame([row_series])
    df_engineered = _compute_engineered_features(df_temp)
    engineered_series = df_engineered.iloc[0]

    # Extract critical_state before reindexing (we need it separately for filtering).
    critical_state = int(engineered_series['critical_state'])

    # Merge in backorder_probability.
    engineered_series['backorder_probability'] = backorder_probability

    # Reindex to the persisted feature_columns order, raising KeyError if any column is missing.
    try:
        final_row = engineered_series[feature_columns]
    except KeyError as e:
        missing_engineered = set(feature_columns) - set(engineered_series.index)
        raise RebalancerMissingFeatureColumnsError(
            f"SKU {sku}: feature engineering failed to produce {missing_engineered}"
        ) from e

    return final_row, critical_state


def rebalance_inventory(input_data: InventoryRebalancerInput) -> InventoryRebalancerOutput:
    """
    Predict urgency scores and recommended replenishment quantities for the given SKUs.

    Args:
        input_data: InventoryRebalancerInput with skus, data, and backorder_probability.

    Returns:
        InventoryRebalancerOutput with recommendations (sorted by batch_rank),
        top_priority_skus, and model_version.

    Raises:
        RebalancerSkuNotFoundError: if any SKU in input_data.skus is missing from
            input_data.data (defensive re-check; Pydantic's validate_sku_coverage
            already guarantees this at construction time).
        MissingBackorderProbabilityError: if any SKU in input_data.skus is missing
            from input_data.backorder_probability (same defensive re-check rationale).
        RebalancerMissingFeatureColumnsError: if required raw columns are absent
            for any SKU.
        RebalancerModelLoadError: if artifacts cannot be loaded (first call only,
            then cached).
    """
    loaded = load_rebalancer_artifacts()

    # Defensive re-checks (belt-and-suspenders, Pydantic validators already enforce these).
    missing_skus = set(input_data.skus) - set(input_data.data.keys())
    if missing_skus:
        raise RebalancerSkuNotFoundError(
            f"Requested SKUs not found in data dict: {missing_skus}"
        )

    missing_bo = set(input_data.skus) - set(input_data.backorder_probability.keys())
    if missing_bo:
        raise MissingBackorderProbabilityError(
            f"Requested SKUs not found in backorder_probability dict: {missing_bo}"
        )

    # Prepare feature matrix for all SKUs, preserving input order (for batch_rank tie-breaking).
    feature_rows = []
    critical_states = []
    for sku in input_data.skus:
        row, critical_state = _prepare_feature_row(
            sku,
            input_data.data[sku],
            input_data.backorder_probability[sku],
            loaded.feature_columns
        )
        feature_rows.append(row)
        critical_states.append(critical_state)

    X = pd.DataFrame(feature_rows)

    # Ensure column order matches loaded feature_columns exactly (defensive reindex).
    X = X[loaded.feature_columns]

    # Inference: impute, then predict.
    X_imputed = pd.DataFrame(
        loaded.imputer.transform(X),
        columns=loaded.feature_columns,
        index=X.index
    )
    raw_predictions = loaded.model.predict(X_imputed)
    urgency_scores = np.clip(raw_predictions, 0.0, 1.0)

    # Compute recommended_qty from raw (non-imputed) values.
    recommended_qtys = []
    for sku in input_data.skus:
        min_bank = input_data.data[sku].get('min_bank', np.nan)
        national_inv = input_data.data[sku].get('national_inv', np.nan)
        # Use np.clip to handle NaN correctly (vs Python's max).
        qty = np.clip(float(min_bank) - float(national_inv), 0.0, None)
        recommended_qtys.append(qty)

    # Compute batch_rank: descending urgency, ties broken by original position (stable sort).
    order = np.argsort(-urgency_scores, kind='stable')
    batch_ranks = np.empty(len(urgency_scores), dtype=int)
    for i, idx in enumerate(order):
        batch_ranks[idx] = i + 1

    # Build RebalancerRecommendation objects.
    recommendations = []
    for sku, urgency, qty, rank in zip(
        input_data.skus, urgency_scores, recommended_qtys, batch_ranks
    ):
        rec = RebalancerRecommendation(
            sku=sku,
            urgency_score_predicted=float(urgency),
            recommended_qty=float(qty),
            batch_rank=int(rank),
            manufacture_rank=None
        )
        recommendations.append(rec)

    # Compute top_priority_skus: recommended_qty > 0 AND critical_state == 1.
    top_priority_skus = [
        sku for sku, rec, cs in zip(input_data.skus, recommendations, critical_states)
        if rec.recommended_qty > 0 and cs == 1
    ]

    # Sort recommendations by batch_rank before returning.
    recommendations.sort(key=lambda r: r.batch_rank)

    return InventoryRebalancerOutput(
        recommendations=recommendations,
        top_priority_skus=top_priority_skus,
        model_version=loaded.model_version
    )
