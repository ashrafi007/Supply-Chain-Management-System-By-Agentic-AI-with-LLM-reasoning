"""
Pydantic v2 schemas for the Risk Detector inference module.
Shared across multiple downstream agents (Inventory Rebalancer, Forecast Optimizer, etc.).
"""

from typing import Any
from pydantic import BaseModel, field_validator, model_validator


class RiskDetectorInput(BaseModel):
    """
    Input to predict_risk().

    `data` is a dict mapping SKU -> feature dict, e.g.:
        {
            "SKU-001": {"national_inv": 100, "lead_time": 5, ...},
            "SKU-002": {"national_inv": 50, "lead_time": 3, ...},
        }

    This format allows O(1) per-SKU lookup, requires no pandas/numpy on the
    caller's side, and is JSON-serializable. Equivalent to:
        df.set_index('sku').to_dict(orient='index')
    """
    skus: list[str]
    data: dict[str, dict[str, Any]]

    @model_validator(mode='after')
    def validate_sku_coverage(self):
        missing = set(self.skus) - set(self.data.keys())
        if missing:
            raise ValueError(f"SKUs in list but not in data dict: {missing}")
        return self


class RiskDetectorPrediction(BaseModel):
    """Single SKU's risk prediction."""
    sku: str
    backorder_probability: float  # [0.0, 1.0]
    predicted_label: bool  # probability >= 0.5 (standard classification threshold)
    is_high_risk: bool  # probability >= threshold_used (e.g. 0.945)

    @field_validator('backorder_probability')
    @classmethod
    def validate_probability(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Probability must be in [0.0, 1.0], got {v}")
        return v


class RiskDetectorOutput(BaseModel):
    """
    Output of predict_risk(). high_risk_skus is derived from predictions,
    never set independently, ensuring consistency.
    """
    predictions: list[RiskDetectorPrediction]
    high_risk_skus: list[str]
    model_version: str
    threshold_used: float

    @field_validator('high_risk_skus')
    @classmethod
    def validate_high_risk_skus_consistency(cls, v, info):
        if 'predictions' in info.data:
            expected = {p.sku for p in info.data['predictions'] if p.is_high_risk}
            actual = set(v)
            if expected != actual:
                raise ValueError(
                    f"high_risk_skus {actual} inconsistent with predictions "
                    f"(expected {expected})"
                )
        return v


# Custom exceptions for the inference module
class RiskDetectorError(Exception):
    """Base exception for all Risk Detector errors."""
    pass


class ModelLoadError(RiskDetectorError):
    """Failed to load the trained model from disk."""
    pass


class SkuNotFoundError(RiskDetectorError):
    """Requested SKU not present in input data."""
    pass


class MissingFeatureColumnsError(RiskDetectorError):
    """Required feature column(s) missing from input row."""
    pass


# ──────────────────────────────────────────────────────────────────────────
# Inventory Rebalancer schemas (Agent 3) — pure inference, no LLM dependencies
# ──────────────────────────────────────────────────────────────────────────


class InventoryRebalancerInput(BaseModel):
    """
    Input to rebalance_inventory().

    `data` maps SKU -> raw feature dict. Each dict must contain all 15 raw
    columns needed to compute the engineered features:
        national_inv, in_transit_qty, sales_1_month, sales_3_month,
        sales_9_month, pieces_past_due, deck_risk, oe_constraint, ppap_risk,
        forecast_3_month, lead_time, perf_gap, min_bank, perf_6_month_avg,
        perf_12_month_avg.
    (`perf_gap` is a raw, precomputed column loaded from Processed_Dataset.pkl
    upstream; its origin formula is not recoverable from this repo — see the
    module docstring in inventory_rebalancer_tool.py.)

    `backorder_probability` maps SKU -> probability in [0.0, 1.0], REQUIRED,
    supplied by the caller/orchestrator from the Risk Detector tool's output.
    This module never computes it itself and never loads stacking_proposed.pkl.

    Same dict-of-dicts interchange format as RiskDetectorInput, for the same
    reasons (O(1) lookup, no pandas required from the caller, JSON-serializable).
    """
    skus: list[str]
    data: dict[str, dict[str, Any]]
    backorder_probability: dict[str, float]

    @model_validator(mode='after')
    def validate_sku_coverage(self):
        missing_data = set(self.skus) - set(self.data.keys())
        if missing_data:
            raise ValueError(f"SKUs in list but not in data dict: {missing_data}")
        missing_bo = set(self.skus) - set(self.backorder_probability.keys())
        if missing_bo:
            raise ValueError(
                f"SKUs in list but not in backorder_probability dict: {missing_bo}"
            )
        return self


class RebalancerRecommendation(BaseModel):
    """Single SKU's rebalancing recommendation."""
    sku: str
    urgency_score_predicted: float       # clipped to [0.0, 1.0] at inference time
    recommended_qty: float               # clip(min_bank - national_inv, 0); >= 0
    batch_rank: int                      # rank within this call's batch only, 1 = most urgent
    manufacture_rank: int | None = None  # global rank; always None in this iteration

    @field_validator('urgency_score_predicted')
    @classmethod
    def validate_urgency_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"urgency_score_predicted must be in [0.0, 1.0], got {v}")
        return v

    @field_validator('recommended_qty')
    @classmethod
    def validate_recommended_qty(cls, v):
        # NaN is intentionally allowed through: NaN < 0 evaluates to False,
        # so the validator naturally passes NaN without raising. This matches
        # the notebook-faithful behavior where raw min_bank/national_inv NaN
        # propagates through recommended_qty = clip(min_bank - national_inv, 0).
        if not (v >= 0 or (isinstance(v, float) and v != v)):  # v != v is True for NaN
            raise ValueError(f"recommended_qty must be >= 0, got {v}")
        return v


class InventoryRebalancerOutput(BaseModel):
    """
    Output of rebalance_inventory().

    `top_priority_skus` = SKUs where recommended_qty > 0 AND an internally
    computed critical_state flag (see inventory_rebalancer_tool.py's
    _compute_engineered_features) is 1. Unlike RiskDetectorOutput's
    high_risk_skus, this field does NOT carry a strict recomputation
    field_validator, because critical_state is not exposed as a field on
    RebalancerRecommendation (it is computed and consumed internally only,
    per the resolved design decision — see the module docstring in
    inventory_rebalancer_tool.py) and therefore cannot be re-derived from
    `recommendations` alone at the schema-validation layer. This is an
    accepted, intentional limitation relative to the RiskDetectorOutput.high_risk_skus
    pattern: top_priority_skus is trusted as constructed by
    rebalance_inventory() rather than independently re-verified by Pydantic.
    """
    recommendations: list[RebalancerRecommendation]  # sorted by batch_rank ascending
    top_priority_skus: list[str]
    model_version: str


# Custom exceptions for the Inventory Rebalancer inference module
class InventoryRebalancerError(Exception):
    """Base exception for all Inventory Rebalancer errors."""
    pass


class RebalancerModelLoadError(InventoryRebalancerError):
    """Failed to load a required rebalancer artifact from disk."""
    pass


class RebalancerSkuNotFoundError(InventoryRebalancerError):
    """Requested SKU not present in input data."""
    pass


class RebalancerMissingFeatureColumnsError(InventoryRebalancerError):
    """Required raw column(s) missing from input row, or feature engineering failed to produce a required column."""
    pass


class MissingBackorderProbabilityError(InventoryRebalancerError):
    """Requested SKU has no entry in backorder_probability."""
    pass


# ──────────────────────────────────────────────────────────────────────────
# Forecast Optimizer schemas (Agent 3) — pure inference, no LLM dependencies
# ──────────────────────────────────────────────────────────────────────────


class ForecastOptimizerInput(BaseModel):
    """
    Input to optimize_forecast().

    `data` maps SKU -> raw feature dict. Each dict must contain all raw
    columns needed for create_features()/create_agent3_features() (see
    REQUIRED_RAW_COLS in forecast_optimizer_tool.py) — every column of
    Processed_Dataset.pkl's real 26-column schema except `went_on_backorder`.

    `backorder_probability` maps SKU -> probability in [0.0, 1.0], REQUIRED.
    `alarm_triggered` maps SKU -> bool, REQUIRED. Both must come from the
    Risk Detector tool's output via the orchestrator — this module never
    computes them itself and never loads stacking_proposed.pkl or
    optimal_thresholds.pkl.

    Same dict-of-dicts interchange format as RiskDetectorInput /
    InventoryRebalancerInput, for the same reasons (O(1) lookup, no pandas
    required from the caller, JSON-serializable).
    """
    skus: list[str]
    data: dict[str, dict[str, Any]]
    backorder_probability: dict[str, float]
    alarm_triggered: dict[str, bool]

    @model_validator(mode='after')
    def validate_sku_coverage(self):
        missing_data = set(self.skus) - set(self.data.keys())
        if missing_data:
            raise ValueError(f"SKUs in list but not in data dict: {missing_data}")
        missing_bo = set(self.skus) - set(self.backorder_probability.keys())
        if missing_bo:
            raise ValueError(
                f"SKUs in list but not in backorder_probability dict: {missing_bo}"
            )
        missing_alarm = set(self.skus) - set(self.alarm_triggered.keys())
        if missing_alarm:
            raise ValueError(
                f"SKUs in list but not in alarm_triggered dict: {missing_alarm}"
            )
        return self


class ForecastOptimizerRecommendation(BaseModel):
    """Single SKU's forecast-correction recommendation."""
    sku: str
    human_forecast_3m: float
    adjusted_forecast_3m: float
    correction_factor: float   # clipped to [0.3, 1.5] at inference time
    bias_detected: bool
    bias_probability: float    # [0.0, 1.0]
    bias_severity: str         # "SEVERE" | "MILD" | "NONE"
    risk_override_applied: bool
    recommendation: str        # "REDUCE_PLAN" | "INCREASE_PLAN" | "HOLD"

    @field_validator('correction_factor')
    @classmethod
    def validate_correction_factor(cls, v):
        if not (0.3 <= v <= 1.5):
            raise ValueError(f"correction_factor must be in [0.3, 1.5], got {v}")
        return v

    @field_validator('bias_probability')
    @classmethod
    def validate_bias_probability(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"bias_probability must be in [0.0, 1.0], got {v}")
        return v

    @field_validator('bias_severity')
    @classmethod
    def validate_bias_severity(cls, v):
        if v not in ('SEVERE', 'MILD', 'NONE'):
            raise ValueError(f"bias_severity must be one of SEVERE/MILD/NONE, got {v!r}")
        return v

    @field_validator('recommendation')
    @classmethod
    def validate_recommendation(cls, v):
        if v not in ('REDUCE_PLAN', 'INCREASE_PLAN', 'HOLD'):
            raise ValueError(
                f"recommendation must be one of REDUCE_PLAN/INCREASE_PLAN/HOLD, got {v!r}"
            )
        return v


class ForecastOptimizerOutput(BaseModel):
    """
    Output of optimize_forecast(). flagged_for_reduction/flagged_for_increase
    are fully derivable from `recommendations` alone, so (unlike
    InventoryRebalancerOutput.top_priority_skus) they carry strict
    consistency field_validators, matching RiskDetectorOutput.high_risk_skus.
    """
    recommendations: list[ForecastOptimizerRecommendation]
    flagged_for_reduction: list[str]   # SKUs with recommendation == REDUCE_PLAN
    flagged_for_increase: list[str]    # SKUs with recommendation == INCREASE_PLAN
    model_version_a: str
    model_version_b: str

    @field_validator('flagged_for_reduction')
    @classmethod
    def validate_flagged_for_reduction(cls, v, info):
        if 'recommendations' in info.data:
            expected = {r.sku for r in info.data['recommendations'] if r.recommendation == 'REDUCE_PLAN'}
            actual = set(v)
            if expected != actual:
                raise ValueError(
                    f"flagged_for_reduction {actual} inconsistent with recommendations "
                    f"(expected {expected})"
                )
        return v

    @field_validator('flagged_for_increase')
    @classmethod
    def validate_flagged_for_increase(cls, v, info):
        if 'recommendations' in info.data:
            expected = {r.sku for r in info.data['recommendations'] if r.recommendation == 'INCREASE_PLAN'}
            actual = set(v)
            if expected != actual:
                raise ValueError(
                    f"flagged_for_increase {actual} inconsistent with recommendations "
                    f"(expected {expected})"
                )
        return v


# Custom exceptions for the Forecast Optimizer inference module
class ForecastOptimizerError(Exception):
    """Base exception for all Forecast Optimizer errors."""
    pass


class ForecastModelLoadError(ForecastOptimizerError):
    """Failed to load a required forecast optimizer artifact from disk."""
    pass


class ForecastSkuNotFoundError(ForecastOptimizerError):
    """Requested SKU not present in input data."""
    pass


class ForecastMissingFeatureColumnsError(ForecastOptimizerError):
    """Required raw column(s) missing from input row, or feature engineering failed to produce a required column."""
    pass


class MissingRiskSignalError(ForecastOptimizerError):
    """Requested SKU has no entry in backorder_probability or alarm_triggered."""
    pass


# ──────────────────────────────────────────────────────────────────────────
# Supplier Auditor schemas (Agent 4) — pure inference, no LLM dependencies
# ──────────────────────────────────────────────────────────────────────────


class SupplierAuditorInput(BaseModel):
    """
    Input to audit_suppliers().

    `data` maps SKU -> raw feature dict. Each dict must contain all 25 raw
    columns in supplier_auditor_tool.REQUIRED_RAW_COLS — every column of
    Processed_Dataset.pkl's 26-column schema except `stop_auto_buy` (the
    target) and `sku` (which is the DataFrame's index in Processed_Dataset.pkl,
    not a column — see supplier_auditor_tool.py's module docstring for why
    this resolves differently than the spec assumed):
        national_inv, lead_time, in_transit_qty, forecast_3_month,
        forecast_6_month, forecast_9_month, sales_1_month, sales_3_month,
        sales_6_month, sales_9_month, min_bank, potential_issue,
        pieces_past_due, perf_6_month_avg, perf_12_month_avg, local_bo_qty,
        deck_risk, oe_constraint, ppap_risk, rev_stop, went_on_backorder,
        inv_velocity, safety_gap, sales_trend, perf_gap.
    (`went_on_backorder` is required because the saved artifact's
    feature_cols includes it as a raw passthrough feature — confirmed by
    directly loading models/supplier_auditor_model.pkl — even though it is
    not needed by any engineer_features() formula.)

    `potential_issue`, `deck_risk`, `oe_constraint`, `ppap_risk`, `rev_stop`
    must arrive already encoded as 0/1 numeric — confirmed against
    Processed_Dataset.pkl's actual contents (see supplier_auditor_tool.py's
    module docstring); this module performs no Yes/No -> 0/1 mapping.

    Same dict-of-dicts interchange format as RiskDetectorInput /
    InventoryRebalancerInput / ForecastOptimizerInput, for the same reasons
    (O(1) lookup, no pandas required from the caller, JSON-serializable).
    """
    skus: list[str]
    data: dict[str, dict[str, Any]]

    @model_validator(mode='after')
    def validate_sku_coverage(self):
        missing = set(self.skus) - set(self.data.keys())
        if missing:
            raise ValueError(f"SKUs in list but not in data dict: {missing}")
        return self


class SupplierAuditResult(BaseModel):
    """Single SKU's supplier audit result."""
    sku: str
    risk_probability: float        # [0.0, 1.0], rounded to 4 decimals
    supplier_grade: str            # "A" | "B" | "C" | "D"
    stop_auto_buy_triggered: bool
    trigger_reason: str            # "MODEL+RULE" | "MODEL" | "RULE" | "NONE"
    delivery_stress: float         # exposed since it drives the rule flag

    @field_validator('risk_probability')
    @classmethod
    def validate_risk_probability(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"risk_probability must be in [0.0, 1.0], got {v}")
        return v

    @field_validator('supplier_grade')
    @classmethod
    def validate_supplier_grade(cls, v):
        if v not in ('A', 'B', 'C', 'D'):
            raise ValueError(f"supplier_grade must be one of A/B/C/D, got {v!r}")
        return v

    @field_validator('trigger_reason')
    @classmethod
    def validate_trigger_reason(cls, v):
        if v not in ('MODEL+RULE', 'MODEL', 'RULE', 'NONE'):
            raise ValueError(
                f"trigger_reason must be one of MODEL+RULE/MODEL/RULE/NONE, got {v!r}"
            )
        return v


class SupplierAuditorOutput(BaseModel):
    """
    Output of audit_suppliers(). flagged_suppliers and grade_distribution are
    fully derivable from `results` alone, so (matching RiskDetectorOutput's
    high_risk_skus / ForecastOptimizerOutput's flagged_for_* pattern) they
    carry strict consistency field_validators.
    """
    results: list[SupplierAuditResult]   # sorted by risk_probability, descending
    flagged_suppliers: list[str]         # SKUs where stop_auto_buy_triggered is True
    grade_distribution: dict[str, int]   # count per grade; always has keys A/B/C/D
    model_version: str                   # e.g. "xgboost_supplier_auditor_v1"
    threshold_used: float                # OPT_THRESH from the loaded artifact

    @field_validator('flagged_suppliers')
    @classmethod
    def validate_flagged_suppliers(cls, v, info):
        if 'results' in info.data:
            expected = {r.sku for r in info.data['results'] if r.stop_auto_buy_triggered}
            actual = set(v)
            if expected != actual:
                raise ValueError(
                    f"flagged_suppliers {actual} inconsistent with results "
                    f"(expected {expected})"
                )
        return v

    @field_validator('grade_distribution')
    @classmethod
    def validate_grade_distribution(cls, v, info):
        if 'results' in info.data:
            expected: dict[str, int] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
            for r in info.data['results']:
                expected[r.supplier_grade] += 1
            if v != expected:
                raise ValueError(
                    f"grade_distribution {v} inconsistent with results (expected {expected})"
                )
        return v


# Custom exceptions for the Supplier Auditor inference module
class SupplierAuditorError(Exception):
    """Base exception for all Supplier Auditor errors."""
    pass


class SupplierAuditorModelLoadError(SupplierAuditorError):
    """Failed to load the supplier auditor artifact from disk, or it is malformed."""
    pass


class SupplierAuditorSkuNotFoundError(SupplierAuditorError):
    """Requested SKU not present in input data."""
    pass


class SupplierAuditorMissingFeatureColumnsError(SupplierAuditorError):
    """Required raw column(s) missing from input row, or feature engineering failed to produce a required column."""
    pass
