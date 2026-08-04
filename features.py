"""
features.py — shared serve-time feature engineering for the agent wrappers.

This module is the single place where raw SKU columns are turned into the
engineered columns the trained models expect. Each agent's wrapper imports the
piece it needs from here so that the *exact same* value-level formulas run at
serve time as ran in the training notebooks. Refitting or re-deriving these
formulas differently at serve time is the classic train/serve skew bug.

Agent 1 (Demand Predictor) surface:
  - ``EPS``
  - ``engineer_demand_features(df)``  -> replicates Demand-Predictor notebook cell 12
  - ``STAGE_SPECS``                   -> per-horizon target + baseline construction rule
  - ``build_demand_matrix(...)``      -> raw row(s) -> (X in FEATURE_COLS order, baseline b)
  - ``to_delta`` / ``from_delta``     -> the log-ratio correction head (notebook cell 16)

Nothing here reads the database or the network; it operates purely on the raw
column values handed to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Matches the notebook's EPS exactly. Changing this changes every ratio feature
# and silently breaks parity, so it is defined once and reused everywhere.
EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared value-level cleaning (database_spec.md §5 "value-level fixes" bucket)
# ---------------------------------------------------------------------------

# The six Yes/No-style flag columns on inventory_current (src/db/columns.py classifies
# these as INTEGER — the DB stores the cleaned 0/1 form, never the raw "Yes"/"No" text).
FLAG_COLS: list[str] = [
    "potential_issue", "deck_risk", "oe_constraint", "ppap_risk",
    "stop_auto_buy", "rev_stop",
]

# Every other raw numeric column stored on inventory_current (spec §3.3).
NUMERIC_RAW_COLS: list[str] = [
    "national_inv", "lead_time", "in_transit_qty",
    "forecast_3_month", "forecast_6_month", "forecast_9_month",
    "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month",
    "min_bank", "pieces_past_due", "perf_6_month_avg", "perf_12_month_avg",
    "local_bo_qty",
]

_FLAG_TOKEN_MAP: dict[str, float] = {
    "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
    "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0,
}


def clean_raw_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Value-level cleaning for the raw columns stored on ``inventory_current``:
    Yes/No -> 1/0 on the flag columns, ``inf``/``-inf`` -> NaN, numeric type
    coercion. This is the "value-level fixes" bucket from database_spec.md §5
    (operates on a single row in isolation; no cross-row statistics).

    Deliberately does **not** impute. `InventoryCurrent` columns are nullable
    by design (models.py) precisely so that missingness survives into storage;
    each agent applies its own (documented, sometimes model-specific) fill
    strategy at serve time. Fabricating a median here would be cleaning that
    no agent actually replicates, defeating the "no-op on seeded rows, real
    work on live data" contract this function exists to satisfy.

    No row dropping, no fitted artifacts. Columns not in ``FLAG_COLS`` or
    ``NUMERIC_RAW_COLS`` (e.g. ``sku``) pass through untouched.
    """
    d = df.copy()

    for col in FLAG_COLS:
        if col not in d.columns:
            continue
        tokens = d[col].map(lambda v: _FLAG_TOKEN_MAP.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
        d[col] = tokens

    numeric_cols = [c for c in (*FLAG_COLS, *NUMERIC_RAW_COLS) if c in d.columns]
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d[numeric_cols] = d[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return d


# ---------------------------------------------------------------------------
# Agent 1 — Demand Predictor
# ---------------------------------------------------------------------------

# Raw columns that engineer_demand_features() consumes directly. These must be
# present on the incoming row(s) — never fabricated. `safety_gap`, `perf_gap`,
# `inv_velocity` and `sales_trend` are *precomputed-upstream* columns (they came
# baked into the training Processed_Dataset, not raw from the source CSV); the
# Demand-Predictor model treats them as ordinary inputs, so they are required
# inputs here too — exactly as they are in the Risk Detector tool.
DEMAND_RAW_INPUT_COLS: list[str] = [
    # source-CSV raw
    "national_inv", "lead_time", "in_transit_qty", "min_bank",
    "perf_6_month_avg", "perf_12_month_avg", "local_bo_qty",
    "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy", "rev_stop",
    "forecast_3_month", "sales_1_month", "sales_3_month",
    # precomputed-upstream (present in the training dataset, treated as inputs)
    "safety_gap", "perf_gap", "inv_velocity", "sales_trend",
]


def engineer_demand_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Verbatim replica of the Demand-Predictor notebook's ``engineer_features``
    (cell 12). Adds the engineered columns the model may reference. No row
    dropping, no imputation, no clipping beyond what the notebook itself does.

    The formulas — denominators, the ``+ EPS`` / ``+ 1.0`` offsets, the
    ``clip(lower=0)`` inside ``log_national_inv`` — must not be "tidied up":
    they are load-bearing for parity.
    """
    d = d.copy()
    d["velocity_1m"]      = d["sales_1_month"] / 30.0
    d["velocity_3m"]      = d["sales_3_month"] / 90.0
    d["accel_3_vs_1"]     = d["velocity_3m"] / (d["velocity_1m"] + EPS)
    d["cover_days_1m"]    = d["national_inv"] / (d["velocity_1m"] + EPS)
    d["transit_cover"]    = d["in_transit_qty"] / (d["velocity_1m"] + EPS)
    d["bo_pressure"]      = d["local_bo_qty"] / (d["sales_1_month"] + 1.0)
    d["inv_to_min_bank"]  = d["national_inv"] / (d["min_bank"] + EPS)
    d["lead_time_demand"] = d["lead_time"] * d["velocity_1m"]
    d["forecast_bias_3m"] = d["forecast_3_month"] - d["sales_3_month"]
    d["log_sales_1m"]     = np.log1p(d["sales_1_month"])
    d["log_national_inv"] = np.log1p(d["national_inv"].clip(lower=0))
    d["risk_flag_count"]  = d[["deck_risk", "oe_constraint", "ppap_risk", "rev_stop"]].sum(axis=1)
    return d


# Per-horizon spec. `baseline_fn` is the *executable* form of the notebook's
# `baseline_name` string ("sales_3_month x 2" for h6), so the anchor `b` the
# log-ratio delta is measured against is reconstructed identically at serve time.
# Only h6 ships a trained bundle today (see the Demand Predictor wrapper spec §2,
# Option A); h3/h9 are recorded for completeness and future bundles.
STAGE_SPECS: dict[str, dict] = {
    "h3": dict(target="sales_3_month", baseline_name="sales_1_month x 3",
               baseline_fn=lambda d: d["sales_1_month"] * 3.0,
               raw_extra=["sales_1_month"]),
    "h6": dict(target="sales_6_month", baseline_name="sales_3_month x 2",
               baseline_fn=lambda d: d["sales_3_month"] * 2.0,
               raw_extra=["sales_1_month", "sales_3_month"]),
    "h9": dict(target="sales_9_month", baseline_name="sales_6_month x 1.5",
               baseline_fn=lambda d: d["sales_6_month"] * 1.5,
               raw_extra=["sales_1_month", "sales_3_month", "sales_6_month"]),
}


def build_demand_matrix(
    raw: pd.DataFrame,
    feature_cols: list[str],
    stage: str = "h6",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Turn raw row(s) into the model's feature matrix and its baseline anchor.

    Returns ``(X, b)`` where:
      - ``X`` has exactly ``feature_cols`` as columns, in that order, as float64.
      - ``b`` is the naive baseline for this stage (the delta anchor).

    ``base_pred`` and ``log_base_pred`` are injected here (they are not produced
    by ``engineer_demand_features``); every other requested column is taken from
    the engineered frame. Requesting a column that cannot be produced raises
    ``KeyError`` rather than silently dropping it — a wrong/short feature matrix
    corrupts LightGBM input without erroring on its own.
    """
    spec = STAGE_SPECS[stage]
    eng = engineer_demand_features(raw)

    b = spec["baseline_fn"](raw).astype(float)
    eng = eng.copy()
    eng["base_pred"] = b.to_numpy()
    eng["log_base_pred"] = np.log1p(b.clip(lower=0)).to_numpy()

    missing = [c for c in feature_cols if c not in eng.columns]
    if missing:
        raise KeyError(f"cannot build feature columns {missing} for stage {stage}")

    X = eng[feature_cols].astype(np.float64)
    return X, b.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Serve-time gap-fill: safety_gap / perf_gap / inv_velocity / sales_trend /
# went_on_backorder — required raw inputs by every one of the five agent tool
# modules, but absent from inventory_current (they only existed precomputed in
# an external Processed_Dataset.pkl that isn't part of this repo). The first
# four are genuine engineered features; their exact formulas were recovered by
# testing candidate expressions against a reference copy of that precomputed
# dataset (1.93M rows) and confirming a bit-exact match (max abs diff ~1e-7,
# float-precision only) across every row — not guesses.
# ---------------------------------------------------------------------------

def compute_safety_gap(national_inv: pd.Series, min_bank: pd.Series) -> pd.Series:
    """national_inv - min_bank. Verified bit-exact against 1.93M reference rows."""
    return national_inv - min_bank


def compute_perf_gap(perf_6_month_avg: pd.Series, perf_12_month_avg: pd.Series) -> pd.Series:
    """abs(perf_6_month_avg - perf_12_month_avg). Verified bit-exact against 1.93M reference rows."""
    return (perf_6_month_avg - perf_12_month_avg).abs()


def compute_inv_velocity(sales_1_month: pd.Series, national_inv: pd.Series) -> pd.Series:
    """sales_1_month / (max(national_inv, 0) + 1). Verified bit-exact against 1.93M reference rows."""
    return sales_1_month / (national_inv.clip(lower=0) + 1)


def compute_sales_trend(sales_1_month: pd.Series, sales_3_month: pd.Series) -> pd.Series:
    """sales_1_month / (sales_3_month / 3 + 1). Verified bit-exact against 1.93M reference rows."""
    return sales_1_month / (sales_3_month / 3 + 1)


def fill_engineered_columns(raw_features: dict) -> dict:
    """
    Return a copy of ``raw_features`` with ``safety_gap``, ``perf_gap``,
    ``inv_velocity``, ``sales_trend`` computed via the verified formulas above,
    and ``went_on_backorder`` defaulted to 0, but only for keys currently
    absent — never overwrites a real stored value (future-proof if
    ``inventory_current`` ever gains these columns directly).

    ``went_on_backorder`` is not an engineered feature — it is a genuine
    historical label with no formula (required as a raw passthrough input by
    the Supplier Auditor tool only). It defaults to 0, the majority class
    (~99.3% of the reference dataset), documented here rather than silently
    assumed.
    """
    d = dict(raw_features)
    row = pd.DataFrame([d])

    if "safety_gap" not in d:
        d["safety_gap"] = float(compute_safety_gap(row["national_inv"], row["min_bank"]).iloc[0])
    if "perf_gap" not in d:
        d["perf_gap"] = float(compute_perf_gap(row["perf_6_month_avg"], row["perf_12_month_avg"]).iloc[0])
    if "inv_velocity" not in d:
        d["inv_velocity"] = float(compute_inv_velocity(row["sales_1_month"], row["national_inv"]).iloc[0])
    if "sales_trend" not in d:
        d["sales_trend"] = float(compute_sales_trend(row["sales_1_month"], row["sales_3_month"]).iloc[0])
    if "went_on_backorder" not in d:
        d["went_on_backorder"] = 0

    return d


def to_delta(y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Training target: log-space delta of demand y against baseline b."""
    return np.log1p(np.maximum(y, 0)) - np.log1p(np.maximum(b, 0))


def from_delta(delta: np.ndarray, b: np.ndarray, smear: float = 1.0) -> np.ndarray:
    """
    Invert the log-ratio delta back to units, apply the Duan smearing factor,
    and clip at zero. This is the notebook's exact transform (cell 16):

        y_hat = max( (1 + max(b, 0)) * exp(delta) * smear - 1, 0 )

    Note this is NOT ``expm1(delta + log1p(b)) * smear``: the smear multiplies
    the whole ``(1+b)*exp(delta)`` term *before* the ``-1``, not after. Getting
    that ordering wrong shifts every prediction.
    """
    return np.maximum((1.0 + np.maximum(b, 0.0)) * np.exp(delta) * smear - 1.0, 0.0)
