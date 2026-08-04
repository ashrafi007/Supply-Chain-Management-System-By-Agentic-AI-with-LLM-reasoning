"""
Read-only data-quality audit over the raw CSV (.CLAUDE/csv_quality_audit_spec.md).

Produces findings only — no row is dropped, no value is changed, and no file outside
``audits/`` is touched. Two downstream specs consume this script's output:
  * ``database_spec.md`` §2 — the ``sku_cardinality_resolution`` finding answers the
    blocking Case A/B/C primary-key question.
  * ``features.py``'s (future) cleaning spec — every finding tagged ``serve_relevant``
    becomes a row-safe fix there. This script never fixes anything itself.

Run via:  python -m audits.csv_quality_audit
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Spec's canonical path is data/raw/rsm.csv; this repo's raw dataset lives at
# Dataset/RSM_Dataset.csv (see src/db/columns.py for the same mismatch, solved the same
# way). Overridable via the RSM_RAW_CSV env var.
RAW_CSV_PATH: Path = Path(
    os.environ.get("RSM_RAW_CSV", PROJECT_ROOT / "Dataset" / "RSM_Dataset.csv")
)
REPORT_JSON_PATH: Path = PROJECT_ROOT / "audits" / "reports" / "csv_quality_report.json"
REPORT_MD_PATH: Path = PROJECT_ROOT / "audits" / "reports" / "csv_quality_report.md"

# ── identity / target ────────────────────────────────────────────────────────────

ID_COL: str = "sku"
TARGET_COL: str = "went_on_backorder"

# ── determinism ──────────────────────────────────────────────────────────────────

SEED: int = 42                 # unused today (no sampling); reserved per spec's determinism note
ROUND_NDIGITS: int = 6
EXAMPLES_PER_FINDING: int = 5  # spec: "3-5 concrete examples"

CHECK_GROUP_ORDER: tuple[str, ...] = (
    "structural", "missingness", "numeric", "categorical", "cross_column", "target",
)

# ── domain constants ────────────────────────────────────────────────────────────

# Columns that should never be negative in this domain. perf_6_month_avg / perf_12_month_avg
# are deliberately excluded here — they use -99 as a documented missingness sentinel and are
# handled by the sentinel-reconciliation rule instead, so they are never double-reported.
NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "national_inv", "lead_time", "in_transit_qty",
    "forecast_3_month", "forecast_6_month", "forecast_9_month",
    "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month",
    "min_bank", "pieces_past_due", "local_bo_qty",
)

FLAG_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy", "rev_stop",
)

# Encoded-null literal text tokens (case-insensitive, stripped).
ENCODED_NULL_TEXT_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "unknown", "-", "?", "missing"}
)
# Numeric sentinel candidates named by the spec, checked generically per numeric column.
ENCODED_NULL_NUMERIC_CANDIDATES: tuple[float, ...] = (-1, -9, -99, -999, -9999, 9999, 99999)

# ── thresholds (visible, justified — see plan's threshold table) ───────────────

OUTLIER_MAX_TO_P999_RATIO: float = 20.0
NEAR_UNIQUE_ID_RATIO: float = 0.95
NEAR_UNIQUE_ID_MIN_CARDINALITY: int = 1000
SALES_VS_INVENTORY_IMPLAUSIBLE_RATIO: float = 100.0
SALES_VS_INVENTORY_SENSITIVITY_RATIOS: tuple[int, ...] = (10, 50, 100, 1000)
SENTINEL_DOMINANCE_RATIO: float = 0.95
MISSINGNESS_CLUSTER_THRESHOLD: float = 0.5
LEAKAGE_RATE_GAP_THRESHOLD: float = 0.2  # 20 percentage points, between target classes


# ── generic helpers ──────────────────────────────────────────────────────────────

def _read_raw_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Full, whole-file read (no chunksize — fits memory comfortably at this size and every
    check needs a full-dataset pass anyway). Deliberately does NOT pass low_memory=False,
    since that would suppress the DtypeWarning that §3.1 requires as a structural finding."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {path}. Set the RSM_RAW_CSV env var to point at it."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pd.read_csv(path)
    dtype_warnings = [
        str(w.message) for w in caught if issubclass(w.category, pd.errors.DtypeWarning)
    ]
    return df, dtype_warnings


def _parse_dtype_warning_columns(dtype_warnings: list[str], header: list[str]) -> set[str]:
    """pandas' DtypeWarning names columns by index, e.g. "Columns (0,2) have mixed types.".
    Map those indices back to column names via the CSV header order."""
    cols: set[str] = set()
    for msg in dtype_warnings:
        m = re.search(r"Columns \(([^)]*)\)", msg)
        if not m:
            continue
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(header):
                    cols.add(header[idx])
    return cols


def _is_textlike(s: pd.Series) -> bool:
    """The only correct generic text test under pandas 3.0's dual object/str default dtypes
    (sku reads as legacy object; the other text columns read as the new str extension dtype).
    Never compare dtype == object — that silently misses the str-extension-dtype columns."""
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _round(x: Any) -> Any:
    """Fixed-precision rounding for determinism; NaN -> None, inf -> a JSON-safe string."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if np.isnan(xf):
            return None
        if np.isinf(xf):
            return "Infinity" if xf > 0 else "-Infinity"
        return round(xf, ROUND_NDIGITS)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _json_safe(obj: Any) -> Any:
    """Recursively converts numpy/pandas scalars into native, deterministic, JSON-safe
    values. Applied once to the whole report right before serialization, so individual
    check functions never have to think about numpy dtypes."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _examples(subset: pd.DataFrame, cols: list[str] | None = None,
              n: int = EXAMPLES_PER_FINDING) -> list[dict[str, Any]]:
    """DataFrame rows -> JSON-safe dicts. Never a DataFrame repr/to_string() blob. Callers
    that need a specific order (e.g. worst-violation-first) must sort `subset` beforehand —
    this function only slices the first n rows of whatever order it's given."""
    if cols is not None:
        subset = subset[cols]
    return [_json_safe(r) for r in subset.head(n).to_dict(orient="records")]


def _finding(id_: str, group: str, title: str, description: str, columns: list[str],
             affected_rows: int, total_rows: int, serve_relevant: bool,
             examples: list[dict[str, Any]] | None = None,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical Finding-dict shape, reused identically by all six check groups so
    downstream tooling can grep/query on `serve_relevant` / `columns` without knowing which
    group produced a given finding."""
    affected_pct = round(affected_rows / total_rows * 100, ROUND_NDIGITS) if total_rows else None
    return {
        "id": id_,
        "check_group": group,
        "title": title,
        "description": description,
        "columns": list(columns),
        "affected_rows": int(affected_rows),
        "total_rows": int(total_rows),
        "affected_pct": affected_pct,
        "serve_relevant": bool(serve_relevant),
        "examples": examples or [],
        "extra": extra or {},
    }


def _split_negatives_sentinel_vs_real(
    series: pd.Series,
    candidates: tuple[float, ...] = ENCODED_NULL_NUMERIC_CANDIDATES,
    dominance: float = SENTINEL_DOMINANCE_RATIO,
) -> dict[str, Any]:
    """Reconciles §3.3's instruction not to double-count a missingness sentinel (e.g. -99 in
    perf_*_avg) as both 'negative value' and 'encoded null'. If a single numeric-sentinel
    candidate accounts for >= dominance of a column's negative values, all of them are
    classified as encoded-null (reported once, in audit_missingness) and excluded from the
    audit_numeric 'negative values' finding, which then reports only the residual."""
    neg = series[series < 0]
    total_neg = int(neg.shape[0])
    if total_neg == 0:
        return {"sentinel_value": None, "sentinel_count": 0, "residual_negative_count": 0}

    counts = neg.value_counts()
    best_value: float | None = None
    best_count = 0
    for cand in candidates:
        if cand < 0 and cand in counts.index:
            c = int(counts.loc[cand])
            if c > best_count:
                best_value, best_count = float(cand), c

    if best_value is not None and (best_count / total_neg) >= dominance:
        return {
            "sentinel_value": best_value,
            "sentinel_count": best_count,
            "residual_negative_count": total_neg - best_count,
        }
    return {"sentinel_value": None, "sentinel_count": 0, "residual_negative_count": total_neg}


def _footer_artifact_mask(df: pd.DataFrame) -> pd.Series:
    """True for Kaggle-export footer rows (sku == "(N rows)", every other column blank).
    Regex-based, never hardcodes a specific row count, so it still matches if the CSV is
    regenerated with different totals."""
    sku_str = df[ID_COL].astype(str).str.strip()
    return sku_str.str.match(r"^\(\d+ rows\)$", na=False)


# ── §3.1 structural ──────────────────────────────────────────────────────────────

def audit_structural(df: pd.DataFrame, dtype_warnings: list[str],
                      footer_mask: pd.Series) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    total_rows = len(df)
    n_footer = int(footer_mask.sum())
    n_data = total_rows - n_footer
    data = df.loc[~footer_mask]

    findings.append(_finding(
        id_="structural.row_col_count",
        group="structural",
        title="Row and column counts",
        description=(
            f"CSV has {total_rows} total rows ({n_footer} footer-artifact rows, {n_data} real "
            f"data rows) and {len(df.columns)} columns."
        ),
        columns=[],
        affected_rows=n_footer,
        total_rows=total_rows,
        serve_relevant=False,
        examples=_examples(df.loc[footer_mask]),
        extra={
            "expected_rows_approx": 1_900_000,
            "total_rows": total_rows,
            "footer_artifact_rows": n_footer,
            "data_rows": n_data,
            "column_count": len(df.columns),
        },
    ))

    full_dup_all = int(df.duplicated().sum())
    full_dup_data = int(data.duplicated().sum())
    findings.append(_finding(
        id_="structural.duplicate_full_rows",
        group="structural",
        title="Fully duplicate rows",
        description=(
            f"{full_dup_data} fully identical rows among real data rows "
            f"({full_dup_all} when footer rows are included)."
        ),
        columns=list(df.columns),
        affected_rows=full_dup_data,
        total_rows=n_data,
        serve_relevant=False,
        examples=_examples(data.loc[data.duplicated(keep=False)]),
        extra={"duplicate_count_including_footer_rows": full_dup_all},
    ))

    n_repeated_sku_rows = int(data[ID_COL].duplicated().sum())
    unique_skus = int(data[ID_COL].nunique())

    if data[ID_COL].duplicated().any():
        top_sku = data[ID_COL].value_counts().index[0]
        rep_rows = data.loc[data[ID_COL] == top_sku]
        examples_of_repeated_sku = _examples(rep_rows, n=min(10, len(rep_rows)))
        case = "UNRESOLVED_MANUAL_REVIEW"
        case_note = (
            "Repeated SKUs found. Inspect examples_of_repeated_sku to determine whether "
            "repeats vary by a date/period column (Case B) or a location/plant column "
            "(Case C) — this script does not guess between them; see database_spec.md §2."
        )
    else:
        examples_of_repeated_sku = []
        case = "A"
        case_note = (
            "No repeated SKUs among real data rows (footer-artifact rows excluded). "
            "nunique(sku) == len(data rows) -> Case A. inventory_current PK = sku_id."
        )

    sku_res = {
        "rows": total_rows,
        "footer_artifact_rows": n_footer,
        "data_rows": n_data,
        "unique_skus": unique_skus,
        "fully_identical_rows": full_dup_data,
        "rows_with_repeated_sku": n_repeated_sku_rows,
        "case": case,
        "case_note": case_note,
        "examples_of_repeated_sku": examples_of_repeated_sku,
    }
    findings.append(_finding(
        id_="structural.sku_cardinality_resolution",
        group="structural",
        title="SKU cardinality resolution (database_spec.md §2 Case A/B/C)",
        description=case_note,
        columns=[ID_COL],
        affected_rows=n_repeated_sku_rows,
        total_rows=n_data,
        serve_relevant=False,
        examples=examples_of_repeated_sku,
        extra=sku_res,
    ))

    warned_cols = _parse_dtype_warning_columns(dtype_warnings, list(df.columns))
    per_column_dtype: list[dict[str, Any]] = []
    numeric_looking_nonnumeric: list[str] = []
    for col in df.columns:
        s = df[col]
        is_numeric = bool(pd.api.types.is_numeric_dtype(s))
        non_null = int(s.notna().sum())
        if is_numeric or non_null == 0:
            parse_rate = 1.0 if is_numeric else 0.0
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            parse_rate = float(coerced.notna().sum() / non_null)
        flagged_by_warning = col in warned_cols
        flagged_by_parse = (not is_numeric) and parse_rate > 0.95
        per_column_dtype.append({
            "column": col,
            "dtype": str(s.dtype),
            "is_numeric": is_numeric,
            "flagged_by_dtype_warning": flagged_by_warning,
            "flagged_by_numeric_parse_heuristic": flagged_by_parse,
            "numeric_parse_rate": _round(parse_rate),
        })
        if flagged_by_warning or flagged_by_parse:
            numeric_looking_nonnumeric.append(col)

    findings.append(_finding(
        id_="structural.dtype_as_read",
        group="structural",
        title="Column dtypes as read; numeric-looking columns read as non-numeric",
        description=(
            f"{len(numeric_looking_nonnumeric)} column(s) flagged as numeric-looking but read "
            f"as non-numeric: {numeric_looking_nonnumeric}."
        ),
        columns=numeric_looking_nonnumeric,
        affected_rows=0,
        total_rows=total_rows,
        serve_relevant=True,
        examples=[],
        extra={"per_column": per_column_dtype, "dtype_warnings_captured": dtype_warnings},
    ))

    summary = {
        "total_rows": total_rows,
        "data_rows": n_data,
        "footer_artifact_rows": n_footer,
        "columns": len(df.columns),
        "sku_case": case,
        "finding_count": len(findings),
    }
    return {"summary": summary, "findings": findings, "sku_cardinality_resolution": sku_res}


# ── §3.2 missingness ─────────────────────────────────────────────────────────────

def audit_missingness(df: pd.DataFrame, footer_mask: pd.Series) -> dict[str, Any]:
    data = df.loc[~footer_mask]
    n_data = len(data)
    findings: list[dict[str, Any]] = []
    cols_with_missing: list[str] = []

    for col in df.columns:
        s_all = df[col]
        s_data = data[col]
        explicit_null_all = int(s_all.isna().sum())
        explicit_null_data = int(s_data.isna().sum())
        footer_contribution = explicit_null_all - explicit_null_data

        if explicit_null_data > 0:
            cols_with_missing.append(col)
            findings.append(_finding(
                id_=f"missingness.explicit_null.{col}",
                group="missingness",
                title=f"Explicit missing values in {col}",
                description=(
                    f"{explicit_null_data} of {n_data} data rows are NaN in {col} "
                    f"({footer_contribution} additional NaN rows are footer artifacts, "
                    f"excluded from this count)."
                ),
                columns=[col],
                affected_rows=explicit_null_data,
                total_rows=n_data,
                serve_relevant=True,
                examples=_examples(data.loc[s_data.isna()]),
                extra={"footer_row_contribution": footer_contribution},
            ))

        if _is_textlike(s_data):
            non_null = s_data.dropna().astype(str)
            whitespace_mask = s_data.notna() & (s_data.astype(str).str.strip() == "")
            whitespace_only = int(whitespace_mask.sum())
            normalized = non_null.str.strip().str.lower()
            encoded_mask_normalized = normalized.isin(ENCODED_NULL_TEXT_TOKENS)
            encoded_null_text_count = int(encoded_mask_normalized.sum())
            if encoded_null_text_count > 0:
                encoded_idx = normalized.index[encoded_mask_normalized]
                findings.append(_finding(
                    id_=f"missingness.encoded_null_text.{col}",
                    group="missingness",
                    title=f"Encoded-null text tokens in {col}",
                    description=(
                        f"{encoded_null_text_count} rows contain a placeholder token "
                        f"(e.g. 'unknown', 'n/a', whitespace-only) in {col}."
                    ),
                    columns=[col],
                    affected_rows=encoded_null_text_count,
                    total_rows=n_data,
                    serve_relevant=True,
                    examples=_examples(data.loc[encoded_idx]),
                    extra={"whitespace_only_count": whitespace_only},
                ))
        else:
            non_null = s_data.dropna()
            sentinel = _split_negatives_sentinel_vs_real(non_null)
            if sentinel["sentinel_count"] > 0:
                mask = s_data == sentinel["sentinel_value"]
                findings.append(_finding(
                    id_=f"missingness.encoded_null_numeric.{col}",
                    group="missingness",
                    title=f"Encoded-null numeric sentinel in {col}",
                    description=(
                        f"{sentinel['sentinel_count']} rows use {sentinel['sentinel_value']} as "
                        f"a missingness sentinel in {col} (excluded from numeric-sanity "
                        f"negative-value findings to avoid double-counting)."
                    ),
                    columns=[col],
                    affected_rows=sentinel["sentinel_count"],
                    total_rows=n_data,
                    serve_relevant=True,
                    examples=_examples(data.loc[mask]),
                ))
            for cand in ENCODED_NULL_NUMERIC_CANDIDATES:
                if cand == sentinel.get("sentinel_value"):
                    continue
                mask = s_data == cand
                c = int(mask.sum())
                if c == 0:
                    continue
                findings.append(_finding(
                    id_=f"missingness.encoded_null_numeric_candidate.{col}.{cand}",
                    group="missingness",
                    title=f"Possible numeric sentinel {cand} in {col}",
                    description=(
                        f"{c} rows equal {cand} in {col}, a common missingness-sentinel value; "
                        f"not confirmed dominant enough to reclassify automatically."
                    ),
                    columns=[col],
                    affected_rows=c,
                    total_rows=n_data,
                    serve_relevant=True,
                    examples=_examples(data.loc[mask]),
                ))

    clustering_pairs: list[dict[str, Any]] = []
    for i, c1 in enumerate(cols_with_missing):
        for c2 in cols_with_missing[i + 1:]:
            m1 = data[c1].isna()
            m2 = data[c2].isna()
            union = int((m1 | m2).sum())
            inter = int((m1 & m2).sum())
            jaccard = (inter / union) if union else 0.0
            if jaccard >= MISSINGNESS_CLUSTER_THRESHOLD:
                clustering_pairs.append({
                    "column_a": c1, "column_b": c2,
                    "jaccard": _round(jaccard), "co_missing_rows": inter,
                })
    findings.append(_finding(
        id_="missingness.clustering_pairs",
        group="missingness",
        title="Missingness clustering across column pairs",
        description=(
            f"{len(clustering_pairs)} column pair(s) with NaN co-occurrence (Jaccard) "
            f">= {MISSINGNESS_CLUSTER_THRESHOLD}, among {len(cols_with_missing)} column(s) "
            f"that have any missing values."
        ),
        columns=cols_with_missing,
        affected_rows=0,
        total_rows=n_data,
        serve_relevant=False,
        examples=[],
        extra={"pairs": clustering_pairs},
    ))

    segment_breakdown: dict[str, dict[str, Any]] = {}
    for flag_col in FLAG_CATEGORICAL_COLUMNS:
        if flag_col not in df.columns:
            continue
        for target_col in cols_with_missing:
            if target_col == flag_col:
                continue
            rates = data[target_col].isna().groupby(data[flag_col], dropna=False).mean()
            segment_breakdown[f"{target_col}_by_{flag_col}"] = {
                str(k): _round(v) for k, v in rates.items()
            }
    findings.append(_finding(
        id_="missingness.by_categorical_segment",
        group="missingness",
        title="Missingness rate by categorical segment",
        description=(
            "Per-segment missingness rate for each column with missing values, broken down "
            "by each Yes/No flag column — a clustered pattern (rate far from the column's "
            "overall rate) usually means 'not collected for this segment'."
        ),
        columns=cols_with_missing,
        affected_rows=0,
        total_rows=n_data,
        serve_relevant=False,
        examples=[],
        extra={"breakdown": segment_breakdown},
    ))

    summary = {"columns_with_missing": cols_with_missing, "finding_count": len(findings)}
    return {"summary": summary, "findings": findings}


# ── §3.3 numeric sanity ──────────────────────────────────────────────────────────

def audit_numeric(df: pd.DataFrame, footer_mask: pd.Series) -> dict[str, Any]:
    data = df.loc[~footer_mask]
    n_data = len(data)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    findings: list[dict[str, Any]] = []
    outlier_stats: dict[str, Any] = {}

    for col in numeric_cols:
        s = data[col]
        non_null = s.dropna()
        if non_null.empty:
            continue

        inf_mask = np.isinf(non_null)
        inf_count = int(inf_mask.sum())
        if inf_count > 0:
            idx = non_null.index[inf_mask]
            findings.append(_finding(
                id_=f"numeric.inf.{col}",
                group="numeric",
                title=f"inf/-inf values in {col}",
                description=f"{inf_count} rows have inf or -inf in {col}.",
                columns=[col],
                affected_rows=inf_count,
                total_rows=n_data,
                serve_relevant=True,
                examples=_examples(data.loc[idx]),
            ))

        finite = non_null[np.isfinite(non_null)]
        if finite.empty:
            continue
        p999 = float(finite.quantile(0.999))
        max_v = float(finite.max())
        ratio = (max_v / p999) if p999 else None
        flagged_outlier = bool(ratio is not None and ratio > OUTLIER_MAX_TO_P999_RATIO)
        outlier_stats[col] = {
            "min": _round(finite.min()),
            "max": _round(max_v),
            "mean": _round(finite.mean()),
            "median": _round(finite.median()),
            "p95": _round(finite.quantile(0.95)),
            "p99": _round(finite.quantile(0.99)),
            "p999": _round(p999),
            "zero_pct": _round((finite == 0).sum() / n_data * 100) if n_data else None,
            "max_to_p999_ratio": _round(ratio),
            "flagged_outlier": flagged_outlier,
        }

    findings.append(_finding(
        id_="numeric.outlier_stats",
        group="numeric",
        title="Outlier statistics per numeric column",
        description=(
            f"min/max/mean/median/p95/p99/p99.9 and max:p99.9 ratio for "
            f"{len(outlier_stats)} numeric column(s); "
            f"{sum(1 for v in outlier_stats.values() if v['flagged_outlier'])} flagged "
            f"(ratio > {OUTLIER_MAX_TO_P999_RATIO}x)."
        ),
        columns=list(outlier_stats.keys()),
        affected_rows=0,
        total_rows=n_data,
        serve_relevant=False,
        examples=[],
        extra={"per_column": outlier_stats},
    ))

    for col in NON_NEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        s = data[col]
        non_null = s.dropna()
        sentinel = _split_negatives_sentinel_vs_real(non_null)
        residual = sentinel["residual_negative_count"]
        if residual <= 0:
            continue
        if sentinel["sentinel_value"] is not None:
            mask = (s < 0) & (s != sentinel["sentinel_value"])
        else:
            mask = s < 0
        findings.append(_finding(
            id_=f"numeric.negative_in_nonneg_column.{col}",
            group="numeric",
            title=f"Negative values in {col}",
            description=(
                f"{residual} rows have a negative, non-sentinel value in {col} "
                f"(should be non-negative)."
            ),
            columns=[col],
            affected_rows=residual,
            total_rows=n_data,
            serve_relevant=True,
            examples=_examples(data.loc[mask]),
            extra=sentinel,
        ))

    if "lead_time" in df.columns and "in_transit_qty" in df.columns:
        mask = (data["lead_time"] == 0) & (data["in_transit_qty"] > 0)
        cnt = int(mask.sum())
        findings.append(_finding(
            id_="numeric.impossible_lead_time_zero_with_transit",
            group="numeric",
            title="lead_time == 0 with in_transit_qty > 0",
            description=(
                f"{cnt} rows have lead_time == 0 while in_transit_qty > 0, which is not "
                f"physically sensible (no lead time implies nothing should be in transit)."
            ),
            columns=["lead_time", "in_transit_qty"],
            affected_rows=cnt,
            total_rows=n_data,
            serve_relevant=True,
            examples=_examples(data.loc[mask]),
        ))

    if "sales_1_month" in df.columns and "national_inv" in df.columns:
        sensitivity = {}
        for r in SALES_VS_INVENTORY_SENSITIVITY_RATIOS:
            m = data["sales_1_month"] > (data["national_inv"] * r)
            sensitivity[str(r)] = int(m.sum())
        main_mask = data["sales_1_month"] > (data["national_inv"] * SALES_VS_INVENTORY_IMPLAUSIBLE_RATIO)
        cnt = int(main_mask.sum())
        findings.append(_finding(
            id_="numeric.impossible_sales_exceeds_inventory",
            group="numeric",
            title="sales_1_month implausibly exceeds national_inv",
            description=(
                f"{cnt} rows have sales_1_month > national_inv * "
                f"{SALES_VS_INVENTORY_IMPLAUSIBLE_RATIO:g} (sensitivity at other ratios "
                f"shown in extra.sensitivity_by_ratio)."
            ),
            columns=["sales_1_month", "national_inv"],
            affected_rows=cnt,
            total_rows=n_data,
            serve_relevant=True,
            examples=_examples(data.loc[main_mask]),
            extra={"sensitivity_by_ratio": sensitivity},
        ))

    summary = {"numeric_columns_checked": numeric_cols, "finding_count": len(findings)}
    return {"summary": summary, "findings": findings}


# ── §3.4 categorical / text sanity ──────────────────────────────────────────────

def audit_categorical(df: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    cardinality_summary: dict[str, Any] = {}

    for col in FLAG_CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        raw_distinct = int(s.nunique())
        normalized = s.str.strip().str.lower()
        norm_distinct = int(normalized.nunique())
        top20 = s.value_counts().head(20)
        cardinality_summary[col] = {
            "cardinality_raw": raw_distinct,
            "cardinality_normalized": norm_distinct,
            "inconsistent_encoding_variants": raw_distinct - norm_distinct,
            "top20": {str(k): int(v) for k, v in top20.items()},
        }
        if raw_distinct != norm_distinct:
            variants: dict[str, list[str]] = {}
            for norm_val, group in s.groupby(normalized):
                uniq = sorted(set(group.tolist()))
                if len(uniq) > 1:
                    variants[str(norm_val)] = uniq
            findings.append(_finding(
                id_=f"categorical.inconsistent_encoding.{col}",
                group="categorical",
                title=f"Inconsistent text encoding in {col}",
                description=(
                    f"{raw_distinct - norm_distinct} extra raw distinct value(s) collapse "
                    f"under case/whitespace normalization in {col}."
                ),
                columns=[col],
                affected_rows=0,
                total_rows=len(df),
                serve_relevant=True,
                examples=[],
                extra={"variant_groups": variants},
            ))

    findings.append(_finding(
        id_="categorical.cardinality_summary",
        group="categorical",
        title="Cardinality and top-20 values for flag columns",
        description=(
            f"Cardinality and top-20 frequency table for {len(cardinality_summary)} "
            f"categorical flag column(s)."
        ),
        columns=list(cardinality_summary.keys()),
        affected_rows=0,
        total_rows=len(df),
        serve_relevant=False,
        examples=[],
        extra={"per_column": cardinality_summary},
    ))

    near_unique: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        if not _is_textlike(s):
            continue
        non_null = int(s.notna().sum())
        if non_null == 0:
            continue
        nunique = int(s.nunique())
        ratio = nunique / non_null
        if ratio >= NEAR_UNIQUE_ID_RATIO and nunique >= NEAR_UNIQUE_ID_MIN_CARDINALITY:
            near_unique.append({
                "column": col,
                "nunique": nunique,
                "non_null_count": non_null,
                "ratio": _round(ratio),
                "expected": col == ID_COL,
            })
    findings.append(_finding(
        id_="categorical.near_unique_identifier_columns",
        group="categorical",
        title="Columns with near-unique (identifier-like) cardinality",
        description=(
            f"{len(near_unique)} text-like column(s) have nunique/non-null ratio "
            f">= {NEAR_UNIQUE_ID_RATIO} and nunique >= {NEAR_UNIQUE_ID_MIN_CARDINALITY}. "
            f"Expected to fire on `{ID_COL}` (the real identifier); any other column here "
            f"would be a genuine misfiled-as-categorical concern."
        ),
        columns=[d["column"] for d in near_unique],
        affected_rows=0,
        total_rows=len(df),
        serve_relevant=False,
        examples=[],
        extra={"columns": near_unique},
    ))

    summary = {"flag_columns_checked": list(cardinality_summary.keys()), "finding_count": len(findings)}
    return {"summary": summary, "findings": findings}


# ── §3.5 cross-column consistency ───────────────────────────────────────────────

def audit_cross_column(df: pd.DataFrame, footer_mask: pd.Series) -> dict[str, Any]:
    data = df.loc[~footer_mask]
    findings: list[dict[str, Any]] = []
    mono_summary: dict[str, Any] = {}

    pairs = (
        ("sales_3_month", "sales_1_month", "sales_3 >= sales_1"),
        ("sales_6_month", "sales_3_month", "sales_6 >= sales_3"),
        ("sales_9_month", "sales_6_month", "sales_9 >= sales_6"),
    )
    for higher, lower, label in pairs:
        if higher not in df.columns or lower not in df.columns:
            continue
        valid = data[[ID_COL, higher, lower]].dropna(subset=[higher, lower])
        gap = valid[lower] - valid[higher]  # positive gap == violation (lower exceeds higher)
        violation_mask = gap > 0
        violation_count = int(violation_mask.sum())
        checked_rows = int(len(valid))
        violation_rate = (violation_count / checked_rows) if checked_rows else 0.0
        mono_summary[label] = {
            "violation_count": violation_count,
            "violation_rate": _round(violation_rate),
            "checked_rows": checked_rows,
        }

        examples: list[dict[str, Any]] = []
        if violation_count > 0:
            worst = valid.loc[violation_mask].copy()
            worst["_gap"] = gap.loc[violation_mask]
            worst = worst.sort_values(["_gap", ID_COL], ascending=[False, True])
            examples = _examples(worst, cols=[ID_COL, higher, lower], n=EXAMPLES_PER_FINDING)

        findings.append(_finding(
            id_=f"cross_column.sales_monotonicity.{higher}_vs_{lower}",
            group="cross_column",
            title=f"Sales window monotonicity violation: {label}",
            description=(
                f"{violation_count} of {checked_rows} rows violate '{label}' "
                f"(violation rate {_round(violation_rate)}); replicates the demand notebook's "
                f"cumulative-window sanity check but reports the violation rate, not the pass rate."
            ),
            columns=[higher, lower],
            affected_rows=violation_count,
            total_rows=checked_rows,
            serve_relevant=True,
            examples=examples,
            extra={"nesting_rate": _round(1 - violation_rate)},
        ))

    if "min_bank" in df.columns and "national_inv" in df.columns:
        valid = data[[ID_COL, "min_bank", "national_inv"]].dropna(subset=["min_bank", "national_inv"])
        mask = valid["min_bank"] > valid["national_inv"]
        cnt = int(mask.sum())
        findings.append(_finding(
            id_="cross_column.min_bank_exceeds_national_inv",
            group="cross_column",
            title="min_bank exceeds national_inv (on-hand vs. reorder-point analog)",
            description=(
                f"{cnt} of {len(valid)} rows have min_bank > national_inv. There is no explicit "
                f"reorder-point column in this schema; min_bank is used as that analog here."
            ),
            columns=["min_bank", "national_inv"],
            affected_rows=cnt,
            total_rows=int(len(valid)),
            serve_relevant=True,
            examples=_examples(valid.loc[mask]),
        ))

    summary = {"sales_monotonicity": mono_summary, "finding_count": len(findings)}
    return {"summary": summary, "findings": findings}


# ── §3.6 target / label column ───────────────────────────────────────────────────

def audit_target(df: pd.DataFrame, footer_mask: pd.Series) -> dict[str, Any]:
    data = df.loc[~footer_mask]
    n_data = len(data)
    findings: list[dict[str, Any]] = []

    target_all_null = int(df[TARGET_COL].isna().sum())
    target_data_null = int(data[TARGET_COL].isna().sum())
    findings.append(_finding(
        id_="target.missingness",
        group="target",
        title=f"Missingness in target column {TARGET_COL}",
        description=(
            f"{target_data_null} of {n_data} data rows have a missing target "
            f"({target_all_null - target_data_null} additional NaNs are footer-artifact rows, "
            f"a distinct category from real feature missingness)."
        ),
        columns=[TARGET_COL],
        affected_rows=target_data_null,
        total_rows=n_data,
        serve_relevant=False,
        examples=_examples(data.loc[data[TARGET_COL].isna()]),
        extra={"footer_row_contribution": target_all_null - target_data_null},
    ))

    labeled = data.dropna(subset=[TARGET_COL])
    counts = labeled[TARGET_COL].value_counts()
    n_yes = int(counts.get("Yes", 0))
    n_no = int(counts.get("No", 0))
    imbalance_ratio = (n_no / n_yes) if n_yes else None
    findings.append(_finding(
        id_="target.class_balance",
        group="target",
        title="Target class balance",
        description=(
            f"{TARGET_COL}: No={n_no}, Yes={n_yes}, imbalance ratio No:Yes = "
            f"{_round(imbalance_ratio)}:1."
        ),
        columns=[TARGET_COL],
        affected_rows=n_yes,
        total_rows=int(len(labeled)),
        serve_relevant=False,
        examples=[],
        extra={"n_yes": n_yes, "n_no": n_no, "imbalance_ratio_no_to_yes": _round(imbalance_ratio)},
    ))

    leakage_flags: list[dict[str, Any]] = []
    target_bool = labeled[TARGET_COL] == "Yes"
    for col in df.columns:
        if col in (TARGET_COL, ID_COL):
            continue
        s = labeled[col]
        if pd.api.types.is_numeric_dtype(s):
            miss_pos = float(s[target_bool].isna().mean()) if len(s[target_bool]) else 0.0
            miss_neg = float(s[~target_bool].isna().mean()) if len(s[~target_bool]) else 0.0
            if abs(miss_pos - miss_neg) > LEAKAGE_RATE_GAP_THRESHOLD:
                leakage_flags.append({
                    "column": col,
                    "reason": "missingness rate differs sharply between target classes",
                    "missingness_rate_target_yes": _round(miss_pos),
                    "missingness_rate_target_no": _round(miss_neg),
                })
        else:
            nn_pos = float(s[target_bool].notna().mean()) if len(s[target_bool]) else 0.0
            nn_neg = float(s[~target_bool].notna().mean()) if len(s[~target_bool]) else 0.0
            if abs(nn_pos - nn_neg) > LEAKAGE_RATE_GAP_THRESHOLD:
                leakage_flags.append({
                    "column": col,
                    "reason": "non-null rate differs sharply between target classes",
                    "non_null_rate_target_yes": _round(nn_pos),
                    "non_null_rate_target_no": _round(nn_neg),
                })

    findings.append(_finding(
        id_="target.leakage_suspicious_columns",
        group="target",
        title="Columns suspiciously correlated with target availability (manual review)",
        description=(
            f"{len(leakage_flags)} column(s) flagged for manual review because their "
            f"missingness/non-null rate differs by more than {LEAKAGE_RATE_GAP_THRESHOLD:.0%} "
            f"between target classes. Flagged only — never auto-excluded."
        ),
        columns=[f["column"] for f in leakage_flags],
        affected_rows=0,
        total_rows=int(len(labeled)),
        serve_relevant=False,
        examples=[],
        extra={"flagged_for_manual_review": leakage_flags},
    ))

    summary = {"n_yes": n_yes, "n_no": n_no, "finding_count": len(findings)}
    return {"summary": summary, "findings": findings}


# ── markdown rendering ───────────────────────────────────────────────────────────

_GROUP_TITLES: dict[str, str] = {
    "structural": "3.1 Structural",
    "missingness": "3.2 Missingness",
    "numeric": "3.3 Numeric Sanity",
    "categorical": "3.4 Categorical / Text Sanity",
    "cross_column": "3.5 Cross-Column Consistency",
    "target": "3.6 Target / Label Column",
}


def render_markdown(report: dict[str, Any], generated_at: str) -> str:
    """Pure formatting of an already-assembled report dict. Never re-reads the CSV or
    recomputes anything. `generated_at` is the only place a timestamp appears anywhere in
    the two output files — it is never written into the JSON, to keep JSON output
    byte-identical across re-runs."""
    lines: list[str] = []
    lines.append("# CSV Data Quality Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append(f"Source CSV: `{report['meta']['raw_csv_path']}` ({report['meta']['raw_csv_path_source']})")
    lines.append(f"Rows: {report['meta']['row_count']}  Columns: {report['meta']['column_count']}")
    lines.append("")

    sku_res = report["sku_cardinality_resolution"]
    lines.append("## SKU Cardinality Resolution (database_spec.md §2 — blocking)")
    lines.append("")
    lines.append(f"- **Case: {sku_res['case']}**")
    lines.append(
        f"- Total rows: {sku_res['rows']} (footer-artifact rows: {sku_res['footer_artifact_rows']}, "
        f"data rows: {sku_res['data_rows']})"
    )
    lines.append(f"- Unique SKUs: {sku_res['unique_skus']}")
    lines.append(f"- Fully identical rows: {sku_res['fully_identical_rows']}")
    lines.append(f"- Rows with a repeated SKU: {sku_res['rows_with_repeated_sku']}")
    lines.append(f"- {sku_res['case_note']}")
    lines.append("")

    all_findings: list[dict[str, Any]] = []
    for group in CHECK_GROUP_ORDER:
        all_findings.extend(report[group]["findings"])
    all_findings.sort(key=lambda f: (f["check_group"], f["id"]))

    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Columns | % Rows Affected | Serve-relevant |")
    lines.append("|---|---|---|---|")
    for f in all_findings:
        cols = ", ".join(f["columns"]) if f["columns"] else "-"
        pct = f["affected_pct"] if f["affected_pct"] is not None else "-"
        serve = "Y" if f["serve_relevant"] else "N"
        lines.append(f"| {f['title']} | {cols} | {pct} | {serve} |")
    lines.append("")

    for group in CHECK_GROUP_ORDER:
        lines.append(f"## {_GROUP_TITLES[group]}")
        lines.append("")
        for f in report[group]["findings"]:
            lines.append(f"### {f['title']}")
            lines.append("")
            lines.append(f["description"])
            lines.append("")
            lines.append(f"- Affected rows: {f['affected_rows']} / {f['total_rows']} ({f['affected_pct']}%)")
            lines.append(f"- Serve-relevant: {'Yes' if f['serve_relevant'] else 'No'}")
            if f["examples"]:
                lines.append("")
                lines.append("Examples:")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(f["examples"], indent=2))
                lines.append("```")
            lines.append("")

    return "\n".join(lines)


# ── entry point ──────────────────────────────────────────────────────────────────

def main() -> int:
    df, dtype_warnings = _read_raw_csv(RAW_CSV_PATH)
    footer_mask = _footer_artifact_mask(df)

    structural = audit_structural(df, dtype_warnings, footer_mask)
    missingness = audit_missingness(df, footer_mask)
    numeric = audit_numeric(df, footer_mask)
    categorical = audit_categorical(df)
    cross_column = audit_cross_column(df, footer_mask)
    target = audit_target(df, footer_mask)

    try:
        raw_csv_path_display = str(RAW_CSV_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        raw_csv_path_display = str(RAW_CSV_PATH)

    report: dict[str, Any] = {
        "meta": {
            "spec_ref": ".CLAUDE/csv_quality_audit_spec.md",
            "raw_csv_path": raw_csv_path_display,
            "raw_csv_path_source": "RSM_RAW_CSV env var" if "RSM_RAW_CSV" in os.environ else "default",
            "read_strategy": "whole_file",
            "peak_memory_mb": None,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_header_order": list(df.columns),
            "dtype_warnings_captured": dtype_warnings,
            "seed": SEED,
            "round_ndigits": ROUND_NDIGITS,
        },
        "structural": structural,
        "missingness": missingness,
        "numeric": numeric,
        "categorical": categorical,
        "cross_column": cross_column,
        "target": target,
    }
    report["sku_cardinality_resolution"] = structural["sku_cardinality_resolution"]
    report = _json_safe(report)

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    generated_at = datetime.now(timezone.utc).isoformat()
    REPORT_MD_PATH.write_text(render_markdown(report, generated_at))

    total_findings = sum(len(report[g]["findings"]) for g in CHECK_GROUP_ORDER)
    serve_relevant_count = sum(
        1 for g in CHECK_GROUP_ORDER for f in report[g]["findings"] if f["serve_relevant"]
    )
    print(f"Rows: {report['meta']['row_count']}  Columns: {report['meta']['column_count']}")
    print(f"SKU cardinality case: {report['sku_cardinality_resolution']['case']}")
    print(f"Findings: {total_findings} ({serve_relevant_count} serve-relevant)")
    print(f"Wrote {REPORT_JSON_PATH}")
    print(f"Wrote {REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
