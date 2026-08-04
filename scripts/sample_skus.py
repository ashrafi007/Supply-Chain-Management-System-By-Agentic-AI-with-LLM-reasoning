"""
scripts/sample_skus.py — stratified SKU sample for seeding (.CLAUDE/seed_data_spec.md §3-4 Step 1).

Importable via :func:`sample_skus`, or standalone:

    python -m scripts.sample_skus            # writes data/seed_sample.parquet for inspection

Returns **raw, uncleaned** rows — value-level cleaning happens later, in the seeder
(``features.clean_raw_row``), not here (spec §4 Step 1).

Source: ``Dataset/RSM_Dataset.csv`` (the raw CSV, per spec §1) — deliberately dirty,
so the "contains missing values" stratum (§3) and ``features.clean_raw_row`` actually
have real work to do. ``Dataset/cleaned_dataset.csv`` (median/mode-imputed upstream by
Notebooks/cleaned_databse.ipynb) was evaluated and rejected as a seed source: it carries
zero missing values, so it starves the missing-values stratum and makes clean_raw_row a
no-op on every row — neither of which demonstrates what this seed is meant to prove.
Override via ``SEED_SOURCE_CSV`` if a different source is ever needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

SOURCE_CSV: Path = Path(
    os.environ.get("SEED_SOURCE_CSV", PROJECT_ROOT / "Dataset" / "RSM_Dataset.csv")
)

SKU_COL: str = "sku"
TARGET_COL: str = "went_on_backorder"

SEED: int = 42
TARGET_SAMPLE_SIZE: int = 5000
ALARM_POSITIVE_TARGET: int = 250
HIGH_URGENCY_TARGET: int = 250
MISSING_VALUES_TARGET: int = 50


def _load_source(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path if csv_path is not None else SOURCE_CSV
    if not path.exists():
        raise FileNotFoundError(f"seed source CSV not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    # Same footer-artifact drop as the cleaning notebook / csv_quality_audit: the RSM
    # export's trailing row has every column but sku blank.
    df = df.dropna(subset=df.columns.difference([SKU_COL]), how="all")
    return df


def _take_stratum(pool: pd.DataFrame, n: int, seed: int, label: str) -> pd.DataFrame:
    if len(pool) < n:
        print(
            f"  [shortfall] {label}: wanted {n}, only {len(pool)} available in source "
            f"— taking all available, not substituting clean rows."
        )
        return pool
    return pool.sample(n=n, random_state=seed)


def _alarm_positive_stratum(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return _take_stratum(df[df[TARGET_COL] == "Yes"], n, seed, "alarm-positive")


def _high_urgency_stratum(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    # No urgency_score exists in raw data — that is an Agent-3 runtime output, not
    # sampleable input. Proxy: on-hand inventory below the min_bank reorder-point
    # analog, the same on-hand-vs-reorder-point relationship the CSV quality audit
    # (audits/csv_quality_audit.py) already flags for this dataset.
    return _take_stratum(df[df["national_inv"] < df["min_bank"]], n, seed, "high-urgency (proxy: national_inv < min_bank)")


def _missing_values_stratum(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return _take_stratum(df[df.isna().any(axis=1)], n, seed, "contains missing values")


def sample_skus(
    df: pd.DataFrame | None = None,
    target_size: int = TARGET_SAMPLE_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build the stratified seed sample (spec §3). Deterministic for a fixed source + seed."""
    if df is None:
        df = _load_source()

    alarm = _alarm_positive_stratum(df, ALARM_POSITIVE_TARGET, seed)
    urgent = _high_urgency_stratum(df, HIGH_URGENCY_TARGET, seed)
    missing = _missing_values_stratum(df, MISSING_VALUES_TARGET, seed)

    combined = pd.concat([alarm, urgent, missing]).drop_duplicates(subset=[SKU_COL])

    remainder_target = max(target_size - len(combined), 0)
    remainder_pool = df[~df[SKU_COL].isin(combined[SKU_COL])]
    remainder_n = min(remainder_target, len(remainder_pool))
    remainder = remainder_pool.sample(n=remainder_n, random_state=seed) if remainder_n > 0 else remainder_pool.iloc[0:0]

    sample = pd.concat([combined, remainder]).drop_duplicates(subset=[SKU_COL]).reset_index(drop=True)

    _print_composition(sample)
    return sample


def _print_composition(sample: pd.DataFrame) -> None:
    mask_alarm = sample[TARGET_COL] == "Yes"
    mask_urgent = sample["national_inv"] < sample["min_bank"]
    mask_missing = sample.isna().any(axis=1)
    mask_any = mask_alarm | mask_urgent | mask_missing

    print("\nRealized stratum composition:")
    print(f"  total sampled SKUs:        {len(sample)}")
    print(f"  alarm-positive:            {int(mask_alarm.sum())}")
    print(f"  high-urgency (proxy):      {int(mask_urgent.sum())}")
    print(f"  contains missing values:   {int(mask_missing.sum())}")
    print(f"  random remainder:          {int((~mask_any).sum())}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the stratified seed sample; write it to parquet for inspection.")
    parser.add_argument("--target-size", type=int, default=TARGET_SAMPLE_SIZE)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "seed_sample.parquet")
    args = parser.parse_args(argv)

    print(f"Source: {SOURCE_CSV}")
    sample = sample_skus(target_size=args.target_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)
    print(f"\nWrote {len(sample)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
