# Data Quality Audit Spec — Raw CSV

**Scope:** Audit `data/raw/rsm.csv` for quality issues. **Produce findings only — fix nothing.** Cleaning logic (`features.py`) is a separate spec that consumes this audit's output.
**Owner:** whoever is refining the CSV.
**Runs:** independent of the database and independent of wrapper parity. No prerequisite beyond having the raw CSV on disk.
**Why this comes first:** every downstream cleaning rule, every `inventory_current` column type, and every imputation constant in `features.py` should be a documented response to something this audit found — not a guess. Writing cleaning code before this audit means re-deriving these facts by trial and error later, usually as a production bug.

---

## 1. Principle

This is a **read-only diagnostic pass.** No column is dropped, no value is imputed, no row is deleted. The output is a report: a structured inventory of every quality problem in the raw CSV, with counts, examples, and a suggested (not applied) handling strategy per issue.

Two audiences read the output: `features.py` (the cleaning spec that follows) and the paper (a documented, quantified account of data quality beats "we cleaned it" in a methods section).

---

## 2. Files to create

```
audits/
    csv_quality_audit.py       # the audit script, produces the report
    reports/
        csv_quality_report.json    # machine-readable findings
        csv_quality_report.md      # human-readable summary, generated from the json
```

No changes to `data/`, `src/`, or the database. This spec touches only `audits/`.

---

## 3. Checks to run

Run every check below over the full raw CSV (not a sample — quality issues can be rare and a sample can miss them). For each check, record: which columns are affected, the count and percentage of affected rows, 3–5 concrete examples, and the value distribution where relevant.

### 3.1 Structural

- **Row and column count.** Sanity check against what you expect (~1.9M rows).
- **Duplicate rows** — fully identical across all columns. Count them; do not drop them here.
- **Duplicate `sku`** — count SKUs appearing more than once. For a sample of repeated SKUs, print the repeated rows side by side to see what varies (date? location? nothing?). **This directly resolves the Case A/B/C question from `database_spec.md` §2 — its output should be pasted into that section, not re-derived by eye.**
- **Column dtypes as read** — flag any numeric-looking column that pandas read as `object` (usually a sign of stray non-numeric values like `"NA"`, `"?"`, or inconsistent formatting).

### 3.2 Missingness

- Per column: count and percentage of `NaN` / empty string / whitespace-only values.
- Distinguish **explicit** nulls (`NaN`) from **encoded** nulls — placeholder values that mean "missing" but aren't recognized as such: `-1`, `9999`, `"N/A"`, `"unknown"`, empty string, `"None"` as literal text. Encoded nulls are the ones that silently corrupt statistics if not caught — a `-1` placeholder dragged into a mean or median poisons it.
- Per column: **missingness pattern** — is it random, or does it cluster (e.g. always missing together with another column, always missing for one supplier)? A clustered pattern usually means "not collected for this segment," which is a different handling decision than random missingness.

### 3.3 Numeric sanity

- **`inf` / `-inf`** — count per column. Note which downstream ratio/rate columns these tend to appear in (division by zero is the usual source).
- **Negative values in columns that should be non-negative** — inventory quantities, lead times, prices, sales. Count and show examples; a small number is often a data entry sign error, a large number suggests the column means something else than assumed.
- **Impossible values** — a lead time of 0 with nonzero transit quantity, a sales figure exceeding on-hand inventory by an implausible margin, negative dates. Define "impossible" per column based on domain meaning, not just statistically.
- **Outliers** — for every numeric column, report min, max, mean, median, p95, p99, p99.9, and flag columns where the max is many orders of magnitude above p99.9 (a strong signal of a data entry error rather than a legitimate rare event).
- **Zero-inflation** — percentage of exact zeros per numeric column. High zero-inflation isn't necessarily bad data, but it changes what "impute with the mean" would do (drag the mean toward zero misleadingly).

### 3.4 Categorical / text sanity

- Per categorical column: cardinality (distinct value count), and the top 20 most frequent values with counts.
- **Inconsistent encoding of the same category** — case differences (`"Yes"` / `"yes"` / `"YES"`), leading/trailing whitespace, synonyms (`"Y"` vs `"Yes"`), which silently multiply the apparent cardinality and will produce an inconsistent one-hot encoding at serve time if not normalized.
- Any categorical column with cardinality high enough to suggest it's actually a near-unique identifier misfiled as categorical.

### 3.5 Cross-column consistency

- **Logical relationships that should hold.** Check `sales_1_month`, `sales_3_month`, `sales_6_month`, `sales_9_month` are monotonically non-decreasing in the expected direction where the notebook assumes it (the demand notebook's Section 2.1 already checks `sales_3 >= sales_1` etc. — replicate that check here as a general audit, and report the violation rate over the full dataset, not just descriptively).
- Any other pair of columns with an implied relationship in the domain (e.g. in-transit quantity vs. lead time, on-hand vs. reorder point) — check and report violation counts.

### 3.6 Target / label column

- Class balance of the backorder target (confirm the 137:1 figure over the full dataset, not a subsample).
- Missingness in the target — rows with no label cannot be used for supervised training and are a distinct category from feature missingness.
- Any leakage-suspicious column: a feature that is suspiciously predictive of the target in a simple univariate check (e.g. a column that is only populated when the target is positive) — flag for manual review, do not auto-exclude.

### 3.7 Train/serve relevance flag

For every issue found above, tag it with which serving path it affects:

- **Batch-only** — an issue that only matters for training (e.g. duplicate rows across train/test, addressed in `verification_spec.md`'s parity concerns, not here).
- **Serve-relevant** — an issue that a *single live row* could also exhibit (missing values, `inf`, encoded nulls, negative-where-impossible). This tag is what determines whether a rule belongs in `features.py`'s value-level cleaning bucket. Everything serve-relevant needs a row-safe fix (impute, clip, coerce) — never a row-drop, since a live single-row request cannot be "dropped."

---

## 4. Step-by-step build

### Step 1 — `audits/csv_quality_audit.py` skeleton
- Load the raw CSV in chunks if memory is a concern on the M2 (`pd.read_csv(..., chunksize=...)`), or whole if it fits; report which approach was used and the peak memory if chunked.
- One function per check group in §3 (`audit_structural`, `audit_missingness`, `audit_numeric`, `audit_categorical`, `audit_cross_column`, `audit_target`). Each returns a plain dict — no side effects, no plotting required (plots are optional and go in a separate notebook cell if wanted, not in the script).

### Step 2 — Assemble the report
- Merge all check-group outputs into one structured dict.
- Write `audits/reports/csv_quality_report.json` — the full machine-readable findings, including every example row (as dicts, not DataFrame reprs) so `features.py` development can query it directly.
- Generate `audits/reports/csv_quality_report.md` from the JSON: one section per check group, a summary table at the top (issue, affected columns, % rows, serve-relevant Y/N), and the SKU-cardinality finding highlighted at the very top since it's blocking for the database spec.

### Step 3 — Run
```
python -m audits.csv_quality_audit
```
No arguments needed beyond the CSV path (put it in a config constant at the top of the script, matching `data/raw/rsm.csv`).

---

## 5. Acceptance criteria

- [ ] Every subsection in §3 has a corresponding entry in the JSON report.
- [ ] The SKU-cardinality finding (§3.1) is explicit enough to directly answer the Case A/B/C question in `database_spec.md` §2 — paste the finding there once this audit runs.
- [ ] Every finding is tagged batch-only or serve-relevant (§3.7).
- [ ] No file outside `audits/` is modified. No row is dropped, no value is changed.
- [ ] The `.md` report is readable without opening the JSON — it's the version for the paper and for teammates.
- [ ] Re-running the script on the same CSV produces byte-identical JSON (fully deterministic — no sampling, or sampling with `SEED=42` if used for anything).

---

## 6. What happens after this spec

This audit's findings feed two places, in order:

1. **`features.py` cleaning spec** — one row-safe fix per serve-relevant finding (impute from a train-fitted statistic, clip, coerce type). Never a row-drop rule for anything tagged serve-relevant.
2. **`database_spec.md` §2** — the SKU-cardinality finding resolves Case A/B/C and confirms (or corrects) `inventory_current`'s primary key.

Do not write cleaning logic before this audit exists — the whole point is that cleaning rules are a response to documented findings, not assumptions.

---

## 7. Notes for the paper

A quantified data-quality account — missingness rates, class imbalance confirmed at full scale, cross-column consistency violation rates — is stronger methods-section material than an unqualified "the dataset was cleaned." It also pre-empts the natural reviewer question of how representative the reported metrics are, since every cleaning decision traces back to a specific, countable issue in this report rather than an ad hoc judgment call made mid-notebook.
