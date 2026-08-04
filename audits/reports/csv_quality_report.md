# CSV Data Quality Report

Generated: 2026-07-28T08:14:12.429496+00:00
Source CSV: `Dataset/RSM_Dataset.csv` (default)
Rows: 1929937  Columns: 23

## SKU Cardinality Resolution (database_spec.md §2 — blocking)

- **Case: A**
- Total rows: 1929937 (footer-artifact rows: 2, data rows: 1929935)
- Unique SKUs: 1929935
- Fully identical rows: 0
- Rows with a repeated SKU: 0
- No repeated SKUs among real data rows (footer-artifact rows excluded). nunique(sku) == len(data rows) -> Case A. inventory_current PK = sku_id.

## Summary

| Check | Columns | % Rows Affected | Serve-relevant |
|---|---|---|---|
| Cardinality and top-20 values for flag columns | potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop | 0.0 | N |
| Columns with near-unique (identifier-like) cardinality | sku | 0.0 | N |
| min_bank exceeds national_inv (on-hand vs. reorder-point analog) | min_bank, national_inv | 4.052209 | Y |
| Sales window monotonicity violation: sales_3 >= sales_1 | sales_3_month, sales_1_month | 0.0 | Y |
| Sales window monotonicity violation: sales_6 >= sales_3 | sales_6_month, sales_3_month | 0.0 | Y |
| Sales window monotonicity violation: sales_9 >= sales_6 | sales_9_month, sales_6_month | 0.0 | Y |
| Missingness rate by categorical segment | lead_time | 0.0 | N |
| Missingness clustering across column pairs | lead_time | 0.0 | N |
| Encoded-null numeric sentinel in perf_12_month_avg | perf_12_month_avg | 7.255426 | Y |
| Encoded-null numeric sentinel in perf_6_month_avg | perf_6_month_avg | 7.698653 | Y |
| Possible numeric sentinel 9999 in forecast_9_month | forecast_9_month | 5.2e-05 | Y |
| Possible numeric sentinel -1 in national_inv | national_inv | 0.076065 | Y |
| Possible numeric sentinel -9 in national_inv | national_inv | 0.006373 | Y |
| Possible numeric sentinel -99 in national_inv | national_inv | 0.000155 | Y |
| Possible numeric sentinel 9999 in national_inv | national_inv | 5.2e-05 | Y |
| Possible numeric sentinel 9999 in sales_9_month | sales_9_month | 5.2e-05 | Y |
| Explicit missing values in lead_time | lead_time | 5.99072 | Y |
| lead_time == 0 with in_transit_qty > 0 | lead_time, in_transit_qty | 0.114719 | Y |
| sales_1_month implausibly exceeds national_inv | sales_1_month, national_inv | 1.066875 | Y |
| Negative values in national_inv | national_inv | 0.344778 | Y |
| Outlier statistics per numeric column | national_inv, lead_time, in_transit_qty, forecast_3_month, forecast_6_month, forecast_9_month, sales_1_month, sales_3_month, sales_6_month, sales_9_month, min_bank, pieces_past_due, perf_6_month_avg, perf_12_month_avg, local_bo_qty | 0.0 | N |
| Column dtypes as read; numeric-looking columns read as non-numeric | sku | 0.0 | Y |
| Fully duplicate rows | sku, national_inv, lead_time, in_transit_qty, forecast_3_month, forecast_6_month, forecast_9_month, sales_1_month, sales_3_month, sales_6_month, sales_9_month, min_bank, potential_issue, pieces_past_due, perf_6_month_avg, perf_12_month_avg, local_bo_qty, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, went_on_backorder | 0.0 | N |
| Row and column counts | - | 0.000104 | N |
| SKU cardinality resolution (database_spec.md §2 Case A/B/C) | sku | 0.0 | N |
| Target class balance | went_on_backorder | 0.724429 | N |
| Columns suspiciously correlated with target availability (manual review) | - | 0.0 | N |
| Missingness in target column went_on_backorder | went_on_backorder | 0.0 | N |

## 3.1 Structural

### Row and column counts

CSV has 1929937 total rows (2 footer-artifact rows, 1929935 real data rows) and 23 columns.

- Affected rows: 2 / 1929937 (0.000104%)
- Serve-relevant: No

Examples:

```json
[
  {
    "sku": "(242075 rows)",
    "national_inv": null,
    "lead_time": null,
    "in_transit_qty": null,
    "forecast_3_month": null,
    "forecast_6_month": null,
    "forecast_9_month": null,
    "sales_1_month": null,
    "sales_3_month": null,
    "sales_6_month": null,
    "sales_9_month": null,
    "min_bank": null,
    "potential_issue": null,
    "pieces_past_due": null,
    "perf_6_month_avg": null,
    "perf_12_month_avg": null,
    "local_bo_qty": null,
    "deck_risk": null,
    "oe_constraint": null,
    "ppap_risk": null,
    "stop_auto_buy": null,
    "rev_stop": null,
    "went_on_backorder": null
  },
  {
    "sku": "(1687860 rows)",
    "national_inv": null,
    "lead_time": null,
    "in_transit_qty": null,
    "forecast_3_month": null,
    "forecast_6_month": null,
    "forecast_9_month": null,
    "sales_1_month": null,
    "sales_3_month": null,
    "sales_6_month": null,
    "sales_9_month": null,
    "min_bank": null,
    "potential_issue": null,
    "pieces_past_due": null,
    "perf_6_month_avg": null,
    "perf_12_month_avg": null,
    "local_bo_qty": null,
    "deck_risk": null,
    "oe_constraint": null,
    "ppap_risk": null,
    "stop_auto_buy": null,
    "rev_stop": null,
    "went_on_backorder": null
  }
]
```

### Fully duplicate rows

0 fully identical rows among real data rows (0 when footer rows are included).

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

### SKU cardinality resolution (database_spec.md §2 Case A/B/C)

No repeated SKUs among real data rows (footer-artifact rows excluded). nunique(sku) == len(data rows) -> Case A. inventory_current PK = sku_id.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

### Column dtypes as read; numeric-looking columns read as non-numeric

1 column(s) flagged as numeric-looking but read as non-numeric: ['sku'].

- Affected rows: 0 / 1929937 (0.0%)
- Serve-relevant: Yes

## 3.2 Missingness

### Possible numeric sentinel -1 in national_inv

1468 rows equal -1 in national_inv, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 1468 / 1929935 (0.076065%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3333345,
    "national_inv": -1.0,
    "lead_time": 2.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 24.0,
    "forecast_6_month": 24.0,
    "forecast_9_month": 24.0,
    "sales_1_month": 22.0,
    "sales_3_month": 22.0,
    "sales_6_month": 22.0,
    "sales_9_month": 22.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.25,
    "perf_12_month_avg": 0.16,
    "local_bo_qty": 1.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3365474,
    "national_inv": -1.0,
    "lead_time": 12.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 3.0,
    "forecast_6_month": 5.0,
    "forecast_9_month": 6.0,
    "sales_1_month": 3.0,
    "sales_3_month": 4.0,
    "sales_6_month": 5.0,
    "sales_9_month": 6.0,
    "min_bank": 2.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.31,
    "perf_12_month_avg": 0.4,
    "local_bo_qty": 2.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3383744,
    "national_inv": -1.0,
    "lead_time": 16.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 3.0,
    "forecast_6_month": 4.0,
    "forecast_9_month": 5.0,
    "sales_1_month": 0.0,
    "sales_3_month": 1.0,
    "sales_6_month": 1.0,
    "sales_9_month": 1.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.31,
    "perf_12_month_avg": 0.4,
    "local_bo_qty": 1.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3437159,
    "national_inv": -1.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 13.0,
    "forecast_6_month": 21.0,
    "forecast_9_month": 25.0,
    "sales_1_month": 1.0,
    "sales_3_month": 5.0,
    "sales_6_month": 10.0,
    "sales_9_month": 16.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 1.0,
    "perf_12_month_avg": 1.0,
    "local_bo_qty": 1.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "Yes"
  },
  {
    "sku": 3443019,
    "national_inv": -1.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 2.0,
    "forecast_6_month": 2.0,
    "forecast_9_month": 2.0,
    "sales_1_month": 3.0,
    "sales_3_month": 3.0,
    "sales_6_month": 45.0,
    "sales_9_month": 45.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.65,
    "perf_12_month_avg": 0.71,
    "local_bo_qty": 1.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Possible numeric sentinel -9 in national_inv

123 rows equal -9 in national_inv, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 123 / 1929935 (0.006373%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3285264,
    "national_inv": -9.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 32.0,
    "forecast_6_month": 32.0,
    "forecast_9_month": 33.0,
    "sales_1_month": 29.0,
    "sales_3_month": 32.0,
    "sales_6_month": 32.0,
    "sales_9_month": 33.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.85,
    "perf_12_month_avg": 0.64,
    "local_bo_qty": 9.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3341433,
    "national_inv": -9.0,
    "lead_time": 2.0,
    "in_transit_qty": 54.0,
    "forecast_3_month": 1519.0,
    "forecast_6_month": 2443.0,
    "forecast_9_month": 3367.0,
    "sales_1_month": 274.0,
    "sales_3_month": 550.0,
    "sales_6_month": 718.0,
    "sales_9_month": 853.0,
    "min_bank": 233.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.93,
    "perf_12_month_avg": 0.89,
    "local_bo_qty": 8.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3341865,
    "national_inv": -9.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 36.0,
    "forecast_6_month": 42.0,
    "forecast_9_month": 48.0,
    "sales_1_month": 0.0,
    "sales_3_month": 9.0,
    "sales_6_month": 17.0,
    "sales_9_month": 28.0,
    "min_bank": 9.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.94,
    "perf_12_month_avg": 0.96,
    "local_bo_qty": 9.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "Yes"
  },
  {
    "sku": 3360592,
    "national_inv": -9.0,
    "lead_time": 2.0,
    "in_transit_qty": 5.0,
    "forecast_3_month": 12.0,
    "forecast_6_month": 17.0,
    "forecast_9_month": 22.0,
    "sales_1_month": 15.0,
    "sales_3_month": 38.0,
    "sales_6_month": 58.0,
    "sales_9_month": 95.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.94,
    "perf_12_month_avg": 0.97,
    "local_bo_qty": 9.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3426715,
    "national_inv": -9.0,
    "lead_time": 12.0,
    "in_transit_qty": 13.0,
    "forecast_3_month": 14.0,
    "forecast_6_month": 22.0,
    "forecast_9_month": 30.0,
    "sales_1_month": 3.0,
    "sales_3_month": 13.0,
    "sales_6_month": 21.0,
    "sales_9_month": 25.0,
    "min_bank": 5.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.73,
    "perf_12_month_avg": 0.79,
    "local_bo_qty": 9.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Possible numeric sentinel -99 in national_inv

3 rows equal -99 in national_inv, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 3 / 1929935 (0.000155%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 1596465,
    "national_inv": -99.0,
    "lead_time": 4.0,
    "in_transit_qty": 72.0,
    "forecast_3_month": 1132.0,
    "forecast_6_month": 1785.0,
    "forecast_9_month": 2417.0,
    "sales_1_month": 91.0,
    "sales_3_month": 1253.0,
    "sales_6_month": 1939.0,
    "sales_9_month": 2461.0,
    "min_bank": 73.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.46,
    "perf_12_month_avg": 0.59,
    "local_bo_qty": 189.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 1620723,
    "national_inv": -99.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 1000.0,
    "forecast_6_month": 1400.0,
    "forecast_9_month": 2000.0,
    "sales_1_month": 1169.0,
    "sales_3_month": 1265.0,
    "sales_6_month": 1439.0,
    "sales_9_month": 2048.0,
    "min_bank": 158.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 1.0,
    "perf_12_month_avg": 1.0,
    "local_bo_qty": 204.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 2062779,
    "national_inv": -99.0,
    "lead_time": 2.0,
    "in_transit_qty": 49.0,
    "forecast_3_month": 75.0,
    "forecast_6_month": 75.0,
    "forecast_9_month": 75.0,
    "sales_1_month": 54.0,
    "sales_3_month": 107.0,
    "sales_6_month": 113.0,
    "sales_9_month": 115.0,
    "min_bank": 24.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.96,
    "perf_12_month_avg": 0.96,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Possible numeric sentinel 9999 in national_inv

1 rows equal 9999 in national_inv, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 1 / 1929935 (5.2e-05%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 1677183,
    "national_inv": 9999.0,
    "lead_time": 12.0,
    "in_transit_qty": 1029.0,
    "forecast_3_month": 13080.0,
    "forecast_6_month": 13080.0,
    "forecast_9_month": 18480.0,
    "sales_1_month": 2521.0,
    "sales_3_month": 9217.0,
    "sales_6_month": 16598.0,
    "sales_9_month": 26283.0,
    "min_bank": 1450.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.69,
    "perf_12_month_avg": 0.69,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Explicit missing values in lead_time

115617 of 1929935 data rows are NaN in lead_time (2 additional NaN rows are footer artifacts, excluded from this count).

- Affected rows: 115617 / 1929935 (5.99072%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3285085,
    "national_inv": 62.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3285131,
    "national_inv": 9.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286073,
    "national_inv": 0.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286113,
    "national_inv": 28.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286206,
    "national_inv": 2.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Possible numeric sentinel 9999 in forecast_9_month

1 rows equal 9999 in forecast_9_month, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 1 / 1929935 (5.2e-05%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 1729009,
    "national_inv": 1784.0,
    "lead_time": 3.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 2574.0,
    "forecast_6_month": 6281.0,
    "forecast_9_month": 9999.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Possible numeric sentinel 9999 in sales_9_month

1 rows equal 9999 in sales_9_month, a common missingness-sentinel value; not confirmed dominant enough to reclassify automatically.

- Affected rows: 1 / 1929935 (5.2e-05%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 2167478,
    "national_inv": 886.0,
    "lead_time": 16.0,
    "in_transit_qty": 385.0,
    "forecast_3_month": 5742.0,
    "forecast_6_month": 8961.0,
    "forecast_9_month": 13137.0,
    "sales_1_month": 780.0,
    "sales_3_month": 3334.0,
    "sales_6_month": 6963.0,
    "sales_9_month": 9999.0,
    "min_bank": 838.0,
    "potential_issue": "No",
    "pieces_past_due": 696.0,
    "perf_6_month_avg": 0.15,
    "perf_12_month_avg": 0.33,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Encoded-null numeric sentinel in perf_6_month_avg

148579 rows use -99.0 as a missingness sentinel in perf_6_month_avg (excluded from numeric-sanity negative-value findings to avoid double-counting).

- Affected rows: 148579 / 1929935 (7.698653%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3285085,
    "national_inv": 62.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3285131,
    "national_inv": 9.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286073,
    "national_inv": 0.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286113,
    "national_inv": 28.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286206,
    "national_inv": 2.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Encoded-null numeric sentinel in perf_12_month_avg

140025 rows use -99.0 as a missingness sentinel in perf_12_month_avg (excluded from numeric-sanity negative-value findings to avoid double-counting).

- Affected rows: 140025 / 1929935 (7.255426%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3285085,
    "national_inv": 62.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3285131,
    "national_inv": 9.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286073,
    "national_inv": 0.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286113,
    "national_inv": 28.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286206,
    "national_inv": 2.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 0.0,
    "sales_3_month": 0.0,
    "sales_6_month": 0.0,
    "sales_9_month": 0.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### Missingness clustering across column pairs

0 column pair(s) with NaN co-occurrence (Jaccard) >= 0.5, among 1 column(s) that have any missing values.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

### Missingness rate by categorical segment

Per-segment missingness rate for each column with missing values, broken down by each Yes/No flag column — a clustered pattern (rate far from the column's overall rate) usually means 'not collected for this segment'.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

## 3.3 Numeric Sanity

### Outlier statistics per numeric column

min/max/mean/median/p95/p99/p99.9 and max:p99.9 ratio for 15 numeric column(s); 12 flagged (ratio > 20.0x).

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

### Negative values in national_inv

6654 rows have a negative, non-sentinel value in national_inv (should be non-negative).

- Affected rows: 6654 / 1929935 (0.344778%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3466610,
    "national_inv": -78.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 9.0,
    "sales_3_month": 40.0,
    "sales_6_month": 128.0,
    "sales_9_month": 296.0,
    "min_bank": 27.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3291498,
    "national_inv": -2.0,
    "lead_time": 9.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 12.0,
    "forecast_6_month": 16.0,
    "forecast_9_month": 20.0,
    "sales_1_month": 0.0,
    "sales_3_month": 5.0,
    "sales_6_month": 8.0,
    "sales_9_month": 13.0,
    "min_bank": 2.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.94,
    "perf_12_month_avg": 0.83,
    "local_bo_qty": 2.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3333345,
    "national_inv": -1.0,
    "lead_time": 2.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 24.0,
    "forecast_6_month": 24.0,
    "forecast_9_month": 24.0,
    "sales_1_month": 22.0,
    "sales_3_month": 22.0,
    "sales_6_month": 22.0,
    "sales_9_month": 22.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.25,
    "perf_12_month_avg": 0.16,
    "local_bo_qty": 1.0,
    "deck_risk": "Yes",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3336471,
    "national_inv": -8.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 17.0,
    "forecast_6_month": 20.0,
    "forecast_9_month": 24.0,
    "sales_1_month": 2.0,
    "sales_3_month": 4.0,
    "sales_6_month": 9.0,
    "sales_9_month": 9.0,
    "min_bank": 7.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.5,
    "perf_12_month_avg": 0.46,
    "local_bo_qty": 8.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3365474,
    "national_inv": -1.0,
    "lead_time": 12.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 3.0,
    "forecast_6_month": 5.0,
    "forecast_9_month": 6.0,
    "sales_1_month": 3.0,
    "sales_3_month": 4.0,
    "sales_6_month": 5.0,
    "sales_9_month": 6.0,
    "min_bank": 2.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.31,
    "perf_12_month_avg": 0.4,
    "local_bo_qty": 2.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### lead_time == 0 with in_transit_qty > 0

2214 rows have lead_time == 0 while in_transit_qty > 0, which is not physically sensible (no lead time implies nothing should be in transit).

- Affected rows: 2214 / 1929935 (0.114719%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3358667,
    "national_inv": 247.0,
    "lead_time": 0.0,
    "in_transit_qty": 6.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 270.0,
    "sales_1_month": 52.0,
    "sales_3_month": 144.0,
    "sales_6_month": 234.0,
    "sales_9_month": 351.0,
    "min_bank": 56.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.0,
    "perf_12_month_avg": 0.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3424966,
    "national_inv": 11.0,
    "lead_time": 0.0,
    "in_transit_qty": 1.0,
    "forecast_3_month": 8.0,
    "forecast_6_month": 8.0,
    "forecast_9_month": 16.0,
    "sales_1_month": 1.0,
    "sales_3_month": 5.0,
    "sales_6_month": 11.0,
    "sales_9_month": 26.0,
    "min_bank": 2.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.0,
    "perf_12_month_avg": 0.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286278,
    "national_inv": 41.0,
    "lead_time": 0.0,
    "in_transit_qty": 10.0,
    "forecast_3_month": 627.0,
    "forecast_6_month": 1045.0,
    "forecast_9_month": 1672.0,
    "sales_1_month": 93.0,
    "sales_3_month": 384.0,
    "sales_6_month": 821.0,
    "sales_9_month": 1194.0,
    "min_bank": 51.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.0,
    "perf_12_month_avg": 0.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3286581,
    "national_inv": 109.0,
    "lead_time": 0.0,
    "in_transit_qty": 3.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 408.0,
    "forecast_9_month": 408.0,
    "sales_1_month": 16.0,
    "sales_3_month": 54.0,
    "sales_6_month": 119.0,
    "sales_9_month": 185.0,
    "min_bank": 22.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.0,
    "perf_12_month_avg": 0.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3288171,
    "national_inv": 332.0,
    "lead_time": 0.0,
    "in_transit_qty": 3.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 81.0,
    "forecast_9_month": 216.0,
    "sales_1_month": 27.0,
    "sales_3_month": 125.0,
    "sales_6_month": 413.0,
    "sales_9_month": 599.0,
    "min_bank": 45.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.0,
    "perf_12_month_avg": 0.0,
    "local_bo_qty": 1.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

### sales_1_month implausibly exceeds national_inv

20590 rows have sales_1_month > national_inv * 100 (sensitivity at other ratios shown in extra.sensitivity_by_ratio).

- Affected rows: 20590 / 1929935 (1.066875%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3327775,
    "national_inv": 0.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 1.0,
    "sales_3_month": 1.0,
    "sales_6_month": 2.0,
    "sales_9_month": 2.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3466610,
    "national_inv": -78.0,
    "lead_time": null,
    "in_transit_qty": 0.0,
    "forecast_3_month": 0.0,
    "forecast_6_month": 0.0,
    "forecast_9_month": 0.0,
    "sales_1_month": 9.0,
    "sales_3_month": 40.0,
    "sales_6_month": 128.0,
    "sales_9_month": 296.0,
    "min_bank": 27.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": -99.0,
    "perf_12_month_avg": -99.0,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "No",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3285175,
    "national_inv": 0.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 8.0,
    "forecast_6_month": 18.0,
    "forecast_9_month": 22.0,
    "sales_1_month": 2.0,
    "sales_3_month": 5.0,
    "sales_6_month": 15.0,
    "sales_9_month": 17.0,
    "min_bank": 0.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.87,
    "perf_12_month_avg": 0.8,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3291498,
    "national_inv": -2.0,
    "lead_time": 9.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 12.0,
    "forecast_6_month": 16.0,
    "forecast_9_month": 20.0,
    "sales_1_month": 0.0,
    "sales_3_month": 5.0,
    "sales_6_month": 8.0,
    "sales_9_month": 13.0,
    "min_bank": 2.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.94,
    "perf_12_month_avg": 0.83,
    "local_bo_qty": 2.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "Yes",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  },
  {
    "sku": 3295717,
    "national_inv": 0.0,
    "lead_time": 8.0,
    "in_transit_qty": 0.0,
    "forecast_3_month": 3.0,
    "forecast_6_month": 4.0,
    "forecast_9_month": 5.0,
    "sales_1_month": 1.0,
    "sales_3_month": 3.0,
    "sales_6_month": 4.0,
    "sales_9_month": 5.0,
    "min_bank": 1.0,
    "potential_issue": "No",
    "pieces_past_due": 0.0,
    "perf_6_month_avg": 0.99,
    "perf_12_month_avg": 0.97,
    "local_bo_qty": 0.0,
    "deck_risk": "No",
    "oe_constraint": "No",
    "ppap_risk": "No",
    "stop_auto_buy": "Yes",
    "rev_stop": "No",
    "went_on_backorder": "No"
  }
]
```

## 3.4 Categorical / Text Sanity

### Cardinality and top-20 values for flag columns

Cardinality and top-20 frequency table for 6 categorical flag column(s).

- Affected rows: 0 / 1929937 (0.0%)
- Serve-relevant: No

### Columns with near-unique (identifier-like) cardinality

1 text-like column(s) have nunique/non-null ratio >= 0.95 and nunique >= 1000. Expected to fire on `sku` (the real identifier); any other column here would be a genuine misfiled-as-categorical concern.

- Affected rows: 0 / 1929937 (0.0%)
- Serve-relevant: No

## 3.5 Cross-Column Consistency

### Sales window monotonicity violation: sales_3 >= sales_1

0 of 1929935 rows violate 'sales_3 >= sales_1' (violation rate 0.0); replicates the demand notebook's cumulative-window sanity check but reports the violation rate, not the pass rate.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: Yes

### Sales window monotonicity violation: sales_6 >= sales_3

0 of 1929935 rows violate 'sales_6 >= sales_3' (violation rate 0.0); replicates the demand notebook's cumulative-window sanity check but reports the violation rate, not the pass rate.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: Yes

### Sales window monotonicity violation: sales_9 >= sales_6

0 of 1929935 rows violate 'sales_9 >= sales_6' (violation rate 0.0); replicates the demand notebook's cumulative-window sanity check but reports the violation rate, not the pass rate.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: Yes

### min_bank exceeds national_inv (on-hand vs. reorder-point analog)

78205 of 1929935 rows have min_bank > national_inv. There is no explicit reorder-point column in this schema; min_bank is used as that analog here.

- Affected rows: 78205 / 1929935 (4.052209%)
- Serve-relevant: Yes

Examples:

```json
[
  {
    "sku": 3293714,
    "min_bank": 1.0,
    "national_inv": 0.0
  },
  {
    "sku": 3346720,
    "min_bank": 2.0,
    "national_inv": 0.0
  },
  {
    "sku": 3389806,
    "min_bank": 1.0,
    "national_inv": 0.0
  },
  {
    "sku": 3395221,
    "min_bank": 1.0,
    "national_inv": 0.0
  },
  {
    "sku": 3419639,
    "min_bank": 1.0,
    "national_inv": 0.0
  }
]
```

## 3.6 Target / Label Column

### Missingness in target column went_on_backorder

0 of 1929935 data rows have a missing target (2 additional NaNs are footer-artifact rows, a distinct category from real feature missingness).

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No

### Target class balance

went_on_backorder: No=1915954, Yes=13981, imbalance ratio No:Yes = 137.03984:1.

- Affected rows: 13981 / 1929935 (0.724429%)
- Serve-relevant: No

### Columns suspiciously correlated with target availability (manual review)

0 column(s) flagged for manual review because their missingness/non-null rate differs by more than 20% between target classes. Flagged only — never auto-excluded.

- Affected rows: 0 / 1929935 (0.0%)
- Serve-relevant: No
