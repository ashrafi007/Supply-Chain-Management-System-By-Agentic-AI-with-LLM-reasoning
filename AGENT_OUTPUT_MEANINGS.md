# Agent Output Meanings — Reference for LLM Feeding

This document exists so an LLM consuming pipeline output knows, for every field every agent can return, **every possible value that field can take and what it means in plain business terms**. It is built from the training notebooks (`Notebooks/`, excluding `Data_Preprocessing`) cross-checked against the actual production code that ships (`inference_tools/*.py`, `inference_tools/schemas.py`, `src/orchestrator/nodes/*.py`). Where the notebook and shipped code disagree, the shipped code wins and the notebook-only behavior is called out explicitly.

Pipeline order: Agent 1 → Agent 2 → Agent 3 → Agent 5 → Agent 6 (Agent 4 "Routing" was cancelled and does not exist).

---

## Agent 1: Demand Predictor

**Model:** LightGBM log-ratio regressor over a naive baseline (`sales_3_month × 2`), with Duan smearing correction.

> ⚠️ **Known production gap:** the notebook defines three output fields (`demand_forecast`, `demand_velocity_band`, `stockout_risk`), but the shipped wrapper (`inference_tools/demand_predictor_tool.py`) currently computes and returns **only `demand_forecast`**. `demand_velocity_band` and `stockout_risk` exist as DB columns and are described below for completeness, but today they are always `None` in real pipeline runs — the notebook logic that would compute them has not been ported into production code.

### `demand_forecast` (numeric — **the only field actually populated today**)
Point forecast of `sales_6_month`: units expected to sell over the next 6 months. Rounded to 2 decimals, never negative (clipped at 0). Heavy-tailed distribution (most SKUs low, a few very high). `None` if a required input is missing or non-finite.

### `demand_velocity_band` (categorical — notebook-only, not shipped)
Bucketed from the forecast via `pd.cut(predicted_demand, bins=[-.1, 0, 10, 50, 200, 1000, inf])`:

| Value | Range | Means |
|---|---|---|
| `Dead` | = 0 | No forecast demand at all → delisting candidate. **Caveat: the notebook itself warns this band is virtually unreachable** — the model almost never outputs exactly 0, so true dead-stock SKUs usually land in `Trickle` instead. Don't trust `Dead` for delisting decisions. |
| `Trickle` | 1–10 | Very low demand → candidate for make-to-order instead of stocked inventory. |
| `Slow` | 10–50 | Low demand → review whether the minimum inventory bank is set too high. |
| `Steady` | 50–200 | Normal demand → standard replenishment cadence, no special action. |
| `Fast` | 200–1000 | High demand → prioritize this SKU in Agent 3's rebalancing queue. |
| `Surge` | >1000 | Very high demand → escalate to production/procurement planning immediately. |

### `stockout_risk` (0/1 — notebook-only, not shipped)
`cover_ratio = national_inv / (predicted_demand + 1)`; `stockout_risk = 1 if cover_ratio < 0.5 else 0`.
- **`1`** = current on-hand inventory covers less than half of forecasted 6-month demand → **this SKU is at high risk of running out of stock / going on backorder**.
- **`0`** = inventory covers at least half of forecasted demand → not currently flagged.

---

## Agent 2: Risk Detector

**Model:** stacking ensemble classifier. Target: `went_on_backorder` (`0` = Safe, `1` = went on backorder).

### `backorder_probability` (numeric, 0.0–1.0)
Probability this SKU will go on backorder / stock out. Higher = riskier.

### `predicted_label` (boolean)
- **`True`**: probability ≥ 0.5 (generic midpoint). Model leans toward "will backorder."
- **`False`**: probability < 0.5. Model leans toward "safe."
- This is *not* the operational flag — see `is_high_risk` below. With backorders being a rare (~0.7%) class, this midpoint threshold is rarely the actionable signal on its own.

### `is_high_risk` (boolean) — **the operational flag used downstream**
- **`True`**: probability ≥ 0.945 (the F2-optimized threshold — tuned to weight missed stockouts as costlier than false alarms). **Means the SKU should be actively treated as high risk of backorder/stockout**, and it feeds Agent 3's urgency scoring.
- **`False`**: below 0.945 — not flagged as high-risk, even if `predicted_label` was `True`. (There's a real band between 0.5 and 0.945 where the model leans "will backorder" but isn't confident enough to trigger action.)

### `high_risk_skus` (list)
Convenience list of every SKU where `is_high_risk == True`. Not an independent signal.

### `threshold_used` (numeric)
Echoes the exact cutoff applied (0.945 in production) — audit/provenance field.

### `model_version` (string)
Provenance identifier (e.g. `"stacking_ensemble_v1"`), not a business signal.

**What survives to the rest of the pipeline:** `backorder_prob` and `alarm_triggered` (`1`="high risk / alarm on", `0`="not flagged" — this is `is_high_risk` renamed).

---

## Agent 3: Inventory Rebalancer

**Model:** XGBoost regressor predicting a composite urgency score.

### `urgency_score_predicted` (numeric, 0.0–1.0, continuous)
Predicted urgency to restock this SKU. `0` = no urgency, `1` = maximum urgency. Composite of: how far inventory is below the safety floor (40%), backorder probability (35%), depletion rate (15%), safety-stock urgency (10%). **There are no discrete tiers in the model itself** — any "critical/high/medium/low" banding is an interpretation layer, not something the model outputs.

### `recommended_qty` (numeric, units, ≥0)
Units needed to bring stock back up to the safety floor: `max(min_bank - national_inv, 0)`.
- **`0`** = SKU is already at or above its safety floor — no reorder needed.
- **`> 0`** = order this many units to reach the safety floor.
- Can be `NaN` if the underlying raw inputs were `NaN` — this is an intentional passthrough, not an error.

### `batch_rank` (integer, 1..N)
This SKU's urgency rank *within the current batch of SKUs sent to the model*, not a global rank across the whole dataset. `1` = most urgent in this batch.

### `manufacture_rank` (integer or null)
**Always `null` today** — intended as a global percentile rank across the entire dataset, but the reference-distribution artifact needed to compute it doesn't exist yet. Treat this as permanently unusable, not as "rank 1" or "no ranking needed."

### `top_priority_skus` (list, top-level)
SKUs flagged for priority action. A SKU appears here when **both**: `recommended_qty > 0` AND it's in a "critical" state (months of stock remaining < lead time, AND already below safety floor, AND forecasted demand exceeds available + incoming inventory). **Means: this SKU will run out before a new order could even arrive — restock immediately.**

---

## Agent 5: Forecast Optimizer

**Models:** LightGBM classifier ("bias detector") + XGBoost regressor ("corrector"), applied together.

### `human_forecast_3m` (numeric)
The original planner/human 3-month forecast, unchanged.

### `adjusted_forecast_3m` (numeric)
`human_forecast_3m × correction_factor` — the corrected forecast planners should actually use.

### `correction_factor` (numeric, 0.3–1.5)
Multiplier applied to the human forecast.
- **`< 1.0`**: human forecast was too high, being corrected downward.
- **`= 1.0`**: no correction needed (or overridden — see `risk_override_applied`).
- **`> 1.0`**: human forecast was too low, being corrected upward.

### `bias_probability` (numeric, 0.0–1.0)
Model's confidence that the human forecast is chronically overestimating demand.

### `bias_severity` (categorical)
| Value | Condition | Means |
|---|---|---|
| `NONE` | correction_factor ≥ 0.90 | No material bias in the human forecast. |
| `MILD` | 0.75 ≤ correction_factor < 0.90 | Moderate overestimate — worth noting. |
| `SEVERE` | correction_factor < 0.75 | Large overestimate — human forecast is significantly too high. |

### `bias_detected` (boolean)
- **`True`**: `bias_probability ≥ 0.765` — classified as a chronic over-forecaster (training rule: forecast was >15% above actual sales historically).
- **`False`**: not flagged as biased.

### `risk_override_applied` (boolean)
- **`True`**: Agent 2 had flagged this SKU as high backorder risk (`alarm_triggered`), which **forces `correction_factor` back to 1.0** regardless of what the corrector model predicted — a deliberate business rule: never shrink the forecast for a SKU that's already at risk of stocking out.
- **`False`**: no override; the model's own correction stands.

### `recommendation` (categorical)
| Value | Condition | Means |
|---|---|---|
| `REDUCE_PLAN` | correction_factor < 0.90 | Cut the production/purchasing plan. |
| `INCREASE_PLAN` | correction_factor > 1.10 | Raise the production/purchasing plan. |
| `HOLD` | 0.90–1.10 | Forecast is roughly accurate — no action needed. |

### `flagged_for_reduction` / `flagged_for_increase` (lists, top-level)
SKUs with `REDUCE_PLAN` / `INCREASE_PLAN` respectively.

**Note:** if Agent 2 was skipped for a SKU, Agent 5 is skipped too and `correction_factor` defaults to `1.0` (no correction, no bias check performed).

---

## Agent 6: Supplier Auditor

**Model:** XGBoost classifier. Target: whether automatic purchasing from this supplier should stop (`stop_auto_buy`).

### `risk_probability` (numeric, 0.0–1.0)
Model's predicted probability that auto-buy should be stopped for this supplier. Higher = riskier.

### `delivery_stress` (numeric, ≥0, unbounded)
`pieces_past_due / national_inv` — ratio of overdue delivery volume to on-hand inventory. Feeds the rule-based override below; higher = more overdue deliveries relative to what's in stock.

### `supplier_grade` (categorical: A/B/C/D)
Probability band:

| Grade | Range | Means |
|---|---|---|
| `A` | [0.00, 0.20) | **Approved** — no action needed. |
| `B` | [0.20, 0.45) | **Watch** — schedule a contract review. |
| `C` | [0.45, 0.70) | **At Risk** — escalate to procurement. |
| `D` | [0.70, 1.00] | **Critical** — this is the band where auto-buy can actually be stopped. |

### `stop_auto_buy_triggered` (boolean)
**True** if EITHER: model probability ≥ decision threshold, OR (`supplier_grade == D` AND `delivery_stress > 0.5`, a rule-based safety net for severe overdue-delivery backlogs the model alone might miss). **Means: halt automatic purchase orders from this supplier** — human review required before further auto-buying.

### `trigger_reason` (categorical)
| Value | Means |
|---|---|
| `MODEL+RULE` | Both the ML model and the business rule agree — strongest stop signal. |
| `MODEL` | Only the ML probability crossed threshold. |
| `RULE` | Only the business rule fired (grade D + high delivery stress) even though the model probability was below threshold — catches a specific failure pattern (severe past-due backlog) the model alone missed. |
| `NONE` | Neither condition fired — not flagged. |

### `flagged_suppliers` (list, top-level)
SKUs/suppliers where `stop_auto_buy_triggered == True`.

### `grade_distribution` (dict: A/B/C/D → count)
How many SKUs fell into each grade this run — a summary stat, not a per-SKU signal.

**What survives to the rest of the pipeline:** only the letter grade (`supplier_risk`). The richer fields (`risk_probability`, `trigger_reason`, `delivery_stress`) are not currently propagated past this agent's own trace.

---

## Quick reference: "what does X=1 mean" cheat sheet

- **`stockout_risk = 1`** (Agent 1, not yet in production) → inventory covers under half of forecasted demand, at risk of running out.
- **`is_high_risk = 1` / `alarm_triggered = 1`** (Agent 2) → SKU is flagged high risk of backorder/stockout; probability ≥ 0.945.
- **`recommended_qty = 0`** (Agent 3) → SKU is already sufficiently stocked, no reorder needed.
- **SKU in `top_priority_skus`** (Agent 3) → will stock out before a reorder could even arrive; restock now.
- **`bias_detected = 1`** (Agent 5) → the human forecast has been chronically too high for this SKU.
- **`risk_override_applied = 1`** (Agent 5) → forecast correction was cancelled because Agent 2 flagged this SKU as backorder-risk; never shrink a forecast for an at-risk SKU.
- **`stop_auto_buy_triggered = 1`** (Agent 6) → stop auto-purchasing from this supplier, human review needed.
- **`supplier_grade = D`** (Agent 6) → critical supplier, the band where auto-buy can be stopped.
