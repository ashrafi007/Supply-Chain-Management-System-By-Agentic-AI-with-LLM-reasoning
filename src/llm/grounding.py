"""
Grounding data for LLM explanations. Source of truth: AGENT_OUTPUT_MEANINGS.md.
Do not add any meaning here that isn't traceable to that document.

Deviation from llm_insertion_spec.md SS5: the spec's draft used keys like
"agent_1_demand"/"agent_2_risk"/etc, but the actual values written to
agent_traces.agent_name (see NODE_TO_AGENT_NAME in src/orchestrator/graph.py) are
"demand_predictor"/"risk_detector"/"inventory_rebalancer"/"forecast_optimizer"/
"supplier_auditor". explainer_service looks up grounding by the real agent_name
column, and test_grounding.py's own acceptance check (SS12.1: "every agent name in
agent_traces has a top-level key in AGENT_GROUNDING") requires the match -- so the
keys below are the real agent names. Field-level content is unchanged from the spec.
"""

AGENT_GROUNDING = {

    "demand_predictor": {
        "shipped_fields": ["demand_forecast"],
        "not_shipped": ["demand_velocity_band", "stockout_risk"],
        "demand_forecast": {
            "description": (
                "Point forecast of units expected to sell over the next 6 months "
                "(sales_6_month), rounded to 2 decimals, never negative."
            ),
        },
    },

    "risk_detector": {
        "shipped_fields": ["backorder_prob", "alarm_triggered"],
        "backorder_prob": {
            "description": "Probability this SKU will go on backorder / stock out. Higher = riskier.",
        },
        "alarm_triggered": {
            "description": (
                "Operational high-risk flag. True when the underlying probability >= 0.945 "
                "(the F2-optimized threshold, tuned to weight missed stockouts as costlier than false alarms). "
                "This is NOT the generic 0.5 midpoint -- there is a real band between 0.5 and 0.945 where the "
                "model leans toward backorder but is not confident enough to trigger action."
            ),
            "threshold": 0.945,
        },
    },

    "inventory_rebalancer": {
        "shipped_fields": ["urgency_score", "recommended_qty", "batch_rank", "top_priority_skus"],
        "urgency_score": {
            "description": (
                "Composite urgency to restock, 0.0-1.0. Composed of: inventory gap below safety floor (40%), "
                "Agent 2's backorder probability (35%), depletion rate (15%), safety-stock urgency (10%). "
                "There are no discrete tiers in the model itself -- any critical/high/medium/low banding is "
                "an interpretation layer, not a model output."
            ),
        },
        "recommended_qty": {
            "description": (
                "Units needed to bring stock to the safety floor: max(min_bank - national_inv, 0). "
                "0 means already at or above the safety floor; NaN is an intentional passthrough from "
                "NaN raw inputs, not an error."
            ),
        },
        "batch_rank": {
            "description": "This SKU's urgency rank within the current batch only, not a global rank.",
        },
        "manufacture_rank": {
            "description": (
                "Always null today -- the global reference-distribution artifact needed to compute it "
                "does not exist yet. Permanently unusable in the current system, not '1' or 'no ranking needed'."
            ),
        },
        "top_priority_skus_membership": {
            "description": (
                "A SKU appears here when recommended_qty > 0 AND it is in a critical state: months of stock "
                "remaining is less than lead time, it is already below safety floor, and forecasted demand "
                "exceeds available plus incoming inventory. Means this SKU will run out before a new order "
                "could even arrive -- restock immediately."
            ),
        },
    },

    "forecast_optimizer": {
        "shipped_fields": ["correction_factor", "adjusted_forecast_3m", "bias_detected",
                           "risk_override_applied", "recommendation"],
        "correction_factor": {
            "description": (
                "Multiplier applied to the human 3-month forecast, range 0.3-1.5. Below 1.0 means the human "
                "forecast was too high and is corrected downward; above 1.0 means it was too low and is "
                "corrected upward; exactly 1.0 means no correction, OR that Agent 2's risk override forced it."
            ),
        },
        "bias_severity": {
            "values": {
                "NONE": "correction_factor >= 0.90 -- no material bias in the human forecast.",
                "MILD": "0.75 <= correction_factor < 0.90 -- moderate overestimate, worth noting.",
                "SEVERE": "correction_factor < 0.75 -- large overestimate, human forecast significantly too high.",
            },
        },
        "bias_detected": {
            "description": (
                "True when bias_probability >= 0.765 -- classified as a chronic over-forecaster "
                "(training rule: forecast >15% above actual sales historically)."
            ),
            "threshold": 0.765,
        },
        "risk_override_applied": {
            "description": (
                "True means Agent 2 flagged this SKU as high backorder risk (alarm_triggered), which forces "
                "correction_factor back to 1.0 regardless of what the corrector model predicted -- a deliberate "
                "rule: never shrink the forecast for a SKU already at risk of stocking out. If Agent 2 was "
                "skipped for this SKU, Agent 5 is skipped too and correction_factor defaults to 1.0 with no "
                "bias check performed at all."
            ),
        },
        "recommendation": {
            "values": {
                "REDUCE_PLAN": "correction_factor < 0.90 -- cut the production/purchasing plan.",
                "INCREASE_PLAN": "correction_factor > 1.10 -- raise the production/purchasing plan.",
                "HOLD": "0.90-1.10 -- forecast is roughly accurate, no action needed.",
            },
        },
    },

    "supplier_auditor": {
        "shipped_fields": ["supplier_risk"],
        "supplier_risk": {
            "description": "Supplier risk grade, letter A-D, based on the model's predicted stop_auto_buy probability.",
            "values": {
                "A": "[0.00, 0.20) -- Approved, no action needed.",
                "B": "[0.20, 0.45) -- Watch, schedule a contract review.",
                "C": "[0.45, 0.70) -- At Risk, escalate to procurement.",
                "D": "[0.70, 1.00] -- Critical, the band where auto-buy can actually be stopped.",
            },
        },
        "stop_auto_buy_triggered": {
            "description": (
                "True if EITHER the model probability crosses the decision threshold, OR "
                "(supplier_grade == D AND delivery_stress > 0.5) -- a rule-based safety net for severe "
                "overdue-delivery backlogs the model alone might miss. Means: halt automatic purchase orders "
                "from this supplier, human review required."
            ),
        },
        "trigger_reason": {
            "values": {
                "MODEL+RULE": "Both the ML model and the business rule agree -- strongest stop signal.",
                "MODEL": "Only the ML probability crossed threshold.",
                "RULE": "Only the business rule fired (grade D + high delivery stress) despite the model "
                        "probability being below threshold -- catches a failure pattern the model alone missed.",
                "NONE": "Neither condition fired -- not flagged.",
            },
        },
    },
}
