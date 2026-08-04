"""
Example script showing how to call the Inventory Rebalancer inference tool.

Run with:
    ./venv/bin/python3 run_rebalancer_example.py
"""

from inference_tools.inventory_rebalancer_tool import rebalance_inventory
from inference_tools.schemas import InventoryRebalancerInput

# 1. Build the input: SKUs, raw feature rows, and backorder_probability
#    (backorder_probability normally comes from the Risk Detector tool's output).
input_data = InventoryRebalancerInput(
    skus=['SKU-001', 'SKU-002', 'SKU-003'],
    data={
        'SKU-001': {
            'national_inv': 100.0, 'in_transit_qty': 50.0, 'sales_1_month': 20.0,
            'sales_3_month': 60.0, 'sales_9_month': 180.0, 'pieces_past_due': 5.0,
            'deck_risk': 0, 'oe_constraint': 0, 'ppap_risk': 0, 'forecast_3_month': 70.0,
            'lead_time': 5.0, 'perf_gap': 0.05, 'min_bank': 20.0,
            'perf_6_month_avg': 0.85, 'perf_12_month_avg': 0.80,
        },
        'SKU-002': {
            'national_inv': 5.0, 'in_transit_qty': 10.0, 'sales_1_month': 15.0,
            'sales_3_month': 45.0, 'sales_9_month': 135.0, 'pieces_past_due': 12.0,
            'deck_risk': 1, 'oe_constraint': 0, 'ppap_risk': 0, 'forecast_3_month': 60.0,
            'lead_time': 7.0, 'perf_gap': 0.15, 'min_bank': 30.0,
            'perf_6_month_avg': 0.70, 'perf_12_month_avg': 0.72,
        },
        'SKU-003': {
            'national_inv': 200.0, 'in_transit_qty': 20.0, 'sales_1_month': 10.0,
            'sales_3_month': 30.0, 'sales_9_month': 90.0, 'pieces_past_due': 0.0,
            'deck_risk': 0, 'oe_constraint': 0, 'ppap_risk': 0, 'forecast_3_month': 25.0,
            'lead_time': 3.0, 'perf_gap': 0.02, 'min_bank': 15.0,
            'perf_6_month_avg': 0.95, 'perf_12_month_avg': 0.93,
        },
    },
    backorder_probability={
        'SKU-001': 0.10,
        'SKU-002': 0.65,
        'SKU-003': 0.03,
    }
)

# 2. Run inference (artifacts load once and are cached for subsequent calls).
output = rebalance_inventory(input_data)

# 3. Use the results.
print(f"Model version: {output.model_version}\n")
print(f"{'SKU':<10} {'Urgency':>8} {'Qty':>8} {'BatchRank':>10}")
for rec in output.recommendations:
    print(f"{rec.sku:<10} {rec.urgency_score_predicted:>8.3f} {rec.recommended_qty:>8.1f} {rec.batch_rank:>10}")

print(f"\nTop priority SKUs: {output.top_priority_skus}")
