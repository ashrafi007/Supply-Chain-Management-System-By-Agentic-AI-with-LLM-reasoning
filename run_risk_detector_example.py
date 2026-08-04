"""
Example script showing how to call the Risk Detector inference tool (Agent 2).

Run with:
    ./venv/bin/python run_risk_detector_example.py
"""

from inference_tools.risk_detector_tool import predict_risk
from inference_tools.schemas import RiskDetectorInput


def _row(national_inv, pieces_past_due, deck_risk, local_bo_qty):
    """A full raw feature row (all 25 columns the Risk Detector needs)."""
    return {
        'national_inv': national_inv, 'lead_time': 8.0, 'in_transit_qty': 40.0,
        'forecast_3_month': 70.0, 'forecast_6_month': 140.0, 'forecast_9_month': 210.0,
        'sales_1_month': 20.0, 'sales_3_month': 60.0, 'sales_6_month': 120.0,
        'sales_9_month': 180.0, 'min_bank': 25.0, 'potential_issue': 0,
        'pieces_past_due': pieces_past_due, 'perf_6_month_avg': 0.85,
        'perf_12_month_avg': 0.82, 'local_bo_qty': local_bo_qty,
        'deck_risk': deck_risk, 'oe_constraint': 0, 'ppap_risk': 0,
        'stop_auto_buy': 0, 'rev_stop': 0, 'inv_velocity': 0.67,
        'safety_gap': national_inv - 25.0, 'sales_trend': 0.0, 'perf_gap': 0.05,
    }


input_data = RiskDetectorInput(
    skus=['SKU-001', 'SKU-002', 'SKU-003'],
    data={
        'SKU-001': _row(national_inv=100.0, pieces_past_due=0.0, deck_risk=0, local_bo_qty=0.0),
        'SKU-002': _row(national_inv=3.0, pieces_past_due=30.0, deck_risk=1, local_bo_qty=25.0),
        'SKU-003': _row(national_inv=200.0, pieces_past_due=0.0, deck_risk=0, local_bo_qty=0.0),
    },
)

output = predict_risk(input_data)

print(f"Model version: {output.model_version}   threshold: {output.threshold_used}\n")
print(f"{'SKU':<10} {'P(backorder)':>13} {'label':>7} {'high_risk':>10}")
for p in output.predictions:
    print(f"{p.sku:<10} {p.backorder_probability:>13.4f} {str(p.predicted_label):>7} {str(p.is_high_risk):>10}")
print(f"\nHigh-risk SKUs: {output.high_risk_skus}")
