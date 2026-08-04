"""
Example script showing how to call the Forecast Optimizer inference tool (Agent 3).

Run with:
    ./venv/bin/python run_forecast_optimizer_example.py

`backorder_probability` and `alarm_triggered` normally come from the Risk
Detector tool's output via the orchestrator; here they are supplied by hand.
"""

from inference_tools.forecast_optimizer_tool import optimize_forecast
from inference_tools.schemas import ForecastOptimizerInput


def _row(national_inv, forecast_3_month, sales_3_month):
    """A full raw feature row (all 25 columns the Forecast Optimizer needs)."""
    return {
        'national_inv': national_inv, 'lead_time': 8.0, 'in_transit_qty': 40.0,
        'forecast_3_month': forecast_3_month, 'forecast_6_month': 140.0,
        'forecast_9_month': 210.0, 'sales_1_month': 20.0, 'sales_3_month': sales_3_month,
        'sales_6_month': 120.0, 'sales_9_month': 180.0, 'min_bank': 25.0,
        'potential_issue': 0, 'pieces_past_due': 5.0, 'perf_6_month_avg': 0.85,
        'perf_12_month_avg': 0.82, 'local_bo_qty': 0.0, 'deck_risk': 0,
        'oe_constraint': 0, 'ppap_risk': 0, 'stop_auto_buy': 0, 'rev_stop': 0,
        'inv_velocity': 0.67, 'safety_gap': national_inv - 25.0,
        'sales_trend': 0.0, 'perf_gap': 0.05,
    }


input_data = ForecastOptimizerInput(
    skus=['SKU-001', 'SKU-002', 'SKU-003'],
    data={
        'SKU-001': _row(national_inv=100.0, forecast_3_month=120.0, sales_3_month=60.0),
        'SKU-002': _row(national_inv=5.0, forecast_3_month=30.0, sales_3_month=60.0),
        'SKU-003': _row(national_inv=200.0, forecast_3_month=70.0, sales_3_month=65.0),
    },
    backorder_probability={'SKU-001': 0.10, 'SKU-002': 0.65, 'SKU-003': 0.03},
    alarm_triggered={'SKU-001': False, 'SKU-002': True, 'SKU-003': False},
)

output = optimize_forecast(input_data)

print(f"Model versions: A={output.model_version_a}  B={output.model_version_b}\n")
print(f"{'SKU':<10} {'human_3m':>9} {'adjusted':>9} {'corr':>6} {'bias':>6} {'recommendation':>16}")
for r in output.recommendations:
    print(f"{r.sku:<10} {r.human_forecast_3m:>9.1f} {r.adjusted_forecast_3m:>9.1f} "
          f"{r.correction_factor:>6.2f} {str(r.bias_detected):>6} {r.recommendation:>16}")
