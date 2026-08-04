"""
Example script showing how to call the Supplier Auditor inference tool (Agent 4).

Run with:
    ./venv/bin/python run_supplier_auditor_example.py

Note: this tool's required columns EXCLUDE `stop_auto_buy` (it is the target)
but INCLUDE `went_on_backorder` (a raw passthrough feature in the saved model).
"""

from inference_tools.supplier_auditor_tool import audit_suppliers
from inference_tools.schemas import SupplierAuditorInput


def _row(perf_6, perf_12, pieces_past_due, local_bo_qty):
    """A full raw feature row (all 25 columns the Supplier Auditor needs)."""
    return {
        'national_inv': 100.0, 'lead_time': 8.0, 'in_transit_qty': 40.0,
        'forecast_3_month': 70.0, 'forecast_6_month': 140.0, 'forecast_9_month': 210.0,
        'sales_1_month': 20.0, 'sales_3_month': 60.0, 'sales_6_month': 120.0,
        'sales_9_month': 180.0, 'min_bank': 25.0, 'potential_issue': 0,
        'pieces_past_due': pieces_past_due, 'perf_6_month_avg': perf_6,
        'perf_12_month_avg': perf_12, 'local_bo_qty': local_bo_qty,
        'deck_risk': 0, 'oe_constraint': 0, 'ppap_risk': 0, 'rev_stop': 0,
        'went_on_backorder': 0, 'inv_velocity': 0.67, 'safety_gap': 75.0,
        'sales_trend': 0.0, 'perf_gap': round(1.0 - perf_6, 4),
    }


input_data = SupplierAuditorInput(
    skus=['SKU-001', 'SKU-002', 'SKU-003'],
    data={
        'SKU-001': _row(perf_6=0.95, perf_12=0.93, pieces_past_due=0.0, local_bo_qty=0.0),
        'SKU-002': _row(perf_6=0.55, perf_12=0.60, pieces_past_due=40.0, local_bo_qty=30.0),
        'SKU-003': _row(perf_6=0.80, perf_12=0.82, pieces_past_due=5.0, local_bo_qty=2.0),
    },
)

output = audit_suppliers(input_data)

print(f"Model version: {output.model_version}\n")
print(f"{'SKU':<10} {'risk_prob':>9} {'grade':>6} {'stop_auto_buy':>14} {'reason':>11}")
for r in output.results:
    print(f"{r.sku:<10} {r.risk_probability:>9.4f} {r.supplier_grade:>6} "
          f"{str(r.stop_auto_buy_triggered):>14} {r.trigger_reason:>11}")
