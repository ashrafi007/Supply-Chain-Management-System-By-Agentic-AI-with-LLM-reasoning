"""Synthetic (non-DB) PipelineState fixtures for orchestrator-only tests -- no repository/DB dependency needed here."""

from __future__ import annotations

import pytest

# The 21 raw columns state_builder.py would put in raw_features for a real
# inventory_current row. The 5 gap-fill columns (safety_gap, perf_gap,
# inv_velocity, sales_trend, went_on_backorder) are deliberately absent here --
# LangGraphExecutor.invoke() computes/defaults them via features.fill_engineered_columns.
def _sample_raw_features() -> dict:
    return dict(
        national_inv=500.0,
        lead_time=10.0,
        in_transit_qty=50.0,
        forecast_3_month=300.0,
        forecast_6_month=600.0,
        forecast_9_month=900.0,
        sales_1_month=100.0,
        sales_3_month=280.0,
        sales_6_month=550.0,
        sales_9_month=820.0,
        min_bank=80.0,
        potential_issue=0,
        pieces_past_due=0.0,
        perf_6_month_avg=0.9,
        perf_12_month_avg=0.85,
        local_bo_qty=0.0,
        deck_risk=0,
        oe_constraint=0,
        ppap_risk=0,
        stop_auto_buy=0,
        rev_stop=0,
    )


@pytest.fixture()
def sample_state() -> dict:
    return dict(
        sku_id="TEST-SKU-0001",
        raw_features=_sample_raw_features(),
        demand_forecast=None,
        demand_velocity_band=None,
        stockout_risk=None,
        backorder_prob=None,
        alarm_triggered=None,
        urgency_score=None,
        correction_factor=None,
        replenishment_qty=None,
        route=None,
        supplier_risk=None,
        recommendation=None,
        trace=[],
        errors=[],
    )
