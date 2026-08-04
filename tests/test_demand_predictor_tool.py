"""
Parity + interface tests for the Demand Predictor inference tool (spec §8, §9).

The fixture ``tests/fixtures/agent_1_reference.parquet`` holds sampled raw rows
(`_raw_idx` + the raw input columns) and ``notebook_demand`` — the demand the
notebook's own inference code (cell 12 engineer_features + cell 16 from_delta,
at best_iteration with the val-tuned smearing factor) produces for the frozen
bundle. If the tool reproduces those values it is serving the same estimator
the paper reports.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference_tools.demand_predictor_tool import DemandPredictorTool

FIXTURE = Path(__file__).parent / "fixtures" / "agent_1_reference.parquet"
RAW_COLS = [
    "national_inv", "lead_time", "in_transit_qty", "min_bank", "perf_6_month_avg",
    "perf_12_month_avg", "local_bo_qty", "deck_risk", "oe_constraint", "ppap_risk",
    "stop_auto_buy", "rev_stop", "forecast_3_month", "sales_1_month", "sales_3_month",
    "safety_gap", "perf_gap", "inv_velocity", "sales_trend",
]


@pytest.fixture(scope="module")
def tool():
    return DemandPredictorTool(stage="h6")


@pytest.fixture(scope="module")
def reference():
    return pd.read_parquet(FIXTURE)


def test_bundle_loaded_once_and_correct(tool):
    b = tool.bundle
    assert b.stage == "h6"
    assert b.target == "sales_6_month"
    assert b.baseline_name == "sales_3_month x 2"
    assert b.best_iteration == 29178
    assert len(b.feature_cols) == 31
    assert b.feature_cols[-2:] == ["base_pred", "log_base_pred"]


def test_parity_batch(tool, reference):
    """The whole reference batch must match the notebook at rtol=1e-5."""
    X = reference[RAW_COLS]
    tool_demand = tool.predict(X)
    notebook_demand = reference["notebook_demand"].to_numpy()
    assert tool_demand.shape == notebook_demand.shape
    assert np.allclose(tool_demand, notebook_demand, rtol=1e-5, atol=1e-8), (
        "max abs diff = %.3e" % np.max(np.abs(tool_demand - notebook_demand))
    )


def test_single_row_equals_batch(tool, reference):
    """
    Running each row alone must equal that row's value from the batch. A mismatch
    would mean per-call state (e.g. a refit transformer) is leaking in — the exact
    bug §1/§3 warn about. Checked on a spread of rows incl. the extremes.
    """
    X = reference[RAW_COLS]
    batch = tool.predict(X)
    check_idx = [0, 1, len(X) // 2, len(X) - 2, len(X) - 1,
                 int(np.argmax(batch)), int(np.argmin(batch))]
    for i in sorted(set(check_idx)):
        one = tool.predict(X.iloc[[i]].to_dict(orient="records")[0])
        assert one.shape == (1,)
        assert one[0] == batch[i], f"row {i}: single {one[0]} != batch {batch[i]}"


def test_run_state_delta_shape(tool, reference):
    """run(state) returns a clean {demand_forecast: float} delta + a trace entry."""
    row = reference[RAW_COLS].iloc[0].to_dict()
    out = tool.run({"raw_features": row, "trace": []})
    assert set(out) == {"demand_forecast", "trace"}
    assert isinstance(out["demand_forecast"], float)
    assert out["demand_forecast"] == reference["notebook_demand"].iloc[0]
    assert out["trace"][-1]["tool"] == "demand_predictor"
    assert out["trace"][-1]["status"] == "ok"


def test_run_appends_to_existing_trace(tool, reference):
    row = reference[RAW_COLS].iloc[0].to_dict()
    out = tool.run({"raw_features": row, "trace": [{"tool": "upstream", "status": "ok"}]})
    assert len(out["trace"]) == 2
    assert out["trace"][0]["tool"] == "upstream"


def test_missing_column_fails_cleanly(tool, reference):
    """Missing required raw column -> demand_forecast=None, reason in trace, no raise."""
    row = reference[RAW_COLS].iloc[0].to_dict()
    del row["sales_3_month"]  # drives both the baseline and forecast_bias
    out = tool.run({"raw_features": row, "trace": []})
    assert out["demand_forecast"] is None
    assert out["trace"][-1]["status"] == "failed"
    assert "sales_3_month" in out["trace"][-1]["reason"]


def test_nan_in_matrix_fails_loudly(tool, reference):
    """A NaN surviving into a baseline-driving column is a cleaning gap -> fail."""
    row = reference[RAW_COLS].iloc[0].to_dict()
    row["sales_3_month"] = float("nan")
    out = tool.run({"raw_features": row, "trace": []})
    assert out["demand_forecast"] is None
    assert out["trace"][-1]["status"] == "failed"


def test_predict_missing_column_raises(tool, reference):
    row = reference[RAW_COLS].iloc[0].to_dict()
    del row["national_inv"]
    with pytest.raises(ValueError, match="missing required raw column"):
        tool.predict(row)


def test_cpu_only_no_gpu_device(tool):
    """Serving must not force a GPU path."""
    params = tool.bundle.model.get_params()
    assert params.get("device") in (None, "cpu")
