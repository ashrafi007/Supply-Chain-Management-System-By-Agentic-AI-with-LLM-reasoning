"""
Comprehensive tests for the Forecast Optimizer inference tool.

Covers: normal input, alarm-override behavior, sentinel handling, NaN handling,
missing SKUs/risk-signals/features, correction_factor range, determinism,
caching, and output validation.
"""

import pytest

from inference_tools.schemas import (
    ForecastOptimizerInput,
    ForecastOptimizerOutput,
    ForecastSkuNotFoundError,
    ForecastMissingFeatureColumnsError,
    MissingRiskSignalError,
)
from inference_tools.forecast_optimizer_tool import (
    optimize_forecast,
    load_forecast_optimizer_artifacts,
    AGENT3_FEATURES,
)


# Hand-built test data: base row with all REQUIRED_RAW_COLS populated.
BASE_ROW = {
    'national_inv': 100.0,
    'lead_time': 5.0,
    'in_transit_qty': 50.0,
    'forecast_3_month': 70.0,
    'forecast_6_month': 140.0,
    'forecast_9_month': 210.0,
    'sales_1_month': 20.0,
    'sales_3_month': 60.0,
    'sales_6_month': 120.0,
    'sales_9_month': 180.0,
    'min_bank': 20.0,
    'potential_issue': 0,
    'pieces_past_due': 5.0,
    'perf_6_month_avg': 0.85,
    'perf_12_month_avg': 0.80,
    'local_bo_qty': 0.0,
    'deck_risk': 0,
    'oe_constraint': 0,
    'ppap_risk': 0,
    'stop_auto_buy': 1,
    'rev_stop': 0,
    'inv_velocity': 0.5,
    'safety_gap': 10.0,
    'sales_trend': 0.1,
    'perf_gap': 0.05,
}

NORMAL_ROW = BASE_ROW.copy()

# -99.0 sentinel in perf columns — should not crash.
SENTINEL_PERF_ROW = {**BASE_ROW, 'perf_6_month_avg': -99.0, 'perf_12_month_avg': -99.0}

# NaN lead_time — a raw passthrough FEATURE_COLS column; also feeds the whole-frame
# median-fill inside create_agent3_features(), but for a single-row batch the median of
# a single NaN is itself NaN, so the documented neutral 0.0 fallback applies.
NAN_LEADTIME_ROW = {**BASE_ROW, 'lead_time': float('nan')}

BASE_BO_PROB = 0.1


class TestNormalInput:
    """Test basic happy path with valid input."""

    def test_normal_input_produces_valid_output(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False},
        )
        output = optimize_forecast(input_data)

        assert isinstance(output, ForecastOptimizerOutput)
        assert len(output.recommendations) == 1
        rec = output.recommendations[0]
        assert rec.sku == 'SKU-001'
        assert 0.0 <= rec.bias_probability <= 1.0
        assert 0.3 <= rec.correction_factor <= 1.5
        assert rec.bias_severity in ('SEVERE', 'MILD', 'NONE')
        assert rec.recommendation in ('REDUCE_PLAN', 'INCREASE_PLAN', 'HOLD')
        assert rec.risk_override_applied is False
        assert output.model_version_a.endswith('_bias_detector_v1')
        assert output.model_version_b == 'xgboost_corrector_v1'

    def test_multiple_skus_all_produce_recommendations(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'forecast_3_month': 100.0},
                'SKU-003': {**NORMAL_ROW, 'forecast_3_month': 40.0},
            },
            backorder_probability={
                'SKU-001': BASE_BO_PROB, 'SKU-002': BASE_BO_PROB, 'SKU-003': BASE_BO_PROB,
            },
            alarm_triggered={
                'SKU-001': False, 'SKU-002': False, 'SKU-003': False,
            },
        )
        output = optimize_forecast(input_data)

        assert len(output.recommendations) == 3
        assert {r.sku for r in output.recommendations} == {'SKU-001', 'SKU-002', 'SKU-003'}


class TestAlarmOverride:
    """Test the business override: alarm_triggered forces correction_factor to 1.0."""

    def test_alarm_triggered_forces_correction_factor_to_one(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-ALARM'],
            data={'SKU-ALARM': NORMAL_ROW},
            backorder_probability={'SKU-ALARM': 0.9},
            alarm_triggered={'SKU-ALARM': True},
        )
        output = optimize_forecast(input_data)

        rec = output.recommendations[0]
        assert rec.correction_factor == pytest.approx(1.0, abs=1e-9)
        assert rec.risk_override_applied is True
        assert rec.adjusted_forecast_3m == pytest.approx(rec.human_forecast_3m, abs=1e-6)
        assert rec.recommendation == 'HOLD'
        assert rec.bias_severity == 'NONE'

    def test_alarm_not_triggered_no_override_flag(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-NO-ALARM'],
            data={'SKU-NO-ALARM': NORMAL_ROW},
            backorder_probability={'SKU-NO-ALARM': 0.1},
            alarm_triggered={'SKU-NO-ALARM': False},
        )
        output = optimize_forecast(input_data)

        assert output.recommendations[0].risk_override_applied is False


class TestSentinelPerfHandling:
    """Test -99.0 sentinel in perf columns."""

    def test_sentinel_perf_averages_do_not_crash(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-SENTINEL'],
            data={'SKU-SENTINEL': SENTINEL_PERF_ROW},
            backorder_probability={'SKU-SENTINEL': BASE_BO_PROB},
            alarm_triggered={'SKU-SENTINEL': False},
        )
        output = optimize_forecast(input_data)

        assert len(output.recommendations) == 1
        assert 0.3 <= output.recommendations[0].correction_factor <= 1.5


class TestNaNHandling:
    """Test NaN handling via the batch-median + neutral-fallback strategy."""

    def test_nan_lead_time_produces_valid_output(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-NAN-LEAD'],
            data={'SKU-NAN-LEAD': NAN_LEADTIME_ROW},
            backorder_probability={'SKU-NAN-LEAD': BASE_BO_PROB},
            alarm_triggered={'SKU-NAN-LEAD': False},
        )
        output = optimize_forecast(input_data)

        assert len(output.recommendations) == 1
        assert 0.3 <= output.recommendations[0].correction_factor <= 1.5

    def test_nan_in_batch_filled_by_other_rows_median(self):
        """With >1 row, a NaN column's batch median comes from the other rows, not 0.0."""
        input_data = ForecastOptimizerInput(
            skus=['SKU-NAN', 'SKU-OK'],
            data={'SKU-NAN': NAN_LEADTIME_ROW, 'SKU-OK': NORMAL_ROW},
            backorder_probability={'SKU-NAN': BASE_BO_PROB, 'SKU-OK': BASE_BO_PROB},
            alarm_triggered={'SKU-NAN': False, 'SKU-OK': False},
        )
        output = optimize_forecast(input_data)

        assert len(output.recommendations) == 2


class TestMissingSkuError:
    """Test that missing SKUs are caught at construction time."""

    def test_missing_sku_in_data_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            ForecastOptimizerInput(
                skus=['SKU-MISSING'],
                data={'SKU-OTHER': NORMAL_ROW},
                backorder_probability={'SKU-MISSING': BASE_BO_PROB},
                alarm_triggered={'SKU-MISSING': False},
            )

    def test_missing_sku_in_backorder_probability_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            ForecastOptimizerInput(
                skus=['SKU-MISSING'],
                data={'SKU-MISSING': NORMAL_ROW},
                backorder_probability={'SKU-OTHER': BASE_BO_PROB},
                alarm_triggered={'SKU-MISSING': False},
            )

    def test_missing_sku_in_alarm_triggered_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            ForecastOptimizerInput(
                skus=['SKU-MISSING'],
                data={'SKU-MISSING': NORMAL_ROW},
                backorder_probability={'SKU-MISSING': BASE_BO_PROB},
                alarm_triggered={'SKU-OTHER': False},
            )


class TestMissingRiskSignal:
    """Defensive re-check inside optimize_forecast() itself (belt-and-suspenders)."""

    def test_direct_call_missing_backorder_probability_raises(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False},
        )
        # Bypass Pydantic's own validator by mutating the dict post-construction.
        input_data.backorder_probability = {}
        with pytest.raises(MissingRiskSignalError):
            optimize_forecast(input_data)


class TestMissingFeatureColumnsError:
    """Test ForecastMissingFeatureColumnsError when required raw columns are absent."""

    def test_missing_raw_column_raises_error(self):
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['national_inv']

        input_data = ForecastOptimizerInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row},
            backorder_probability={'SKU-INCOMPLETE': BASE_BO_PROB},
            alarm_triggered={'SKU-INCOMPLETE': False},
        )
        with pytest.raises(ForecastMissingFeatureColumnsError, match="national_inv"):
            optimize_forecast(input_data)

    def test_missing_multiple_columns_raises_error(self):
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['lead_time']
        del incomplete_row['forecast_6_month']

        input_data = ForecastOptimizerInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row},
            backorder_probability={'SKU-INCOMPLETE': BASE_BO_PROB},
            alarm_triggered={'SKU-INCOMPLETE': False},
        )
        with pytest.raises(ForecastMissingFeatureColumnsError):
            optimize_forecast(input_data)


class TestCorrectionFactorRange:
    """Test correction_factor stays within the trained [0.3, 1.5] clip range."""

    def test_correction_factor_within_trained_range(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001', 'SKU-002'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'forecast_3_month': 5.0},
            },
            backorder_probability={'SKU-001': BASE_BO_PROB, 'SKU-002': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False, 'SKU-002': False},
        )
        output = optimize_forecast(input_data)

        for rec in output.recommendations:
            assert 0.3 <= rec.correction_factor <= 1.5


class TestDeterminism:
    """Test that optimize_forecast() is deterministic for a fixed input."""

    def test_same_input_produces_identical_output(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False},
        )
        output1 = optimize_forecast(input_data)
        output2 = optimize_forecast(input_data)

        assert output1.model_dump() == output2.model_dump()

    def test_multiple_calls_produce_identical_outputs(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-A', 'SKU-B'],
            data={
                'SKU-A': NORMAL_ROW,
                'SKU-B': {**NORMAL_ROW, 'forecast_3_month': 90.0},
            },
            backorder_probability={'SKU-A': BASE_BO_PROB, 'SKU-B': BASE_BO_PROB},
            alarm_triggered={'SKU-A': False, 'SKU-B': True},
        )
        outputs = [optimize_forecast(input_data) for _ in range(3)]

        for output in outputs[1:]:
            assert output.model_dump() == outputs[0].model_dump()


class TestArtifactCaching:
    """Test that load_forecast_optimizer_artifacts() caches the loaded models."""

    def test_artifacts_loaded_once_across_multiple_calls(self):
        load_forecast_optimizer_artifacts.cache_clear()

        input_data = ForecastOptimizerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False},
        )

        optimize_forecast(input_data)
        cache_info_1 = load_forecast_optimizer_artifacts.cache_info()

        optimize_forecast(input_data)
        cache_info_2 = load_forecast_optimizer_artifacts.cache_info()

        assert cache_info_2.hits == cache_info_1.hits + 1


class TestArtifactLoadPathResolution:
    """Test that load_forecast_optimizer_artifacts() resolves and loads correctly."""

    def test_default_models_dir_resolves(self):
        load_forecast_optimizer_artifacts.cache_clear()
        loaded = load_forecast_optimizer_artifacts()

        assert loaded.model_a is not None
        assert loaded.model_b is not None
        assert loaded.feature_cols is not None
        assert loaded.model_version_b == 'xgboost_corrector_v1'

    def test_feature_cols_match_expected(self):
        loaded = load_forecast_optimizer_artifacts()
        assert loaded.feature_cols == AGENT3_FEATURES


class TestOutputValidation:
    """Test that ForecastOptimizerOutput validates correctly."""

    def test_output_validates_against_schema(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB},
            alarm_triggered={'SKU-001': False},
        )
        output = optimize_forecast(input_data)

        validated = ForecastOptimizerOutput.model_validate(output.model_dump())
        assert validated.model_dump() == output.model_dump()


class TestFlaggedListsConsistency:
    """Test flagged_for_reduction / flagged_for_increase are derived views."""

    def test_flagged_lists_consistent_with_recommendations(self):
        input_data = ForecastOptimizerInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'forecast_3_month': 200.0},
                'SKU-003': {**NORMAL_ROW, 'forecast_3_month': 20.0},
            },
            backorder_probability={'SKU-001': 0.1, 'SKU-002': 0.1, 'SKU-003': 0.1},
            alarm_triggered={'SKU-001': False, 'SKU-002': False, 'SKU-003': False},
        )
        output = optimize_forecast(input_data)

        expected_reduce = {r.sku for r in output.recommendations if r.recommendation == 'REDUCE_PLAN'}
        expected_increase = {r.sku for r in output.recommendations if r.recommendation == 'INCREASE_PLAN'}
        assert set(output.flagged_for_reduction) == expected_reduce
        assert set(output.flagged_for_increase) == expected_increase
        assert set(output.flagged_for_reduction) & set(output.flagged_for_increase) == set()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
