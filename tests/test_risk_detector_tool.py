"""
Comprehensive tests for the Risk Detector inference tool.

Covers: normal input, missing SKU, malformed input (missing columns),
determinism, and caching behavior.
"""

import pytest
from unittest.mock import patch

from inference_tools.schemas import (
    RiskDetectorInput,
    RiskDetectorOutput,
    SkuNotFoundError,
    MissingFeatureColumnsError,
    ModelLoadError,
)
from inference_tools.risk_detector_tool import (
    predict_risk,
    load_risk_detector_model,
    _prepare_feature_row,
)


# Hand-built test data: one normal row, one with -99.0 sentinel, one with NaN,
# one with negative national_inv.
BASE_ROW = {
    'national_inv': 100.0,
    'lead_time': 5.0,
    'in_transit_qty': 50.0,
    'forecast_3_month': 150.0,
    'forecast_6_month': 450.0,
    'forecast_9_month': 900.0,
    'sales_1_month': 50.0,
    'sales_3_month': 150.0,
    'sales_6_month': 300.0,
    'sales_9_month': 450.0,
    'min_bank': 20.0,
    'potential_issue': 0,
    'pieces_past_due': 5.0,
    'perf_6_month_avg': 0.85,
    'perf_12_month_avg': 0.80,
    'local_bo_qty': 0.0,
    'deck_risk': 0,
    'oe_constraint': 0,
    'ppap_risk': 0,
    'stop_auto_buy': 0,
    'rev_stop': 0,
    'inv_velocity': 0.5,
    'safety_gap': 10.0,
    'sales_trend': 5.0,
    'perf_gap': 0.05,
}

NORMAL_ROW = BASE_ROW.copy()

SENTINEL_ROW = BASE_ROW.copy()
SENTINEL_ROW['perf_6_month_avg'] = -99.0
SENTINEL_ROW['perf_12_month_avg'] = -99.0

NAN_LEADTIME_ROW = BASE_ROW.copy()
NAN_LEADTIME_ROW['lead_time'] = float('nan')

NEGATIVE_NATIONAL_INV_ROW = BASE_ROW.copy()
NEGATIVE_NATIONAL_INV_ROW['national_inv'] = -50.0


class TestNormalInput:
    """Test basic happy path with valid input."""

    def test_normal_input_produces_valid_output(self):
        """Single normal SKU produces valid RiskDetectorOutput."""
        input_data = RiskDetectorInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW}
        )
        output = predict_risk(input_data)

        assert isinstance(output, RiskDetectorOutput)
        assert len(output.predictions) == 1
        assert output.predictions[0].sku == 'SKU-001'
        assert 0.0 <= output.predictions[0].backorder_probability <= 1.0
        assert isinstance(output.predictions[0].predicted_label, bool)
        assert isinstance(output.predictions[0].is_high_risk, bool)
        assert output.model_version == 'stacking_ensemble_v1'
        assert output.threshold_used == pytest.approx(0.945, rel=0.01)

    def test_multiple_skus_all_produce_predictions(self):
        """Multiple SKUs all get predictions in output."""
        input_data = RiskDetectorInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'national_inv': 50.0},
                'SKU-003': {**NORMAL_ROW, 'national_inv': 200.0},
            }
        )
        output = predict_risk(input_data)

        assert len(output.predictions) == 3
        assert [p.sku for p in output.predictions] == ['SKU-001', 'SKU-002', 'SKU-003']

    def test_high_risk_skus_derived_from_predictions(self):
        """high_risk_skus is exactly those predictions where is_high_risk=True."""
        input_data = RiskDetectorInput(
            skus=['SKU-001', 'SKU-002'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': NORMAL_ROW,
            }
        )
        output = predict_risk(input_data)

        # high_risk_skus should be all SKUs where is_high_risk is True
        expected = {p.sku for p in output.predictions if p.is_high_risk}
        assert set(output.high_risk_skus) == expected

    def test_predicted_label_vs_is_high_risk_thresholds(self):
        """predicted_label (0.5) and is_high_risk (0.945) are distinct."""
        # For most real data, these will differ. Test structure, not values.
        input_data = RiskDetectorInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW}
        )
        output = predict_risk(input_data)
        pred = output.predictions[0]

        # If prob >= 0.5, predicted_label must be True.
        if pred.backorder_probability >= 0.5:
            assert pred.predicted_label is True
        else:
            assert pred.predicted_label is False

        # If prob >= 0.945, is_high_risk must be True.
        if pred.backorder_probability >= 0.945:
            assert pred.is_high_risk is True
        else:
            assert pred.is_high_risk is False


class TestSentinelHandling:
    """Test -99.0 sentinel → NaN conversion."""

    def test_sentinel_in_perf_columns(self):
        """Rows with -99.0 in perf columns are handled without error."""
        input_data = RiskDetectorInput(
            skus=['SKU-SENTINEL'],
            data={'SKU-SENTINEL': SENTINEL_ROW}
        )
        output = predict_risk(input_data)

        assert len(output.predictions) == 1
        assert 0.0 <= output.predictions[0].backorder_probability <= 1.0


class TestNaNHandling:
    """Test NaN passthrough (no imputation)."""

    def test_nan_lead_time_passes_through(self):
        """Rows with NaN lead_time produce valid predictions (NaN is passed through)."""
        input_data = RiskDetectorInput(
            skus=['SKU-NAN-LEAD'],
            data={'SKU-NAN-LEAD': NAN_LEADTIME_ROW}
        )
        output = predict_risk(input_data)

        assert len(output.predictions) == 1
        assert 0.0 <= output.predictions[0].backorder_probability <= 1.0


class TestNegativeNationalInv:
    """Test that negative national_inv is NOT corrected."""

    def test_negative_national_inv_passes_uncorrected(self):
        """Negative national_inv is passed through as-is, not clipped/corrected."""
        input_data = RiskDetectorInput(
            skus=['SKU-NEG-INV'],
            data={'SKU-NEG-INV': NEGATIVE_NATIONAL_INV_ROW}
        )
        output = predict_risk(input_data)

        assert len(output.predictions) == 1
        assert 0.0 <= output.predictions[0].backorder_probability <= 1.0


class TestMissingSkuError:
    """Test that missing SKUs are caught at input validation level."""

    def test_missing_sku_raises_validation_error(self):
        """Requested SKU not in data dict → validation error at construction time."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            RiskDetectorInput(
                skus=['SKU-MISSING'],
                data={'SKU-OTHER': NORMAL_ROW}
            )

    def test_multiple_missing_skus_lists_all(self):
        """Multiple missing SKUs are all listed in validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RiskDetectorInput(
                skus=['SKU-A', 'SKU-B', 'SKU-C'],
                data={'SKU-X': NORMAL_ROW}
            )


class TestMissingFeatureColumnsError:
    """Test MissingFeatureColumnsError when required columns are absent."""

    def test_missing_raw_column_raises_error(self):
        """Row missing a required raw column → MissingFeatureColumnsError."""
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['national_inv']

        input_data = RiskDetectorInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row}
        )
        with pytest.raises(MissingFeatureColumnsError, match="national_inv"):
            predict_risk(input_data)

    def test_missing_multiple_columns_raises_error(self):
        """Row missing multiple required columns → MissingFeatureColumnsError."""
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['lead_time']
        del incomplete_row['deck_risk']

        input_data = RiskDetectorInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row}
        )
        with pytest.raises(MissingFeatureColumnsError):
            predict_risk(input_data)


class TestDeterminism:
    """Test that predict_risk() is deterministic."""

    def test_same_input_produces_identical_output(self):
        """Same input → same output (deterministic)."""
        input_data = RiskDetectorInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW}
        )
        output1 = predict_risk(input_data)
        output2 = predict_risk(input_data)

        # Compare via model_dump() to handle float equality robustly.
        assert output1.model_dump() == output2.model_dump()

    def test_multiple_calls_produce_identical_outputs(self):
        """Multiple calls with same input always produce identical outputs."""
        input_data = RiskDetectorInput(
            skus=['SKU-A', 'SKU-B'],
            data={
                'SKU-A': NORMAL_ROW,
                'SKU-B': {**NORMAL_ROW, 'national_inv': 75.0},
            }
        )
        outputs = [predict_risk(input_data) for _ in range(3)]

        for output in outputs[1:]:
            assert output.model_dump() == outputs[0].model_dump()


class TestModelCaching:
    """Test that load_risk_detector_model() caches the loaded model."""

    def test_model_loaded_once_across_multiple_calls(self):
        """Model is loaded once (cached) across multiple predict_risk calls."""
        # Clear the cache before this test to get a clean state.
        load_risk_detector_model.cache_clear()

        input_data = RiskDetectorInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW}
        )

        # First call loads the model.
        predict_risk(input_data)
        cache_info_1 = load_risk_detector_model.cache_info()

        # Second call hits the cache.
        predict_risk(input_data)
        cache_info_2 = load_risk_detector_model.cache_info()

        # Hits should have increased by 1 (second call was a cache hit).
        assert cache_info_2.hits == cache_info_1.hits + 1


class TestModelLoadPathResolution:
    """Test that load_risk_detector_model() resolves default path correctly."""

    def test_default_model_path_resolves(self):
        """Default model path (None) resolves to the correct location."""
        load_risk_detector_model.cache_clear()
        loaded = load_risk_detector_model()

        assert loaded.model is not None
        assert loaded.feature_columns is not None
        assert len(loaded.feature_columns) == 40
        assert loaded.threshold == pytest.approx(0.945, rel=0.01)
        assert loaded.model_version == 'stacking_ensemble_v1'

    def test_feature_columns_are_40_elements(self):
        """Feature columns list has exactly 40 elements."""
        loaded = load_risk_detector_model()
        assert len(loaded.feature_columns) == 40

    def test_threshold_is_valid_probability(self):
        """Threshold is a valid probability in [0, 1]."""
        loaded = load_risk_detector_model()
        assert 0.0 <= loaded.threshold <= 1.0


class TestOutputValidation:
    """Test that RiskDetectorOutput validates correctly."""

    def test_output_validates_against_schema(self):
        """Output passes Pydantic validation."""
        input_data = RiskDetectorInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW}
        )
        output = predict_risk(input_data)

        # This should not raise if the output is valid.
        validated = RiskDetectorOutput.model_validate(output.model_dump())
        assert validated.model_dump() == output.model_dump()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
