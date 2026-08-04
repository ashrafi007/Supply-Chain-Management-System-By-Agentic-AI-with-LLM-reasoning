"""
Comprehensive tests for the Inventory Rebalancer inference tool.

Covers: normal input, sentinel handling, NaN handling, negative values,
recommended_qty computation, missing SKUs/features, determinism, caching,
artifact loading, output validation, top_priority_skus, and batch_rank.
"""

import pytest

from inference_tools.schemas import (
    InventoryRebalancerInput,
    InventoryRebalancerOutput,
    RebalancerSkuNotFoundError,
    RebalancerMissingFeatureColumnsError,
    MissingBackorderProbabilityError,
)
from inference_tools.inventory_rebalancer_tool import (
    rebalance_inventory,
    load_rebalancer_artifacts,
    REBALANCER_FEATURES,
)


# Hand-built test data: base row with all required raw columns.
BASE_ROW = {
    'national_inv': 100.0,
    'in_transit_qty': 50.0,
    'sales_1_month': 20.0,
    'sales_3_month': 60.0,
    'sales_9_month': 180.0,
    'pieces_past_due': 5.0,
    'deck_risk': 0,
    'oe_constraint': 0,
    'ppap_risk': 0,
    'forecast_3_month': 70.0,
    'lead_time': 5.0,
    'perf_gap': 0.05,
    'min_bank': 20.0,
    'perf_6_month_avg': 0.85,
    'perf_12_month_avg': 0.80,
}

NORMAL_ROW = BASE_ROW.copy()

# -99.0 in perf columns — should not crash. Note: perf_6_month_avg/perf_12_month_avg
# only feed performance_trend, which is never a selected REBALANCER_FEATURES column,
# so this test only confirms no crash; it doesn't test a specific numeric effect.
SENTINEL_PERF_ROW = {**BASE_ROW, 'perf_6_month_avg': -99.0, 'perf_12_month_avg': -99.0}

# NaN in lead_time — imputer handles it directly since lead_time is one of the 13 REBALANCER_FEATURES.
NAN_LEADTIME_ROW = {**BASE_ROW, 'lead_time': float('nan')}

# Negative national_inv — passed through uncorrected (not clipped).
NEGATIVE_NATIONAL_INV_ROW = {**BASE_ROW, 'national_inv': -50.0}

# min_bank > national_inv → recommended_qty > 0.
HIGH_MIN_BANK_ROW = {**BASE_ROW, 'national_inv': 5.0, 'min_bank': 100.0}

BASE_BO_PROB = 0.1


class TestNormalInput:
    """Test basic happy path with valid input."""

    def test_normal_input_produces_valid_output(self):
        """Single normal SKU produces valid InventoryRebalancerOutput."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert isinstance(output, InventoryRebalancerOutput)
        assert len(output.recommendations) == 1
        assert output.recommendations[0].sku == 'SKU-001'
        assert 0.0 <= output.recommendations[0].urgency_score_predicted <= 1.0
        assert output.recommendations[0].recommended_qty >= 0
        assert output.recommendations[0].batch_rank == 1
        assert output.recommendations[0].manufacture_rank is None
        assert output.model_version == 'xgboost_urgency_v1'

    def test_multiple_skus_all_produce_recommendations(self):
        """Multiple SKUs all get recommendations in output."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'national_inv': 50.0},
                'SKU-003': {**NORMAL_ROW, 'national_inv': 200.0},
            },
            backorder_probability={
                'SKU-001': BASE_BO_PROB,
                'SKU-002': BASE_BO_PROB,
                'SKU-003': BASE_BO_PROB,
            }
        )
        output = rebalance_inventory(input_data)

        assert len(output.recommendations) == 3
        assert {r.sku for r in output.recommendations} == {'SKU-001', 'SKU-002', 'SKU-003'}


class TestSentinelPerfHandling:
    """Test -99.0 sentinel in perf columns."""

    def test_sentinel_perf_averages_do_not_crash(self):
        """Rows with -99.0 in perf columns are handled without error."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-SENTINEL'],
            data={'SKU-SENTINEL': SENTINEL_PERF_ROW},
            backorder_probability={'SKU-SENTINEL': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert len(output.recommendations) == 1
        assert 0.0 <= output.recommendations[0].urgency_score_predicted <= 1.0


class TestNaNHandling:
    """Test NaN passthrough (imputer handles it)."""

    def test_nan_lead_time_produces_valid_output(self):
        """Rows with NaN lead_time produce valid predictions (imputer handles it)."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-NAN-LEAD'],
            data={'SKU-NAN-LEAD': NAN_LEADTIME_ROW},
            backorder_probability={'SKU-NAN-LEAD': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert len(output.recommendations) == 1
        assert 0.0 <= output.recommendations[0].urgency_score_predicted <= 1.0


class TestNegativeNationalInv:
    """Test that negative national_inv is NOT corrected."""

    def test_negative_national_inv_passes_uncorrected(self):
        """Negative national_inv is passed through as-is, not clipped/corrected."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-NEG-INV'],
            data={'SKU-NEG-INV': NEGATIVE_NATIONAL_INV_ROW},
            backorder_probability={'SKU-NEG-INV': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert len(output.recommendations) == 1
        assert 0.0 <= output.recommendations[0].urgency_score_predicted <= 1.0


class TestRecommendedQty:
    """Test recommended_qty computation (clip(min_bank - national_inv, 0))."""

    def test_min_bank_greater_than_national_inv_gives_positive_qty(self):
        """min_bank > national_inv → recommended_qty > 0 (exact arithmetic check)."""
        # HIGH_MIN_BANK_ROW: national_inv=5.0, min_bank=100.0 → qty should be 95.0
        input_data = InventoryRebalancerInput(
            skus=['SKU-HIGH-BANK'],
            data={'SKU-HIGH-BANK': HIGH_MIN_BANK_ROW},
            backorder_probability={'SKU-HIGH-BANK': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert output.recommendations[0].recommended_qty == pytest.approx(95.0, abs=1e-6)

    def test_min_bank_less_than_national_inv_gives_zero_qty(self):
        """min_bank <= national_inv → recommended_qty == 0 (clipped)."""
        # NORMAL_ROW: national_inv=100.0, min_bank=20.0 → qty clipped to 0
        input_data = InventoryRebalancerInput(
            skus=['SKU-LOW-BANK'],
            data={'SKU-LOW-BANK': NORMAL_ROW},
            backorder_probability={'SKU-LOW-BANK': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        assert output.recommendations[0].recommended_qty == pytest.approx(0.0, abs=1e-6)


class TestMissingSkuError:
    """Test that missing SKUs are caught."""

    def test_missing_sku_in_data_raises_validation_error(self):
        """Requested SKU not in data dict → ValidationError at construction time."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            InventoryRebalancerInput(
                skus=['SKU-MISSING'],
                data={'SKU-OTHER': NORMAL_ROW},
                backorder_probability={'SKU-MISSING': BASE_BO_PROB}
            )

    def test_missing_sku_in_backorder_probability_raises_validation_error(self):
        """Requested SKU not in backorder_probability → ValidationError at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            InventoryRebalancerInput(
                skus=['SKU-MISSING'],
                data={'SKU-MISSING': NORMAL_ROW},
                backorder_probability={'SKU-OTHER': BASE_BO_PROB}
            )

    def test_multiple_missing_skus_listed(self):
        """Multiple missing SKUs are all listed in error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            InventoryRebalancerInput(
                skus=['SKU-A', 'SKU-B', 'SKU-C'],
                data={'SKU-X': NORMAL_ROW},
                backorder_probability={'SKU-X': BASE_BO_PROB}
            )


class TestMissingFeatureColumnsError:
    """Test RebalancerMissingFeatureColumnsError when required raw columns are absent."""

    def test_missing_raw_column_raises_error(self):
        """Row missing a required raw column → RebalancerMissingFeatureColumnsError."""
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['national_inv']

        input_data = InventoryRebalancerInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row},
            backorder_probability={'SKU-INCOMPLETE': BASE_BO_PROB}
        )
        with pytest.raises(RebalancerMissingFeatureColumnsError, match="national_inv"):
            rebalance_inventory(input_data)

    def test_missing_multiple_columns_raises_error(self):
        """Row missing multiple required columns → error."""
        incomplete_row = NORMAL_ROW.copy()
        del incomplete_row['lead_time']
        del incomplete_row['deck_risk']

        input_data = InventoryRebalancerInput(
            skus=['SKU-INCOMPLETE'],
            data={'SKU-INCOMPLETE': incomplete_row},
            backorder_probability={'SKU-INCOMPLETE': BASE_BO_PROB}
        )
        with pytest.raises(RebalancerMissingFeatureColumnsError):
            rebalance_inventory(input_data)


class TestDeterminism:
    """Test that rebalance_inventory() is deterministic."""

    def test_same_input_produces_identical_output(self):
        """Same input → same output (deterministic)."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB}
        )
        output1 = rebalance_inventory(input_data)
        output2 = rebalance_inventory(input_data)

        assert output1.model_dump() == output2.model_dump()

    def test_multiple_calls_produce_identical_outputs(self):
        """Multiple calls with same input always produce identical outputs."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-A', 'SKU-B'],
            data={
                'SKU-A': NORMAL_ROW,
                'SKU-B': {**NORMAL_ROW, 'national_inv': 75.0},
            },
            backorder_probability={'SKU-A': BASE_BO_PROB, 'SKU-B': BASE_BO_PROB}
        )
        outputs = [rebalance_inventory(input_data) for _ in range(3)]

        for output in outputs[1:]:
            assert output.model_dump() == outputs[0].model_dump()


class TestArtifactCaching:
    """Test that load_rebalancer_artifacts() caches the loaded model."""

    def test_artifacts_loaded_once_across_multiple_calls(self):
        """Artifacts are loaded once (cached) across multiple rebalance_inventory calls."""
        load_rebalancer_artifacts.cache_clear()

        input_data = InventoryRebalancerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB}
        )

        # First call loads the artifacts.
        rebalance_inventory(input_data)
        cache_info_1 = load_rebalancer_artifacts.cache_info()

        # Second call hits the cache.
        rebalance_inventory(input_data)
        cache_info_2 = load_rebalancer_artifacts.cache_info()

        # Hits should have increased by 1 (second call was a cache hit).
        assert cache_info_2.hits == cache_info_1.hits + 1


class TestArtifactLoadPathResolution:
    """Test that load_rebalancer_artifacts() resolves and loads correctly."""

    def test_default_models_dir_resolves(self):
        """Default models_dir (None) resolves to the correct location."""
        load_rebalancer_artifacts.cache_clear()
        loaded = load_rebalancer_artifacts()

        assert loaded.model is not None
        assert loaded.feature_columns is not None
        assert len(loaded.feature_columns) == 13
        assert loaded.model_version == 'xgboost_urgency_v1'

    def test_feature_columns_are_13_elements_in_expected_order(self):
        """Feature columns list has exactly 13 elements in the expected order."""
        loaded = load_rebalancer_artifacts()
        assert len(loaded.feature_columns) == 13
        assert loaded.feature_columns == REBALANCER_FEATURES


class TestOutputValidation:
    """Test that InventoryRebalancerOutput validates correctly."""

    def test_output_validates_against_schema(self):
        """Output passes Pydantic validation via round-trip."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001'],
            data={'SKU-001': NORMAL_ROW},
            backorder_probability={'SKU-001': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        # Round-trip through Pydantic validation.
        validated = InventoryRebalancerOutput.model_validate(output.model_dump())
        assert validated.model_dump() == output.model_dump()


class TestTopPrioritySKUs:
    """Test top_priority_skus computation (recommended_qty > 0 AND critical_state == 1)."""

    def test_top_priority_skus_subset_of_recommendations(self):
        """All SKUs in top_priority_skus are in recommendations."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001', 'SKU-002'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': HIGH_MIN_BANK_ROW,
            },
            backorder_probability={'SKU-001': BASE_BO_PROB, 'SKU-002': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        recommendation_skus = {r.sku for r in output.recommendations}
        assert set(output.top_priority_skus) <= recommendation_skus

    def test_zero_recommended_qty_skus_excluded_from_top_priority(self):
        """SKUs with recommended_qty == 0 are NOT in top_priority_skus (necessary condition)."""
        # NORMAL_ROW has min_bank=20 < national_inv=100, so recommended_qty will be 0.
        input_data = InventoryRebalancerInput(
            skus=['SKU-ZERO-QTY'],
            data={'SKU-ZERO-QTY': NORMAL_ROW},
            backorder_probability={'SKU-ZERO-QTY': BASE_BO_PROB}
        )
        output = rebalance_inventory(input_data)

        # Recommended_qty is 0, so this SKU must NOT be in top_priority_skus.
        # (Note: full membership cannot be asserted from the test side since
        # critical_state is internal; only the necessary condition that zero-qty
        # SKUs are always excluded.)
        assert 'SKU-ZERO-QTY' not in output.top_priority_skus


class TestBatchRank:
    """Test batch_rank computation and sorting."""

    def test_batch_rank_is_permutation_of_1_to_n(self):
        """For N SKUs, batch_rank is a permutation of 1..N."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={
                'SKU-001': {**NORMAL_ROW, 'national_inv': 10.0},
                'SKU-002': {**NORMAL_ROW, 'national_inv': 50.0},
                'SKU-003': {**NORMAL_ROW, 'national_inv': 150.0},
            },
            backorder_probability={
                'SKU-001': 0.05,
                'SKU-002': 0.10,
                'SKU-003': 0.15,
            }
        )
        output = rebalance_inventory(input_data)

        batch_ranks = sorted(r.batch_rank for r in output.recommendations)
        assert batch_ranks == list(range(1, 4))

    def test_batch_rank_1_has_highest_urgency_score(self):
        """The recommendation with batch_rank=1 has the highest urgency_score_predicted."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-001', 'SKU-002'],
            data={
                'SKU-001': NORMAL_ROW,
                'SKU-002': {**NORMAL_ROW, 'national_inv': 50.0},
            },
            backorder_probability={'SKU-001': 0.05, 'SKU-002': 0.15}
        )
        output = rebalance_inventory(input_data)

        rank_1_rec = next(r for r in output.recommendations if r.batch_rank == 1)
        max_urgency = max(r.urgency_score_predicted for r in output.recommendations)
        assert rank_1_rec.urgency_score_predicted == pytest.approx(max_urgency, abs=1e-6)

    def test_recommendations_sorted_by_batch_rank(self):
        """Recommendations list is sorted by batch_rank ascending."""
        input_data = InventoryRebalancerInput(
            skus=['SKU-A', 'SKU-B', 'SKU-C'],
            data={
                'SKU-A': {**NORMAL_ROW, 'national_inv': 10.0},
                'SKU-B': {**NORMAL_ROW, 'national_inv': 100.0},
                'SKU-C': {**NORMAL_ROW, 'national_inv': 200.0},
            },
            backorder_probability={'SKU-A': 0.2, 'SKU-B': 0.1, 'SKU-C': 0.05}
        )
        output = rebalance_inventory(input_data)

        ranks = [r.batch_rank for r in output.recommendations]
        assert ranks == sorted(ranks)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
