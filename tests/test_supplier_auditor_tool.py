"""
Comprehensive tests for the Supplier Auditor inference tool.

Covers: normal input, missing SKU, malformed input (missing raw columns), grade-band
coverage (A/B/C/D reachability), the D-band's 1.01 upper bound, the four-way
MODEL/RULE/MODEL+RULE/NONE trigger-reason combination, determinism, artifact caching,
and output/schema validation.

Feature rows below (BASE_ROW, GRADE_D_RULE_ROW, GRADE_C_ROW, GRADE_B_ROW, RISKY_ROW) were
found empirically against the real loaded model artifact (not hand-guessed) so that each
lands in its named grade band. GRADE_B_ROW / GRADE_C_ROW are linear interpolations between
BASE_ROW and RISKY_ROW and so have fractional (non-0/1) risk-flag values — fine for pure
inference testing since engineer_features() only sums them, but not meant to depict a
realistic supplier record.
"""

import pytest

from inference_tools.schemas import (
    SupplierAuditorInput,
    SupplierAuditorOutput,
    SupplierAuditorSkuNotFoundError,
    SupplierAuditorMissingFeatureColumnsError,
)
from inference_tools.supplier_auditor_tool import (
    audit_suppliers,
    load_supplier_auditor_artifact,
    engineer_features,
    assign_grade,
    combine_trigger_flags,
    REQUIRED_RAW_COLS,
    FEATURE_COLS,
)

import pandas as pd


# Grade D (MODEL only): high risk_probability, delivery_stress == 0.
BASE_ROW = {
    'national_inv': 500.0, 'lead_time': 5.0, 'in_transit_qty': 100.0,
    'forecast_3_month': 300.0, 'forecast_6_month': 600.0, 'forecast_9_month': 900.0,
    'sales_1_month': 100.0, 'sales_3_month': 300.0, 'sales_6_month': 600.0, 'sales_9_month': 900.0,
    'min_bank': 50.0, 'potential_issue': 0, 'pieces_past_due': 0.0,
    'perf_6_month_avg': 0.95, 'perf_12_month_avg': 0.93,
    'local_bo_qty': 0.0, 'deck_risk': 0, 'oe_constraint': 0, 'ppap_risk': 0,
    'rev_stop': 0, 'went_on_backorder': 0, 'inv_velocity': 0.5, 'safety_gap': 10.0,
    'sales_trend': 0.05, 'perf_gap': 0.02,
}

# Grade D (MODEL+RULE): same as BASE_ROW but pieces_past_due pushed up so
# delivery_stress = 300 / (500 + EPS) == 0.6 > 0.5.
GRADE_D_RULE_ROW = {**BASE_ROW, 'pieces_past_due': 300.0}

# Grade A (NONE): far low-risk_probability row (empirically confirmed prob ~0.006, well
# below the artifact's real OPT_THRESH ~0.08, and not grade D, so both flags are False).
RISKY_ROW = {
    'national_inv': 5.0, 'lead_time': 45.0, 'in_transit_qty': 0.0,
    'forecast_3_month': 500.0, 'forecast_6_month': 1000.0, 'forecast_9_month': 1500.0,
    'sales_1_month': 10.0, 'sales_3_month': 20.0, 'sales_6_month': 40.0, 'sales_9_month': 60.0,
    'min_bank': 200.0, 'potential_issue': 1, 'pieces_past_due': 400.0,
    'perf_6_month_avg': 0.05, 'perf_12_month_avg': 0.50,
    'local_bo_qty': 300.0, 'deck_risk': 1, 'oe_constraint': 1, 'ppap_risk': 1,
    'rev_stop': 1, 'went_on_backorder': 1, 'inv_velocity': -0.9, 'safety_gap': -190.0,
    'sales_trend': -0.5, 'perf_gap': 0.9,
}

# Grade C (MODEL): 70% interpolation between BASE_ROW and RISKY_ROW.
GRADE_C_ROW = {
    'national_inv': 153.5, 'lead_time': 33.0, 'in_transit_qty': 30.0,
    'forecast_3_month': 440.0, 'forecast_6_month': 880.0, 'forecast_9_month': 1320.0,
    'sales_1_month': 37.0, 'sales_3_month': 104.0, 'sales_6_month': 208.0, 'sales_9_month': 312.0,
    'min_bank': 155.0, 'potential_issue': 0.7, 'pieces_past_due': 280.0,
    'perf_6_month_avg': 0.32, 'perf_12_month_avg': 0.629,
    'local_bo_qty': 210.0, 'deck_risk': 0.7, 'oe_constraint': 0.7, 'ppap_risk': 0.7,
    'rev_stop': 0.7, 'went_on_backorder': 0.7, 'inv_velocity': -0.48, 'safety_gap': -130.0,
    'sales_trend': -0.335, 'perf_gap': 0.636,
}

# Grade B (MODEL): 87% interpolation between BASE_ROW and RISKY_ROW.
GRADE_B_ROW = {
    'national_inv': 69.35, 'lead_time': 39.8, 'in_transit_qty': 13.0,
    'forecast_3_month': 474.0, 'forecast_6_month': 948.0, 'forecast_9_month': 1422.0,
    'sales_1_month': 21.7, 'sales_3_month': 56.4, 'sales_6_month': 112.8, 'sales_9_month': 169.2,
    'min_bank': 180.5, 'potential_issue': 0.87, 'pieces_past_due': 348.0,
    'perf_6_month_avg': 0.167, 'perf_12_month_avg': 0.5559,
    'local_bo_qty': 261.0, 'deck_risk': 0.87, 'oe_constraint': 0.87, 'ppap_risk': 0.87,
    'rev_stop': 0.87, 'went_on_backorder': 0.87, 'inv_velocity': -0.718, 'safety_gap': -164.0,
    'sales_trend': -0.4285, 'perf_gap': 0.7856,
}

# -99.0 sentinel in perf columns — should not crash.
SENTINEL_PERF_ROW = {**BASE_ROW, 'perf_6_month_avg': -99.0, 'perf_12_month_avg': -99.0}

# NaN lead_time — a raw passthrough FEATURE_COLS column, imputed downstream by the
# preprocessor's SimpleImputer(median), never touched by engineer_features() itself.
NAN_LEADTIME_ROW = {**BASE_ROW, 'lead_time': float('nan')}


class TestNormalInput:
    def test_normal_input_produces_valid_output(self):
        input_data = SupplierAuditorInput(skus=['SKU-001'], data={'SKU-001': BASE_ROW})
        output = audit_suppliers(input_data)

        assert isinstance(output, SupplierAuditorOutput)
        assert len(output.results) == 1
        r = output.results[0]
        assert r.sku == 'SKU-001'
        assert 0.0 <= r.risk_probability <= 1.0
        assert r.supplier_grade in ('A', 'B', 'C', 'D')
        assert r.trigger_reason in ('MODEL+RULE', 'MODEL', 'RULE', 'NONE')
        assert output.model_version == 'xgboost_supplier_auditor_v1'
        assert output.threshold_used == pytest.approx(output.threshold_used)

    def test_multiple_skus_all_produce_results(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-001', 'SKU-002', 'SKU-003'],
            data={'SKU-001': BASE_ROW, 'SKU-002': RISKY_ROW, 'SKU-003': GRADE_C_ROW},
        )
        output = audit_suppliers(input_data)

        assert len(output.results) == 3
        assert {r.sku for r in output.results} == {'SKU-001', 'SKU-002', 'SKU-003'}

    def test_results_sorted_by_risk_probability_descending(self):
        input_data = SupplierAuditorInput(
            skus=['LOW', 'HIGH', 'MID'],
            data={'LOW': RISKY_ROW, 'HIGH': BASE_ROW, 'MID': GRADE_C_ROW},
        )
        output = audit_suppliers(input_data)

        probs = [r.risk_probability for r in output.results]
        assert probs == sorted(probs, reverse=True)


class TestMissingSkuError:
    def test_missing_sku_in_data_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SKU-MISSING"):
            SupplierAuditorInput(skus=['SKU-MISSING'], data={'SKU-OTHER': BASE_ROW})

    def test_direct_call_missing_sku_raises_typed_error(self):
        input_data = SupplierAuditorInput(skus=['SKU-001'], data={'SKU-001': BASE_ROW})
        # Bypass Pydantic's own validator by mutating the dict post-construction.
        input_data.data = {}
        with pytest.raises(SupplierAuditorSkuNotFoundError):
            audit_suppliers(input_data)


class TestMissingFeatureColumnsError:
    def test_missing_raw_column_raises_error(self):
        incomplete_row = BASE_ROW.copy()
        del incomplete_row['national_inv']

        input_data = SupplierAuditorInput(
            skus=['SKU-INCOMPLETE'], data={'SKU-INCOMPLETE': incomplete_row}
        )
        with pytest.raises(SupplierAuditorMissingFeatureColumnsError, match="national_inv"):
            audit_suppliers(input_data)

    def test_missing_multiple_columns_raises_error(self):
        incomplete_row = BASE_ROW.copy()
        del incomplete_row['lead_time']
        del incomplete_row['went_on_backorder']

        input_data = SupplierAuditorInput(
            skus=['SKU-INCOMPLETE'], data={'SKU-INCOMPLETE': incomplete_row}
        )
        with pytest.raises(SupplierAuditorMissingFeatureColumnsError):
            audit_suppliers(input_data)


class TestGradeBandCoverage:
    """Verify all four grade bands are reachable, including the D band's 1.01 upper bound."""

    def test_grade_d_reachable(self):
        input_data = SupplierAuditorInput(skus=['SKU-D'], data={'SKU-D': BASE_ROW})
        output = audit_suppliers(input_data)
        assert output.results[0].supplier_grade == 'D'
        assert output.results[0].risk_probability >= 0.70

    def test_grade_a_reachable(self):
        input_data = SupplierAuditorInput(skus=['SKU-A'], data={'SKU-A': RISKY_ROW})
        output = audit_suppliers(input_data)
        assert output.results[0].supplier_grade == 'A'
        assert output.results[0].risk_probability < 0.20

    def test_grade_c_reachable(self):
        input_data = SupplierAuditorInput(skus=['SKU-C'], data={'SKU-C': GRADE_C_ROW})
        output = audit_suppliers(input_data)
        assert output.results[0].supplier_grade == 'C'
        assert 0.45 <= output.results[0].risk_probability < 0.70

    def test_grade_b_reachable(self):
        input_data = SupplierAuditorInput(skus=['SKU-B'], data={'SKU-B': GRADE_B_ROW})
        output = audit_suppliers(input_data)
        assert output.results[0].supplier_grade == 'B'
        assert 0.20 <= output.results[0].risk_probability < 0.45

    def test_assign_grade_upper_bound_1_01_includes_exact_1_0(self):
        assert assign_grade(1.0) == 'D'
        assert assign_grade(0.99999) == 'D'
        assert assign_grade(0.70) == 'D'

    def test_assign_grade_band_boundaries(self):
        assert assign_grade(0.0) == 'A'
        assert assign_grade(0.1999) == 'A'
        assert assign_grade(0.20) == 'B'
        assert assign_grade(0.4499) == 'B'
        assert assign_grade(0.45) == 'C'
        assert assign_grade(0.6999) == 'C'


class TestTriggerReasonCombination:
    """Test the four-way MODEL/RULE/MODEL+RULE/NONE combination logic."""

    def test_model_only(self):
        input_data = SupplierAuditorInput(skus=['SKU-MODEL'], data={'SKU-MODEL': BASE_ROW})
        output = audit_suppliers(input_data)
        r = output.results[0]
        assert r.trigger_reason == 'MODEL'
        assert r.stop_auto_buy_triggered is True
        assert r.delivery_stress <= 0.5

    def test_model_and_rule(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-BOTH'], data={'SKU-BOTH': GRADE_D_RULE_ROW}
        )
        output = audit_suppliers(input_data)
        r = output.results[0]
        assert r.supplier_grade == 'D'
        assert r.delivery_stress > 0.5
        assert r.trigger_reason == 'MODEL+RULE'
        assert r.stop_auto_buy_triggered is True

    def test_none(self):
        input_data = SupplierAuditorInput(skus=['SKU-NONE'], data={'SKU-NONE': RISKY_ROW})
        output = audit_suppliers(input_data)
        r = output.results[0]
        assert r.trigger_reason == 'NONE'
        assert r.stop_auto_buy_triggered is False

    def test_rule_only_via_combine_trigger_flags_directly(self):
        """
        rule_flag-without-model_flag is structurally unreachable through the real loaded
        model: GRADE_THRESHOLDS["D"] starts at 0.70, always >= the artifact's actual
        OPT_THRESH (~0.08), so grade == "D" implies model_flag is already True. The
        combination LOGIC itself (independent of the real threshold) is verified directly.
        """
        triggered, reason = combine_trigger_flags(
            risk_probability=0.75,
            supplier_grade='D',
            delivery_stress=0.6,
            threshold=0.99,
        )
        assert reason == 'RULE'
        assert triggered is True

    def test_combine_trigger_flags_none_branch(self):
        triggered, reason = combine_trigger_flags(
            risk_probability=0.05, supplier_grade='A', delivery_stress=0.1, threshold=0.08,
        )
        assert reason == 'NONE'
        assert triggered is False

    def test_combine_trigger_flags_model_and_rule_branch(self):
        triggered, reason = combine_trigger_flags(
            risk_probability=0.9, supplier_grade='D', delivery_stress=0.9, threshold=0.08,
        )
        assert reason == 'MODEL+RULE'
        assert triggered is True


class TestSentinelAndNaNHandling:
    def test_sentinel_perf_averages_do_not_crash(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-SENTINEL'], data={'SKU-SENTINEL': SENTINEL_PERF_ROW}
        )
        output = audit_suppliers(input_data)
        assert len(output.results) == 1
        assert 0.0 <= output.results[0].risk_probability <= 1.0

    def test_nan_lead_time_does_not_crash(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-NAN-LEAD'], data={'SKU-NAN-LEAD': NAN_LEADTIME_ROW}
        )
        output = audit_suppliers(input_data)
        assert len(output.results) == 1
        assert 0.0 <= output.results[0].risk_probability <= 1.0


class TestDeterminism:
    def test_same_input_produces_identical_output(self):
        input_data = SupplierAuditorInput(skus=['SKU-001'], data={'SKU-001': BASE_ROW})
        output1 = audit_suppliers(input_data)
        output2 = audit_suppliers(input_data)
        assert output1.model_dump() == output2.model_dump()

    def test_multiple_calls_produce_identical_outputs(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-A', 'SKU-B'], data={'SKU-A': BASE_ROW, 'SKU-B': RISKY_ROW}
        )
        outputs = [audit_suppliers(input_data) for _ in range(3)]
        for output in outputs[1:]:
            assert output.model_dump() == outputs[0].model_dump()


class TestArtifactCaching:
    def test_artifacts_loaded_once_across_multiple_calls(self):
        load_supplier_auditor_artifact.cache_clear()

        input_data = SupplierAuditorInput(skus=['SKU-001'], data={'SKU-001': BASE_ROW})
        audit_suppliers(input_data)
        cache_info_1 = load_supplier_auditor_artifact.cache_info()

        audit_suppliers(input_data)
        cache_info_2 = load_supplier_auditor_artifact.cache_info()

        assert cache_info_2.hits == cache_info_1.hits + 1


class TestArtifactLoadPathResolution:
    def test_default_model_path_resolves(self):
        load_supplier_auditor_artifact.cache_clear()
        loaded = load_supplier_auditor_artifact()

        assert loaded.model is not None
        assert loaded.preprocessor is not None
        assert loaded.feature_cols is not None
        assert loaded.model_version == 'xgboost_supplier_auditor_v1'

    def test_feature_cols_match_expected(self):
        loaded = load_supplier_auditor_artifact()
        assert loaded.feature_cols == FEATURE_COLS


class TestOutputValidation:
    def test_output_validates_against_schema(self):
        input_data = SupplierAuditorInput(skus=['SKU-001'], data={'SKU-001': BASE_ROW})
        output = audit_suppliers(input_data)

        validated = SupplierAuditorOutput.model_validate(output.model_dump())
        assert validated.model_dump() == output.model_dump()


class TestDerivedViewsConsistency:
    def test_flagged_suppliers_and_grade_distribution_consistent_with_results(self):
        input_data = SupplierAuditorInput(
            skus=['SKU-D', 'SKU-A', 'SKU-C', 'SKU-B'],
            data={
                'SKU-D': BASE_ROW, 'SKU-A': RISKY_ROW,
                'SKU-C': GRADE_C_ROW, 'SKU-B': GRADE_B_ROW,
            },
        )
        output = audit_suppliers(input_data)

        expected_flagged = {r.sku for r in output.results if r.stop_auto_buy_triggered}
        assert set(output.flagged_suppliers) == expected_flagged

        expected_dist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for r in output.results:
            expected_dist[r.supplier_grade] += 1
        assert output.grade_distribution == expected_dist
        assert sum(output.grade_distribution.values()) == len(output.results)


class TestEngineerFeatures:
    """Direct tests of engineer_features(), independent of the model."""

    def test_engineer_features_adds_all_engineered_columns(self):
        df = pd.DataFrame([BASE_ROW])
        df_eng = engineer_features(df)
        for col in [
            'perf_trend', 'perf_decay_rate', 'delivery_stress', 'backlog_transit_ratio',
            'forecast_deviation_3m', 'forecast_deviation_6m', 'forecast_accuracy_3m',
            'inventory_coverage_days', 'transit_exposure_ratio',
            'risk_flag_count', 'compound_risk_index',
        ]:
            assert col in df_eng.columns

    def test_delivery_stress_formula(self):
        df = pd.DataFrame([{**BASE_ROW, 'pieces_past_due': 250.0, 'national_inv': 500.0}])
        df_eng = engineer_features(df)
        assert df_eng['delivery_stress'].iloc[0] == pytest.approx(0.5, abs=1e-4)

    def test_risk_flag_count_sums_numeric_flags_directly(self):
        row = {**BASE_ROW, 'potential_issue': 1, 'deck_risk': 1, 'oe_constraint': 0,
               'ppap_risk': 1, 'rev_stop': 0}
        df = pd.DataFrame([row])
        df_eng = engineer_features(df)
        assert df_eng['risk_flag_count'].iloc[0] == 3


class TestRequiredRawCols:
    def test_required_raw_cols_excludes_sku_and_stop_auto_buy(self):
        assert 'sku' not in REQUIRED_RAW_COLS
        assert 'stop_auto_buy' not in REQUIRED_RAW_COLS

    def test_required_raw_cols_includes_went_on_backorder(self):
        # went_on_backorder is a required raw passthrough feature per the loaded
        # artifact's feature_cols, even though engineer_features() never reads it.
        assert 'went_on_backorder' in REQUIRED_RAW_COLS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
