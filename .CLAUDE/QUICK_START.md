# Risk Detector Inference Tool — Quick Start Guide

## Activate Virtual Environment

```bash
source venv/bin/activate
```

## Run All Tests

```bash
# Full test suite with verbose output
python -m pytest tests/test_risk_detector_tool.py -v

# Run specific test class
python -m pytest tests/test_risk_detector_tool.py::TestNormalInput -v

# Run single test
python -m pytest tests/test_risk_detector_tool.py::TestNormalInput::test_normal_input_produces_valid_output -v

# Run with short output
python -m pytest tests/test_risk_detector_tool.py -q

# Run with detailed failure info
python -m pytest tests/test_risk_detector_tool.py -vvs
```

## Test Coverage

```bash
# Install pytest-cov (if not already installed)
pip install pytest-cov

# Run tests with coverage report
python -m pytest tests/test_risk_detector_tool.py --cov=inference_tools --cov-report=term-missing
```

## Quick Inference Test

```bash
python3 << 'PYTHON'
from inference_tools.risk_detector_tool import predict_risk
from inference_tools.schemas import RiskDetectorInput

# Create test data
test_sku_data = {
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

# Run prediction
input_data = RiskDetectorInput(
    skus=['SKU-TEST-001'],
    data={'SKU-TEST-001': test_sku_data}
)
output = predict_risk(input_data)

# Print results
print("✓ Inference successful!")
print(f"  SKU: {output.predictions[0].sku}")
print(f"  Backorder Probability: {output.predictions[0].backorder_probability:.4f}")
print(f"  Predicted Label (≥0.5): {output.predictions[0].predicted_label}")
print(f"  Is High Risk (≥0.945): {output.predictions[0].is_high_risk}")
print(f"  Threshold Used: {output.threshold_used}")
print(f"  Model Version: {output.model_version}")
PYTHON
```

## Multiple SKU Prediction

```bash
python3 << 'PYTHON'
from inference_tools.risk_detector_tool import predict_risk
from inference_tools.schemas import RiskDetectorInput

# Create data for 3 SKUs with different inventory levels
base_data = {
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

input_data = RiskDetectorInput(
    skus=['SKU-001', 'SKU-002', 'SKU-003'],
    data={
        'SKU-001': base_data,
        'SKU-002': {**base_data, 'national_inv': 50.0},  # Low inventory
        'SKU-003': {**base_data, 'national_inv': 200.0},  # High inventory
    }
)

output = predict_risk(input_data)

print(f"✓ Predicted {len(output.predictions)} SKUs")
for pred in output.predictions:
    print(f"  {pred.sku}: prob={pred.backorder_probability:.4f}, "
          f"label={pred.predicted_label}, high_risk={pred.is_high_risk}")

print(f"\nHigh-risk SKUs: {output.high_risk_skus if output.high_risk_skus else 'None'}")
PYTHON
```

## Check Model Cache

```bash
python3 << 'PYTHON'
from inference_tools.risk_detector_tool import load_risk_detector_model

# Clear cache (starts fresh)
load_risk_detector_model.cache_clear()
print(f"Cache cleared: {load_risk_detector_model.cache_info()}")

# Load model (cache miss)
model1 = load_risk_detector_model()
print(f"After 1st load: {load_risk_detector_model.cache_info()}")

# Load again (cache hit)
model2 = load_risk_detector_model()
print(f"After 2nd load: {load_risk_detector_model.cache_info()}")

# Verify same object
print(f"Same instance: {model1 is model2}")
PYTHON
```

## Verify No LangChain/LangGraph Dependencies

```bash
grep -rE "langchain|langgraph" inference_tools/risk_detector_tool.py || echo "✓ Clean"
```

## Check Dependencies

```bash
pip list | grep -E "pandas|numpy|scikit-learn|xgboost|lightgbm|catboost|pydantic|pytest"
```

## Deactivate Virtual Environment

```bash
deactivate
```

## File Structure

```
SupplyChainAgenticAi/
├── inference_tools/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models & exceptions
│   └── risk_detector_tool.py   # Main inference module
├── tests/
│   ├── __init__.py
│   └── test_risk_detector_tool.py  # 18 comprehensive tests
├── Models/
│   └── risk_detector&Inventory_rebalencer/
│       ├── stacking_proposed.pkl
│       ├── feature_columns.pkl
│       └── optimal_threshold.pkl
├── venv/                       # Python virtual environment
└── requirements.txt            # Dependencies
```

## Common Issues & Fixes

### Issue: "No module named 'inference_tools'"
**Fix:** Make sure you're in the repo root and venv is activated
```bash
cd /Users/aunggonchowdhury/Documents/SupplyChainAgenticAi
source venv/bin/activate
```

### Issue: Model file not found
**Fix:** Ensure Models directory structure exists
```bash
ls -la Models/risk_detector&Inventory_rebalencer/stacking_proposed.pkl
```

### Issue: Tests are slow (first run)
**Normal:** First inference loads the model from disk (~30s). Subsequent calls use cache (<1s).

### Issue: Deprecation warnings about NumPy
**Normal:** Joblib has minor NumPy 2.5 deprecation warnings. Doesn't affect functionality.
