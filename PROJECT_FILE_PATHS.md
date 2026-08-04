# Project File Paths

Full inventory of tracked project files (excludes `venv/`, `.git/`, `__pycache__/`, `.DS_Store`).

## `.CLAUDE/` — spec documents
- `.CLAUDE/csv_quality_audit_spec.md`
- `.CLAUDE/database_spec.md`
- `.CLAUDE/demand_predictor_wrapper_spec.md`
- `.CLAUDE/forecast_optimizer_inference_tool_SPEC.md`
- `.CLAUDE/inventory_rebalancer_inference_tool_SPEC.md`
- `.CLAUDE/orchestrator_spec.md`
- `.CLAUDE/queue_migration_spec.md`
- `.CLAUDE/repository_layer_spec.md`
- `.CLAUDE/risk_detector_inference_tool_SPEC.md`
- `.CLAUDE/seed_data_spec.md`
- `.CLAUDE/settings.local.json`
- `.CLAUDE/supplier_auditor_inference_tool_SPEC.md`

## Root
- `.gitattributes`
- `.gitignore`
- `.gitignore.save`
- `QUICK_START.md`
- `requirements.txt`
- `features.py`
- `run_demand_predictor_example.py`
- `run_forecast_optimizer_example.py`
- `run_rebalancer_example.py`
- `run_risk_detector_example.py`
- `run_supplier_auditor_example.py`

## `.pytest_cache/`
- `.pytest_cache/.gitignore`
- `.pytest_cache/CACHEDIR.TAG`
- `.pytest_cache/README.md`
- `.pytest_cache/v/cache/lastfailed`
- `.pytest_cache/v/cache/nodeids`

## `audits/` — data quality audit
- `audits/__init__.py`
- `audits/csv_quality_audit.py`
- `audits/reports/csv_quality_report.json`
- `audits/reports/csv_quality_report.md`

## `data/` — runtime data
- `data/app.db`
- `data/seed_sample.parquet`

## `Dataset/` — raw/processed source data
- `Dataset/cleaned_dataset.csv`
- `Dataset/Processed_Dataset_with_SKU_Column.csv`
- `Dataset/RSM_Dataset.csv`

## `inference_tools/` — agent inference wrappers
- `inference_tools/__init__.py`
- `inference_tools/demand_predictor_tool.py`
- `inference_tools/forecast_optimizer_tool.py`
- `inference_tools/inventory_rebalancer_tool.py`
- `inference_tools/risk_detector_tool.py`
- `inference_tools/schemas.py`
- `inference_tools/supplier_auditor_tool.py`

## `Models/` — trained model artifacts
- `Models/Demand_Predictor/agent1_h6_model.joblib`
- `Models/Demand_Predictor/agent1_h6_model.pkl`
- `Models/forecast/agent3_bias_detector_catboost.pkl`
- `Models/forecast/agent3_bias_detector_lgbm.pkl`
- `Models/forecast/agent3_business_metrics.pkl`
- `Models/forecast/agent3_corrector_catboost.pkl`
- `Models/forecast/agent3_corrector_lgbm.pkl`
- `Models/forecast/agent3_corrector_xgb.pkl`
- `Models/forecast/agent3_feature_cols.pkl`
- `Models/forecast/agent3_thresholds.pkl`
- `Models/risk_detector&Inventory_rebalencer/ablation_results.csv`
- `Models/risk_detector&Inventory_rebalencer/brf_model.pkl`
- `Models/risk_detector&Inventory_rebalencer/catboost_model.pkl`
- `Models/risk_detector&Inventory_rebalencer/feature_columns.pkl`
- `Models/risk_detector&Inventory_rebalencer/feature_importance.csv`
- `Models/risk_detector&Inventory_rebalencer/final_test_metrics.json`
- `Models/risk_detector&Inventory_rebalencer/lightgbm_model.pkl`
- `Models/risk_detector&Inventory_rebalencer/model_q1.pkl`
- `Models/risk_detector&Inventory_rebalencer/optimal_threshold.pkl`
- `Models/risk_detector&Inventory_rebalencer/optimal_thresholds.pkl`
- `Models/risk_detector&Inventory_rebalencer/priority_list.csv`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_best_params.pkl`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_feature_columns.pkl`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_imputer.pkl`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_linear_baseline.pkl`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_metrics.json`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_urgency_weights.pkl`
- `Models/risk_detector&Inventory_rebalencer/rebalancer_xgboost.pkl`
- `Models/risk_detector&Inventory_rebalencer/scale_pos_weight.pkl`
- `Models/risk_detector&Inventory_rebalencer/stacking_proposed.pkl`
- `Models/risk_detector&Inventory_rebalencer/xgboost_model.pkl`
- `Models/supply_auditor/supplier_auditor_model.pkl`

## `Notebooks/` — original training notebooks
- `Notebooks/Data_Preprocessing (1).ipynb`
- `Notebooks/Demand Predictor Model (1).ipynb`
- `Notebooks/Forcast_opt.ipynb`
- `Notebooks/Inventory_Rebalencer.ipynb`
- `Notebooks/Risk_Detector_Classification_Model.ipynb`
- `Notebooks/supplier_auditor_model_train.ipynb`

## `run_commands/` — CLI cheat-sheets
- `run_commands/database_run.txt`
- `run_commands/ORCHESTRATOR_RUN_COMMANDS.txt`
- `run_commands/RUN_COMMANDS.txt`
- `run_commands/VERIFICATION_COMMANDS.txt`

## `scripts/` — operational CLI scripts
- `scripts/__init__.py`
- `scripts/add_new_sku.py`
- `scripts/run_migration.py`
- `scripts/run_single_prediction.py`
- `scripts/sample_skus.py`

## `src/` — application source

### `src/` (top level)
- `src/__init__.py`
- `src/pipeline_state.py`

### `src/db/`
- `src/db/__init__.py`
- `src/db/base.py`
- `src/db/columns.py`
- `src/db/create_db.py`
- `src/db/models.py`
- `src/db/seed.py`
- `src/db/verify_db.py`

### `src/orchestrator/`
- `src/orchestrator/__init__.py`
- `src/orchestrator/edges.py`
- `src/orchestrator/executor.py`
- `src/orchestrator/graph.py`
- `src/orchestrator/manifest.json`
- `src/orchestrator/manifest.py`
- `src/orchestrator/state.py`
- `src/orchestrator/tracing.py`

### `src/orchestrator/nodes/`
- `src/orchestrator/nodes/__init__.py`
- `src/orchestrator/nodes/agent_1_demand.py`
- `src/orchestrator/nodes/agent_2_risk.py`
- `src/orchestrator/nodes/agent_3_rebalancer.py`
- `src/orchestrator/nodes/agent_5_forecast_opt.py`
- `src/orchestrator/nodes/agent_6_auditor.py`

### `src/queue/`
- `src/queue/__init__.py`
- `src/queue/deletion_service.py`
- `src/queue/ingestion_service.py`
- `src/queue/models.py`
- `src/queue/queue_repository.py`
- `src/queue/sweep_service.py`

### `src/repository/`
- `src/repository/__init__.py`
- `src/repository/pipeline_service.py`
- `src/repository/run_repository.py`
- `src/repository/sku_ingestion.py`
- `src/repository/state_builder.py`

## `tests/` — test suite

### `tests/` (top level)
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_demand_predictor_tool.py`
- `tests/test_forecast_optimizer_tool.py`
- `tests/test_inventory_rebalancer_tool.py`
- `tests/test_repository.py`
- `tests/test_risk_detector_tool.py`
- `tests/test_sku_ingestion.py`
- `tests/test_supplier_auditor_tool.py`

### `tests/fixtures/`
- `tests/fixtures/__init__.py`
- `tests/fixtures/agent_1_reference.parquet`
- `tests/fixtures/stub_executor.py`

### `tests/orchestrator/`
- `tests/orchestrator/__init__.py`
- `tests/orchestrator/conftest.py`
- `tests/orchestrator/test_end_to_end.py`
- `tests/orchestrator/test_executor_contract.py`
- `tests/orchestrator/test_graph_topology.py`
- `tests/orchestrator/test_suppression.py`

### `tests/queue/`
- `tests/queue/__init__.py`
- `tests/queue/conftest.py`
- `tests/queue/test_deletion_service.py`
- `tests/queue/test_ingestion_service.py`
- `tests/queue/test_queue_repository.py`
- `tests/queue/test_sweep_service.py`
