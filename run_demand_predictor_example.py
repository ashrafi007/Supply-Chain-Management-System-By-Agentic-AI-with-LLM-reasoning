"""
Runnable example / smoke-test for the Demand Predictor inference tool (Agent 1).

Usage (from the repo root, inside the venv):

    ./venv/bin/python run_demand_predictor_example.py

It loads the h6 bundle once, then runs the tool three ways:
  1. batch predict() over a few real SKUs spanning demand regimes,
  2. run(state) as a LangGraph node (single SKU -> state delta + trace),
  3. the failure path (missing required column -> demand_forecast=None).
"""

import pandas as pd

from inference_tools.demand_predictor_tool import DemandPredictorTool

FIXTURE = "tests/fixtures/agent_1_reference.parquet"


def main() -> None:
    tool = DemandPredictorTool(stage="h6")
    b = tool.bundle
    print(f"Loaded Demand Predictor  stage={b.stage}  target={b.target}")
    print(f"  baseline='{b.baseline_name}'  rounds={b.best_iteration}  "
          f"smear={b.smearing_factor:.5f}  features={len(b.feature_cols)}\n")

    ref = pd.read_parquet(FIXTURE)
    raw = [c for c in ref.columns if c not in ("_raw_idx", "notebook_demand")]

    # A few real SKUs across demand regimes.
    sample = pd.concat([
        ref.iloc[[ref.notebook_demand.idxmax()]],          # biggest mover
        ref[ref.notebook_demand.between(50, 200)].head(1),  # steady
        ref[ref.notebook_demand == 0].head(1),              # dead
    ])

    print("1) batch predict()")
    for (_, r), p in zip(sample.iterrows(), tool.predict(sample[raw])):
        print(f"   sales_1m={r.sales_1_month:>9.0f} sales_3m={r.sales_3_month:>9.0f} "
              f"nat_inv={r.national_inv:>9.0f}  ->  6-mo demand = {p:>12.2f}")

    print("\n2) run(state) — LangGraph node, single SKU")
    one = sample[raw].iloc[0].to_dict()
    out = tool.run({"raw_features": one, "trace": []})
    print("   state delta:", {"demand_forecast": out["demand_forecast"]})
    print("   trace entry:", out["trace"][-1])

    print("\n3) failure path — missing required column")
    bad = dict(one)
    del bad["sales_3_month"]
    fail = tool.run({"raw_features": bad, "trace": []})
    print("   demand_forecast:", fail["demand_forecast"])
    print("   trace entry:", fail["trace"][-1])


if __name__ == "__main__":
    main()
