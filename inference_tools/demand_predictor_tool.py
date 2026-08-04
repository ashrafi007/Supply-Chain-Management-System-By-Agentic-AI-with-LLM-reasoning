"""
Demand Predictor Inference Tool — Agent 1. Pure model inference, no
LangChain/LangGraph/LLM dependencies (mirrors the other tools in this package).

Multi-stage decision (spec §2): **Option A.** The 6-month horizon (`h6`,
predicting `sales_6_month`) is the single canonical demand signal that populates
`PipelineState.demand_forecast`. It is the usual replenishment horizon and the
only stage with a trained bundle today. The tool accepts an optional `stage`
argument so an h3/h9 bundle can be dropped in later without an interface change;
`PipelineState` and the `predictions` table are unchanged (no dict-valued forecast).

What this model is (spec §1): a LightGBM regressor with a **log-ratio correction
head**. It does not predict units directly — it predicts a log-space delta against
a naive baseline `b = sales_3_month * 2` (for h6), which is inverted, multiplied by
a validation-tuned Duan smearing factor, and clipped at zero. Three tuned/learned
quantities are loaded from the bundle and never recomputed at serve time:
`best_iteration` (boosting rounds), `smearing_factor` (Duan), and the baseline rule.

Bundle audit result (spec §3): the shipped bundle contains
`model, stage, target, features, baseline_name, smearing_factor, best_iteration,
test_metrics, naive_metrics, transform`. It does **not** contain an imputer/scaler
pipeline — and correctly so: the proposed LightGBM model was trained on the *raw*
(unscaled) feature frame (notebook cell 16), and every notebook prediction uses raw
features. The SimpleImputer→RobustScaler in the notebook exists only for the linear
/RF comparison baselines, never for this model. Applying a scaler here would *break*
parity, so §5's transform step is deliberately skipped. LightGBM consumes NaN
natively; no imputation is invented at serve time.

The tool never reads or writes the database — it operates purely on in-memory
state and is an entry-point graph node (no upstream dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

import features as F

# Bundle location, resolved relative to the repo root (parent of this package) —
# same style as the sibling tools. Only the h6 Demand Predictor ships today.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUNDLE_PATHS: dict[str, Path] = {
    "h6": _REPO_ROOT / "Models" / "Demand_Predictor" / "agent1_h6_model.joblib",
}


@dataclass(frozen=True)
class DemandBundle:
    """Immutable view of the loaded on-disk artifacts."""
    model: Any
    stage: str
    target: str
    feature_cols: list[str]
    baseline_name: str
    smearing_factor: float
    best_iteration: int
    model_version: str


class DemandPredictorTool:
    """
    Uniform-interface inference tool wrapping the trained Demand Predictor.

    Construct once (loads the bundle from disk), then call:
      - ``run(state)``      -> LangGraph node; reads ``state["raw_features"]``,
                               returns a state delta ``{"demand_forecast": ...}``
                               plus an appended ``trace`` entry.
      - ``predict(rows)``   -> batch inference for tests/orchestration; returns a
                               numpy array of demand values.
    """

    NAME = "demand_predictor"

    def __init__(self, stage: str = "h6", bundle_file: str | Path | None = None):
        if bundle_file is not None:
            path = Path(bundle_file)
        else:
            if stage not in _BUNDLE_PATHS:
                raise KeyError(f"no bundle for stage {stage!r}; known: {sorted(_BUNDLE_PATHS)}")
            path = _BUNDLE_PATHS[stage]
        if not path.exists():
            raise FileNotFoundError(f"Demand Predictor bundle not found at {path}")

        raw = joblib.load(path)

        # Guard: a stage mismatch between the requested horizon and the shipped
        # bundle would silently serve the wrong model/baseline.
        if raw["stage"] != stage:
            raise ValueError(
                f"bundle at {path} is stage {raw['stage']!r}, but tool asked for {stage!r}"
            )

        model = raw["model"]
        # Serving is CPU-only (Apple Silicon). If the saved params ever carry a
        # GPU device, strip it so predict cannot force a GPU path at inference.
        try:
            if getattr(model, "get_params", None) and model.get_params().get("device") == "gpu":
                model.set_params(device="cpu")
        except Exception:
            pass

        self.bundle = DemandBundle(
            model=model,
            stage=raw["stage"],
            target=raw["target"],
            feature_cols=list(raw["features"]),
            baseline_name=raw["baseline_name"],
            smearing_factor=float(raw["smearing_factor"]),
            best_iteration=int(raw["best_iteration"]),
            model_version=f"agent1_{raw['stage']}_lgbm_logratio",
        )

        # Sanity: the executable baseline rule must match the string the notebook saved.
        if F.STAGE_SPECS[stage]["baseline_name"] != raw["baseline_name"]:
            raise ValueError(
                f"baseline rule mismatch: features.py has "
                f"{F.STAGE_SPECS[stage]['baseline_name']!r}, bundle has {raw['baseline_name']!r}"
            )

        # Required raw inputs: features.py's raw set plus this stage's baseline extras.
        self.required_cols: list[str] = list(
            dict.fromkeys(F.DEMAND_RAW_INPUT_COLS + F.STAGE_SPECS[stage]["raw_extra"])
        )

    # -- core batch inference (spec §5) -------------------------------------
    def predict(self, rows: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> np.ndarray:
        """
        Run the full §5 sequence on one or many raw rows and return demand units.

        Accepts a DataFrame, a list of raw-column dicts, or a single dict.
        Raises ``ValueError`` on missing required columns or on non-finite values
        surviving into the feature matrix (fail loudly rather than feed LightGBM
        garbage). Does not mutate the input.
        """
        df = self._as_frame(rows)

        # 1. validate inputs
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing required raw column(s): {missing}")

        # 2. clean (value-level): none beyond the notebook's own formulas — the
        #    proposed model was trained on the raw frame as-is, so extra cleaning
        #    would diverge from training. features.engineer_demand_features holds
        #    every value-level transform the model expects.
        # 3-4. build baseline b and the FEATURE_COLS matrix (in order)
        X, b = F.build_demand_matrix(df, self.bundle.feature_cols, self.bundle.stage)

        # 5. transform: SKIPPED — no scaler in the bundle; model expects raw features.

        # guard against inf/NaN surviving into X (spec §7): a cleaning gap, not a
        # value LightGBM should silently bin.
        arr = X.to_numpy()
        if not np.isfinite(arr).all():
            bad = X.columns[~np.isfinite(arr).all(axis=0)].tolist()
            raise ValueError(f"non-finite values in feature matrix column(s): {bad}")

        # 6. predict the delta at the tuned number of rounds
        delta = self.bundle.model.predict(X, num_iteration=self.bundle.best_iteration)

        # 7. invert + smear + clip at zero
        demand = F.from_delta(np.asarray(delta), b, self.bundle.smearing_factor)
        demand = np.maximum(demand, 0.0)

        # 8. round consistently with the notebook
        return np.round(demand, 2)

    # -- uniform interface (spec §6) ----------------------------------------
    def run(self, state: Mapping[str, Any]) -> dict:
        """
        LangGraph node. Reads ``state["raw_features"]`` (a single raw-column dict
        for the canonical single-SKU path), runs inference, and returns a state
        delta ``{"demand_forecast": <float|None>}`` plus an appended ``trace`` entry.

        On failure the delta records the reason in ``trace`` and leaves
        ``demand_forecast`` as ``None`` so the graph continues; downstream tools
        that do not depend on demand still run. Never raises.
        """
        trace = list(state.get("trace", []))
        raw = state.get("raw_features")
        try:
            if raw is None:
                raise ValueError("state['raw_features'] is missing")
            # Canonical path is a single SKU -> a single float.
            if isinstance(raw, Mapping):
                value = float(self.predict(raw)[0])
            else:
                # A batch was handed in; return a list, first element unused by the
                # single-valued PipelineState but preserved for callers that batch.
                value = [float(v) for v in self.predict(raw)]
            trace.append({
                "tool": self.NAME,
                "status": "ok",
                "model_version": self.bundle.model_version,
                "stage": self.bundle.stage,
            })
            return {"demand_forecast": value, "trace": trace}
        except Exception as exc:
            trace.append({
                "tool": self.NAME,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return {"demand_forecast": None, "trace": trace}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _as_frame(rows) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            return rows
        if isinstance(rows, Mapping):
            return pd.DataFrame([dict(rows)])
        return pd.DataFrame([dict(r) for r in rows])
