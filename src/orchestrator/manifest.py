"""Loads models/manifest.json, validates + warms every artifact at import time, exposes MANIFEST_VERSION and per-agent path constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inference_tools.demand_predictor_tool import DemandPredictorTool
from inference_tools.forecast_optimizer_tool import load_forecast_optimizer_artifacts
from inference_tools.inventory_rebalancer_tool import load_rebalancer_artifacts
from inference_tools.risk_detector_tool import load_risk_detector_model
from inference_tools.supplier_auditor_tool import load_supplier_auditor_artifact


class ManifestError(Exception):
    """Raised when a manifest-listed artifact is missing or fails to load."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Co-located with this module, not under a top-level models/ dir: this repo's
# Models/ (capital M) already holds binary artifacts, and on case-insensitive
# filesystems (macOS default) models/ and Models/ collide onto the same
# directory -- see this file's own docstring note in manifest.json.
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

with open(MANIFEST_PATH) as _f:
    _RAW_REGISTRY: dict[str, list[str]] = {
        k: v for k, v in json.load(_f).items() if not k.startswith("_")
    }

REGISTRY: dict[str, list[Path]] = {}
for _agent, _rel_paths in _RAW_REGISTRY.items():
    resolved = []
    for _rel in _rel_paths:
        _p = PROJECT_ROOT / _rel
        if not _p.exists():
            raise ManifestError(f"manifest entry {_agent!r} points at missing file: {_p}")
        resolved.append(_p)
    REGISTRY[_agent] = resolved


def _artifact_fingerprint() -> str:
    """
    Deterministic version string derived from (relative_path, size, mtime_ns)
    of every registered artifact, sorted for stability. Changes automatically
    if any artifact file is replaced or updated.
    """
    parts = []
    for agent in sorted(REGISTRY):
        for p in REGISTRY[agent]:
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            stat = p.stat()
            parts.append((rel, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return f"sha256:{digest[:16]}"


MANIFEST_VERSION: str = _artifact_fingerprint()

# Per-agent path constants for node files to import (single source of truth).
DEMAND_PREDICTOR_MODEL_PATH: Path = REGISTRY["agent_1_demand_predictor"][0]
RISK_DETECTOR_MODEL_PATH: Path = REGISTRY["agent_2_risk_detector"][0]
REBALANCER_MODEL_PATH: Path = REGISTRY["agent_3_rebalancer"][0]
FORECAST_OPTIMIZER_MODELS_DIR: Path = REGISTRY["agent_5_forecast_optimizer"][0].parent
SUPPLIER_AUDITOR_MODEL_PATH: Path = REGISTRY["agent_6_supplier_auditor"][0]

# Warm + validate each artifact via its own tool's loader at import time, so a
# malformed (not just missing) artifact also fails loudly before any node runs.
DEMAND_PREDICTOR_TOOL = DemandPredictorTool(bundle_file=DEMAND_PREDICTOR_MODEL_PATH)
load_risk_detector_model(str(RISK_DETECTOR_MODEL_PATH))
load_rebalancer_artifacts(str(REBALANCER_MODEL_PATH.parent))
load_forecast_optimizer_artifacts(str(FORECAST_OPTIMIZER_MODELS_DIR))
load_supplier_auditor_artifact(str(SUPPLIER_AUDITOR_MODEL_PATH))


if __name__ == "__main__":
    print("MANIFEST_VERSION:", MANIFEST_VERSION)
    for agent, paths in REGISTRY.items():
        print(f"\n{agent}:")
        for p in paths:
            print(f"  {p}")
