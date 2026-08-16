import { ExplainResult, RunDetail } from "../api";
import { SparkleIcon } from "../icons";
import { RadarAxis, RadarChart } from "./RadarChart";

export const GRADE_COLOR: Record<string, string> = {
  A: "var(--status-good)",
  B: "var(--status-warn)",
  C: "var(--status-orange)",
  D: "var(--status-bad)",
};

// Grade -> probability-band midpoint, matching AGENT_OUTPUT_MEANINGS.md's real
// A:[0,.2) B:[.2,.45) C:[.45,.7) D:[.7,1] bands -- not an arbitrary score.
const GRADE_MIDPOINT: Record<string, number> = { A: 0.1, B: 0.325, C: 0.575, D: 0.85 };

/** Per-SKU "risk profile" -- four axes that are all genuinely 0-1 "higher = needs
 * more attention" scores already, by this system's own design (no invented
 * normalization). demand_forecast is deliberately left out: it's an unbounded
 * volume number, not a 0-1 risk score, and forcing it onto this scale would be
 * the kind of misleading normalization a radar chart is already prone to. */
function buildRiskProfileAxes(p: RunDetail["prediction"]): RadarAxis[] {
  const correctionDeviation =
    p?.correction_factor != null ? Math.max(0, Math.min(1, Math.abs(p.correction_factor - 1) / 0.5)) : 0;
  return [
    {
      label: "Backorder Risk",
      value: p?.backorder_prob ?? 0,
      max: 1,
      displayValue: p?.backorder_prob != null ? `${(p.backorder_prob * 100).toFixed(0)}%` : "n/a",
    },
    {
      label: "Restock Urgency",
      value: p?.urgency_score ?? 0,
      max: 1,
      displayValue: p?.urgency_score != null ? p.urgency_score.toFixed(2) : "n/a",
    },
    {
      label: "Supplier Risk",
      value: p?.supplier_risk ? (GRADE_MIDPOINT[p.supplier_risk] ?? 0) : 0,
      max: 1,
      displayValue: p?.supplier_risk ?? "n/a",
    },
    {
      label: "Forecast Correction",
      value: correctionDeviation,
      max: 1,
      displayValue: p?.correction_factor != null ? `${p.correction_factor.toFixed(2)}x` : "n/a",
    },
  ];
}

type AgentTile = {
  label: string;
  render: (p: RunDetail["prediction"]) => string;
  accent: (p: RunDetail["prediction"]) => string | undefined;
};

const AGENT_TILES: AgentTile[] = [
  {
    label: "Demand Predictor",
    render: (p) => (p?.demand_forecast != null ? `${p.demand_forecast.toFixed(2)} units / 6mo` : "n/a"),
    accent: () => undefined,
  },
  {
    label: "Risk Detector",
    render: (p) => (p?.backorder_prob != null ? `${(p.backorder_prob * 100).toFixed(1)}% backorder risk` : "n/a"),
    accent: (p) => (p?.alarm_triggered ? "var(--status-bad)" : "var(--status-good)"),
  },
  {
    label: "Inventory Rebalancer",
    render: (p) => (p?.urgency_score != null ? `${p.urgency_score.toFixed(2)} urgency` : "n/a"),
    accent: () => undefined,
  },
  {
    label: "Forecast Optimizer",
    render: (p) => (p?.correction_factor != null ? `${p.correction_factor.toFixed(2)}x adjustment` : "n/a"),
    accent: () => undefined,
  },
  {
    label: "Supplier Auditor",
    render: (p) => (p?.supplier_risk ? `Grade ${p.supplier_risk}` : "n/a"),
    accent: (p) => (p?.supplier_risk ? GRADE_COLOR[p.supplier_risk] : undefined),
  },
];

/** The full "what did the orchestrator do, what does the AI say" card -- shared by
 * the Dashboard (one per queued SKU) and Add Order (the SKU you just ran). */
export function SkuResultCard({
  skuId,
  run,
  explanation,
  error,
  titlePrefix = "SKU",
}: {
  skuId: string;
  run: RunDetail | null;
  explanation: ExplainResult | null;
  error: string | null;
  titlePrefix?: string;
}) {
  return (
    <div className="card sku-card">
      <div className="sku-card__header">
        <h2 className="card__title">
          {titlePrefix} {skuId}
        </h2>
        {run && <span className={`pill pill--${run.status}`}>{run.status}</span>}
      </div>

      {error && <div className="banner banner--error banner--tight">{error}</div>}

      {run && (
        <>
          <div className="agent-grid">
            {AGENT_TILES.map((tile) => (
              <div className="agent-tile" key={tile.label}>
                <div className="agent-tile__label">{tile.label}</div>
                <div className="agent-tile__value" style={{ color: tile.accent(run.prediction) ?? "var(--text-primary)" }}>
                  {tile.render(run.prediction)}
                </div>
              </div>
            ))}
          </div>

          <div className="sku-card__profile">
            <div className="sku-card__profile-radar">
              <RadarChart axes={buildRiskProfileAxes(run.prediction)} size={300} />
            </div>
            <div className="sku-card__profile-side">
              <div className="sku-card__profile-title">Risk profile</div>
              <p className="prose prose--muted sku-card__profile-note">
                Backorder risk, restock urgency, supplier risk, and forecast correction —
                all 0–1 scores, higher means more attention needed. Demand forecast isn't
                on this scale (it's a unit count, not a risk score):
              </p>
              <div className="sku-card__demand-stat">
                {run.prediction?.demand_forecast != null ? `${run.prediction.demand_forecast.toFixed(0)} units` : "n/a"}
                <span className="muted"> / 6 months</span>
              </div>
            </div>
          </div>

          <div className="explanation">
            <div className="explanation__header">
              <SparkleIcon size={15} />
              <span>AI Explanation</span>
              {explanation && (
                <span className={`badge ${explanation.was_polished ? "badge--polished" : "badge--draft"}`}>
                  {explanation.was_polished ? explanation.model_used : "template (no LLM)"}
                </span>
              )}
            </div>
            <p className="explanation__text">
              {explanation ? explanation.explanation : "No explanation available for this run."}
            </p>
          </div>
        </>
      )}

      {!run && !error && <p className="muted">This SKU hasn't been run through the pipeline yet.</p>}
    </div>
  );
}
