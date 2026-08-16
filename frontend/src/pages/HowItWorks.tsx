import { ArchitectureDiagram } from "../components/ArchitectureDiagram";
import { BarCompare } from "../components/BarCompare";
import { PipelineDiagram } from "../components/PipelineDiagram";
import { RadarChart } from "../components/RadarChart";
import { StatTile } from "../components/StatTile";

const AGENT_TABLE: { name: string; model: string; decides: string; key: string }[] = [
  {
    name: "Demand Predictor",
    model: "LightGBM (log-ratio regression + Duan smearing correction)",
    decides: "Units expected to sell over the next 6 months",
    key: "R² = 0.99 on held-out test data (vs. 0.95 for a naive guess — see chart below)",
  },
  {
    name: "Risk Detector",
    model: "Stacking ensemble classifier",
    decides: "Probability this product goes on backorder",
    key: "Alert threshold τ = 0.945 (F2-optimized — deliberately favors catching real risk over avoiding false alarms)",
  },
  {
    name: "Inventory Rebalancer",
    model: "XGBoost regressor",
    decides: "Restock urgency, 0–1 composite score",
    key: "Ranks urgency 8× more accurately than a linear-regression baseline (statistically significant, p < 0.001)",
  },
  {
    name: "Forecast Optimizer",
    model: "LightGBM (bias detector) + XGBoost (corrector)",
    decides: "Whether the human-made forecast needs correcting",
    key: "Business rule: never shrinks the forecast for a product already at risk of stocking out",
  },
  {
    name: "Supplier Auditor",
    model: "XGBoost classifier + rule-based safety net",
    decides: "Supplier reliability grade, A (best) to D (critical)",
    key: "99.4% F2-score on held-out test data (see chart below)",
  },
];

// All figures below are read directly from this repo's own evaluation artifacts
// (Models/*/*.json, the demand model's embedded test_metrics, agent3_business_metrics.pkl)
// -- nothing here is estimated or invented for the page.

const DEMAND_NAIVE_MAE = 48.22;
const DEMAND_MODEL_MAE = 33.56;
const DEMAND_NAIVE_RMSE = 2497.71;
const DEMAND_MODEL_RMSE = 1075.92;
const DEMAND_MAE_IMPROVEMENT = Math.round((1 - DEMAND_MODEL_MAE / DEMAND_NAIVE_MAE) * 100);
const DEMAND_RMSE_IMPROVEMENT = Math.round((1 - DEMAND_MODEL_RMSE / DEMAND_NAIVE_RMSE) * 100);

const RISK_DETECTOR_AXES = [
  { label: "Recall", value: 0.5744, max: 1, displayValue: "0.57" },
  { label: "Precision", value: 0.2373, max: 1, displayValue: "0.24" },
  { label: "F2-Score", value: 0.4473, max: 1, displayValue: "0.45" },
  { label: "ROC-AUC", value: 0.9655, max: 1, displayValue: "0.97" },
  { label: "MCC", value: 0.3624, max: 1, displayValue: "0.36" },
];

const SUPPLIER_AUDITOR_AXES = [
  { label: "Accuracy", value: 0.9895, max: 1, displayValue: "0.99" },
  { label: "F2-Score", value: 0.9944, max: 1, displayValue: "0.99" },
  { label: "ROC-AUC", value: 0.9938, max: 1, displayValue: "0.99" },
  { label: "PR-AUC", value: 0.9997, max: 1, displayValue: "1.00" },
  { label: "MCC", value: 0.8524, max: 1, displayValue: "0.85" },
];

const REBALANCER_RANK_BARS = [
  { label: "XGBoost (used in production)", value: 0.9761, max: 1, displayValue: "ρ = 0.98", highlight: true },
  { label: "Linear regression (baseline)", value: 0.1239, max: 1, displayValue: "ρ = 0.12" },
];

export function HowItWorks() {
  return (
    <div className="page page--wide">
      <div className="page__header">
        <div>
          <h1>How This Works</h1>
          <p className="page__subtitle">What the system does, in plain English — and how it actually works, underneath.</p>
        </div>
      </div>

      {/* ---------------- Section 1: plain English ---------------- */}
      <section className="card">
        <span className="section-eyebrow">In plain English</span>
        <h2 className="card__title card__title--lg">What does this actually do?</h2>
        <p className="prose">
          Imagine a warehouse with thousands of products. For every single one, someone has to answer:
          how much will we sell? Will we run out? How urgently do we need to reorder? Is our supplier
          reliable? Normally, a person has to guess at all of this using spreadsheets and experience.
        </p>
        <p className="prose">
          This system answers all four questions <strong>automatically</strong>, using AI models trained
          on real sales history — and instead of handing back a wall of numbers, it writes the answer out
          in plain English, like a colleague explaining it to you.
        </p>
        <div className="diagram-wrap">
          <PipelineDiagram />
        </div>
        <p className="prose prose--muted">
          Each box above is a different check. They run one after another, and each one can see what the
          checks before it decided — for example, if a product is flagged as high-risk, the system will
          not suggest cutting its production plan, even if the raw numbers might otherwise suggest that.
          That's a deliberate safety rule, not an accident.
        </p>

        <h3 className="form-section__title">Is the AI actually better than guessing?</h3>
        <p className="prose">
          For the sales-forecasting check specifically, here's a direct comparison: predicting demand by
          simply doubling last quarter's sales (a common manual rule of thumb) versus the trained AI model
          — both measured against what actually happened.
        </p>
        <div className="stat-row">
          <StatTile
            label="Average error — simple guess"
            value={`${DEMAND_NAIVE_MAE.toFixed(0)} units`}
            sublabel="doubling last quarter's sales"
          />
          <StatTile
            label="Average error — AI model"
            value={`${DEMAND_MODEL_MAE.toFixed(0)} units`}
            sublabel={`${DEMAND_MAE_IMPROVEMENT}% lower error`}
            tone="good"
          />
        </div>
        <p className="prose prose--muted">
          Same story on bigger misses (the errors that actually hurt): {DEMAND_RMSE_IMPROVEMENT}% lower
          than the simple-guess approach. Both numbers are measured on real held-out sales data the model
          never saw during training.
        </p>
      </section>

      {/* ---------------- Section 2: technical ---------------- */}
      <section className="card">
        <span className="section-eyebrow">For a technical reviewer</span>
        <h2 className="card__title card__title--lg">Architecture &amp; process</h2>

        <div className="tech-summary">
          <strong>30-second technical summary:</strong> a LangGraph state machine orchestrates 5
          independently-trained ML models (LightGBM, XGBoost, and a stacking ensemble) in a fixed
          sequence, with one hard-coded business-rule override between the Risk Detector and Forecast
          Optimizer. A FastAPI backend
          persists every run (inputs, per-agent trace, outputs) to SQLite. Explanations are template-first
          and fact-grounded — an LLM only rephrases them, under a no-new-facts constraint, with automatic
          fallback to the deterministic draft on any API failure. Frontend is Tauri + React over that same
          REST API.
        </div>

        <p className="prose">
          Five machine-learning agents run in a fixed sequence, orchestrated as a{" "}
          <strong>LangGraph state machine</strong>. Every run's inputs, outputs, and per-agent trace are
          persisted, so any prediction can be audited after the fact. (A sixth agent, "Routing," was
          planned and explicitly cancelled before launch.)
        </p>
        <div className="diagram-wrap">
          <ArchitectureDiagram />
        </div>

        <h3 className="form-section__title">The five agents</h3>
        <div className="table-scroll">
          <table className="table agent-table">
            <thead>
              <tr>
                <th>Agent</th>
                <th>Model</th>
                <th>Decides</th>
                <th>Key detail</th>
              </tr>
            </thead>
            <tbody>
              {AGENT_TABLE.map((a) => (
                <tr key={a.name}>
                  <td className="agent-table__name">{a.name}</td>
                  <td className="muted">{a.model}</td>
                  <td>{a.decides}</td>
                  <td className="muted">{a.key}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className="form-section__title">Explaining the results honestly</h3>
        <p className="prose">
          A separate layer turns raw predictions into plain English. It always builds a deterministic,
          fact-grounded draft first — every number in it is traced directly to a real prediction, never
          invented. An LLM (via OpenRouter) then optionally rewrites that draft in more natural language,
          under a strict rule: it may improve phrasing, but it can never add or change a fact or number.
          If the AI call fails or is unavailable, the system falls back to the deterministic draft
          automatically — the explanation is never lost, just less polished.
        </p>
      </section>

      {/* ---------------- Section 3: model performance, one subsection per agent ---------------- */}
      <section className="card">
        <span className="section-eyebrow">Model performance</span>
        <h2 className="card__title card__title--lg">Real evaluation results, per agent</h2>
        <p className="prose prose--muted">
          Every number below comes from this project's own held-out test evaluations — read directly from
          the trained model artifacts, not estimated for this page.
        </p>

        <h3 className="form-section__title">Demand Predictor</h3>
        <div className="stat-row">
          <StatTile label="R² (model)" value="0.99" sublabel="vs. 0.95 for naive baseline" tone="good" />
          <StatTile label="MAE" value={DEMAND_MODEL_MAE.toFixed(1)} sublabel={`units, ${DEMAND_MAE_IMPROVEMENT}% lower than naive`} tone="good" />
          <StatTile label="RMSE" value={DEMAND_MODEL_RMSE.toFixed(0)} sublabel={`${DEMAND_RMSE_IMPROVEMENT}% lower than naive`} tone="good" />
        </div>

        <h3 className="form-section__title">Risk Detector</h3>
        <p className="prose prose--muted">
          Precision is intentionally low here — the decision threshold (τ = 0.945) is F2-optimized to
          prioritize recall, because missing a real stockout costs far more than an extra false alarm.
          Backorders are also rare in the training data (~0.7% of rows), which naturally caps precision at
          any recall-favoring threshold.
        </p>
        <div className="radar-wrap">
          <RadarChart axes={RISK_DETECTOR_AXES} size={300} />
        </div>

        <h3 className="form-section__title">Inventory Rebalancer</h3>
        <p className="prose prose--muted">
          What matters operationally is <em>ranking</em> SKUs by urgency correctly, not the exact number —
          measured here with Spearman rank correlation (ρ) against a linear-regression baseline, on 1,000
          bootstrap samples (Wilcoxon test, p &lt; 0.001).
        </p>
        <BarCompare items={REBALANCER_RANK_BARS} />
        <div className="stat-row stat-row--compact">
          <StatTile label="R²" value="0.998" tone="good" />
          <StatTile label="RMSE" value="0.0055" />
          <StatTile label="MAE" value="0.0027" />
        </div>

        <h3 className="form-section__title">Forecast Optimizer</h3>
        <p className="prose prose--muted">
          This agent's evaluation is model-selection-based rather than a single accuracy score: three
          candidate correction models were trained and compared, and the best performer was selected for
          production.
        </p>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Candidate model</th>
                <th>Selected threshold</th>
                <th>Chosen for production?</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>LightGBM</td>
                <td className="mono">0.765</td>
                <td>
                  <span className="pill" style={{ background: "var(--status-good)" }}>yes</span>
                </td>
              </tr>
              <tr>
                <td>CatBoost</td>
                <td className="mono">0.780</td>
                <td className="muted">no</td>
              </tr>
              <tr>
                <td>Balanced Random Forest</td>
                <td className="mono">0.715</td>
                <td className="muted">no</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 className="form-section__title">Supplier Auditor</h3>
        <p className="prose prose--muted">
          Held-out test set. Every axis capped at 1.0 for comparison — MCC's true range is −1 to 1, so
          0.85 here means the same as it always does, just plotted on the same scale as the others.
        </p>
        <div className="radar-wrap">
          <RadarChart axes={SUPPLIER_AUDITOR_AXES} size={300} />
        </div>
      </section>
    </div>
  );
}
