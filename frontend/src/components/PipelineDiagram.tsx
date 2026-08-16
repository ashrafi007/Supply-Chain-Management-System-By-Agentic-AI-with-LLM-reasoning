interface Stage {
  plain: string;
  technical: string;
  accent?: string;
}

const STAGES: Stage[] = [
  { plain: "New order comes in", technical: "SKU added to order_queue" },
  { plain: "How much will it sell?", technical: "Demand Predictor (LightGBM)" },
  { plain: "Will it run out?", technical: "Risk Detector (Stacking Ensemble)" },
  { plain: "How urgent to restock?", technical: "Inventory Rebalancer (XGBoost)" },
  { plain: "Is the human forecast right?", technical: "Forecast Optimizer (LightGBM+XGBoost)" },
  { plain: "Can we trust the supplier?", technical: "Supplier Auditor (XGBoost)" },
  { plain: "Explain it in plain English", technical: "LLM Explanation Layer (grounded + polished)" },
  { plain: "You see the result", technical: "Dashboard" },
];

export function PipelineDiagram() {
  return (
    <div className="pipeline-diagram">
      {STAGES.map((stage, i) => (
        <div className="pipeline-diagram__item" key={i}>
          <div className="pipeline-diagram__stage">
            <span className="pipeline-diagram__number">{i + 1}</span>
            <div className="pipeline-diagram__plain">{stage.plain}</div>
            <div className="pipeline-diagram__technical">{stage.technical}</div>
          </div>
          {i < STAGES.length - 1 && (
            <svg className="pipeline-diagram__arrow" width="20" height="14" viewBox="0 0 20 14" aria-hidden="true">
              <path
                d="M0 7h16M11 2l5 5-5 5"
                strokeWidth="1.8"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ stroke: "var(--text-muted)" }}
              />
            </svg>
          )}
        </div>
      ))}
    </div>
  );
}
