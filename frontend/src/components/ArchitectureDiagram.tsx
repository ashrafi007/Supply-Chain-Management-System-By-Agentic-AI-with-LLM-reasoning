interface Layer {
  title: string;
  detail: string;
  tech: string;
}

const LAYERS: Layer[] = [
  { title: "Desktop App", detail: "What you're looking at right now", tech: "Tauri (Rust) + React + TypeScript" },
  { title: "API Layer", detail: "Every screen talks to this over HTTP", tech: "FastAPI (Python), REST endpoints" },
  { title: "Orchestrator", detail: "Runs the 5 ML agents in a fixed sequence, applies business-rule overrides", tech: "LangGraph state machine" },
  { title: "Data + AI", detail: "Every run's inputs/outputs stored; explanations generated on demand", tech: "SQLite (10 tables) + OpenRouter LLM" },
];

export function ArchitectureDiagram() {
  return (
    <div className="arch-diagram">
      {LAYERS.map((layer, i) => (
        <div className="arch-diagram__item" key={i}>
          <div className="arch-diagram__layer">
            <div className="arch-diagram__title">{layer.title}</div>
            <div className="arch-diagram__detail">{layer.detail}</div>
            <div className="arch-diagram__tech">{layer.tech}</div>
          </div>
          {i < LAYERS.length - 1 && (
            <svg className="arch-diagram__arrow" width="14" height="22" viewBox="0 0 14 22" aria-hidden="true">
              <path
                d="M7 0v16M2 11l5 5 5-5"
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
