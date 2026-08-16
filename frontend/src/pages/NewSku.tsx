import { FormEvent, useEffect, useState } from "react";
import { api, ApiError, ExplainResult, RunDetail, Supplier } from "../api";
import { SkuResultCard } from "../components/SkuResultCard";
import { ChevronDownIcon } from "../icons";

const NUMERIC_FIELDS: { key: string; label: string; step?: string }[] = [
  { key: "national_inv", label: "Current inventory (units)" },
  { key: "in_transit_qty", label: "In-transit quantity" },
  { key: "min_bank", label: "Safety stock minimum" },
  { key: "lead_time", label: "Supplier lead time (days)" },
  { key: "sales_1_month", label: "Sales — last 1 month" },
  { key: "sales_3_month", label: "Sales — last 3 months" },
  { key: "sales_6_month", label: "Sales — last 6 months" },
  { key: "sales_9_month", label: "Sales — last 9 months" },
  { key: "forecast_3_month", label: "Forecast — next 3 months" },
  { key: "forecast_6_month", label: "Forecast — next 6 months" },
  { key: "forecast_9_month", label: "Forecast — next 9 months" },
  { key: "perf_6_month_avg", label: "Supplier performance, 6mo avg (0–1)", step: "0.01" },
  { key: "perf_12_month_avg", label: "Supplier performance, 12mo avg (0–1)", step: "0.01" },
  { key: "pieces_past_due", label: "Pieces past due" },
  { key: "local_bo_qty", label: "Local backorder quantity" },
];

const FLAG_FIELDS: { key: string; label: string }[] = [
  { key: "potential_issue", label: "Potential issue flagged" },
  { key: "deck_risk", label: "Deck risk" },
  { key: "oe_constraint", label: "OE constraint" },
  { key: "ppap_risk", label: "PPAP risk" },
  { key: "stop_auto_buy", label: "Stop auto-buy" },
  { key: "rev_stop", label: "Revenue stop" },
];

const emptyNumbers = () => Object.fromEntries(NUMERIC_FIELDS.map((f) => [f.key, ""]));
const emptyFlags = () => Object.fromEntries(FLAG_FIELDS.map((f) => [f.key, false]));

export function NewSku() {
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);

  const [skuId, setSkuId] = useState("");
  const [supplierId, setSupplierId] = useState("");
  const [description, setDescription] = useState("");
  const [numbers, setNumbers] = useState<Record<string, string>>(emptyNumbers);
  const [flags, setFlags] = useState<Record<string, boolean>>(emptyFlags);
  const [dueInDays, setDueInDays] = useState(7);
  const [runNow, setRunNow] = useState(true);

  const [phase, setPhase] = useState<"idle" | "saving" | "running">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ run: RunDetail | null; explanation: ExplainResult | null } | null>(null);

  useEffect(() => {
    api.listSuppliers().then(setSuppliers).catch(() => setSuppliers([]));
  }, []);

  function resetForm() {
    setSkuId("");
    setSupplierId("");
    setDescription("");
    setNumbers(emptyNumbers());
    setFlags(emptyFlags());
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const raw_features: Record<string, unknown> = {};
    for (const f of NUMERIC_FIELDS) {
      const v = numbers[f.key].trim();
      raw_features[f.key] = v === "" ? null : Number(v);
    }
    for (const f of FLAG_FIELDS) {
      raw_features[f.key] = flags[f.key] ? 1 : 0;
    }

    try {
      setPhase("saving");
      const response = await api.createSku({
        sku_id: skuId,
        raw_features,
        supplier_id: supplierId || null,
        description: description || null,
        due_in_days: dueInDays,
        run_now: runNow,
      });

      if (runNow && response.run_id) {
        setPhase("running");
        const run = await api.getRun(response.run_id);
        setResult({ run, explanation: response.explanation });
      } else {
        setResult({ run: null, explanation: null });
      }
      resetForm();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong creating this SKU.");
    } finally {
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1>New SKU</h1>
          <p className="page__subtitle">Onboard a brand-new product into the system.</p>
        </div>
      </div>

      <section className="card">
        <form onSubmit={handleSubmit}>
          <h3 className="form-section__title">Identity</h3>
          <div className="form-grid">
            <label className="field">
              <span className="field__label">
                SKU ID<span className="field__required">*</span>
              </span>
              <input
                className="field__input"
                value={skuId}
                onChange={(e) => setSkuId(e.target.value)}
                placeholder="e.g. SKU-DEMO-001"
                required
                disabled={busy}
              />
            </label>
            <label className="field">
              <span className="field__label">Supplier</span>
              <div className="select-wrap">
                <select className="field__input" value={supplierId} onChange={(e) => setSupplierId(e.target.value)} disabled={busy}>
                  <option value="">— none / unknown —</option>
                  {suppliers.map((s) => (
                    <option value={s.supplier_id} key={s.supplier_id}>
                      {s.name} ({s.supplier_id})
                    </option>
                  ))}
                </select>
                <span className="select-wrap__chevron">
                  <ChevronDownIcon size={14} />
                </span>
              </div>
            </label>
            <label className="field">
              <span className="field__label">Description</span>
              <input
                className="field__input"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="optional"
                disabled={busy}
              />
            </label>
          </div>

          <h3 className="form-section__title">Inventory, sales &amp; forecast</h3>
          <p className="muted field-hint">
            Leave anything you don't know blank — it's stored as unknown, not zero.
          </p>
          <div className="form-grid">
            {NUMERIC_FIELDS.map((f) => (
              <label className="field" key={f.key}>
                <span className="field__label">{f.label}</span>
                <input
                  className="field__input"
                  type="number"
                  step={f.step ?? "any"}
                  value={numbers[f.key]}
                  onChange={(e) => setNumbers({ ...numbers, [f.key]: e.target.value })}
                  disabled={busy}
                />
              </label>
            ))}
          </div>

          <h3 className="form-section__title">Risk flags</h3>
          <div className="flag-grid">
            {FLAG_FIELDS.map((f) => (
              <label className="flag-check" key={f.key}>
                <input
                  type="checkbox"
                  checked={flags[f.key]}
                  onChange={(e) => setFlags({ ...flags, [f.key]: e.target.checked })}
                  disabled={busy}
                />
                {f.label}
              </label>
            ))}
          </div>

          <h3 className="form-section__title">Queue options</h3>
          <div className="form-grid">
            <label className="field">
              <span className="field__label">Due in (days)</span>
              <input
                className="field__input"
                type="number"
                min={0}
                value={dueInDays}
                onChange={(e) => setDueInDays(Number(e.target.value))}
                disabled={busy}
              />
            </label>
            <label className="flag-check flag-check--standalone">
              <input type="checkbox" checked={runNow} onChange={(e) => setRunNow(e.target.checked)} disabled={busy} />
              Run through the orchestrator immediately
            </label>
          </div>

          <div className="modal__actions modal__actions--left">
            <button className="btn btn--primary" type="submit" disabled={busy}>
              {phase === "saving" && "Creating SKU…"}
              {phase === "running" && "Running orchestrator…"}
              {phase === "idle" && "Create SKU"}
            </button>
          </div>
        </form>
      </section>

      {error && (
        <div className="banner banner--error">
          <strong>Couldn't create this SKU.</strong> {error}
        </div>
      )}

      {result && !result.run && (
        <div className="banner banner--success">SKU created and added to the order queue.</div>
      )}

      {result && result.run && (
        <SkuResultCard
          skuId={result.run.sku_id}
          run={result.run}
          explanation={result.explanation}
          error={null}
          titlePrefix="Result — SKU"
        />
      )}
    </div>
  );
}
