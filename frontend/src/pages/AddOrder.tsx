import { FormEvent, useEffect, useState } from "react";
import { api, ApiError, ExplainResult, RunDetail } from "../api";
import { SkuResultCard } from "../components/SkuResultCard";
import { ChevronDownIcon } from "../icons";

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export function AddOrder() {
  const [skuOptions, setSkuOptions] = useState<string[]>([]);
  const [skuId, setSkuId] = useState("");
  const [dueDate, setDueDate] = useState(today());
  const [source, setSource] = useState<"scheduled" | "manual_add">("manual_add");

  const [phase, setPhase] = useState<"idle" | "enqueueing" | "running">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ run: RunDetail; explanation: ExplainResult | null } | null>(null);

  useEffect(() => {
    api
      .listSkus(500)
      .then((skus) => setSkuOptions(skus.map((s) => s.sku_id)))
      .catch(() => setSkuOptions([]));
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
      setPhase("enqueueing");
      await api.enqueueSku({ sku_id: skuId, due_date: dueDate, source });

      setPhase("running");
      const { run_id, explanation } = await api.runNow(skuId, true);
      const run = await api.getRun(run_id);
      setResult({ run, explanation });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong adding this order.");
    } finally {
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1>Add Order</h1>
          <p className="page__subtitle">
            Queue an existing SKU and run it through the orchestrator immediately.
          </p>
        </div>
      </div>

      <section className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <label className="field">
              <span className="field__label">
                SKU ID<span className="field__required">*</span>
              </span>
              <input
                className="field__input"
                list="sku-suggestions"
                value={skuId}
                onChange={(e) => setSkuId(e.target.value)}
                placeholder="e.g. 1111949"
                required
                disabled={busy}
              />
              <datalist id="sku-suggestions">
                {skuOptions.map((id) => (
                  <option value={id} key={id} />
                ))}
              </datalist>
            </label>

            <label className="field">
              <span className="field__label">
                Due date<span className="field__required">*</span>
              </span>
              <input
                className="field__input"
                type="date"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
                required
                disabled={busy}
              />
            </label>

            <label className="field">
              <span className="field__label">Source</span>
              <div className="select-wrap">
                <select
                  className="field__input"
                  value={source}
                  onChange={(e) => setSource(e.target.value as "scheduled" | "manual_add")}
                  disabled={busy}
                >
                  <option value="manual_add">Manual add</option>
                  <option value="scheduled">Scheduled</option>
                </select>
                <span className="select-wrap__chevron">
                  <ChevronDownIcon size={14} />
                </span>
              </div>
            </label>
          </div>

          <p className="muted field-hint">
            This immediately runs the 5-agent pipeline for this SKU and generates an AI explanation —
            it doesn't wait for the due date, that's only used for future automated sweeps.
          </p>

          <div className="modal__actions modal__actions--left">
            <button className="btn btn--primary" type="submit" disabled={busy}>
              {phase === "enqueueing" && "Adding to queue…"}
              {phase === "running" && "Running orchestrator…"}
              {phase === "idle" && "Add & Run Order"}
            </button>
          </div>
        </form>
      </section>

      {error && (
        <div className="banner banner--error">
          <strong>Couldn't add this order.</strong> {error}
        </div>
      )}

      {result && (
        <SkuResultCard
          skuId={skuId}
          run={result.run}
          explanation={result.explanation}
          error={null}
          titlePrefix="Result — SKU"
        />
      )}
    </div>
  );
}
