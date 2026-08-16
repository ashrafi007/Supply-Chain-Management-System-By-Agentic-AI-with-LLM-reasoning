import { useEffect, useState } from "react";
import { api, ApiError, ExplainResult, QueueEntry, RunDetail, StatsOut } from "../api";
import { GradeDistribution } from "../components/GradeDistribution";
import { SkuResultCard } from "../components/SkuResultCard";
import { StatTile } from "../components/StatTile";
import { RefreshIcon } from "../icons";

const STATUS_COLOR: Record<QueueEntry["status"], string> = {
  pending: "var(--status-neutral)",
  due_today: "var(--status-warn)",
  expired: "var(--status-bad)",
};

interface SkuResult {
  entry: QueueEntry;
  run: RunDetail | null;
  explanation: ExplainResult | null;
  error: string | null;
}

export function Dashboard() {
  const [results, setResults] = useState<SkuResult[] | null>(null);
  const [stats, setStats] = useState<StatsOut | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setLoadError(null);
    try {
      const [entries, statsResult] = await Promise.all([api.listQueue(), api.getStats()]);
      setStats(statsResult);
      const withDetail = await Promise.all(
        entries.map(async (entry): Promise<SkuResult> => {
          if (!entry.last_run_id) {
            return { entry, run: null, explanation: null, error: null };
          }
          try {
            const [run, explanation] = await Promise.all([
              api.getRun(entry.last_run_id),
              api.explainRun(entry.last_run_id),
            ]);
            return { entry, run, explanation, error: null };
          } catch (err) {
            const message = err instanceof ApiError ? err.message : "Failed to load this SKU's result.";
            return { entry, run: null, explanation: null, error: message };
          }
        }),
      );
      setResults(withDetail);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Unknown error loading the dashboard.");
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1>Dashboard</h1>
          <p className="page__subtitle">Order queue, orchestrator results, and AI explanations per SKU.</p>
        </div>
        <button className="btn btn--ghost" onClick={load} disabled={loading} type="button">
          <span className={loading ? "icon-spin" : undefined}>
            <RefreshIcon size={15} />
          </span>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {loadError && (
        <div className="banner banner--error">
          <strong>Couldn't load the dashboard.</strong> {loadError}
        </div>
      )}

      {!loadError && stats && (
        <section className="card">
          <h2 className="card__title">System Overview</h2>
          <div className="stat-row">
            <StatTile label="Total SKUs" value={String(stats.total_skus)} />
            <StatTile label="Pending" value={String(stats.queue_counts_by_status.pending ?? 0)} />
            <StatTile label="Due today" value={String(stats.queue_counts_by_status.due_today ?? 0)} />
            <StatTile label="Expired" value={String(stats.queue_counts_by_status.expired ?? 0)} />
            <StatTile
              label="High-risk SKUs"
              value={String(stats.high_risk_sku_count)}
              sublabel="alarm triggered, ≥0.945"
              tone={stats.high_risk_sku_count > 0 ? undefined : "good"}
            />
          </div>
          <h3 className="form-section__title">Supplier grade distribution (queued SKUs)</h3>
          <GradeDistribution distribution={stats.supplier_grade_distribution} />
        </section>
      )}

      {!loadError && results && (
        <>
          <section className="card">
            <h2 className="card__title">Order Queue</h2>
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>SKU</th>
                    <th>Status</th>
                    <th>Due date</th>
                    <th>Source</th>
                    <th>Last run</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map(({ entry }) => (
                    <tr key={entry.sku_id}>
                      <td className="mono">{entry.sku_id}</td>
                      <td>
                        <span className="pill" style={{ background: STATUS_COLOR[entry.status] }}>
                          {entry.status.replace("_", " ")}
                        </span>
                      </td>
                      <td>{entry.due_date}</td>
                      <td className="muted">{entry.source.replace("_", " ")}</td>
                      <td className="mono muted">{entry.last_run_id ? entry.last_run_id.slice(0, 8) : "—"}</td>
                    </tr>
                  ))}
                  {results.length === 0 && (
                    <tr>
                      <td colSpan={5} className="muted table-empty">
                        No SKUs in the order queue yet. Add one from the "Add Order" tab.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="sku-results">
            {results.map(({ entry, run, explanation, error }) => (
              <SkuResultCard key={entry.sku_id} skuId={entry.sku_id} run={run} explanation={explanation} error={error} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}
