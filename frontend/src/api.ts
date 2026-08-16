// Typed client for the FastAPI backend (src/api/ in the Python project).
// The backend must be running separately: `uvicorn src.api.main:app` from the repo root.
// CORS is already configured on that side (CORS_ORIGINS, defaults to "*") specifically
// so this webview can call it directly with plain fetch() -- no Tauri IPC involved.

export const API_BASE = "http://127.0.0.1:8000";

// ---- Types (mirror src/api/schemas.py) --------------------------------------------

export interface QueueEntry {
  sku_id: string;
  status: "pending" | "due_today" | "expired";
  queued_at: string;
  due_date: string;
  last_run_id: string | null;
  last_evaluated_at: string | null;
  source: "scheduled" | "manual_add";
}

export interface Prediction {
  demand_forecast: number | null;
  demand_velocity_band: string | null;
  stockout_risk: number | null;
  backorder_prob: number | null;
  alarm_triggered: number | null;
  urgency_score: number | null;
  correction_factor: number | null;
  supplier_risk: string | null;
  recommendation: string | null;
}

export interface AgentTrace {
  agent_name: string;
  sequence: number;
  status: string;
  latency_ms: number | null;
  output: Record<string, unknown> | null;
  note: string | null;
}

export interface RunDetail {
  run_id: string;
  sku_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  latency_ms: number | null;
  manifest_version: string;
  error: string | null;
  prediction: Prediction | null;
  traces: AgentTrace[];
}

export interface ExplainResult {
  explanation: string;
  cached: boolean;
  was_polished: boolean;
  fallback_reason: string | null;
  model_used: string;
}

export interface StatsOut {
  total_skus: number;
  queue_counts_by_status: Record<string, number>;
  supplier_grade_distribution: Record<string, number>;
  high_risk_sku_count: number;
}

export interface Supplier {
  supplier_id: string;
  name: string;
  country: string | null;
  lead_time_avg_days: number | null;
  created_at: string;
}

export interface SkuOut {
  sku_id: string;
  supplier_id: string | null;
  description: string | null;
  created_at: string;
}

// ---- Generic DB browser/CRUD types (mirror src/api/routers/db.py) -----------------

export type ColumnType = "text" | "integer" | "float" | "datetime" | "date" | "json" | "boolean";

export interface ColumnMeta {
  name: string;
  type: ColumnType;
  nullable: boolean;
  primary_key: boolean;
  foreign_key: string | null;
}

export interface TableInfo {
  name: string;
  row_count: number;
  primary_key: string | null;
}

export interface TableDetail {
  name: string;
  primary_key: string;
  columns: ColumnMeta[];
  total: number;
  rows: Record<string, unknown>[];
}

// ---- Fetch helpers ------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(
      `Can't reach the backend at ${API_BASE}. Is \`uvicorn src.api.main:app\` running?`,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON -- keep statusText
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listQueue: (status?: QueueEntry["status"]) =>
    request<QueueEntry[]>(`/queue${status ? `?status=${status}` : ""}`),

  enqueueSku: (body: { sku_id: string; due_date: string; source: "scheduled" | "manual_add" }) =>
    request<QueueEntry>("/queue", { method: "POST", body: JSON.stringify(body) }),

  runNow: (skuId: string, explain = true) =>
    request<{ run_id: string; status: string; explanation: ExplainResult | null }>(
      `/queue/${skuId}/run?explain=${explain}`,
      { method: "POST" },
    ),

  triggerSweep: (explain = true) =>
    request<{
      as_of: string;
      expired_count: number;
      evaluated_sku_ids: string[];
      run_ids: string[];
      explanations: Record<string, ExplainResult>;
    }>(`/queue/sweep?explain=${explain}`, { method: "POST" }),

  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),

  explainRun: (runId: string) =>
    request<ExplainResult>(`/predictions/${runId}/explain`, { method: "POST" }),

  getStats: () => request<StatsOut>("/stats"),

  listSuppliers: () => request<Supplier[]>("/suppliers"),

  listSkus: (limit = 50) => request<SkuOut[]>(`/skus?limit=${limit}`),

  createSku: (body: {
    sku_id: string;
    raw_features: Record<string, unknown>;
    supplier_id?: string | null;
    description?: string | null;
    due_in_days?: number;
    run_now?: boolean;
  }) =>
    request<{ sku_id: string; queued: QueueEntry; run_id: string | null; explanation: ExplainResult | null }>(
      "/skus",
      { method: "POST", body: JSON.stringify(body) },
    ),

  // ---- Generic DB browser/CRUD ----------------------------------------------------

  listTables: () => request<TableInfo[]>("/db/tables"),

  getTable: (table: string, limit = 50, offset = 0) =>
    request<TableDetail>(`/db/tables/${table}?limit=${limit}&offset=${offset}`),

  createTableRow: (table: string, values: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/db/tables/${table}`, {
      method: "POST",
      body: JSON.stringify(values),
    }),

  updateTableRow: (table: string, pk: string, values: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/db/tables/${table}/${encodeURIComponent(pk)}`, {
      method: "PUT",
      body: JSON.stringify(values),
    }),

  deleteTableRow: async (table: string, pk: string) => {
    const response = await fetch(`${API_BASE}/db/tables/${table}/${encodeURIComponent(pk)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new ApiError(detail, response.status);
    }
  },
};
