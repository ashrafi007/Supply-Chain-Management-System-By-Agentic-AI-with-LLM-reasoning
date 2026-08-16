import { useEffect, useState } from "react";
import { api, ApiError, ColumnMeta, TableDetail, TableInfo } from "../api";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { RowFormModal } from "../components/RowFormModal";
import { UndoAction, UndoToast } from "../components/UndoToast";
import { RefreshIcon } from "../icons";

const PAGE_SIZE = 25;

function formatCell(value: unknown, col: ColumnMeta): string {
  if (value === null || value === undefined) return "—";
  if (col.type === "json") return JSON.stringify(value);
  if (col.type === "datetime" && typeof value === "string") return value.replace("T", " ").slice(0, 19);
  return String(value);
}

export function Database() {
  const [tables, setTables] = useState<TableInfo[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<TableDetail | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [modal, setModal] = useState<{ mode: "create" | "edit"; row: Record<string, unknown> } | null>(null);
  const [saving, setSaving] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [undo, setUndo] = useState<UndoAction | null>(null);

  async function loadTables(keepSelection = true) {
    try {
      const list = await api.listTables();
      setTables(list);
      if (!keepSelection || !selected) {
        const preferred = list.find((t) => t.name === "order_queue") ?? list[0];
        setSelected(preferred?.name ?? null);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load tables.");
    }
  }

  async function loadDetail(table: string, at: number) {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getTable(table, PAGE_SIZE, at);
      setDetail(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load table data.");
      setDetail(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadTables(false);
  }, []);

  useEffect(() => {
    if (selected) {
      setOffset(0);
      setUndo(null); // an undo action references rows in the table you just left
      loadDetail(selected, 0);
    }
  }, [selected]);

  function refresh() {
    loadTables();
    if (selected) loadDetail(selected, offset);
  }

  async function handleSave(values: Record<string, unknown>) {
    if (!selected || !detail || !modal) return;
    setSaving(true);
    try {
      if (modal.mode === "create") {
        const created = await api.createTableRow(selected, values);
        const pkValue = String(created[detail.primary_key]);
        const table = selected;
        setUndo({
          message: `Row added to ${table}.`,
          perform: async () => {
            await api.deleteTableRow(table, pkValue);
            await loadTables();
            if (selected === table) await loadDetail(table, offset);
            setUndo(null);
          },
        });
      } else {
        const pkValue = String(modal.row[detail.primary_key]);
        const previousValues = { ...modal.row }; // snapshot before this edit overwrites it
        const table = selected;
        await api.updateTableRow(selected, pkValue, values);
        setUndo({
          message: `Row ${pkValue} updated in ${table}.`,
          perform: async () => {
            await api.updateTableRow(table, pkValue, previousValues);
            await loadTables();
            if (selected === table) await loadDetail(table, offset);
            setUndo(null);
          },
        });
      }
      setModal(null);
      await loadTables();
      await loadDetail(selected, offset);
    } catch (err) {
      alert(err instanceof ApiError ? err.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!selected || !detail || pendingDelete === null) return;
    const deletedRow = detail.rows.find((r) => String(r[detail.primary_key]) === pendingDelete);
    const table = selected;
    setDeleting(true);
    try {
      await api.deleteTableRow(selected, pendingDelete);
      setPendingDelete(null);
      await loadTables();
      await loadDetail(selected, offset);
      if (deletedRow) {
        setUndo({
          message: `Row ${pendingDelete} deleted from ${table}.`,
          perform: async () => {
            await api.createTableRow(table, deletedRow);
            await loadTables();
            if (selected === table) await loadDetail(table, offset);
            setUndo(null);
          },
        });
      }
    } catch (err) {
      alert(err instanceof ApiError ? err.message : "Delete failed.");
    } finally {
      setDeleting(false);
    }
  }

  const emptyRowTemplate = detail
    ? Object.fromEntries(detail.columns.map((c) => [c.name, null]))
    : {};

  return (
    <div className="page page--wide">
      <div className="page__header">
        <div>
          <h1>Database</h1>
          <p className="page__subtitle">Browse and edit every table directly.</p>
        </div>
        <button className="btn btn--ghost" onClick={refresh} disabled={loading} type="button">
          <span className={loading ? "icon-spin" : undefined}>
            <RefreshIcon size={15} />
          </span>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="banner banner--error">
          <strong>Couldn't load the database view.</strong> {error}
        </div>
      )}

      {tables && (
        <div className="table-tabs">
          {tables.map((t) => (
            <button
              key={t.name}
              type="button"
              className={`table-tab${t.name === selected ? " table-tab--active" : ""}`}
              onClick={() => setSelected(t.name)}
            >
              {t.name}
              <span className="table-tab__count">{t.row_count}</span>
            </button>
          ))}
        </div>
      )}

      {detail && (
        <section className="card">
          <div className="card__header-row">
            <h2 className="card__title">
              {detail.name} <span className="muted">({detail.total} rows)</span>
            </h2>
            {detail.name === "order_queue" ? (
              <span className="muted card__header-hint">Add new orders from the "Add Order" tab →</span>
            ) : (
              <button
                className="btn btn--primary"
                type="button"
                onClick={() => setModal({ mode: "create", row: emptyRowTemplate })}
              >
                + Add row
              </button>
            )}
          </div>

          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  {detail.columns.map((c) => (
                    <th key={c.name}>
                      {c.name}
                      {c.primary_key && <span className="th-tag">PK</span>}
                    </th>
                  ))}
                  <th className="table-actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {detail.rows.map((row) => {
                  const pkValue = String(row[detail.primary_key]);
                  return (
                    <tr key={pkValue}>
                      {detail.columns.map((c) => (
                        <td key={c.name} className={c.type === "json" ? "mono cell-json" : undefined} title={formatCell(row[c.name], c)}>
                          {formatCell(row[c.name], c)}
                        </td>
                      ))}
                      <td className="table-actions-col">
                        <button className="btn btn--icon" type="button" onClick={() => setModal({ mode: "edit", row })}>
                          Edit
                        </button>
                        <button className="btn btn--icon btn--danger-ghost" type="button" onClick={() => setPendingDelete(pkValue)}>
                          Delete
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {detail.rows.length === 0 && (
                  <tr>
                    <td colSpan={detail.columns.length + 1} className="muted table-empty">
                      No rows in this table yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <button
              className="btn"
              type="button"
              disabled={offset === 0}
              onClick={() => {
                const next = Math.max(0, offset - PAGE_SIZE);
                setOffset(next);
                loadDetail(detail.name, next);
              }}
            >
              ← Prev
            </button>
            <span className="muted">
              {detail.total === 0 ? "0" : `${offset + 1}–${Math.min(offset + PAGE_SIZE, detail.total)}`} of {detail.total}
            </span>
            <button
              className="btn"
              type="button"
              disabled={offset + PAGE_SIZE >= detail.total}
              onClick={() => {
                const next = offset + PAGE_SIZE;
                setOffset(next);
                loadDetail(detail.name, next);
              }}
            >
              Next →
            </button>
          </div>
        </section>
      )}

      {modal && detail && (
        <RowFormModal
          tableName={detail.name}
          columns={detail.columns}
          mode={modal.mode}
          initialValues={modal.row}
          busy={saving}
          onSave={handleSave}
          onCancel={() => setModal(null)}
        />
      )}

      {pendingDelete !== null && (
        <ConfirmDialog
          title="Delete this row?"
          message={`This deletes ${detail?.name} row "${pendingDelete}". You'll get a brief Undo option right after, but it won't last if you navigate away or make another change.`}
          busy={deleting}
          onConfirm={handleDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}

      {undo && <UndoToast action={undo} onDismiss={() => setUndo(null)} />}
    </div>
  );
}
