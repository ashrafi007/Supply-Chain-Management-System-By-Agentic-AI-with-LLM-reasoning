import { FormEvent, useState } from "react";
import { ColumnMeta } from "../api";

// All form state is kept as strings (what inputs naturally produce) and only
// converted to the real JSON type at submit time -- keeps every input a plain
// controlled <input>/<textarea> regardless of the underlying column type.
function toFormValue(value: unknown, type: ColumnMeta["type"]): string {
  if (value === null || value === undefined) return "";
  if (type === "datetime" && typeof value === "string") return value.slice(0, 16);
  if (type === "json") return JSON.stringify(value, null, 2);
  return String(value);
}

function fromFormValue(raw: string, col: ColumnMeta): unknown {
  if (raw.trim() === "") return null;
  switch (col.type) {
    case "integer":
      return Math.trunc(Number(raw));
    case "float":
      return Number(raw);
    case "json":
      return JSON.parse(raw); // caller catches SyntaxError
    default:
      return raw;
  }
}

export function RowFormModal({
  tableName,
  columns,
  mode,
  initialValues,
  busy,
  onSave,
  onCancel,
}: {
  tableName: string;
  columns: ColumnMeta[];
  mode: "create" | "edit";
  initialValues: Record<string, unknown>;
  busy: boolean;
  onSave: (values: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const [form, setForm] = useState<Record<string, string>>(() =>
    Object.fromEntries(columns.map((c) => [c.name, toFormValue(initialValues[c.name], c.type)])),
  );
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const values: Record<string, unknown> = {};
    for (const col of columns) {
      if (mode === "edit" && col.primary_key) continue; // PK immutable via PUT
      const raw = form[col.name] ?? "";
      // Create mode: a blank field is OMITTED entirely, not sent as null -- several
      // columns (order_queue.status, .queued_at, ...) have server-side defaults that
      // only apply when the key is absent from the insert; sending an explicit null
      // overrides the default and trips their NOT NULL constraint instead. Edit mode
      // keeps sending null for a blank field, since clearing a field IS the intent there.
      if (mode === "create" && raw.trim() === "" && !col.primary_key) continue;
      try {
        values[col.name] = fromFormValue(raw, col);
      } catch {
        setError(`"${col.name}" isn't valid JSON.`);
        return;
      }
    }
    onSave(values);
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">
          {mode === "create" ? `Add row — ${tableName}` : `Edit row — ${tableName}`}
        </h3>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {columns.map((col) => {
              const disabled = mode === "edit" && col.primary_key;
              const required = !col.nullable && !disabled;
              return (
                <label className="field" key={col.name}>
                  <span className="field__label">
                    {col.name}
                    {required && <span className="field__required">*</span>}
                    {col.primary_key && <span className="field__tag">PK</span>}
                    {col.foreign_key && <span className="field__tag field__tag--fk">FK → {col.foreign_key}</span>}
                  </span>
                  {col.type === "json" ? (
                    <textarea
                      className="field__input field__input--textarea"
                      value={form[col.name] ?? ""}
                      disabled={disabled}
                      onChange={(e) => setForm({ ...form, [col.name]: e.target.value })}
                      rows={4}
                    />
                  ) : (
                    <input
                      className="field__input"
                      type={
                        col.type === "integer" || col.type === "float"
                          ? "number"
                          : col.type === "date"
                            ? "date"
                            : col.type === "datetime"
                              ? "datetime-local"
                              : "text"
                      }
                      step={col.type === "float" ? "any" : undefined}
                      value={form[col.name] ?? ""}
                      disabled={disabled}
                      required={required}
                      onChange={(e) => setForm({ ...form, [col.name]: e.target.value })}
                    />
                  )}
                </label>
              );
            })}
          </div>

          {error && <div className="banner banner--error banner--tight">{error}</div>}

          <div className="modal__actions">
            <button className="btn" type="button" onClick={onCancel} disabled={busy}>
              Cancel
            </button>
            <button className="btn btn--primary" type="submit" disabled={busy}>
              {busy ? "Saving…" : mode === "create" ? "Create" : "Save changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
