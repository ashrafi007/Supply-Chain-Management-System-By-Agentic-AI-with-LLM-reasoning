import { useEffect, useState } from "react";

const AUTO_DISMISS_MS = 8000;

export interface UndoAction {
  message: string;
  perform: () => Promise<void>;
}

export function UndoToast({ action, onDismiss }: { action: UndoAction; onDismiss: () => void }) {
  const [undoing, setUndoing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(onDismiss, AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [action, onDismiss]);

  async function handleUndo() {
    setUndoing(true);
    setError(null);
    try {
      await action.perform();
    } catch {
      setError("Undo failed — the row may have changed since. Check the table.");
      setUndoing(false);
    }
  }

  return (
    <div className="undo-toast">
      <span className="undo-toast__message">{error ?? action.message}</span>
      {!error && (
        <button className="undo-toast__btn" type="button" onClick={handleUndo} disabled={undoing}>
          {undoing ? "Undoing…" : "Undo"}
        </button>
      )}
      <button className="undo-toast__close" type="button" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
