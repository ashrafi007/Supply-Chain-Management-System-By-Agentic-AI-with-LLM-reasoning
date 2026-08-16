import { APP_NAME } from "../constants";

export function TopBar() {
  return (
    <header className="top-bar" data-tauri-drag-region>
      <div className="top-bar__title" data-tauri-drag-region>
        <span className="top-bar__live-dot" aria-hidden="true" />
        {APP_NAME}
      </div>
    </header>
  );
}
