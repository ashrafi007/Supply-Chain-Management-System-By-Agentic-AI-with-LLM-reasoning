// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

/// Holds the backend child process (if this instance started one) so it can be
/// killed on app exit. `None` if a backend was already running on :8000 when we
/// started (someone else's process -- not ours to kill) or if spawning failed.
struct BackendProcess(Mutex<Option<Child>>);

fn backend_already_running() -> bool {
    let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// Dev-only: launches `venv/bin/python -m uvicorn src.api.main:app --port 8000`
/// from the project root, using the project's own virtualenv. CARGO_MANIFEST_DIR
/// is baked in at compile time as `frontend/src-tauri`, so the project root is
/// two levels up -- this only works running from source (this repo's own venv +
/// Python package present alongside the compiled binary). It will NOT work in a
/// distributed/bundled build (`tauri build`) copied to another machine -- there's
/// no packaging step for this app yet, so that's not a regression, just a known
/// boundary of what "auto-start" covers today.
fn spawn_backend() -> Option<Child> {
    if backend_already_running() {
        println!("[backend] already running on :8000 -- not starting a second one.");
        return None;
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = Path::new(manifest_dir)
        .parent() // frontend/
        .and_then(Path::parent) // repo root
        .expect("frontend/src-tauri should have two parent directories");
    let python = project_root.join("venv/bin/python");

    if !python.exists() {
        eprintln!(
            "[backend] {} not found -- venv missing, not starting backend automatically.",
            python.display()
        );
        return None;
    }

    match Command::new(&python)
        .args(["-m", "uvicorn", "src.api.main:app", "--port", "8000"])
        .current_dir(project_root)
        .spawn()
    {
        Ok(child) => {
            println!("[backend] started (pid {}).", child.id());
            Some(child)
        }
        Err(e) => {
            eprintln!("[backend] failed to start: {e}");
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .setup(|app| {
            let child = spawn_backend();
            app.manage(BackendProcess(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<BackendProcess>() {
                    if let Some(mut child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                        println!("[backend] stopped.");
                    }
                }
            }
        });
}
