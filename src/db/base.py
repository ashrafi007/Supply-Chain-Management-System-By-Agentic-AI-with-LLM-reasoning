"""
Engine, session factory, and declarative Base for the runtime SQLite database.

Design (see .CLAUDE/database_spec.md §1, §2/step-2):
- SQLite, one file at ``data/app.db`` resolved from the *project root* (never the CWD),
  so ``python -m src.db.create_db`` works from anywhere.
- A ``connect`` event listener enables ``PRAGMA foreign_keys=ON`` on *every* connection.
  SQLite does not enforce foreign keys by default and the setting is per-connection —
  omitting this would silently disable every FK in the schema.
- ``PRAGMA journal_mode=WAL`` for better concurrent-read behaviour in the desktop app.

``APP_DB_PATH`` env var overrides the DB file path -- unset in normal use (app,
scripts, uvicorn all get ``data/app.db``), set only by ``tests/conftest.py`` to
point the entire test suite at an isolated copy (``data/test.db``) instead, so
running the app for real never corrupts a test's "this table starts empty"
assumptions and vice versa.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# Project root = two levels up from this file: src/db/base.py -> src/db -> src -> <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"

_db_path_override = os.environ.get("APP_DB_PATH")
if _db_path_override:
    _override_path = Path(_db_path_override)
    DB_PATH: Path = _override_path if _override_path.is_absolute() else PROJECT_ROOT / _override_path
else:
    DB_PATH = DATA_DIR / "app.db"

# Create data/ if missing so the engine can open the file.
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite URL. future=True is the default in SQLAlchemy 2.0 but stated for clarity.
DATABASE_URL: str = f"sqlite:///{DB_PATH}"

engine: Engine = create_engine(DATABASE_URL, echo=False, future=True)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ANN001
    """Enforce FKs and enable WAL on every new connection (both are per-connection)."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base shared by every model in :mod:`src.db.models`."""
