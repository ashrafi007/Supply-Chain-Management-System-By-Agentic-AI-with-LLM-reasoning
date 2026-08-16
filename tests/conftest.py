"""
DB session fixture for repository-layer tests.

Binds a Session to a connection-level transaction + SAVEPOINT against an isolated
test database (data/test.db -- a checkpointed copy of the real, seeded data/app.db,
refreshed automatically whenever app.db is newer). Code under test
(create_pending_run/persist_success/persist_failure) calls session.commit()
internally; the after_transaction_end listener restarts the SAVEPOINT each time one
ends, so those commits only ever close the SAVEPOINT, never the outer transaction.
The outer transaction is rolled back at teardown, so nothing written during a test
survives it -- and now nothing the *real app* does to data/app.db (adding real
order_queue rows via the frontend, running a real sweep, ...) can touch what tests
see either, since they're different files entirely.

The env var + one-time copy below MUST run before anything imports src.db.base --
that module reads APP_DB_PATH once, at import time, to build its engine. This file
is pytest's root conftest.py, so it's guaranteed to load before any test module or
subdirectory conftest.py gets a chance to import src.db.base first.

Deliberately NOT auto-refreshed from app.db on every run (e.g. "copy whenever
app.db is newer"): app.db's mtime changes constantly from ordinary use (the
frontend adding an order, a real sweep running, ...), and re-syncing on that
signal would pull today's live operational data straight into the "supposed to
be isolated" test DB the next time pytest runs -- silently reintroducing the
exact problem this file exists to prevent. test.db is created once, from
whatever app.db's baseline seed data looks like at that moment, and after that
it only changes if you delete it yourself (e.g. after a real schema migration
you want reflected in tests) -- an explicit, deliberate step, matching how
migrations are already handled elsewhere in this project (queue_migration_spec.md).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DB = _PROJECT_ROOT / "data" / "app.db"
_TEST_DB = _PROJECT_ROOT / "data" / "test.db"

if not _TEST_DB.exists() and _SOURCE_DB.exists():
    # Flush WAL into the main file first -- app.db runs in WAL mode (src/db/base.py),
    # so recent writes can still be sitting in app.db-wal, not yet in app.db itself.
    # Copying app.db alone without this could silently copy stale/incomplete data.
    _conn = sqlite3.connect(str(_SOURCE_DB))
    _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _conn.close()
    shutil.copyfile(_SOURCE_DB, _TEST_DB)

os.environ.setdefault("APP_DB_PATH", str(_TEST_DB))

import pytest  # noqa: E402
from sqlalchemy import event, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src.db.base import engine  # noqa: E402
from src.db.models import Sku  # noqa: E402
from src.queue.models import OrderQueue  # noqa: E402

# Not present in `skus` at all — safe for read-only lookups (state_builder does a
# plain SELECT against inventory_current, so no FK constraint is involved).
UNKNOWN_SKU_ID = "SKU-DOES-NOT-EXIST-999999"


@pytest.fixture()
def db_session():
    connection = engine.connect()
    trans = connection.begin()
    connection.begin_nested()

    TestingSessionLocal = sessionmaker(
        bind=connection, autoflush=False, expire_on_commit=False, future=True
    )
    session = TestingSessionLocal()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, transaction):
        if not connection.in_nested_transaction():
            connection.begin_nested()

    yield session

    session.close()
    trans.rollback()
    connection.close()


@pytest.fixture()
def known_sku_id(db_session) -> str:
    """A real sku_id, guaranteed NOT already sitting in order_queue -- the app now
    persists real demo/seed rows there (e.g. for the frontend dashboard) outside of
    any test transaction, so a plain "first sku in `skus`" pick can collide with an
    already-committed order_queue row the moment a queue test tries to enqueue it
    (UNIQUE constraint on order_queue.sku_id). Excluding queued SKUs here keeps every
    other fixture/test that assumes "known_sku_id starts with a clean queue slate"
    true regardless of what's been seeded into the live DB for non-test purposes."""
    stmt = (
        select(Sku.sku_id)
        .outerjoin(OrderQueue, OrderQueue.sku_id == Sku.sku_id)
        .where(OrderQueue.sku_id.is_(None))
        .limit(1)
    )
    return db_session.execute(stmt).scalar_one()


@pytest.fixture()
def orphan_sku_id(db_session) -> str:
    """A SKU that exists in `skus` (satisfies pipeline_runs.sku_id's NOT NULL FK)
    but has no `inventory_current` row — the realistic "not found" case, since
    state_builder only ever queries inventory_current, never `skus` directly."""
    sku_id = "TEST-ORPHAN-SKU-NO-INVENTORY"
    db_session.add(Sku(sku_id=sku_id, supplier_id=None, description="test orphan"))
    db_session.commit()
    return sku_id
