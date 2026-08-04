"""
DB session fixture for repository-layer tests.

Binds a Session to a connection-level transaction + SAVEPOINT against the real,
seeded data/app.db. Code under test (create_pending_run/persist_success/persist_failure)
calls session.commit() internally; the after_transaction_end listener restarts the
SAVEPOINT each time one ends, so those commits only ever close the SAVEPOINT, never
the outer transaction. The outer transaction is rolled back at teardown, so nothing
written during a test survives it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker

from src.db.base import engine
from src.db.models import Sku

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
    return db_session.execute(select(Sku.sku_id).limit(1)).scalar_one()


@pytest.fixture()
def orphan_sku_id(db_session) -> str:
    """A SKU that exists in `skus` (satisfies pipeline_runs.sku_id's NOT NULL FK)
    but has no `inventory_current` row — the realistic "not found" case, since
    state_builder only ever queries inventory_current, never `skus` directly."""
    sku_id = "TEST-ORPHAN-SKU-NO-INVENTORY"
    db_session.add(Sku(sku_id=sku_id, supplier_id=None, description="test orphan"))
    db_session.commit()
    return sku_id
