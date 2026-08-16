"""Generic table browser + CRUD over every table on the shared Base.metadata --
one reflection-based router instead of ten hand-written ones.

    GET    /db/tables                        -- table names, row counts, primary key
    GET    /db/tables/{table}                 -- column metadata + paginated rows
    POST   /db/tables/{table}                 -- insert a row
    PUT    /db/tables/{table}/{pk_value}      -- update a row
    DELETE /db/tables/{table}/{pk_value}      -- delete a row

Safety notes:
- Table/column identifiers are NEVER taken as raw user strings for SQL -- every
  operation resolves through `Base.metadata.tables[name]`, a whitelist of the
  actual mapped tables. If the name isn't in that dict, it's a 404, not a query.
- All values are bound parameters via SQLAlchemy Core (table.insert()/.update()/
  .delete()), so this is not vulnerable to SQL injection the way raw string-built
  SQL would be.
- NOT NULL / CHECK / FK / UNIQUE violations surface as 400 (bad input) or 409
  (delete blocked by a foreign key elsewhere) with the DB's own message, not a
  500 -- the constraints Postgres/SQLite already enforce are the real safety net
  here, this layer doesn't try to reimplement them.
- Several tables here are pipeline OUTPUT, not authored data (`pipeline_runs`,
  `predictions`, `agent_traces`, `llm_explanations`, `forecast_actuals`) --
  CRUD is exposed uniformly because that's what was asked for, but editing rows
  in those tables by hand means the row no longer reflects what the orchestrator
  / LLM actually produced. Nothing technical stops it; worth knowing before doing it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Table, delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import src.db.models  # noqa: F401 -- registers suppliers/skus/.../inventory_current on Base.metadata
import src.llm.models  # noqa: F401 -- registers llm_explanations
import src.queue.models  # noqa: F401 -- registers order_queue/order_queue_log
from src.api.deps import get_db
from src.db.base import Base

router = APIRouter(prefix="/db")

_TYPE_LABELS = {
    "VARCHAR": "text", "TEXT": "text", "CHAR": "text",
    "INTEGER": "integer", "BIGINT": "integer", "SMALLINT": "integer",
    "FLOAT": "float", "REAL": "float", "NUMERIC": "float",
    "DATETIME": "datetime", "TIMESTAMP": "datetime",
    "DATE": "date",
    "JSON": "json",
    "BOOLEAN": "boolean",
}


def _type_label(col) -> str:
    return _TYPE_LABELS.get(type(col.type).__name__.upper(), "text")


def _get_table(table_name: str) -> Table:
    table = Base.metadata.tables.get(table_name)
    if table is None:
        raise HTTPException(status_code=404, detail=f"no such table {table_name!r}")
    return table


def _pk_column(table: Table):
    pk_cols = list(table.primary_key.columns)
    if len(pk_cols) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"table {table.name!r} has a composite or missing primary key -- not supported by this endpoint",
        )
    return pk_cols[0]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _serialize_row(row) -> dict:
    return {key: _serialize_value(value) for key, value in row._mapping.items()}


def _coerce_value(col, value: Any) -> Any:
    """Incoming JSON -> the Python type SQLAlchemy expects for this column. JSON
    doesn't distinguish date/datetime from string, so ISO strings are parsed
    explicitly for those two types; everything else passes through as-is (Core
    binds it as a parameter, the DB driver / column type handles the rest)."""
    if value is None:
        return None
    label = _type_label(col)
    if label == "datetime" and isinstance(value, str):
        return datetime.fromisoformat(value)
    if label == "date" and isinstance(value, str):
        return date.fromisoformat(value)
    return value


def _integrity_error_detail(exc: IntegrityError) -> str:
    # SQLite/Postgres both put the useful part in orig's string form; str(exc) also
    # works but includes the full failed SQL statement, noisier for a UI to show.
    return str(getattr(exc, "orig", exc))


@router.get("/tables")
def list_tables(session: Session = Depends(get_db)) -> list[dict]:
    out = []
    for name, table in sorted(Base.metadata.tables.items()):
        pk_cols = list(table.primary_key.columns)
        count = session.execute(select(func.count()).select_from(table)).scalar_one()
        out.append({
            "name": name,
            "row_count": count,
            "primary_key": pk_cols[0].name if len(pk_cols) == 1 else None,
        })
    return out


@router.get("/tables/{table_name}")
def get_table(
    table_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> dict:
    table = _get_table(table_name)
    pk_col = _pk_column(table)

    columns = [
        {
            "name": col.name,
            "type": _type_label(col),
            "nullable": col.nullable,
            "primary_key": col.primary_key,
            "foreign_key": next(iter(col.foreign_keys)).target_fullname if col.foreign_keys else None,
        }
        for col in table.columns
    ]

    total = session.execute(select(func.count()).select_from(table)).scalar_one()
    rows = session.execute(
        select(table).order_by(pk_col).limit(limit).offset(offset)
    ).all()

    return {
        "name": table_name,
        "primary_key": pk_col.name,
        "columns": columns,
        "total": total,
        "rows": [_serialize_row(row) for row in rows],
    }


@router.post("/tables/{table_name}", status_code=201)
def create_row(table_name: str, body: dict, session: Session = Depends(get_db)) -> dict:
    table = _get_table(table_name)
    pk_col = _pk_column(table)

    values = {
        col.name: _coerce_value(col, body[col.name])
        for col in table.columns
        if col.name in body
    }

    try:
        session.execute(insert(table).values(**values))
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=_integrity_error_detail(exc))

    pk_value = values.get(pk_col.name)
    row = session.execute(select(table).where(pk_col == pk_value)).first()
    return _serialize_row(row) if row is not None else values


@router.put("/tables/{table_name}/{pk_value}")
def update_row(table_name: str, pk_value: str, body: dict, session: Session = Depends(get_db)) -> dict:
    table = _get_table(table_name)
    pk_col = _pk_column(table)

    existing = session.execute(select(table).where(pk_col == pk_value)).first()
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no row with {pk_col.name}={pk_value!r} in {table_name!r}")

    values = {
        col.name: _coerce_value(col, body[col.name])
        for col in table.columns
        if col.name in body and col.name != pk_col.name  # PK is immutable via this endpoint
    }

    if values:
        try:
            session.execute(update(table).where(pk_col == pk_value).values(**values))
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=_integrity_error_detail(exc))

    row = session.execute(select(table).where(pk_col == pk_value)).first()
    return _serialize_row(row)


@router.delete("/tables/{table_name}/{pk_value}", status_code=204)
def delete_row(table_name: str, pk_value: str, session: Session = Depends(get_db)) -> None:
    table = _get_table(table_name)
    pk_col = _pk_column(table)

    existing = session.execute(select(table).where(pk_col == pk_value)).first()
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no row with {pk_col.name}={pk_value!r} in {table_name!r}")

    try:
        session.execute(delete(table).where(pk_col == pk_value))
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete: referenced by another table (foreign key) -- {_integrity_error_detail(exc)}",
        )
