"""Read-only supplier surface.

    GET /suppliers               -- list
    GET /suppliers/{supplier_id} -- detail
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.schemas import SupplierOut
from src.db.models import Supplier

router = APIRouter()


@router.get("/suppliers", response_model=list[SupplierOut])
def list_suppliers(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_db),
) -> list[Supplier]:
    stmt = select(Supplier).order_by(Supplier.supplier_id).limit(limit)
    return list(session.execute(stmt).scalars().all())


@router.get("/suppliers/{supplier_id}", response_model=SupplierOut)
def get_supplier(supplier_id: str, session: Session = Depends(get_db)) -> Supplier:
    supplier = session.get(Supplier, supplier_id)
    if supplier is None:
        raise HTTPException(status_code=404, detail=f"supplier_id {supplier_id!r} not found")
    return supplier
