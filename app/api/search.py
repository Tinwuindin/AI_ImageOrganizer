"""Búsqueda por lenguaje natural."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from app.api.images import context

router = APIRouter(prefix="/api/search", tags=["búsqueda"])


@router.get("")
def search(
    request: Request,
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(60, ge=1, le=200),
    category: str | None = None,
) -> dict[str, Any]:
    return context(request).search.search(q, limit, category)
