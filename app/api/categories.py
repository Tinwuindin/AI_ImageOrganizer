"""Administración de categorías."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.images import context
from app.database.models import FALLBACK_CATEGORY

router = APIRouter(prefix="/api/categories", tags=["categorías"])


class CategoryInput(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=300)


@router.get("")
def list_categories(request: Request) -> dict[str, Any]:
    return {"categories": context(request).repository.category_tree()}


@router.post("", status_code=201)
def create_category(body: CategoryInput, request: Request) -> dict[str, Any]:
    try:
        category = context(request).repository.create_category(
            body.name.strip(), body.description.strip()
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "Ya existe una categoría con ese nombre.") from error
    return {"id": category.id, "name": category.name, "description": category.description}


@router.patch("/{category_id}")
def update_category(category_id: int, body: CategoryInput, request: Request) -> dict[str, bool]:
    repository = context(request).repository
    current = repository.get_category(category_id)
    if current is None:
        raise HTTPException(404, "La categoría no existe.")
    if current.name == FALLBACK_CATEGORY and body.name.strip() != FALLBACK_CATEGORY:
        raise HTTPException(409, f"La categoría {FALLBACK_CATEGORY} no se puede renombrar.")
    try:
        repository.update_category(category_id, body.name.strip(), body.description.strip())
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "Ya existe una categoría con ese nombre.") from error
    return {"ok": True}


@router.delete("/{category_id}")
def delete_category(
    category_id: int,
    request: Request,
    reassign_to: str = FALLBACK_CATEGORY,
) -> dict[str, bool]:
    repository = context(request).repository
    current = repository.get_category(category_id)
    if current is None:
        raise HTTPException(404, "La categoría no existe.")
    if current.name == FALLBACK_CATEGORY:
        raise HTTPException(409, f"La categoría {FALLBACK_CATEGORY} no se puede eliminar.")
    names = {category.name for category in repository.list_categories()}
    if reassign_to not in names or reassign_to == current.name:
        raise HTTPException(422, "La categoría de destino no es válida.")
    repository.delete_category(category_id, reassign_to)
    return {"ok": True}
