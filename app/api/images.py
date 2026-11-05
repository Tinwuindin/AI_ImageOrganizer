"""Biblioteca de imágenes.

Los archivos solo se sirven a partir de un ``id`` registrado: el cliente nunca
indica una ruta del disco.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.context import AppContext
from app.core.thumbnails import encode_jpeg, open_normalized, thumbnail_path
from app.database.models import ImageFilters, ImageRecord
from app.engine.duplicate_detector import promote_copy
from app.engine.similarity import similar_images

router = APIRouter(prefix="/api/images", tags=["imágenes"])

_PREVIEW_SIDE = 1600
_LONG_CACHE = {"Cache-Control": "public, max-age=31536000, immutable"}


def context(request: Request) -> AppContext:
    return request.app.state.context


def _require_image(ctx: AppContext, image_id: int) -> ImageRecord:
    record = ctx.repository.get_image(image_id)
    if record is None:
        raise HTTPException(404, "La imagen no existe.")
    return record


class ClassificationUpdate(BaseModel):
    category: str = Field(min_length=1, max_length=80)
    subcategory: str = Field(default="", max_length=80)


class KeepCopy(BaseModel):
    image_id: int


@router.get("")
def list_images(
    request: Request,
    category: str | None = None,
    subcategory: str | None = None,
    status: str | None = None,
    source: str | None = None,
    low_confidence: bool = False,
    sort: str = "date_desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(60, ge=1, le=200),
) -> dict[str, Any]:
    ctx = context(request)
    filters = ImageFilters(
        category=category,
        subcategory=subcategory,
        status=status,
        source=source,
        max_confidence=ctx.config.organization.min_confidence if low_confidence else None,
    )
    items, total = ctx.repository.list_images(filters, page_size, (page - 1) * page_size, sort)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/stats")
def stats(request: Request) -> dict[str, Any]:
    return context(request).repository.stats()


@router.get("/duplicates")
def duplicates(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
) -> dict[str, Any]:
    repository = context(request).repository
    groups, total = repository.duplicate_groups(page_size, (page - 1) * page_size)
    return {
        "groups": groups,
        "total": total,
        "recoverable_bytes": repository.duplicate_recoverable_bytes(),
        "page": page,
        "page_size": page_size,
    }


@router.post("/duplicates/{group_id}/keep")
def keep_copy(group_id: int, body: KeepCopy, request: Request) -> dict[str, bool]:
    ctx = context(request)
    if ctx.jobs.is_busy:
        raise HTTPException(409, "Espera a que termine la tarea en curso.")
    if not promote_copy(ctx.repository, ctx.vectors, group_id, body.image_id):
        raise HTTPException(404, "Esa copia no pertenece al grupo.")
    return {"ok": True}


@router.get("/{image_id}")
def image_detail(image_id: int, request: Request) -> dict[str, Any]:
    item = context(request).repository.library_item(image_id)
    if item is None:
        raise HTTPException(404, "La imagen no existe.")
    return item


@router.get("/{image_id}/thumbnail")
def thumbnail(image_id: int, request: Request, v: str | None = None) -> FileResponse:
    """Miniatura. Con ``v`` (huella del contenido) el navegador la guarda para siempre."""
    ctx = context(request)
    record = _require_image(ctx, image_id)
    path = thumbnail_path(ctx.config.thumbnails_dir, record.sha256) if record.sha256 else None
    if path is None or not path.exists():
        raise HTTPException(404, "La imagen no tiene miniatura.")
    headers = _LONG_CACHE if v else {"Cache-Control": "no-cache"}
    return FileResponse(path, media_type="image/jpeg", headers=headers)


@router.get("/{image_id}/preview")
def preview(image_id: int, request: Request) -> Response:
    """Versión grande en JPEG: también sirve para formatos como HEIC."""
    record = _require_image(context(request), image_id)
    try:
        image = open_normalized(Path(record.path), _PREVIEW_SIDE)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise HTTPException(404, f"No se pudo abrir el archivo: {error}") from error
    return Response(
        encode_jpeg(image),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/{image_id}/similar")
def similar(
    image_id: int,
    request: Request,
    limit: int = Query(12, ge=1, le=60),
) -> dict[str, Any]:
    ctx = context(request)
    _require_image(ctx, image_id)
    matches = similar_images(ctx.vectors, image_id, limit)
    items = ctx.repository.library_items([match for match, _ in matches])
    return {
        "results": [
            {**items[match], "similarity": score}
            for match, score in matches
            if match in items
        ]
    }


@router.patch("/{image_id}/classification")
def update_classification(
    image_id: int,
    body: ClassificationUpdate,
    request: Request,
) -> dict[str, Any]:
    """Corrección manual: el análisis nunca vuelve a sobrescribirla."""
    ctx = context(request)
    _require_image(ctx, image_id)
    names = {category.name for category in ctx.repository.list_categories()}
    if body.category not in names:
        raise HTTPException(422, "La categoría no existe.")
    ctx.repository.set_manual_category(image_id, body.category, body.subcategory.strip())
    item = ctx.repository.library_item(image_id)
    assert item is not None
    return item
