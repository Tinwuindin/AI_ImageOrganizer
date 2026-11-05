"""Propuesta, revisión y ejecución de la organización física."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.images import context
from app.database.models import PLAN_DRAFT, PLAN_EXECUTED
from app.services.jobs import JobBusyError, JobContext
from app.services.organization import OrganizationError

router = APIRouter(prefix="/api/organizer", tags=["organizador"])


class PlanRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=1024)
    template: str | None = Field(default=None, max_length=200)


class OperationUpdate(BaseModel):
    included: bool


class FolderUpdate(BaseModel):
    folder: str
    included: bool | None = None
    new_folder: str | None = Field(default=None, max_length=300)


def _plan_detail(ctx: Any, plan_id: int) -> dict[str, Any]:
    plan = ctx.repository.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, "El plan no existe.")
    folders = ctx.repository.plan_folders(plan_id)
    return {
        **asdict(plan),
        "folders": folders,
        "total": sum(folder["total"] for folder in folders),
        "included": sum(folder["included"] or 0 for folder in folders),
        "done": sum(folder["done"] or 0 for folder in folders),
        "failed": sum(folder["failed"] or 0 for folder in folders),
    }


@router.get("/plans")
def list_plans(request: Request) -> dict[str, Any]:
    return {"plans": context(request).repository.list_plans()}


@router.post("/plans", status_code=201)
def create_plan(body: PlanRequest, request: Request) -> dict[str, Any]:
    ctx = context(request)
    try:
        plan_id = ctx.organization.create_plan(body.destination, body.template)
    except OrganizationError as error:
        raise HTTPException(422, str(error)) from error
    return _plan_detail(ctx, plan_id)


@router.get("/plans/{plan_id}")
def plan_detail(plan_id: int, request: Request) -> dict[str, Any]:
    return _plan_detail(context(request), plan_id)


@router.delete("/plans/{plan_id}")
def discard_plan(plan_id: int, request: Request) -> dict[str, bool]:
    repository = context(request).repository
    plan = repository.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, "El plan no existe.")
    if plan.status != PLAN_DRAFT:
        raise HTTPException(409, "Solo se puede descartar una propuesta sin ejecutar.")
    repository.delete_plan(plan_id)
    return {"ok": True}


@router.get("/plans/{plan_id}/operations")
def plan_operations(
    plan_id: int,
    request: Request,
    folder: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(120, ge=1, le=500),
) -> dict[str, Any]:
    operations = context(request).repository.plan_operations(
        plan_id, folder, page_size, (page - 1) * page_size
    )
    return {"operations": [asdict(operation) for operation in operations]}


@router.patch("/plans/{plan_id}/folders")
def update_folder(plan_id: int, body: FolderUpdate, request: Request) -> dict[str, Any]:
    ctx = context(request)
    try:
        if body.included is not None:
            ctx.repository.set_folder_included(plan_id, body.folder, body.included)
        if body.new_folder is not None:
            ctx.organization.retarget_folder(plan_id, body.folder, body.new_folder)
    except OrganizationError as error:
        raise HTTPException(422, str(error)) from error
    return _plan_detail(ctx, plan_id)


@router.patch("/operations/{operation_id}")
def update_operation(
    operation_id: int,
    body: OperationUpdate,
    request: Request,
) -> dict[str, bool]:
    repository = context(request).repository
    if repository.get_operation(operation_id) is None:
        raise HTTPException(404, "El movimiento no existe.")
    repository.set_operation_included(operation_id, body.included)
    return {"ok": True}


def _start_job(ctx: Any, kind: str, target: Any) -> dict[str, Any]:
    try:
        ctx.jobs.start(kind, target)
    except JobBusyError as error:
        raise HTTPException(409, str(error)) from error
    return ctx.jobs.snapshot()


@router.post("/plans/{plan_id}/execute", status_code=202)
def execute_plan(plan_id: int, request: Request) -> dict[str, Any]:
    ctx = context(request)
    plan = ctx.repository.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, "El plan no existe.")
    if plan.status != PLAN_DRAFT:
        raise HTTPException(409, "Este plan ya se ejecutó.")

    def run(job: JobContext) -> dict[str, Any]:
        job.stage("organize")
        return ctx.organization.execute(plan_id, job.advance, job.cancel)

    return _start_job(ctx, "organize", run)


@router.post("/plans/{plan_id}/undo", status_code=202)
def undo_plan(plan_id: int, request: Request) -> dict[str, Any]:
    ctx = context(request)
    plan = ctx.repository.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, "El plan no existe.")
    if plan.status != PLAN_EXECUTED:
        raise HTTPException(409, "Solo se puede deshacer un plan ejecutado.")

    def run(job: JobContext) -> dict[str, Any]:
        job.stage("undo")
        return ctx.organization.undo(plan_id, job.advance, job.cancel)

    return _start_job(ctx, "undo", run)
