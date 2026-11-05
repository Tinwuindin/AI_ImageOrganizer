"""Control del análisis y utilidades del sistema local."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.images import context
from app.services.jobs import JobBusyError, JobContext

router = APIRouter(prefix="/api", tags=["análisis"])

# Se abre en un proceso aparte: Tkinter debe vivir en el hilo principal de su proceso.
_FOLDER_DIALOG = """
import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
print(filedialog.askdirectory(title="Selecciona una carpeta") or "", end="")
"""
_DIALOG_TIMEOUT_SECONDS = 600


class AnalysisRequest(BaseModel):
    roots: list[str] = Field(min_length=1, max_length=20)


@router.post("/pipeline/start", status_code=202)
def start_analysis(body: AnalysisRequest, request: Request) -> dict[str, Any]:
    ctx = context(request)
    roots: list[Path] = []
    for raw in body.roots:
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise HTTPException(422, f"La carpeta no existe: {raw}")
        roots.append(path.resolve())

    def run(job: JobContext) -> None:
        ctx.pipeline.run(job, roots)

    try:
        ctx.jobs.start("analysis", run)
    except JobBusyError as error:
        raise HTTPException(409, str(error)) from error
    return ctx.jobs.snapshot()


@router.post("/pipeline/cancel")
def cancel_analysis(request: Request) -> dict[str, bool]:
    return {"cancelling": context(request).jobs.cancel()}


@router.get("/pipeline/status")
def analysis_status(request: Request) -> dict[str, Any]:
    return context(request).jobs.snapshot()


@router.get("/system/info")
def system_info(request: Request) -> dict[str, Any]:
    """Indica si los modelos están descargados y los ajustes que usa la interfaz."""
    config = context(request).config
    items = {
        "EmbeddingGemma 2": config.resolve(config.embedding.model_path) / "model.safetensors",
        "Gemma 4 E2B": config.resolve(config.gemma.model_path),
        "Proyector de visión": config.resolve(config.gemma.mmproj_path),
        "llama-server": config.resolve(config.paths.llama_server),
    }
    missing = [name for name, path in items.items() if not path.exists()]
    return {
        "ready": not missing,
        "missing": missing,
        "min_confidence": config.organization.min_confidence,
    }


@router.post("/system/pick-folder")
def pick_folder() -> dict[str, str | None]:
    """Abre el selector de carpetas nativo del equipo."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _FOLDER_DIALOG],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_DIALOG_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise HTTPException(500, "No se pudo abrir el selector de carpetas.") from error
    selected = result.stdout.strip()
    return {"path": str(Path(selected)) if selected else None}
