"""Aplicación FastAPI: API local e interfaz web."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import categories, images, organizer, pipeline, search
from app.context import AppContext, build_context
from app.core.config import PROJECT_ROOT, AppConfig, load_config

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _is_local_request(request: Request) -> bool:
    """Bloquea peticiones de otros sitios web abiertos en el navegador.

    La API mueve archivos del disco. Escuchar en 127.0.0.1 no basta: una página
    maliciosa podría enviar peticiones al puerto local (CSRF) o resolver su dominio
    a 127.0.0.1 (DNS rebinding). Por eso se validan ``Host`` y ``Origin``.
    """
    host = urlsplit(f"//{request.headers.get('host', '')}").hostname
    if host not in _LOCAL_HOSTS:
        return False
    origin = request.headers.get("origin")
    if request.method in _SAFE_METHODS or origin is None:
        return True
    return urlsplit(origin).hostname in _LOCAL_HOSTS


def create_app(config: AppConfig | None = None, app_context: AppContext | None = None) -> FastAPI:
    config = config or load_config()
    context = app_context or build_context(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        context.jobs.cancel()
        context.jobs.wait(timeout=60)
        context.classifier.unload()
        context.vectors.save()

    app = FastAPI(title="Organizador de imágenes", lifespan=lifespan)
    app.state.context = context

    @app.middleware("http")
    async def local_only(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not _is_local_request(request):
            return JSONResponse({"detail": "Solo se admiten peticiones locales."}, status_code=403)
        return await call_next(request)

    app.include_router(images.router)
    app.include_router(search.router)
    app.include_router(categories.router)
    app.include_router(organizer.router)
    app.include_router(pipeline.router)
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
    return app
