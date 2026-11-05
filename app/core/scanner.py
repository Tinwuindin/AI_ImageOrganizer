"""Escaneo recursivo de carpetas.

Solo se vuelven a procesar los archivos nuevos o modificados (por tamaño y fecha
de modificación), de modo que repetir un escaneo sobre 10,000 imágenes tarda
segundos. Si un archivo aparece en otra ruta con el mismo contenido, se trata
como un movimiento: conserva su análisis.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import ScannerConfig
from app.core.hashing import sha256_file
from app.core.metadata import read_metadata
from app.core.thumbnails import ensure_thumbnail
from app.database.models import STATUS_ERROR, STATUS_MISSING, ImageRecord, ScannedFile
from app.database.repository import Repository

logger = logging.getLogger(__name__)

# Se escribe a SQLite por lotes para no abrir una transacción por archivo.
_WRITE_BATCH = 100

_IMAGE_ERRORS = (
    UnidentifiedImageError,
    OSError,
    ValueError,
    SyntaxError,
    Image.DecompressionBombError,
)

ProgressCallback = Callable[[int, int], None]


@dataclass
class ScanSummary:
    discovered: int = 0
    added: int = 0
    changed: int = 0
    relocated: int = 0
    unchanged: int = 0
    failed: int = 0
    missing: int = 0
    cancelled: bool = False


def iter_image_files(
    roots: Iterable[Path],
    extensions: Iterable[str],
    exclude_dirs: Iterable[str],
    exclude_paths: Iterable[Path] = (),
) -> Iterator[Path]:
    """Recorre las carpetas y entrega los archivos con extensión admitida."""
    allowed = {extension.lower() for extension in extensions}
    excluded_names = {name.casefold() for name in exclude_dirs}
    excluded_paths = {os.path.normcase(str(path)) for path in exclude_paths}
    visited: set[str] = set()

    def on_error(error: OSError) -> None:
        logger.warning("No se pudo leer la carpeta %s: %s", error.filename, error.strerror)

    for root in roots:
        for directory, subdirectories, filenames in os.walk(root, onerror=on_error):
            key = os.path.normcase(directory)
            if key in visited:  # Raíces solapadas: cada carpeta se recorre una vez.
                subdirectories[:] = []
                continue
            visited.add(key)
            subdirectories[:] = [
                name for name in subdirectories
                if name.casefold() not in excluded_names
                and os.path.normcase(os.path.join(directory, name)) not in excluded_paths
            ]
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in allowed:
                    yield Path(directory, filename)


def inspect_file(
    path: Path,
    size: int,
    mtime: float,
    thumbnails_dir: Path,
    thumbnail_size: int,
) -> ScannedFile:
    """Calcula hash, metadatos y miniatura. Un archivo dañado no detiene el escaneo."""
    try:
        sha256 = sha256_file(path)
    except OSError as error:
        return ScannedFile(str(path), "", size, mtime, error=f"No se pudo leer: {error}")
    try:
        metadata = read_metadata(path)
        ensure_thumbnail(path, sha256, thumbnails_dir, thumbnail_size)
    except _IMAGE_ERRORS as error:
        return ScannedFile(str(path), sha256, size, mtime, error=f"Imagen no válida: {error}")
    return ScannedFile(
        str(path), sha256, size, mtime,
        width=metadata.width, height=metadata.height, date_taken=metadata.date_taken,
    )


def _is_inside(path: str, roots: Iterable[Path]) -> bool:
    normalized = os.path.normcase(path)
    return any(
        normalized.startswith(os.path.normcase(str(root)).rstrip("\\/") + os.sep)
        for root in roots
    )


class _ResultWriter:
    """Aplica en la base de datos cada archivo inspeccionado."""

    def __init__(
        self,
        repository: Repository,
        known: dict[str, ImageRecord],
        summary: ScanSummary,
    ) -> None:
        self._repository = repository
        self._known = known
        self._summary = summary
        self.relocated_ids: set[int] = set()

    def write(self, file: ScannedFile) -> None:
        existing = self._known.get(file.path)
        if existing is not None:
            self._write_existing(existing, file)
        elif file.sha256 and (relocated := self._find_relocated(file)) is not None:
            self._repository.relocate_image(relocated.id, file.path, file.mtime)
            self._repository.restore_status([relocated.id])
            self.relocated_ids.add(relocated.id)
            self._summary.relocated += 1
        else:
            self._repository.insert_image(file)
            self._summary.added += 1
        if file.error:
            self._summary.failed += 1
            logger.warning("%s: %s", file.path, file.error)

    def _write_existing(self, existing: ImageRecord, file: ScannedFile) -> None:
        if existing.sha256 == file.sha256 and file.error is None:
            self._repository.touch_image(existing.id, file.mtime)
            self._repository.restore_status([existing.id])
            self._summary.unchanged += 1
        else:
            self._repository.update_image_content(existing.id, file)
            self._summary.changed += 1

    def _find_relocated(self, file: ScannedFile) -> ImageRecord | None:
        for candidate in self._repository.find_by_sha256(file.sha256):
            if candidate.id in self.relocated_ids:
                continue
            renamed = os.path.normcase(candidate.path) == os.path.normcase(file.path)
            if renamed or not os.path.exists(candidate.path):
                return candidate
        return None


def _write_results(
    futures: list[Future[ScannedFile]],
    repository: Repository,
    writer: _ResultWriter,
    summary: ScanSummary,
    on_progress: ProgressCallback | None,
    cancel: threading.Event,
) -> None:
    total = len(futures)
    done = 0
    batch: list[ScannedFile] = []

    def flush() -> None:
        with repository.transaction():
            for file in batch:
                writer.write(file)
        batch.clear()
        if on_progress:
            on_progress(done, total)

    for future in as_completed(futures):
        if cancel.is_set():
            summary.cancelled = True
            break
        batch.append(future.result())
        done += 1
        if len(batch) >= _WRITE_BATCH:
            flush()
    flush()


def scan_directories(
    roots: list[Path],
    repository: Repository,
    config: ScannerConfig,
    thumbnails_dir: Path,
    exclude_paths: Iterable[Path] = (),
    on_progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> ScanSummary:
    summary = ScanSummary()
    cancel = cancel or threading.Event()
    known = repository.all_images()
    seen: set[str] = set()
    pending: list[tuple[Path, int, float]] = []
    restored: list[int] = []

    for path in iter_image_files(roots, config.extensions, config.exclude_dirs, exclude_paths):
        try:
            stat = path.stat()
        except OSError as error:
            logger.warning("No se pudo consultar %s: %s", path, error)
            continue
        key = str(path)
        seen.add(key)
        summary.discovered += 1
        record = known.get(key)
        unchanged = (
            record is not None
            and record.size == stat.st_size
            and record.mtime == stat.st_mtime
            and record.status != STATUS_ERROR  # Un error puede ser pasajero: se reintenta.
        )
        if unchanged:
            summary.unchanged += 1
            if record.status == STATUS_MISSING:
                restored.append(record.id)
        else:
            pending.append((path, stat.st_size, stat.st_mtime))

    if restored:
        repository.restore_status(restored)
    if on_progress:
        on_progress(0, len(pending))

    writer = _ResultWriter(repository, known, summary)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(inspect_file, path, size, mtime, thumbnails_dir, config.thumbnail_size)
            for path, size, mtime in pending
        ]
        try:
            _write_results(futures, repository, writer, summary, on_progress, cancel)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    if summary.cancelled:
        return summary

    gone = [
        record.id
        for path, record in known.items()
        if path not in seen
        and record.status != STATUS_MISSING
        and record.id not in writer.relocated_ids
        and _is_inside(path, roots)
        and not os.path.exists(path)
    ]
    if gone:
        repository.set_status(gone, STATUS_MISSING)
        summary.missing = len(gone)
    return summary
