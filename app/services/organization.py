"""Organización física de los archivos.

El flujo tiene tres pasos, y ninguno destruye datos:

1. ``create_plan`` calcula una propuesta sin tocar el disco.
2. La persona la revisa: excluye imágenes o carpetas, o cambia destinos.
3. ``execute`` mueve los archivos; ``undo`` los devuelve a su lugar.

Nunca se sobrescribe un archivo ni se borra nada. Si el destino está en otro
disco, primero se copia y se verifica el hash; el original solo se retira después.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Any

from app.core.config import OrganizationConfig
from app.core.hashing import sha256_file
from app.database.models import (
    OP_DONE,
    OP_PLANNED,
    OP_UNDONE,
    PLAN_DRAFT,
    PLAN_EXECUTED,
    PLAN_RUNNING,
    PLAN_UNDONE,
    REASON_CLASSIFIED,
    REASON_DUPLICATE,
    REASON_LOW_CONFIDENCE,
    NewOperation,
    Operation,
)
from app.database.repository import Repository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

TEMPLATE_FIELDS = frozenset({"category", "subcategory", "year", "month"})
MISCELLANEOUS_FOLDER = "Varios"
_MAX_COMPONENT_LENGTH = 60

_INVALID_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


class OrganizationError(ValueError):
    """La operación pedida no es válida."""


def safe_component(text: str, fallback: str) -> str:
    """Convierte un texto en un nombre de carpeta válido en Windows."""
    cleaned = _INVALID_CHARACTERS.sub(" ", text)
    cleaned = " ".join(cleaned.split())[:_MAX_COMPONENT_LENGTH].rstrip(". ")
    if not cleaned or cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        return fallback
    return cleaned


def validate_template(template: str) -> str:
    """Comprueba que la plantilla solo use campos conocidos y carpetas relativas."""
    normalized = template.strip().strip("/\\").replace("\\", "/")
    if not normalized:
        raise OrganizationError("La plantilla está vacía.")
    try:
        fields = {name for _, name, _, _ in Formatter().parse(normalized) if name is not None}
    except ValueError as error:
        raise OrganizationError(f"Plantilla no válida: {error}") from error
    unknown = fields - TEMPLATE_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise OrganizationError(f"Campos desconocidos en la plantilla: {names}.")
    if not fields:
        raise OrganizationError("La plantilla debe usar al menos un campo.")
    if any(part in ("", ".", "..") for part in normalized.split("/")):
        raise OrganizationError("La plantilla no puede subir de carpeta.")
    return normalized


def move_file(source: Path, target: Path, expected_sha256: str) -> None:
    """Mueve un archivo sin sobrescribir nunca el destino."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(f"Ya existe {target}")
    if source.stat().st_dev == target.parent.stat().st_dev:
        os.rename(source, target)
        return

    temporary = target.with_name(f"{target.name}.moviendo")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise OSError(f"La copia de {source.name} no coincide con el original.")
        os.rename(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    os.unlink(source)


def _unique_target(folder: Path, filename: str, source: str, reserved: set[str]) -> Path | None:
    """Elige una ruta libre. ``None`` indica que la imagen ya está en su sitio."""
    stem, suffix = os.path.splitext(filename)
    counter = 1
    while True:
        name = filename if counter == 1 else f"{stem}_{counter}{suffix}"
        candidate = folder / name
        key = os.path.normcase(str(candidate))
        if key == os.path.normcase(source):
            return None
        if key not in reserved and not os.path.lexists(candidate):
            reserved.add(key)
            return candidate
        counter += 1


def _capture_date(image: dict[str, Any]) -> datetime:
    if image.get("date_taken"):
        return datetime.fromisoformat(image["date_taken"])
    return datetime.fromtimestamp(image["mtime"])


class OrganizationService:
    def __init__(self, repository: Repository, config: OrganizationConfig) -> None:
        self._repository = repository
        self._config = config

    # ------------------------------------------------------------------
    # Propuesta
    # ------------------------------------------------------------------

    def create_plan(self, destination: str, template: str | None = None) -> int:
        """Genera una propuesta nueva. No toca ningún archivo."""
        template = validate_template(template or self._config.default_template)
        root = Path(destination).expanduser()
        if not root.is_absolute():
            raise OrganizationError("El destino debe ser una ruta absoluta.")
        root = root.resolve()
        if root.exists() and not root.is_dir():
            raise OrganizationError("El destino no es una carpeta.")

        operations = self._plan_classified(root, template)
        operations += self._plan_duplicates(root)

        for plan in self._repository.list_plans():
            if plan["status"] == PLAN_DRAFT:
                self._repository.delete_plan(plan["id"])
        return self._repository.create_plan(str(root), template, operations)

    def _plan_classified(self, root: Path, template: str) -> list[NewOperation]:
        images = self._repository.organizable_images()
        sizes = Counter((image["category"], image["subcategory"]) for image in images)
        reserved: set[str] = set()
        operations: list[NewOperation] = []
        for image in images:
            folder = self._folder_for(image, template, sizes)
            target = _unique_target(root / folder, image["filename"], image["path"], reserved)
            if target is None:
                continue
            low_confidence = image["confidence"] < self._config.min_confidence
            operations.append(NewOperation(
                image_id=image["id"],
                source_path=image["path"],
                target_path=str(target),
                folder=folder,
                reason=REASON_LOW_CONFIDENCE if low_confidence else REASON_CLASSIFIED,
                included=not low_confidence,
            ))
        return operations

    def _folder_for(
        self,
        image: dict[str, Any],
        template: str,
        sizes: Counter[tuple[str, str]],
    ) -> str:
        subcategory = image["subcategory"]
        # Sin este límite habría cientos de carpetas con una sola foto.
        if sizes[(image["category"], subcategory)] < self._config.min_subcategory_size:
            subcategory = MISCELLANEOUS_FOLDER
        date = _capture_date(image)
        # Se limpian antes de aplicar la plantilla: un "/" en un valor no crea carpetas.
        return template.format(
            category=safe_component(image["category"], MISCELLANEOUS_FOLDER),
            subcategory=safe_component(subcategory, MISCELLANEOUS_FOLDER),
            year=f"{date.year:04d}",
            month=f"{date.month:02d}",
        )

    def _plan_duplicates(self, root: Path) -> list[NewOperation]:
        folder = safe_component(self._config.duplicates_folder, "_Duplicados")
        reserved: set[str] = set()
        operations: list[NewOperation] = []
        for record in self._repository.duplicate_copies():
            target = _unique_target(root / folder, record.filename, record.path, reserved)
            if target is None:
                continue
            operations.append(NewOperation(
                image_id=record.id,
                source_path=record.path,
                target_path=str(target),
                folder=folder,
                reason=REASON_DUPLICATE,
            ))
        return operations

    def retarget_folder(self, plan_id: int, folder: str, new_folder: str) -> None:
        """Renombra una carpeta de la propuesta antes de ejecutarla."""
        plan = self._draft(plan_id)
        parts = [part for part in new_folder.replace("\\", "/").split("/") if part.strip()]
        cleaned = [safe_component(part, "") for part in parts]
        if not cleaned or any(not part or part in (".", "..") for part in cleaned):
            raise OrganizationError("El nombre de la carpeta no es válido.")
        normalized = "/".join(cleaned)
        root = Path(plan.destination)
        reserved = {os.path.normcase(path) for path in self._repository.plan_targets(plan_id)}
        with self._repository.transaction():
            for operation in self._repository.plan_operations(plan_id, folder=folder):
                filename = Path(operation.target_path).name
                target = _unique_target(root / normalized, filename, operation.source_path, reserved)
                if target is None:
                    target = root / normalized / filename
                self._repository.retarget_operation(operation.id, str(target), normalized)

    def _draft(self, plan_id: int) -> Any:
        plan = self._repository.get_plan(plan_id)
        if plan is None:
            raise OrganizationError("El plan no existe.")
        if plan.status != PLAN_DRAFT:
            raise OrganizationError("Solo se puede cambiar una propuesta sin ejecutar.")
        return plan

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(
        self,
        plan_id: int,
        on_progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, int]:
        self._draft(plan_id)
        cancel = cancel or threading.Event()
        self._repository.set_plan_status(plan_id, PLAN_RUNNING)
        operations = [
            operation for operation in self._repository.plan_operations(plan_id)
            if operation.included and operation.status == OP_PLANNED
        ]
        moved = failed = 0
        try:
            for index, operation in enumerate(operations):
                if cancel.is_set():
                    break
                if self._apply(operation):
                    moved += 1
                else:
                    failed += 1
                if on_progress:
                    on_progress(index + 1, len(operations))
        finally:
            self._repository.set_plan_status(plan_id, PLAN_EXECUTED, executed=True)
        return {"moved": moved, "failed": failed, "total": len(operations)}

    def _apply(self, operation: Operation) -> bool:
        image = self._repository.get_image(operation.image_id)
        if image is None or image.path != operation.source_path:
            self._repository.fail_operation(operation.id, "La imagen cambió de lugar.")
            return False
        source = Path(operation.source_path)
        target = Path(operation.target_path)
        try:
            if source.stat().st_size != image.size:
                raise OSError("El archivo cambió desde que se analizó.")
            move_file(source, target, image.sha256)
            mtime = target.stat().st_mtime
        except OSError as error:
            self._repository.fail_operation(operation.id, str(error))
            logger.warning("No se pudo mover %s: %s", source, error)
            return False
        self._repository.finish_operation(operation, str(target), mtime, OP_DONE)
        return True

    def recover_interrupted(self) -> None:
        """Completa el registro de un plan que quedó a medias por un cierre inesperado.

        El archivo se mueve antes de anotarlo, así que puede estar ya en su destino
        con el movimiento sin registrar. Sin esto, el plan quedaría en curso para
        siempre y esos archivos no se podrían deshacer.
        """
        for plan in self._repository.list_plans():
            if plan["status"] != PLAN_RUNNING:
                continue
            for operation in self._repository.plan_operations(plan["id"]):
                if operation.status != OP_PLANNED or not operation.included:
                    continue
                target = Path(operation.target_path)
                if target.exists() and not Path(operation.source_path).exists():
                    self._repository.finish_operation(
                        operation, str(target), target.stat().st_mtime, OP_DONE
                    )
            self._repository.set_plan_status(plan["id"], PLAN_EXECUTED, executed=True)
            logger.warning("Se recuperó el plan %d, interrumpido a medias.", plan["id"])

    def undo(
        self,
        plan_id: int,
        on_progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, int]:
        """Devuelve a su lugar los archivos que el plan movió."""
        plan = self._repository.get_plan(plan_id)
        if plan is None or plan.status != PLAN_EXECUTED:
            raise OrganizationError("Solo se puede deshacer un plan ejecutado.")
        cancel = cancel or threading.Event()
        operations = [
            operation for operation in self._repository.plan_operations(plan_id)
            if operation.status == OP_DONE
        ]
        restored = failed = 0
        for index, operation in enumerate(reversed(operations)):
            if cancel.is_set():
                break
            if self._revert(operation):
                restored += 1
            else:
                failed += 1
            if on_progress:
                on_progress(index + 1, len(operations))
        if not cancel.is_set() and failed == 0:
            self._repository.set_plan_status(plan_id, PLAN_UNDONE)
        self._remove_empty_folders(Path(plan.destination), operations)
        return {"restored": restored, "failed": failed, "total": len(operations)}

    def _revert(self, operation: Operation) -> bool:
        image = self._repository.get_image(operation.image_id)
        source = Path(operation.target_path)
        target = Path(operation.source_path)
        if image is None or image.path != operation.target_path:
            self._repository.note_operation_error(operation.id, "La imagen cambió de lugar.")
            return False
        try:
            # Un deshacer interrumpido pudo mover el archivo sin llegar a anotarlo.
            if source.exists() or not target.exists():
                move_file(source, target, image.sha256)
            mtime = target.stat().st_mtime
        except OSError as error:
            self._repository.note_operation_error(operation.id, str(error))
            logger.warning("No se pudo restaurar %s: %s", target, error)
            return False
        self._repository.finish_operation(operation, str(target), mtime, OP_UNDONE)
        return True

    @staticmethod
    def _remove_empty_folders(root: Path, operations: list[Operation]) -> None:
        """Quita las carpetas de destino que quedaron vacías, nunca otra cosa."""
        folders = {Path(operation.target_path).parent for operation in operations}
        for folder in sorted(folders, key=lambda path: len(path.parts), reverse=True):
            current = folder
            while current != root and root in current.parents:
                try:
                    current.rmdir()  # Solo funciona si está vacía.
                except OSError:
                    break
                current = current.parent
