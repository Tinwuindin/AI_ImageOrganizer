"""Detección de duplicados exactos por SHA-256.

Dos archivos con el mismo hash son idénticos aunque se llamen distinto. Solo la
copia canónica pasa por los modelos; las demás comparten su análisis y, al
organizar, se proponen para ``_Duplicados``. Nunca se elimina ningún archivo.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.database.models import (
    STATUS_CLASSIFIED,
    STATUS_DUPLICATE,
    STATUS_EMBEDDED,
    STATUS_SCANNED,
    ImageRecord,
)
from app.database.repository import Repository
from app.database.vector_index import VectorIndex

# Se conserva la copia con más análisis hecho: así no se repite trabajo.
_ANALYSIS_RANK = {STATUS_CLASSIFIED: 3, STATUS_EMBEDDED: 2, STATUS_SCANNED: 1, STATUS_DUPLICATE: 0}


def choose_canonical(copies: Sequence[ImageRecord]) -> ImageRecord:
    return min(
        copies,
        key=lambda record: (-_ANALYSIS_RANK.get(record.status, 0), len(record.path), record.id),
    )


def detect_exact_duplicates(repository: Repository) -> int:
    """Recalcula los grupos y devuelve cuántos hay."""
    groups: list[tuple[str, int]] = []
    canonical_ids: list[int] = []
    copy_ids: list[int] = []
    for sha256, copies in repository.duplicate_candidates().items():
        canonical = choose_canonical(copies)
        groups.append((sha256, canonical.id))
        if canonical.status == STATUS_DUPLICATE:
            canonical_ids.append(canonical.id)
        copy_ids.extend(record.id for record in copies if record.id != canonical.id)

    with repository.transaction():
        repository.replace_duplicate_groups(groups)
        repository.delete_embeddings(copy_ids)
        repository.set_status(copy_ids, STATUS_DUPLICATE)
        repository.restore_status(canonical_ids)
        repository.restore_status(repository.orphaned_duplicates())
    return len(groups)


def promote_copy(
    repository: Repository,
    vectors: VectorIndex,
    group_id: int,
    image_id: int,
) -> bool:
    """Elige otra copia como canónica; el análisis y el vector pasan a ella."""
    group = repository.duplicate_group(group_id)
    if group is None:
        return False
    previous = group["canonical_image_id"]
    vector = vectors.get(previous)
    if not repository.set_canonical(group_id, image_id):
        return False
    if vector is not None and previous != image_id:
        vectors.add([image_id], vector.reshape(1, -1))
        vectors.remove([previous])
        vectors.save()
    return True
