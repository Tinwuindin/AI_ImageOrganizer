"""Orquestación del análisis por etapas.

Escaneo → duplicados → embeddings → agrupación → clasificación. Cada etapa toma
solo lo que le falta, de modo que el análisis se puede cancelar y retomar.

Los modelos se turnan la GPU: EmbeddingGemma se descarga antes de iniciar Gemma.
Juntos cabrían en 8 GB (1.6 + 2.9 GB medidos), pero la VRAM libre depende de otras
aplicaciones abiertas y alternarlos no cuesta casi nada: cada uno corre una vez.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import AppConfig
from app.core.scanner import scan_directories
from app.core.thumbnails import open_normalized
from app.database.models import STATUS_ERROR
from app.database.repository import Repository
from app.database.vector_index import VectorIndex
from app.engine.categorizer import Categorizer, ImageClassifier
from app.engine.clustering import NOISE, cluster_vectors
from app.engine.duplicate_detector import detect_exact_duplicates
from app.models.embedding_model import GpuMemoryError
from app.services.jobs import JobContext

logger = logging.getLogger(__name__)

STAGE_SCAN = "scan"
STAGE_DUPLICATES = "duplicates"
STAGE_EMBEDDINGS = "embeddings"
STAGE_CLUSTERING = "clustering"
STAGE_CLASSIFICATION = "classification"

_DECODE_WORKERS = 4


class ImageEncoderProtocol(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def batch_size(self) -> int: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def reconcile_vectors(repository: Repository, vectors: VectorIndex) -> None:
    """Corrige diferencias entre SQLite y FAISS tras una interrupción."""
    registered = set(repository.embedding_ids())
    indexed = {int(identifier) for identifier in vectors.ids()}
    orphans = sorted(indexed - registered)
    lost = sorted(registered - indexed)
    if orphans:
        vectors.remove(orphans)
        vectors.save()
    if lost:
        logger.warning("%d vectores no están en FAISS: se volverán a calcular.", len(lost))
        repository.delete_embeddings(lost)


def _batches(items: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class IndexingPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: Repository,
        vectors: VectorIndex,
        image_encoder: ImageEncoderProtocol,
        classifier: ImageClassifier,
    ) -> None:
        self._config = config
        self._repository = repository
        self._vectors = vectors
        self._image_encoder = image_encoder
        self._classifier = classifier

    def run(self, job: JobContext, roots: list[Path]) -> None:
        try:
            self._scan(job, roots)
            if job.cancelled:
                return
            self._find_duplicates(job)
            self._embed(job)
            if job.cancelled:
                return
            self._cluster(job)
            self._classify(job)
        finally:
            self._image_encoder.unload()
            self._classifier.unload()
            self._vectors.save()

    def _scan(self, job: JobContext, roots: list[Path]) -> None:
        job.stage(STAGE_SCAN)
        summary = scan_directories(
            roots,
            self._repository,
            self._config.scanner,
            self._config.thumbnails_dir,
            exclude_paths=[self._config.data_dir, self._config.root / ".venv"],
            on_progress=job.advance,
            cancel=job.cancel,
        )
        job.note("scan", asdict(summary))

    def _find_duplicates(self, job: JobContext) -> None:
        job.stage(STAGE_DUPLICATES)
        groups = detect_exact_duplicates(self._repository)
        reconcile_vectors(self._repository, self._vectors)
        job.note("duplicate_groups", groups)

    def _embed(self, job: JobContext) -> None:
        pending = self._repository.ids_pending_embedding()
        job.stage(STAGE_EMBEDDINGS, len(pending))
        if not pending:
            return
        self._image_encoder.load()
        records = self._repository.get_images(pending)
        unsaved: list[int] = []
        done = 0
        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as decoder:
            for batch in _batches(pending, self._image_encoder.batch_size):
                if job.cancelled:
                    break
                paths = [Path(records[image_id].path) for image_id in batch]
                decoded = list(decoder.map(self._decode, paths))
                valid = self._drop_unreadable(batch, decoded)
                encoded = self._encode(
                    [image_id for image_id, _ in valid], [image for _, image in valid]
                )
                if encoded:
                    self._vectors.add(
                        [image_id for image_id, _ in encoded],
                        np.stack([vector for _, vector in encoded]),
                    )
                    unsaved.extend(image_id for image_id, _ in encoded)
                done += len(batch)
                job.advance(done)
                if len(unsaved) >= self._config.embedding.checkpoint_every:
                    self._checkpoint(unsaved)
        self._checkpoint(unsaved)
        self._image_encoder.unload()
        job.note("embedded", done)

    def _decode(self, path: Path) -> Image.Image | str:
        try:
            return open_normalized(path, self._config.embedding.image_max_side)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            return str(error) or type(error).__name__

    def _drop_unreadable(
        self,
        ids: Sequence[int],
        decoded: Sequence[Image.Image | str],
    ) -> list[tuple[int, Image.Image]]:
        valid: list[tuple[int, Image.Image]] = []
        for image_id, result in zip(ids, decoded):
            if isinstance(result, str):
                self._repository.set_status([image_id], STATUS_ERROR, f"No se pudo leer: {result}")
                logger.warning("Imagen %d: %s", image_id, result)
            else:
                valid.append((image_id, result))
        return valid

    def _encode(
        self,
        ids: Sequence[int],
        images: Sequence[Image.Image],
    ) -> list[tuple[int, np.ndarray]]:
        """Codifica un lote. Si una imagen lo hace fallar, la aísla y sigue con el resto."""
        if not ids:
            return []
        try:
            return list(zip(ids, self._image_encoder.encode(images)))
        except GpuMemoryError:
            raise
        except (RuntimeError, ValueError) as error:
            if len(ids) == 1:
                self._repository.set_status(ids, STATUS_ERROR, f"No se pudo analizar: {error}")
                logger.warning("Imagen %d: %s", ids[0], error)
                return []
        return [
            pair
            for image_id, image in zip(ids, images)
            for pair in self._encode([image_id], [image])
        ]

    def _checkpoint(self, unsaved: list[int]) -> None:
        """Guarda primero FAISS: un vector sin registro se limpia al reconciliar."""
        if not unsaved:
            return
        self._vectors.save()
        self._repository.save_embeddings(unsaved, self._image_encoder.model_name)
        unsaved.clear()

    def _cluster(self, job: JobContext) -> None:
        job.stage(STAGE_CLUSTERING)
        ids = self._repository.embedded_active_ids()
        if not ids:
            return
        labels = cluster_vectors(self._vectors.matrix(ids), self._config.clustering)
        self._repository.set_clusters([
            (image_id, None if int(label) == NOISE else int(label))
            for image_id, label in zip(ids, labels)
        ])
        job.note("clusters", int(len({int(label) for label in labels} - {NOISE})))
        job.note("unclustered", int((labels == NOISE).sum()))

    def _classify(self, job: JobContext) -> None:
        job.stage(STAGE_CLASSIFICATION)
        categorizer = Categorizer(
            self._repository,
            self._vectors,
            self._classifier,
            self._config.classification,
            self._config.gemma,
        )
        job.note("classification", categorizer.run(job))
