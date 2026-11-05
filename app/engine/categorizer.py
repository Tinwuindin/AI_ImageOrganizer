"""Clasificación semántica guiada por los grupos visuales.

Para no enviar las 10,000 imágenes a Gemma:

1. De cada grupo se analizan unos representantes, repartidos por todo el grupo.
2. Si coinciden en la categoría, el grupo queda validado: sus miembros heredan la
   clasificación del representante más parecido.
3. Se analizan una por una las imágenes ambiguas: las que no entraron en ningún
   grupo, las de grupos sin acuerdo y las que quedan lejos del centro de su grupo.

Las imágenes con una clasificación previa no se vuelven a analizar, así que
retomar un análisis interrumpido no repite trabajo.
"""

from __future__ import annotations

import logging
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import UnidentifiedImageError

from app.core.config import ClassificationConfig, GemmaConfig
from app.database.models import (
    SOURCE_CLUSTER,
    SOURCE_DIRECT,
    STATUS_ERROR,
    Category,
    Classification,
    ImageRecord,
)
from app.database.repository import Repository
from app.database.vector_index import VectorIndex
from app.models.gemma_classifier import GemmaError, ImageAnalysis
from app.services.jobs import JobContext

logger = logging.getLogger(__name__)

_MIN_VOTERS = 2


class ImageClassifier(Protocol):
    @property
    def version(self) -> str: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def analyze(self, image_path: Path, categories: Sequence[Category]) -> ImageAnalysis: ...


def subcategory_key(name: str) -> str:
    """Clave que iguala variantes como ``Gatos``, ``gato`` y ``Gáto``."""
    decomposed = unicodedata.normalize("NFKD", name)
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    key = " ".join(plain.casefold().split())
    return key[:-1] if len(key) > 3 and key.endswith("s") else key


class SubcategoryNames:
    """Unifica la escritura de las subcategorías para no multiplicar carpetas."""

    def __init__(self, existing: Iterable[tuple[str, str]]) -> None:
        self._names: dict[tuple[str, str], str] = {}
        for category, subcategory in existing:
            self.canonical(category, subcategory)

    def canonical(self, category: str, subcategory: str) -> str:
        if not subcategory:
            return subcategory
        return self._names.setdefault((category, subcategory_key(subcategory)), subcategory)


def representative_count(size: int, config: ClassificationConfig) -> int:
    wanted = math.ceil(math.sqrt(size))
    return min(size, max(config.min_representatives, min(wanted, config.max_representatives)))


def select_representatives(vectors: np.ndarray, count: int) -> list[int]:
    """Elige posiciones que cubren el grupo: el centro y luego lo más alejado."""
    centroid = vectors.mean(axis=0)
    first = int(np.argmax(vectors @ centroid))
    chosen = [first]
    closest = vectors @ vectors[first]
    while len(chosen) < count:
        candidate = int(np.argmin(closest))
        chosen.append(candidate)
        closest = np.maximum(closest, vectors @ vectors[candidate])
    return chosen


@dataclass(frozen=True)
class ClusterPlan:
    cluster_id: int
    members: tuple[int, ...]
    representatives: tuple[int, ...]


@dataclass
class ClusterDecision:
    inherited: list[Classification] = field(default_factory=list)
    individual: list[int] = field(default_factory=list)


def evaluate_cluster(
    plan: ClusterPlan,
    matrix: np.ndarray,
    classified: dict[int, Classification],
    config: ClassificationConfig,
) -> ClusterDecision:
    """Valida un grupo con sus representantes y reparte a sus miembros pendientes."""
    position = {image_id: index for index, image_id in enumerate(plan.members)}
    pending = [image_id for image_id in plan.members if image_id not in classified]
    voters = [classified[image_id] for image_id in plan.representatives if image_id in classified]
    if len(voters) < _MIN_VOTERS:
        return ClusterDecision(individual=pending)

    category, votes = Counter(voter.category for voter in voters).most_common(1)[0]
    agreement = votes / len(voters)
    if agreement < config.min_agreement:
        return ClusterDecision(individual=pending)

    centroid = matrix.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarity = matrix @ centroid
    threshold = float(similarity.mean() - config.ambiguity_std * similarity.std())

    agreeing = [voter for voter in voters if voter.category == category]
    agreeing_vectors = matrix[[position[voter.image_id] for voter in agreeing]]
    decision = ClusterDecision()
    for image_id in pending:
        row = position[image_id]
        if similarity[row] < threshold:
            decision.individual.append(image_id)
            continue
        nearest = agreeing[int(np.argmax(agreeing_vectors @ matrix[row]))]
        decision.inherited.append(Classification(
            image_id=image_id,
            category=category,
            subcategory=nearest.subcategory,
            tags=nearest.tags,
            confidence=nearest.confidence * agreement,
            model_version=nearest.model_version,
            source=SOURCE_CLUSTER,
        ))
    return decision


class Categorizer:
    def __init__(
        self,
        repository: Repository,
        vectors: VectorIndex,
        classifier: ImageClassifier,
        config: ClassificationConfig,
        gemma_config: GemmaConfig,
    ) -> None:
        self._repository = repository
        self._vectors = vectors
        self._classifier = classifier
        self._config = config
        self._gemma_config = gemma_config

    def run(self, job: JobContext) -> dict[str, int]:
        ids = self._repository.embedded_active_ids()
        records = self._repository.get_images(ids)
        classified = self._repository.get_classifications(ids)
        pending = [image_id for image_id in ids if image_id not in classified]
        if not pending:
            return {"analyzed": 0, "inherited": 0, "failed": 0}

        categories = self._repository.list_categories()
        run = _ClassificationRun(
            self._repository, self._classifier, self._gemma_config, records, categories,
            classified, job,
        )
        if not self._config.use_clusters:
            job.advance(0, len(pending))
            run.analyze(pending)
            return run.summary()

        plans, noise = self._plan_clusters(ids, records, classified)
        representatives = [
            image_id
            for plan in plans
            for image_id in plan.representatives
            if image_id not in classified
        ]
        job.advance(0, len(representatives) + len(noise))
        run.analyze(representatives)
        if job.cancelled:
            return run.summary()

        individual = list(noise)
        for plan in plans:
            matrix = self._vectors.matrix(plan.members)
            decision = evaluate_cluster(plan, matrix, classified, self._config)
            run.inherit(decision.inherited)
            individual.extend(decision.individual)
        job.advance(run.attempted, run.attempted + len(individual))
        run.analyze(individual)
        return run.summary()

    def _plan_clusters(
        self,
        ids: Sequence[int],
        records: dict[int, ImageRecord],
        classified: dict[int, Classification],
    ) -> tuple[list[ClusterPlan], list[int]]:
        groups: dict[int, list[int]] = {}
        noise: list[int] = []
        for image_id in ids:
            cluster = records[image_id].cluster_id
            if cluster is not None:
                groups.setdefault(cluster, []).append(image_id)
            elif image_id not in classified:
                noise.append(image_id)

        plans: list[ClusterPlan] = []
        for cluster, members in groups.items():
            if all(image_id in classified for image_id in members):
                continue
            matrix = self._vectors.matrix(members)
            count = representative_count(len(members), self._config)
            positions = select_representatives(matrix, count)
            plans.append(ClusterPlan(
                cluster, tuple(members), tuple(members[index] for index in positions)
            ))
        return plans, noise


class _ClassificationRun:
    """Estado de una pasada: errores consecutivos, contadores y progreso."""

    def __init__(
        self,
        repository: Repository,
        classifier: ImageClassifier,
        gemma_config: GemmaConfig,
        records: dict[int, ImageRecord],
        categories: Sequence[Category],
        classified: dict[int, Classification],
        job: JobContext,
    ) -> None:
        self._repository = repository
        self._classifier = classifier
        self._gemma_config = gemma_config
        self._records = records
        self._categories = categories
        self._classified = classified
        self._job = job
        self._names = SubcategoryNames(repository.subcategory_names())
        self._consecutive_failures = 0
        self.analyzed = 0
        self.inherited = 0
        self.failed = 0

    @property
    def attempted(self) -> int:
        return self.analyzed + self.failed

    def summary(self) -> dict[str, int]:
        return {"analyzed": self.analyzed, "inherited": self.inherited, "failed": self.failed}

    def analyze(self, ids: Sequence[int]) -> None:
        for image_id in ids:
            if self._job.cancelled:
                return
            self._analyze_one(image_id)
            self._job.advance(self.attempted)

    def _analyze_one(self, image_id: int) -> None:
        record = self._records[image_id]
        try:
            analysis = self._classifier.analyze(Path(record.path), self._categories)
        except GemmaError as error:
            self.failed += 1
            self._consecutive_failures += 1
            self._repository.record_error(image_id, str(error))
            logger.warning("%s: %s", record.path, error)
            if self._consecutive_failures >= self._gemma_config.max_consecutive_failures:
                raise GemmaError(
                    f"{self._consecutive_failures} fallos seguidos. Último: {error}"
                ) from error
            return
        except (OSError, ValueError, UnidentifiedImageError) as error:
            self.failed += 1
            self._repository.set_status([image_id], STATUS_ERROR, f"No se pudo leer: {error}")
            logger.warning("%s: %s", record.path, error)
            return

        self._consecutive_failures = 0
        classification = Classification(
            image_id=image_id,
            category=analysis.category,
            subcategory=self._names.canonical(analysis.category, analysis.subcategory),
            description=analysis.description,
            tags=analysis.tags,
            ocr_text=analysis.ocr_text,
            confidence=analysis.confidence,
            model_version=self._classifier.version,
            source=SOURCE_DIRECT,
        )
        self._repository.save_classifications([classification])
        self._classified[image_id] = classification
        self.analyzed += 1

    def inherit(self, items: Sequence[Classification]) -> None:
        if not items:
            return
        self._repository.save_classifications(items)
        self._classified.update({item.image_id: item for item in items})
        self.inherited += len(items)
