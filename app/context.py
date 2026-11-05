"""Componentes compartidos de la aplicación.

Se construyen en un solo lugar para que las pruebas puedan sustituir los modelos
por dobles sin GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.config import AppConfig, GemmaConfig
from app.core.thumbnails import configure_pillow
from app.database.repository import Repository
from app.database.vector_index import VectorIndex
from app.engine.categorizer import ImageClassifier
from app.engine.indexing import ImageEncoderProtocol, IndexingPipeline, reconcile_vectors
from app.models.embedding_model import ImageEncoder, QueryEncoder
from app.models.gemma_classifier import GemmaClassifier
from app.services.jobs import JobRunner
from app.services.organization import OrganizationService
from app.services.search_service import QueryEncoderProtocol, SearchService


@dataclass
class AppContext:
    config: AppConfig
    repository: Repository
    vectors: VectorIndex
    jobs: JobRunner
    pipeline: IndexingPipeline
    search: SearchService
    organization: OrganizationService
    classifier: ImageClassifier


def build_context(
    config: AppConfig,
    image_encoder: ImageEncoderProtocol | None = None,
    query_encoder: QueryEncoderProtocol | None = None,
    classifier: ImageClassifier | None = None,
) -> AppContext:
    configure_pillow(config.scanner.max_image_pixels)
    repository = Repository(config.database_path)
    vectors = VectorIndex(config.vector_index_path, config.embedding.dimension)
    reconcile_vectors(repository, vectors)

    gemma = classifier or GemmaClassifier(
        _resolved_gemma(config),
        config.resolve(config.paths.llama_server),
        config.data_dir / "llama-server.log",
    )
    pipeline = IndexingPipeline(
        config=config,
        repository=repository,
        vectors=vectors,
        image_encoder=image_encoder or ImageEncoder(config),
        classifier=gemma,
    )
    organization = OrganizationService(repository, config.organization)
    organization.recover_interrupted()
    return AppContext(
        config=config,
        repository=repository,
        vectors=vectors,
        jobs=JobRunner(),
        pipeline=pipeline,
        search=SearchService(
            repository, vectors, query_encoder or QueryEncoder(config), config.search
        ),
        organization=organization,
        classifier=gemma,
    )


def _resolved_gemma(config: AppConfig) -> GemmaConfig:
    return replace(
        config.gemma,
        model_path=config.resolve(config.gemma.model_path),
        mmproj_path=config.resolve(config.gemma.mmproj_path),
    )
