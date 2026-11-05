"""Búsqueda por lenguaje natural.

Combina dos señales con fusión de rangos recíprocos (RRF):

- Semántica: la consulta y las imágenes viven en el mismo espacio de EmbeddingGemma 2.
- Texto: coincidencias en descripción, etiquetas, texto visible y nombre de archivo.

RRF usa las posiciones y no las puntuaciones, porque las dos escalas no son
comparables. La similitud texto-imagen, además, se concentra en un rango estrecho
(cerca de 0.55 para lo no relacionado y 0.70 para lo relevante), así que un umbral
fijo descartaría buenos resultados o dejaría pasar malos.

Tampoco se recorta la lista con un umbral relativo (media + N desviaciones): cuando
un tema domina la biblioteca, algo normal en una fototeca personal, descartaba hasta
el 85 % de las imágenes relevantes. El resultado es un ranking: lo mejor va primero
y la búsqueda no distingue que algo no exista, solo muestra lo más parecido.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from app.core.config import SearchConfig
from app.database.repository import Repository
from app.database.vector_index import VectorIndex

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
_MIN_TERM_LENGTH = 2

# Palabras que no ayudan a encontrar una imagen por su contenido.
_STOPWORDS = frozenset("""
    a al ante bajo con contra de del desde durante en entre hacia hasta mi mis para por
    sin sobre su sus tu tus un una unas unos el la las lo los y o e u que como donde cuando
    es son fue estaba este esta estos estas ese esa esos esas aquel aquella
    foto fotos fotografia fotografias imagen imagenes
    quiero busca buscar buscame muestra muestrame encuentra encontrar ver tengo tenemos
""".split())

_ACCENTS = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")


class QueryEncoderProtocol(Protocol):
    def encode(self, query: str) -> np.ndarray: ...


def extract_terms(query: str) -> list[str]:
    """Palabras significativas de la consulta, sin repetir."""
    terms: list[str] = []
    for word in _WORD.findall(query.lower()):
        if len(word) < _MIN_TERM_LENGTH or word.translate(_ACCENTS) in _STOPWORDS:
            continue
        if word not in terms:
            terms.append(word)
    return terms


def build_match_expression(terms: Sequence[str]) -> str:
    """Expresión FTS5. El prefijo cubre plurales simples: ``perro`` halla ``perros``."""
    prefixes = []
    for term in terms:
        stem = term[:-1] if len(term) > 3 and term.endswith("s") else term
        prefixes.append(f'"{stem}"*')
    return " OR ".join(prefixes)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))


class SearchService:
    def __init__(
        self,
        repository: Repository,
        vectors: VectorIndex,
        encoder: QueryEncoderProtocol,
        config: SearchConfig,
    ) -> None:
        self._repository = repository
        self._vectors = vectors
        self._encoder = encoder
        self._config = config

    def search(
        self,
        query: str,
        limit: int | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        limit = limit or self._config.default_limit
        semantic, semantic_available = self._semantic(query)
        lexical = self._lexical(query)
        ranked = reciprocal_rank_fusion(
            [[image_id for image_id, _ in semantic], lexical], self._config.rrf_k
        )

        similarities = dict(semantic)
        lexical_ids = set(lexical)
        items = self._repository.library_items(ranked)
        results: list[dict[str, Any]] = []
        for image_id in ranked:
            item = items.get(image_id)
            if item is None or (category and item["category"] != category):
                continue
            results.append({
                **item,
                "similarity": similarities.get(image_id),
                "text_match": image_id in lexical_ids,
            })
            if len(results) >= limit:
                break
        return {"query": query, "semantic": semantic_available, "results": results}

    def _semantic(self, query: str) -> tuple[list[tuple[int, float]], bool]:
        if len(self._vectors) == 0:
            return [], False
        try:
            vector = self._encoder.encode(query)
        except Exception:  # Sin encoder sigue funcionando la búsqueda por texto.
            logger.exception("No se pudo codificar la consulta")
            return [], False
        return self._vectors.search(vector, self._config.candidates), True

    def _lexical(self, query: str) -> list[int]:
        terms = extract_terms(query)
        if not terms:
            return []
        return self._repository.search_text(
            build_match_expression(terms), self._config.candidates
        )
