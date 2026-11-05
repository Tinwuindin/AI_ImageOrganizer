"""Consultas de similitud visual sobre el índice vectorial."""

from __future__ import annotations

from app.database.vector_index import VectorIndex


def similar_images(vectors: VectorIndex, image_id: int, limit: int) -> list[tuple[int, float]]:
    """Imágenes más parecidas a una dada, sin incluirla a ella."""
    vector = vectors.get(image_id)
    if vector is None:
        return []
    matches = vectors.search(vector, limit + 1)
    return [(match, score) for match, score in matches if match != image_id][:limit]
