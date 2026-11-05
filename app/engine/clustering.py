"""Agrupación de imágenes parecidas.

Se reduce la dimensión con PCA antes de HDBSCAN: la densidad se estima mejor con
pocas dimensiones, y 10,000 vectores se agrupan en segundos. Las imágenes que no
encajan en ningún grupo reciben la etiqueta ``NOISE`` y se analizan una por una.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from app.core.config import ClusteringConfig

NOISE = -1


def cluster_vectors(vectors: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Devuelve una etiqueta de grupo por vector; ``NOISE`` si no pertenece a ninguno."""
    count = len(vectors)
    if count < max(config.min_cluster_size, 3):
        return np.full(count, NOISE, dtype=np.int64)

    components = min(config.pca_components, count - 1, vectors.shape[1])
    reduced = PCA(n_components=components, random_state=0).fit_transform(vectors)
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-12)

    labels = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        cluster_selection_method=config.selection_method,
        copy=True,
    ).fit_predict(reduced)
    return labels.astype(np.int64)
