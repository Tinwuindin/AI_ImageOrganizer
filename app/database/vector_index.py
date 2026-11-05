"""Índice vectorial en FAISS.

Con unas 10,000 imágenes una búsqueda exacta (producto interno sobre vectores
normalizados, es decir, similitud coseno) responde en milisegundos, así que no
hace falta un índice aproximado.

El archivo se escribe de forma atómica. Si el proceso se interrumpe entre guardar
FAISS y confirmar SQLite, ``Repository`` y este índice pueden diferir;
``reconcile_vectors`` lo corrige al arrancar y antes de cada etapa.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, path: Path, dimension: int) -> None:
        self._path = path
        self._dimension = dimension
        self._lock = threading.RLock()
        self._dirty = False
        self._index = self._load()

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))

    def _load(self) -> faiss.IndexIDMap2:
        if not self._path.exists():
            return self._new_index()
        # La lectura vía bytes admite rutas con caracteres no ASCII en Windows.
        data = np.frombuffer(self._path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(data)
        if index.d != self._dimension:
            raise ValueError(
                f"{self._path.name} tiene vectores de {index.d} dimensiones "
                f"y el modelo produce {self._dimension}."
            )
        return index

    def __len__(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def ids(self) -> np.ndarray:
        with self._lock:
            return faiss.vector_to_array(self._index.id_map).copy()

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        """Agrega vectores; si un identificador ya existe, lo reemplaza."""
        identifiers = np.asarray(ids, dtype=np.int64)
        with self._lock:
            self._index.remove_ids(identifiers)
            self._index.add_with_ids(np.asarray(vectors, dtype=np.float32), identifiers)
            self._dirty = True

    def remove(self, ids: Sequence[int]) -> None:
        if len(ids) == 0:
            return
        with self._lock:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            self._dirty = True

    def search(self, vector: np.ndarray, limit: int) -> list[tuple[int, float]]:
        """Devuelve pares ``(id, similitud)`` de mayor a menor similitud."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            scores, ids = self._index.search(query, min(limit, int(self._index.ntotal)))
        return [
            (int(identifier), float(score))
            for identifier, score in zip(ids[0], scores[0])
            if identifier != -1
        ]

    def get(self, image_id: int) -> np.ndarray | None:
        with self._lock:
            try:
                return self._index.reconstruct(int(image_id))
            except RuntimeError:
                return None

    def matrix(self, ids: Sequence[int]) -> np.ndarray:
        """Vectores de los identificadores indicados, en el mismo orden."""
        with self._lock:
            result = np.empty((len(ids), self._dimension), dtype=np.float32)
            for row, image_id in enumerate(ids):
                result[row] = self._index.reconstruct(int(image_id))
        return result

    def save(self) -> None:
        with self._lock:
            if not self._dirty and self._path.exists():
                return
            data = faiss.serialize_index(self._index)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_bytes(data.tobytes())
            temporary.replace(self._path)
            self._dirty = False
