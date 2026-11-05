from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import AppConfig
from app.database.models import ScannedFile
from app.database.repository import Repository
from app.database.vector_index import VectorIndex
from app.engine.indexing import reconcile_vectors


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(4, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


@pytest.fixture
def index(tmp_path: Path) -> VectorIndex:
    return VectorIndex(tmp_path / "faiss.index", 4)


def test_search_returns_closest_first(index: VectorIndex) -> None:
    index.add([10, 20], np.stack([unit(1, 0), unit(0, 1)]))

    results = index.search(unit(1, 0.1), 2)

    assert [identifier for identifier, _ in results] == [10, 20]
    assert results[0][1] > results[1][1]


def test_search_on_empty_index(index: VectorIndex) -> None:
    assert index.search(unit(1), 5) == []


def test_limit_larger_than_index_is_safe(index: VectorIndex) -> None:
    index.add([1], unit(1).reshape(1, -1))

    assert len(index.search(unit(1), 50)) == 1


def test_adding_an_existing_id_replaces_it(index: VectorIndex) -> None:
    index.add([1], unit(1, 0).reshape(1, -1))
    index.add([1], unit(0, 1).reshape(1, -1))

    assert len(index) == 1
    assert index.search(unit(0, 1), 1)[0][1] == pytest.approx(1.0)


def test_remove_and_get(index: VectorIndex) -> None:
    index.add([1, 2], np.stack([unit(1), unit(0, 1)]))

    index.remove([1])

    assert index.get(1) is None
    assert index.get(2) == pytest.approx(unit(0, 1))
    assert index.ids().tolist() == [2]


def test_matrix_preserves_requested_order(index: VectorIndex) -> None:
    index.add([1, 2, 3], np.stack([unit(1), unit(0, 1), unit(0, 0, 1)]))

    matrix = index.matrix([3, 1])

    assert matrix[0] == pytest.approx(unit(0, 0, 1))
    assert matrix[1] == pytest.approx(unit(1))


def test_index_survives_reopening_on_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "índice de años" / "faiss.index"
    first = VectorIndex(path, 4)
    first.add([7], unit(1).reshape(1, -1))
    first.save()

    second = VectorIndex(path, 4)

    assert second.ids().tolist() == [7]
    assert second.search(unit(1), 1)[0][0] == 7


def test_wrong_dimension_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "faiss.index"
    VectorIndex(path, 4).save()

    with pytest.raises(ValueError, match="dimensiones"):
        VectorIndex(path, 8)


def _insert(repository: Repository, name: str) -> int:
    return repository.insert_image(ScannedFile(f"C:/fotos/{name}", name * 8, 10, 1.0, 10, 10))


def test_reconcile_removes_vectors_without_a_record(
    config: AppConfig, index: VectorIndex
) -> None:
    repository = Repository(config.database_path)
    image = _insert(repository, "a")
    repository.save_embeddings([image], "modelo")
    index.add([image, 999], np.stack([unit(1), unit(0, 1)]))

    reconcile_vectors(repository, index)

    assert index.ids().tolist() == [image]


def test_reconcile_requeues_records_without_a_vector(
    config: AppConfig, index: VectorIndex
) -> None:
    repository = Repository(config.database_path)
    image = _insert(repository, "a")
    repository.save_embeddings([image], "modelo")

    reconcile_vectors(repository, index)

    assert repository.ids_pending_embedding() == [image]
