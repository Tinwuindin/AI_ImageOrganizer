from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.core.config import AppConfig, ClassificationConfig
from app.database.models import (
    SOURCE_CLUSTER,
    SOURCE_DIRECT,
    Classification,
    ScannedFile,
)
from app.database.repository import Repository
from app.database.vector_index import VectorIndex
from app.engine.categorizer import (
    Categorizer,
    ClusterPlan,
    SubcategoryNames,
    evaluate_cluster,
    representative_count,
    select_representatives,
    subcategory_key,
)
from app.models.gemma_classifier import GemmaError, ImageAnalysis
from app.services.jobs import JobContext, JobRunner
from tests.conftest import FakeClassifier, make_image

CONFIG = ClassificationConfig(
    min_representatives=3, max_representatives=12, min_agreement=0.8, ambiguity_std=1.5
)


def spread(count: int, seed: int = 0, center_axis: int = 0, noise: float = 0.05) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vectors = generator.normal(0, noise, (count, 16)).astype(np.float32)
    vectors[:, center_axis] += 1.0
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, 1), (2, 2), (3, 3), (4, 3), (16, 4), (100, 10), (10_000, 12)],
)
def test_representative_count_grows_slowly(size: int, expected: int) -> None:
    assert representative_count(size, CONFIG) == expected


def test_representatives_are_distinct_and_cover_the_spread() -> None:
    near = spread(20, seed=1)
    far = spread(5, seed=2, center_axis=1)
    vectors = np.concatenate([near, far])

    chosen = select_representatives(vectors, 4)

    assert len(set(chosen)) == 4
    assert any(index >= 20 for index in chosen), "Debe incluir la zona minoritaria del grupo."


def plan_for(size: int, representatives: tuple[int, ...]) -> ClusterPlan:
    return ClusterPlan(0, tuple(range(size)), representatives)


def vote(image_id: int, category: str, subcategory: str = "Tema") -> Classification:
    return Classification(image_id, category, subcategory, tags=("x",), confidence=0.9)


def test_agreeing_group_is_inherited() -> None:
    matrix = spread(10)
    classified = {index: vote(index, "Comida y bebida") for index in (0, 1, 2)}

    decision = evaluate_cluster(plan_for(10, (0, 1, 2)), matrix, classified, CONFIG)

    assert len(decision.inherited) + len(decision.individual) == 7
    assert decision.inherited
    inherited = decision.inherited[0]
    assert inherited.category == "Comida y bebida"
    assert inherited.source == SOURCE_CLUSTER
    assert inherited.description == "", "Una descripción ajena sería falsa para esta imagen."
    assert inherited.tags == ("x",)
    assert inherited.confidence == pytest.approx(0.9)


def test_inherits_from_the_nearest_representative() -> None:
    first = spread(5, seed=1, noise=0.01)
    second = spread(5, seed=2, center_axis=1, noise=0.01)
    matrix = np.concatenate([first, second])
    classified = {0: vote(0, "Comida y bebida", "Pizza"), 5: vote(5, "Comida y bebida", "Pasta")}
    config = replace(CONFIG, ambiguity_std=10)

    decision = evaluate_cluster(plan_for(10, (0, 5)), matrix, classified, config)

    subcategories = {item.image_id: item.subcategory for item in decision.inherited}
    assert subcategories[1] == "Pizza"
    assert subcategories[6] == "Pasta"


def test_group_without_agreement_goes_to_individual_analysis() -> None:
    matrix = spread(6)
    classified = {0: vote(0, "Personas"), 1: vote(1, "Vehículos"), 2: vote(2, "Otros")}

    decision = evaluate_cluster(plan_for(6, (0, 1, 2)), matrix, classified, CONFIG)

    assert decision.inherited == []
    assert sorted(decision.individual) == [3, 4, 5]


def test_outlier_member_is_analyzed_individually() -> None:
    matrix = np.concatenate([spread(19, noise=0.01), spread(1, seed=9, center_axis=2)])
    classified = {index: vote(index, "Personas") for index in (0, 1, 2)}

    decision = evaluate_cluster(plan_for(20, (0, 1, 2)), matrix, classified, CONFIG)

    assert 19 in decision.individual
    assert 19 not in {item.image_id for item in decision.inherited}


def test_failed_representatives_do_not_validate_a_group() -> None:
    matrix = spread(5)
    classified = {0: vote(0, "Personas")}

    decision = evaluate_cluster(plan_for(5, (0, 1, 2)), matrix, classified, CONFIG)

    assert decision.inherited == []
    assert sorted(decision.individual) == [1, 2, 3, 4]


def test_confidence_drops_with_partial_agreement() -> None:
    matrix = spread(12)
    classified = {index: vote(index, "Personas") for index in range(4)}
    classified[4] = vote(4, "Otros")
    config = replace(CONFIG, min_agreement=0.75)

    decision = evaluate_cluster(plan_for(12, (0, 1, 2, 3, 4)), matrix, classified, config)

    assert decision.inherited[0].confidence == pytest.approx(0.9 * 0.8)


@pytest.mark.parametrize(
    ("left", "right"),
    [("Gatos", "gato"), ("Cumpleaños", "cumpleaños"), ("Café", "cafe"), ("  Playa ", "playa")],
)
def test_subcategory_variants_share_a_key(left: str, right: str) -> None:
    assert subcategory_key(left) == subcategory_key(right)


def test_subcategory_names_keep_the_first_spelling_per_category() -> None:
    names = SubcategoryNames([("Mascotas", "Gatos")])

    assert names.canonical("Mascotas", "gato") == "Gatos"
    assert names.canonical("Mascotas", "Perros") == "Perros"
    assert names.canonical("Memes", "gato") == "gato"
    assert names.canonical("Mascotas", "") == ""


class Fixture:
    """Biblioteca con vectores y grupos ya calculados."""

    def __init__(self, tmp_path: Path, config: AppConfig) -> None:
        self.repository = Repository(config.database_path)
        self.vectors = VectorIndex(config.vector_index_path, 16)
        self.classifier = FakeClassifier()
        self.tmp_path = tmp_path
        self.gemma_config = config.gemma
        self.ids: list[int] = []
        self.created = 0

    def add_cluster(self, cluster: int | None, vectors: np.ndarray, color: tuple[int, int, int]):
        ids = []
        for _ in range(len(vectors)):
            self.created += 1
            name = f"grupo{cluster}_{self.created}.png"
            path = make_image(self.tmp_path / "fotos" / name, color=color, size=(32, 32))
            image_id = self.repository.insert_image(
                ScannedFile(str(path), f"hash-{self.created}", 10, 1.0, 32, 32)
            )
            ids.append(image_id)
        self.vectors.add(ids, vectors)
        self.repository.save_embeddings(ids, "modelo")
        self.repository.set_clusters([(image_id, cluster) for image_id in ids])
        self.ids.extend(ids)
        return ids

    def run(self, config: ClassificationConfig = CONFIG) -> tuple[dict[str, int], JobRunner]:
        runner = JobRunner()
        job = JobContext(runner)
        categorizer = Categorizer(
            self.repository, self.vectors, self.classifier, config, self.gemma_config
        )
        return categorizer.run(job), runner


@pytest.fixture
def library(tmp_path: Path, config: AppConfig) -> Fixture:
    return Fixture(tmp_path, config)


def test_clusters_reduce_model_calls(library: Fixture) -> None:
    library.add_cluster(1, spread(100, noise=0.01), (220, 30, 30))

    summary, _ = library.run()

    assert summary["analyzed"] < 30
    assert summary["analyzed"] + summary["inherited"] == 100
    sources = {
        item.source
        for item in library.repository.get_classifications(library.ids).values()
    }
    assert sources == {SOURCE_DIRECT, SOURCE_CLUSTER}


def test_images_outside_any_group_are_analyzed_one_by_one(library: Fixture) -> None:
    library.add_cluster(None, spread(7), (220, 30, 30))

    summary, _ = library.run()

    assert summary == {"analyzed": 7, "inherited": 0, "failed": 0}


def test_full_mode_analyzes_every_image(library: Fixture) -> None:
    library.add_cluster(1, spread(20), (220, 30, 30))

    summary, _ = library.run(replace(CONFIG, use_clusters=False))

    assert summary["analyzed"] == 20


def test_second_run_repeats_nothing(library: Fixture) -> None:
    library.add_cluster(1, spread(30), (220, 30, 30))
    library.add_cluster(None, spread(3), (30, 30, 220))
    library.run()
    calls = len(library.classifier.analyzed)

    summary, _ = library.run()

    assert summary == {"analyzed": 0, "inherited": 0, "failed": 0}
    assert len(library.classifier.analyzed) == calls


def test_classified_members_vote_when_new_images_join_a_group(library: Fixture) -> None:
    ids = library.add_cluster(1, spread(20, seed=3), (220, 30, 30))
    library.run()
    new = library.add_cluster(1, spread(9, seed=4), (220, 30, 30))
    library.repository.set_clusters([(image_id, 1) for image_id in ids + new])
    calls = len(library.classifier.analyzed)

    summary, _ = library.run()

    assert summary["analyzed"] + summary["inherited"] == 9
    assert len(library.classifier.analyzed) - calls < 9


def test_manual_category_is_never_overwritten(library: Fixture) -> None:
    ids = library.add_cluster(1, spread(12), (220, 30, 30))
    library.repository.set_manual_category(ids[0], "Documentos", "Mis papeles")

    library.run()

    kept = library.repository.get_classification(ids[0])
    assert (kept.category, kept.subcategory) == ("Documentos", "Mis papeles")


def test_a_failed_image_stays_pending_for_the_next_run(library: Fixture) -> None:
    ids = library.add_cluster(None, spread(4), (220, 30, 30))
    failing = Path(library.repository.get_image(ids[0]).path).name
    library.classifier.failures.add(failing)

    summary, _ = library.run()

    assert summary == {"analyzed": 3, "inherited": 0, "failed": 1}
    assert library.repository.get_classification(ids[0]) is None
    assert "fallo simulado" in library.repository.get_image(ids[0]).error

    library.classifier.failures.clear()
    summary, _ = library.run()

    assert summary["analyzed"] == 1
    assert library.repository.get_image(ids[0]).error is None


def test_too_many_consecutive_failures_stop_the_analysis(library: Fixture) -> None:
    library.add_cluster(None, spread(30), (220, 30, 30))
    library.classifier.always_fail = True

    with pytest.raises(GemmaError, match="fallos seguidos"):
        library.run()

    assert len(library.classifier.analyzed) == library.gemma_config.max_consecutive_failures


def test_unreadable_file_is_marked_and_skipped(library: Fixture) -> None:
    ids = library.add_cluster(None, spread(3), (220, 30, 30))
    Path(library.repository.get_image(ids[0]).path).unlink()

    class Reader(FakeClassifier):
        def analyze(self, image_path: Path, categories):
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            return super().analyze(image_path, categories)

    library.classifier = Reader()

    summary, _ = library.run()

    assert summary["analyzed"] == 2
    assert summary["failed"] == 1
    assert library.repository.get_image(ids[0]).status == "error"


def test_subcategories_are_unified_across_images(library: Fixture) -> None:
    ids = library.add_cluster(None, spread(2), (220, 30, 30))
    names = [Path(library.repository.get_image(image_id).path).name for image_id in ids]
    library.classifier.overrides = {
        names[0]: ImageAnalysis("Mascotas y animales", "Gatos", "Uno.", (), "", 0.9),
        names[1]: ImageAnalysis("Mascotas y animales", "gato", "Dos.", (), "", 0.9),
    }

    library.run()

    stored = library.repository.get_classifications(ids)
    assert {item.subcategory for item in stored.values()} == {"Gatos"}


def test_progress_total_reflects_the_real_number_of_calls(library: Fixture) -> None:
    library.add_cluster(1, spread(40, noise=0.01), (220, 30, 30))
    library.add_cluster(None, spread(3), (30, 30, 220))

    summary, runner = library.run()

    snapshot = runner.snapshot()
    assert snapshot["done"] == snapshot["total"] == summary["analyzed"]
