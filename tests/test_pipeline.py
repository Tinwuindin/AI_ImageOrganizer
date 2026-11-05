from __future__ import annotations

import shutil
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.context import AppContext, build_context
from app.core.config import AppConfig, ClusteringConfig
from app.database.models import (
    STATUS_CLASSIFIED,
    STATUS_DUPLICATE,
    STATUS_ERROR,
)
from app.engine.clustering import NOISE, cluster_vectors
from app.engine.duplicate_detector import choose_canonical, promote_copy
from app.models.embedding_model import GpuMemoryError
from app.services.jobs import (
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    JobBusyError,
    JobContext,
    JobRunner,
)
from tests.conftest import FakeClassifier, FakeImageEncoder, FakeQueryEncoder, make_image

Fakes = tuple[FakeImageEncoder, FakeQueryEncoder, FakeClassifier]

RED = (220, 40, 40)
GREEN = (40, 200, 40)
BLUE = (40, 40, 220)


def build_library(root: Path, per_color: int = 6) -> None:
    for name, color in (("rojo", RED), ("verde", GREEN), ("azul", BLUE)):
        for index in range(per_color):
            make_image(root / name / f"{name}_{index}.png", color, (48, 48), seed=index)


def run(context: AppContext, root: Path) -> JobRunner:
    runner = JobRunner()
    context.pipeline.run(JobContext(runner), [root])
    return runner


def test_full_analysis(context: AppContext, tmp_path: Path, fakes: Fakes) -> None:
    root = tmp_path / "fotos"
    build_library(root)
    shutil.copy(root / "rojo" / "rojo_0.png", root / "copia.png")
    (root / "dañada.jpg").write_bytes(b"nada")

    runner = run(context, root)

    stats = context.repository.stats()
    assert stats["statuses"] == {STATUS_CLASSIFIED: 18, STATUS_DUPLICATE: 1, STATUS_ERROR: 1}
    assert len(context.vectors) == 18
    assert stats["clusters"] == 3
    assert stats["duplicate_groups"] == 1
    assert context.config.vector_index_path.exists()
    tree = {node["name"]: node["total"] for node in context.repository.category_tree()}
    assert tree["Comida y bebida"] == 6
    assert tree["Paisajes y naturaleza"] == 6
    assert tree["Capturas de pantalla"] == 6
    summary = runner.snapshot()["summary"]
    assert summary["scan"]["added"] == 20
    assert summary["clusters"] == 3
    # Gemma no vio las 18 imágenes: los grupos redujeron las llamadas.
    assert 0 < len(fakes[2].analyzed) <= 18


def test_only_one_copy_reaches_the_models(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    make_image(root / "a.png", RED, (32, 32), seed=1)
    for index in range(4):
        shutil.copy(root / "a.png", root / f"copia{index}.png")

    run(context, root)

    assert fakes[0].encoded == 1
    assert len(context.vectors) == 1


def test_models_take_turns_on_the_gpu(
    config: AppConfig, tmp_path: Path, fakes: Fakes
) -> None:
    image_encoder, query_encoder, _ = fakes
    root = tmp_path / "fotos"
    build_library(root, per_color=2)

    class WatchingClassifier(FakeClassifier):
        encoder_was_loaded = False

        def analyze(self, image_path, categories):
            self.encoder_was_loaded |= image_encoder.loaded
            return super().analyze(image_path, categories)

    classifier = WatchingClassifier()
    context = build_context(config, image_encoder, query_encoder, classifier)

    run(context, root)

    assert classifier.analyzed
    assert not classifier.encoder_was_loaded
    assert not image_encoder.loaded and not classifier.loaded


def test_models_are_released_when_the_analysis_fails(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    image_encoder, _, classifier = fakes
    classifier.loaded = True
    classifier.always_fail = True

    with pytest.raises(Exception, match="fallos seguidos"):
        run_with_low_failure_limit(context, root)

    assert not image_encoder.loaded
    assert not classifier.loaded


def run_with_low_failure_limit(context: AppContext, root: Path) -> None:
    gemma = replace(context.config.gemma, max_consecutive_failures=2)
    config = replace(context.config, gemma=gemma)
    context.pipeline._config = config
    run(context, root)


def test_cancelled_analysis_resumes_where_it_stopped(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=8)
    image_encoder, _, classifier = fakes
    runner = JobRunner()
    job = JobContext(runner)
    original = image_encoder.encode

    def encode_then_cancel(images):
        vectors = original(images)
        if image_encoder.encoded >= 8:
            job.cancel.set()
        return vectors

    image_encoder.encode = encode_then_cancel

    context.pipeline.run(job, [root])

    processed = image_encoder.encoded
    assert 0 < processed < 24
    assert len(context.vectors) == processed
    assert classifier.analyzed == []

    image_encoder.encode = original
    run(context, root)

    assert image_encoder.encoded == 24, "No debe repetir los vectores ya guardados."
    assert context.repository.stats()["statuses"] == {STATUS_CLASSIFIED: 24}


def test_interruption_between_faiss_and_sqlite_is_repaired(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    run(context, root)
    image_id = context.repository.embedded_active_ids()[0]
    context.repository.delete_embeddings([image_id])  # Como si SQLite no hubiera confirmado.

    rebuilt = build_context(context.config, *fakes)

    assert len(rebuilt.vectors) == 5
    run(rebuilt, root)
    assert len(rebuilt.vectors) == 6


def test_second_run_only_processes_new_images(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=5)
    run(context, root)
    encoded = fakes[0].encoded

    make_image(root / "rojo" / "nueva.png", RED, (48, 48), seed=99)
    run(context, root)

    assert fakes[0].encoded == encoded + 1
    assert context.repository.stats()["statuses"] == {STATUS_CLASSIFIED: 16}


def test_modified_image_is_analyzed_again(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=3)
    run(context, root)
    target = root / "rojo" / "rojo_0.png"

    make_image(target, BLUE, (48, 48), seed=50)
    run(context, root)

    record = context.repository.all_images()[str(target)]
    assert context.repository.get_classification(record.id).category == "Capturas de pantalla"
    assert len(context.vectors) == 9


def test_duplicate_appearing_later_keeps_the_analyzed_copy(
    context: AppContext, tmp_path: Path
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    run(context, root)
    original = context.repository.all_images()[str(root / "azul" / "azul_0.png")]

    shutil.copy(root / "azul" / "azul_0.png", root / "a.png")
    run(context, root)

    assert context.repository.get_image(original.id).status == STATUS_CLASSIFIED
    copy = context.repository.all_images()[str(root / "a.png")]
    assert copy.status == STATUS_DUPLICATE
    assert context.vectors.get(copy.id) is None


def test_duplicate_group_disappears_when_a_copy_is_removed(
    context: AppContext, tmp_path: Path
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    copy = root / "copia.png"
    shutil.copy(root / "azul" / "azul_0.png", copy)
    run(context, root)
    assert context.repository.stats()["duplicate_groups"] == 1

    copy.unlink()
    run(context, root)

    assert context.repository.stats()["duplicate_groups"] == 0
    assert STATUS_DUPLICATE not in context.repository.stats()["statuses"]


def test_keeping_another_copy_moves_analysis_and_vector(
    context: AppContext, tmp_path: Path
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    shutil.copy(root / "azul" / "azul_0.png", root / "preferida.png")
    run(context, root)
    groups, _ = context.repository.duplicate_groups(10, 0)
    group = groups[0]
    previous = group["canonical_image_id"]
    other = next(m["id"] for m in group["members"] if m["id"] != previous)
    classification = context.repository.get_classification(previous)

    assert promote_copy(context.repository, context.vectors, group["id"], other)

    assert context.repository.get_image(other).status == STATUS_CLASSIFIED
    assert context.repository.get_image(previous).status == STATUS_DUPLICATE
    assert context.repository.get_classification(other).category == classification.category
    assert context.repository.get_classification(previous) is None
    assert context.vectors.get(other) is not None
    assert context.vectors.get(previous) is None


def test_a_copy_from_another_group_cannot_be_kept(
    context: AppContext, tmp_path: Path
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)
    shutil.copy(root / "azul" / "azul_0.png", root / "copia.png")
    run(context, root)
    group = context.repository.duplicate_groups(10, 0)[0][0]
    unrelated = context.repository.all_images()[str(root / "rojo" / "rojo_0.png")]

    assert not promote_copy(context.repository, context.vectors, group["id"], unrelated.id)


def test_canonical_prefers_the_copy_with_most_analysis(
    context: AppContext, tmp_path: Path
) -> None:
    root = tmp_path / "fotos"
    make_image(root / "ruta" / "muy" / "larga" / "a.png", RED, (32, 32), seed=1)
    run(context, root)
    analyzed = next(iter(context.repository.all_images().values()))
    shutil.copy(analyzed.path, root / "b.png")
    run(context, root)

    assert choose_canonical(list(context.repository.all_images().values())).id == analyzed.id


def test_clustering_separates_distinct_groups() -> None:
    generator = np.random.default_rng(0)
    centers = np.eye(16, dtype=np.float32)[:3]
    vectors = np.concatenate([
        center + generator.normal(0, 0.03, (30, 16)) for center in centers
    ]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    labels = cluster_vectors(vectors, ClusteringConfig(pca_components=5, selection_method="eom"))

    assert len(set(labels) - {NOISE}) == 3
    assert (labels[:30] == labels[0]).mean() > 0.8


def test_clustering_handles_tiny_libraries() -> None:
    vectors = np.eye(16, dtype=np.float32)[:2]

    assert cluster_vectors(vectors, ClusteringConfig()).tolist() == [NOISE, NOISE]


def test_job_runner_reports_progress_and_blocks_a_second_job() -> None:
    runner = JobRunner()
    release = threading.Event()

    def target(job: JobContext) -> dict:
        job.stage("uno", 10)
        job.advance(4)
        release.wait(5)
        return {"resultado": 1}

    runner.start("prueba", target)
    with pytest.raises(JobBusyError):
        runner.start("otra", target)
    release.set()
    runner.wait(5)

    snapshot = runner.snapshot()
    assert snapshot["state"] == STATE_COMPLETED
    assert snapshot["summary"] == {"resultado": 1}


def test_job_runner_reports_failures_and_cancellation() -> None:
    runner = JobRunner()

    def failing(_: JobContext) -> None:
        raise RuntimeError("sin espacio en disco")

    runner.start("fallida", failing)
    runner.wait(5)
    assert runner.snapshot()["state"] == STATE_FAILED
    assert runner.snapshot()["error"] == "sin espacio en disco"

    started = threading.Event()

    def waiting(job: JobContext) -> None:
        started.set()
        job.cancel.wait(5)

    runner.start("cancelable", waiting)
    started.wait(5)
    assert runner.cancel()
    runner.wait(5)
    assert runner.snapshot()["state"] == STATE_CANCELLED


def test_an_image_that_breaks_the_encoder_is_isolated(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=3)
    make_image(root / "extrana.png", (1, 2, 3), (48, 48))
    image_encoder = fakes[0]
    original = image_encoder.encode

    def fragile_encode(images):
        if any(tuple(image.getpixel((0, 0))) == (1, 2, 3) for image in images):
            raise ValueError("relación de aspecto no admitida")
        return original(images)

    image_encoder.encode = fragile_encode

    run(context, root)

    statuses = context.repository.stats()["statuses"]
    assert statuses == {STATUS_CLASSIFIED: 9, STATUS_ERROR: 1}
    broken = context.repository.all_images()[str(root / "extrana.png")]
    assert "aspecto" in broken.error


def test_running_out_of_vram_stops_instead_of_failing_every_image(
    context: AppContext, tmp_path: Path, fakes: Fakes
) -> None:
    root = tmp_path / "fotos"
    build_library(root, per_color=2)

    def no_memory(images):
        raise GpuMemoryError("sin VRAM")

    fakes[0].encode = no_memory

    with pytest.raises(GpuMemoryError):
        run(context, root)

    assert STATUS_ERROR not in context.repository.stats()["statuses"]
