"""Utilidades de prueba: imágenes reales en disco y modelos simulados sin GPU."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.context import AppContext, build_context
from app.core.config import AppConfig, ClusteringConfig, load_config
from app.database.models import Category
from app.models.gemma_classifier import GemmaError, ImageAnalysis

DIMENSION = 16


def make_image(
    path: Path,
    color: tuple[int, int, int] = (200, 30, 30),
    size: tuple[int, int] = (640, 480),
    exif: Image.Exif | None = None,
    seed: int | None = None,
) -> Path:
    """Escribe una imagen. ``seed`` agrega ruido para que su contenido sea único."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if seed is not None:
        generator = np.random.default_rng(seed)
        noise = generator.integers(-12, 12, (size[1], size[0], 3))
        pixels = np.clip(np.asarray(image, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")
    options = {"exif": exif} if exif is not None else {}
    image.save(path, **options)
    return path


def color_vector(image: Image.Image) -> np.ndarray:
    """Vector determinista: imágenes de colores parecidos quedan cerca."""
    red, green, blue = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255
    digest = hashlib.sha256(np.asarray(image).tobytes()).digest()
    jitter = np.frombuffer(digest[:DIMENSION], dtype=np.uint8).astype(np.float32) / 255 - 0.5
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[:3] = [red, green, blue]
    vector += jitter * 0.02
    return vector / np.linalg.norm(vector)


class FakeImageEncoder:
    model_name = "fake-embedding"
    batch_size = 4

    def __init__(self) -> None:
        self.encoded = 0
        self.loaded = False
        self.fail_on: set[int] = set()

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.encoded += len(images)
        return np.stack([color_vector(image) for image in images])


class FakeQueryEncoder:
    """Traduce palabras de color a la zona del espacio donde están esas imágenes."""

    AXES = {"rojo": 0, "verde": 1, "azul": 2}

    def __init__(self) -> None:
        self.fail = False

    def encode(self, query: str) -> np.ndarray:
        if self.fail:
            raise RuntimeError("encoder no disponible")
        vector = np.zeros(DIMENSION, dtype=np.float32)
        for word, axis in self.AXES.items():
            if word in query.lower():
                vector[axis] = 1.0
        if not vector.any():
            vector[3] = 1.0
        return vector / np.linalg.norm(vector)


class FakeClassifier:
    """Clasifica por color dominante; ``overrides`` cambia el resultado por archivo."""

    version = "fake-gemma"

    def __init__(self) -> None:
        self.analyzed: list[Path] = []
        self.loaded = False
        self.overrides: dict[str, ImageAnalysis] = {}
        self.failures: set[str] = set()
        self.always_fail = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def analyze(self, image_path: Path, categories: Sequence[Category]) -> ImageAnalysis:
        self.analyzed.append(image_path)
        if self.always_fail or image_path.name in self.failures:
            raise GemmaError("fallo simulado")
        if image_path.name in self.overrides:
            return self.overrides[image_path.name]
        with Image.open(image_path) as image:
            red, green, blue = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        if red >= green and red >= blue:
            label = "Comida y bebida", "Tomates", "Una imagen roja"
        elif green >= blue:
            label = "Paisajes y naturaleza", "Bosques", "Una imagen verde"
        else:
            label = "Capturas de pantalla", "Azules", "Una imagen azul"
        return ImageAnalysis(label[0], label[1], label[2], ("color",), "", 0.9)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    base = load_config()
    return replace(
        base,
        root=tmp_path,
        paths=replace(base.paths, data_dir=tmp_path / "data"),
        embedding=replace(base.embedding, dimension=DIMENSION, checkpoint_every=4),
        clustering=ClusteringConfig(
            pca_components=3, min_cluster_size=4, min_samples=2, selection_method="eom"
        ),
    )


@pytest.fixture
def fakes() -> tuple[FakeImageEncoder, FakeQueryEncoder, FakeClassifier]:
    return FakeImageEncoder(), FakeQueryEncoder(), FakeClassifier()


@pytest.fixture
def context(
    config: AppConfig,
    fakes: tuple[FakeImageEncoder, FakeQueryEncoder, FakeClassifier],
) -> AppContext:
    image_encoder, query_encoder, classifier = fakes
    return build_context(config, image_encoder, query_encoder, classifier)
