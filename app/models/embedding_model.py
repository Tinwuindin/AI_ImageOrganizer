"""EmbeddingGemma 2: vectores de imagen y texto en un mismo espacio.

Se usa de dos formas, separadas a propósito:

- ``ImageEncoder`` corre en la GPU con los encoders de texto y visión, sin el de
  audio (unos 440 M de parámetros), y solo se carga durante la etapa de embeddings.
- ``QueryEncoder`` corre en la CPU con el encoder de texto. Una consulta tarda
  alrededor de 100 ms, así que la búsqueda funciona aunque Gemma ocupe la GPU.

Medido en una RTX 4060: el tamaño de lote no cambia el rendimiento (unas 8 img/s)
y sí la VRAM, por lo que conviene un lote pequeño. ``float16`` no es compatible
con este modelo; el formato recomendado en GPU es ``bfloat16``.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import AppConfig, EmbeddingConfig

if TYPE_CHECKING:
    from PIL import Image
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Todo el análisis es local: nada debe consultar Hugging Face.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class EmbeddingError(RuntimeError):
    """El modelo de embeddings no está disponible."""


class GpuMemoryError(RuntimeError):
    """La VRAM libre no alcanza ni para una imagen."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza en ``float32``: en ``bfloat16`` la norma queda algo lejos de 1."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _model_path(config: AppConfig) -> str:
    path = config.resolve(config.embedding.model_path)
    if not (path / "model.safetensors").exists():
        raise EmbeddingError(f"No se encontró {path}. Ejecuta: python scripts/download_models.py")
    return str(path)


class ImageEncoder:
    """Encoder de imágenes para la etapa de indexación."""

    def __init__(self, config: AppConfig) -> None:
        self._app_config = config
        self._config: EmbeddingConfig = config.embedding
        self._batch_size = config.embedding.batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        path = _model_path(self._app_config)
        use_gpu = torch.cuda.is_available()
        logger.info("Cargando EmbeddingGemma 2 en %s", "GPU" if use_gpu else "CPU")
        self._model = SentenceTransformer(
            path,
            device="cuda" if use_gpu else "cpu",
            config_kwargs={"audio_config": None},
            model_kwargs={"dtype": torch.bfloat16 if use_gpu else torch.float32},
        )

    def unload(self) -> None:
        if self._model is None:
            return
        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Devuelve vectores normalizados. Reduce el lote si se agota la VRAM."""
        import torch

        self.load()
        assert self._model is not None
        while True:
            try:
                with torch.inference_mode():
                    vectors = self._model.encode(
                        [{"image": image} for image in images],
                        batch_size=self._batch_size,
                        processing_kwargs={"image": {"max_soft_tokens": self._config.image_tokens}},
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                return _normalize(vectors)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                if self._batch_size == 1:
                    raise GpuMemoryError(
                        "La VRAM libre no alcanza: cierra otras aplicaciones que usen la GPU."
                    ) from error
                self._batch_size = max(1, self._batch_size // 2)
                logger.warning("VRAM agotada: el lote baja a %d.", self._batch_size)


class QueryEncoder:
    """Encoder de consultas de texto, residente en la CPU."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _load(self) -> SentenceTransformer:
        import torch
        from sentence_transformers import SentenceTransformer

        path = _model_path(self._config)
        torch.set_num_threads(self._config.embedding.query_threads)
        logger.info("Cargando el encoder de consultas en CPU")
        return SentenceTransformer(
            path,
            device="cpu",
            config_kwargs={"audio_config": None, "vision_config": None},
            model_kwargs={"dtype": torch.float32},
        )

    def encode(self, query: str) -> np.ndarray:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            vector = self._model.encode(
                [query],
                prompt_name="query",
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return _normalize(vector)[0]
