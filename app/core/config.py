"""Carga de ``config.yaml`` en estructuras tipadas.

Las rutas relativas se interpretan desde la raíz del proyecto. Una clave
desconocida en el YAML es un error: así un descuido al escribir no se convierte
en un ajuste que se ignora en silencio.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(ValueError):
    """El archivo de configuración no es válido."""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data")
    llama_server: Path = Path("tools/llama.cpp/llama-server.exe")


@dataclass(frozen=True)
class ScannerConfig:
    extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif",
    )
    exclude_dirs: tuple[str, ...] = ("$RECYCLE.BIN", "System Volume Information", ".git")
    workers: int = 4
    thumbnail_size: int = 384
    max_image_pixels: int = 500_000_000


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: Path = Path("models/embeddinggemma-2")
    model_name: str = "google/embeddinggemma-2"
    dimension: int = 768
    batch_size: int = 8
    image_max_side: int = 768
    image_tokens: int = 280
    query_threads: int = 4
    checkpoint_every: int = 256


@dataclass(frozen=True)
class GemmaConfig:
    model_path: Path = Path("models/gemma-4-e2b-q4/gemma-4-E2B_q4_0-it.gguf")
    mmproj_path: Path = Path("models/gemma-4-e2b-q4/gemma-4-E2B-it-mmproj.gguf")
    model_version: str = "gemma-4-E2B-it-qat-q4_0"
    context_size: int = 4096
    gpu_layers: int = 999
    max_tokens: int = 500
    temperature: float = 0.1
    image_max_side: int = 768
    image_tokens: int = 280
    startup_timeout: float = 180.0
    request_timeout: float = 180.0
    max_retries: int = 2
    max_consecutive_failures: int = 10


@dataclass(frozen=True)
class ClusteringConfig:
    pca_components: int = 50
    min_cluster_size: int = 5
    min_samples: int = 3
    selection_method: str = "leaf"


@dataclass(frozen=True)
class ClassificationConfig:
    use_clusters: bool = True
    min_representatives: int = 3
    max_representatives: int = 12
    min_agreement: float = 0.8
    ambiguity_std: float = 1.5


@dataclass(frozen=True)
class SearchConfig:
    default_limit: int = 60
    candidates: int = 200
    rrf_k: int = 60


@dataclass(frozen=True)
class OrganizationConfig:
    default_template: str = "{category}/{subcategory}"
    min_subcategory_size: int = 5
    min_confidence: float = 0.8
    duplicates_folder: str = "_Duplicados"


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    organization: OrganizationConfig = field(default_factory=OrganizationConfig)
    root: Path = PROJECT_ROOT

    def resolve(self, path: Path) -> Path:
        """Convierte una ruta de la configuración en absoluta."""
        return path if path.is_absolute() else self.root / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "organizer.db"

    @property
    def vector_index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"


def _build(cls: type, data: dict[str, Any], location: str) -> Any:
    """Construye un dataclass desde un diccionario, validando sus claves."""
    if not isinstance(data, dict):
        raise ConfigError(f"Se esperaba una sección en '{location}'.")
    hints = get_type_hints(cls)
    known = {item.name for item in fields(cls)}
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"Clave desconocida en la configuración: '{location}.{key}'.")
        expected = hints[key]
        if is_dataclass(expected):
            values[key] = _build(expected, value, f"{location}.{key}")
        elif expected is Path:
            values[key] = Path(value)
        elif getattr(expected, "__origin__", None) is tuple:
            values[key] = tuple(value)
        else:
            values[key] = value
    return cls(**values)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Lee el YAML indicado. Si no existe, usa los valores por defecto."""
    if not path.exists():
        return AppConfig()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _build(AppConfig, raw, "config")
