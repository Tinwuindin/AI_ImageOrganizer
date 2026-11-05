"""Esquema de SQLite y tipos de registro.

SQLite es el registro principal. FAISS solo guarda los vectores, y cada vector
se identifica con el ``id`` de la imagen a la que pertenece.
"""

from __future__ import annotations

from dataclasses import dataclass

# Estado de análisis de una imagen.
STATUS_SCANNED = "scanned"  # Registrada, a la espera de embedding.
STATUS_EMBEDDED = "embedded"  # Con vector, a la espera de clasificación.
STATUS_CLASSIFIED = "classified"  # Análisis completo.
STATUS_DUPLICATE = "duplicate"  # Copia exacta de otra: comparte su análisis.
STATUS_ERROR = "error"  # No se pudo leer o analizar.
STATUS_MISSING = "missing"  # Ya no está en la ruta registrada.

# Estados cuyas imágenes siguen en el flujo de análisis.
ACTIVE_STATUSES = (STATUS_SCANNED, STATUS_EMBEDDED, STATUS_CLASSIFIED)

# Origen de una clasificación.
SOURCE_DIRECT = "direct"  # Gemma analizó esta imagen.
SOURCE_CLUSTER = "cluster"  # Heredada de un representante de su grupo.
SOURCE_MANUAL = "manual"  # La persona la asignó.

# Estado de un plan de organización.
PLAN_DRAFT = "draft"
PLAN_RUNNING = "running"
PLAN_EXECUTED = "executed"
PLAN_UNDONE = "undone"

# Estado de cada movimiento.
OP_PLANNED = "planned"
OP_DONE = "done"
OP_FAILED = "failed"
OP_UNDONE = "undone"

# Motivo de cada movimiento.
REASON_CLASSIFIED = "classified"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_DUPLICATE = "duplicate"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    sha256      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    extension   TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    date_taken  TEXT,
    status      TEXT NOT NULL,
    error       TEXT,
    cluster_id  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_cluster ON images(cluster_id);

CREATE TABLE IF NOT EXISTS classifications (
    image_id      INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    category      TEXT NOT NULL,
    subcategory   TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    ocr_text      TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL,
    model_version TEXT NOT NULL,
    source        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_classifications_category
    ON classifications(category, subcategory);

CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id  INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    vector_id INTEGER NOT NULL UNIQUE,
    model     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL DEFAULT '',
    position    INTEGER NOT NULL DEFAULT 0
);

-- Cada grupo reúne las copias de un mismo contenido. Sus miembros son las
-- imágenes con ese sha256; la canónica conserva el análisis.
CREATE TABLE IF NOT EXISTS duplicate_groups (
    id                 INTEGER PRIMARY KEY,
    sha256             TEXT NOT NULL UNIQUE,
    canonical_image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS organization_plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    destination TEXT NOT NULL,
    template    TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    executed_at TEXT
);

CREATE TABLE IF NOT EXISTS operations (
    id          INTEGER PRIMARY KEY,
    plan_id     INTEGER NOT NULL REFERENCES organization_plans(id) ON DELETE CASCADE,
    image_id    INTEGER NOT NULL REFERENCES images(id),
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    folder      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    included    INTEGER NOT NULL DEFAULT 1,
    status      TEXT NOT NULL DEFAULT 'planned',
    error       TEXT,
    executed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_operations_plan ON operations(plan_id, folder);

CREATE VIRTUAL TABLE IF NOT EXISTS image_search USING fts5(
    filename, category, subcategory, description, tags, ocr_text,
    tokenize = "unicode61 remove_diacritics 2"
);
"""


@dataclass(frozen=True)
class ImageRecord:
    id: int
    path: str
    sha256: str
    filename: str
    extension: str
    width: int | None
    height: int | None
    size: int
    mtime: float
    date_taken: str | None
    status: str
    error: str | None
    cluster_id: int | None


@dataclass(frozen=True)
class ScannedFile:
    """Resultado de inspeccionar un archivo en disco."""

    path: str
    sha256: str
    size: int
    mtime: float
    width: int | None = None
    height: int | None = None
    date_taken: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Classification:
    image_id: int
    category: str
    subcategory: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    ocr_text: str = ""
    confidence: float = 0.0
    model_version: str = ""
    source: str = SOURCE_DIRECT


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    description: str
    position: int


@dataclass(frozen=True)
class NewOperation:
    image_id: int
    source_path: str
    target_path: str
    folder: str
    reason: str
    included: bool = True


@dataclass(frozen=True)
class Operation:
    id: int
    plan_id: int
    image_id: int
    source_path: str
    target_path: str
    folder: str
    reason: str
    included: bool
    status: str
    error: str | None
    executed_at: str | None


@dataclass(frozen=True)
class Plan:
    id: int
    destination: str
    template: str
    status: str
    created_at: str
    executed_at: str | None


@dataclass(frozen=True)
class ImageFilters:
    """Filtros del listado de la biblioteca."""

    category: str | None = None
    subcategory: str | None = None
    status: str | None = None
    source: str | None = None
    max_confidence: float | None = None


# Taxonomía inicial: la persona puede cambiarla desde la interfaz.
#
# El orden influye en el modelo, que tiende a quedarse con la primera opción que
# encaja: en las pruebas, las capturas de pantalla pasaron de 15/30 a 23/30 aciertos
# al subirlas en la lista. Por eso va de lo específico a lo genérico, y las palabras
# de cada descripción también cuentan ("retratos" en Personas atraía a los perros).
DEFAULT_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Capturas de pantalla", "Capturas de computadora o teléfono: páginas web, chats e interfaces de aplicaciones."),
    ("Documentos", "Papeles, recibos, identificaciones, pizarrones y fotos de texto."),
    ("Memes y gráficos", "Memes, ilustraciones, logotipos, diagramas e imágenes diseñadas."),
    ("Mascotas y animales", "Perros, gatos, aves y otros animales, aunque los acompañe una persona."),
    ("Comida y bebida", "Platillos, bebidas, recetas y restaurantes."),
    ("Vehículos", "Autos, motos, bicicletas, trenes, aviones y barcos."),
    ("Electrónica y tecnología", "Circuitos, componentes, computadoras, gadgets y proyectos técnicos."),
    ("Paisajes y naturaleza", "Montañas, playas, bosques, cielos, plantas y escenas al aire libre."),
    ("Ciudad y arquitectura", "Calles, edificios, monumentos e interiores arquitectónicos."),
    ("Viajes y eventos", "Celebraciones, conciertos, excursiones y recuerdos de viaje."),
    ("Personas", "Selfies, grupos y fotos cuyo tema principal son las personas."),
    ("Otros", "Todo lo que no encaja claramente en ninguna otra categoría."),
)

# Categoría de reserva: siempre existe, porque el clasificador la necesita.
FALLBACK_CATEGORY = "Otros"
