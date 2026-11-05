"""Acceso a SQLite.

Cada hilo usa su propia conexión. WAL permite que la interfaz lea mientras el
análisis escribe, y todas las escrituras pasan por transacciones explícitas.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.database.models import (
    ACTIVE_STATUSES,
    DEFAULT_CATEGORIES,
    OP_DONE,
    OP_FAILED,
    OP_PLANNED,
    OP_UNDONE,
    PLAN_DRAFT,
    SCHEMA,
    SOURCE_MANUAL,
    STATUS_CLASSIFIED,
    STATUS_DUPLICATE,
    STATUS_EMBEDDED,
    STATUS_ERROR,
    STATUS_MISSING,
    STATUS_SCANNED,
    Category,
    Classification,
    ImageFilters,
    ImageRecord,
    NewOperation,
    Operation,
    Plan,
    ScannedFile,
)

# SQLite limita los parámetros por sentencia: las listas largas van por tramos.
_CHUNK_SIZE = 900

_SORT_COLUMNS = {
    "date_desc": "COALESCE(i.date_taken, datetime(i.mtime, 'unixepoch')) DESC, i.id DESC",
    "date_asc": "COALESCE(i.date_taken, datetime(i.mtime, 'unixepoch')) ASC, i.id ASC",
    "name": "i.filename COLLATE NOCASE ASC, i.id ASC",
    "size": "i.size DESC, i.id ASC",
    "confidence": "c.confidence ASC, i.id ASC",
}

_LIBRARY_COLUMNS = """
    i.id, i.path, i.sha256, i.filename, i.extension, i.width, i.height, i.size,
    i.mtime, i.date_taken, i.status, i.error, i.cluster_id,
    c.category, c.subcategory, c.description, c.tags, c.ocr_text, c.confidence,
    c.model_version, c.source
"""

# El estado se deduce de lo que ya existe, así nunca se repite trabajo hecho.
_DERIVED_STATUS = f"""
    CASE
        WHEN id IN (SELECT image_id FROM classifications) THEN '{STATUS_CLASSIFIED}'
        WHEN id IN (SELECT image_id FROM image_embeddings) THEN '{STATUS_EMBEDDED}'
        ELSE '{STATUS_SCANNED}'
    END
"""


def _chunks(items: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), _CHUNK_SIZE):
        yield items[start:start + _CHUNK_SIZE]


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _to_image(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(**dict(row))


def _to_classification(row: sqlite3.Row) -> Classification:
    data = dict(row)
    data["tags"] = tuple(json.loads(data["tags"]))
    return Classification(**data)


def _to_operation(row: sqlite3.Row) -> Operation:
    data = dict(row)
    data["included"] = bool(data["included"])
    return Operation(**data)


def _library_item(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["tags"] = json.loads(item["tags"]) if item["tags"] else []
    return item


def _extension_of(path: str) -> str:
    return Path(path).suffix.lower()


class Repository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection.executescript(SCHEMA)
        self._seed_categories()

    @property
    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection
        if connection.in_transaction:
            yield connection
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def _seed_categories(self) -> None:
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM categories LIMIT 1").fetchone():
                return
            connection.executemany(
                "INSERT INTO categories (name, description, position) VALUES (?, ?, ?)",
                [(name, text, index) for index, (name, text) in enumerate(DEFAULT_CATEGORIES)],
            )

    # ------------------------------------------------------------------
    # Imágenes
    # ------------------------------------------------------------------

    def get_image(self, image_id: int) -> ImageRecord | None:
        row = self.connection.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return _to_image(row) if row else None

    def get_images(self, ids: Sequence[int]) -> dict[int, ImageRecord]:
        records: dict[int, ImageRecord] = {}
        for chunk in _chunks(list(ids)):
            rows = self.connection.execute(
                f"SELECT * FROM images WHERE id IN ({_placeholders(len(chunk))})", chunk
            )
            records.update({row["id"]: _to_image(row) for row in rows})
        return records

    def all_images(self) -> dict[str, ImageRecord]:
        """Todas las imágenes registradas, indexadas por ruta."""
        rows = self.connection.execute("SELECT * FROM images")
        return {row["path"]: _to_image(row) for row in rows}

    def find_by_sha256(self, sha256: str) -> list[ImageRecord]:
        rows = self.connection.execute("SELECT * FROM images WHERE sha256 = ?", (sha256,))
        return [_to_image(row) for row in rows]

    def insert_image(self, file: ScannedFile) -> int:
        status = STATUS_ERROR if file.error else STATUS_SCANNED
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO images (path, sha256, filename, extension, width, height, size,
                                    mtime, date_taken, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file.path, file.sha256, Path(file.path).name, _extension_of(file.path),
                    file.width, file.height, file.size, file.mtime, file.date_taken,
                    status, file.error,
                ),
            )
            image_id = int(cursor.lastrowid)
            self._reindex(connection, image_id)
        return image_id

    def update_image_content(self, image_id: int, file: ScannedFile) -> None:
        """Registra que el archivo cambió de contenido: su análisis ya no vale."""
        status = STATUS_ERROR if file.error else STATUS_SCANNED
        with self.transaction() as connection:
            connection.execute("DELETE FROM classifications WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM image_embeddings WHERE image_id = ?", (image_id,))
            connection.execute(
                """
                UPDATE images
                SET sha256 = ?, width = ?, height = ?, size = ?, mtime = ?, date_taken = ?,
                    status = ?, error = ?, cluster_id = NULL
                WHERE id = ?
                """,
                (
                    file.sha256, file.width, file.height, file.size, file.mtime,
                    file.date_taken, status, file.error, image_id,
                ),
            )
            self._reindex(connection, image_id)

    def touch_image(self, image_id: int, mtime: float) -> None:
        """El archivo se tocó pero su contenido es el mismo."""
        with self.transaction() as connection:
            connection.execute("UPDATE images SET mtime = ? WHERE id = ?", (mtime, image_id))

    def relocate_image(self, image_id: int, path: str, mtime: float) -> None:
        """Actualiza la ruta de una imagen conservando todo su análisis."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE images SET path = ?, filename = ?, extension = ?, mtime = ? WHERE id = ?",
                (path, Path(path).name, _extension_of(path), mtime, image_id),
            )
            self._reindex(connection, image_id)

    def set_status(self, ids: Sequence[int], status: str, error: str | None = None) -> None:
        with self.transaction() as connection:
            for chunk in _chunks(list(ids)):
                connection.execute(
                    f"UPDATE images SET status = ?, error = ? "
                    f"WHERE id IN ({_placeholders(len(chunk))})",
                    (status, error, *chunk),
                )

    def record_error(self, image_id: int, error: str) -> None:
        """Anota un fallo pasajero: la imagen se reintenta en el siguiente análisis."""
        with self.transaction() as connection:
            connection.execute("UPDATE images SET error = ? WHERE id = ?", (error, image_id))

    def restore_status(self, ids: Sequence[int]) -> None:
        """Devuelve imágenes al flujo, en la etapa que les corresponda."""
        with self.transaction() as connection:
            for chunk in _chunks(list(ids)):
                connection.execute(
                    f"UPDATE images SET status = {_DERIVED_STATUS}, error = NULL "
                    f"WHERE id IN ({_placeholders(len(chunk))})",
                    chunk,
                )

    def ids_pending_embedding(self) -> list[int]:
        rows = self.connection.execute(
            f"""
            SELECT id FROM images
            WHERE status IN ({_placeholders(len(ACTIVE_STATUSES))})
              AND id NOT IN (SELECT image_id FROM image_embeddings)
            ORDER BY id
            """,
            ACTIVE_STATUSES,
        )
        return [row["id"] for row in rows]

    def embedded_active_ids(self) -> list[int]:
        """Imágenes con vector que participan en agrupación y clasificación."""
        rows = self.connection.execute(
            f"""
            SELECT i.id FROM images i
            JOIN image_embeddings e ON e.image_id = i.id
            WHERE i.status IN ({_placeholders(len(ACTIVE_STATUSES))})
            ORDER BY i.id
            """,
            ACTIVE_STATUSES,
        )
        return [row["id"] for row in rows]

    def set_clusters(self, assignments: Iterable[tuple[int, int | None]]) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "UPDATE images SET cluster_id = ? WHERE id = ?",
                [(cluster, image_id) for image_id, cluster in assignments],
            )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embeddings(self, ids: Sequence[int], model: str) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO image_embeddings (image_id, vector_id, model) "
                "VALUES (?, ?, ?)",
                [(image_id, image_id, model) for image_id in ids],
            )
            for chunk in _chunks(list(ids)):
                connection.execute(
                    f"UPDATE images SET status = '{STATUS_EMBEDDED}' "
                    f"WHERE status = '{STATUS_SCANNED}' "
                    f"AND id IN ({_placeholders(len(chunk))})",
                    chunk,
                )

    def embedding_ids(self) -> list[int]:
        rows = self.connection.execute("SELECT vector_id FROM image_embeddings")
        return [row["vector_id"] for row in rows]

    def delete_embeddings(self, ids: Sequence[int]) -> None:
        with self.transaction() as connection:
            for chunk in _chunks(list(ids)):
                connection.execute(
                    f"DELETE FROM image_embeddings WHERE image_id IN ({_placeholders(len(chunk))})",
                    chunk,
                )

    # ------------------------------------------------------------------
    # Clasificaciones
    # ------------------------------------------------------------------

    def get_classification(self, image_id: int) -> Classification | None:
        row = self.connection.execute(
            "SELECT * FROM classifications WHERE image_id = ?", (image_id,)
        ).fetchone()
        return _to_classification(row) if row else None

    def get_classifications(self, ids: Sequence[int]) -> dict[int, Classification]:
        result: dict[int, Classification] = {}
        for chunk in _chunks(list(ids)):
            rows = self.connection.execute(
                f"SELECT * FROM classifications WHERE image_id IN ({_placeholders(len(chunk))})",
                chunk,
            )
            result.update({row["image_id"]: _to_classification(row) for row in rows})
        return result

    def save_classifications(self, items: Sequence[Classification]) -> None:
        with self.transaction() as connection:
            for item in items:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO classifications
                        (image_id, category, subcategory, description, tags, ocr_text,
                         confidence, model_version, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.image_id, item.category, item.subcategory, item.description,
                        json.dumps(list(item.tags), ensure_ascii=False), item.ocr_text,
                        item.confidence, item.model_version, item.source,
                    ),
                )
                connection.execute(
                    f"UPDATE images SET status = '{STATUS_CLASSIFIED}', error = NULL "
                    f"WHERE id = ? AND status IN ('{STATUS_SCANNED}', '{STATUS_EMBEDDED}', "
                    f"'{STATUS_ERROR}')",
                    (item.image_id,),
                )
                self._reindex(connection, item.image_id)

    def set_manual_category(self, image_id: int, category: str, subcategory: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                f"""
                INSERT INTO classifications
                    (image_id, category, subcategory, confidence, model_version, source)
                VALUES (?, ?, ?, 1.0, '{SOURCE_MANUAL}', '{SOURCE_MANUAL}')
                ON CONFLICT(image_id) DO UPDATE SET
                    category = excluded.category,
                    subcategory = excluded.subcategory,
                    confidence = 1.0,
                    source = '{SOURCE_MANUAL}'
                """,
                (image_id, category, subcategory),
            )
            connection.execute(
                f"UPDATE images SET status = '{STATUS_CLASSIFIED}' WHERE id = ?", (image_id,)
            )
            self._reindex(connection, image_id)

    def subcategory_names(self) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            "SELECT DISTINCT category, subcategory FROM classifications WHERE subcategory != ''"
        )
        return [(row["category"], row["subcategory"]) for row in rows]

    def _reindex(self, connection: sqlite3.Connection, image_id: int) -> None:
        """Mantiene sincronizado el índice de texto completo de una imagen."""
        row = connection.execute(
            """
            SELECT i.filename, c.category, c.subcategory, c.description, c.tags, c.ocr_text
            FROM images i LEFT JOIN classifications c ON c.image_id = i.id
            WHERE i.id = ?
            """,
            (image_id,),
        ).fetchone()
        connection.execute("DELETE FROM image_search WHERE rowid = ?", (image_id,))
        if row is None:
            return
        tags = " ".join(json.loads(row["tags"])) if row["tags"] else ""
        connection.execute(
            """
            INSERT INTO image_search
                (rowid, filename, category, subcategory, description, tags, ocr_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, row["filename"], row["category"] or "", row["subcategory"] or "",
                row["description"] or "", tags, row["ocr_text"] or "",
            ),
        )

    # ------------------------------------------------------------------
    # Búsqueda y biblioteca
    # ------------------------------------------------------------------

    def search_text(self, match_expression: str, limit: int) -> list[int]:
        """Identificadores que coinciden en el índice de texto, del mejor al peor."""
        rows = self.connection.execute(
            "SELECT rowid FROM image_search WHERE image_search MATCH ? ORDER BY rank LIMIT ?",
            (match_expression, limit),
        )
        return [row["rowid"] for row in rows]

    def library_items(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Datos de presentación de las imágenes indicadas que siguen disponibles."""
        items: dict[int, dict[str, Any]] = {}
        for chunk in _chunks(list(ids)):
            rows = self.connection.execute(
                f"""
                SELECT {_LIBRARY_COLUMNS}
                FROM images i LEFT JOIN classifications c ON c.image_id = i.id
                WHERE i.id IN ({_placeholders(len(chunk))})
                  AND i.status IN ({_placeholders(len(ACTIVE_STATUSES))})
                """,
                (*chunk, *ACTIVE_STATUSES),
            )
            items.update({row["id"]: _library_item(row) for row in rows})
        return items

    def library_item(self, image_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            f"""
            SELECT {_LIBRARY_COLUMNS}
            FROM images i LEFT JOIN classifications c ON c.image_id = i.id
            WHERE i.id = ?
            """,
            (image_id,),
        ).fetchone()
        return _library_item(row) if row else None

    def list_images(
        self,
        filters: ImageFilters,
        limit: int,
        offset: int,
        sort: str = "date_desc",
    ) -> tuple[list[dict[str, Any]], int]:
        conditions: list[str] = []
        parameters: list[Any] = []
        if filters.status:
            conditions.append("i.status = ?")
            parameters.append(filters.status)
        else:
            conditions.append(f"i.status IN ({_placeholders(len(ACTIVE_STATUSES))})")
            parameters.extend(ACTIVE_STATUSES)
        if filters.category:
            conditions.append("c.category = ?")
            parameters.append(filters.category)
        if filters.subcategory is not None:
            conditions.append("c.subcategory = ?")
            parameters.append(filters.subcategory)
        if filters.source:
            conditions.append("c.source = ?")
            parameters.append(filters.source)
        if filters.max_confidence is not None:
            conditions.append("c.confidence < ?")
            parameters.append(filters.max_confidence)
        where = " AND ".join(conditions)
        order = _SORT_COLUMNS.get(sort, _SORT_COLUMNS["date_desc"])
        base = f"FROM images i LEFT JOIN classifications c ON c.image_id = i.id WHERE {where}"
        total = self.connection.execute(f"SELECT COUNT(*) {base}", parameters).fetchone()[0]
        rows = self.connection.execute(
            f"SELECT {_LIBRARY_COLUMNS} {base} ORDER BY {order} LIMIT ? OFFSET ?",
            (*parameters, limit, offset),
        )
        return [_library_item(row) for row in rows], total

    def category_tree(self) -> list[dict[str, Any]]:
        """Categorías con sus subcategorías y conteos."""
        counts = self.connection.execute(
            f"""
            SELECT c.category, c.subcategory, COUNT(*) AS total
            FROM classifications c JOIN images i ON i.id = c.image_id
            WHERE i.status IN ({_placeholders(len(ACTIVE_STATUSES))})
            GROUP BY c.category, c.subcategory
            """,
            ACTIVE_STATUSES,
        ).fetchall()
        tree: dict[str, dict[str, Any]] = {
            category.name: {
                "id": category.id,
                "name": category.name,
                "description": category.description,
                "total": 0,
                "subcategories": [],
            }
            for category in self.list_categories()
        }
        for row in counts:
            node = tree.setdefault(
                row["category"],
                {"id": None, "name": row["category"], "description": "", "total": 0,
                 "subcategories": []},
            )
            node["total"] += row["total"]
            if row["subcategory"]:
                node["subcategories"].append({"name": row["subcategory"], "total": row["total"]})
        for node in tree.values():
            node["subcategories"].sort(key=lambda item: (-item["total"], item["name"]))
        return list(tree.values())

    def stats(self) -> dict[str, Any]:
        statuses = {
            row["status"]: row["total"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS total FROM images GROUP BY status"
            )
        }
        sources = {
            row["source"]: row["total"]
            for row in self.connection.execute(
                "SELECT source, COUNT(*) AS total FROM classifications GROUP BY source"
            )
        }
        clusters = self.connection.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM images WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        return {
            "total": sum(statuses.values()),
            "library": sum(statuses.get(status, 0) for status in ACTIVE_STATUSES),
            "statuses": statuses,
            "sources": sources,
            "clusters": clusters,
            "duplicate_groups": self.connection.execute(
                "SELECT COUNT(*) FROM duplicate_groups"
            ).fetchone()[0],
        }

    # ------------------------------------------------------------------
    # Categorías
    # ------------------------------------------------------------------

    def list_categories(self) -> list[Category]:
        rows = self.connection.execute("SELECT * FROM categories ORDER BY position, name")
        return [Category(**dict(row)) for row in rows]

    def get_category(self, category_id: int) -> Category | None:
        row = self.connection.execute(
            "SELECT * FROM categories WHERE id = ?", (category_id,)
        ).fetchone()
        return Category(**dict(row)) if row else None

    def create_category(self, name: str, description: str) -> Category:
        """Agrega una categoría al inicio: las propias suelen ser las más específicas."""
        with self.transaction() as connection:
            position = connection.execute(
                "SELECT COALESCE(MIN(position), 1) - 1 FROM categories"
            ).fetchone()[0]
            cursor = connection.execute(
                "INSERT INTO categories (name, description, position) VALUES (?, ?, ?)",
                (name, description, position),
            )
            return Category(int(cursor.lastrowid), name, description, position)

    def update_category(self, category_id: int, name: str, description: str) -> None:
        current = self.get_category(category_id)
        if current is None:
            return
        with self.transaction() as connection:
            connection.execute(
                "UPDATE categories SET name = ?, description = ? WHERE id = ?",
                (name, description, category_id),
            )
            if name != current.name:
                self._move_classifications(connection, current.name, name)

    def delete_category(self, category_id: int, reassign_to: str) -> None:
        current = self.get_category(category_id)
        if current is None:
            return
        with self.transaction() as connection:
            self._move_classifications(connection, current.name, reassign_to)
            connection.execute("DELETE FROM categories WHERE id = ?", (category_id,))

    def _move_classifications(
        self, connection: sqlite3.Connection, source: str, target: str
    ) -> None:
        ids = [
            row["image_id"]
            for row in connection.execute(
                "SELECT image_id FROM classifications WHERE category = ?", (source,)
            )
        ]
        connection.execute(
            "UPDATE classifications SET category = ? WHERE category = ?", (target, source)
        )
        for image_id in ids:
            self._reindex(connection, image_id)

    # ------------------------------------------------------------------
    # Duplicados
    # ------------------------------------------------------------------

    def duplicate_candidates(self) -> dict[str, list[ImageRecord]]:
        """Copias presentes de cada contenido que aparece más de una vez."""
        excluded = (STATUS_MISSING, STATUS_ERROR)
        rows = self.connection.execute(
            f"""
            SELECT * FROM images
            WHERE status NOT IN ({_placeholders(len(excluded))})
              AND sha256 IN (
                  SELECT sha256 FROM images
                  WHERE status NOT IN ({_placeholders(len(excluded))})
                  GROUP BY sha256 HAVING COUNT(*) > 1
              )
            ORDER BY sha256, id
            """,
            (*excluded, *excluded),
        )
        groups: dict[str, list[ImageRecord]] = {}
        for row in rows:
            groups.setdefault(row["sha256"], []).append(_to_image(row))
        return groups

    def replace_duplicate_groups(self, groups: Sequence[tuple[str, int]]) -> None:
        """Sustituye los grupos por ``(sha256, id_canónico)``."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM duplicate_groups")
            connection.executemany(
                "INSERT INTO duplicate_groups (sha256, canonical_image_id) VALUES (?, ?)", groups
            )

    def orphaned_duplicates(self) -> list[int]:
        """Imágenes marcadas como copia cuyo grupo ya no existe."""
        rows = self.connection.execute(
            f"""
            SELECT id FROM images
            WHERE status = '{STATUS_DUPLICATE}'
              AND sha256 NOT IN (SELECT sha256 FROM duplicate_groups)
            """
        )
        return [row["id"] for row in rows]

    def duplicate_groups(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        total = self.connection.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]
        groups = self.connection.execute(
            """
            SELECT g.id, g.sha256, g.canonical_image_id, i.size,
                   (SELECT COUNT(*) FROM images m
                    WHERE m.sha256 = g.sha256 AND m.status != ?) AS copies
            FROM duplicate_groups g JOIN images i ON i.id = g.canonical_image_id
            ORDER BY copies * i.size DESC, g.id
            LIMIT ? OFFSET ?
            """,
            (STATUS_MISSING, limit, offset),
        ).fetchall()
        result = []
        for group in groups:
            members = self.connection.execute(
                """
                SELECT id, path, filename, width, height, size, date_taken, status
                FROM images WHERE sha256 = ? AND status != ? ORDER BY path
                """,
                (group["sha256"], STATUS_MISSING),
            ).fetchall()
            result.append({
                "id": group["id"],
                "sha256": group["sha256"],
                "canonical_image_id": group["canonical_image_id"],
                "size": group["size"],
                "copies": group["copies"],
                "members": [dict(member) for member in members],
            })
        return result, total

    def duplicate_recoverable_bytes(self) -> int:
        """Espacio que ocupan las copias sobrantes."""
        row = self.connection.execute(
            f"SELECT COALESCE(SUM(size), 0) FROM images WHERE status = '{STATUS_DUPLICATE}'"
        ).fetchone()
        return int(row[0])

    def duplicate_group(self, group_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM duplicate_groups WHERE id = ?", (group_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_canonical(self, group_id: int, image_id: int) -> bool:
        """Cambia la copia que se conserva; el análisis acompaña a la canónica."""
        with self.transaction() as connection:
            group = connection.execute(
                "SELECT sha256, canonical_image_id FROM duplicate_groups WHERE id = ?", (group_id,)
            ).fetchone()
            target = self.get_image(image_id)
            if group is None or target is None or target.sha256 != group["sha256"]:
                return False
            if target.status == STATUS_MISSING:
                return False
            previous = group["canonical_image_id"]
            if previous == image_id:
                return True
            cluster = connection.execute(
                "SELECT cluster_id FROM images WHERE id = ?", (previous,)
            ).fetchone()["cluster_id"]
            connection.execute(
                "UPDATE classifications SET image_id = ? WHERE image_id = ?", (image_id, previous)
            )
            connection.execute(
                "UPDATE image_embeddings SET image_id = ?, vector_id = ? WHERE image_id = ?",
                (image_id, image_id, previous),
            )
            connection.execute(
                "UPDATE images SET status = ?, cluster_id = NULL WHERE id = ?",
                (STATUS_DUPLICATE, previous),
            )
            connection.execute(
                f"UPDATE images SET status = {_DERIVED_STATUS}, cluster_id = ? WHERE id = ?",
                (cluster, image_id),
            )
            connection.execute(
                "UPDATE duplicate_groups SET canonical_image_id = ? WHERE id = ?",
                (image_id, group_id),
            )
            self._reindex(connection, previous)
            self._reindex(connection, image_id)
        return True

    # ------------------------------------------------------------------
    # Planes de organización
    # ------------------------------------------------------------------

    def organizable_images(self) -> list[dict[str, Any]]:
        """Imágenes clasificadas que pueden entrar en un plan."""
        rows = self.connection.execute(
            f"""
            SELECT {_LIBRARY_COLUMNS}
            FROM images i JOIN classifications c ON c.image_id = i.id
            WHERE i.status = '{STATUS_CLASSIFIED}'
            ORDER BY i.id
            """
        )
        return [_library_item(row) for row in rows]

    def duplicate_copies(self) -> list[ImageRecord]:
        rows = self.connection.execute(
            f"SELECT * FROM images WHERE status = '{STATUS_DUPLICATE}' ORDER BY id"
        )
        return [_to_image(row) for row in rows]

    def create_plan(
        self, destination: str, template: str, operations: Sequence[NewOperation]
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO organization_plans (destination, template, status) VALUES (?, ?, ?)",
                (destination, template, PLAN_DRAFT),
            )
            plan_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO operations
                    (plan_id, image_id, source_path, target_path, folder, reason, included)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        plan_id, item.image_id, item.source_path, item.target_path,
                        item.folder, item.reason, int(item.included),
                    )
                    for item in operations
                ],
            )
        return plan_id

    def get_plan(self, plan_id: int) -> Plan | None:
        row = self.connection.execute(
            "SELECT * FROM organization_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return Plan(**dict(row)) if row else None

    def list_plans(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT p.*,
                   COUNT(o.id) AS total,
                   SUM(o.included) AS included,
                   SUM(o.status = ?) AS done,
                   SUM(o.status = ?) AS failed,
                   SUM(o.status = ?) AS undone
            FROM organization_plans p LEFT JOIN operations o ON o.plan_id = p.id
            GROUP BY p.id ORDER BY p.id DESC LIMIT ?
            """,
            (OP_DONE, OP_FAILED, OP_UNDONE, limit),
        )
        return [dict(row) for row in rows]

    def set_plan_status(self, plan_id: int, status: str, executed: bool = False) -> None:
        with self.transaction() as connection:
            if executed:
                connection.execute(
                    "UPDATE organization_plans SET status = ?, "
                    "executed_at = datetime('now', 'localtime') WHERE id = ?",
                    (status, plan_id),
                )
            else:
                connection.execute(
                    "UPDATE organization_plans SET status = ? WHERE id = ?", (status, plan_id)
                )

    def delete_plan(self, plan_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM organization_plans WHERE id = ?", (plan_id,))

    def plan_folders(self, plan_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT folder, COUNT(*) AS total, SUM(included) AS included,
                   SUM(status = ?) AS done, SUM(status = ?) AS failed,
                   SUM(reason = 'low_confidence') AS low_confidence
            FROM operations WHERE plan_id = ?
            GROUP BY folder ORDER BY folder COLLATE NOCASE
            """,
            (OP_DONE, OP_FAILED, plan_id),
        ).fetchall()
        folders = []
        for row in rows:
            samples = self.connection.execute(
                "SELECT image_id FROM operations WHERE plan_id = ? AND folder = ? LIMIT 6",
                (plan_id, row["folder"]),
            )
            folders.append({**dict(row), "sample_ids": [item["image_id"] for item in samples]})
        return folders

    def plan_operations(
        self,
        plan_id: int,
        folder: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Operation]:
        query = "SELECT * FROM operations WHERE plan_id = ?"
        parameters: list[Any] = [plan_id]
        if folder is not None:
            query += " AND folder = ?"
            parameters.append(folder)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            parameters.extend([limit, offset])
        return [_to_operation(row) for row in self.connection.execute(query, parameters)]

    def plan_targets(self, plan_id: int) -> set[str]:
        rows = self.connection.execute(
            "SELECT target_path FROM operations WHERE plan_id = ?", (plan_id,)
        )
        return {row["target_path"] for row in rows}

    def get_operation(self, operation_id: int) -> Operation | None:
        row = self.connection.execute(
            "SELECT * FROM operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return _to_operation(row) if row else None

    def set_operation_included(self, operation_id: int, included: bool) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET included = ? WHERE id = ? AND status = ?",
                (int(included), operation_id, OP_PLANNED),
            )

    def set_folder_included(self, plan_id: int, folder: str, included: bool) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET included = ? "
                "WHERE plan_id = ? AND folder = ? AND status = ?",
                (int(included), plan_id, folder, OP_PLANNED),
            )

    def retarget_operation(self, operation_id: int, target_path: str, folder: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET target_path = ?, folder = ? WHERE id = ? AND status = ?",
                (target_path, folder, operation_id, OP_PLANNED),
            )

    def finish_operation(
        self,
        operation: Operation,
        new_path: str,
        new_mtime: float,
        status: str,
    ) -> None:
        """Registra en una sola transacción el movimiento y la nueva ruta."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET status = ?, error = NULL, "
                "executed_at = datetime('now', 'localtime') WHERE id = ?",
                (status, operation.id),
            )
            self.relocate_image(operation.image_id, new_path, new_mtime)

    def fail_operation(self, operation_id: int, error: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET status = ?, error = ? WHERE id = ?",
                (OP_FAILED, error, operation_id),
            )

    def note_operation_error(self, operation_id: int, error: str) -> None:
        """Anota un error sin cambiar el estado: el archivo sigue donde estaba."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE operations SET error = ? WHERE id = ?", (error, operation_id)
            )
