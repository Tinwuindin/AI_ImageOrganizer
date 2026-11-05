from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from app.core.config import AppConfig, OrganizationConfig
from app.core.hashing import sha256_file
from app.core.scanner import scan_directories
from app.database.models import (
    OP_DONE,
    OP_FAILED,
    OP_PLANNED,
    OP_UNDONE,
    PLAN_DRAFT,
    PLAN_EXECUTED,
    PLAN_RUNNING,
    PLAN_UNDONE,
    REASON_DUPLICATE,
    REASON_LOW_CONFIDENCE,
    STATUS_DUPLICATE,
    Classification,
)
from app.database.repository import Repository
from app.engine.duplicate_detector import detect_exact_duplicates
from app.services import organization
from app.services.organization import (
    OrganizationError,
    OrganizationService,
    move_file,
    safe_component,
    validate_template,
)
from tests.conftest import make_image


@pytest.fixture
def repository(config: AppConfig) -> Repository:
    return Repository(config.database_path)


@pytest.fixture
def service(repository: Repository) -> OrganizationService:
    return OrganizationService(
        repository,
        OrganizationConfig(min_subcategory_size=2, min_confidence=0.5),
    )


def add_images(
    root: Path,
    repository: Repository,
    config: AppConfig,
    specs: dict[str, tuple[str, str, float]],
) -> dict[str, int]:
    """Crea imágenes clasificadas: ``{nombre: (categoría, subcategoría, confianza)}``."""
    for index, name in enumerate(specs):
        make_image(root / name, seed=index + 100)
    scan_directories([root], repository, config.scanner, config.thumbnails_dir)
    ids: dict[str, int] = {}
    for path, record in repository.all_images().items():
        name = Path(path).relative_to(root).as_posix()
        ids[name] = record.id
        category, subcategory, confidence = specs[name]
        repository.save_classifications([
            Classification(record.id, category, subcategory, confidence=confidence)
        ])
    return ids


def test_plan_does_not_touch_the_disk(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    add_images(root, repository, config, {
        "a.jpg": ("Comida y bebida", "Pizza", 0.9),
        "b.jpg": ("Comida y bebida", "Pizza", 0.9),
    })
    destination = tmp_path / "ordenadas"

    plan_id = service.create_plan(str(destination))

    assert not destination.exists()
    assert (root / "a.jpg").exists()
    operations = repository.plan_operations(plan_id)
    assert {operation.folder for operation in operations} == {"Comida y bebida/Pizza"}
    assert all(operation.status == OP_PLANNED for operation in operations)
    assert all(
        Path(operation.target_path).parent == destination / "Comida y bebida" / "Pizza"
        for operation in operations
    )


def test_rare_subcategories_are_grouped(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
        "c.jpg": ("Personas", "Bautizo", 0.9),
    })

    plan_id = service.create_plan(str(tmp_path / "ordenadas"))

    folders = {item["folder"]: item["total"] for item in repository.plan_folders(plan_id)}
    assert folders == {"Personas/Boda": 2, "Personas/Varios": 1}


def test_low_confidence_images_start_unchecked(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.2),
        "b.jpg": ("Personas", "Boda", 0.9),
    })

    plan_id = service.create_plan(str(tmp_path / "ordenadas"))

    flagged = {op.reason: op.included for op in repository.plan_operations(plan_id)}
    assert flagged[REASON_LOW_CONFIDENCE] is False


def test_name_collisions_get_a_suffix_and_never_overwrite(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "uno/foto.jpg": ("Personas", "Boda", 0.9),
        "dos/foto.jpg": ("Personas", "Boda", 0.9),
        "tres/FOTO.jpg": ("Personas", "Boda", 0.9),
    })
    occupied = make_image(tmp_path / "ordenadas" / "Personas" / "Boda" / "foto.jpg", seed=999)
    original = occupied.read_bytes()

    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    service.execute(plan_id)

    names = sorted(path.name for path in occupied.parent.iterdir())
    assert len(names) == 4
    assert occupied.read_bytes() == original


def test_files_already_in_place_are_left_alone(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "ordenadas"
    add_images(root, repository, config, {
        "Personas/Boda/a.jpg": ("Personas", "Boda", 0.9),
        "Personas/Boda/b.jpg": ("Personas", "Boda", 0.9),
    })

    plan_id = service.create_plan(str(root))

    assert repository.plan_operations(plan_id) == []


def test_exact_duplicates_go_to_their_own_folder(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    make_image(root / "original.jpg", seed=7)
    shutil.copy(root / "original.jpg", root / "copia.jpg")
    scan_directories([root], repository, config.scanner, config.thumbnails_dir)
    assert detect_exact_duplicates(repository) == 1
    records = {record.filename: record for record in repository.all_images().values()}
    canonical = min(records.values(), key=lambda record: len(record.path))
    assert records["copia.jpg"].status == STATUS_DUPLICATE or canonical.filename == "copia.jpg"
    repository.save_classifications([
        Classification(canonical.id, "Personas", "Boda", confidence=0.9)
    ])

    plan_id = service.create_plan(str(tmp_path / "ordenadas"))

    duplicates = [
        op for op in repository.plan_operations(plan_id) if op.reason == REASON_DUPLICATE
    ]
    assert len(duplicates) == 1
    assert duplicates[0].folder == "_Duplicados"


def test_execute_moves_files_and_updates_paths(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    ids = add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    digest = sha256_file(root / "a.jpg")
    destination = tmp_path / "ordenadas"
    plan_id = service.create_plan(str(destination))

    result = service.execute(plan_id)

    assert result == {"moved": 2, "failed": 0, "total": 2}
    target = destination / "Personas" / "Boda" / "a.jpg"
    assert target.exists() and not (root / "a.jpg").exists()
    assert sha256_file(target) == digest
    assert repository.get_image(ids["a.jpg"]).path == str(target)
    assert repository.get_plan(plan_id).status == PLAN_EXECUTED
    assert all(op.status == OP_DONE for op in repository.plan_operations(plan_id))


def test_excluded_operations_are_not_executed(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    first = repository.plan_operations(plan_id)[0]
    repository.set_operation_included(first.id, False)

    result = service.execute(plan_id)

    assert result["moved"] == 1
    assert Path(first.source_path).exists()


def test_target_created_after_planning_is_not_overwritten(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    operation = repository.plan_operations(plan_id)[0]
    intruder = make_image(Path(operation.target_path), seed=321)
    content = intruder.read_bytes()

    result = service.execute(plan_id)

    assert result["failed"] == 1
    assert intruder.read_bytes() == content
    assert Path(operation.source_path).exists()
    failed = repository.get_operation(operation.id)
    assert failed.status == OP_FAILED
    assert "Ya existe" in failed.error


def test_file_changed_after_analysis_is_not_moved(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    with (root / "a.jpg").open("ab") as handle:
        handle.write(b"cambio")

    result = service.execute(plan_id)

    assert result["failed"] == 1
    assert (root / "a.jpg").exists()


def test_undo_restores_everything_and_cleans_folders(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    ids = add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Comida y bebida", "Pizza", 0.9),
        "c.jpg": ("Comida y bebida", "Pizza", 0.9),
    })
    original_path = repository.get_image(ids["a.jpg"]).path
    digest = sha256_file(root / "a.jpg")
    destination = tmp_path / "ordenadas"
    destination.mkdir()
    unrelated = make_image(destination / "mia.jpg", seed=55)
    plan_id = service.create_plan(str(destination))
    service.execute(plan_id)

    result = service.undo(plan_id)

    assert result == {"restored": 3, "failed": 0, "total": 3}
    assert sha256_file(root / "a.jpg") == digest
    assert repository.get_image(ids["a.jpg"]).path == original_path
    assert repository.get_plan(plan_id).status == PLAN_UNDONE
    assert all(op.status == OP_UNDONE for op in repository.plan_operations(plan_id))
    assert sorted(path.name for path in destination.iterdir()) == [unrelated.name]


def test_undo_never_overwrites_a_new_file_at_the_origin(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    service.execute(plan_id)
    newcomer = make_image(root / "a.jpg", seed=777)
    content = newcomer.read_bytes()

    result = service.undo(plan_id)

    assert result["restored"] == 1
    assert result["failed"] == 1
    assert newcomer.read_bytes() == content
    assert repository.get_plan(plan_id).status == PLAN_EXECUTED
    blocked = [op for op in repository.plan_operations(plan_id) if op.status == OP_DONE]
    assert len(blocked) == 1 and "Ya existe" in blocked[0].error
    assert Path(blocked[0].target_path).exists()


def test_plan_cannot_run_twice(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    service.execute(plan_id)

    with pytest.raises(OrganizationError):
        service.execute(plan_id)


def test_new_plan_replaces_previous_draft_only(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    first = service.create_plan(str(tmp_path / "uno"))
    second = service.create_plan(str(tmp_path / "dos"))

    assert repository.get_plan(first) is None
    assert repository.get_plan(second).status == PLAN_DRAFT


def test_folder_can_be_renamed_before_executing(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    destination = tmp_path / "ordenadas"
    plan_id = service.create_plan(str(destination))

    service.retarget_folder(plan_id, "Personas/Boda", "Familia/Boda de Ana")
    service.execute(plan_id)

    assert (destination / "Familia" / "Boda de Ana" / "a.jpg").exists()


@pytest.mark.parametrize("name", ["../fuera", "..", "/", "  "])
def test_folder_rename_cannot_escape_destination(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
    name: str,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))

    with pytest.raises(OrganizationError):
        service.retarget_folder(plan_id, "Personas/Boda", name)


def test_template_with_date_fields(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    ids = add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
    })
    repository.connection.execute(
        "UPDATE images SET date_taken = '2019-03-02T10:00:00' WHERE id = ?", (ids["a.jpg"],)
    )

    plan_id = service.create_plan(str(tmp_path / "ordenadas"), "{year}/{month}/{category}")

    assert repository.plan_operations(plan_id)[0].folder == "2019/03/Personas"


@pytest.mark.parametrize(
    "template",
    ["", "{usuario}", "fijo", "../{category}", "{category}/../..", "{category!r}/{", "{}"],
)
def test_invalid_templates_are_rejected(template: str) -> None:
    with pytest.raises(OrganizationError):
        validate_template(template)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Autos: clásicos", "Autos clásicos"),
        ("a/b\\c", "a b c"),
        ("CON", "Varios"),
        ("com1.txt", "Varios"),
        ("final. ", "final"),
        ("...", "Varios"),
        ("", "Varios"),
    ],
)
def test_safe_component(text: str, expected: str) -> None:
    assert safe_component(text, "Varios") == expected


def test_destination_must_be_absolute(service: OrganizationService) -> None:
    with pytest.raises(OrganizationError):
        service.create_plan("relativa/carpeta")


class _OtherDisk:
    """Hace creer a ``move_file`` que el destino está en otro volumen."""

    def __init__(self, stat_result: os.stat_result) -> None:
        self._stat = stat_result

    def __getattr__(self, name: str):
        return getattr(self._stat, name)

    @property
    def st_dev(self) -> int:
        return self._stat.st_dev + 1


def _simulate_other_disk(monkeypatch: pytest.MonkeyPatch, target_parent: Path) -> None:
    real_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        return _OtherDisk(result) if self == target_parent else result

    monkeypatch.setattr(Path, "stat", fake_stat)


def test_move_across_disks_verifies_before_removing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_image(tmp_path / "a.jpg", seed=1)
    digest = sha256_file(source)
    target = tmp_path / "otro" / "a.jpg"
    target.parent.mkdir()
    _simulate_other_disk(monkeypatch, target.parent)

    move_file(source, target, digest)

    assert target.exists() and not source.exists()
    assert sha256_file(target) == digest


def test_corrupt_copy_across_disks_keeps_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_image(tmp_path / "a.jpg", seed=1)
    content = source.read_bytes()
    target = tmp_path / "otro" / "a.jpg"
    target.parent.mkdir()
    _simulate_other_disk(monkeypatch, target.parent)

    def corrupting_copy(origin: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"copia incompleta")

    monkeypatch.setattr(organization.shutil, "copy2", corrupting_copy)

    with pytest.raises(OSError, match="no coincide"):
        move_file(source, target, sha256_file(source))

    assert source.read_bytes() == content
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_slash_in_a_subcategory_does_not_create_folders(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    add_images(tmp_path / "fotos", repository, config, {
        "a.jpg": ("Personas", "Perro/Gato", 0.9),
        "b.jpg": ("Personas", "Perro/Gato", 0.9),
    })

    plan_id = service.create_plan(str(tmp_path / "ordenadas"))

    assert {op.folder for op in repository.plan_operations(plan_id)} == {"Personas/Perro Gato"}


def test_interrupted_execution_is_recovered_at_startup(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    ids = add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    operations = {Path(op.source_path).name: op for op in repository.plan_operations(plan_id)}
    moved = operations["a.jpg"]
    # Simula un cierre justo después de mover el archivo y antes de anotarlo.
    repository.set_plan_status(plan_id, PLAN_RUNNING)
    move_file(root / "a.jpg", Path(moved.target_path), repository.get_image(ids["a.jpg"]).sha256)

    service.recover_interrupted()

    assert repository.get_plan(plan_id).status == PLAN_EXECUTED
    assert repository.get_operation(moved.id).status == OP_DONE
    assert repository.get_image(ids["a.jpg"]).path == moved.target_path
    assert repository.get_operation(operations["b.jpg"].id).status == OP_PLANNED

    result = service.undo(plan_id)

    assert result["restored"] == 1
    assert (root / "a.jpg").exists()


def test_undo_finishes_a_file_that_was_already_restored(
    tmp_path: Path,
    repository: Repository,
    config: AppConfig,
    service: OrganizationService,
) -> None:
    root = tmp_path / "fotos"
    ids = add_images(root, repository, config, {
        "a.jpg": ("Personas", "Boda", 0.9),
        "b.jpg": ("Personas", "Boda", 0.9),
    })
    plan_id = service.create_plan(str(tmp_path / "ordenadas"))
    service.execute(plan_id)
    image = repository.get_image(ids["a.jpg"])
    Path(image.path).rename(root / "a.jpg")  # Un deshacer anterior se cortó aquí.

    result = service.undo(plan_id)

    assert result == {"restored": 2, "failed": 0, "total": 2}
    assert repository.get_image(ids["a.jpg"]).path == str(root / "a.jpg")
