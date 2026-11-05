from __future__ import annotations

import os
import threading
from pathlib import Path

import pillow_heif
import pytest
from PIL import ExifTags, Image

from app.core.config import AppConfig
from app.core.metadata import parse_exif_date, read_metadata
from app.core.scanner import iter_image_files, scan_directories
from app.core.thumbnails import ensure_thumbnail, open_normalized, thumbnail_path
from app.database.models import STATUS_ERROR, STATUS_MISSING, STATUS_SCANNED, Classification
from app.database.repository import Repository
from tests.conftest import make_image


@pytest.fixture
def repository(config: AppConfig) -> Repository:
    return Repository(config.database_path)


def scan(root: Path, repository: Repository, config: AppConfig, **kwargs):
    return scan_directories([root], repository, config.scanner, config.thumbnails_dir, **kwargs)


def test_walks_recursively_and_filters_files(tmp_path: Path) -> None:
    root = tmp_path / "fotos"
    make_image(root / "a.jpg")
    make_image(root / "viaje" / "2023" / "b.PNG")
    (root / "notas.txt").write_text("hola")
    make_image(root / "$RECYCLE.BIN" / "borrada.jpg")
    make_image(root / "miniaturas" / "c.jpg")

    found = list(iter_image_files(
        [root], [".jpg", ".png"], ["$RECYCLE.BIN"], [root / "miniaturas"]
    ))

    assert sorted(path.name for path in found) == ["a.jpg", "b.PNG"]


def test_overlapping_roots_visit_each_folder_once(tmp_path: Path) -> None:
    root = tmp_path / "fotos"
    make_image(root / "viaje" / "a.jpg")

    found = list(iter_image_files([root, root / "viaje"], [".jpg"], []))

    assert len(found) == 1


def test_scan_registers_hash_metadata_and_thumbnail(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = "2021:07:04 18:30:00"
    make_image(root / "foto.jpg", size=(800, 600), exif=exif)

    summary = scan(root, repository, config)

    assert summary.added == 1
    record = next(iter(repository.all_images().values()))
    assert (record.width, record.height) == (800, 600)
    assert record.date_taken == "2021-07-04T18:30:00"
    assert record.status == STATUS_SCANNED
    assert len(record.sha256) == 64
    assert thumbnail_path(config.thumbnails_dir, record.sha256).exists()


def test_orientation_swaps_reported_dimensions(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    path = make_image(tmp_path / "vertical.jpg", size=(800, 600), exif=exif)

    metadata = read_metadata(path)

    assert (metadata.width, metadata.height) == (600, 800)
    assert open_normalized(path, 400).size == (300, 400)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023:05:17 14:03:22", "2023-05-17T14:03:22"),
        (b"2023:05:17 14:03:22\x00", "2023-05-17T14:03:22"),
        ("0000:00:00 00:00:00", None),
        ("mañana", None),
        (None, None),
    ],
)
def test_parse_exif_date(value: object, expected: str | None) -> None:
    assert parse_exif_date(value) == expected


def test_corrupt_file_is_recorded_without_stopping_the_scan(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    (root).mkdir()
    (root / "dañada.jpg").write_bytes(b"esto no es una imagen")
    make_image(root / "buena.jpg")

    summary = scan(root, repository, config)

    assert summary.added == 2
    assert summary.failed == 1
    records = {record.filename: record for record in repository.all_images().values()}
    assert records["dañada.jpg"].status == STATUS_ERROR
    assert "no válida" in records["dañada.jpg"].error
    assert records["buena.jpg"].status == STATUS_SCANNED


def test_truncated_jpeg_is_reported_as_error(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "cortada.jpg", seed=1)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    scan(root, repository, config)

    record = next(iter(repository.all_images().values()))
    assert record.status == STATUS_ERROR


def test_transparent_png_and_unicode_path(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos de años"
    root.mkdir()
    image = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    image.save(root / "ñandú.png")

    scan(root, repository, config)

    record = next(iter(repository.all_images().values()))
    assert record.status == STATUS_SCANNED
    # La transparencia se compone sobre blanco, no sobre negro.
    assert open_normalized(Path(record.path), 32).getpixel((0, 0)) == (255, 255, 255)


def test_heic_files_are_supported(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    root.mkdir()
    pillow_heif.from_pillow(Image.new("RGB", (320, 240), (10, 200, 10))).save(root / "iphone.heic")

    scan(root, repository, config)

    record = next(iter(repository.all_images().values()))
    assert record.status == STATUS_SCANNED
    assert (record.width, record.height) == (320, 240)


def test_rescan_skips_unchanged_files(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    make_image(root / "a.jpg")
    scan(root, repository, config)

    summary = scan(root, repository, config)

    assert summary.unchanged == 1
    assert summary.added == summary.changed == 0


def test_modified_file_loses_its_analysis(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "a.jpg", color=(10, 10, 10))
    scan(root, repository, config)
    record = next(iter(repository.all_images().values()))
    repository.save_classifications([Classification(record.id, "Otros", confidence=0.9)])

    make_image(path, color=(240, 240, 240), seed=2)
    summary = scan(root, repository, config)

    assert summary.changed == 1
    updated = repository.get_image(record.id)
    assert updated.status == STATUS_SCANNED
    assert updated.sha256 != record.sha256
    assert repository.get_classification(record.id) is None


def test_touched_file_keeps_its_analysis(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "a.jpg")
    scan(root, repository, config)
    record = next(iter(repository.all_images().values()))
    repository.save_classifications([Classification(record.id, "Otros", confidence=0.9)])

    os.utime(path, (1_000_000_000, 1_000_000_000))
    summary = scan(root, repository, config)

    assert summary.changed == 0
    assert repository.get_classification(record.id) is not None


def test_moved_file_keeps_identity_and_analysis(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "a.jpg")
    scan(root, repository, config)
    record = next(iter(repository.all_images().values()))
    repository.save_classifications([Classification(record.id, "Otros", confidence=0.9)])

    new_path = root / "movida" / "renombrada.jpg"
    new_path.parent.mkdir()
    path.rename(new_path)
    summary = scan(root, repository, config)

    assert summary.relocated == 1
    assert summary.added == summary.missing == 0
    assert len(repository.all_images()) == 1
    updated = repository.get_image(record.id)
    assert updated.path == str(new_path)
    assert updated.filename == "renombrada.jpg"
    assert repository.get_classification(record.id) is not None


def test_deleted_file_is_marked_missing_and_can_return(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "a.jpg", seed=3)
    content = path.read_bytes()
    stat = path.stat()
    scan(root, repository, config)
    record = next(iter(repository.all_images().values()))

    path.unlink()
    summary = scan(root, repository, config)
    assert summary.missing == 1
    assert repository.get_image(record.id).status == STATUS_MISSING

    path.write_bytes(content)
    os.utime(path, (stat.st_atime, stat.st_mtime))
    scan(root, repository, config)
    assert repository.get_image(record.id).status == STATUS_SCANNED


def test_missing_detection_only_applies_inside_scanned_roots(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    make_image(tmp_path / "uno" / "a.jpg")
    make_image(tmp_path / "dos" / "b.jpg", seed=4)
    scan(tmp_path / "uno", repository, config)
    scan(tmp_path / "dos", repository, config)

    (tmp_path / "uno" / "a.jpg").unlink()
    summary = scan(tmp_path / "dos", repository, config)

    assert summary.missing == 0


def test_cancelled_scan_does_not_mark_anything_missing(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "a.jpg")
    scan(root, repository, config)
    path.unlink()
    make_image(root / "b.jpg", seed=5)
    cancel = threading.Event()
    cancel.set()

    summary = scan(root, repository, config, cancel=cancel)

    assert summary.cancelled
    assert summary.missing == 0


def test_identical_copies_racing_for_one_thumbnail_are_not_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_image(tmp_path / "a.jpg")
    thumbnails = tmp_path / "miniaturas"
    target = thumbnail_path(thumbnails, "ab" * 32)
    real_replace = Path.replace

    def losing_replace(self: Path, destination: Path) -> Path:
        # Otro hilo escribió la misma miniatura y Windows niega el reemplazo.
        destination.write_bytes(b"miniatura del otro hilo")
        raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr(Path, "replace", losing_replace)

    result = ensure_thumbnail(source, "ab" * 32, thumbnails, 64)

    monkeypatch.setattr(Path, "replace", real_replace)
    assert result == target
    assert target.read_bytes() == b"miniatura del otro hilo"
    assert [path.name for path in target.parent.iterdir()] == [target.name]


def test_renaming_only_the_case_keeps_a_single_record(
    tmp_path: Path, repository: Repository, config: AppConfig
) -> None:
    root = tmp_path / "fotos"
    path = make_image(root / "Foto.JPG", seed=8)
    scan(root, repository, config)
    record = next(iter(repository.all_images().values()))

    renamed = root / "foto.jpg"
    path.rename(renamed)
    summary = scan(root, repository, config)

    assert summary.added == 0
    assert list(repository.all_images()) == [str(renamed)]
    assert repository.get_image(record.id).filename == "foto.jpg"
