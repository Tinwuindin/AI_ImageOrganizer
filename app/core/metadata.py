"""Lectura de dimensiones y fecha EXIF."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

_TAG_ORIENTATION = 0x0112
_TAG_DATETIME = 0x0132
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME_DIGITIZED = 0x9004

# En estas orientaciones la imagen se muestra girada 90°: ancho y alto se invierten.
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    date_taken: str | None


def parse_exif_date(value: object) -> str | None:
    """Convierte una fecha EXIF (``2023:05:17 14:03:22``) a ISO 8601."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    if not isinstance(value, str):
        return None
    text = value.strip().strip("\x00")
    try:
        return datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _date_taken(exif: Image.Exif) -> str | None:
    try:
        details = exif.get_ifd(ExifTags.IFD.Exif)
    except (OSError, ValueError, KeyError, TypeError):
        details = {}
    candidates = (
        details.get(_TAG_DATETIME_ORIGINAL),
        details.get(_TAG_DATETIME_DIGITIZED),
        exif.get(_TAG_DATETIME),
    )
    for candidate in candidates:
        parsed = parse_exif_date(candidate)
        if parsed:
            return parsed
    return None


def read_metadata(path: Path) -> ImageMetadata:
    """Lee los metadatos. Un EXIF dañado no invalida la imagen."""
    with Image.open(path) as image:
        width, height = image.size
        try:
            exif = image.getexif()
        except (OSError, ValueError, SyntaxError):
            return ImageMetadata(width, height, None)
    if exif.get(_TAG_ORIENTATION) in _ROTATED_ORIENTATIONS:
        width, height = height, width
    return ImageMetadata(width, height, _date_taken(exif))
