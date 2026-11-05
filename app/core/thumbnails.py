"""Normalización de imágenes y miniaturas en disco."""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

_JPEG_QUALITY = 82
_BACKGROUND = (255, 255, 255)


def configure_pillow(max_image_pixels: int) -> None:
    """Fija el límite de píxeles que Pillow acepta decodificar."""
    Image.MAX_IMAGE_PIXELS = max_image_pixels


def _flatten(image: Image.Image) -> Image.Image:
    """Pasa a RGB y compone la transparencia sobre fondo blanco."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, _BACKGROUND)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image.convert("RGB")


def open_normalized(path: Path, max_side: int) -> Image.Image:
    """Abre una imagen en RGB, con la orientación EXIF aplicada y reducida.

    ``draft`` hace que el decodificador JPEG reduzca desde el dominio de la DCT:
    una foto de 20 MP se abre varias veces más rápido que decodificándola entera.
    """
    with Image.open(path) as source:
        source.draft("RGB", (max_side, max_side))
        image = ImageOps.exif_transpose(source)
        image = _flatten(image)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def encode_jpeg(image: Image.Image, quality: int = _JPEG_QUALITY) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def thumbnail_path(thumbnails_dir: Path, sha256: str) -> Path:
    """Ruta direccionada por contenido: sobrevive a mover o renombrar el archivo."""
    return thumbnails_dir / sha256[:2] / f"{sha256}.jpg"


def ensure_thumbnail(source: Path, sha256: str, thumbnails_dir: Path, size: int) -> Path:
    """Genera la miniatura si no existe, de forma atómica."""
    target = thumbnail_path(thumbnails_dir, sha256)
    if target.exists():
        return target
    image = open_normalized(source, size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.{id(image)}.tmp")
    try:
        temporary.write_bytes(encode_jpeg(image))
        try:
            temporary.replace(target)
        except PermissionError:
            # Dos copias idénticas comparten miniatura. Si otro hilo ganó la carrera,
            # Windows rechaza el reemplazo, pero el resultado ya está en su lugar.
            if not target.exists():
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return target
