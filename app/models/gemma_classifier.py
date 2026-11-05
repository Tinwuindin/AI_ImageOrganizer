"""Clasificación multimodal con Gemma 4 E2B, servido por ``llama-server``.

Gemma corre en un proceso aparte: al cerrarlo se libera toda su VRAM, y el
servidor limita cada respuesta con un esquema JSON, por lo que la salida siempre
es válida y la categoría siempre pertenece a la lista.

El orden de los campos del esquema importa, y se midió con imágenes reales:

- El modelo escribe qué ve y nombra lo que aparece (``subject``) antes de elegir
  la categoría. Si la elige primero, decide sin haber observado la imagen.
- El nombre del campo también cuenta: con ``subcategory``, el 62 % de las respuestas
  repetía la categoría ("Personas/Personas"); con ``subject``, el 4 %.
- También importa el orden de la lista: tiende a quedarse con la primera opción
  que encaja. Por eso las categorías van de lo específico a lo genérico.

La confianza no se le pregunta al modelo, porque respondía 0.95 acierte o falle.
Es la probabilidad que el propio modelo asignó a los tokens de la categoría.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import math
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import GemmaConfig
from app.core.thumbnails import encode_jpeg, open_normalized
from app.database.models import Category

logger = logging.getLogger(__name__)

_MAX_TAGS = 6
_JPEG_QUALITY = 88
_USER_PROMPT = "Cataloga esta imagen."

_SYSTEM_PROMPT = """\
Eres un catalogador de fotografías personales. Observa la imagen y responde \
únicamente con el JSON solicitado, en español.

Categorías disponibles:
{categories}

Instrucciones:
- description: una frase breve y concreta sobre lo que se ve.
- text_in_image: transcribe el texto legible más importante; cadena vacía si no hay texto.
- tags: de dos a seis palabras clave en minúsculas.
- subject: qué es concretamente lo que aparece, en singular, de una a tres palabras \
(por ejemplo: Perro, Gato, Pizza, Playa, Factura, Placa de circuito, Noticias, Boda). \
Nunca uses el nombre de una categoría.
- category: la categoría de la lista que mejor corresponda al tema. Decide primero \
por el tipo de imagen: una captura de pantalla, una página web, un documento o un \
meme pertenecen a esa categoría aunque muestren personas, edificios u otros temas."""


class GemmaError(RuntimeError):
    """El servidor de Gemma falló o no respondió como se esperaba."""


@dataclass(frozen=True)
class ImageAnalysis:
    category: str
    subcategory: str
    description: str
    tags: tuple[str, ...]
    ocr_text: str
    confidence: float


def build_schema(category_names: Sequence[str]) -> dict[str, Any]:
    """Esquema JSON de la respuesta. El orden de ``properties`` es el de escritura."""
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string", "minLength": 10, "maxLength": 200},
            "text_in_image": {"type": "string", "maxLength": 240},
            "tags": {
                "type": "array",
                "items": {"type": "string", "minLength": 2, "maxLength": 24},
                "minItems": 2,
                "maxItems": _MAX_TAGS,
            },
            "subject": {"type": "string", "minLength": 2, "maxLength": 32},
            "category": {"type": "string", "enum": list(category_names)},
        },
        "required": ["description", "text_in_image", "tags", "subject", "category"],
        "additionalProperties": False,
    }


def build_system_prompt(categories: Sequence[Category]) -> str:
    listing = "\n".join(f"- {item.name}: {item.description}" for item in categories)
    return _SYSTEM_PROMPT.format(categories=listing)


def category_probability(tokens: Sequence[dict[str, Any]], category: str) -> float:
    """Probabilidad conjunta de los tokens que escriben el valor de ``category``.

    Se trabaja con bytes porque un token puede partir un carácter acentuado.
    """
    written = b"".join(bytes(token.get("bytes") or []) for token in tokens)
    key = written.rfind(b'"category"')
    if key < 0:
        return 0.0
    encoded = json.dumps(category, ensure_ascii=False).encode("utf-8")[1:-1]
    start = written.find(encoded, key + len(b'"category"'))
    if start < 0:
        return 0.0
    end = start + len(encoded)
    total = 0.0
    position = 0
    for token in tokens:
        length = len(token.get("bytes") or [])
        if position < end and position + length > start:
            total += float(token["logprob"])
        position += length
    return min(math.exp(total), 1.0)


def parse_analysis(
    content: str,
    category_names: Sequence[str],
    tokens: Sequence[dict[str, Any]],
) -> ImageAnalysis:
    """Convierte la respuesta en un análisis normalizado."""
    try:
        data = json.loads(content)
        category = str(data["category"])
        raw_tags = data["tags"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise GemmaError(f"Respuesta no válida del modelo: {content[:200]!r}") from error
    if category not in category_names:
        raise GemmaError(f"Categoría fuera de la lista: {category!r}")

    tags: list[str] = []
    for tag in raw_tags:
        cleaned = " ".join(str(tag).split()).lower()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    subject = " ".join(str(data.get("subject", "")).split())
    return ImageAnalysis(
        category=category,
        subcategory=subject[:1].upper() + subject[1:],
        description=" ".join(str(data.get("description", "")).split()),
        tags=tuple(tags[:_MAX_TAGS]),
        ocr_text=" ".join(str(data.get("text_in_image", "")).split()),
        confidence=category_probability(tokens, category),
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _kill_with_parent(process: subprocess.Popen[bytes]) -> Any:
    """Asegura que el servidor muera con la aplicación, aun tras un cierre forzado.

    Sin esto, un proceso huérfano seguiría reteniendo varios GB de VRAM. En Windows
    se logra con un Job Object: al cerrarse el último identificador (lo hace el
    sistema cuando termina este proceso), todos sus miembros se terminan.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("No se pudo crear el Job Object de llama-server.")
        return None
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    configured = kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(limits), ctypes.sizeof(limits),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(job, int(process._handle))
    if not assigned:
        logger.warning("No se pudo vincular llama-server a la aplicación.")
    return job


class GemmaClassifier:
    def __init__(self, config: GemmaConfig, llama_server: Path, log_path: Path) -> None:
        self._config = config
        self._llama_server = llama_server
        self._log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._job: Any = None
        self._base_url = ""
        self._client: httpx.Client | None = None
        atexit.register(self.unload)

    @property
    def version(self) -> str:
        return self._config.model_version

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def load(self) -> None:
        if self.is_loaded:
            return
        self.unload()
        for required in (self._llama_server, self._config.model_path, self._config.mmproj_path):
            if not required.exists():
                raise GemmaError(
                    f"No se encontró {required}. Ejecuta: python scripts/download_models.py"
                )
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        arguments = [
            str(self._llama_server),
            "--model", str(self._config.model_path),
            "--mmproj", str(self._config.mmproj_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--gpu-layers", str(self._config.gpu_layers),
            "--ctx-size", str(self._config.context_size),
            "--parallel", "1",  # Una sola GPU: sin paralelismo.
            "--image-min-tokens", str(self._config.image_tokens),
            "--image-max-tokens", str(self._config.image_tokens),
            "--reasoning", "off",
            "--cache-ram", "0",
            "--no-webui",
        ]
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("wb") as log:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        self._job = _kill_with_parent(self._process)
        self._client = httpx.Client(timeout=self._config.request_timeout)
        try:
            self._wait_until_ready()
        except GemmaError:
            self.unload()
            raise
        logger.info("Gemma listo en %s", self._base_url)

    def _wait_until_ready(self) -> None:
        assert self._client is not None and self._process is not None
        deadline = time.monotonic() + self._config.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise GemmaError(
                    f"llama-server terminó con código {self._process.returncode}. "
                    f"Revisa {self._log_path}."
                )
            try:
                if self._client.get(f"{self._base_url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise GemmaError(f"llama-server no respondió a tiempo. Revisa {self._log_path}.")

    def unload(self) -> None:
        """Detiene el servidor. Al salir el proceso, Windows libera toda su VRAM."""
        if self._client is not None:
            self._client.close()
            self._client = None
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._job = None

    def analyze(self, image_path: Path, categories: Sequence[Category]) -> ImageAnalysis:
        """Analiza una imagen. Reinicia el servidor si se cayó y reintenta."""
        image = open_normalized(image_path, self._config.image_max_side)
        payload = base64.b64encode(encode_jpeg(image, _JPEG_QUALITY)).decode("ascii")
        names = [item.name for item in categories]
        body = self._request_body(payload, categories, names)

        last_error: Exception | None = None
        for attempt in range(1 + self._config.max_retries):
            try:
                self.load()
                content, tokens = self._complete(body)
                return parse_analysis(content, names, tokens)
            except (httpx.HTTPError, GemmaError) as error:
                last_error = error
                logger.warning("Fallo de Gemma (intento %d): %s", attempt + 1, error)
                if not self.is_loaded:
                    self.unload()
        raise GemmaError(f"Gemma no pudo analizar la imagen: {last_error}")

    def _request_body(
        self, image_base64: str, categories: Sequence[Category], names: Sequence[str]
    ) -> dict[str, Any]:
        # El sistema va primero y es fijo: llama-server reutiliza su evaluación
        # entre imágenes, así que cada petición solo procesa la imagen.
        return {
            "messages": [
                {"role": "system", "content": build_system_prompt(categories)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "cache_prompt": True,
            "logprobs": True,
            "top_logprobs": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "image_analysis",
                    "strict": True,
                    "schema": build_schema(names),
                },
            },
        }

    def _complete(self, body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        assert self._client is not None
        response = self._client.post(f"{self._base_url}/v1/chat/completions", json=body)
        if response.status_code != 200:
            raise GemmaError(f"llama-server respondió {response.status_code}: {response.text[:300]}")
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise GemmaError("La respuesta se truncó: aumenta gemma.max_tokens.")
        tokens = (choice.get("logprobs") or {}).get("content") or []
        return str(choice["message"]["content"]), tokens
