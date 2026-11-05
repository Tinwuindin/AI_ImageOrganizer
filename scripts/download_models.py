"""Descarga los pesos de los modelos y el binario de llama.cpp.

Todo queda dentro del proyecto (``models/`` y ``tools/``) con versiones fijadas,
y cada archivo pesado se verifica por SHA-256. Se puede volver a ejecutar sin
riesgo: lo que ya está descargado y verificado no se toca.

    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools"

EMBEDDING_REPO = "google/embeddinggemma-2"
EMBEDDING_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EMBEDDING_WEIGHTS_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"

GEMMA_REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
GEMMA_REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
GEMMA_FILES = {
    "gemma-4-E2B_q4_0-it.gguf": "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
    "gemma-4-E2B-it-mmproj.gguf": "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14",
}

LLAMA_TAG = "b12098"
LLAMA_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
LLAMA_ARCHIVES = {
    f"llama-{LLAMA_TAG}-bin-win-cuda-13.4-x64.zip":
        "6399e31fd2e2f455288ce7d717331441708b21e9e57f1a0bb93233e47d1f398c",
    "cudart-llama-bin-win-cuda-13.4-x64.zip":
        "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise SystemExit(
            f"La suma SHA-256 de {path.name} no coincide "
            f"(esperada {expected}, obtenida {actual}). Borra el archivo y reintenta."
        )


def download_embedding_model() -> None:
    target = MODELS_DIR / "embeddinggemma-2"
    weights = target / "model.safetensors"
    print(f"[1/3] EmbeddingGemma 2 -> {target}")
    snapshot_download(EMBEDDING_REPO, revision=EMBEDDING_REVISION, local_dir=target)
    verify(weights, EMBEDDING_WEIGHTS_SHA256)


def download_gemma() -> None:
    target = MODELS_DIR / "gemma-4-e2b-q4"
    print(f"[2/3] Gemma 4 E2B Q4 -> {target}")
    for filename, expected in GEMMA_FILES.items():
        path = Path(hf_hub_download(GEMMA_REPO, filename, revision=GEMMA_REVISION, local_dir=target))
        verify(path, expected)


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
    partial.replace(destination)


def download_llama_server() -> None:
    target = TOOLS_DIR / "llama.cpp"
    binary = target / "llama-server.exe"
    marker = target / f".version-{LLAMA_TAG}"
    print(f"[3/3] llama.cpp {LLAMA_TAG} -> {target}")
    if binary.exists() and marker.exists():
        return

    # Cada archivo se verifica antes de extraerlo y va a una carpeta limpia.
    archives_dir = TOOLS_DIR / "downloads"
    archives_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, expected in LLAMA_ARCHIVES.items():
        archive = archives_dir / name
        if not archive.exists() or sha256_of(archive) != expected:
            download_file(f"{LLAMA_BASE_URL}/{name}", archive)
        verify(archive, expected)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    if not binary.exists():
        raise SystemExit(f"El paquete de llama.cpp no contiene {binary.name}.")
    marker.touch()
    shutil.rmtree(archives_dir)


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    TOOLS_DIR.mkdir(exist_ok=True)
    download_embedding_model()
    download_gemma()
    download_llama_server()
    print("Modelos y llama-server listos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
