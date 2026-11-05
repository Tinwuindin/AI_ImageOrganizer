"""Inicia el organizador de imágenes en http://127.0.0.1:8765."""

from __future__ import annotations

import logging
import threading
import webbrowser

import uvicorn

from app.core.config import load_config
from app.main import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    app = create_app(config)
    url = f"http://{config.server.host}:{config.server.port}"
    print(f"Organizador de imágenes en {url}  (Ctrl+C para salir)")
    if config.server.open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="warning")


if __name__ == "__main__":
    main()
