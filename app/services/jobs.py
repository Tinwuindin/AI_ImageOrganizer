"""Tareas largas en segundo plano.

Solo corre una a la vez: el análisis y los movimientos de archivos no deben
pisarse, y hay una sola GPU. La interfaz consulta el avance con ``snapshot``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_CANCELLING = "cancelling"
STATE_COMPLETED = "completed"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"


class JobBusyError(RuntimeError):
    """Ya hay una tarea en curso."""


class JobContext:
    """Lo que una tarea usa para informar su avance y detectar la cancelación."""

    def __init__(self, runner: JobRunner) -> None:
        self._runner = runner
        self.cancel = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def stage(self, name: str, total: int = 0) -> None:
        self._runner.begin_stage(name, total)

    def advance(self, done: int, total: int | None = None) -> None:
        self._runner.advance(done, total)

    def note(self, key: str, value: Any) -> None:
        """Agrega un dato al resumen que verá la interfaz."""
        self._runner.note(key, value)


class JobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._context: JobContext | None = None
        self._kind: str | None = None
        self._state = STATE_IDLE
        self._stage: str | None = None
        self._done = 0
        self._total = 0
        self._started_at: float | None = None
        self._stage_started_at: float | None = None
        self._finished_at: float | None = None
        self._error: str | None = None
        self._summary: dict[str, Any] = {}

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state in (STATE_RUNNING, STATE_CANCELLING)

    def start(
        self,
        kind: str,
        target: Callable[[JobContext], dict[str, Any] | None],
    ) -> None:
        with self._lock:
            if self._state in (STATE_RUNNING, STATE_CANCELLING):
                raise JobBusyError("Ya hay una tarea en curso.")
            context = JobContext(self)
            self._context = context
            self._kind = kind
            self._state = STATE_RUNNING
            self._stage = None
            self._done = self._total = 0
            self._started_at = self._stage_started_at = time.monotonic()
            self._finished_at = None
            self._error = None
            self._summary = {}
            self._thread = threading.Thread(
                target=self._run, args=(context, target), name=f"job-{kind}", daemon=True
            )
            self._thread.start()

    def _run(
        self,
        context: JobContext,
        target: Callable[[JobContext], dict[str, Any] | None],
    ) -> None:
        state = STATE_COMPLETED
        error: str | None = None
        result: dict[str, Any] | None = None
        try:
            result = target(context)
            if context.cancelled:
                state = STATE_CANCELLED
        except Exception as exception:  # La tarea debe terminar en un estado visible.
            logger.exception("La tarea falló")
            state = STATE_FAILED
            error = str(exception) or type(exception).__name__
        with self._lock:
            self._state = state
            self._error = error
            self._finished_at = time.monotonic()
            if result:
                self._summary.update(result)

    def cancel(self) -> bool:
        with self._lock:
            if self._state != STATE_RUNNING or self._context is None:
                return False
            self._state = STATE_CANCELLING
            self._context.cancel.set()
            return True

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def begin_stage(self, name: str, total: int) -> None:
        with self._lock:
            self._stage = name
            self._done = 0
            self._total = total
            self._stage_started_at = time.monotonic()

    def advance(self, done: int, total: int | None) -> None:
        with self._lock:
            self._done = done
            if total is not None:
                self._total = total

    def note(self, key: str, value: Any) -> None:
        with self._lock:
            self._summary[key] = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._finished_at or time.monotonic()
            elapsed = now - self._started_at if self._started_at else 0.0
            stage_elapsed = now - self._stage_started_at if self._stage_started_at else 0.0
            eta = None
            rate = None
            if self._state == STATE_RUNNING and self._done > 0 and stage_elapsed > 0:
                rate = self._done / stage_elapsed
                eta = max(self._total - self._done, 0) / rate
            return {
                "kind": self._kind,
                "state": self._state,
                "stage": self._stage,
                "done": self._done,
                "total": self._total,
                "elapsed_seconds": round(elapsed, 1),
                "rate_per_second": round(rate, 2) if rate else None,
                "eta_seconds": round(eta) if eta is not None else None,
                "error": self._error,
                "summary": dict(self._summary),
            }
