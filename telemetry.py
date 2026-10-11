from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from models import IntegrationResult


def _bounded(value: Any, limit: int = 1000) -> str:
    rendered = json.dumps(value, default=str, separators=(",", ":"))
    return rendered[:limit]


@dataclass
class Telemetry:
    run_id: str
    model: str
    configured: bool = field(init=False)
    latency_ms: dict[str, float] = field(default_factory=dict)
    _sdk: Any = field(init=False, default=None)
    _error: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        dsn = os.getenv("SENTRY_DSN")
        self.configured = bool(dsn)
        if not dsn:
            return
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=dsn,
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
                send_default_pii=False,
                auto_session_tracking=False,
            )
            sentry_sdk.set_tag("run_id", self.run_id)
            sentry_sdk.set_tag("model", self.model)
            self._sdk = sentry_sdk
        except Exception as exc:  # telemetry must not hide pipeline output
            self._error = f"Sentry initialization failed: {exc}"

    @contextmanager
    def trace(self) -> Iterator[None]:
        if not self._sdk:
            yield
            return
        with self._sdk.start_transaction(op="hermeneus.pipeline", name="Hermeneus Core run") as transaction:
            transaction.set_data("run_id", self.run_id)
            transaction.set_data("model", self.model)
            yield

    @contextmanager
    def stage(self, name: str, summary: Any | None = None) -> Iterator[None]:
        start = time.perf_counter()
        span_context = self._sdk.start_span(op=f"hermeneus.{name}", description=name) if self._sdk else None
        try:
            if span_context:
                with span_context as span:
                    span.set_data("run_id", self.run_id)
                    if summary is not None:
                        span.set_data("summary", _bounded(summary))
                    yield
                    span.set_data("stage_status", "ok")
            else:
                yield
        except Exception as exc:
            if self._sdk:
                self._sdk.capture_exception(exc)
            raise
        finally:
            duration = (time.perf_counter() - start) * 1000
            self.latency_ms[name] = round(self.latency_ms.get(name, 0.0) + duration, 3)

    def annotate(self, summary: Any) -> None:
        """Attach a bounded, sanitized summary to the currently active span."""
        if not self._sdk:
            return
        span = self._sdk.get_current_span()
        if span:
            span.set_data("result_summary", _bounded(summary))

    def finish(self) -> IntegrationResult:
        if not self.configured:
            return IntegrationResult(configured=False, succeeded=False, message="SENTRY_DSN is not set; no trace was sent")
        if self._error or not self._sdk:
            return IntegrationResult(configured=True, succeeded=False, message=self._error or "Sentry is unavailable")
        try:
            flushed = self._sdk.flush(timeout=5)
            message = "Sentry trace flushed" if flushed is not False else "Sentry flush timed out"
            return IntegrationResult(configured=True, succeeded=flushed is not False, message=message)
        except Exception as exc:
            return IntegrationResult(configured=True, succeeded=False, message=f"Sentry flush failed: {exc}")
