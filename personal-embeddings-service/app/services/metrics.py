"""Prometheus collectors for the embedding service.

Langfuse export stays in telemetry.py. These collectors are scrape-only.
Labels are endpoint, HTTP status, and model name — never request text.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram
from starlette.exceptions import HTTPException as StarletteHTTPException

# CPU encode of one call is typically a few tenths of a second, up to several
# seconds for a larger batch. Buckets cover about 0.05 s to 10 s.
ENCODE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
TEXT_COUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

embedder_requests_total = Counter(
    "embedder_requests_total",
    "Embedding HTTP requests.",
    labelnames=("endpoint", "status"),
)
embedder_texts_per_request = Histogram(
    "embedder_texts_per_request",
    "Number of texts submitted in one embedding request.",
    buckets=TEXT_COUNT_BUCKETS,
)
embedder_encode_seconds = Histogram(
    "embedder_encode_seconds",
    "Wall time of one encode call, excluding HTTP handling.",
    buckets=ENCODE_BUCKETS,
)
embedder_model_loaded = Gauge(
    "embedder_model_loaded",
    "1 when the named model is loaded, otherwise 0.",
    labelnames=("model",),
)
embedder_model_load_seconds = Gauge(
    "embedder_model_load_seconds",
    "Seconds spent on the most recent model load attempt.",
)


class track_request:
    """Count one embedding request when the handler returns or raises."""

    def __init__(self, endpoint: str, text_count: int) -> None:
        self.endpoint = endpoint
        self.text_count = text_count
        self.status = "500"

    def __enter__(self) -> "track_request":
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is None:
            self.status = "200"
        elif exc is not None and isinstance(exc, StarletteHTTPException):
            self.status = str(exc.status_code)
        embedder_requests_total.labels(
            endpoint=self.endpoint, status=self.status
        ).inc()
        embedder_texts_per_request.observe(self.text_count)
        return False
