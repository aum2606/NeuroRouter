"""Small monotonic timing utilities for telemetry."""

from time import perf_counter


class Timer:
    """Context manager exposing elapsed milliseconds."""

    def __init__(self) -> None:
        self._started: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "Timer":
        self._started = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self._started is not None:
            self.elapsed_ms = (perf_counter() - self._started) * 1000
