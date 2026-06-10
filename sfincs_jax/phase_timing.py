"""Small JSON-friendly phase timing helpers for benchmark and audit scripts."""

from __future__ import annotations

import contextlib
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


def maxrss_mb(*, platform: str = sys.platform, raw_value: int | None = None) -> float:
    """Return process maximum resident set size in MB.

    ``resource.ru_maxrss`` is reported in bytes on macOS and kilobytes on most
    Linux systems, so normalize it here for all run-audit JSON artifacts.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_value is None else int(raw_value)
    if str(platform).startswith("darwin"):
        return float(raw) / (1024.0 * 1024.0)
    return float(raw) / 1024.0


@dataclass
class PhaseRecord:
    """One timed phase in a benchmark or audit run."""

    name: str
    elapsed_s: float
    status: str = "ok"
    maxrss_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PhaseTimer:
    """Collect bounded phase timings for JSON run reports."""

    def __init__(self) -> None:
        self._start_s = time.perf_counter()
        self.records: list[PhaseRecord] = []

    @contextlib.contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[None]:
        start_s = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.records.append(
                PhaseRecord(
                    name=name,
                    elapsed_s=round(max(0.0, time.perf_counter() - start_s), 6),
                    status=status,
                    maxrss_mb=round(maxrss_mb(), 6),
                    metadata=dict(metadata),
                )
            )

    def summary(self) -> dict[str, Any]:
        elapsed = max(0.0, time.perf_counter() - self._start_s)
        return {
            "elapsed_s": round(elapsed, 6),
            "maxrss_mb": round(maxrss_mb(), 6),
            "phase_count": len(self.records),
            "phases": [record.to_json() for record in self.records],
        }
