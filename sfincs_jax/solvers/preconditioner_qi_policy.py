"""Promotion gates for QI hard-seed and production ladder artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class QIRunEvidence:
    """Minimal QI run evidence required for policy decisions."""

    seed: int
    backend: str
    resolution: tuple[int, int, int, int]
    converged: bool
    residual_ratio: float
    timed_out: bool = False
    output_written: bool = True
    solver_trace_written: bool = True
    observable_rel_diff: float = 0.0
    host_fallback_used: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly run evidence."""

        return asdict(self)


@dataclass(frozen=True)
class QILadderPromotionResult:
    """Fail-closed QI ladder policy decision."""

    promoted: bool
    required_seeds: tuple[int, ...]
    present_seeds: tuple[int, ...]
    backends: tuple[str, ...]
    min_resolution: tuple[int, int, int, int]
    max_residual_ratio: float
    max_observable_rel_diff: float
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly policy metadata."""

        return asdict(self)


def _normalize_backend(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_evidence(value: QIRunEvidence | Mapping[str, object]) -> QIRunEvidence:
    if isinstance(value, QIRunEvidence):
        return value
    resolution_raw = value.get("resolution", ())
    resolution = tuple(int(v) for v in resolution_raw)  # type: ignore[arg-type]
    if len(resolution) != 4:
        raise ValueError("QI run evidence resolution must have four entries")
    return QIRunEvidence(
        seed=int(value.get("seed", 0)),
        backend=_normalize_backend(value.get("backend", "")),
        resolution=resolution,  # type: ignore[arg-type]
        converged=bool(value.get("converged", False)),
        residual_ratio=float(value.get("residual_ratio", float("inf"))),
        timed_out=bool(value.get("timed_out", False)),
        output_written=bool(value.get("output_written", False)),
        solver_trace_written=bool(value.get("solver_trace_written", False)),
        observable_rel_diff=float(value.get("observable_rel_diff", float("inf"))),
        host_fallback_used=bool(value.get("host_fallback_used", False)),
    )


def evaluate_qi_production_ladder_promotion(
    runs: list[QIRunEvidence | Mapping[str, object]]
    | tuple[QIRunEvidence | Mapping[str, object], ...],
    *,
    required_seeds: tuple[int, ...],
    required_backends: tuple[str, ...] = ("cpu", "gpu"),
    min_resolution: tuple[int, int, int, int] = (15, 31, 60, 5),
    max_residual_ratio: float = 1.0e-6,
    max_observable_rel_diff: float = 1.0e-8,
    allow_host_fallback: bool = False,
) -> QILadderPromotionResult:
    """Evaluate whether QI ladder evidence is strong enough for policy.

    A production-resolution QI claim requires every requested seed/backend pair
    to converge, write output and trace artifacts, meet the residual/observable
    gates, and run on the true device path unless a non-autodiff host fallback
    claim is explicitly requested.
    """

    evidence = tuple(_normalize_evidence(run) for run in runs)
    required_seed_set = {int(seed) for seed in required_seeds}
    required_backend_set = {
        _normalize_backend(backend) for backend in required_backends
    }
    min_resolution_use = tuple(int(v) for v in min_resolution)
    if len(min_resolution_use) != 4:
        raise ValueError("min_resolution must have four entries")

    by_key = {(int(run.seed), _normalize_backend(run.backend)): run for run in evidence}
    failures: list[str] = []
    warnings: list[str] = []
    for seed in sorted(required_seed_set):
        for backend in sorted(required_backend_set):
            run = by_key.get((seed, backend))
            if run is None:
                failures.append(f"missing seed={seed} backend={backend}")
                continue
            resolution = tuple(int(v) for v in run.resolution)
            if any(
                got < need
                for got, need in zip(resolution, min_resolution_use, strict=True)
            ):
                failures.append(
                    f"seed={seed} backend={backend} resolution={run.resolution} below {min_resolution_use}"
                )
            if run.timed_out:
                failures.append(f"seed={seed} backend={backend} timed out")
            if not run.converged:
                failures.append(f"seed={seed} backend={backend} did not converge")
            if not run.output_written:
                failures.append(f"seed={seed} backend={backend} did not write output")
            if not run.solver_trace_written:
                failures.append(
                    f"seed={seed} backend={backend} did not write solver trace"
                )
            if float(run.residual_ratio) > float(max_residual_ratio):
                failures.append(
                    f"seed={seed} backend={backend} residual_ratio={run.residual_ratio:.3g} "
                    f"above {float(max_residual_ratio):.3g}"
                )
            if float(run.observable_rel_diff) > float(max_observable_rel_diff):
                failures.append(
                    f"seed={seed} backend={backend} observable_rel_diff={run.observable_rel_diff:.3g} "
                    f"above {float(max_observable_rel_diff):.3g}"
                )
            if run.host_fallback_used and not bool(allow_host_fallback):
                failures.append(f"seed={seed} backend={backend} used host fallback")
            elif run.host_fallback_used:
                warnings.append(f"seed={seed} backend={backend} used host fallback")

    present_seeds = tuple(sorted({int(run.seed) for run in evidence}))
    backends = tuple(sorted({_normalize_backend(run.backend) for run in evidence}))
    return QILadderPromotionResult(
        promoted=not failures,
        required_seeds=tuple(sorted(required_seed_set)),
        present_seeds=present_seeds,
        backends=backends,
        min_resolution=min_resolution_use,  # type: ignore[arg-type]
        max_residual_ratio=float(max_residual_ratio),
        max_observable_rel_diff=float(max_observable_rel_diff),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


__all__ = [
    "QILadderPromotionResult",
    "QIRunEvidence",
    "evaluate_qi_production_ladder_promotion",
]
