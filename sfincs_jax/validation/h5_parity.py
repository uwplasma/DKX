"""Strict HDF5 parity checks for SFINCS-style output files.

The high-level :mod:`sfincs_jax.compare` helper intentionally includes
SFINCS-specific tolerance and skip rules for known Fortran/JAX convention
differences.  This module is stricter and campaign-oriented: it records missing
datasets, shape mismatches, and max errors for every numeric dataset so parity
audits can detect output-contract regressions early.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


@dataclass(frozen=True)
class H5DatasetParity:
    """Parity status for one numeric HDF5 dataset."""

    key: str
    status: str
    reference_shape: tuple[int, ...] | None
    candidate_shape: tuple[int, ...] | None
    max_abs: float | None
    max_rel: float | None
    atol: float
    rtol: float

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "extra_in_candidate", "non_numeric"}

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_shape"] = list(self.reference_shape) if self.reference_shape is not None else None
        payload["candidate_shape"] = list(self.candidate_shape) if self.candidate_shape is not None else None
        return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _is_numeric_array(array: np.ndarray) -> bool:
    return array.dtype.kind in {"b", "i", "u", "f", "c"}


def _numeric_datasets(path: Path) -> dict[str, np.ndarray]:
    """Read all numeric datasets from an HDF5 file into memory."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    datasets: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            value = np.asarray(obj[...])
            if _is_numeric_array(value):
                datasets[name] = value

        h5.visititems(visit)
    return datasets


def _dataset_tolerance(
    key: str,
    *,
    atol: float,
    rtol: float,
    tolerances: Mapping[str, Mapping[str, float]] | None,
) -> tuple[float, float]:
    if not tolerances or key not in tolerances:
        return atol, rtol
    local = tolerances[key]
    return float(local.get("atol", atol)), float(local.get("rtol", rtol))


def _max_errors(reference: np.ndarray, candidate: np.ndarray, *, atol: float) -> tuple[float, float]:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    absdiff = np.abs(cand - ref)
    max_abs = float(np.nanmax(absdiff)) if absdiff.size else 0.0
    denom_floor = max(float(atol), np.finfo(np.float64).tiny)
    denom = np.maximum(np.abs(ref), denom_floor)
    rel = absdiff / denom
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    return max_abs, max_rel


def compare_h5_outputs(
    *,
    reference_path: Path,
    candidate_path: Path,
    keys: Iterable[str] | None = None,
    ignore_keys: Iterable[str] = (),
    include_extra: bool = True,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    tolerances: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Compare numeric datasets from two HDF5 files and return a JSON-ready report.

    Parameters
    ----------
    reference_path:
        Baseline HDF5 file, usually a SFINCS Fortran v3 output.
    candidate_path:
        Candidate HDF5 file, usually a SFINCS-JAX output.
    keys:
        Optional explicit dataset keys to compare.  If omitted, all keys from
        the reference are compared.
    ignore_keys:
        Dataset keys to skip.
    include_extra:
        Whether to record candidate-only numeric datasets as non-failing extras.
    atol, rtol:
        Default absolute and relative tolerances.
    tolerances:
        Per-dataset overrides, e.g. ``{"FSABjHat": {"rtol": 1e-8}}``.
    """
    reference = _numeric_datasets(reference_path)
    candidate = _numeric_datasets(candidate_path)
    ignored = set(ignore_keys)
    selected = set(keys) if keys is not None else set(reference)
    selected -= ignored

    results: list[H5DatasetParity] = []
    for key in sorted(selected):
        local_atol, local_rtol = _dataset_tolerance(key, atol=atol, rtol=rtol, tolerances=tolerances)
        if key not in reference:
            results.append(H5DatasetParity(key, "missing_in_reference", None, None, None, None, local_atol, local_rtol))
            continue
        if key not in candidate:
            results.append(
                H5DatasetParity(key, "missing_in_candidate", reference[key].shape, None, None, None, local_atol, local_rtol)
            )
            continue
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            results.append(
                H5DatasetParity(key, "shape_mismatch", ref.shape, cand.shape, None, None, local_atol, local_rtol)
            )
            continue
        max_abs, max_rel = _max_errors(ref, cand, atol=local_atol)
        ok = bool(np.allclose(ref, cand, atol=local_atol, rtol=local_rtol, equal_nan=True))
        results.append(H5DatasetParity(key, "ok" if ok else "value_mismatch", ref.shape, cand.shape, max_abs, max_rel, local_atol, local_rtol))

    if include_extra:
        for key in sorted(set(candidate) - set(reference) - ignored):
            local_atol, local_rtol = _dataset_tolerance(key, atol=atol, rtol=rtol, tolerances=tolerances)
            results.append(
                H5DatasetParity(key, "extra_in_candidate", None, candidate[key].shape, None, None, local_atol, local_rtol)
            )

    failing = [
        item
        for item in results
        if item.status
        in {
            "missing_in_reference",
            "missing_in_candidate",
            "shape_mismatch",
            "value_mismatch",
        }
    ]
    compared = [item for item in results if item.status in {"ok", "value_mismatch"}]
    max_abs = max((item.max_abs or 0.0 for item in compared), default=0.0)
    max_rel = max((item.max_rel or 0.0 for item in compared), default=0.0)
    counts: dict[str, int] = {}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1

    return {
        "schema_version": 1,
        "reference_path": str(Path(reference_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "overall_status": "pass" if not failing else "fail",
        "numeric_reference_dataset_count": len(reference),
        "numeric_candidate_dataset_count": len(candidate),
        "compared_dataset_count": len(compared),
        "failing_dataset_count": len(failing),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "status_counts": counts,
        "datasets": [item.to_json() for item in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict numeric HDF5 parity check.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--ignore-key", action="append", default=[])
    parser.add_argument("--key", action="append", default=None, help="Dataset key to compare; may be repeated")
    parser.add_argument("--no-extra", action="store_true", help="Do not report candidate-only datasets")
    args = parser.parse_args(argv)

    report = compare_h5_outputs(
        reference_path=args.reference,
        candidate_path=args.candidate,
        keys=args.key,
        ignore_keys=args.ignore_key,
        include_extra=not args.no_extra,
        atol=args.atol,
        rtol=args.rtol,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
