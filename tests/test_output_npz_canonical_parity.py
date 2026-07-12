"""Canonical ``.npz`` output parity for the RHSMode=1/2/3 run drivers.

The canonical writer (:mod:`sfincs_jax.writer`) now emits ``.npz`` archives
directly, so ``sfincs_jax write-output --out *.npz`` no longer falls back to the
retired legacy pipeline.  These tests pin two
properties on a tiny RHSMode=1 case and a tiny RHSMode=3 case:

* the canonical ``.npz`` datasets equal the Fortran reference ``sfincsOutput.h5``
  (the same ground truth the h5 end-to-end tests use); and
* the canonical ``.npz`` equals the legacy ``.npz`` writer field-by-field for
  every shared dataset (the legacy writer only adds JAX ``linearSolver*``
  provenance the canonical operator does not emit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.api import write_output
from sfincs_jax.io import read_sfincs_h5, read_sfincs_output_file
from sfincs_jax.run import run_profile, run_transport_matrix

REF = Path(__file__).parent / "ref"

# Legacy-only JAX solver provenance (never a Fortran dataset); the canonical
# writer intentionally omits these, exactly as it does for the h5 path.
_LEGACY_ONLY_KEYS = frozenset(
    {
        "linearSolverAcceptanceCriterion",
        "linearSolverAccepted",
        "linearSolverConverged",
        "linearSolverMethod",
        "linearSolverRequestedMethod",
        "linearSolverResidualNorm",
        "linearSolverResidualTarget",
        "linearSolverResidualTargetRatio",
        "linearSolverTrueResidualConverged",
    }
)


def _is_numeric(x) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "U", "O"}:
        return False
    try:
        np.asarray(x, dtype=np.float64)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize(
    "base, rhs_mode",
    (
        ("pas_1species_PAS_noEr_tiny_scheme1", 1),
        ("monoenergetic_PAS_tiny_scheme11", 3),
    ),
)
def test_canonical_npz_matches_fortran_reference_and_legacy_npz(
    base: str, rhs_mode: int, tmp_path: Path
) -> None:
    input_path = REF / f"{base}.input.namelist"
    ref = read_sfincs_h5(REF / f"{base}.sfincsOutput.h5")

    canonical = tmp_path / f"{base}.canonical.npz"
    legacy = tmp_path / f"{base}.legacy.npz"

    if rhs_mode == 1:
        run = run_profile(input_path, out_path=canonical, emit=None)
    else:
        run = run_transport_matrix(input_path, out_path=canonical, emit=None)
    assert Path(run.output_path).suffix == ".npz"

    write_output(input_path, legacy)

    out = read_sfincs_output_file(canonical)
    leg = read_sfincs_output_file(legacy)

    # 1. Canonical .npz reproduces the Fortran reference datasets.
    atol = 5e-8
    for k in sorted(ref.keys()):
        if k in {"input.namelist", "elapsed time (s)"} or not _is_numeric(ref[k]):
            continue
        assert k in out, f"canonical .npz missing {k}"
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=1e-12,
            atol=atol,
            err_msg=f"{base}:{k}",
        )

    # 2. Canonical .npz == legacy .npz for every shared numeric dataset.
    shared = (set(out) & set(leg)) - {"input.namelist", "elapsed time (s)"}
    assert (set(leg) - set(out)) <= _LEGACY_ONLY_KEYS
    for k in sorted(shared):
        if not _is_numeric(leg[k]):
            continue
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(leg[k], dtype=np.float64),
            rtol=1e-12,
            atol=atol,
            err_msg=f"{base}:{k}",
        )

    # Provenance: timings are written but expected to differ from the reference.
    assert np.asarray(out["elapsed time (s)"]).shape == np.asarray(ref["elapsed time (s)"]).shape
