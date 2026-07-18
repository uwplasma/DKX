from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx.api import write_output
from dkx.io import read_sfincs_h5


def _is_numeric_dataset(x) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "U", "O"}:
        return False
    try:
        np.asarray(x, dtype=np.float64)
        return True
    except Exception:  # noqa: BLE001
        return False


def _assert_fortran_keys_and_solver_metadata(out: dict, ref: dict) -> None:
    """Require Fortran-compatible fields while allowing JAX solver provenance."""
    assert set(ref.keys()).issubset(out.keys())
    assert "linearSolverMethod" in out
    assert "linearSolverResidualNorm" in out


def _assert_bootstrap_current_closure(out: dict, *, atol: float) -> None:
    n_species = int(np.asarray(out["Nspecies"]).reshape(-1)[0])
    charges = np.asarray(out["Zs"], dtype=np.float64).reshape((n_species,))
    flow = np.asarray(out["FSABFlow"], dtype=np.float64)
    if flow.ndim == 1:
        flow = flow.reshape((n_species, 1))
    assert flow.shape[0] == n_species
    current = np.asarray(out["FSABjHat"], dtype=np.float64).reshape((-1,))
    expected_current = np.sum(charges[:, None] * flow, axis=0)

    np.testing.assert_allclose(current, expected_current, rtol=0.0, atol=float(atol))
    np.testing.assert_allclose(
        np.asarray(out["FSABjHatOverB0"], dtype=np.float64).reshape((-1,)),
        current / float(np.asarray(out["B0OverBBar"], dtype=np.float64).reshape(-1)[0]),
        rtol=0.0,
        atol=float(atol),
    )
    np.testing.assert_allclose(
        np.asarray(out["FSABjHatOverRootFSAB2"], dtype=np.float64).reshape((-1,)),
        current / np.sqrt(float(np.asarray(out["FSABHat2"], dtype=np.float64).reshape(-1)[0])),
        rtol=0.0,
        atol=float(atol),
    )


@pytest.mark.parametrize(
    "base",
    (
        "pas_1species_PAS_noEr_tiny_withPhi1_linear",
        "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear",
        "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision",
        "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear",
    ),
)
def test_write_output_rhsmode1_phi1_fixtures_match_fortran_end_to_end(base: str, tmp_path: Path) -> None:
    """End-to-end: solve RHSMode=1 includePhi1 fixtures and write a v3-style sfincsOutput.h5."""
    here = Path(__file__).parent
    input_path = here / "ref" / f"{base}.input.namelist"
    ref_path = here / "ref" / f"{base}.sfincsOutput.h5"
    out_path = tmp_path / f"{base}.sfincsOutput_jax.h5"

    write_output(input_path, out_path)

    out = read_sfincs_h5(out_path)
    ref = read_sfincs_h5(ref_path)

    _assert_fortran_keys_and_solver_metadata(out, ref)
    _assert_bootstrap_current_closure(out, atol=5.0e-11)

    # Newton-based includePhi1 runs can differ at ~1e-9 due to floating-point and inner-solve details.
    # Keep a tight absolute tolerance consistent with other Phi1 fixture tests.
    atol = 5e-8

    for k in sorted(ref.keys()):
        if k == "input.namelist":
            continue
        if k == "elapsed time (s)":
            # Wall-clock provenance: the retired legacy writer froze this to
            # zeros; the canonical writer records the real solve time, which
            # legitimately differs from the Fortran fixture (asserted below).
            continue
        if not _is_numeric_dataset(ref[k]):
            continue
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=0.0,
            atol=atol,
        )

    # Timings are expected to differ between Fortran and JAX runs, but we
    # still write them for provenance.
    assert "elapsed time (s)" in out
    assert np.asarray(out["elapsed time (s)"]).shape == np.asarray(ref["elapsed time (s)"]).shape
    assert np.all(np.asarray(out["elapsed time (s)"], dtype=np.float64) >= 0.0)


def test_write_output_default_qn_phi1_path_regression(tmp_path: Path) -> None:
    """QN-only default quasineutrality should use physics parameters without crashing."""
    src = Path(__file__).parent / "ref" / "include_phi1_linear_subset_tiny.input.namelist"
    text = src.read_text()
    text = text.replace("  quasineutralityOption = 2\n", "")
    text = text.replace("Nzeta = 5", "Nzeta = 1")
    text = text.replace("Nxi = 6", "Nxi = 4")
    text = text.replace("Nx = 4", "Nx = 2")

    input_path = tmp_path / "default_qn_phi1.input.namelist"
    input_path.write_text(text)
    out_path = tmp_path / "default_qn_phi1.sfincsOutput_jax.h5"

    write_output(input_path, out_path)

    out = read_sfincs_h5(out_path)
    assert int(np.asarray(out["NIterations"]).ravel()[-1]) >= 1
    assert float(np.asarray(out["linearSolverResidualNorm"]).item()) < 1.0e-12
    np.testing.assert_allclose(np.asarray(out["FSABFlow"], dtype=np.float64), 0.0, atol=1e-12)
