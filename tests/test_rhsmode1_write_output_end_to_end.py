from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


@pytest.fixture(autouse=True)
def _clear_solver_policy_env(monkeypatch) -> None:
    """Keep end-to-end parity fixtures independent of earlier solver-policy tests."""

    for key in (
        "SFINCS_JAX_RHSMODE1_PRECONDITIONER",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP",
    ):
        monkeypatch.delenv(key, raising=False)


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
    """Require all Fortran fixture fields while allowing JAX-only solver metadata."""
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
        "pas_1species_PAS_noEr_tiny_scheme1",
        "pas_1species_PAS_noEr_tiny_scheme5",
        "pas_1species_PAS_noEr_tiny_scheme11",
        "pas_1species_PAS_noEr_tiny_scheme12",
    ),
)
def test_write_output_rhsmode1_solution_fields_match_fortran_fixture(base: str, tmp_path: Path) -> None:
    """End-to-end: from input.namelist, solve RHSMode=1 and write solution-derived fields."""
    here = Path(__file__).parent
    input_path = here / "ref" / f"{base}.input.namelist"
    ref_path = here / "ref" / f"{base}.sfincsOutput.h5"
    out_path = tmp_path / f"{base}.sfincsOutput_jax.h5"

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_solution=True,
    )

    out = read_sfincs_h5(out_path)
    ref = read_sfincs_h5(ref_path)
    _assert_fortran_keys_and_solver_metadata(out, ref)
    _assert_bootstrap_current_closure(out, atol=5.0e-12)

    # Full-file numeric parity (excluding embedded input text).
    # uHat involves long FFTs/reductions and is compared with a slightly looser atol.
    # GeometryScheme=12 uses non-stellarator-symmetric Boozer `.bc` inputs and can show slightly
    # larger floating-point differences in some geometry *derivative* arrays due to reduction
    # ordering and cancellation in large-magnitude terms.
    atol = 2e-9 if base.endswith("scheme12") else 5e-10
    atol_uhat = 1e-8
    scheme12_key_atol = {
        # These derivatives can accumulate small absolute differences ~1e-8.
        "dBHat_sup_theta_dpsiHat": 1e-7,
        "dBHat_sup_zeta_dpsiHat": 1e-7,
        # This metric coefficient can differ at the ~1e-3 level in absolute terms while still
        # matching to ~1e-10 relative on typical v3 fixtures (values are O(1e6)).
        "gpsiHatpsiHat": 1e-3,
    }

    for k in sorted(ref.keys()):
        if k == "input.namelist":
            continue
        if not _is_numeric_dataset(ref[k]):
            continue
        if base.endswith("scheme12") and k in scheme12_key_atol:
            this_atol = scheme12_key_atol[k]
        else:
            this_atol = atol_uhat if k == "uHat" else atol
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=0.0,
            atol=float(this_atol),
        )

    # Timings are expected to differ between Fortran and JAX runs, but we still write them for provenance.
    assert "elapsed time (s)" in out
    assert np.asarray(out["elapsed time (s)"]).shape == np.asarray(ref["elapsed time (s)"]).shape
    assert np.all(np.asarray(out["elapsed time (s)"], dtype=np.float64) >= 0.0)
