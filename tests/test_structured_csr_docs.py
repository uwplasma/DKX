from __future__ import annotations

from pathlib import Path

from sfincs_jax.problems.profile_setup import STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _doc_text() -> str:
    return "\n".join(
        [
            (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs" / "examples.rst").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs" / "usage.rst").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs" / "outputs.rst").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs" / "performance_techniques.rst").read_text(encoding="utf-8"),
        ]
    )


def test_structured_csr_solve_method_names_are_documented() -> None:
    methods = STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS
    docs = _doc_text()

    for method in ("structured_csr", "host_structured_csr"):
        assert method in methods
        assert method in docs

    for alias in ("structured_full_csr", "host_full_csr", "structured_full_csr_host_gmres"):
        assert alias in methods
        assert alias in docs


def test_structured_csr_xblock_environment_knobs_are_documented() -> None:
    docs = _doc_text()

    for knob in (
        "SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER",
        "SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_LMAX",
        "SFINCS_JAX_RHS1_FULL_CSR_MAX_MB",
        "SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB",
        "SFINCS_JAX_RHS1_FULL_CSR_KRYLOV",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COARSE_SOLVER",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_LMAX",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_DROP_TOL",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_FILL_FACTOR",
    ):
        assert knob in docs

    assert "xblock_tz_low_l_schur" in docs
    assert "active projected direct" in docs
    assert "active_low_l_schur" in docs
    assert "active_overlap_schwarz" in docs
    assert "active_schwarz_low_l_schur" in docs
    assert "active_xblock" in docs
    assert "active_coarse" in docs
    assert "active_ilu" in docs
    assert "SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO" in docs


def test_structured_csr_auto_selection_is_documented() -> None:
    docs = _doc_text()

    assert "Zenodo QA/QH negative gate" in docs
    assert "SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO=1" in docs
    assert "auto" in docs
    assert "structured full-CSR" in docs


def test_fortran_reduced_auto_selection_is_documented() -> None:
    docs = _doc_text()

    assert "fortran_reduced_pc_gmres" in docs
    assert "Fortran-reduced sparse-PC GMRES" in docs
    assert "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO" in docs
    assert "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE" in docs
    assert "full-operator true-residual acceptance" in docs


def test_full_grid_qa_qh_active_lu_route_and_cap_are_documented() -> None:
    docs = _doc_text()

    assert "full-grid finite-beta QA/QH" in docs
    assert "25 x 39 x 60 x 7" in docs
    assert "Fortran-reduced direct-tail active LU" in docs
    assert "active LU preconditioner" in docs
    assert "507004" in docs
    assert "14708.1 MiB" in docs
    assert "13,303,259,384" in docs
    assert "no manual ``PC_BACKEND=global``" in docs
    assert "``DIRECT_TAIL_PC_MAX_MB`` override" in docs
