from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from sfincs_jax.problems.profile_response.auto_solve import (
    RHS1AutoHostSolveContext,
    RHS1StructuredCSRSolveContext,
    solve_rhs1_structured_full_csr_explicit,
    try_rhs1_auto_host_solve,
)


@dataclass(frozen=True)
class _FakeGMRES:
    residual_norm: float


@dataclass(frozen=True)
class _FakeResult:
    gmres: _FakeGMRES
    metadata: dict[str, Any]


def _full_fp_op(*, total_size: int = 20, constraint_scheme: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        total_size=total_size,
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=constraint_scheme,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )


def _context(**overrides: Any) -> RHS1AutoHostSolveContext:
    calls = overrides.pop("calls", [])

    def solve_driver(**kwargs: Any) -> _FakeResult:
        calls.append(kwargs)
        solve_method = str(kwargs.get("solve_method", ""))
        metadata = {"solver_kind": solve_method, "accepted_converged": solve_method == "structured_full_csr"}
        return _FakeResult(gmres=_FakeGMRES(residual_norm=1.0e-12), metadata=metadata)

    values: dict[str, Any] = {
        "nml": object(),
        "which_rhs": None,
        "op": _full_fp_op(),
        "x0": None,
        "tol": 1.0e-10,
        "atol": 0.0,
        "restart": 40,
        "maxiter": 120,
        "solve_method": "auto",
        "identity_shift": 0.0,
        "phi1_hat_base": None,
        "differentiable": False,
        "emit": None,
        "recycle_basis": None,
        "solve_driver": solve_driver,
        "solve_method_kind_requested": "auto",
        "structured_full_csr_explicit_requested": False,
        "use_implicit": False,
        "structured_auto_allowed": False,
        "structured_sharded_multidevice": False,
    }
    values.update(overrides)
    return RHS1AutoHostSolveContext(**values)


def test_auto_host_solve_selects_fortran_reduced_sparse_pc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", "1")
    calls: list[dict[str, Any]] = []

    result = try_rhs1_auto_host_solve(_context(calls=calls))

    assert result is not None
    assert len(calls) == 1
    assert calls[0]["solve_method"] == "fortran_reduced_pc_gmres"
    assert result.metadata["auto_solver_selected"] is True
    assert result.metadata["auto_solver_policy"] == "fortran_reduced_pc_gmres"
    assert result.metadata["auto_solver_size"] == 20
    assert result.metadata["auto_solver_min_size"] == 1


def test_auto_host_solve_skips_fortran_reduced_when_implicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", "1")
    calls: list[dict[str, Any]] = []

    result = try_rhs1_auto_host_solve(_context(calls=calls, use_implicit=True))

    assert result is None
    assert calls == []


def test_auto_host_solve_accepts_structured_csr_after_fortran_reduced_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO", "0")
    calls: list[dict[str, Any]] = []

    result = try_rhs1_auto_host_solve(_context(calls=calls, structured_auto_allowed=True))

    assert result is not None
    assert len(calls) == 1
    assert calls[0]["solve_method"] == "structured_full_csr"
    assert result.metadata["auto_solver_selected"] is True
    assert result.metadata["auto_solver_policy"] == "structured_full_csr"


def test_auto_host_solve_rejects_nonconverged_structured_csr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO", "0")
    messages: list[str] = []

    def solve_driver(**kwargs: Any) -> _FakeResult:
        assert kwargs["solve_method"] == "structured_full_csr"
        return _FakeResult(
            gmres=_FakeGMRES(residual_norm=2.0),
            metadata={"accepted_converged": False, "reported_residual_norm": 2.0},
        )

    result = try_rhs1_auto_host_solve(
        _context(
            solve_driver=solve_driver,
            structured_auto_allowed=True,
            emit=lambda _level, msg: messages.append(msg),
        )
    )

    assert result is None
    assert any("did not converge" in message for message in messages)


def test_structured_full_csr_explicit_normalizes_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER", "active_xblock")
    calls: list[dict[str, Any]] = []

    def structured_solver(**kwargs: Any) -> _FakeResult:
        calls.append(kwargs)
        return _FakeResult(
            gmres=_FakeGMRES(residual_norm=1.0e-12),
            metadata={
                "structured_full_csr": {
                    "residual_norm": 1.0e-12,
                    "converged": True,
                    "solve_s": 0.25,
                    "info": 0,
                    "residual_history": (1.0, 1.0e-12),
                    "selection": {"metadata": {"nnz": 17, "csr_nbytes_actual": 512}},
                    "metadata": {
                        "target": 1.0e-10,
                        "factor_s": 0.05,
                        "factor_nbytes_actual": 1024,
                        "active_dof": True,
                        "active_size": 7,
                        "full_size": 11,
                        "preconditioner": {
                            "kind": "active_xblock",
                            "setup_s": 0.125,
                            "metadata": {"factor_nbytes_actual": 2048},
                        },
                    },
                }
            },
        )

    result = solve_rhs1_structured_full_csr_explicit(
        RHS1StructuredCSRSolveContext(
            nml=object(),
            op=_full_fp_op(total_size=11),
            x0=None,
            rhs_norm=2.0,
            tol=1.0e-10,
            atol=0.0,
            restart=30,
            maxiter=90,
            solve_method="structured_csr",
            identity_shift=0.0,
            phi1_hat_base=None,
            differentiable=False,
            emit=None,
            structured_solver=structured_solver,
        )
    )

    assert calls and calls[0]["method"] == "direct"
    assert calls[0]["active_dof"] is True
    assert calls[0]["max_csr_nbytes"] == 2 * 1024 * 1024
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["reported_residual_norm"] == pytest.approx(1.0e-12)
    assert result.metadata["iterations"] == 2
    assert result.metadata["csr_nnz"] == 17
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == 2048
    assert result.metadata["structured_active_size"] == 7
