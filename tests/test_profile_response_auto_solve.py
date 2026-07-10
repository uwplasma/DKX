"""Absence tests for the removed RHSMode=1 auto host solver routes.

The Fortran-reduced sparse-PC and assembled full-CSR ``auto`` lanes were
deleted with the legacy sparse solver families; ``try_rhs1_auto_host_solve``
must decline every context so the matrix-free policy owns retained legacy
solves.
"""

from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.problems.profile_dense import RHS1AutoHostSolveContext, try_rhs1_auto_host_solve


def test_auto_host_solve_always_declines() -> None:
    context = RHS1AutoHostSolveContext(
        nml=None, which_rhs=None, op=SimpleNamespace(total_size=10),
        x0=None, tol=1e-10, atol=0.0, restart=80, maxiter=None,
        solve_method="auto", identity_shift=0.0, phi1_hat_base=None,
        differentiable=False, emit=None, recycle_basis=None,
        solve_driver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not solve")),
        solve_method_kind_requested="auto",
        structured_full_csr_explicit_requested=False,
        use_implicit=False,
        structured_auto_allowed=False,
        structured_sharded_multidevice=False,
    )
    assert try_rhs1_auto_host_solve(context) is None
