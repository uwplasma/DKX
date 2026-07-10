from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
from sfincs_jax.solvers.preconditioner_xblock_block_jacobi import (
    build_rhs1_xblock_tz_lmax_preconditioner,
)
from sfincs_jax.solvers.preconditioner_xblock_radial import (
    _rhsmode1_precond_cache_key,
    build_rhs1_xmg_preconditioner,
)
from sfincs_jax.solvers.preconditioning import _RHSMODE1_XUPWIND_PRECOND_CACHE


def test_rhsmode1_xmg_includes_er_xdot_x_coupling_for_pas(monkeypatch) -> None:
    """RHSMode=1 xmg preconditioner must capture Er xDot dense-x coupling for PAS-only cases.

    Without including Er xDot in the coarse-x matrix, PAS-only systems are diagonal in x and
    xmg devolves to a pointwise scaling. With Er xDot enabled, the coarse inverse should
    become non-diagonal in x.
    """
    input_path = (
        Path(__file__).parent
        / "reduced_inputs"
        / "tokamak_1species_PASCollisions_withEr_fullTrajectories.input.namelist"
    )
    assert input_path.exists()
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    assert op.fblock.pas is not None
    assert op.fblock.fp is None
    assert op.fblock.er_xdot is not None

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XMG_STRIDE", "2")
    build_rhs1_xmg_preconditioner(op=op)

    # For PAS+Er, the xmg builder should route to the stable x-upwind preconditioner
    # (dense ddx-based x-block inversions can be extremely ill-conditioned).
    cache_key = _rhsmode1_precond_cache_key(op, "xupwind")
    cached = _RHSMODE1_XUPWIND_PRECOND_CACHE[cache_key]
    sub = np.asarray(cached.sub)[0, :, 0]  # (X,) for s=0, L=0
    assert float(np.max(np.abs(sub[1:]))) > 0.0


def test_rhsmode1_xmg_preconditioner_is_transpose_safe(monkeypatch) -> None:
    """JAX GMRES transposes right-preconditioned matvecs; xmg scatters must support that path."""

    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XMG_STRIDE", "2")
    preconditioner = build_rhs1_xmg_preconditioner(op=op)
    rhs = jnp.linspace(0.1, 1.0, int(op.total_size), dtype=jnp.float64)
    weights = jnp.linspace(1.0, 0.2, int(op.total_size), dtype=jnp.float64)

    def objective(residual):
        return jnp.vdot(weights, preconditioner(residual))

    value, pullback = jax.vjp(objective, rhs)
    (grad_rhs,) = pullback(jnp.asarray(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad_rhs)).all()
    assert float(jnp.linalg.norm(grad_rhs)) > 0.0


def test_rhsmode1_xblock_tz_lmax_preconditioner_is_transpose_safe(monkeypatch) -> None:
    """The strong fallback used by GPU GMRES must stay linear and transposable."""

    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    monkeypatch.setenv("SFINCS_JAX_PRECOND_PAS_MAX_COLS", "8")
    preconditioner = build_rhs1_xblock_tz_lmax_preconditioner(op=op, lmax=2)
    rhs = jnp.linspace(0.2, 0.9, int(op.total_size), dtype=jnp.float64)
    weights = jnp.linspace(0.7, -0.1, int(op.total_size), dtype=jnp.float64)

    def objective(residual):
        return jnp.vdot(weights, preconditioner(residual))

    value, pullback = jax.vjp(objective, rhs)
    (grad_rhs,) = pullback(jnp.asarray(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad_rhs)).all()
    assert float(jnp.linalg.norm(grad_rhs)) > 0.0
