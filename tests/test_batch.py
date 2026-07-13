"""Batched multi-surface / multi-``E_r`` vmap productization (:mod:`sfincs_jax.batch`).

Gates for the first-class batched-solve API:

- batched result == serial per-element result to ~1e-12 (``E_r`` scan and
  surface scan), for both the states and the moments;
- ``jax.grad`` through a batched scalar (sum of the bootstrap current over the
  batch) matches the per-element gradients and a central finite difference;
- ``jax.jit(batched_solve)`` compiles and runs, matching the eager result;
- the memory-budgeted auto-chunking respects the budget / ``max_batch`` and the
  chunked result is identical to the single-chunk result;
- batching a discretization leaf, or a surface batch that disagrees on the
  discretization, is rejected with a clear error;
- the :mod:`sfincs_jax.api` facades route to the same result.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest


def _pas_deck(
    er: float = 0.0,
    *,
    epsilon_h: float = 0.05,
    dndr: float = -0.5,
    n_theta: int = 7,
    n_zeta: int = 7,
    n_xi: int = 8,
    n_x: int = 3,
) -> str:
    """A tiny non-axisymmetric two-species PAS deck (RHSMode=1, ``Er`` knob)."""
    return f"""&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = 1.31
  epsilon_t = 0.1
  epsilon_h = {epsilon_h}
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 0.000545509
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = {dndr} {dndr}
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _write(tmp_path: Path, text: str, name: str = "input.namelist") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _build_op(tmp_path: Path, *, epsilon_h: float, dndr: float, name: str):
    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
    from sfincs_jax.inputs import load_sfincs_input

    deck = _pas_deck(epsilon_h=epsilon_h, dndr=dndr)
    raw = load_sfincs_input(_write(tmp_path, deck, name)).raw
    return kinetic_operator_from_namelist(raw)


# ---------------------------------------------------------------------------
# 1a. E_r scan: batched == serial per-element to ~1e-12
# ---------------------------------------------------------------------------


def test_batched_er_scan_matches_serial(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 0.7], dtype=jnp.float64)

    result = batch_mod.batched_er_scan(prob, er_values)

    assert result.states.shape[0] == er_values.shape[0]
    assert result.radial_current.shape == er_values.shape

    for i, er in enumerate(np.asarray(er_values)):
        j_r, gamma, state = er_mod.radial_current(prob, float(er))
        np.testing.assert_allclose(
            np.asarray(result.radial_current[i]), float(j_r), rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(result.moments["particleFlux_vm_psiHat"][i]),
            np.asarray(gamma, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(result.states[i]),
            np.asarray(jnp.reshape(state.result.x, (-1,))),
            rtol=0.0,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 1b. Surface scan: batched == serial per-element to ~1e-12
# ---------------------------------------------------------------------------


def test_batched_surface_scan_matches_serial(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax.run import profile_moments_from_operator
    from sfincs_jax.solve import solve

    specs = [(0.03, -0.5), (0.05, -0.6), (0.07, -0.4)]
    ops = [
        _build_op(tmp_path, epsilon_h=eh, dndr=dn, name=f"surface_{i}.namelist")
        for i, (eh, dn) in enumerate(specs)
    ]

    result = batch_mod.batched_surface_scan(ops)
    assert result.states.shape[0] == len(ops)

    for i, op in enumerate(ops):
        state = jnp.reshape(solve(op, op.rhs(), method="auto", tol=1e-10).x, (-1,))
        moments = profile_moments_from_operator(op, state)
        np.testing.assert_allclose(
            np.asarray(result.states[i]), np.asarray(state), rtol=0.0, atol=1e-12
        )
        for key in ("FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
            np.testing.assert_allclose(
                np.asarray(result.moments[key][i]),
                np.asarray(moments[key]),
                rtol=0.0,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# 2. Differentiable: grad of a batched scalar == per-element grads == central FD
# ---------------------------------------------------------------------------


def test_batched_grad_matches_per_element_and_fd(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod
    from sfincs_jax.run import profile_moments_from_operator
    from sfincs_jax.solve import solve

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -0.5, 0.5], dtype=jnp.float64)
    dphi = jnp.asarray(prob.dphi_per_er, dtype=jnp.float64) * er_values
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}
    base_dn = prob.operator.dn_hat_dpsi_hat

    def batched_bootstrap_sum(scale):
        op = dataclasses.replace(prob.operator, dn_hat_dpsi_hat=base_dn * scale)
        res = batch_mod.batched_solve(op, batch_leaves, differentiable=True)
        return jnp.sum(res.moments["FSABjHat"])

    grad_batched = float(jax.grad(batched_bootstrap_sum)(1.0))

    # Per-element gradient sum (the batch scalar is a sum, so grads add).
    def element_bootstrap(scale, dphi_scalar):
        op = dataclasses.replace(
            prob.operator,
            dn_hat_dpsi_hat=base_dn * scale,
            dphi_hat_dpsi_hat=dphi_scalar,
            dphi_hat_dpsi_hat_kinetic=dphi_scalar,
        )
        state = jnp.reshape(
            solve(op, op.rhs(), method="auto", tol=1e-10, differentiable=True).x, (-1,)
        )
        return profile_moments_from_operator(op, state)["FSABjHat"]

    grad_elements = sum(
        float(jax.grad(element_bootstrap)(1.0, float(d))) for d in np.asarray(dphi)
    )

    # Central finite difference of the batched scalar.
    eps = 1e-4
    fd = float(
        (batched_bootstrap_sum(1.0 + eps) - batched_bootstrap_sum(1.0 - eps)) / (2 * eps)
    )

    assert abs(grad_batched - grad_elements) <= 1e-9
    assert abs(grad_batched - fd) / (abs(fd) + 1e-30) <= 1e-6


# ---------------------------------------------------------------------------
# 3. jit-safety: jax.jit(batched_solve) compiles and runs, matches eager
# ---------------------------------------------------------------------------


def test_jit_batched_solve_compiles_and_matches(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-1.5, 0.0, 0.6], dtype=jnp.float64)
    dphi = jnp.asarray(prob.dphi_per_er, dtype=jnp.float64) * er_values
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}

    eager = batch_mod.batched_solve(prob.operator, batch_leaves)
    jitted = jax.jit(lambda leaves: batch_mod.batched_solve(prob.operator, leaves))(
        batch_leaves
    )

    np.testing.assert_allclose(
        np.asarray(jitted.states), np.asarray(eager.states), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(jitted.moments["FSABjHat"]),
        np.asarray(eager.moments["FSABjHat"]),
        rtol=0.0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 4. Memory-budgeted auto-chunking: budget/max_batch honored, chunked == single
# ---------------------------------------------------------------------------


def test_auto_chunk_size_respects_budget_and_max_batch(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    op = prob.operator

    assert batch_mod.solve_footprint_bytes(op) > 0.0
    assert batch_mod.resolve_memory_budget_bytes(4.0) == pytest.approx(4.0 * 2**30)

    # A generous budget fits the whole batch in one chunk.
    assert batch_mod.auto_chunk_size(op, 8, memory_budget_gb=64.0) == 8
    # A tiny budget forces one solve at a time (never zero).
    assert batch_mod.auto_chunk_size(op, 8, memory_budget_gb=1e-6) == 1
    # max_batch caps the chunk; the batch length caps it too.
    assert batch_mod.auto_chunk_size(op, 8, max_batch=3, memory_budget_gb=64.0) == 3
    assert batch_mod.auto_chunk_size(op, 2, memory_budget_gb=64.0) == 2

    with pytest.raises(ValueError):
        batch_mod.auto_chunk_size(op, 0)
    with pytest.raises(ValueError):
        batch_mod.auto_chunk_size(op, 4, max_batch=0)


def test_chunked_batch_matches_single_chunk(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5], dtype=jnp.float64)

    single = batch_mod.batched_er_scan(prob, er_values)
    chunked = batch_mod.batched_er_scan(prob, er_values, max_batch=2)

    assert single.chunk_size == er_values.shape[0]
    assert chunked.chunk_size == 2
    assert chunked.n_chunks == 3  # ceil(5 / 2)
    np.testing.assert_allclose(
        np.asarray(chunked.radial_current),
        np.asarray(single.radial_current),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked.states), np.asarray(single.states), rtol=0.0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Validation: discretization leaves may not be batched / must agree
# ---------------------------------------------------------------------------


def test_batching_discretization_leaf_is_rejected(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    op = prob.operator

    with pytest.raises(ValueError, match="discretization leaf"):
        batch_mod.batched_solve(op, {"ddtheta": jnp.stack([op.ddtheta, op.ddtheta])})
    with pytest.raises(ValueError, match="not a batchable"):
        batch_mod.batched_solve(op, {"not_a_field": jnp.zeros((2,))})
    with pytest.raises(ValueError, match="empty"):
        batch_mod.batched_solve(op, {})


def test_surface_scan_requires_shared_discretization(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from sfincs_jax import batch as batch_mod
    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
    from sfincs_jax.inputs import load_sfincs_input

    op_a = _build_op(tmp_path, epsilon_h=0.05, dndr=-0.5, name="a.namelist")
    # A different Ntheta changes the discretization -> must be rejected.
    raw_b = load_sfincs_input(
        _write(tmp_path, _pas_deck(n_theta=9), "b.namelist")
    ).raw
    op_b = kinetic_operator_from_namelist(raw_b)

    with pytest.raises(ValueError, match="discretization|structure"):
        batch_mod.batched_surface_scan([op_a, op_b])

    with pytest.raises(ValueError, match="at least one"):
        batch_mod.batched_surface_scan([])


# ---------------------------------------------------------------------------
# 6. API facades route to the same result
# ---------------------------------------------------------------------------


def test_api_batched_facades_route(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from sfincs_jax import api
    from sfincs_jax import batch as batch_mod
    from sfincs_jax import er as er_mod

    deck_path = _write(tmp_path, _pas_deck())
    er_values = jnp.asarray([-1.5, -0.3, 0.4], dtype=jnp.float64)

    prob = er_mod.prepare(deck_path, er_bracket=(-5.0, 5.0))
    direct = batch_mod.batched_er_scan(prob, er_values)
    via_api = api.batched_er_scan(str(deck_path), er_values, er_bracket=(-5.0, 5.0))
    np.testing.assert_allclose(
        np.asarray(via_api.radial_current),
        np.asarray(direct.radial_current),
        rtol=0.0,
        atol=1e-12,
    )

    specs = [(0.04, -0.5), (0.06, -0.55)]
    ops = [
        _build_op(tmp_path, epsilon_h=eh, dndr=dn, name=f"api_surface_{i}.namelist")
        for i, (eh, dn) in enumerate(specs)
    ]
    direct_surf = batch_mod.batched_surface_scan(ops)
    via_api_surf = api.batched_surface_scan(ops)
    np.testing.assert_allclose(
        np.asarray(via_api_surf.moments["FSABjHat"]),
        np.asarray(direct_surf.moments["FSABjHat"]),
        rtol=0.0,
        atol=1e-12,
    )
