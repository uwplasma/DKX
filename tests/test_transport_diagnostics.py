from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.geometry import BoozerGeometry
from sfincs_jax.problems import transport_diagnostics as td


class _ToyFBlock:
    def __init__(self, *, n_species: int = 2, n_x: int = 3, n_xi: int = 4, n_theta: int = 2, n_zeta: int = 3):
        self.n_species = int(n_species)
        self.n_x = int(n_x)
        self.n_xi = int(n_xi)
        self.n_theta = int(n_theta)
        self.n_zeta = int(n_zeta)
        self.f_shape = (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)
        self.flat_size = int(np.prod(self.f_shape))
        n_xi_for_x = np.asarray([self.n_xi, max(1, self.n_xi - 1), 1], dtype=np.int32)[: self.n_x]
        self.collisionless = SimpleNamespace(n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32))


class _ToyOp:
    """Small weakref-able operator carrying the fields used by diagnostics.F90 formulas."""

    def __init__(self, *, rhs_mode: int = 2, constraint_scheme: int = 2, include_phi1: bool = False, n_xi: int = 4):
        self.fblock = _ToyFBlock(n_xi=n_xi)
        self.constraint_scheme = int(constraint_scheme)
        self.point_at_x0 = False
        self.include_phi1 = bool(include_phi1)
        self.quasineutrality_option = 2
        self.with_adiabatic = False
        self.alpha = jnp.asarray(1.15, dtype=jnp.float64)
        self.delta = jnp.asarray(0.025, dtype=jnp.float64)
        self.adiabatic_z = jnp.asarray(0.0, dtype=jnp.float64)
        self.adiabatic_nhat = jnp.asarray(0.0, dtype=jnp.float64)
        self.adiabatic_that = jnp.asarray(1.0, dtype=jnp.float64)
        self.include_phi1_in_kinetic = False
        self.dphi_hat_dpsi_hat = jnp.asarray(0.0, dtype=jnp.float64)
        self.rhs_mode = int(rhs_mode)
        self.e_parallel_hat = jnp.asarray(0.0, dtype=jnp.float64)
        self.e_parallel_hat_spec = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        self.fsab_hat2 = jnp.asarray(1.7, dtype=jnp.float64)
        self.z_s = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
        self.m_hat = jnp.asarray([2.0, 0.75], dtype=jnp.float64)
        self.t_hat = jnp.asarray([1.3, 0.8], dtype=jnp.float64)
        self.n_hat = jnp.asarray([0.9, 1.1], dtype=jnp.float64)
        self.dn_hat_dpsi_hat = jnp.asarray([-0.2, -0.1], dtype=jnp.float64)
        self.dt_hat_dpsi_hat = jnp.asarray([-0.3, -0.25], dtype=jnp.float64)
        self.theta_weights = jnp.asarray([0.55, 0.45], dtype=jnp.float64)
        self.zeta_weights = jnp.asarray([0.2, 0.35, 0.45], dtype=jnp.float64)
        self.d_hat = jnp.asarray([[1.2, 1.1, 1.3], [1.05, 1.25, 1.4]], dtype=jnp.float64)
        self.b_hat = jnp.asarray([[1.1, 1.25, 1.35], [1.18, 1.4, 1.5]], dtype=jnp.float64)
        self.db_hat_dtheta = jnp.asarray([[0.04, -0.03, 0.05], [-0.02, 0.06, -0.04]], dtype=jnp.float64)
        self.db_hat_dzeta = jnp.asarray([[0.07, 0.02, -0.05], [0.01, -0.06, 0.04]], dtype=jnp.float64)
        self.b_hat_sup_theta = jnp.asarray([[0.3, 0.25, 0.2], [0.22, 0.18, 0.16]], dtype=jnp.float64)
        self.b_hat_sup_zeta = jnp.asarray([[0.15, 0.12, 0.1], [0.11, 0.09, 0.08]], dtype=jnp.float64)
        self.b_hat_sub_theta = jnp.asarray([[0.8, 0.7, 0.6], [0.65, 0.55, 0.45]], dtype=jnp.float64)
        self.b_hat_sub_zeta = jnp.asarray([[0.35, 0.3, 0.25], [0.28, 0.22, 0.18]], dtype=jnp.float64)
        self.x = jnp.asarray([0.4, 0.9, 1.4], dtype=jnp.float64)
        self.x_weights = jnp.asarray([0.3, 0.45, 0.25], dtype=jnp.float64)
        self.ddx = jnp.eye(self.n_x, dtype=jnp.float64)
        self.phi1_hat_base = jnp.asarray([[0.02, -0.01, 0.015], [-0.005, 0.01, -0.02]], dtype=jnp.float64)

    @property
    def n_species(self) -> int:
        return int(self.fblock.n_species)

    @property
    def n_x(self) -> int:
        return int(self.fblock.n_x)

    @property
    def n_xi(self) -> int:
        return int(self.fblock.n_xi)

    @property
    def n_theta(self) -> int:
        return int(self.fblock.n_theta)

    @property
    def n_zeta(self) -> int:
        return int(self.fblock.n_zeta)

    @property
    def f_size(self) -> int:
        return int(self.fblock.flat_size)

    @property
    def phi1_size(self) -> int:
        if self.include_phi1:
            return int(self.n_theta * self.n_zeta + 1)
        return 0

    @property
    def extra_size(self) -> int:
        if self.constraint_scheme == 2:
            return int(self.n_species * self.n_x)
        if self.constraint_scheme in {1, 3, 4}:
            return int(2 * self.n_species)
        if self.constraint_scheme == 0:
            return 0
        raise NotImplementedError

    @property
    def total_size(self) -> int:
        return int(self.f_size + self.phi1_size + self.extra_size)


def _state(op: _ToyOp, *, scale: float = 1.0) -> jnp.ndarray:
    f_delta = (jnp.arange(op.f_size, dtype=jnp.float64) - 0.4 * op.f_size) * (0.002 * scale)
    x = jnp.zeros((op.total_size,), dtype=jnp.float64).at[: op.f_size].set(f_delta)
    if op.include_phi1:
        phi1 = jnp.asarray([[0.06, -0.04, 0.02], [0.01, -0.03, 0.05]], dtype=jnp.float64).reshape(-1)
        x = x.at[op.f_size : op.f_size + op.n_theta * op.n_zeta].set(phi1)
        x = x.at[op.f_size + op.n_theta * op.n_zeta].set(-0.125)
    extra0 = op.f_size + op.phi1_size
    extra = (jnp.arange(op.extra_size, dtype=jnp.float64) + 1.0) * (0.01 * scale)
    return x.at[extra0:].set(extra)


def _geom(*, zero_flux_functions: bool = False) -> BoozerGeometry:
    op = _ToyOp()
    zeros = jnp.zeros_like(op.b_hat)
    return BoozerGeometry(
        n_periods=5,
        b0_over_bbar=0.0 if zero_flux_functions else 1.4,
        iota=0.42,
        g_hat=0.0 if zero_flux_functions else 1.8,
        i_hat=0.25,
        b_hat=op.b_hat,
        db_hat_dtheta=op.db_hat_dtheta,
        db_hat_dzeta=op.db_hat_dzeta,
        d_hat=op.d_hat,
        b_hat_sup_theta=op.b_hat_sup_theta,
        b_hat_sup_zeta=op.b_hat_sup_zeta,
        b_hat_sub_theta=op.b_hat_sub_theta,
        b_hat_sub_zeta=op.b_hat_sub_zeta,
        b_hat_sub_psi=zeros,
        db_hat_dpsi_hat=zeros,
        db_hat_sub_psi_dtheta=zeros,
        db_hat_sub_psi_dzeta=zeros,
        db_hat_sub_theta_dpsi_hat=zeros,
        db_hat_sub_zeta_dpsi_hat=zeros,
        db_hat_sub_theta_dzeta=zeros,
        db_hat_sub_zeta_dtheta=zeros,
        db_hat_sup_theta_dpsi_hat=zeros,
        db_hat_sup_theta_dzeta=zeros,
        db_hat_sup_zeta_dpsi_hat=zeros,
        db_hat_sup_zeta_dtheta=zeros,
    )


def _assert_tree_close(left, right) -> None:
    for field in dataclasses.fields(left):
        np.testing.assert_allclose(
            np.asarray(getattr(left, field.name)),
            np.asarray(getattr(right, field.name)),
            rtol=2e-13,
            atol=2e-13,
        )


def test_pytree_roundtrip_and_fortran_order_weighted_sums() -> None:
    diag = td.V3TransportDiagnostics(
        vprime_hat=jnp.asarray(1.25),
        particle_flux_vm_psi_hat=jnp.asarray([1.0, -2.0]),
        heat_flux_vm_psi_hat=jnp.asarray([3.0, 4.0]),
        fsab_flow=jnp.asarray([0.2, -0.1]),
        particle_flux_before_surface_integral_vm=jnp.ones((2, 2, 3)),
        heat_flux_before_surface_integral_vm=2.0 * jnp.ones((2, 2, 3)),
        particle_flux_before_surface_integral_vm0=3.0 * jnp.ones((2, 2, 3)),
        heat_flux_before_surface_integral_vm0=4.0 * jnp.ones((2, 2, 3)),
        particle_flux_vm_psi_hat_vs_x=jnp.ones((3, 2)),
        heat_flux_vm_psi_hat_vs_x=2.0 * jnp.ones((3, 2)),
        fsab_flow_vs_x=3.0 * jnp.ones((3, 2)),
    )
    flat, tree = jax.tree_util.tree_flatten(diag)
    restored = jax.tree_util.tree_unflatten(tree, flat)
    _assert_tree_close(restored, diag)

    pre = td.V3TransportDiagnosticsPrecomputed(
        vprime_hat=jnp.asarray(1.0),
        theta_w=jnp.asarray([0.6, 0.4]),
        zeta_w=jnp.asarray([0.25, 0.75]),
        factor_vm=jnp.ones((2, 2)),
        wpf0=jnp.asarray([1.0, 2.0]),
        wpf2=jnp.asarray([0.5, 0.0]),
        whf0=jnp.asarray([1.5, 2.5]),
        whf2=jnp.asarray([0.25, 0.75]),
        wf1=jnp.asarray([0.2, 0.3]),
        particle_flux_factor_vm=jnp.asarray([1.0]),
        heat_flux_factor_vm=jnp.asarray([2.0]),
        flow_factor=jnp.asarray([3.0]),
        b_over_d=jnp.ones((2, 2)),
    )
    flat_pre, tree_pre = jax.tree_util.tree_flatten(pre)
    _assert_tree_close(jax.tree_util.tree_unflatten(tree_pre, flat_pre), pre)

    values_sxtz = jnp.arange(2 * 3 * 2 * 3, dtype=jnp.float64).reshape((2, 3, 2, 3))
    w_x = jnp.asarray([0.2, 0.5, 0.3], dtype=jnp.float64)
    expected_x = np.einsum("x,sxtz->stz", np.asarray(w_x), np.asarray(values_sxtz))
    np.testing.assert_allclose(np.asarray(td._weighted_sum_x_fortran(w_x, values_sxtz, strict=False)), expected_x)
    np.testing.assert_allclose(np.asarray(td._weighted_sum_x_fortran(w_x, values_sxtz, strict=True)), expected_x)

    values_stz = jnp.arange(2 * 2 * 3, dtype=jnp.float64).reshape((2, 2, 3))
    w_t = jnp.asarray([0.4, 0.6], dtype=jnp.float64)
    w_z = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float64)
    expected_tz = np.einsum("t,z,stz->s", np.asarray(w_t), np.asarray(w_z), np.asarray(values_stz))
    np.testing.assert_allclose(np.asarray(td._weighted_sum_tz_fortran(w_t, w_z, values_stz)), expected_tz)
    expected_sx = np.einsum("t,z,sxtz->sx", np.asarray(w_t), np.asarray(w_z), np.asarray(values_sxtz))
    np.testing.assert_allclose(np.asarray(td._weighted_sum_tz_fortran_sx(w_t, w_z, values_sxtz, strict=True)), expected_sx)

    with pytest.raises(ValueError, match="w_x has length"):
        td._weighted_sum_x_fortran(jnp.ones((2,)), values_sxtz)
    with pytest.raises(ValueError, match="Weight shapes"):
        td._weighted_sum_tz_fortran(jnp.ones((3,)), w_z, values_stz)
    with pytest.raises(ValueError, match="Weight shapes"):
        td._weighted_sum_tz_fortran_sx(w_t, jnp.ones((2,)), values_sxtz)


def test_vm_diagnostics_precompute_matches_direct_and_pins_moment_identities() -> None:
    op = _ToyOp()
    x_full = _state(op)
    direct = td.v3_transport_diagnostics_vm_only(op, x_full=x_full)
    precomputed = td.v3_transport_diagnostics_vm_only_precompute(op)
    via_precompute = td._v3_transport_diagnostics_vm_only_from_precomputed(
        precomputed,
        x_full=x_full,
        f0_l0=td.f0_l0_v3_from_operator(op),
        n_xi=op.n_xi,
        f_shape=op.fblock.f_shape,
        f_size=op.f_size,
    )

    _assert_tree_close(via_precompute, direct)
    np.testing.assert_allclose(
        np.asarray(direct.particle_flux_vm_psi_hat_vs_x).sum(axis=0),
        np.asarray(direct.particle_flux_vm_psi_hat),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(direct.heat_flux_vm_psi_hat_vs_x).sum(axis=0),
        np.asarray(direct.heat_flux_vm_psi_hat),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(direct.fsab_flow_vs_x).sum(axis=0),
        np.asarray(direct.fsab_flow),
        rtol=2e-13,
        atol=2e-13,
    )

    f0_full = td.f0_v3_from_operator(op)
    assert f0_full.shape == op.fblock.f_shape
    np.testing.assert_allclose(np.asarray(f0_full[:, :, 1:, :, :]), 0.0, atol=0.0)
    phi1_override = jnp.zeros((op.n_theta, op.n_zeta), dtype=jnp.float64)
    f0_phi1 = td.f0_l0_v3_from_operator_phi1(op, phi1_override)
    assert f0_phi1.shape == (op.n_species, op.n_x, op.n_theta, op.n_zeta)
    assert not np.allclose(np.asarray(f0_phi1), np.asarray(td.f0_l0_v3_from_operator(op)))

    op_low_pitch = _ToyOp(n_xi=1)
    low_pitch = td.v3_transport_diagnostics_vm_only(op_low_pitch, x_full=_state(op_low_pitch))
    np.testing.assert_allclose(np.asarray(low_pitch.fsab_flow), 0.0, atol=0.0)

    with pytest.raises(ValueError, match="x_full must have shape"):
        td._v3_transport_diagnostics_vm_only_from_f0_l0(op, x_full=x_full[:-1], f0_l0=td.f0_l0_v3_from_operator(op))


def test_transport_precompute_cache_eviction_and_batch_shape_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    op1 = _ToyOp()
    op2 = _ToyOp(rhs_mode=3)
    td._TRANSPORT_DIAG_PRECOMPUTE_CACHE.clear()
    td._TRANSPORT_DIAG_PRECOMPUTE_ORDER.clear()
    monkeypatch.setattr(td, "_TRANSPORT_DIAG_PRECOMPUTE_CACHE_MAX", 1)

    first = td._transport_diag_precompute_cached(op1)
    assert td._transport_diag_precompute_cached(op1) is first
    second = td._transport_diag_precompute_cached(op2)
    assert second is not first
    assert id(op1) not in td._TRANSPORT_DIAG_PRECOMPUTE_CACHE
    assert id(op2) in td._TRANSPORT_DIAG_PRECOMPUTE_CACHE

    with pytest.raises(ValueError, match="non-empty"):
        td._stack_full_system_operators([])
    with pytest.raises(ValueError, match="x_full_stack must have shape"):
        td.v3_transport_diagnostics_vm_only_batch(op_stack=op1, x_full_stack=jnp.zeros((op1.total_size,)))
    with pytest.raises(ValueError, match="x_full_stack must have shape"):
        td.v3_transport_diagnostics_vm_only_batch_op0(op0=op1, x_full_stack=jnp.zeros((op1.total_size,)))
    with pytest.raises(ValueError, match="x_full_stack must have shape"):
        td.v3_transport_diagnostics_vm_only_batch_op0_precomputed(
            op0=op1,
            precomputed=first,
            x_full_stack=jnp.zeros((op1.total_size,)),
        )


def test_rhsmode1_output_fields_sources_phi1_and_batch_consistency() -> None:
    op = _ToyOp(constraint_scheme=2)
    x_full = _state(op)
    fields = td.v3_rhsmode1_output_fields_vm_only(op, x_full=x_full)
    sources = np.asarray(fields["sources"], dtype=np.float64)
    expected_sources = np.asarray(x_full[op.f_size :].reshape((op.n_species, op.n_x)).T)
    np.testing.assert_allclose(sources, expected_sources, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(fields["particleFlux_vm_psiHat_vs_x"]).sum(axis=0),
        np.asarray(fields["particleFlux_vm_psiHat"]),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(np.asarray(fields["particleFluxBeforeSurfaceIntegral_vE"]), 0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(fields["FSABjHat"]),
        np.dot(np.asarray(op.z_s), np.asarray(fields["FSABFlow"])),
        rtol=2e-13,
        atol=2e-13,
    )

    batch_one = td.v3_rhsmode1_output_fields_vm_only_batch(op, x_full_stack=x_full)
    np.testing.assert_allclose(np.asarray(batch_one["FSABFlow"][0]), np.asarray(fields["FSABFlow"]))
    batch_two = td.v3_rhsmode1_output_fields_vm_only_batch(op, x_full_stack=jnp.stack([x_full, 0.5 * x_full]))
    assert batch_two["FSABFlow"].shape == (2, op.n_species)

    op_phi1 = _ToyOp(constraint_scheme=1, include_phi1=True)
    x_phi1 = _state(op_phi1)
    fields_phi1 = td.v3_rhsmode1_output_fields_vm_only_phi1(op_phi1, x_full=x_phi1)
    expected_sources_phi1 = np.asarray(
        x_phi1[op_phi1.f_size + op_phi1.phi1_size :].reshape((op_phi1.n_species, 2)).T
    )
    np.testing.assert_allclose(np.asarray(fields_phi1["sources"]), expected_sources_phi1, rtol=0.0, atol=0.0)
    batch_phi1 = td.v3_rhsmode1_output_fields_vm_only_phi1_batch(op_phi1, x_full_stack=x_phi1)
    np.testing.assert_allclose(np.asarray(batch_phi1["FSABFlow"][0]), np.asarray(fields_phi1["FSABFlow"]))

    with pytest.raises(ValueError, match="x_full_stack must have shape"):
        td.v3_rhsmode1_output_fields_vm_only_batch(op, x_full_stack=jnp.zeros((2, op.total_size - 1)))
    with pytest.raises(ValueError, match="x_full_stack must have shape"):
        td.v3_rhsmode1_output_fields_vm_only_phi1_batch(op_phi1, x_full_stack=jnp.zeros((2, op_phi1.total_size - 1)))


def test_radial_current_coordinate_conversions_and_observable_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _ToyOp()
    x_full = _state(op)
    psi_current = td.radial_current_vm_psi_hat_from_state(op, x_full=x_full)
    np.testing.assert_allclose(td.radial_current_vm_from_state(op, x_full=x_full), psi_current)

    psi_a_hat = 0.7
    a_hat = 1.8
    r_n = 0.6
    np.testing.assert_allclose(
        td.radial_current_vm_from_state(
            op,
            x_full=x_full,
            radial_coordinate="rHat",
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
            r_n=r_n,
        ),
        psi_current * a_hat / (2.0 * psi_a_hat * r_n),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        td.radial_current_vm_from_state(
            op,
            x_full=x_full,
            radial_coordinate="rN",
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
            r_n=r_n,
        ),
        psi_current / (2.0 * psi_a_hat * r_n),
        rtol=2e-13,
        atol=2e-13,
    )
    with pytest.raises(ValueError, match="required"):
        td.radial_current_vm_from_state(op, x_full=x_full, radial_coordinate="rN")
    with pytest.raises(ValueError, match="radial_coordinate"):
        td.radial_current_vm_from_state(op, x_full=x_full, radial_coordinate="psiN", psi_a_hat=1.0, a_hat=1.0, r_n=1.0)

    calls: list[tuple[int, int, float]] = []

    def fake_probe(observable, *, size: int, chunk_size: int):
        value_at_zero = float(observable(jnp.zeros((size,), dtype=jnp.float64)))
        calls.append((int(size), int(chunk_size), value_at_zero))
        return jnp.arange(size, dtype=jnp.float64), value_at_zero

    monkeypatch.setattr("sfincs_jax.sensitivity.probe_linear_observable_vector", fake_probe)
    c_psi, j0_psi = td.radial_current_vm_psi_hat_observable_vector(op, chunk_size=7)
    c_rn, j0_rn = td.radial_current_vm_observable_vector(
        op,
        radial_coordinate="rN",
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        r_n=r_n,
        chunk_size=5,
    )
    assert c_psi.shape == (op.total_size,)
    assert c_rn.shape == (op.total_size,)
    assert np.isfinite(float(j0_psi))
    assert np.isfinite(float(j0_rn))
    assert calls[0][:2] == (op.total_size, 7)
    assert calls[1][:2] == (op.total_size, 5)


def test_transport_output_fields_chunked_and_matrix_formula_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        td,
        "v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit",
        td.v3_transport_diagnostics_vm_only_batch_op0_precomputed,
    )
    monkeypatch.setattr(
        td,
        "v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit",
        td.v3_transport_diagnostics_vm_only_batch_op0_precomputed,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DIAG_REMAT", "0")
    td._TRANSPORT_DIAG_PRECOMPUTE_CACHE.clear()
    td._TRANSPORT_DIAG_PRECOMPUTE_ORDER.clear()

    op = _ToyOp(rhs_mode=2, constraint_scheme=2)
    states = {1: _state(op, scale=1.0), 2: _state(op, scale=0.4), 3: _state(op, scale=-0.25)}
    full = td.v3_transport_output_fields_vm_only(op0=op, state_vectors_by_rhs=states, chunk_size=0)
    chunked = td.v3_transport_output_fields_vm_only(op0=op, state_vectors_by_rhs=states, chunk_size=1)
    for key in (
        "FSABFlow",
        "particleFlux_vm_psiHat",
        "heatFlux_vm_psiHat",
        "particleFluxBeforeSurfaceIntegral_vm",
        "heatFluxBeforeSurfaceIntegral_vm",
        "particleFlux_vm_psiHat_vs_x",
        "sources",
    ):
        np.testing.assert_allclose(np.asarray(chunked[key]), np.asarray(full[key]), rtol=2e-13, atol=2e-13)

    with pytest.raises(ValueError, match="Missing state vector"):
        td.v3_transport_output_fields_vm_only(op0=op, state_vectors_by_rhs={1: states[1], 2: states[2]})
    with pytest.raises(ValueError, match="RHSMode=2 expects"):
        td.v3_transport_matrix_column(op=op, geom=_geom(), which_rhs=4, diag=td.v3_transport_diagnostics_vm_only(op, x_full=states[1]))
    with pytest.raises(ValueError, match="expected shape"):
        td.v3_transport_matrix_from_flux_arrays(
            op=op,
            geom=_geom(),
            particle_flux_vm_psi_hat=jnp.zeros((op.n_species, 2)),
            heat_flux_vm_psi_hat=jnp.zeros((op.n_species, 3)),
            fsab_flow=jnp.zeros((op.n_species, 3)),
        )
    with pytest.raises(ValueError, match="only defined"):
        td.transport_matrix_size_from_rhs_mode(1)

    diag_cols = [td.v3_transport_diagnostics_vm_only(op, x_full=states[i]) for i in (1, 2, 3)]
    pf = jnp.stack([diag.particle_flux_vm_psi_hat for diag in diag_cols], axis=1)
    hf = jnp.stack([diag.heat_flux_vm_psi_hat for diag in diag_cols], axis=1)
    flow = jnp.stack([diag.fsab_flow for diag in diag_cols], axis=1)
    matrix = td.v3_transport_matrix_from_flux_arrays(
        op=op,
        geom=_geom(),
        particle_flux_vm_psi_hat=pf,
        heat_flux_vm_psi_hat=hf,
        fsab_flow=flow,
    )
    column1 = td.v3_transport_matrix_column(op=op, geom=_geom(), which_rhs=1, diag=diag_cols[0])
    np.testing.assert_allclose(np.asarray(matrix[:, 0]), np.asarray(column1), rtol=2e-13, atol=2e-13)

    fallback_matrix = td.v3_transport_matrix_from_flux_arrays(
        op=op,
        geom=_geom(zero_flux_functions=True),
        particle_flux_vm_psi_hat=pf,
        heat_flux_vm_psi_hat=hf,
        fsab_flow=flow,
    )
    assert np.all(np.isfinite(np.asarray(fallback_matrix)))

    op3 = _ToyOp(rhs_mode=3, constraint_scheme=1)
    states3 = {1: _state(op3, scale=0.2), 2: _state(op3, scale=-0.1)}
    fields3 = td.v3_transport_output_fields_vm_only(op0=op3, state_vectors_by_rhs=states3, chunk_size=1)
    assert fields3["sources"].shape == (2, op3.n_species, 2)
    diag3_1 = td.v3_transport_diagnostics_vm_only(op3, x_full=states3[1])
    diag3_2 = td.v3_transport_diagnostics_vm_only(op3, x_full=states3[2])
    pf3 = jnp.stack([diag3_1.particle_flux_vm_psi_hat, diag3_2.particle_flux_vm_psi_hat], axis=1)
    hf3 = jnp.stack([diag3_1.heat_flux_vm_psi_hat, diag3_2.heat_flux_vm_psi_hat], axis=1)
    flow3 = jnp.stack([diag3_1.fsab_flow, diag3_2.fsab_flow], axis=1)
    matrix3 = td.v3_transport_matrix_from_flux_arrays(
        op=op3,
        geom=_geom(),
        particle_flux_vm_psi_hat=pf3,
        heat_flux_vm_psi_hat=hf3,
        fsab_flow=flow3,
    )
    np.testing.assert_allclose(
        np.asarray(matrix3[:, 2 - 1]),
        np.asarray(td.v3_transport_matrix_column(op=op3, geom=_geom(), which_rhs=2, diag=diag3_2)),
        rtol=2e-13,
        atol=2e-13,
    )
    with pytest.raises(ValueError, match="RHSMode=3 expects"):
        td.v3_transport_matrix_column(op=op3, geom=_geom(), which_rhs=3, diag=diag3_1)
