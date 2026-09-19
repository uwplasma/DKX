"""The opt-in upper-triangle speed coupling in the coarse preconditioner.

Phase 2 step 2. The coarse preconditioner keeps only ``mat[s, s, l, x, x]`` of
the collision operator, Fortran ``preconditioner_x = 1``. Because the linearized
Fokker-Planck operator is an integral operator in speed, and so upper triangular
in this basis, retaining its upper triangle (``preconditioner_x = 2``) is very
nearly the full coupling. These tests pin that the option is correct and off by
default; whether it is *worth* enabling is a measurement, recorded in
``docs/experiments/``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.coarse_precond import (
    _coarse_operator,
    _strict_upper_speed_coupling,
    build_coarse_preconditioner,
)

DECK = """&general
/
&geometryParameters
 geometryScheme=1
 inputRadialCoordinate=3
 rN_wish=0.3
 B0OverBBar=1.0
 GHat=1.0
 IHat=0.0
 iota=1.31
 epsilon_t=0.1
 epsilon_h=0.05
 helicity_l=2
 helicity_n=5
 psiAHat=0.045
 aHat=0.1
/
&speciesParameters
 Zs=1
 mHats=1.0
 nHats=1.0
 THats=1.0
 dNHatdrHats=-0.5
 dTHatdrHats=-1.0
/
&physicsParameters
 Delta=4.5694d-3
 alpha=1.0
 nu_n=8.4774d-3
 Er=0.0
 collisionOperator={collision}
 includeXDotTerm=.false.
 includeElectricFieldTermInXiDot=.false.
 useDKESExBDrift=.true.
 includePhi1=.false.
/
&resolutionParameters
 Ntheta=5
 Nzeta=5
 Nxi=6
 NL=4
 Nx={nx}
 solverTolerance=1d-10
/
&otherNumericalParameters
 Nxi_for_x_option=0
/
"""


def _operator(tmp_path: Path, *, nx: int = 4, collision: int = 0):
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    path = tmp_path / "input.namelist"
    path.write_text(DECK.format(nx=nx, collision=collision))
    return kinetic_operator_from_namelist(load_sfincs_input(path).raw)


def _triangular_operator(op):
    """The operator the option's preconditioner is meant to invert exactly."""
    base = _coarse_operator(op)
    collision = op.fp if op.fp is not None else op.sugama
    n_x = op.f_shape[1]
    keep = (
        jnp.asarray(np.eye(collision.mat.shape[0]))[:, :, None, None, None]
        * jnp.asarray(np.triu(np.ones((n_x, n_x))))[None, None, None, :, :]
    )
    retained = replace(collision, mat=collision.mat * keep)
    field = "fp" if op.fp is not None else "sugama"
    return replace(base, **{field: retained})


def test_the_option_is_off_by_default(tmp_path: Path) -> None:
    """The default preconditioner must be byte-for-byte the one in use."""
    op = _operator(tmp_path)
    default, _ = build_coarse_preconditioner(op)
    explicit, _ = build_coarse_preconditioner(op, retain_speed_triangle=False)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))
    assert float(jnp.linalg.norm(default(v) - explicit(v))) == 0.0


def test_the_retained_block_is_strictly_upper(tmp_path: Path) -> None:
    op = _operator(tmp_path)
    upper = np.asarray(_strict_upper_speed_coupling(op, op._mask()))
    n_x = upper.shape[-1]
    rows, columns = np.tril_indices(n_x, 0)
    assert np.all(upper[..., rows, columns] == 0.0), "the diagonal belongs to D, not U"
    assert np.any(upper != 0.0), "a full-Fokker-Planck deck has coupling above it"


def test_pitch_angle_scattering_has_nothing_to_retain(tmp_path: Path) -> None:
    """Pitch-angle scattering is already speed-diagonal, so the option is inert."""
    op = _operator(tmp_path, collision=1)
    upper = _strict_upper_speed_coupling(op, op._mask())
    assert upper is None or float(jnp.linalg.norm(upper)) == 0.0


def test_it_inverts_the_operator_it_claims_to(tmp_path: Path) -> None:
    """``D^-1 U`` is nilpotent of index ``n_x``, so the sweeps reach the exact
    inverse of ``D + U`` rather than an approximation of it."""
    op = _operator(tmp_path)
    precond, _ = build_coarse_preconditioner(op, retain_speed_triangle=True)
    triangular = _triangular_operator(op)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))
    recovered = precond(triangular.apply(v))
    error = float(jnp.linalg.norm(recovered - v)) / float(jnp.linalg.norm(v))
    assert error < 1e-3, error


def test_it_approximates_the_real_operator_far_better(tmp_path: Path) -> None:
    """The point of the option: M moves much closer to A."""
    op = _operator(tmp_path)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))
    exact = op.apply(v)
    scale = float(jnp.linalg.norm(exact))
    diagonal = float(jnp.linalg.norm(exact - _coarse_operator(op).apply(v))) / scale
    triangle = float(jnp.linalg.norm(exact - _triangular_operator(op).apply(v))) / scale
    assert triangle < diagonal / 10.0, (diagonal, triangle)


def test_truncating_the_series_is_allowed_and_ordered(tmp_path: Path) -> None:
    """An integer truncates the sweeps, trading exactness in M for a cheaper
    apply. More sweeps must not move M further from the operator it inverts."""
    op = _operator(tmp_path)
    triangular = _triangular_operator(op)
    v = jnp.asarray(np.random.default_rng(1).standard_normal(op.total_size))
    rhs = triangular.apply(v)
    errors = []
    for sweeps in (1, 2, True):
        precond, _ = build_coarse_preconditioner(op, retain_speed_triangle=sweeps)
        errors.append(float(jnp.linalg.norm(precond(rhs) - v)) / float(jnp.linalg.norm(v)))
    assert errors[-1] <= errors[0], errors


def test_the_exact_triangle_solves_one_speed_at_a_time(tmp_path, monkeypatch) -> None:
    """How the exact inverse is reached, which is what it costs.

    ``D^-1 U`` is nilpotent, so sweeping the whole band ``n_x - 1`` times also
    lands on the exact inverse -- while reading every chain's factors once per
    sweep. Back-substitution over ``x`` reads each speed's factors once. Pin
    the calls, because the two are indistinguishable in the map they produce.
    """
    import dkx.coarse_precond as cp

    op = _operator(tmp_path, nx=4)
    calls = {"triangle": 0, "band": 0}
    for key, name in (("triangle", "_apply_dense_coarse_triangle"), ("band", "_apply_dense_coarse_factors")):
        real = getattr(cp, name)

        def counted(*args, _real=real, _key=key, **kwargs):
            calls[_key] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(cp, name, counted)

    v = jnp.asarray(np.random.default_rng(4).standard_normal(op.total_size))
    # The bordered operator applies the coarse inverse more than once per
    # preconditioner application, so measure that factor rather than assume it.
    default, _ = cp.build_coarse_preconditioner(op)
    default(v)
    per_application = calls["band"]
    assert per_application >= 1

    calls.update(triangle=0, band=0)
    exact, _ = cp.build_coarse_preconditioner(op, retain_speed_triangle=True)
    exact(v)
    # One compiled program per application, holding all n_x speeds, and no sweep
    # of the whole band.
    assert calls == {"triangle": per_application, "band": 0}

    calls.update(triangle=0, band=0)
    series, _ = cp.build_coarse_preconditioner(op, retain_speed_triangle=2)
    series(v)
    assert calls == {"triangle": 0, "band": 3 * per_application}


def test_back_substitution_and_the_full_series_are_the_same_map(tmp_path) -> None:
    """Both reach the inverse of ``D + U``, so they must agree to rounding."""
    op = _operator(tmp_path, nx=5)
    n_x = op.f_shape[1]
    v = jnp.asarray(np.random.default_rng(5).standard_normal(op.total_size))
    for exact, series in zip(
        build_coarse_preconditioner(op, retain_speed_triangle=True),
        build_coarse_preconditioner(op, retain_speed_triangle=n_x - 1),
    ):
        difference = float(jnp.linalg.norm(exact(v) - series(v)))
        assert difference / float(jnp.linalg.norm(series(v))) < 1e-10, difference


def test_the_exact_triangle_adjoint_is_its_transpose(tmp_path) -> None:
    """The adjoint runs the substitution the other way over ``x``, on the
    transposed couplings. That is only right if it is the transposed map."""
    op = _operator(tmp_path, nx=4)
    precond, precond_t = build_coarse_preconditioner(op, retain_speed_triangle=True)
    rng = np.random.default_rng(6)
    u = jnp.asarray(rng.standard_normal(op.total_size))
    w = jnp.asarray(rng.standard_normal(op.total_size))
    left = float(jnp.dot(precond(u), w))
    right = float(jnp.dot(u, precond_t(w)))
    assert abs(left - right) <= 1e-9 * max(abs(left), abs(right)), (left, right)


def test_it_refuses_the_route_it_does_not_implement(tmp_path, monkeypatch) -> None:
    """Where the bands do not fit, the coarse preconditioner is eliminated from
    generated rows and there is no band to retain the triangle in. Refuse
    loudly rather than silently returning the diagonal preconditioner, which
    would look like the option working while changing nothing."""
    import dkx.coarse_precond as cp

    op = _operator(tmp_path)
    monkeypatch.setattr(cp, "_coarse_bands_fit", lambda _op: False)
    with pytest.raises(NotImplementedError, match="dense-band route only"):
        cp.build_coarse_preconditioner(op, retain_speed_triangle=True)


def test_the_solver_offers_it_as_a_named_krylov_preconditioner(tmp_path) -> None:
    """``coarse_triangle`` reaches the same object through the solver's factory,
    which is how a measurement selects it without importing internals."""
    from dkx.solve import _TIER2_PRECONDITIONERS, build_tier2_preconditioner

    assert "coarse_triangle" in _TIER2_PRECONDITIONERS
    op = _operator(tmp_path)
    precond, precond_t = build_tier2_preconditioner(op, "coarse_triangle")
    direct, _ = build_coarse_preconditioner(op, retain_speed_triangle=True)
    v = jnp.asarray(np.random.default_rng(3).standard_normal(op.total_size))
    assert float(jnp.linalg.norm(precond(v) - direct(v))) == 0.0
    # the transpose is a distinct operator, not the forward one returned twice
    assert float(jnp.linalg.norm(precond_t(v) - precond(v))) > 0.0
