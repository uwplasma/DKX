"""Splitting ``A - M`` by physical mechanism.

Every deck that leaves the pitch-angle-scattering family is preconditioned by
the SFINCS-simplified operator ``M``, inverted exactly by block-Thomas. When the
recycled Krylov route then converges slowly, the useful question is which of the
couplings ``M`` drops is responsible. These tests pin the answer's exactness:
the drift-kinetic operator is a sum of terms, so the split is an identity, not
an estimate.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.coarse_precond import _coarse_operator, _DROPPED_F_TERMS, dropped_couplings


def _deck(
    *,
    collision_operator: int = 1,
    x_dot: str = ".false.",
    xi_dot: str = ".false.",
    er: float = 0.0,
    n_theta: int = 5,
    n_zeta: int = 5,
    n_xi: int = 6,
    n_x: int = 3,
) -> str:
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
  epsilon_h = 0.05
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
  dNHatdrHats = -0.5 -0.5
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = {collision_operator}
  includeXDotTerm = {x_dot}
  includeElectricFieldTermInXiDot = {xi_dot}
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


def _operator(tmp_path: Path, **kwargs):
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    path = tmp_path / "input.namelist"
    path.write_text(_deck(**kwargs))
    return kinetic_operator_from_namelist(load_sfincs_input(path).raw)


def test_a_pitch_angle_deck_is_represented_exactly(tmp_path: Path) -> None:
    """``M`` is the whole operator when nothing structure-breaking is on.

    Pitch-angle scattering is already diagonal in speed and species, so the
    preconditioner's simplification removes nothing.

    This is why the structured direct route owns these decks: there is nothing
    for a Krylov method to correct.
    """
    diagnosis = dropped_couplings(_operator(tmp_path))
    assert diagnosis.relative_residual_operator == pytest.approx(0.0, abs=1e-14)
    assert diagnosis.dominant is None
    assert not any(diagnosis.weights.values())


def test_full_fokker_planck_is_what_the_preconditioner_drops(tmp_path: Path) -> None:
    diagnosis = dropped_couplings(_operator(tmp_path, collision_operator=0))
    assert diagnosis.dominant == "fokker_planck"
    assert diagnosis.relative_residual_operator > 0.0
    assert diagnosis.weights["fokker_planck"] == pytest.approx(1.0, rel=1e-9)


def test_the_electric_field_terms_are_separated_from_the_collisions(
    tmp_path: Path,
) -> None:
    """A deck with both must attribute to both, or the diagnosis is useless."""
    diagnosis = dropped_couplings(
        _operator(tmp_path, collision_operator=0, x_dot=".true.", xi_dot=".true.", er=12.0)
    )
    for mechanism in ("fokker_planck", "er_xdot", "er_xidot"):
        assert diagnosis.weights[mechanism] > 0.0, mechanism
    assert diagnosis.dominant in {"fokker_planck", "er_xdot", "er_xidot"}


def test_the_split_is_an_identity_not_an_estimate(tmp_path: Path) -> None:
    """``(A - M) f`` is exactly the sum of the dropped terms applied to ``f``.

    The operator is a sum of terms, so restoring one mechanism at a time onto
    the stripped operator and adding the responses must reproduce the whole
    difference to round-off. If this ever fails, a term is being double counted
    or one is missing from ``_DROPPED_F_TERMS`` and the weights are wrong.
    """
    op = _operator(tmp_path, collision_operator=0, x_dot=".true.", xi_dot=".true.", er=12.0)
    coarse = _coarse_operator(op)
    f = jnp.asarray(np.random.default_rng(0).standard_normal(op.f_shape), dtype=jnp.float64)

    base = coarse.apply_f(f)
    whole = op.apply_f(f) - base
    parts = sum(
        replace(coarse, **{field: getattr(op, field) for field in fields}).apply_f(f) - base
        for _, fields in _DROPPED_F_TERMS
        if any(getattr(op, field) is not getattr(coarse, field) for field in fields)
    )
    assert float(jnp.linalg.norm(parts - whole)) <= 1e-10 * float(jnp.linalg.norm(whole))


def test_the_diagnosis_does_not_depend_on_the_probe_draw(tmp_path: Path) -> None:
    """Different random probes must agree on which mechanism dominates."""
    op = _operator(tmp_path, collision_operator=0, x_dot=".true.", er=12.0)
    first = dropped_couplings(op, probes=3, seed=0)
    second = dropped_couplings(op, probes=3, seed=17)
    assert first.dominant == second.dominant
    for mechanism, weight in first.weights.items():
        assert second.weights[mechanism] == pytest.approx(weight, rel=0.25), mechanism


def test_probes_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="probes must be positive"):
        dropped_couplings(_operator(tmp_path), probes=0)


# --------------------------------------------------------------------------
# The shape of the dropped coupling, which decides how to recover it
# --------------------------------------------------------------------------


def _collision_matrix(tmp_path: Path, *, n_x: int):
    op = _operator(tmp_path, collision_operator=0, n_x=n_x)
    collision = op.fp if op.fp is not None else op.sugama
    assert collision is not None, "a full-Fokker-Planck deck must carry a dense operator"
    # (S, S, L, X, X): one speed block per Legendre index, so the operator is
    # diagonal in L and couples only species and speed.
    return np.asarray(collision.mat)


def test_the_collision_operator_is_upper_triangular_in_speed(tmp_path: Path) -> None:
    """Rosenbluth potentials make collisions an integral operator in speed.

    A row at one speed draws on integrals over the speeds above it, so in this
    basis the matrix is upper triangular to round-off. The plan's phase 2 rests
    on that: keeping the upper triangle (Fortran ``preconditioner_x = 2``) is
    then very nearly the full coupling, and it inverts by back-substitution
    reusing the block-Thomas factors rather than by a dense solve. If a change
    of speed basis ever breaks the triangularity, that argument breaks with it.
    """
    mat = _collision_matrix(tmp_path, n_x=8)
    rows, columns = np.tril_indices(mat.shape[-1], -1)
    lower = mat[..., rows, columns]
    assert np.linalg.norm(lower) / np.linalg.norm(mat) < 1e-2


def test_the_collision_mass_sits_in_the_corner_not_in_a_band(tmp_path: Path) -> None:
    """Why banding the speed coupling cannot help.

    The mass is at the largest ``|x - x'|``, not beside the diagonal, so the
    tridiagonal and diagonal-plus-superdiagonal simplifications SFINCS offers
    (``preconditioner_x = 3`` and ``4``) recover almost nothing. Measured:
    bandwidths 1 and 2 leave the residual operator where the diagonal already
    left it. This test pins the structural reason.
    """
    mat = _collision_matrix(tmp_path, n_x=8)
    n_x = mat.shape[-1]
    separation = np.abs(np.arange(n_x)[:, None] - np.arange(n_x)[None, :])
    total = np.linalg.norm(mat)
    first_superdiagonal = np.linalg.norm(mat[..., separation == 1]) / total
    corner = np.linalg.norm(mat[..., separation == n_x - 1]) / total
    assert corner > 10.0 * first_superdiagonal, (
        f"corner {corner:.3f} vs first superdiagonal {first_superdiagonal:.3f}: "
        "a band would be the right shape after all, and phase 2 needs rethinking"
    )
