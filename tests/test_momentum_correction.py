"""Sugama-Nishimura parallel-momentum correction: physics and AD gates.

Consistency gates for :mod:`dkx.momentum_correction` (moment method of
H. Sugama and S. Nishimura, Phys. Plasmas 9, 4637 (2002); 15, 042502 (2008);
monoenergetic-database application of H. Maassberg, C. D. Beidler, and Y.
Turkin, Phys. Plasmas 16, 072504 (2009)).

Measured reference values (recorded 2026-07-13, float64):

===========================================================  ===============
quantity                                                     measured
===========================================================  ===============
single-species restoring factor M0/Mcorr (scheme-1 tokamak)  0.9310
single-species solve corrected/uncorrected vs that factor    0.0 (exact)
friction-matrix momentum conservation sum_a Lambda_ab        <= 1e-18
2-species H+C6 tokamak bootstrap: |PAS - FP|                 ~3.81e-2
2-species H+C6 tokamak bootstrap: |corrected - FP|           ~6.9e-3 (82% cut)
jax.grad(corrected bootstrap) w.r.t. T_a vs central FD       1.5e-10 rel
===========================================================  ===============

The multi-species gate compares a pitch-angle-scattering (PAS) bootstrap to a
full Fokker-Planck (FP) bootstrap on the *same* deck.  PAS is momentum-
deficient: the impurity parallel flow is a factor ~700 too small because the
Lorentz operator lets it slip relative to the bulk.  The momentum correction
couples the flows through the parallel friction matrix and locks the trace
impurity to the bulk, which -- weighted by its charge Z=6 -- moves ``<B j_||>``
most of the way to the momentum-conserving FP value.  The two are *not* equal:
the correction is the leading-order (single-moment) parallel-momentum
restoration and does not capture the collision operator's energy-scattering
difference; the residual (~15% of the FP bootstrap here) is measured and
asserted with a comfortable envelope, not driven to zero.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

REF_DIR = Path(__file__).parent / "ref"
MONO_DECK = REF_DIR / "monoenergetic_PAS_tiny_scheme1.input.namelist"

# A two-species (hydrogen + fully stripped carbon) tokamak deck.  geometryScheme
# = 1 keeps GHat > 0 (the Beidler monoenergetic normalization needs a positive
# aspect ratio); collisionOperator is templated so the same deck runs both PAS
# (=1) and full Fokker-Planck (=0).
_TWO_SPECIES_DECK = """\
&general
  RHSMode = 1
  saveMatricesAndVectorsInBinary = .false.
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0d+0
  GHat = 1.0d+0
  IHat = 0.0d+0
  iota = 1.31d+0
  epsilon_t = 0.1d+0
  epsilon_h = 0.0d+0
  helicity_l = 1
  helicity_n = 1
  psiAHat = 0.045d+0
  aHat = 0.1
/
&speciesParameters
  Zs = 1 6
  mHats = 1.0d+0 6.0d+0
  nHats = 0.6d+0 0.0667d+0
  THats = 0.5d+0 0.5d+0
  dNHatdrHats = -6.0d+0 -0.5d+0
  dTHatdrHats = -3.0d+0 -3.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.0d+0
  collisionOperator = {collop}
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 9
  Nzeta = 1
  Nxi = 12
  NL = 4
  Nx = 6
  solverTolerance = 1d-9
/
&otherNumericalParameters
  Nxi_for_x_option = 0
  xGridScheme = 5
/
&preconditionerOptions
/
"""


def _independent_restoring_factor(db, *, z, m, t, n, nu_n, x, w):
    """Recompute the single-species Sugama-Nishimura factor from the database.

    Independent (test-side) implementation of the parallel-viscosity energy
    convolution used by :func:`momentum_correction.parallel_viscosity`, to
    cross-check the module: ``M0 = <(m^2/T) nu_D^2 D33_SN>`` and
    ``Mcorr = <(m^2/T) nu_D^2 D33_SN F>`` with ``D33_SN = D33_PS (1 - D33*)``,
    ``F = 1/(1 - 3 m nu_D D33_SN / (2 T K <B^2>))``.
    """
    import jax.numpy as jnp

    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.monoenergetic import _dstar_lookup

    g = float(db.g_hat)
    ih = float(db.i_hat)
    io = float(db.iota)
    b0 = float(db.b0_over_bbar)
    fsab2 = float(np.asarray(db.fsab_hat2))
    x0 = float(db.x0)
    nudref = float(db.nu_d_hat_x0)
    gp = g + io * ih
    vth = np.sqrt(t / m)
    v = np.asarray(x) * vth
    nud = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=jnp.asarray(x), z_s=jnp.asarray([z]), m_hats=jnp.asarray([m]),
            n_hats=jnp.asarray([n]), t_hats=jnp.asarray([t]),
        )
    )[0]  # fmt: skip
    nprime = gp / b0 * (x0 / nudref) * nu_n * nud / v
    _d11, _d13, _d31, d33s = (
        np.asarray(a)
        for a in _dstar_lookup(db, jnp.asarray(nprime), jnp.zeros_like(jnp.asarray(nprime)))
    )
    nu_d = nu_n * nud
    d33_ps = v * v * fsab2 / (3.0 * nu_d * b0 * b0)
    d33_sn = d33_ps * (1.0 - d33s)
    k = np.asarray(x) ** 2
    quad = (4.0 / np.sqrt(np.pi)) * np.asarray(w) * k * np.exp(-k)
    arg = (3.0 * m * nu_d * d33_sn) / (2.0 * t * k * fsab2)
    factor = 1.0 / (1.0 - arg)
    base = (m * m / t) * nu_d * nu_d * d33_sn
    m0 = n * np.sum(quad * base)
    mc = n * np.sum(quad * base * factor)
    return m0 / mc


def test_single_species_reduces_to_momentum_restoring_factor() -> None:
    """Single species: corrected flow == uncorrected flow * restoring factor.

    With one species the parallel friction coupling vanishes identically
    (``sum_b l_ab = 0`` makes the single-species friction Laplacian zero), so
    the coupled solve collapses to the scalar Sugama-Nishimura momentum-
    restoring factor ``M^(0)/M`` acting on the uncorrected flow.  The factor is
    cross-checked against an independent energy convolution to ~1e-10.
    """
    import jax.numpy as jnp

    from dkx.monoenergetic import monoenergetic_database
    from dkx.momentum_correction import (
        parallel_friction_matrix,
        parallel_viscosity,
        solve_corrected_flows,
    )

    db = monoenergetic_database(MONO_DECK, list(np.geomspace(1e-3, 30.0, 16)), [0.0], emit=None)
    kw = dict(z_s=[1.0], m_hats=[1.0], n_hats=[1.0], t_hats=[1.0], nu_n=3.0e-4)
    visc = parallel_viscosity(db, n_x=48, x_max=5.0, **kw)
    factor = float(np.asarray(visc.restoring_factor)[0])

    # The scalar factor is a genuine momentum restoration (differs from 1).
    assert 0.5 < factor < 1.5
    assert abs(factor - 1.0) > 1e-3

    # The coupled solve reduces to (M0/Mcorr) * V_unc exactly for one species.
    gamma = parallel_friction_matrix(**kw)
    v_unc = 0.037
    res = solve_corrected_flows([v_unc], viscosity=visc, friction_matrix=gamma, z_s=[1.0])
    ratio = float(np.asarray(res.corrected_flows)[0]) / v_unc
    assert abs(ratio - factor) < 1e-10
    # Bootstrap of a single species is Z * corrected flow.
    assert float(np.asarray(res.corrected_bootstrap)) == pytest.approx(1.0 * v_unc * factor, abs=1e-12)

    # Cross-check the factor against an independent convolution.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    xq = 0.5 * 5.0 * (nodes + 1.0)
    wq = 0.5 * 5.0 * weights
    factor_indep = _independent_restoring_factor(
        db, z=1.0, m=1.0, t=1.0, n=1.0, nu_n=3.0e-4, x=jnp.asarray(xq), w=jnp.asarray(wq)
    )
    assert abs(factor - factor_indep) / abs(factor_indep) < 1e-10


def test_friction_matrix_is_symmetric_and_conserves_momentum() -> None:
    """The parallel friction matrix conserves total parallel momentum.

    ``gamma_ab`` is symmetric (self-adjointness ``l_ab = l_ba``) and the
    friction Laplacian ``Lambda_ab = delta_ab sum_c gamma_ac - gamma_ab`` has
    columns that sum to zero, so ``sum_a <B F_a> = 0`` -- like- and unlike-
    particle collisions exchange but do not create parallel momentum.
    """
    from dkx.momentum_correction import parallel_friction_matrix

    gamma = np.asarray(
        parallel_friction_matrix(
            z_s=[-1.0, 1.0, 6.0], m_hats=[2.72e-4, 1.0, 6.0],
            n_hats=[1.06, 1.0, 0.01], t_hats=[1.2, 1.0, 1.0], nu_n=8.4774e-3,
        )  # fmt: skip
    )
    assert gamma.shape == (3, 3)
    assert np.all(gamma > 0.0)
    np.testing.assert_allclose(gamma, gamma.T, rtol=0, atol=1e-300)
    lam = np.diag(gamma.sum(axis=1)) - gamma
    np.testing.assert_allclose(lam.sum(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(lam.sum(axis=1), 0.0, atol=1e-15)


def _run_profile_bootstrap(collop: int, tmp: Path):
    from dkx.run import run_profile

    path = tmp / f"two_species_c{collop}.namelist"
    path.write_text(_TWO_SPECIES_DECK.format(collop=collop))
    run = run_profile(path, emit=None)
    return run, path


def test_multispecies_correction_moves_bootstrap_toward_full_fp() -> None:
    """2-species PAS-corrected bootstrap moves toward the full-FP bootstrap.

    Runs the same H+C6 tokamak deck with pitch-angle (PAS) and full Fokker-
    Planck (FP) collisions, feeds the per-species PAS parallel flows to the
    momentum correction (viscosity/friction from a monoenergetic database of
    the same deck), and checks the corrected ``<B j_||>`` is substantially
    closer to the momentum-conserving FP value.  Measured: PAS 7.1e-3, FP
    4.5e-2, corrected 3.8e-2 -- the correction removes ~82% of the gap; the
    ~15% residual is the leading-order limitation and is only loosely bounded.
    """
    import jax.numpy as jnp

    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.drift_kinetic import _geometry_and_radial
    from dkx.inputs import load_sfincs_input
    from dkx.monoenergetic import monoenergetic_database
    from dkx.momentum_correction import (
        parallel_friction_matrix,
        parallel_viscosity,
        solve_corrected_flows,
    )
    from dkx.run import _grids_from_input, _raw_with_validated_overrides

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fp_run, _ = _run_profile_bootstrap(0, tmp)
        pas_run, pas_path = _run_profile_bootstrap(1, tmp)

        op = pas_run.operator
        nu_n = float(op.pas.nu_n)
        fp_boot = float(np.asarray(fp_run.moments["FSABjHat"]))
        pas_boot = float(np.asarray(pas_run.moments["FSABjHat"]))
        pas_flow = np.asarray(pas_run.moments["FSABFlow"])

        # Node-equivalent nuPrime range for a well-covered database.
        inp = load_sfincs_input(pas_path)
        raw = _raw_with_validated_overrides(inp)
        grids = _grids_from_input(inp, raw)
        geom, _radial = _geometry_and_radial(nml=raw, grids=grids)
        g_plus = float(geom.g_hat) + float(geom.iota) * float(geom.i_hat)
        b0 = float(geom.b0_over_bbar)
        x = np.asarray(op.x)
        vth = np.sqrt(np.asarray(op.t_hat) / np.asarray(op.m_hat))
        nud = np.asarray(
            nu_d_hat_pitch_angle_scattering_v3(
                x=op.x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
            )
        )
        nud_ref = float(
            np.asarray(
                nu_d_hat_pitch_angle_scattering_v3(
                    x=jnp.asarray([1.0]), z_s=op.z_s, m_hats=op.m_hat,
                    n_hats=op.n_hat, t_hats=op.t_hat,
                )
            )[0, 0]
        )  # fmt: skip
        nuprime = np.abs((g_plus / b0) * (1.0 / nud_ref) * nu_n * nud / (x[None, :] * vth[:, None]))
        grid = np.geomspace(max(nuprime.min() / 3, 1e-4), min(nuprime.max() * 3, 100.0), 14)
        db = monoenergetic_database(pas_path, [float(v) for v in grid], [0.0], emit=None)

        visc = parallel_viscosity(
            db, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat,
            nu_n=nu_n, x=op.x, x_weights=op.x_weights,
        )  # fmt: skip
        gamma = parallel_friction_matrix(
            z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat, nu_n=nu_n
        )
        res = solve_corrected_flows(pas_flow, viscosity=visc, friction_matrix=gamma, z_s=op.z_s)
        corr_boot = float(np.asarray(res.corrected_bootstrap))

    gap_pas = abs(pas_boot - fp_boot)
    gap_corr = abs(corr_boot - fp_boot)

    # PAS is strongly momentum-deficient here, FP much larger and same sign.
    assert np.sign(pas_boot) == np.sign(fp_boot)
    assert abs(fp_boot) > 3.0 * abs(pas_boot)
    # The correction moves the bootstrap toward FP and removes most of the gap.
    assert gap_corr < gap_pas
    assert gap_corr < 0.4 * gap_pas  # measured 0.14; envelope with margin
    # Corrected stays on the FP side of PAS without wildly overshooting.
    assert abs(corr_boot) > abs(pas_boot)
    assert abs(corr_boot) < 1.5 * abs(fp_boot)
    # The impurity (species 1) parallel flow is dragged up toward the bulk.
    assert abs(np.asarray(res.corrected_flows)[1]) > 10.0 * abs(pas_flow[1])


def test_gradient_of_corrected_bootstrap_matches_finite_differences() -> None:
    """d(corrected <B j_||>)/dT_a through the viscosity, friction, and solve.

    Differentiates the corrected bootstrap w.r.t. a species temperature (which
    flows through the deflection frequency, the parallel viscosity, the
    friction matrix, the uncorrected monoenergetic flows, and the linear
    solve) and compares ``jax.grad`` to central finite differences (measured
    relative deviation 1.5e-10; asserted rtol <= 1e-5).
    """
    import jax
    import jax.numpy as jnp

    from dkx.monoenergetic import monoenergetic_database
    from dkx.momentum_correction import momentum_corrected_bootstrap

    db = monoenergetic_database(MONO_DECK, list(np.geomspace(1e-3, 30.0, 12)), [0.0], emit=None)

    def bootstrap_of_temperature(t0: jnp.ndarray) -> jnp.ndarray:
        result = momentum_corrected_bootstrap(
            db, z_s=jnp.array([1.0, 6.0]), m_hats=jnp.array([1.0, 6.0]),
            n_hats=jnp.array([0.6, 0.0667]), t_hats=jnp.array([t0, 0.5]),
            nu_n=8.4774e-3, dn_hat_dpsi_hat=jnp.array([-6.0, -0.5]),
            dt_hat_dpsi_hat=jnp.array([-3.0, -3.0]), n_x=24, x_max=4.0,
        )  # fmt: skip
        return result.corrected_bootstrap

    t0 = jnp.asarray(0.5)
    grad_ad = float(jax.grad(bootstrap_of_temperature)(t0))
    eps = 1.0e-5
    grad_fd = float(
        (bootstrap_of_temperature(t0 + eps) - bootstrap_of_temperature(t0 - eps)) / (2.0 * eps)
    )
    assert np.isfinite(grad_ad) and abs(grad_fd) > 0.0
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5)


def test_public_api_facade_matches_module() -> None:
    """The api.py facade returns the same corrected bootstrap as the module."""
    import jax.numpy as jnp

    from dkx.api import momentum_corrected_bootstrap as facade
    from dkx.monoenergetic import monoenergetic_database
    from dkx.momentum_correction import momentum_corrected_bootstrap as direct

    db = monoenergetic_database(MONO_DECK, list(np.geomspace(1e-3, 30.0, 10)), [0.0], emit=None)
    kw = dict(
        z_s=jnp.array([1.0, 6.0]), m_hats=jnp.array([1.0, 6.0]), n_hats=jnp.array([0.6, 0.0667]),
        t_hats=jnp.array([0.5, 0.5]), nu_n=8.4774e-3, dn_hat_dpsi_hat=jnp.array([-6.0, -0.5]),
        dt_hat_dpsi_hat=jnp.array([-3.0, -3.0]), n_x=24, x_max=4.0,
    )  # fmt: skip
    a = facade(db, **kw)
    b = direct(db, **kw)
    assert float(np.asarray(a.corrected_bootstrap)) == pytest.approx(
        float(np.asarray(b.corrected_bootstrap)), rel=0, abs=0
    )
