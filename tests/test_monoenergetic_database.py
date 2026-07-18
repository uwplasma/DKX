"""Monoenergetic-database mode: scans, normalization, convolution, gradients.

Physics and consistency gates for :mod:`dkx.monoenergetic` (conventions
of C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011); monoenergetic
formulation of S.P. Hirshman et al., Phys. Fluids 29, 2951 (1986)).

Measured reference values (recorded 2026-07-12, float64, tier-1 direct
solves unless noted):

===========================================================  ===============
quantity                                                     measured
===========================================================  ===============
database point vs direct run_transport_matrix (2x2 grid)     <= ~1e-15 rel
D33* at nuPrime=10 (collisional limit -> 1)                  |1-D33*|=3.0e-4
Onsager |D13*+D31*|/|D31*| at nuPrime=1.2e-3                 7.9e-4
convolution vs full RHSMode=2 3x3, database at exact nodes   5.8e-14 max rel
convolution via 12-point log-nuPrime interpolated database   L11 3.0e-3,
                                                             L33 8.2e-4,
                                                             L31/L13 1.0e-1
jax.grad of convolved L11 w.r.t. Boozer amplitude vs FD      5.5e-10 rel
4x3 (nuPrime, EStar) scan wall time, tiny scheme-1 deck      1.8 s
===========================================================  ===============

The exact-node convolution agreement is a structural identity, not an
approximation check: for pitch-angle scattering with DKES trajectories the
kinetic equation is diagonal in speed, so the energy convolution of the
monoenergetic coefficients *is* the full-kinetic thermal transport matrix
evaluated with the same speed quadrature.  The interpolated-database numbers
measure the database grid resolution only (they tighten with the nuPrime
grid; D31/D13 carry the largest interpolation error because they change
sign across the scanned collisionality range).  With energy-scattering
(Fokker-Planck) collisions the convolution is the standard monoenergetic
approximation and exact agreement is not expected.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REF_DIR = Path(__file__).parent / "ref"
MONO_DECK = REF_DIR / "monoenergetic_PAS_tiny_scheme1.input.namelist"
RHS2_DECK = REF_DIR / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"


def _write_mono_deck(tmp_path: Path, *, nu_prime: float, e_star: float, name: str) -> Path:
    text = MONO_DECK.read_text()

    def sub(key: str, value: str, source: str) -> str:
        out, count = re.subn(rf"(?mi)^\s*{key}\s*=.*$", f"  {key} = {value}", source)
        assert count == 1, key
        return out

    text = sub("saveMatricesAndVectorsInBinary", ".false.", text)
    text = sub("nuPrime", f"{nu_prime!r}", text)
    text = sub("EStar", f"{e_star!r}", text)
    path = tmp_path / name
    path.write_text(text)
    return path


def test_database_points_match_direct_transport_matrix_solves(tmp_path: Path) -> None:
    """Every scan entry equals a direct namelist-level RHSMode=3 solve.

    The scan re-solves the deck with only the (nuPrime, EStar) coefficient
    fields replaced (`dataclasses.replace`); a fresh `run_transport_matrix`
    at the same point must give the same normalized coefficients to solver
    precision (~1e-12) -- a consistency check of the scan plumbing.
    """
    from dkx.monoenergetic import (
        monoenergetic_database,
        monoenergetic_dstar_from_transport_matrix,
    )
    from dkx.run import run_transport_matrix

    nu_values = [1.196132e-3, 0.3]
    er_values = [0.0, 2.0e-3]
    db = monoenergetic_database(MONO_DECK, nu_values, er_values)

    for i, nu_prime in enumerate(nu_values):
        for j, e_star in enumerate(er_values):
            deck = _write_mono_deck(
                tmp_path, nu_prime=nu_prime, e_star=e_star, name=f"point_{i}{j}.namelist"
            )
            run = run_transport_matrix(deck, emit=None)
            point = monoenergetic_dstar_from_transport_matrix(
                run.transport_matrix,
                nu_prime=nu_prime,
                delta=db.delta,
                g_hat=db.g_hat,
                i_hat=db.i_hat,
                iota=db.iota,
                b0_over_bbar=db.b0_over_bbar,
                fsab_hat2=db.fsab_hat2,
                r_hat=db.r_hat,
                x0=db.x0,
                w0=db.w0,
                nu_d_hat_x0=db.nu_d_hat_x0,
            )
            for key, direct in (
                ("d11_star", point.d11_star),
                ("d13_star", point.d13_star),
                ("d31_star", point.d31_star),
                ("d33_star", point.d33_star),
            ):
                scanned = float(np.asarray(getattr(db, key))[i, j])
                rel = abs(scanned - float(direct)) / max(abs(float(direct)), 1e-300)
                assert rel < 1e-12, f"{key} at ({nu_prime}, {e_star}): rel={rel:.3e}"


def test_normalization_physics_gates() -> None:
    """The ICNTS normalization has exact anchors that pin the conversion.

    - ``D33* -> 1`` in the collisional limit (the Lorentz parallel
      conductivity equals ``(v^2/3nu) <B^2>/B0^2`` exactly, geometry free),
      which validates the ``D33`` conversion including the deflection
      frequency actually applied at the speed node.
    - Onsager symmetry ``D13* = -D31*`` at ``EStar = 0`` couples the two
      independently converted off-diagonal channels.
    - ``D11* > 0`` and ``nu_star`` reproduces the scan collisionality.
    """
    from dkx.monoenergetic import monoenergetic_database

    db = monoenergetic_database(MONO_DECK, [1.196132e-3, 10.0], [0.0])

    d33_hi = float(np.asarray(db.d33_star)[1, 0])
    assert abs(d33_hi - 1.0) < 5.0e-4  # measured 3.0e-4

    d13 = float(np.asarray(db.d13_star)[0, 0])
    d31 = float(np.asarray(db.d31_star)[0, 0])
    assert abs(d13 + d31) / abs(d31) < 5.0e-3  # measured 7.9e-4

    assert np.all(np.asarray(db.d11_star) > 0.0)
    assert np.all(np.asarray(db.nu_star) > 0.0)
    assert db.eps_t > 0.0
    assert np.all(np.asarray(db.v_e) == 0.0)


def _full_kinetic_lb_matrix():
    """The RHSMode=2 3x3 matrix of the PAS/DKES deck in Beidler L^B form.

    Applies the hat-unit ``transportMatrix -> L^B`` conversion factors of the
    :mod:`dkx.monoenergetic` docstring to the full-kinetic solve
    (rows/columns 1,2 share the radial factor; index 3 is the parallel
    channel).
    """
    from dkx.drift_kinetic import _geometry_and_radial
    from dkx.inputs import load_sfincs_input
    from dkx.run import _grids_from_input, _raw_with_validated_overrides, run_transport_matrix

    run = run_transport_matrix(RHS2_DECK, emit=None)
    op = run.operator
    tm3 = np.asarray(run.transport_matrix)

    inp = load_sfincs_input(RHS2_DECK)
    raw = _raw_with_validated_overrides(inp)
    grids = _grids_from_input(inp, raw)
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)

    delta = float(op.delta)
    g_hat, i_hat = float(geom.g_hat), float(geom.i_hat)
    iota, b0 = float(geom.iota), float(geom.b0_over_bbar)
    z = float(op.z_s[0])
    m_hat, t_hat = float(op.m_hat[0]), float(op.t_hat[0])
    r_hat = float(np.sqrt(2.0 * abs(float(radial.psi_hat)) / abs(b0)))
    dr_dpsi = 1.0 / (b0 * r_hat)
    g_plus = g_hat + iota * i_hat
    vth = np.sqrt(t_hat / m_hat)

    c11 = -(delta**2 / 4.0) * dr_dpsi**2 * t_hat**2 * g_hat**2 / (z**2 * b0 * vth * g_plus)
    c13 = +(delta / 2.0) * (t_hat * g_hat / (z * b0)) * dr_dpsi
    c31 = -(delta / 2.0) * (t_hat * g_hat / (z * b0)) * dr_dpsi
    c33 = vth * g_plus / b0
    lb_full = np.array(
        [
            [c11 * tm3[0, 0], c11 * tm3[0, 1], c13 * tm3[0, 2]],
            [c11 * tm3[1, 0], c11 * tm3[1, 1], c13 * tm3[1, 2]],
            [c31 * tm3[2, 0], c31 * tm3[2, 1], c33 * tm3[2, 2]],
        ]
    )
    return run, lb_full


def _mapped_nu_primes(op) -> np.ndarray:
    """Database nuPrime values equivalent to each deck speed node (module docstring)."""
    import jax.numpy as jnp

    from dkx.collisions import nu_d_hat_pitch_angle_scattering_v3
    from dkx.drift_kinetic import _geometry_and_radial
    from dkx.inputs import load_sfincs_input
    from dkx.run import _grids_from_input, _raw_with_validated_overrides

    inp = load_sfincs_input(RHS2_DECK)
    raw = _raw_with_validated_overrides(inp)
    grids = _grids_from_input(inp, raw)
    geom, _radial = _geometry_and_radial(nml=raw, grids=grids)
    g_plus = float(geom.g_hat) + float(geom.iota) * float(geom.i_hat)
    b0 = float(geom.b0_over_bbar)

    x = np.asarray(op.x)
    vth = float(np.sqrt(op.t_hat[0] / op.m_hat[0]))
    nu_n = float(op.pas.nu_n)
    nud = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=op.x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
        )
    )[0]
    nud_ref = float(
        np.asarray(
            nu_d_hat_pitch_angle_scattering_v3(
                x=jnp.asarray([1.0]), z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
            )
        )[0, 0]
    )
    return (g_plus / b0) * (1.0 / nud_ref) * nu_n * nud / (x * vth)


def test_energy_convolution_reproduces_full_kinetic_transport_matrix() -> None:
    """Convolved database == full RHSMode=2 3x3 matrix (PAS/DKES identity).

    With the database evaluated at the exact node-equivalent nuPrime values
    and the deck's own speed quadrature, all nine thermal matrix entries
    must match the full-kinetic solve to solver precision (measured max
    relative deviation 5.8e-14; asserted < 1e-10).
    """
    from dkx.monoenergetic import energy_convolution, monoenergetic_database

    run, lb_full = _full_kinetic_lb_matrix()
    op = run.operator
    nu_map = _mapped_nu_primes(op)

    db = monoenergetic_database(RHS2_DECK, sorted(float(v) for v in nu_map), [0.0])
    assert db.x0 == pytest.approx(1.0)

    thermal = energy_convolution(
        db,
        z_s=op.z_s,
        m_hats=op.m_hat,
        t_hats=op.t_hat,
        n_hats=op.n_hat,
        nu_n=float(op.pas.nu_n),
        x=op.x,
        x_weights=op.x_weights,
    )
    lb_conv = np.asarray(thermal.l_matrix[0])
    rel = np.abs(lb_conv - lb_full) / np.abs(lb_full)
    assert rel.max() < 1e-10, rel


def test_energy_convolution_interpolated_database_envelope() -> None:
    """A generic log-spaced database reproduces the thermal matrix within
    the measured database-resolution envelope.

    Twelve log-spaced nuPrime points spanning the node-equivalent range gave
    (measured 2026-07-12): L11 3.0e-3, L33 8.2e-4, L31/L13 1.0e-1 relative.
    D31/D13 dominate the interpolation error because they change sign inside
    the scanned collisionality decade; the envelope tightens with the grid
    (16 points: 4.0e-2).  Asserted with ~2x margins.
    """
    from dkx.monoenergetic import energy_convolution, monoenergetic_database

    run, lb_full = _full_kinetic_lb_matrix()
    op = run.operator
    nu_map = _mapped_nu_primes(op)

    grid = np.geomspace(nu_map.min() / 1.5, nu_map.max() * 1.5, 12)
    db = monoenergetic_database(RHS2_DECK, [float(v) for v in grid], [0.0])
    thermal = energy_convolution(
        db,
        z_s=op.z_s,
        m_hats=op.m_hat,
        t_hats=op.t_hat,
        n_hats=op.n_hat,
        nu_n=float(op.pas.nu_n),
        x=op.x,
        x_weights=op.x_weights,
    )
    lb_conv = np.asarray(thermal.l_matrix[0])
    rel = np.abs(lb_conv - lb_full) / np.abs(lb_full)
    assert rel[0, 0] < 7e-3, rel  # L11, measured 3.0e-3
    assert rel[2, 2] < 2e-3, rel  # L33, measured 8.2e-4
    assert rel[0, 2] < 2e-1, rel  # L13, measured 1.0e-1
    assert rel[2, 0] < 2e-1, rel  # L31, measured 1.0e-1


def test_gradient_of_convolved_l11_matches_finite_differences() -> None:
    """d(L11)/d(B_mn) through scan + conversion + convolution (headline AD).

    Builds the scan operator on a differentiable ``from_fourier`` geometry,
    runs a 2-point nuPrime database with implicit-differentiation solves,
    convolves to the thermal L11, and compares ``jax.grad`` against central
    finite differences (measured relative deviation 5.5e-10; asserted
    rtol <= 1e-5).
    """
    import jax
    import jax.numpy as jnp

    from dkx.drift_kinetic import (
        _geometry_and_radial,
        kinetic_operator_from_namelist,
    )
    from dkx.inputs import load_sfincs_input
    from dkx.magnetic_geometry import FluxSurfaceGeometry
    from dkx.monoenergetic import (
        energy_convolution,
        monoenergetic_database_from_operator,
    )
    from dkx.run import _grids_from_input, _raw_with_validated_overrides

    inp = load_sfincs_input(MONO_DECK)
    raw = _raw_with_validated_overrides(inp)
    grids = _grids_from_input(inp, raw)
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)
    op0 = kinetic_operator_from_namelist(raw)

    g_hat, i_hat = float(geom.g_hat), float(geom.i_hat)
    iota, b0 = float(geom.iota), float(geom.b0_over_bbar)
    r_hat = float(np.sqrt(2.0 * abs(float(radial.psi_hat)) / abs(b0)))
    theta = jnp.asarray(grids.theta)
    zeta = jnp.asarray(grids.zeta)
    m_modes = jnp.asarray([0, 1, 2])
    n_modes = jnp.asarray([0, 0, 1])

    def l11_of(amp: jnp.ndarray) -> jnp.ndarray:
        bmnc = jnp.stack([jnp.asarray(1.0), amp, jnp.asarray(0.03)])
        fourier = FluxSurfaceGeometry.from_fourier(
            theta=theta, zeta=zeta, bmnc=bmnc, m=m_modes, n=n_modes,
            n_periods=10, iota=iota, g_hat=g_hat, i_hat=i_hat,
        )  # fmt: skip
        op_traced = replace(
            op0,
            b_hat=fourier.b_hat,
            db_hat_dtheta=fourier.db_hat_dtheta,
            db_hat_dzeta=fourier.db_hat_dzeta,
            d_hat=fourier.d_hat,
            b_hat_sup_theta=fourier.b_hat_sup_theta,
            b_hat_sup_zeta=fourier.b_hat_sup_zeta,
            b_hat_sub_theta=fourier.b_hat_sub_theta,
            b_hat_sub_zeta=fourier.b_hat_sub_zeta,
            fsab_hat2=fourier.fsab_hat2(
                theta_weights=op0.theta_weights, zeta_weights=op0.zeta_weights
            ),
        )
        db = monoenergetic_database_from_operator(
            op_traced, [0.2, 2.0], (0.0,),
            g_hat=g_hat, i_hat=i_hat, iota=iota, b0_over_bbar=b0, r_hat=r_hat,
            differentiable=True,
        )  # fmt: skip
        thermal = energy_convolution(
            db, z_s=[1.0], m_hats=[1.0], t_hats=[1.0], n_hats=[1.0],
            nu_n=3.0e-4, n_x=16, x_max=4.0,
        )  # fmt: skip
        return thermal.l11[0]

    amp0 = jnp.asarray(0.06)
    grad_ad = float(jax.grad(l11_of)(amp0))
    eps = 1.0e-5
    grad_fd = float((l11_of(amp0 + eps) - l11_of(amp0 - eps)) / (2.0 * eps))
    assert np.isfinite(grad_ad) and np.isfinite(grad_fd) and abs(grad_fd) > 0.0
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5)


def test_database_save_load_roundtrip(tmp_path: Path) -> None:
    """The compact npz format round-trips grids, coefficients, and provenance."""
    from dkx.monoenergetic import load_database, monoenergetic_database, save_database

    db = monoenergetic_database(MONO_DECK, [1.196132e-3, 0.5], [0.0, 1.0e-3])
    path = save_database(tmp_path / "db.npz", db)
    loaded = load_database(path)

    for key in ("nu_prime", "e_star", "d11_star", "d13_star", "d31_star", "d33_star"):
        np.testing.assert_array_equal(np.asarray(getattr(db, key)), np.asarray(getattr(loaded, key)))
    for key in (
        "x0", "w0", "nu_d_hat_x0", "delta", "alpha",
        "g_hat", "i_hat", "iota", "b0_over_bbar", "r_hat",
    ):  # fmt: skip
        assert float(getattr(db, key)) == float(getattr(loaded, key)), key
    assert float(np.asarray(db.fsab_hat2)) == float(np.asarray(loaded.fsab_hat2))
    assert "RHSMode" in loaded.deck_text
    np.testing.assert_allclose(np.asarray(loaded.nu_star), np.asarray(db.nu_star))

    with pytest.raises(FileExistsError):
        save_database(path, db, overwrite=False)


def test_cli_monoenergetic_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The thin CLI subcommand writes the npz database and prints the table."""
    from dkx import cli
    from dkx.monoenergetic import load_database

    out = tmp_path / "database.npz"
    rc = cli.main(
        [
            "monoenergetic-database",
            "--input", str(MONO_DECK),
            "--nu-prime", "1.196132e-3",
            "--e-star", "0.0",
            "--out", str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "D11*" in captured
    db = load_database(out)
    assert np.asarray(db.d11_star).shape == (1, 1)
    assert float(np.asarray(db.d11_star)[0, 0]) > 0.0
