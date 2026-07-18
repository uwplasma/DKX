"""End-to-end tests for the canonical RHSMode=1 driver and writer.

- ``dkx.run.run_profile`` solved states must reproduce the frozen
  Fortran v3 ``stateVector`` fixtures, and the written h5 file must match the
  recorded Fortran ``sfincsOutput.h5`` golden where one exists (the tiny PAS
  scheme-1 axisymmetric fixture);
- stdout must contain the diagnostics.F90 species-results block rendered by
  the golden-pinned ``console.species_results_lines`` helpers
  (format byte-parity is pinned in ``tests/test_inputs_console.py``), with
  values matching the reference-data-v2 Fortran log;
- the auto solver policy must pick tier 1 (``block_tridiagonal``) for the PAS
  family and tier 2 (``gcrot``) for Fokker-Planck;
- ``jax.grad`` of FSABjHat must flow through the pure moment/solve path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "ref"
_REFERENCE = Path("/Users/rogerio/local/reference-data-v2")

WITH_ER_BASE = "pas_dkes_withEr_tiny"
FIXTURES = (
    "pas_1species_PAS_noEr_tiny_scheme1",
    "quick_2species_FPCollisions_noEr",
    WITH_ER_BASE,
)

EXPECTED_SOLVE_METHOD = {
    "pas_1species_PAS_noEr_tiny_scheme1": "block_tridiagonal",
    "quick_2species_FPCollisions_noEr": "gcrot",
    WITH_ER_BASE: "block_tridiagonal",
}

# Tier 1 is a direct solve, but the frozen Fortran states carry the Fortran
# Krylov tolerance: a tight deck keeps the state referee sharp.
_WITH_ER_TOLERANCE_LINE = "  Nx = 3\n  solverTolerance = 1d-13\n"

_CACHE: dict[str, dict] = {}


def _read_h5(path: Path) -> dict[str, np.ndarray]:
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[...]))
    return out


def _deck_path(base: str, tmp_dir: Path) -> Path:
    if base != WITH_ER_BASE:
        return REF / f"{base}.input.namelist"
    from test_drift_kinetic import PAS_DKES_ER_TEXT

    text = PAS_DKES_ER_TEXT.replace("  Nx = 3\n", _WITH_ER_TOLERANCE_LINE)
    assert "solverTolerance" in text
    deck = tmp_dir / f"{base}.input.namelist"
    deck.write_text(text)
    return deck


def _case(base: str, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One canonical run per fixture, cached module-wide."""
    if base not in _CACHE:
        from dkx.run import run_profile

        tmp_dir = tmp_path_factory.mktemp(f"rhsmode1_{base}")
        deck = _deck_path(base, tmp_dir)
        lines: list[str] = []
        run = run_profile(deck, out_path=tmp_dir / f"{base}.canonical.h5", emit=lines.append)
        _CACHE[base] = {
            "deck": deck,
            "run": run,
            "lines": tuple(lines),
            "canonical": _read_h5(run.output_path),
        }
    return _CACHE[base]


def _assert_scaled_close(a: np.ndarray, b: np.ndarray, *, tol: float, label: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    if a.size == 0:
        return
    scale = max(1.0, float(np.max(np.abs(a))))
    err = float(np.max(np.abs(a - b)))
    assert err <= tol * scale, f"{label}: max|diff|={err:g} > {tol:g}*scale({scale:g})"


# ---------------------------------------------------------------------------
# Solved-state and physics anchors vs the frozen Fortran references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_run_profile_state_and_bootstrap_closure(
    base: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from dkx.validation.fortran import read_petsc_vec

    case = _case(base, tmp_path_factory)
    run = case["run"]
    assert run.solve_result.converged
    assert run.state_vector.shape == (run.operator.total_size,)

    # Direct Fortran state referee where a frozen stateVector exists.
    statevec = REF / f"{base}.stateVector.petscbin"
    if statevec.exists():
        x_ref = read_petsc_vec(statevec).values
        scale = max(1.0, float(np.max(np.abs(x_ref))))
        err = float(np.max(np.abs(np.asarray(run.state_vector) - x_ref)))
        assert err <= 1e-8 * scale, f"state referee: max|diff|={err:g}"

    # Bootstrap-current closure as a physics anchor.
    z_s = np.asarray(run.operator.z_s, dtype=np.float64)
    np.testing.assert_allclose(
        float(run.moments["FSABjHat"]),
        float(np.dot(z_s, np.asarray(run.moments["FSABFlow"]))),
        rtol=0.0,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# h5 field-by-field equality vs the recorded Fortran golden
# ---------------------------------------------------------------------------


def test_profile_h5_matches_fortran_golden(tmp_path_factory: pytest.TempPathFactory) -> None:
    from dkx.compare import compare_sfincs_outputs

    base = "pas_1species_PAS_noEr_tiny_scheme1"
    case = _case(base, tmp_path_factory)
    golden = REF / f"{base}.sfincsOutput.h5"
    results = compare_sfincs_outputs(
        a_path=golden, b_path=Path(str(case["run"].output_path)), rtol=1e-8, atol=1e-9
    )
    failures = [r for r in results if not r.ok]
    assert not failures, f"canonical vs Fortran golden mismatches: {[f.key for f in failures]}"


# ---------------------------------------------------------------------------
# Console: the diagnostics.F90 species-results block
# ---------------------------------------------------------------------------


def _expected_species_block(run) -> tuple[str, ...]:
    """Re-render the species-results block from the run's moment table."""
    from dkx import console

    moments = run.moments
    op = run.operator
    mach = np.asarray(moments["MachUsingFSAThermalSpeed"], dtype=np.float64)
    sources = np.asarray(moments["sources"], dtype=np.float64)
    entries = []
    for s in range(op.n_species):
        entry = {
            key: float(np.asarray(moments[key], dtype=np.float64)[s])
            for key in (
                "FSADensityPerturbation", "FSABFlow", "FSAPressurePerturbation", "NTV",
                "particleFlux_vm0_psiHat", "particleFlux_vm_psiHat",
                "momentumFlux_vm0_psiHat", "momentumFlux_vm_psiHat",
                "heatFlux_vm0_psiHat", "heatFlux_vm_psiHat",
            )  # fmt: skip
        }
        entry["classicalParticleFlux"] = float(moments["classicalParticleFlux_psiHat"][s])
        entry["classicalHeatFlux"] = float(moments["classicalHeatFlux_psiHat"][s])
        entry["MachMax"] = float(np.max(mach[s]))
        entry["MachMin"] = float(np.min(mach[s]))
        if op.constraint_scheme in (1, 3, 4):
            entry["particleSource"] = float(sources[0, s])
            entry["heatSource"] = float(sources[1, s])
        elif op.constraint_scheme == 2:
            entry["sources"] = [float(v) for v in sources[:, s]]
        entries.append(entry)
    return console.species_results_lines(
        species_results=entries,
        fsab_j_hat=float(np.asarray(moments["FSABjHat"], dtype=np.float64)),
        include_phi1=False,
        constraint_scheme=op.constraint_scheme,
    )


def test_console_species_results_block_quick2species(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from dkx import console

    case = _case("quick_2species_FPCollisions_noEr", tmp_path_factory)
    run = case["run"]
    lines = [ln.rstrip() for ln in case["lines"]]

    # The whole v3 console flow surrounds the block.
    for banner_line in console.banner_lines(n_procs=1):
        assert banner_line.rstrip() in lines
    assert console.entering_solver_line().rstrip() in lines
    assert console.main_solve_begin_line().rstrip() in lines
    assert lines[-1] == " Goodbye!"

    # Species-results block: byte parity vs the golden-pinned format helpers
    # (test_inputs_console.py pins species_results_lines to the Fortran log).
    expected = [ln.rstrip() for ln in _expected_species_block(run)]
    start = lines.index(" Results for species            1 :")
    assert lines[start : start + len(expected)] == expected

    # Values match the reference-data-v2 Fortran run (solverTolerance=1e-6).
    golden_log = _REFERENCE / "quick_2species_FPCollisions_noEr" / "stdout.log"
    if not golden_log.exists():
        pytest.skip(f"reference file not available: {golden_log}")
    golden = golden_log.read_text().splitlines()
    fsab_j_golden = float(
        next(ln for ln in golden if ln.startswith(" FSABjHat")).split(":")[1]
    )
    np.testing.assert_allclose(
        float(run.moments["FSABjHat"]), fsab_j_golden, rtol=1e-4, atol=0.0
    )
    flow_golden = [float(ln.split(":")[1]) for ln in golden if ln.startswith("    FSABFlow:")]
    np.testing.assert_allclose(
        np.asarray(run.moments["FSABFlow"], dtype=np.float64), flow_golden, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# Solver policy: tier 1 for PAS, tier 2 for Fokker-Planck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_auto_policy_tier_selection(base: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    case = _case(base, tmp_path_factory)
    assert case["run"].solve_result.method == EXPECTED_SOLVE_METHOD[base]


def test_run_profile_rejects_transport_modes() -> None:
    from dkx.run import run_profile

    with pytest.raises(NotImplementedError, match="RHSMode"):
        run_profile(REF / "monoenergetic_PAS_tiny_scheme1.input.namelist", emit=None)


# ---------------------------------------------------------------------------
# End-to-end parity for the tangential magnetic drifts (magneticDriftScheme=1)
# ---------------------------------------------------------------------------


def test_run_profile_magnetic_drifts_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical run_profile with tangential magnetic drifts runs end to end.

    The ``magneticDriftScheme=1`` Boozer (geometryScheme 11) deck routes through
    the canonical stack.  Because the drift couples L±2 the operator is not
    block-tridiagonal, so :func:`dkx.solve.solve` routes it to tier-2
    GCROT (falling back to the exact tier-3 direct solve on this tiny
    collisionless fixture).  The drift assembly is validated element-wise
    against Fortran v3 in ``tests/test_magnetic_drifts_parity.py`` and the
    scheme 2-9 output families against Fortran goldens in
    ``tests/test_output_h5_magdrift_schemes_parity.py``.
    """
    from dkx.run import run_profile

    monkeypatch.setenv("DKX_EQUILIBRIA_DIRS", str(REF))
    deck = REF / "magdrift_1species_tiny.input.namelist"

    run = run_profile(deck, out_path=tmp_path / "magdrift.canonical.h5", emit=None)
    assert run.solve_result.converged
    canonical = _read_h5(run.output_path)

    for key in ("particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "FSABFlow", "FSABjHat"):
        assert key in canonical, f"missing flux dataset {key!r}"
        assert np.all(np.isfinite(np.asarray(canonical[key], dtype=np.float64))), key


# ---------------------------------------------------------------------------
# Differentiability: jax.grad of FSABjHat through the pure solve + moments
# ---------------------------------------------------------------------------


def test_grad_fsabjhat_wrt_t_hat_matches_finite_differences() -> None:
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from dkx.drift_kinetic import KineticOperator
    from dkx.namelist import read_sfincs_input
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve

    op0 = KineticOperator.from_namelist(
        read_sfincs_input(REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    )

    def fsab_j_hat(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        # Thread THat through the operator pytree (streaming/mirror and the
        # RHS drive depend on it; the PAS collision matrices stay frozen, so
        # finite differences see the same function) — as in tests/test_solve.py.
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        result = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
        return profile_moments_from_operator(op, result.x)["FSABjHat"]

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(fsab_j_hat)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float(
        (fsab_j_hat(jnp.asarray(t0 + eps)) - fsab_j_hat(jnp.asarray(t0 - eps))) / (2.0 * eps)
    )
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)
