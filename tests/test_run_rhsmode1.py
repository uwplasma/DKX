"""End-to-end tests for the canonical RHSMode=1 driver and writer.

Vertical slice #3 referee (plan_final.md "File-Level Execution Queues"):

- ``sfincs_jax.run.run_profile`` moment tables must equal the legacy
  ``problems.transport_diagnostics.v3_rhsmode1_output_fields_vm_only`` values
  on the same solved state to 1e-10 (scaled) for the tiny PAS scheme-1
  axisymmetric fixture, the 2-species Fokker-Planck fixture, and a withEr
  PAS/DKES case (the ``tests/test_drift_kinetic.py`` inline deck);
- the ``sfincs_jax.writer.write_profile_output`` h5 file must contain every
  dataset the legacy ``outputs`` writer emits for RHSMode=1, equal to 1e-10
  (scaled), with the known-missing set enumerated explicitly: the legacy
  JAX-only ``linearSolver*`` metadata and — where the deck requests it — the
  export_f data family, which the canonical operator defers (unlike Phi1 and
  the tangential magnetic drifts, which are now canonical — see
  ``test_run_profile_magnetic_drifts_match_legacy`` below);
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

# Datasets present in the legacy RHSMode=1 output file but not written by the
# canonical writer.  ``linearSolver*`` is legacy-JAX solver metadata (not a
# Fortran dataset); the export_f data family is deferred by the canonical
# operator (plan_final.md "Explicit Deferred Items").  Phi1 and
# magnetic-drift families never appear here because the canonical operator
# refuses those decks at construction (tests/test_drift_kinetic.py).
_LEGACY_SOLVER_METADATA = frozenset({
    "linearSolverAcceptanceCriterion", "linearSolverAccepted", "linearSolverConverged",
    "linearSolverMethod", "linearSolverRequestedMethod", "linearSolverResidualNorm",
    "linearSolverResidualTarget", "linearSolverResidualTargetRatio",
    "linearSolverTrueResidualConverged",
})  # fmt: skip
_EXPORT_F_MISSING = frozenset({
    "N_export_f_theta", "N_export_f_x", "N_export_f_xi", "N_export_f_zeta",
    "delta_f", "full_f",
    "export_f_theta", "export_f_theta_option", "export_f_x", "export_f_x_option",
    "export_f_xi", "export_f_xi_option", "export_f_zeta", "export_f_zeta_option",
})  # fmt: skip
KNOWN_MISSING = {
    "pas_1species_PAS_noEr_tiny_scheme1": _LEGACY_SOLVER_METADATA,
    # The quick_2species deck sets export_full_f/export_delta_f = .true.
    "quick_2species_FPCollisions_noEr": _LEGACY_SOLVER_METADATA | _EXPORT_F_MISSING,
    WITH_ER_BASE: _LEGACY_SOLVER_METADATA,
}

EXPECTED_SOLVE_METHOD = {
    "pas_1species_PAS_noEr_tiny_scheme1": "block_tridiagonal",
    "quick_2species_FPCollisions_noEr": "gcrot",
    WITH_ER_BASE: "block_tridiagonal",
}

# The legacy write path solves iteratively at the deck's solverTolerance while
# tier 1 is direct: the moment-table and h5 referee needs a tight deck.
_WITH_ER_TOLERANCE_LINE = "  Nx = 3\n  solverTolerance = 1d-13\n"

# Wall-clock content differs run to run; compare shape/dtype only.
TIMING_KEYS = frozenset({"elapsed time (s)"})

# Scaled comparison tolerance |a-b| <= tol * max(1, max|a|).
DEFAULT_TOL = 1e-10

# The legacy moments helper leaves NTV as a zero placeholder (the writer fills
# it separately); the canonical table carries the real NTV, which the h5
# comparison referees instead.
_NTV_KEYS = frozenset({"NTV", "NTVBeforeSurfaceIntegral"})

_CACHE: dict[str, dict] = {}


@pytest.fixture(autouse=True)
def _clear_legacy_solver_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the legacy write path independent of earlier solver-policy tests."""
    for key in (
        "SFINCS_JAX_RHSMODE1_PRECONDITIONER",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP",
    ):
        monkeypatch.delenv(key, raising=False)


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
    """One canonical run + one legacy write per fixture, cached module-wide."""
    if base not in _CACHE:
        from sfincs_jax.io import write_sfincs_jax_output_h5
        from sfincs_jax.run import run_profile

        tmp_dir = tmp_path_factory.mktemp(f"rhsmode1_{base}")
        deck = _deck_path(base, tmp_dir)
        lines: list[str] = []
        run = run_profile(deck, out_path=tmp_dir / f"{base}.canonical.h5", emit=lines.append)
        legacy_path = tmp_dir / f"{base}.legacy.h5"
        write_sfincs_jax_output_h5(
            input_namelist=deck,
            output_path=legacy_path,
            compute_solution=True,
            overwrite=True,
            verbose=False,
        )
        _CACHE[base] = {
            "deck": deck,
            "run": run,
            "lines": tuple(lines),
            "canonical": _read_h5(run.output_path),
            "legacy": _read_h5(legacy_path),
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
# Moment-table equality vs the legacy diagnostics on the same solved state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_run_profile_moments_match_legacy_diagnostics(
    base: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    import jax.numpy as jnp

    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
    from sfincs_jax.problems import transport_diagnostics as td

    case = _case(base, tmp_path_factory)
    run = case["run"]
    assert run.solve_result.converged
    assert run.state_vector.shape == (run.operator.total_size,)

    op_old = full_system_operator_from_namelist(
        nml=read_sfincs_input(case["deck"]), identity_shift=0.0
    )
    old = td.v3_rhsmode1_output_fields_vm_only(op_old, x_full=jnp.asarray(run.state_vector))
    assert set(old).issubset(set(run.moments))
    for key in sorted(set(old) - _NTV_KEYS):
        _assert_scaled_close(old[key], run.moments[key], tol=DEFAULT_TOL, label=key)

    # Bootstrap-current closure as a physics anchor (not just old-vs-new).
    z_s = np.asarray(run.operator.z_s, dtype=np.float64)
    np.testing.assert_allclose(
        float(run.moments["FSABjHat"]),
        float(np.dot(z_s, np.asarray(run.moments["FSABFlow"]))),
        rtol=0.0,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# h5 field-by-field equality vs the legacy writer file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_profile_h5_matches_legacy_writer(
    base: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    case = _case(base, tmp_path_factory)
    legacy, canonical = case["legacy"], case["canonical"]

    missing = set(legacy) - set(canonical)
    extra = set(canonical) - set(legacy)
    assert missing == set(KNOWN_MISSING[base])
    assert extra == set()

    for key in sorted(set(legacy) & set(canonical)):
        a, b = legacy[key], canonical[key]
        if a.dtype.kind in "SOU" or b.dtype.kind in "SOU":
            assert np.array_equal(a, b), f"{key}: string dataset mismatch"
            continue
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        if key in TIMING_KEYS:
            assert a.shape == b.shape
            continue
        _assert_scaled_close(a, b, tol=DEFAULT_TOL, label=key)


# ---------------------------------------------------------------------------
# Console: the diagnostics.F90 species-results block
# ---------------------------------------------------------------------------


def _expected_species_block(run) -> tuple[str, ...]:
    """Re-render the species-results block from the run's moment table."""
    from sfincs_jax import console

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
    from sfincs_jax import console

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
    from sfincs_jax.run import run_profile

    with pytest.raises(NotImplementedError, match="RHSMode"):
        run_profile(REF / "monoenergetic_PAS_tiny_scheme1.input.namelist", emit=None)


# ---------------------------------------------------------------------------
# End-to-end parity for the tangential magnetic drifts (magneticDriftScheme=1)
# ---------------------------------------------------------------------------


def test_run_profile_magnetic_drifts_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical run_profile with tangential magnetic drifts matches the legacy pipeline.

    The ``magneticDriftScheme=1`` Boozer (geometryScheme 11) deck now routes
    through the canonical stack.  Because the drift couples L±2 the operator is
    not block-tridiagonal, so :func:`sfincs_jax.solve.solve` routes it to tier-2
    GCROT (falling back to the exact tier-3 direct solve on this tiny
    collisionless fixture).  Its fluxes must equal the retained legacy pipeline,
    whose magnetic-drift assembly is validated element-wise against Fortran v3 in
    ``tests/test_magnetic_drifts_parity.py``.
    """
    from sfincs_jax.cli import deck_requires_legacy_pipeline
    from sfincs_jax.io import write_sfincs_jax_output_h5
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.run import run_profile

    monkeypatch.setenv("SFINCS_JAX_EQUILIBRIA_DIRS", str(REF))
    deck = REF / "magdrift_1species_tiny.input.namelist"

    # The deck routes canonically now (no longer deferred to the legacy pipeline).
    assert deck_requires_legacy_pipeline(read_sfincs_input(deck)) is None

    run = run_profile(deck, out_path=tmp_path / "magdrift.canonical.h5", emit=None)
    legacy_path = tmp_path / "magdrift.legacy.h5"
    write_sfincs_jax_output_h5(
        input_namelist=deck,
        output_path=legacy_path,
        compute_solution=True,
        overwrite=True,
        verbose=False,
    )
    canonical = _read_h5(run.output_path)
    legacy = _read_h5(legacy_path)

    for key in ("particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "FSABFlow", "FSABjHat"):
        assert key in canonical and key in legacy, f"missing flux dataset {key!r}"
        _assert_scaled_close(canonical[key], legacy[key], tol=1e-9, label=key)


# ---------------------------------------------------------------------------
# Differentiability: jax.grad of FSABjHat through the pure solve + moments
# ---------------------------------------------------------------------------


def test_grad_fsabjhat_wrt_t_hat_matches_finite_differences() -> None:
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from sfincs_jax.drift_kinetic import KineticOperator
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.run import profile_moments_from_operator
    from sfincs_jax.solve import solve

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
