"""End-to-end RHSMode=1/2/3 runs on the canonical stack.

The single-call drivers for the SFINCS v3 linear modes (``sfincs_main.F90`` /
``solver.F90``): parse and validate the input (:mod:`sfincs_jax.inputs`),
build the drift-kinetic operator (:mod:`sfincs_jax.drift_kinetic`), solve with
the three-tier policy (:mod:`sfincs_jax.solve`), assemble the diagnostic
moments (:mod:`sfincs_jax.moments`), emit the Fortran-parity stdout blocks
(:mod:`sfincs_jax.console`), and write ``sfincsOutput``
(:mod:`sfincs_jax.writer`).  No legacy ``problems``/``operators``/``outputs``
modules are imported.

- :func:`run_transport_matrix` — RHSMode=2/3, the whichRHS transport-matrix
  loop (tier-1 structured direct for the PAS/DKES family).
- :func:`run_profile` — RHSMode=1, the single-RHS profile-gradient solve with
  the full per-species diagnostic table (tier 1 for PAS, tier-2 recycled
  Krylov for Fokker-Planck).  The validateInput.F90 RHSMode=3 monoenergetic
  forcing adapter is *not* applied here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np

from sfincs_jax import console
from sfincs_jax.constants import RadialCoordinates
from sfincs_jax.drift_kinetic import (
    KineticOperator,
    _geometry_and_radial,
    _n_periods_from_namelist,
    kinetic_operator_from_namelist,
)
from sfincs_jax.inputs import RawNamelist, SfincsInput, load_sfincs_input
from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry
from sfincs_jax.moments import (
    classical_fluxes,
    ntv_kernel,
    ntv_moments,
    rhsmode1_moments,
    transport_matrix_from_state_vectors,
    transport_moments_table,
)
from sfincs_jax.phase_space import Grids, make_grids
from sfincs_jax.solve import SolveResult, solve
from sfincs_jax.writer import (
    _effective_flux_functions,
    _geometry_extras,
    _u_hat,
    operator_containers,
    write_profile_output,
    write_run_solver_trace,
    write_transport_output,
)

__all__ = [
    "ProfileRun",
    "TransportRun",
    "profile_moments_from_operator",
    "run_profile",
    "run_transport_matrix",
]


@dataclass(frozen=True)
class TransportRun:
    """Result of one RHSMode=2/3 canonical-stack run.

    Attributes:
        input: the validated typed input.
        operator: the drift-kinetic operator the states were solved with.
        transport_matrix: RHSMode=3 monoenergetic 2x2 or RHSMode=2 Onsager 3x3
            matrix (``diagnostics.F90`` normalization).
        state_vectors: solved states, shape ``(n_rhs, total_size)``.
        solve_result: the :class:`sfincs_jax.solve.SolveResult` (method,
            residuals, timings) of the shared multi-RHS solve.
        moments: RHSMode=2/3 diagnostic table keyed by sfincsOutput.h5 names
            (:func:`sfincs_jax.moments.transport_moments_table` orders).
        output_path: written output file, or ``None``.
    """

    input: SfincsInput
    operator: KineticOperator
    transport_matrix: np.ndarray
    state_vectors: np.ndarray
    solve_result: SolveResult
    moments: Dict[str, np.ndarray]
    output_path: Path | None


@dataclass(frozen=True)
class ProfileRun:
    """Result of one RHSMode=1 canonical-stack run.

    Attributes:
        input: the validated typed input.
        operator: the drift-kinetic operator the state was solved with.
        state_vector: the solved state, shape ``(total_size,)``.
        solve_result: the :class:`sfincs_jax.solve.SolveResult` (method,
            residuals, timings) of the single-RHS solve.
        moments: the full RHSMode=1 per-species diagnostic table keyed by
            sfincsOutput.h5 names (:func:`sfincs_jax.moments.rhsmode1_moments`
            orders, species axis leading), including NTV and the classical
            fluxes (``classicalParticleFlux_psiHat``/``classicalHeatFlux_psiHat``).
        output_path: written output file, or ``None``.
    """

    input: SfincsInput
    operator: KineticOperator
    state_vector: np.ndarray
    solve_result: SolveResult
    moments: Dict[str, np.ndarray]
    output_path: Path | None


def _emit_lines(emit: Callable[[str], None] | None, lines: Iterable[str]) -> None:
    if emit is None:
        return
    for line in lines:
        emit(line)


def _raw_with_validated_overrides(inp: SfincsInput) -> RawNamelist:
    """Fold the validateInput.F90 RHSMode=3 hard overrides back into the raw deck.

    The operator builder reads the raw namelist directly; monoenergetic decks
    may rely on v3's forced settings (``Nx=1``, PAS collisions, DKES ExB, no
    xDot/xiDot Er terms), which :func:`sfincs_jax.inputs._validate` applies to
    the typed sections only.
    """
    raw = inp.raw
    if raw is None:
        raise ValueError("run_transport_matrix requires an input parsed from a namelist file.")
    if inp.general.rhs_mode != 3:
        return raw
    groups = {name: dict(values) for name, values in raw.groups.items()}
    phys = groups.setdefault("physicsparameters", {})
    phys["COLLISIONOPERATOR"] = inp.physics.collision_operator
    phys["USEDKESEXBDRIFT"] = inp.physics.use_dkes_exb_drift
    phys["INCLUDEXDOTTERM"] = inp.physics.include_x_dot_term
    phys["INCLUDEELECTRICFIELDTERMINXIDOT"] = inp.physics.include_electric_field_term_in_xi_dot
    phys["INCLUDEPHI1"] = inp.physics.include_phi1
    phys["INCLUDETEMPERATUREEQUILIBRATIONTERM"] = inp.physics.include_temperature_equilibration_term
    groups.setdefault("resolutionparameters", {})["NX"] = inp.resolution.n_x
    groups.setdefault("othernumericalparameters", {})["NXI_FOR_X_OPTION"] = inp.other.n_xi_for_x_option
    return replace(raw, groups=groups)


def _grids_from_input(inp: SfincsInput, raw: RawNamelist) -> Grids:
    """The same :func:`make_grids` call the operator builder performs."""
    res, other = inp.resolution, inp.other
    return make_grids(
        n_theta=res.n_theta,
        n_zeta=res.n_zeta,
        n_xi=res.n_xi,
        n_x=res.n_x,
        n_l=res.n_l,
        n_periods=_n_periods_from_namelist(nml=raw),
        theta_derivative_scheme=other.theta_derivative_scheme,
        zeta_derivative_scheme=other.zeta_derivative_scheme,
        magnetic_drift_derivative_scheme=other.magnetic_drift_derivative_scheme,
        x_grid_scheme=other.x_grid_scheme,
        x_grid_k=other.x_grid_k,
        n_xi_for_x_option=other.n_xi_for_x_option,
        monoenergetic=(inp.general.rhs_mode == 3),
    )


def _min_x_for_l(n_xi_for_x: np.ndarray, n_xi: int) -> list[int]:
    """First (1-based) speed index carrying each Legendre mode (createGrids.F90)."""
    n_x = int(n_xi_for_x.shape[0])
    out: list[int] = []
    for ell in range(int(n_xi)):
        idx = np.nonzero(n_xi_for_x > ell)[0]
        out.append(int(idx[0] + 1) if idx.size else n_x)
    return out


def _startup_lines(
    *, inp: SfincsInput, op: KineticOperator, grids: Grids, input_name: str
) -> list[str]:
    """Banner through 'The matrix is N x N elements.' (sfincs_main/createGrids)."""
    lines: list[str] = []
    lines += console.banner_lines(n_procs=1)
    lines += console.namelist_read_lines(input_name=input_name)
    if inp.general.rhs_mode == 3:
        lines.append(
            console.list_print(
                "Since RHSMode=3, ignoring the requested values of Zs, nHats, THats, "
                "nu_n, Er, and dPhiHatd*."
            )
        )
    lines += console.physics_parameter_lines(
        n_species=op.n_species,
        delta=inp.physics.delta,
        alpha=inp.physics.alpha,
        nu_n=inp.physics.nu_n,
        include_phi1=inp.physics.include_phi1,
    )
    n_xi_for_x = np.asarray(grids.n_xi_for_x)
    lines += console.grid_summary_lines(
        n_theta=op.n_theta,
        n_zeta=op.n_zeta,
        n_xi=op.n_xi,
        n_l=inp.resolution.n_l,
        n_x=op.n_x,
        solver_tolerance=inp.resolution.solver_tolerance,
        theta_derivative_scheme=inp.other.theta_derivative_scheme,
        zeta_derivative_scheme=inp.other.zeta_derivative_scheme,
        use_iterative_linear_solver=inp.other.use_iterative_linear_solver,
        n_xi_for_x_option=inp.other.n_xi_for_x_option,
        x=[float(v) for v in np.asarray(grids.x)],
        n_xi_for_x=[int(v) for v in n_xi_for_x],
        min_x_for_l=_min_x_for_l(n_xi_for_x, op.n_xi),
        matrix_size=op.total_size,
        x_grid_scheme=inp.other.x_grid_scheme,
        n_x_potentials_per_vth=inp.resolution.n_x_potentials_per_vth,
        x_max=inp.resolution.x_max,
    )
    return lines


def run_transport_matrix(
    namelist_path: str | Path,
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    out_path: str | Path | None = None,
    solver_trace_path: str | Path | None = None,
    emit: Callable[[str], None] | None = print,
) -> TransportRun:
    """Run a SFINCS v3 RHSMode=2/3 transport-matrix calculation end to end.

    Args:
        namelist_path: SFINCS ``input.namelist`` file (validated on load; the
            RHSMode=3 monoenergetic hard overrides of validateInput.F90 apply).
        solve_method: :func:`sfincs_jax.solve.solve` method (``"auto"`` picks
            the tier-1 structured direct path for the PAS/DKES family).
        tol: relative residual tolerance per whichRHS column.
        out_path: optional ``sfincsOutput`` file (``.h5``, ``.nc``, or ``.npz``)
            written by :func:`sfincs_jax.writer.write_transport_output`.
        solver_trace_path: optional JSON sidecar path; when set, a versioned
            :class:`sfincs_jax.solvers.diagnostics.SolverTrace` is written from
            the shared multi-RHS :class:`sfincs_jax.solve.SolveResult`.
        emit: per-line stdout sink for the Fortran-parity print blocks
            (``print`` reproduces the v3 console flow); ``None`` silences it.

    Returns:
        A :class:`TransportRun` with the transport matrix, states, solver
        stats, and moments tables.
    """
    namelist_path = Path(namelist_path)
    inp = load_sfincs_input(namelist_path)
    rhs_mode = inp.general.rhs_mode
    if rhs_mode not in (2, 3):
        raise NotImplementedError(
            "run_transport_matrix supports RHSMode 2 and 3; use run_profile for RHSMode=1."
        )

    raw = _raw_with_validated_overrides(inp)
    op = kinetic_operator_from_namelist(raw)
    grids = _grids_from_input(inp, raw)
    geom: FluxSurfaceGeometry
    radial: RadialCoordinates
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)

    _emit_lines(emit, _startup_lines(inp=inp, op=op, grids=grids, input_name=namelist_path.name))

    import jax.numpy as jnp  # noqa: PLC0415

    n_rhs = 3 if rhs_mode == 2 else 2
    rhs = jnp.stack([op.rhs(which_rhs) for which_rhs in range(1, n_rhs + 1)], axis=1)

    _emit_lines(emit, [console.entering_solver_line(), console.main_solve_begin_line()])
    t0 = time.perf_counter()
    result = solve(op, rhs, method=solve_method, tol=tol)
    solve_seconds = time.perf_counter() - t0
    _emit_lines(emit, [console.main_solve_done_line(seconds=solve_seconds)])
    if not result.converged:
        raise RuntimeError(
            f"transport-matrix solve did not converge (method={result.method}, "
            f"residuals={np.asarray(result.residual_norms)!r})"
        )

    state_vectors = np.asarray(result.x, dtype=np.float64).T  # (n_rhs, total_size)
    layout, vgrid, surface, species = operator_containers(op)
    transport_matrix = np.asarray(
        transport_matrix_from_state_vectors(
            layout, vgrid, surface, species, jnp.asarray(state_vectors),
            rhs_mode=rhs_mode, delta=op.delta, alpha=op.alpha,
            g_hat=float(geom.g_hat), i_hat=float(geom.i_hat),
            iota=float(geom.iota), b0_over_bbar=float(geom.b0_over_bbar),
        ),  # fmt: skip
        dtype=np.float64,
    )
    moments_table: Dict[str, np.ndarray] = {
        key: np.asarray(val, dtype=np.float64)
        for key, val in transport_moments_table(
            layout, vgrid, surface, species, jnp.asarray(state_vectors),
            rhs_mode=rhs_mode, delta=op.delta, alpha=op.alpha,
        ).items()  # fmt: skip
    }

    _emit_lines(emit, console.transport_matrix_lines(transport_matrix))

    output_path: Path | None = None
    if out_path is not None:
        elapsed = np.full((n_rhs,), solve_seconds / n_rhs, dtype=np.float64)
        output_path = write_transport_output(
            path=out_path, inp=inp, op=op, grids=grids, geom=geom, radial=radial,
            state_vectors=state_vectors, transport_matrix=transport_matrix,
            elapsed_times=elapsed,
        )  # fmt: skip

    if solver_trace_path is not None:
        rhs_norm = float(np.max(np.linalg.norm(np.asarray(rhs, dtype=np.float64), axis=0)))
        write_run_solver_trace(
            path=solver_trace_path, inp=inp, op=op, solve_result=result,
            rhs_norm=rhs_norm, solver_tol=float(tol), selected_path="transport_matrix",
            elapsed_seconds=solve_seconds, input_namelist=namelist_path, output_path=out_path,
            compute_solution=False, compute_transport_matrix=True,
        )  # fmt: skip

    _emit_lines(emit, [console.goodbye_line()])
    return TransportRun(
        input=inp,
        operator=op,
        transport_matrix=transport_matrix,
        state_vectors=state_vectors,
        solve_result=result,
        moments=moments_table,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# RHSMode=1: single-RHS profile-gradient run
# ---------------------------------------------------------------------------


def profile_moments_from_operator(
    op: KineticOperator,
    state_vector,
    *,
    ntv_kernel_tz=None,
) -> Dict[str, "np.ndarray"]:
    """Pure RHSMode=1 per-species moment table of a solved state (differentiable).

    Thin functional wrapper over :func:`sfincs_jax.moments.rhsmode1_moments`
    on the operator's own containers; every output is a jax array traced from
    ``op``'s species/geometry fields and ``state_vector``, so ``jax.grad`` of
    any entry (for example ``FSABjHat``) flows through both the moment
    integrals and — when the state comes from a ``differentiable=True``
    :func:`sfincs_jax.solve.solve` — the implicit solve.

    Args:
        op: the canonical operator (defines layout, grids, geometry, species).
        state_vector: solved state, shape ``(total_size,)``.
        ntv_kernel_tz: optional NTV geometric kernel ``(T, Z)``
            (:func:`sfincs_jax.moments.ntv_kernel`); when given, the ``NTV``
            and ``NTVBeforeSurfaceIntegral`` placeholders are replaced.

    Returns:
        The h5-named moment table with the species axis leading.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    layout, vgrid, surface, species = operator_containers(op)
    x_full = jnp.asarray(state_vector, dtype=jnp.float64)
    table = dict(
        rhsmode1_moments(
            layout, vgrid, surface, species, x_full,
            delta=op.delta, alpha=op.alpha, phi1_from_state=bool(op.include_phi1),
        )  # fmt: skip
    )
    if ntv_kernel_tz is not None:
        before, ntv = ntv_moments(layout, vgrid, surface, species, x_full, kernel=ntv_kernel_tz)
        table["NTVBeforeSurfaceIntegral"] = before
        table["NTV"] = ntv
    return table


def _ntv_kernel_for(inp: SfincsInput, op: KineticOperator, geom: FluxSurfaceGeometry):
    """NTV kernel for the run's geometry (zero for VMEC scheme 5, as in v3)."""
    import jax.numpy as jnp  # noqa: PLC0415

    if inp.geometry.geometry_scheme == 5:
        return jnp.zeros_like(jnp.asarray(op.b_hat))
    _, _, surface, _ = operator_containers(op)
    _b0_eff, g_eff, i_eff = _effective_flux_functions(op, geom)
    return ntv_kernel(
        surface, u_hat=jnp.asarray(_u_hat(geom)), g_hat=g_eff, i_hat=i_eff, iota=float(geom.iota)
    )


def _species_results_console_lines(
    *, op: KineticOperator, moments: Dict[str, np.ndarray]
) -> tuple[str, ...]:
    """The diagnostics.F90 per-species results table from the moment dict."""
    mach = np.asarray(moments["MachUsingFSAThermalSpeed"], dtype=np.float64)  # (S,T,Z)
    sources = np.asarray(moments["sources"], dtype=np.float64) if "sources" in moments else None
    entries: list[dict] = []
    for s in range(op.n_species):
        entry: dict = {
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
        if sources is not None:
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


def run_profile(
    namelist_path: str | Path,
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    out_path: str | Path | None = None,
    solver_trace_path: str | Path | None = None,
    emit: Callable[[str], None] | None = print,
) -> ProfileRun:
    """Run a SFINCS v3 RHSMode=1 profile-gradient calculation end to end.

    Args:
        namelist_path: SFINCS ``input.namelist`` file (validated on load).  The
            RHSMode=3 monoenergetic forcing adapter is *not* applied: RHSMode=1
            keeps the deck's collision operator, Er terms, and speed grid.
        solve_method: :func:`sfincs_jax.solve.solve` method (``"auto"`` picks
            tier 1 for the PAS/DKES family and tier-2 recycled Krylov for
            Fokker-Planck collisions).
        tol: relative residual tolerance for the single-RHS solve.
        out_path: optional ``sfincsOutput`` file (``.h5``, ``.nc``, or ``.npz``)
            written by :func:`sfincs_jax.writer.write_profile_output`.
        solver_trace_path: optional JSON sidecar path; when set, a versioned
            :class:`sfincs_jax.solvers.diagnostics.SolverTrace` is written from
            the single-RHS :class:`sfincs_jax.solve.SolveResult`.
        emit: per-line stdout sink for the Fortran-parity print blocks
            (``print`` reproduces the v3 console flow); ``None`` silences it.

    Returns:
        A :class:`ProfileRun` with the state, solver stats, and the full
        per-species moment table.
    """
    namelist_path = Path(namelist_path)
    inp = load_sfincs_input(namelist_path)
    if inp.general.rhs_mode != 1:
        raise NotImplementedError(
            "run_profile supports RHSMode=1; use run_transport_matrix for RHSMode 2/3."
        )
    raw = inp.raw
    if raw is None:
        raise ValueError("run_profile requires an input parsed from a namelist file.")

    op = kinetic_operator_from_namelist(raw)
    grids = _grids_from_input(inp, raw)
    geom: FluxSurfaceGeometry
    radial: RadialCoordinates
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)

    _emit_lines(emit, _startup_lines(inp=inp, op=op, grids=grids, input_name=namelist_path.name))

    _emit_lines(emit, [console.entering_solver_line(), console.main_solve_begin_line()])
    t0 = time.perf_counter()
    if op.include_phi1:
        # includePhi1 makes the DKE nonlinear (quasineutrality); the canonical
        # Newton solve in sfincs_jax.phi1 wraps solve() as its inner linear step.
        import jax.numpy as jnp  # noqa: PLC0415

        from sfincs_jax.phi1 import solve_phi1  # noqa: PLC0415

        phi1_result = solve_phi1(op, tol=tol, emit=emit)
        state_vector = np.asarray(phi1_result.x, dtype=np.float64).reshape((-1,))
        op = phi1_result.operator  # carries phi1_lin_state = solved state
        result = SolveResult(
            x=jnp.reshape(phi1_result.x, (-1, 1)),
            method="phi1_newton_krylov",
            iterations=phi1_result.inner_iterations_total,
            residual_norms=jnp.asarray([phi1_result.residual_norm], dtype=jnp.float64),
            converged=phi1_result.converged,
            recycle=None,
            timings=phi1_result.timings or {},
        )
    else:
        rhs = op.rhs()
        result = solve(op, rhs, method=solve_method, tol=tol)
        state_vector = np.asarray(result.x, dtype=np.float64).reshape((-1,))
    solve_seconds = time.perf_counter() - t0
    _emit_lines(emit, [console.main_solve_done_line(seconds=solve_seconds)])
    if not result.converged:
        raise RuntimeError(
            f"RHSMode=1 solve did not converge (method={result.method}, "
            f"residuals={np.asarray(result.residual_norms)!r})"
        )

    table = profile_moments_from_operator(
        op, state_vector, ntv_kernel_tz=_ntv_kernel_for(inp, op, geom)
    )
    moments: Dict[str, np.ndarray] = {
        key: np.asarray(val, dtype=np.float64) for key, val in table.items()
    }

    # Classical fluxes at the run's gradients (classicalTransport.F90).
    _, _, surface, species = operator_containers(op)
    gpsipsi, _diota = _geometry_extras(inp=inp, grids=grids, geom=geom, radial=radial)
    pf, hf = classical_fluxes(
        use_phi1=False, surface=surface, species=species,
        gpsipsi=gpsipsi, phi1_hat=np.zeros_like(gpsipsi),
        alpha=op.alpha, delta=op.delta, nu_n=inp.physics.nu_n,
        dn_hat_dpsi_hat=op.dn_hat_dpsi_hat, dt_hat_dpsi_hat=op.dt_hat_dpsi_hat,
    )  # fmt: skip
    moments["classicalParticleFlux_psiHat"] = np.asarray(pf, dtype=np.float64)
    moments["classicalHeatFlux_psiHat"] = np.asarray(hf, dtype=np.float64)

    _emit_lines(emit, _species_results_console_lines(op=op, moments=moments))

    output_path: Path | None = None
    if out_path is not None:
        output_path = write_profile_output(
            path=out_path, inp=inp, op=op, grids=grids, geom=geom, radial=radial,
            state_vector=state_vector, elapsed_seconds=solve_seconds,
        )  # fmt: skip

    if solver_trace_path is not None:
        try:
            rhs_norm = float(np.linalg.norm(np.asarray(op.rhs(), dtype=np.float64)))
        except Exception:  # noqa: BLE001
            rhs_norm = 0.0
        write_run_solver_trace(
            path=solver_trace_path, inp=inp, op=op, solve_result=result,
            rhs_norm=rhs_norm, solver_tol=float(tol), selected_path="rhsmode1_solution",
            elapsed_seconds=solve_seconds, input_namelist=namelist_path, output_path=out_path,
            compute_solution=True, compute_transport_matrix=False,
        )  # fmt: skip

    _emit_lines(emit, [console.goodbye_line()])
    return ProfileRun(
        input=inp,
        operator=op,
        state_vector=state_vector,
        solve_result=result,
        moments=moments,
        output_path=output_path,
    )
