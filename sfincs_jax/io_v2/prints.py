"""Fortran-parity stdout blocks for SFINCS v3 runs.

Replicates, byte-for-byte (timing values aside), the gfortran output of
``sfincs_main.F90`` / ``createGrids.F90`` / ``indices.F90`` / ``solver.F90``
/ ``diagnostics.F90``, as captured in the golden logs
``reference-data-v2/*/stdout.log``.

gfortran list-directed (``print *``) fields, verified against the golden logs:

- ``real(8)``: 26-character field. If ``0.1 <= |x| < 1e17`` the value is
  written in fixed form with 17 significant digits, right-justified in 21
  characters followed by 5 blanks; otherwise scientific with 16 decimals and
  a signed 3-digit exponent (``4.5694000000000004E-003``), right-justified
  in 26. Zero prints as ``0.0000000000000000`` (fixed form).
- ``integer(4)``: right-justified in a 12-character field.
- A single blank separates a numeric item from a *following* character item;
  no separator precedes a numeric item. Every record starts with one blank.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "fortran_real_field",
    "fortran_int_field",
    "list_print",
    "banner_lines",
    "namelist_read_lines",
    "physics_parameter_lines",
    "grid_summary_lines",
    "matrix_size_line",
    "entering_solver_line",
    "main_solve_begin_line",
    "main_solve_done_line",
    "species_results_lines",
    "goodbye_line",
    "NAMELIST_GROUP_ORDER",
]

NAMELIST_GROUP_ORDER: Tuple[str, ...] = (
    "general",
    "geometryParameters",
    "speciesParameters",
    "physicsParameters",
    "resolutionParameters",
    "otherNumericalParameters",
    "preconditionerOptions",
    "export_f",
)


def fortran_real_field(x: float) -> str:
    """Render a double the way gfortran list-directed output does (26 chars)."""
    x = float(x)
    if x == 0.0:
        return "0.0000000000000000".rjust(21) + " " * 5
    ax = abs(x)
    if 0.1 <= ax < 1.0e17:
        # Fixed form, 17 significant digits: k digits before the point.
        k = max(int(math.floor(math.log10(ax))) + 1, 0)
        body = f"{x:.{17 - k}f}"
        # Rounding can push e.g. 0.09999... to 0.1000... or 9.99...9 to 10.0...;
        # gfortran chooses the form from the rounded value, so re-derive k.
        digits_before_point = len(body.split(".", 1)[0].lstrip("-"))
        if digits_before_point != max(k, 1):
            body = f"{x:.{17 - digits_before_point}f}"
        return body.rjust(21) + " " * 5
    mantissa, exp = f"{x:.16E}".split("E")
    sign = exp[0] if exp[0] in "+-" else "+"
    return f"{mantissa}E{sign}{exp.lstrip('+-').zfill(3)}".rjust(26)


def fortran_int_field(value: int) -> str:
    """Render a default-kind integer as gfortran list-directed output (12 chars)."""
    return f"{int(value):12d}"


def list_print(*items: object) -> str:
    """Emulate gfortran ``print *, items...`` for str/int/float items."""
    out: List[str] = [" "]
    prev_numeric = False
    for item in items:
        if isinstance(item, str):
            if prev_numeric:
                out.append(" ")
            out.append(item)
            prev_numeric = False
        elif isinstance(item, bool) or isinstance(item, int):
            out.append(fortran_int_field(item))
            prev_numeric = True
        else:
            out.append(fortran_real_field(float(item)))
            prev_numeric = True
    return "".join(out)


# --------------------------------------------------------------------------
# Startup banner (sfincs_main.F90:35-47)
# --------------------------------------------------------------------------


def banner_lines(*, n_procs: int = 1, single_precision: bool = False) -> Tuple[str, ...]:
    lines = [
        list_print("*" * 76),
        list_print("SFINCS: Stellarator Fokker-Plank Iterative Neoclassical Conservative Solver"),
        list_print("Version 3"),
        list_print("Using single precision." if single_precision else "Using double precision."),
    ]
    if n_procs == 1:
        lines.append(list_print("Serial job (1 process) detected."))
    else:
        lines.append(f" Parallel job ({int(n_procs):4d} processes) detected.")
    return tuple(lines)


def namelist_read_lines(
    *, input_name: str = "input.namelist", groups: Iterable[str] = NAMELIST_GROUP_ORDER
) -> Tuple[str, ...]:
    """The per-namelist success lines from readInput.F90."""
    return tuple(
        list_print(f"Successfully read parameters from {group} namelist in {input_name}.")
        for group in groups
    )


# --------------------------------------------------------------------------
# Physics-parameter block (sfincs_main.F90:80-102)
# --------------------------------------------------------------------------


def physics_parameter_lines(
    *,
    n_species: int,
    delta: float,
    alpha: float,
    nu_n: float,
    include_phi1: bool = False,
    include_phi1_in_kinetic_equation: bool = True,
    quasineutrality_option: int = 1,
    read_external_phi1: bool = False,
) -> Tuple[str, ...]:
    lines = [
        list_print("---- Physics parameters: ----"),
        list_print("Number of particle species = ", n_species),
        list_print("Delta (rho* at reference parameters)          = ", delta),
        list_print("alpha (e Phi / T at reference parameters)     = ", alpha),
        list_print("nu_n (collisionality at reference parameters) = ", nu_n),
    ]
    if include_phi1 and not read_external_phi1:
        lines.append(list_print("Nonlinear run"))
        lines.append(
            list_print("with Phi1 included in the kinetic equation")
            if include_phi1_in_kinetic_equation
            else list_print("but with Phi1 excluded from the kinetic equation")
        )
        lines.append(
            list_print("Using full quasi-neutrality equation")
            if quasineutrality_option == 1
            else list_print("Using EUTERPE quasi-neutrality equation")
        )
    elif include_phi1 and include_phi1_in_kinetic_equation and read_external_phi1:
        lines.append(
            list_print(
                "Linear run but with Phi1 read from external file and included in the kinetic equation"
            )
        )
    else:
        lines.append(list_print("Linear run"))
    return tuple(lines)


# --------------------------------------------------------------------------
# Grid summary (createGrids.F90:60-99,171 and 1074-1119; indices.F90:361)
# --------------------------------------------------------------------------

_DERIVATIVE_DESCRIPTIONS = {
    0: "spectral collocation",
    1: "centered finite differences, 3-point stencil",
    2: "centered finite differences, 5-point stencil",
}


def grid_summary_lines(
    *,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    n_l: int,
    n_x: int,
    solver_tolerance: float,
    theta_derivative_scheme: int = 2,
    zeta_derivative_scheme: int = 2,
    use_iterative_linear_solver: bool = True,
    n_xi_for_x_option: int = 1,
    x: Sequence[float],
    n_xi_for_x: Sequence[int],
    min_x_for_l: Sequence[int],
    matrix_size: int,
    x_grid_scheme: int = 5,
    n_x_potentials_per_vth: float = 40.0,
    x_max: float = 5.0,
) -> Tuple[str, ...]:
    """The "---- Numerical parameters: ----" block through the matrix size."""
    lines = [
        list_print("---- Numerical parameters: ----"),
        list_print("Ntheta             = ", int(n_theta)),
        list_print("Nzeta              = ", int(n_zeta)),
        list_print("Nxi                = ", int(n_xi)),
        list_print("NL                 = ", int(n_l)),
        list_print("Nx                 = ", int(n_x)),
    ]
    if x_grid_scheme < 5:
        lines.append(list_print("NxPotentialsPerVth = ", float(n_x_potentials_per_vth)))
        lines.append(list_print("xMax               = ", float(x_max)))
    lines.append(list_print("solverTolerance    = ", float(solver_tolerance)))
    lines.append(list_print(f"Theta derivative: {_DERIVATIVE_DESCRIPTIONS[theta_derivative_scheme]}"))
    lines.append(list_print(f"Zeta derivative: {_DERIVATIVE_DESCRIPTIONS[zeta_derivative_scheme]}"))
    lines.append(
        list_print("For solving large linear systems, an iterative Krylov solver will be used.")
        if use_iterative_linear_solver
        else list_print("For solving large linear systems, a direct solver will be used.")
    )
    # createGrids.F90:171 formatted write "(a,i4,a,i3,a,i3,a,i3,a,i3,a)", trim'ed.
    lines.append(
        f" Processor {0:4d} owns theta indices {1:3d} to {int(n_theta):3d}"
        f" and zeta indices {1:3d} to {int(n_zeta):3d}"
    )
    lines.append(list_print("Nxi_for_x_option:", int(n_xi_for_x_option)))
    lines.append(list_print("x:", *(float(v) for v in x)))
    lines.append(list_print("Nxi for each x:", *(int(v) for v in n_xi_for_x)))
    lines.append(list_print("min_x_for_L:", *(int(v) for v in min_x_for_l)))
    lines.append(matrix_size_line(matrix_size=matrix_size))
    return tuple(lines)


def matrix_size_line(*, matrix_size: int) -> str:
    """indices.F90:361."""
    return list_print("The matrix is ", int(matrix_size), "x", int(matrix_size), " elements.")


# --------------------------------------------------------------------------
# Solver banners (solver.F90:91,442,473)
# --------------------------------------------------------------------------


def entering_solver_line() -> str:
    return list_print("Entering main solver loop.")


def main_solve_begin_line() -> str:
    return list_print("Beginning the main solve.  This could take a while ...")


def main_solve_done_line(*, seconds: float) -> str:
    return list_print("Done with the main solve.  Time to solve: ", float(seconds), " seconds.")


# --------------------------------------------------------------------------
# Per-species results table (diagnostics.F90:895-943)
# --------------------------------------------------------------------------

# (label as it appears in diagnostics.F90, result key) — order matters.
_SPECIES_RESULT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("   FSADensityPerturbation:  ", "FSADensityPerturbation"),
    ("   FSABFlow:                ", "FSABFlow"),
    ("   FSAPressurePerturbation: ", "FSAPressurePerturbation"),
    ("   NTV:                     ", "NTV"),
    ("   particleFlux_vm0_psiHat  ", "particleFlux_vm0_psiHat"),
    ("   particleFlux_vm_psiHat   ", "particleFlux_vm_psiHat"),
    ("   classicalParticleFlux    ", "classicalParticleFlux"),
    ("   classicalHeatFlux        ", "classicalHeatFlux"),
    ("   momentumFlux_vm0_psiHat  ", "momentumFlux_vm0_psiHat"),
    ("   momentumFlux_vm_psiHat   ", "momentumFlux_vm_psiHat"),
    ("   heatFlux_vm0_psiHat      ", "heatFlux_vm0_psiHat"),
    ("   heatFlux_vm_psiHat       ", "heatFlux_vm_psiHat"),
)

_PHI1_RESULT_FIELDS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "particleFlux_vm_psiHat": (
        ("   particleFlux_vE0_psiHat  ", "particleFlux_vE0_psiHat"),
        ("   particleFlux_vE_psiHat   ", "particleFlux_vE_psiHat"),
        ("   particleFlux_vd1_psiHat  ", "particleFlux_vd1_psiHat"),
        ("   particleFlux_vd_psiHat   ", "particleFlux_vd_psiHat"),
    ),
    "momentumFlux_vm_psiHat": (
        ("   momentumFlux_vE0_psiHat  ", "momentumFlux_vE0_psiHat"),
        ("   momentumFlux_vE_psiHat   ", "momentumFlux_vE_psiHat"),
        ("   momentumFlux_vd1_psiHat  ", "momentumFlux_vd1_psiHat"),
        ("   momentumFlux_vd_psiHat   ", "momentumFlux_vd_psiHat"),
    ),
    "heatFlux_vm_psiHat": (
        ("   heatFlux_vE0_psiHat      ", "heatFlux_vE0_psiHat"),
        ("   heatFlux_vE_psiHat       ", "heatFlux_vE_psiHat"),
        ("   heatFlux_vd1_psiHat      ", "heatFlux_vd1_psiHat"),
        ("   heatFlux_vd_psiHat       ", "heatFlux_vd_psiHat"),
        ("   heatFlux_withoutPhi1_psiHat ", "heatFlux_withoutPhi1_psiHat"),
    ),
}


def species_results_lines(
    *,
    species_results: Sequence[Mapping[str, float]],
    fsab_j_hat: float,
    include_phi1: bool = False,
    constraint_scheme: int = 1,
) -> Tuple[str, ...]:
    """The per-species table plus the bootstrap-current line.

    Each entry of ``species_results`` maps the Fortran diagnostic name to its
    value; ``MachMax``/``MachMin`` carry the Mach extrema and, for
    ``constraintScheme`` 1/3/4, ``particleSource``/``heatSource`` the sources.
    """
    n_species = len(species_results)
    lines: List[str] = []
    for i, result in enumerate(species_results, start=1):
        if n_species > 1:
            lines.append(list_print("Results for species ", i, ":"))
        for label, key in _SPECIES_RESULT_FIELDS[:2]:
            lines.append(list_print(label, float(result[key])))
        lines.append(
            list_print("   max and min Mach #:      ", float(result["MachMax"]), float(result["MachMin"]))
        )
        for label, key in _SPECIES_RESULT_FIELDS[2:]:
            lines.append(list_print(label, float(result[key])))
            if include_phi1 and key in _PHI1_RESULT_FIELDS:
                for p_label, p_key in _PHI1_RESULT_FIELDS[key]:
                    lines.append(list_print(p_label, float(result[p_key])))
        if constraint_scheme in (1, 3, 4):
            lines.append(list_print("   particle source          ", float(result["particleSource"])))
            lines.append(list_print("   heat source              ", float(result["heatSource"])))
        elif constraint_scheme == 2 and "sources" in result:
            lines.append(list_print("   sources: ", *(float(s) for s in result["sources"])))  # type: ignore[arg-type]
    lines.append(list_print("FSABjHat (bootstrap current): ", float(fsab_j_hat)))
    return tuple(lines)


def goodbye_line() -> str:
    """sfincs_main.F90:193."""
    return list_print("Goodbye!")
