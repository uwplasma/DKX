"""SFINCS v3 ``input.namelist`` reading, defaults, and validation.

Fortran counterparts (``/Users/rogerio/local/sfincs/fortran/version3``):

- ``readInput.F90``   — namelist group membership and species-array handling.
- ``globalVariables.F90`` (``gV`` in comments below) — the default value of
  every input parameter; each typed field cites the line declaring it.
- ``validateInput.F90``  — option-range checks and the RHSMode=3
  (monoenergetic transport matrix) hard overrides.

The low-level parser is the proven hand-rolled one from
``dkx/namelist.py`` (no f90nml dependency: f90nml is not a project
dependency and the hand-rolled parser already round-trips every upstream
example deck). It is consolidated here so the legacy module can be deleted
at the end of the Phase-3 refactor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

Number = Union[int, float]
Scalar = Union[str, bool, Number]
Value = Union[Scalar, List[Scalar]]

# --------------------------------------------------------------------------
# Low-level Fortran-namelist parser (proven logic from dkx/namelist.py)
# --------------------------------------------------------------------------


def _strip_fortran_comments(line: str) -> str:
    """Remove ``!`` comments, respecting quoted strings."""
    out: List[str] = []
    quote_char: str | None = None
    for ch in line:
        if ch in {"'", '"'}:
            if quote_char is None:
                quote_char = ch
            elif quote_char == ch:
                quote_char = None
            out.append(ch)
        elif ch == "!" and quote_char is None:
            break
        else:
            out.append(ch)
    return "".join(out)


_GROUP_START_RE = re.compile(r"^\s*&\s*(?P<name>[A-Za-z_]\w*)\s*$", flags=re.IGNORECASE)
_GROUP_END_RE = re.compile(r"^\s*/\s*$")
_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)
_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}


def _tokenize_value_chunk(chunk: str) -> List[str]:
    """Split a value chunk into tokens, keeping quoted strings intact."""
    tokens: List[str] = []
    buf: List[str] = []
    quote_char: str | None = None
    for ch in chunk.strip():
        if ch in {"'", '"'}:
            if quote_char is None:
                quote_char = ch
            elif quote_char == ch:
                quote_char = None
            buf.append(ch)
        elif quote_char is None and ch in {",", "\n", "\t", " ", "\r"}:
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
                buf = []
        else:
            buf.append(ch)
    if buf:
        tok = "".join(buf).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _parse_scalar(tok: str) -> Scalar:
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in {"'", '"'}:
        return tok[1:-1]
    up = tok.upper()
    if up in _BOOL_TRUE:
        return True
    if up in _BOOL_FALSE:
        return False
    if re.fullmatch(r"[+-]?\d+", tok):
        return int(tok)
    try:
        return float(tok.replace("D", "E").replace("d", "E"))
    except ValueError:
        return tok


def _parse_key(key: str) -> Tuple[str, Tuple[int, ...] | None]:
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    idx = tuple(int(x.strip()) for x in rest.rstrip(")").split(",") if x.strip() != "")
    return base.upper(), idx


@dataclass(frozen=True)
class RawNamelist:
    """Parsed namelist groups: ``groups[group][KEY] -> value`` (keys uppercased)."""

    groups: Dict[str, Dict[str, Value]]
    indexed: Dict[str, Dict[str, Dict[Tuple[int, ...], Scalar]]]
    source_path: Path | None = None
    source_text: str | None = None

    def group(self, name: str) -> Dict[str, Value]:
        return self.groups.get(name.lower(), {})


def parse_sfincs_input_text(text: str, *, source_path: str | Path | None = None) -> RawNamelist:
    """Parse SFINCS namelist text into groups (case-insensitive group names)."""
    source = None if source_path is None else Path(source_path)
    lines = [_strip_fortran_comments(ln) for ln in text.splitlines()]

    groups: Dict[str, List[str]] = {}
    current_name: str | None = None
    current_lines: List[str] = []
    for raw in lines:
        m = _GROUP_START_RE.match(raw)
        if m:
            if current_name is not None:
                raise ValueError(f"Nested namelist group found while in &{current_name}")
            current_name = m.group("name").lower()
            current_lines = []
            continue
        if current_name is None:
            continue
        if _GROUP_END_RE.match(raw):
            groups[current_name] = current_lines
            current_name = None
            current_lines = []
            continue
        current_lines.append(raw)
    if current_name is not None:
        raise ValueError(f"Namelist &{current_name} not terminated by '/'")

    parsed_groups: Dict[str, Dict[str, Value]] = {}
    parsed_indexed: Dict[str, Dict[str, Dict[Tuple[int, ...], Scalar]]] = {}
    for gname, glines in groups.items():
        cleaned = "\n".join(glines)
        scalars: Dict[str, Value] = {}
        indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {}
        matches = list(_ASSIGN_RE.finditer(cleaned))
        for i, m in enumerate(matches):
            key_base, idx = _parse_key(m.group("key"))
            val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            chunk = re.sub(r",\s*$", "", cleaned[m.end() : val_end].strip())
            toks = _tokenize_value_chunk(chunk)
            if not toks:
                continue
            parsed = [_parse_scalar(t) for t in toks]
            value: Value = parsed[0] if len(parsed) == 1 else parsed
            if idx is None:
                scalars[key_base] = value
            else:
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError(f"Indexed assignment {m.group('key')} has multiple values")
                    value = value[0]
                indexed.setdefault(key_base, {})[idx] = value
        parsed_groups[gname] = scalars
        parsed_indexed[gname] = indexed
    return RawNamelist(groups=parsed_groups, indexed=parsed_indexed, source_path=source, source_text=text)


def read_sfincs_input(path: str | Path) -> RawNamelist:
    """Parse a SFINCS ``input.namelist`` file into groups."""
    source_path = Path(path).resolve()
    return parse_sfincs_input_text(source_path.read_text(), source_path=source_path)


# --------------------------------------------------------------------------
# Typed sections. Each field is declared as ``_f(default, "FortranName")``;
# the default's Python type selects the converter (bool/int/float/str/tuple).
# Defaults verified against globalVariables.F90 ("gV").
# --------------------------------------------------------------------------


def _f(default: Any, fortran_name: str) -> Any:
    return field(default=default, metadata={"nml": fortran_name})


def _convert(value: Any, default: Any) -> Any:
    if isinstance(value, list) and not isinstance(default, tuple):
        value = value[0]
    if isinstance(default, tuple):
        seq = value if isinstance(value, (list, tuple)) else [value]
        return tuple(float(v) for v in seq)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


@dataclass(frozen=True)
class GeneralParams:
    """&general — readInput.F90:23."""

    rhs_mode: int = _f(1, "RHSMode")  # gV:33 (1 solve, 2/3 transport matrix, 4/5 adjoint)
    save_matlab_output: bool = _f(False, "saveMatlabOutput")  # gV:27
    save_matrices_and_vectors_in_binary: bool = _f(False, "saveMatricesAndVectorsInBinary")  # gV:27
    output_filename: str = _f("sfincsOutput.h5", "outputFilename")  # gV:30
    solve_system: bool = _f(True, "solveSystem")  # gV:32
    ambipolar_solve: bool = _f(False, "ambipolarSolve")  # gV:37
    ambipolar_solve_option: int = _f(2, "ambipolarSolveOption")  # gV:38
    n_er_ambipolar_solve: int = _f(20, "NEr_ambipolarSolve")  # gV:39
    er_search_tolerance_dx: float = _f(1.0e-8, "Er_search_tolerance_dx")  # gV:40
    er_search_tolerance_f: float = _f(1.0e-10, "Er_search_tolerance_f")  # gV:41
    er_min: float = _f(-100.0, "Er_min")  # gV:42
    er_max: float = _f(100.0, "Er_max")  # gV:43


@dataclass(frozen=True)
class GeometryParams:
    """&geometryParameters — readInput.F90:31."""

    geometry_scheme: int = _f(1, "geometryScheme")  # gV:53
    g_hat: float = _f(3.7481, "GHat")  # gV:54
    i_hat: float = _f(0.0, "IHat")  # gV:54
    iota: float = _f(0.4542, "iota")  # gV:54
    b0_over_bbar: float = _f(1.0, "B0OverBBar")  # gV:54
    psi_a_hat: float = _f(0.15596, "psiAHat")  # gV:54
    epsilon_t: float = _f(-0.07053, "epsilon_t")  # gV:56
    epsilon_h: float = _f(0.05067, "epsilon_h")  # gV:56
    epsilon_antisymm: float = _f(0.0, "epsilon_antisymm")  # gV:56
    n_periods: int = _f(0, "NPeriods")  # gV:57
    helicity_l: int = _f(2, "helicity_l")  # gV:57
    helicity_n: int = _f(10, "helicity_n")  # gV:57
    helicity_antisymm_l: int = _f(1, "helicity_antisymm_l")  # gV:57
    helicity_antisymm_n: int = _f(0, "helicity_antisymm_n")  # gV:57
    equilibrium_file: str = _f("", "equilibriumFile")  # gV:58
    min_bmn_to_load: float = _f(0.0, "min_Bmn_to_load")  # gV:60
    a_hat: float = _f(0.5585, "aHat")  # gV:60
    psi_hat_wish: float = _f(-1.0, "psiHat_wish")  # gV:61
    psi_n_wish: float = _f(0.25, "psiN_wish")  # gV:61
    r_hat_wish: float = _f(-1.0, "rHat_wish")  # gV:61
    r_n_wish: float = _f(0.5, "rN_wish")  # gV:61
    input_radial_coordinate: int = _f(3, "inputRadialCoordinate")  # gV:63 (3 = rN)
    input_radial_coordinate_for_gradients: int = _f(4, "inputRadialCoordinateForGradients")  # gV:64
    vmec_radial_option: int = _f(1, "VMECRadialOption")  # gV:66
    vmec_nyquist_option: int = _f(1, "VMEC_Nyquist_option")  # gV:67
    ripple_scale: float = _f(1.0, "rippleScale")  # gV:68


@dataclass(frozen=True)
class SpeciesParams:
    """&speciesParameters — readInput.F90:40.

    The Fortran arrays default to a ``speciesNotInitialized`` sentinel
    (readInput.F90:103-114); the number of species is the count of
    contiguously-initialized ``Zs`` entries. Empty tuples model that here.
    """

    z_s: Tuple[float, ...] = _f((), "Zs")
    m_hats: Tuple[float, ...] = _f((), "mHats")
    n_hats: Tuple[float, ...] = _f((), "nHats")
    t_hats: Tuple[float, ...] = _f((), "THats")
    d_n_hat_d_psi_hats: Tuple[float, ...] = _f((), "dNHatdpsiHats")
    d_t_hat_d_psi_hats: Tuple[float, ...] = _f((), "dTHatdpsiHats")
    d_n_hat_d_psi_ns: Tuple[float, ...] = _f((), "dNHatdpsiNs")
    d_t_hat_d_psi_ns: Tuple[float, ...] = _f((), "dTHatdpsiNs")
    d_n_hat_d_r_hats: Tuple[float, ...] = _f((), "dNHatdrHats")
    d_t_hat_d_r_hats: Tuple[float, ...] = _f((), "dTHatdrHats")
    d_n_hat_d_r_ns: Tuple[float, ...] = _f((), "dNHatdrNs")
    d_t_hat_d_r_ns: Tuple[float, ...] = _f((), "dTHatdrNs")
    with_adiabatic: bool = _f(False, "withAdiabatic")  # gV:100
    adiabatic_z: float = _f(-1.0, "adiabaticZ")  # gV:99
    adiabatic_m_hat: float = _f(5.446170214e-4, "adiabaticMHat")  # gV:99
    adiabatic_n_hat: float = _f(1.0, "adiabaticNHat")  # gV:99
    adiabatic_t_hat: float = _f(1.0, "adiabaticTHat")  # gV:99
    with_nbi_spec: bool = _f(False, "withNBIspec")  # gV:104
    nbi_spec_z: float = _f(1.0, "NBIspecZ")  # gV:103
    nbi_spec_n_hat: float = _f(0.0, "NBIspecNHat")  # gV:103

    @property
    def n_species(self) -> int:
        return len(self.z_s)


@dataclass(frozen=True)
class PhysicsParams:
    """&physicsParameters — readInput.F90:57."""

    delta: float = _f(4.5694e-3, "Delta")  # rho* at reference parameters, gV:133
    alpha: float = _f(1.0, "alpha")  # e Phi / T at reference parameters, gV:134
    nu_n: float = _f(8.330e-3, "nu_n")  # collisionality at reference parameters, gV:135
    e_parallel_hat: float = _f(0.0, "EParallelHat")  # gV:137
    d_phi_hat_d_psi_hat: float = _f(0.0, "dPhiHatdpsiHat")  # gV:138
    d_phi_hat_d_psi_n: float = _f(0.0, "dPhiHatdpsiN")  # gV:138
    d_phi_hat_d_r_hat: float = _f(0.0, "dPhiHatdrHat")  # gV:138
    d_phi_hat_d_r_n: float = _f(0.0, "dPhiHatdrN")  # gV:138
    er: float = _f(0.0, "Er")  # gV:138
    collision_operator: int = _f(0, "collisionOperator")  # gV:140 (0 = full FP, 1 = PAS, 3 = improved Sugama)
    constraint_scheme: int = _f(-1, "constraintScheme")  # gV:214 (-1 = auto: 1 for FP, 2 for PAS)
    include_x_dot_term: bool = _f(True, "includeXDotTerm")  # gV:144
    include_electric_field_term_in_xi_dot: bool = _f(True, "includeElectricFieldTermInXiDot")  # gV:145
    use_dkes_exb_drift: bool = _f(False, "useDKESExBDrift")  # gV:146
    include_f_div_ve_term: bool = _f(False, "include_fDivVE_term")  # gV:147
    include_temperature_equilibration_term: bool = _f(False, "includeTemperatureEquilibrationTerm")  # gV:148
    include_phi1: bool = _f(False, "includePhi1")  # gV:149
    include_phi1_in_collision_operator: bool = _f(False, "includePhi1InCollisionOperator")  # gV:150
    include_phi1_in_kinetic_equation: bool = _f(True, "includePhi1InKineticEquation")  # gV:152
    read_external_phi1: bool = _f(False, "readExternalPhi1")  # gV:153
    nu_prime: float = _f(1.0, "nuPrime")  # gV:155
    e_star: float = _f(0.0, "EStar")  # gV:155
    magnetic_drift_scheme: int = _f(0, "magneticDriftScheme")  # gV:157
    quasineutrality_option: int = _f(1, "quasineutralityOption")  # gV:159
    krook: float = _f(0.0, "Krook")  # gV:161


@dataclass(frozen=True)
class ResolutionParams:
    """&resolutionParameters — readInput.F90:73."""

    force_odd_ntheta_and_nzeta: bool = _f(True, "forceOddNthetaAndNzeta")  # gV:195
    n_theta: int = _f(15, "Ntheta")  # gV:186
    n_zeta: int = _f(15, "Nzeta")  # gV:187
    n_xi: int = _f(16, "Nxi")  # gV:188
    n_l: int = _f(4, "NL")  # collision-coupling depth, gV:189
    n_x: int = _f(5, "Nx")  # gV:190
    x_max: float = _f(5.0, "xMax")  # gV:192 (only used for xGridScheme<5)
    solver_tolerance: float = _f(1.0e-6, "solverTolerance")  # gV:193
    n_x_potentials_per_vth: float = _f(40.0, "NxPotentialsPerVth")  # gV:191


@dataclass(frozen=True)
class OtherNumericalParams:
    """&otherNumericalParameters — readInput.F90:83."""

    use_iterative_linear_solver: bool = _f(True, "useIterativeLinearSolver")  # gV:201
    theta_derivative_scheme: int = _f(2, "thetaDerivativeScheme")  # gV:171 (0 spectral, 1/2 FD)
    zeta_derivative_scheme: int = _f(2, "zetaDerivativeScheme")  # gV:172
    exb_derivative_scheme_theta: int = _f(0, "ExBDerivativeSchemeTheta")  # gV:177
    exb_derivative_scheme_zeta: int = _f(0, "ExBDerivativeSchemeZeta")  # gV:178
    magnetic_drift_derivative_scheme: int = _f(3, "magneticDriftDerivativeScheme")  # gV:179
    x_dot_derivative_scheme: int = _f(0, "xDotDerivativeScheme")  # gV:180
    x_grid_scheme: int = _f(5, "xGridScheme")  # gV:183 (5 = Landreman-Ernst grid)
    x_potentials_grid_scheme: int = _f(2, "xPotentialsGridScheme")  # gV:182
    x_grid_k: float = _f(0.0, "xGrid_k")  # xGrid.F90:70
    which_parallel_solver_to_factor_preconditioner: int = _f(
        1, "whichParallelSolverToFactorPreconditioner"
    )  # gV:203
    petsc_preallocation_strategy: int = _f(1, "PETSCPreallocationStrategy")  # gV:216
    n_xi_for_x_option: int = _f(1, "Nxi_for_x_option")  # gV:217


@dataclass(frozen=True)
class PreconditionerOptions:
    """&preconditionerOptions — readInput.F90:90. Defaults gV:208-212."""

    preconditioner_x: int = _f(1, "preconditioner_x")
    preconditioner_x_min_l: int = _f(0, "preconditioner_x_min_L")
    preconditioner_zeta: int = _f(0, "preconditioner_zeta")
    preconditioner_theta: int = _f(0, "preconditioner_theta")
    preconditioner_xi: int = _f(1, "preconditioner_xi")
    preconditioner_species: int = _f(1, "preconditioner_species")
    preconditioner_theta_min_l: int = _f(0, "preconditioner_theta_min_L")
    preconditioner_zeta_min_l: int = _f(0, "preconditioner_zeta_min_L")
    reuse_preconditioner: bool = _f(True, "reusePreconditioner")
    preconditioner_magnetic_drifts_max_l: int = _f(2, "preconditioner_magnetic_drifts_max_L")


_SECTION_CLASSES: Dict[str, type] = {
    "general": GeneralParams,
    "geometryparameters": GeometryParams,
    "speciesparameters": SpeciesParams,
    "physicsparameters": PhysicsParams,
    "resolutionparameters": ResolutionParams,
    "othernumericalparameters": OtherNumericalParams,
    "preconditioneroptions": PreconditionerOptions,
}

# (SfincsInput attribute, Fortran namelist group name) in canonical deck order.
_SECTION_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("general", "general"),
    ("geometry", "geometryParameters"),
    ("species", "speciesParameters"),
    ("physics", "physicsParameters"),
    ("resolution", "resolutionParameters"),
    ("other", "otherNumericalParameters"),
    ("preconditioner", "preconditionerOptions"),
)


# --------------------------------------------------------------------------
# Serializer: typed input -> Fortran-namelist text (the parser's inverse)
# --------------------------------------------------------------------------


def _format_scalar(value: Scalar) -> str:
    """Render one value in Fortran-namelist syntax the parser round-trips."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)  # shortest text that round-trips the float64 exactly
    text = str(value)
    quote = "'" if '"' in text else '"'
    return f"{quote}{text}{quote}"


def _format_value(value: Value | Tuple[float, ...]) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_format_scalar(v) for v in value)
    return _format_scalar(value)


# Written even by compact decks: the geometry-family dispatch key has no
# usable fallback in the raw-namelist readers (they fail loudly on a missing
# geometryScheme rather than silently assuming the Fortran default).
_COMPACT_ALWAYS_WRITE = frozenset({"GEOMETRYSCHEME"})


def _section_lines(
    section: Any,
    *,
    include_defaults: bool,
    raw_group: Mapping[str, Value],
    raw_indexed: Mapping[str, Mapping[Tuple[int, ...], Scalar]],
) -> List[str]:
    """One namelist group body: typed fields, untyped raw keys, rank-2 arrays."""
    lines: List[str] = []
    claimed: set[str] = set()
    for fld in fields(type(section)):
        name = fld.metadata.get("nml")
        if name is None:
            continue
        claimed.add(name.upper())
        value = getattr(section, fld.name)
        if isinstance(value, tuple) and not value:
            continue  # uninitialized species-style array (Fortran sentinel)
        if (
            not include_defaults
            and value == fld.default
            and name.upper() not in _COMPACT_ALWAYS_WRITE
        ):
            continue
        lines.append(f"  {name} = {_format_value(value)}")
    # Untyped keys the parser retained in ``raw`` (e.g. the legacy Boozer
    # equilibrium aliases or externalPhi1Filename) survive the round trip.
    for key, value in raw_group.items():
        if key.upper() in claimed:
            continue
        lines.append(f"  {key} = {_format_value(value)}")
    # Rank-2+ indexed assignments (the boozer_bmnc(m,n)/boozer_bmns(m,n)
    # spectra of geometryScheme=13) are consumed from ``raw.indexed`` by the
    # geometry builder and re-emitted verbatim here.
    for base, entries in raw_indexed.items():
        for idx, val in sorted(entries.items()):
            if len(idx) < 2:
                continue  # rank-1 assignments were folded into typed vectors
            idx_txt = ",".join(str(i) for i in idx)
            lines.append(f"  {base}({idx_txt}) = {_format_scalar(val)}")
    return lines


def _build_flat_fields() -> Dict[str, Tuple[str, str, Any]]:
    """UPPERCASE Fortran name -> (section attribute, field name, default).

    The flat constructor namespace of :meth:`SfincsInput.from_params`; built
    once and asserted collision-free (every Fortran input name is unique
    across the namelist groups, as in readInput.F90).
    """
    mapping: Dict[str, Tuple[str, str, Any]] = {}
    for attr, display in _SECTION_GROUPS:
        for fld in fields(_SECTION_CLASSES[display.lower()]):
            name = fld.metadata.get("nml")
            if name is None:
                continue
            key = name.upper()
            if key in mapping:  # pragma: no cover - guarded by construction
                raise AssertionError(f"Duplicate Fortran input name across sections: {name}")
            mapping[key] = (attr, fld.name, fld.default)
    return mapping


_FLAT_FIELDS: Dict[str, Tuple[str, str, Any]] = _build_flat_fields()


@dataclass(frozen=True)
class SfincsInput:
    """Typed SFINCS input, grouped by namelist section.

    ``export_f`` and anything not (yet) typed remain reachable through
    ``raw`` (the full parsed namelist) — the ``.raw`` fallback contract.

    Build one from a file with :func:`load_sfincs_input`, or programmatically
    with :meth:`from_params` using the flat Fortran parameter names
    (``SfincsInput.from_params(Ntheta=17, geometryScheme=11, ...)``); the
    nested section dataclasses use snake_case Python names (``n_theta``)
    whose Fortran spelling is recorded in each field's ``nml`` metadata.
    Serialize back to Fortran-namelist text with :meth:`to_namelist` or
    :meth:`write`.
    """

    general: GeneralParams = field(default_factory=GeneralParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    species: SpeciesParams = field(default_factory=SpeciesParams)
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    resolution: ResolutionParams = field(default_factory=ResolutionParams)
    other: OtherNumericalParams = field(default_factory=OtherNumericalParams)
    preconditioner: PreconditionerOptions = field(default_factory=PreconditionerOptions)
    export_f: Mapping[str, Value] = field(default_factory=dict)
    raw: RawNamelist | None = None
    warnings: Tuple[str, ...] = ()

    @classmethod
    def from_params(cls, *, validate: bool = True, **params: Any) -> "SfincsInput":
        """Build a typed input from flat Fortran-named parameters.

        Parameter names are the Fortran ``input.namelist`` names
        (``Ntheta``, ``geometryScheme``, ``equilibriumFile``, ``Zs``, ...),
        matched case-insensitively and routed to the owning namelist section
        automatically.  Species arrays accept lists or tuples.  Unknown names
        raise ``ValueError`` (the ``.raw`` fallback of parsed decks has no
        programmatic equivalent).

        Args:
            validate: run the validateInput.F90-equivalent checks and the
                RHSMode=3 hard overrides (default), as :func:`load_sfincs_input`
                does for files.
            **params: flat Fortran-named values, e.g.
                ``SfincsInput.from_params(geometryScheme=1, Ntheta=15, Zs=[1.0])``.

        Returns:
            The typed (and, by default, validated) :class:`SfincsInput` with
            ``raw=None``; the run drivers synthesize the namelist via
            :meth:`to_namelist` when needed.
        """
        section_kwargs: Dict[str, Dict[str, Any]] = {attr: {} for attr, _ in _SECTION_GROUPS}
        for name, value in params.items():
            entry = _FLAT_FIELDS.get(name.upper())
            if entry is None:
                raise ValueError(
                    f"Unknown SFINCS input parameter {name!r}. Parameter names are the "
                    "Fortran input.namelist names (case-insensitive), e.g. Ntheta, "
                    "geometryScheme, equilibriumFile, Zs, nHats."
                )
            attr, field_name, default = entry
            section_kwargs[attr][field_name] = _convert(value, default)
        inp = cls(
            **{
                attr: _SECTION_CLASSES[display.lower()](**section_kwargs[attr])
                for attr, display in _SECTION_GROUPS
            }
        )
        return _validate(inp) if validate else inp

    def to_namelist(self, *, include_defaults: bool = False) -> str:
        """Serialize to SFINCS-style Fortran ``input.namelist`` text.

        The output is the parser's inverse: ``load_sfincs_input`` of the
        written text reproduces every typed section field, the ``export_f``
        group, untyped keys retained in ``raw`` (legacy aliases such as
        ``JGboozer_file``), and the rank-2 ``boozer_bmnc(m,n)`` spectra of
        geometryScheme=13 decks.  The formatting (``&group`` ... ``/``,
        ``.true.``/``.false.``, quoted strings, space-separated arrays) is the
        Fortran-namelist subset both this parser and the SFINCS v3 Fortran
        reader accept.

        Args:
            include_defaults: write every typed field; the default writes a
                compact deck with only the fields that differ from the
                Fortran defaults (globalVariables.F90).
        """
        raw_groups: Dict[str, Dict[str, Value]] = (
            {name: dict(vals) for name, vals in self.raw.groups.items()} if self.raw is not None else {}
        )
        raw_indexed: Dict[str, Dict[str, Dict[Tuple[int, ...], Scalar]]] = (
            {name: dict(vals) for name, vals in self.raw.indexed.items()} if self.raw is not None else {}
        )
        lines: List[str] = ["! SFINCS-style input.namelist written by dkx (SfincsInput.to_namelist)."]
        known_groups = {"export_f"}
        for attr, display in _SECTION_GROUPS:
            group_key = display.lower()
            known_groups.add(group_key)
            lines.append(f"&{display}")
            lines.extend(
                _section_lines(
                    getattr(self, attr),
                    include_defaults=include_defaults,
                    raw_group=raw_groups.get(group_key, {}),
                    raw_indexed=raw_indexed.get(group_key, {}),
                )
            )
            lines.append("/")
        lines.append("&export_f")
        lines.extend(f"  {key} = {_format_value(value)}" for key, value in self.export_f.items())
        lines.append("/")
        # Unknown groups the parser retained in ``raw`` survive verbatim.
        for group_name, group_values in raw_groups.items():
            if group_name in known_groups:
                continue
            lines.append(f"&{group_name}")
            lines.extend(f"  {key} = {_format_value(value)}" for key, value in group_values.items())
            for base, entries in raw_indexed.get(group_name, {}).items():
                for idx, val in sorted(entries.items()):
                    idx_txt = ",".join(str(i) for i in idx)
                    lines.append(f"  {base}({idx_txt}) = {_format_scalar(val)}")
            lines.append("/")
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path, *, include_defaults: bool = False) -> Path:
        """Write :meth:`to_namelist` text to ``path`` and return the ``Path``."""
        out = Path(path)
        out.write_text(self.to_namelist(include_defaults=include_defaults), encoding="utf-8")
        return out


def _merge_indexed(
    group_values: Dict[str, Value], indexed: Mapping[str, Mapping[Tuple[int, ...], Scalar]]
) -> Dict[str, Value]:
    """Fold Fortran indexed assignments (``mHats(2) = 6.0``) into vectors.

    Rank-2 indexed assignments (the ``boozer_bmnc(m,n)`` / ``boozer_bmns(m,n)``
    Boozer |B| spectrum used by geometryScheme=13) are not modeled as a typed
    vector here; the geometry builder reads them directly from
    :attr:`RawNamelist.indexed`.  Such bases are left out of the merged typed
    section rather than raising.
    """
    merged = dict(group_values)
    for base, entries in indexed.items():
        if any(len(idx) != 1 for idx in entries):
            continue  # rank-2+ array (e.g. boozer_bmnc(m,n)); consumed via .indexed
        existing = merged.get(base)
        vec: List[Scalar] = list(existing) if isinstance(existing, list) else ([existing] if base in merged else [])
        for idx, val in sorted(entries.items()):
            while len(vec) < idx[0]:
                vec.append(0)
            vec[idx[0] - 1] = val
        merged[base] = vec if len(vec) != 1 else vec[0]
    return merged


def _build_section(section_name: str, nml: RawNamelist) -> Any:
    cls = _SECTION_CLASSES[section_name]
    group = _merge_indexed(dict(nml.group(section_name)), nml.indexed.get(section_name, {}))
    kwargs: Dict[str, Any] = {}
    for fld in fields(cls):
        key = fld.metadata.get("nml")
        if key is not None and key.upper() in group:
            kwargs[fld.name] = _convert(group[key.upper()], fld.default)
    return cls(**kwargs)


def _validate(inp: SfincsInput) -> SfincsInput:
    """Replicate the checks of validateInput.F90 that dkx relies on.

    Hard errors raise ``ValueError``; the RHSMode=3 monoenergetic overrides
    are applied (as in Fortran) and recorded in ``warnings``.
    """
    warnings: List[str] = list(inp.warnings)
    gen, phys, res, other, pre, spec = (
        inp.general,
        inp.physics,
        inp.resolution,
        inp.other,
        inp.preconditioner,
        inp.species,
    )

    # --- Option ranges (validateInput.F90) ---
    if gen.rhs_mode < 1:
        raise ValueError("RHSMode must be at least 1.")
    if gen.rhs_mode > 3:
        raise ValueError(
            f"RHSMode must be 1, 2, or 3 (got {gen.rhs_mode}). The Fortran adjoint modes "
            "(RHSMode=4/5) are replaced by jax.grad differentiation of the dkx outputs."
        )
    if phys.collision_operator not in (0, 1, 3):
        raise ValueError(
            "collisionOperator must be 0 (full Fokker-Planck) or 1 (pitch-angle scattering), "
            "or 3 (improved Sugama momentum/energy-conserving model operator); "
            f"got {phys.collision_operator}."
        )
    if phys.constraint_scheme not in (-1, 0, 1, 2, 3, 4):
        raise ValueError(
            f"constraintScheme must be -1 (auto), 0, 1, 2, 3, or 4; got {phys.constraint_scheme}."
        )
    if (
        phys.include_phi1
        and not phys.read_external_phi1
        and phys.quasineutrality_option not in (1, 2)
    ):
        raise ValueError(
            "quasineutralityOption must be 1 (full quasineutrality) or 2 (EUTERPE); "
            f"got {phys.quasineutrality_option}."
        )
    if gen.rhs_mode == 2 and phys.include_phi1:
        raise ValueError("RHSMode = 2 is incompatible with includePhi1 = .true..")
    if gen.rhs_mode == 2 and spec.n_species > 1:
        raise ValueError("The transport matrix (RHSMode=2) is only available for 1 species.")
    if res.n_theta < 5:
        raise ValueError("Ntheta must be at least 5.")
    if res.n_zeta < 1:
        raise ValueError("Nzeta must be positive.")
    if res.n_xi < 1:
        raise ValueError("Nxi must be positive.")
    if res.n_x < 1:
        raise ValueError("Nx must be positive.")
    if res.n_l < 0:
        raise ValueError("NL must be at least 0.")
    if res.x_max <= 0:
        raise ValueError("xMax must be positive.")
    if res.solver_tolerance <= 0:
        raise ValueError("solverTolerance must be positive.")
    if other.theta_derivative_scheme not in (0, 1, 2):
        raise ValueError("thetaDerivativeScheme must be 0, 1, or 2.")
    if other.zeta_derivative_scheme not in (0, 1, 2):
        raise ValueError("zetaDerivativeScheme must be 0, 1, or 2.")
    if not 1 <= other.x_grid_scheme <= 8:
        raise ValueError("xGridScheme must be between 1 and 8.")
    if not -2 <= other.x_dot_derivative_scheme <= 11:
        raise ValueError("xDotDerivativeScheme must be between -2 and 11.")
    if (
        other.x_dot_derivative_scheme > 0
        and other.x_dot_derivative_scheme != 11
        and other.x_grid_scheme not in (3, 4)
    ):
        raise ValueError(
            "If xDotDerivativeScheme is >0 and not 11, then xGridScheme must be either 3 or 4."
        )
    if not 1 <= other.x_potentials_grid_scheme <= 4:
        raise ValueError("xPotentialsGridScheme must be between 1 and 4.")
    if not 0 <= pre.preconditioner_species <= 1:
        raise ValueError("preconditioner_species must be 0 or 1.")
    if not 0 <= pre.preconditioner_x <= 4:
        raise ValueError("preconditioner_x must be between 0 and 4.")
    if not 0 <= pre.preconditioner_xi <= 1:
        raise ValueError("preconditioner_xi must be 0 or 1.")
    if not 0 <= pre.preconditioner_theta <= 3:
        raise ValueError("preconditioner_theta must be between 0 and 3.")
    if not 0 <= pre.preconditioner_zeta <= 3:
        raise ValueError("preconditioner_zeta must be between 0 and 3.")
    if pre.preconditioner_x_min_l < 0 or pre.preconditioner_theta_min_l < 0 or pre.preconditioner_zeta_min_l < 0:
        raise ValueError("preconditioner_*_min_L must be at least 0.")
    if any(z == 0 for z in spec.z_s):
        raise ValueError("Charges Zs cannot be zero.")
    if any(m <= 0 for m in spec.m_hats):
        raise ValueError("Masses mHats must be positive.")
    if any(t <= 0 for t in spec.t_hats):
        raise ValueError("Temperatures THats must be positive.")
    if any(n <= 0 for n in spec.n_hats):
        raise ValueError("Densities nHats must be positive.")

    # --- RHSMode=3 monoenergetic hard overrides (validateInput.F90:51-192) ---
    if gen.rhs_mode == 3:
        if phys.nu_prime == 0:
            raise ValueError("When running with RHSMode=3, you must set nuPrime to a nonzero value.")
        if other.n_xi_for_x_option != 0:
            warnings.append("Setting Nxi_for_x_option=0, since RHSMode=3.")
            other = replace(other, n_xi_for_x_option=0)
        if res.n_x != 1:
            warnings.append("RHSMode=3 with Nx > 1 is incompatible. Setting Nx = 1.")
            res = replace(res, n_x=1)
        if not phys.use_dkes_exb_drift:
            warnings.append("RHSMode=3 requires useDKESExBDrift = .true.. Setting it.")
            phys = replace(phys, use_dkes_exb_drift=True)
        if phys.include_x_dot_term:
            warnings.append("RHSMode=3 is incompatible with includeXDotTerm. Setting it false.")
            phys = replace(phys, include_x_dot_term=False)
        if phys.include_electric_field_term_in_xi_dot:
            warnings.append(
                "RHSMode=3 is incompatible with includeElectricFieldTermInXiDot. Setting it false."
            )
            phys = replace(phys, include_electric_field_term_in_xi_dot=False)
        if phys.collision_operator != 1:
            warnings.append("RHSMode=3 requires collisionOperator = 1 (pitch-angle scattering). Setting it.")
            phys = replace(phys, collision_operator=1)
        if phys.include_phi1:
            warnings.append("RHSMode=3 is incompatible with includePhi1. Setting it false.")
            phys = replace(phys, include_phi1=False)
        if phys.include_temperature_equilibration_term:
            warnings.append(
                "RHSMode=3 is incompatible with includeTemperatureEquilibrationTerm. Setting it false."
            )
            phys = replace(phys, include_temperature_equilibration_term=False)
        if spec.n_species > 1:
            warnings.append("RHSMode=3 with >1 species is incompatible. Using only the first species.")
            spec = replace(
                spec,
                **{
                    f.name: getattr(spec, f.name)[:1]
                    for f in fields(spec)
                    if isinstance(getattr(spec, f.name), tuple)
                },
            )

    return replace(
        inp,
        general=gen,
        physics=phys,
        resolution=res,
        other=other,
        preconditioner=pre,
        species=spec,
        warnings=tuple(warnings),
    )


def sfincs_input_from_raw(nml: RawNamelist, *, validate: bool = True) -> SfincsInput:
    """Build a typed :class:`SfincsInput` from a parsed :class:`RawNamelist`."""
    inp = SfincsInput(
        general=_build_section("general", nml),
        geometry=_build_section("geometryparameters", nml),
        species=_build_section("speciesparameters", nml),
        physics=_build_section("physicsparameters", nml),
        resolution=_build_section("resolutionparameters", nml),
        other=_build_section("othernumericalparameters", nml),
        preconditioner=_build_section("preconditioneroptions", nml),
        export_f=dict(nml.group("export_f")),
        raw=nml,
    )
    return _validate(inp) if validate else inp


def load_sfincs_input(path: str | Path, *, validate: bool = True) -> SfincsInput:
    """Read and (optionally) validate a SFINCS ``input.namelist`` file."""
    return sfincs_input_from_raw(read_sfincs_input(path), validate=validate)
