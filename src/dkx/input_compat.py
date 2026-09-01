from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from .namelist import Namelist, read_sfincs_input
from .paths import resolve_existing_path
from .paths import repository_root


def _group_get(group: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = group.get(key.upper(), None)
        if value is not None:
            return value
    return None


def lookup_config_value(config: Any, groups: tuple[str, ...], key: str, default: Any = None) -> Any:
    """Read a SFINCS option from either a ``Namelist`` or nested mapping.

    This is intentionally small but shared: problem modules need the same
    Fortran-style case-insensitive lookup when validating source-compatible
    ambipolar and adjoint-sensitivity settings.
    """

    key_upper = key.upper()
    for group in groups:
        group_data: Any
        if hasattr(config, "group"):
            group_data = config.group(group)
        elif isinstance(config, Mapping):
            group_data = config.get(group, config.get(group.lower(), config.get(group.upper(), {})))
        else:
            group_data = {}
        if isinstance(group_data, Mapping):
            if key_upper in group_data:
                return group_data[key_upper]
            if key in group_data:
                return group_data[key]
            lower_map = {str(k).lower(): v for k, v in group_data.items()}
            if key.lower() in lower_map:
                return lower_map[key.lower()]
    if isinstance(config, Mapping):
        if key_upper in config:
            return config[key_upper]
        if key in config:
            return config[key]
        lower_map = {str(k).lower(): v for k, v in config.items()}
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return default


def first_config_value(value: Any, default: Any = None) -> Any:
    """Return the first scalar from a namelist value or ``default`` if empty."""

    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def bool_config_values(value: Any) -> tuple[bool, ...]:
    """Return a tuple of booleans from scalar or vector namelist values."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(bool(item) for item in value)
    return (bool(value),)


def config_bool(config: Any, groups: tuple[str, ...], key: str, default: bool = False) -> bool:
    return bool(first_config_value(lookup_config_value(config, groups, key, default), default))


def config_int(config: Any, groups: tuple[str, ...], key: str, default: int = 0) -> int:
    return int(first_config_value(lookup_config_value(config, groups, key, default), default))


def config_float(config: Any, groups: tuple[str, ...], key: str, default: float = 0.0) -> float:
    return float(first_config_value(lookup_config_value(config, groups, key, default), default))


def effective_equilibrium_file(*, geom_params: Mapping[str, Any]) -> Any | None:
    geometry_scheme = int(_group_get(geom_params, "geometryScheme") or -1)
    equilibrium_file = _group_get(geom_params, "equilibriumFile")
    if equilibrium_file is not None:
        return equilibrium_file
    if geometry_scheme == 10:
        return _group_get(geom_params, "fort996boozer_file")
    if geometry_scheme == 11:
        return _group_get(geom_params, "JGboozer_file")
    if geometry_scheme == 12:
        return _group_get(geom_params, "JGboozer_file_NonStelSym")
    return None


def _resolve_equilibrium_file_from_namelist(*, nml: Namelist) -> Path:
    """Resolve the effective VMEC/Boozer equilibrium referenced by a namelist.

    The resolver follows SFINCS-v3 input conventions, including the legacy
    Boozer alias keys and the VMEC ASCII-to-NetCDF sibling preference used by
    mixed upstream benchmark directories.
    """

    geom_params = nml.group("geometryParameters")
    equilibrium_file = effective_equilibrium_file(geom_params=geom_params)
    if equilibrium_file is None:
        raise ValueError("Missing geometryParameters.equilibriumFile")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = repository_root()
    extra = (
        (repo_root / "tests" / "ref", repo_root / "src" / "dkx" / "data" / "equilibria")
        if repo_root is not None
        else ()
    )
    geometry_scheme = int(_group_get(geom_params, "geometryScheme") or -1)

    raw = str(equilibrium_file).strip().strip('"').strip("'")
    p = Path(raw)
    if geometry_scheme == 5 and p.suffix.lower() in {".txt", ".dat"}:
        p_nc = p.with_suffix(".nc")
        try:
            return resolve_existing_path(str(p_nc), base_dir=base_dir, extra_search_dirs=extra).path
        except FileNotFoundError:
            pass
    return resolve_existing_path(raw, base_dir=base_dir, extra_search_dirs=extra).path


def localize_equilibrium_file_in_place(*, input_namelist: Path, overwrite: bool = False) -> Path | None:
    """Copy the effective equilibrium next to an input file and patch the input.

    Example and benchmark decks often refer to equilibria relative to an
    upstream source tree. Localizing keeps a staged run directory self-contained
    for both DKX and SFINCS Fortran v3 comparisons.
    """

    input_namelist = Path(input_namelist).resolve()
    nml = read_sfincs_input(input_namelist)
    geom_params = nml.group("geometryParameters")
    equilibrium_file = effective_equilibrium_file(geom_params=geom_params)
    if equilibrium_file is None:
        return None

    resolved = _resolve_equilibrium_file_from_namelist(nml=nml)
    dst = input_namelist.parent / resolved.name
    if overwrite or (not dst.exists()):
        shutil.copyfile(resolved, dst)

    txt = input_namelist.read_text()
    geometry_scheme = int(_group_get(geom_params, "geometryScheme") or -1)
    if geometry_scheme == 10:
        key_candidates = ("fort996boozer_file", "equilibriumFile")
    elif geometry_scheme == 11:
        key_candidates = ("JGboozer_file", "equilibriumFile")
    elif geometry_scheme == 12:
        key_candidates = ("JGboozer_file_NonStelSym", "equilibriumFile")
    else:
        key_candidates = ("equilibriumFile",)

    txt2 = txt
    for key_name in key_candidates:
        pat = re.compile(rf"(?im)^\s*{re.escape(key_name)}\s*=\s*(['\"])(.*?)\1\s*$")
        m = pat.search(txt)
        if m is not None:
            quote = m.group(1)
            txt2 = txt.replace(m.group(0), f"  {key_name} = {quote}{dst.name}{quote}")
            break
        pat2 = re.compile(rf"(?im)^\s*{re.escape(key_name)}\s*=\s*([^!\n\r]+)\s*$")
        m2 = pat2.search(txt)
        if m2 is not None:
            txt2 = txt.replace(m2.group(0), f'  {key_name} = "{dst.name}"')
            break

    if txt2 != txt:
        input_namelist.write_text(txt2)
    return dst


def canonical_equilibrium_override(
    *,
    equilibrium_file: str | Path | None = None,
    wout_path: str | Path | None = None,
) -> str | None:
    """Return a single canonical equilibrium override string.

    ``wout_path`` is kept as a compatibility alias for VMEC-centric callers. When
    both arguments are provided they must resolve to the same textual path.
    """

    def _norm(value: str | Path | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    eq = _norm(equilibrium_file)
    wout = _norm(wout_path)
    if eq is None:
        return wout
    if wout is None:
        return eq
    if eq != wout:
        raise ValueError(
            "Received conflicting equilibrium overrides: "
            f"equilibrium_file={eq!r} and wout_path={wout!r}"
        )
    return eq


def render_input_with_equilibrium_override(
    *,
    source_text: str,
    equilibrium_override: str,
) -> str:
    """Return input text with ``equilibriumFile`` replaced or inserted."""
    pat = re.compile(r"(?im)^(\s*equilibriumFile\s*=\s*)(['\"])(.*?)\2(\s*)$")
    replacement = rf'\1"{equilibrium_override}"\4'
    if pat.search(source_text):
        return pat.sub(replacement, source_text, count=1)

    group_pat = re.compile(r"(?im)^(\s*&geometryParameters\s*$)")
    if not group_pat.search(source_text):
        return source_text
    return group_pat.sub(rf'\1\n  equilibriumFile = "{equilibrium_override}"', source_text, count=1)


def with_equilibrium_override(
    *,
    nml: Namelist,
    equilibrium_file: str | Path | None = None,
    wout_path: str | Path | None = None,
) -> Namelist:
    """Return a copy of ``nml`` with the effective equilibrium file overridden."""
    override = canonical_equilibrium_override(
        equilibrium_file=equilibrium_file,
        wout_path=wout_path,
    )
    if override is None:
        return nml

    groups = {name: dict(group) for name, group in nml.groups.items()}
    indexed = {name: {key: dict(value) for key, value in group.items()} for name, group in nml.indexed.items()}
    geom = dict(groups.get("geometryparameters", {}))
    geom["EQUILIBRIUMFILE"] = str(override)
    groups["geometryparameters"] = geom
    source_text = nml.source_text
    if source_text is not None:
        source_text = render_input_with_equilibrium_override(
            source_text=source_text,
            equilibrium_override=str(override),
        )
    return Namelist(
        groups=groups,
        indexed=indexed,
        source_path=nml.source_path,
        source_text=source_text,
    )


def effective_r_n_wish(*, geom_params: Mapping[str, Any], default: float = 0.5) -> float:
    value = _group_get(geom_params, "rN_wish", "normradius_wish")
    return float(value) if value is not None else float(default)


def effective_psi_n_wish(
    *,
    geom_params: Mapping[str, Any],
    default_r_n: float = 0.5,
    psi_a_hat: float | None = None,
    a_hat: float | None = None,
) -> float:
    """Return the requested normalized toroidal flux for v3 radial-coordinate inputs.

    SFINCS v3 lets users select the input surface with ``inputRadialCoordinate``:
    ``psiHat`` (0), ``psiN`` (1), ``rHat`` (2), or ``rN`` (3). Most examples use
    ``rN``, but the Redl/SFINCS benchmark decks specify ``psiN_wish`` directly.
    This helper centralizes the conversion so geometry selection and radial-gradient
    normalization use the same surface.
    """
    input_radial_value = _group_get(geom_params, "inputRadialCoordinate")
    input_radial = int(input_radial_value) if input_radial_value is not None else 3
    if input_radial == 0:
        value = _group_get(geom_params, "psiHat_wish")
        if value is None:
            return float(default_r_n) * float(default_r_n)
        if psi_a_hat is None:
            raise ValueError("psi_a_hat is required to convert psiHat_wish to psiN_wish.")
        return float(value) / float(psi_a_hat)
    if input_radial == 1:
        value = _group_get(geom_params, "psiN_wish")
        return float(value) if value is not None else float(default_r_n) * float(default_r_n)
    if input_radial == 2:
        value = _group_get(geom_params, "rHat_wish")
        if value is None:
            return float(default_r_n) * float(default_r_n)
        if a_hat is None:
            raise ValueError("a_hat is required to convert rHat_wish to psiN_wish.")
        return (float(value) / float(a_hat)) ** 2
    if input_radial == 3:
        r_n = effective_r_n_wish(geom_params=geom_params, default=default_r_n)
        return float(r_n) * float(r_n)
    raise ValueError(f"Invalid inputRadialCoordinate={input_radial}.")


def effective_psi_a_hat(
    *,
    geom_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: float,
) -> float:
    value = _group_get(geom_params, "psiAHat")
    if value is None:
        value = _group_get(phys_params, "psiAHat")
    return float(value) if value is not None else float(default)




def scheme4_radial_constants() -> tuple[float, float]:
    """Return v3's built-in ``geometryScheme=4`` radial normalization constants."""

    psi_a_hat = -0.384935
    a_hat = 0.5109
    return psi_a_hat, a_hat




_scheme4_radial_constants = scheme4_radial_constants


def infer_species_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    species_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    if _group_get(species_params, "dNHatdrHats", "dTHatdrHats") is not None:
        return 2
    if _group_get(species_params, "dNHatdpsiHats", "dTHatdpsiHats") is not None:
        return 0
    if _group_get(species_params, "dNHatdpsiNs", "dTHatdpsiNs") is not None:
        return 1
    if _group_get(species_params, "dNHatdrNs", "dTHatdrNs") is not None:
        return 3
    return int(default)


def infer_phi_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    if _group_get(phys_params, "Er") is not None:
        return 4
    if _group_get(phys_params, "dPhiHatdrHat") is not None:
        return 2
    if _group_get(phys_params, "dPhiHatdpsiHat") is not None:
        return 0
    if _group_get(phys_params, "dPhiHatdpsiN") is not None:
        return 1
    if _group_get(phys_params, "dPhiHatdrN") is not None:
        return 3
    return int(default)


def infer_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    species_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    phi_coord = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=geom_params,
        phys_params=phys_params,
        default=default,
    )
    if _group_get(phys_params, "dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN", "Er") is not None:
        return int(phi_coord)

    return infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params,
        species_params=species_params,
        default=default,
    )


def effective_use_iterative_linear_solver(*, other_params: Mapping[str, Any], default: int = 1) -> int:
    value = _group_get(other_params, "useIterativeLinearSolver", "useIterativeSolver")
    return int(value) if value is not None else int(default)


# ===========================================================================
# SFINCS ``input.namelist`` -> native DKX case (the ``dkx convert`` migration)
# ===========================================================================
#
# The two descriptions differ in more than spelling, and both differences are
# handled here rather than by the caller:
#
# 1. A deck is DIMENSIONLESS.  ``Zs``/``mHats``/``nHats``/``THats``/``Er`` are
#    ratios to the SFINCS reference set, which is pinned rather than free
#    (``globalVariables.F90:133-135``; see :mod:`dkx.units`).  A case is SI:
#    ``density_m3``, ``temperature_keV``, ``mass_amu``, ``value_kV_m``.  Every
#    rescaling below goes through :mod:`dkx.units`; there are no literal
#    conversion factors in this file.
#
# 2. A deck states ONE flux surface plus prescribed radial gradients.  A case
#    states a radial PROFILE, and :mod:`dkx.execution` recovers the gradients by
#    differentiating it (``numpy.gradient`` against ``rHat``).  The deck's local
#    value-plus-slope therefore expands into the smallest profile that carries
#    exactly that slope: three surfaces bracketing the deck's, holding the
#    profile that is LINEAR in ``rHat``.  ``numpy.gradient`` is exact on linear
#    data at every node, edges included, so the operator the native path builds
#    at the deck's surface sees the deck's own ``nHat``, ``THat`` and gradients.
#
# Anything the case schema cannot carry REFUSES (plan.md operating rule 11)
# through :class:`dkx.config.CaseValidationError`, naming the namelist key and
# its value.  Converting such a deck into an adjacent model would produce a case
# that runs and is quietly wrong, which is the failure mode the rule exists to
# prevent.  The refusals track what ``dkx.execution.run_case`` actually builds,
# not what the case *schema* can spell: a schema field the native route refuses
# (``magnetic_drifts = "full"``) is no more convertible than a missing one.

#: ``geometryScheme`` -> the analytic-equilibrium token ``geometry.file`` names
#: (``dkx.execution._analytic_scheme``).
_ANALYTIC_GEOMETRY_FILES: Mapping[int, str] = {
    1: "tokamak",
    2: "lhd_standard",
    3: "lhd_inward",
    4: "w7x_standard",
}

#: ``(psiAHat, aHat, rN)`` that ``geometry.F90`` fixes internally for the
#: built-in models; upstream ignores the ``*_wish`` keys for these, forcing
#: ``rN = 0.5``, so the converted case pins the surface at ``psiN = 0.25``.
_FIXED_ANALYTIC_RADIAL: Mapping[int, tuple[float, float, float]] = {
    2: (0.5585 * 0.5585 / 2.0, 0.5585, 0.5),
    3: (0.5400 * 0.5400 / 2.0, 0.5400, 0.5),
    4: (-0.384935, 0.5109, 0.5),
}

#: ``geometryScheme=1`` reads its three-helicity model from the deck, but the
#: native route calls ``FluxSurfaceGeometry.from_scheme(1)`` with no overrides,
#: so only a deck that leaves the model at the ``globalVariables.F90`` defaults
#: describes the same magnetic field.
_SCHEME1_PINNED_MODEL: tuple[tuple[str, float], ...] = (
    ("epsilon_t", -0.07053),
    ("epsilon_h", 0.05067),
    ("epsilon_antisymm", 0.0),
    ("iota", 0.4542),
    ("GHat", 3.7481),
    ("IHat", 0.0),
    ("B0OverBBar", 1.0),
    ("helicity_l", 2.0),
    ("helicity_n", 10.0),
    ("helicity_antisymm_l", 1.0),
    ("helicity_antisymm_n", 0.0),
    ("aHat", 0.5585),
)

_COLLISION_OPERATORS: Mapping[int, str] = {
    0: "linearized_fokker_planck",
    1: "pitch_angle_scattering",
}

#: ``(round(Z), round(mass_amu))`` -> the species name a case declares.
_SPECIES_NAMES: Mapping[tuple[int, int], str] = {
    (1, 1): "hydrogen",
    (1, 2): "deuterium",
    (1, 3): "tritium",
    (2, 3): "helium3",
    (2, 4): "helium",
    (5, 11): "boron",
    (6, 12): "carbon",
}

#: ``inputRadialCoordinateForGradients`` -> the ``Phi`` gradient key it selects.
_PHI_GRADIENT_KEYS: Mapping[int, str] = {
    0: "dPhiHatdpsiHat",
    1: "dPhiHatdpsiN",
    2: "dPhiHatdrHat",
    3: "dPhiHatdrN",
}

#: ``inputRadialCoordinateForGradients`` -> the species-gradient key suffix.
_SPECIES_GRADIENT_SUFFIX: Mapping[int, str] = {
    0: "psiHats",
    1: "psiNs",
    2: "rHats",
    3: "rNs",
    4: "rHats",
}

#: Largest half-width, in ``rN``, of the three-surface profile stencil.  Small
#: enough that the linear profile stays physical across it and that the extra
#: two surfaces sit near the deck's, large enough to stay far from float noise.
_MAX_STENCIL_HALF_WIDTH_R_N = 0.05
_MIN_STENCIL_HALF_WIDTH_R_N = 1.0e-9


def _refuse(group: str, key: str, value: Any, expected: str, correction: str) -> Any:
    """Refuse one namelist option, in the style of ``inputs._unsupported_option``."""
    from .config import CaseValidationError  # noqa: PLC0415

    raise CaseValidationError(f"{group}.{key}", value, expected, correction)


def _deck_vector(nml: Any, group: str, key: str) -> tuple[float, ...] | None:
    raw = lookup_config_value(nml, (group,), key, None)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(float(item) for item in raw)
    return (float(raw),)


def _species_vector(
    nml: Any, key: str, *, default: float, count: int, broadcast: bool = False
) -> tuple[float, ...]:
    """One ``&speciesParameters`` array, with ``readInput.F90``'s defaults."""
    values = _deck_vector(nml, "speciesParameters", key)
    if values is None:
        return (float(default),) * count
    if broadcast and len(values) == 1 and count > 1:
        return values * count
    if len(values) != count:
        _refuse(
            "speciesParameters",
            key,
            list(values),
            f"exactly {count} entries, one per Zs entry",
            "Give every species array the same length as Zs.",
        )
    return values


def _species_name(*, charge: float, mass_amu: float, index: int, taken: set[str]) -> str:
    """A physical, unique name for one species; the case schema requires one."""
    if charge < 0.0 and mass_amu < 0.1:
        base = "electron"
    else:
        key = (round(charge), round(mass_amu))
        base = _SPECIES_NAMES.get(key)
        if (
            base is None
            or abs(charge - key[0]) > 1.0e-6
            or abs(mass_amu - key[1]) > 0.05
        ):
            base = f"species{index + 1}"
    name = base
    counter = 2
    while name in taken:
        name = f"{base}_{counter}"
        counter += 1
    taken.add(name)
    return name


def _stencil_half_width(
    *,
    r_n: float,
    a_hat: float,
    r_n_bounds: tuple[float, float],
    profiles: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """Half-width in ``rN`` of the profile stencil around the deck's surface.

    Three constraints, all of which only shrink the stencil and none of which
    touch the value or slope at the deck's own surface:

    - **the equilibrium's radial support.** ``r_n_bounds`` is ``(0, 1]`` for the
      analytic models, but a ``.bc`` or ``wout`` file only carries the surfaces
      it stores, and the native route interpolates between them. A neighbour
      outside that span makes ``dkx run`` fail on a geometry lookup, which is a
      failure the conversion can simply avoid.
    - **positivity.** A case declares densities and temperatures that must be
      positive at every surface, and a linear profile crosses zero at
      ``value/slope``; half that distance keeps every neighbour physical.
    - **locality**, so the extra surfaces stay near the deck's.
    """
    limits = [_MAX_STENCIL_HALF_WIDTH_R_N, 0.9 * (float(r_n) - float(r_n_bounds[0]))]
    for values, slopes in profiles:
        for value, slope in zip(values, slopes):
            if slope != 0.0:
                limits.append(0.5 * abs(value / (slope * float(a_hat))))
    return min(limits)


def _analytic_geometry(nml: Any, scheme: int) -> tuple[str, float, float, float]:
    """``(file token, psiAHat, aHat, psiN)`` for ``geometryScheme`` 1-4."""
    geom_params = nml.group("geometryParameters")
    phys_params = nml.group("physicsParameters")
    if scheme in _FIXED_ANALYTIC_RADIAL:
        psi_a_hat, a_hat, r_n = _FIXED_ANALYTIC_RADIAL[scheme]
        return _ANALYTIC_GEOMETRY_FILES[scheme], psi_a_hat, a_hat, r_n * r_n

    psi_a_hat = effective_psi_a_hat(
        geom_params=geom_params, phys_params=phys_params, default=0.15596
    )
    a_hat = config_float(nml, ("geometryParameters",), "aHat", 0.5585)
    for key, pinned in _SCHEME1_PINNED_MODEL:
        value = config_float(nml, ("geometryParameters",), key, pinned)
        if value != pinned:
            _refuse(
                "geometryParameters",
                key,
                value,
                f"{pinned!r}, the globalVariables.F90 default",
                "geometryScheme=1 is the three-helicity analytic model, and a case "
                "names it by the token geometry.file = 'tokamak' only -- the case "
                "schema has nowhere to put a modified epsilon/iota/GHat/helicity, "
                "so dkx.execution builds it from the defaults. Write this "
                "equilibrium as a Boozer .bc or VMEC wout file, or run the deck "
                "through the SFINCS-compatibility path instead.",
            )
    if psi_a_hat != 0.15596:
        _refuse(
            "geometryParameters",
            "psiAHat",
            psi_a_hat,
            "0.15596, the globalVariables.F90 default",
            "The native analytic route pins the radial normalization of "
            "geometryScheme=1 to the default (dkx.execution._prepare_geometry); "
            "a different psiAHat rescales every radial gradient and flux.",
        )
    psi_n = effective_psi_n_wish(
        geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
    )
    return _ANALYTIC_GEOMETRY_FILES[1], psi_a_hat, a_hat, float(psi_n)


def _file_geometry(
    nml: Any, scheme: int
) -> tuple[Path, str, float, float, float, tuple[float, float]]:
    """``(path, format, psiAHat, aHat, psiN, rN bounds)`` for scheme 5/11/12.

    The equilibrium is read at convert time because ``psiAHat`` and ``aHat``
    live in the file, and the deck's gradients cannot be expressed as an SI
    profile without them.
    """
    geom_params = nml.group("geometryParameters")
    radial_option = config_int(nml, ("geometryParameters",), "VMECRadialOption", 1)
    if radial_option != 0:
        _refuse(
            "geometryParameters",
            "VMECRadialOption",
            radial_option,
            "0 (interpolate to the requested surface)",
            "The native route interpolates the equilibrium to geometry.surfaces "
            "(dkx.execution._geometry_context passes vmec_radial_option=0). "
            "VMECRadialOption=1 snaps to the nearest stored surface instead, so "
            "the deck and the converted case would solve on different surfaces.",
        )
    try:
        path = _resolve_equilibrium_file_from_namelist(nml=nml)
    except (FileNotFoundError, ValueError) as exc:
        _refuse(
            "geometryParameters",
            "equilibriumFile",
            effective_equilibrium_file(geom_params=geom_params),
            "a readable VMEC wout or Boozer .bc file",
            f"A case records the equilibrium by path, so it must resolve now: {exc}",
        )
        raise AssertionError("unreachable")  # pragma: no cover

    if scheme in {11, 12}:
        from .magnetic_geometry import read_native_boozer  # noqa: PLC0415

        data = read_native_boozer(path)
        psi_a_hat = float(data.header.psi_a_hat)
        a_hat = float(data.header.a_hat)
        psi_n = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        stored = [float(surface.r_n) for surface in data.surfaces]
        return (
            path,
            "boozer",
            psi_a_hat,
            a_hat,
            float(psi_n),
            (min(stored), max(stored)),
        )

    from .magnetic_geometry import psi_a_hat_from_wout, read_vmec_wout  # noqa: PLC0415

    for key, pinned in (("rippleScale", 1.0), ("min_Bmn_to_load", 0.0)):
        value = config_float(nml, ("geometryParameters",), key, pinned)
        if value != pinned:
            _refuse(
                "geometryParameters",
                key,
                value,
                f"{pinned!r}, the globalVariables.F90 default",
                "The native VMEC route builds the surface from the wout alone "
                "(dkx.execution._geometry_context); a case has no field for a "
                "modified spectrum, so the converted case would use a different "
                "magnetic field from the deck.",
            )
    nyquist = config_int(nml, ("geometryParameters",), "VMEC_Nyquist_option", 1)
    if nyquist != 1:
        _refuse(
            "geometryParameters",
            "VMEC_Nyquist_option",
            nyquist,
            "1, the globalVariables.F90 default",
            "The native VMEC route reads the Nyquist spectrum with the default "
            "option and a case cannot record another choice.",
        )
    wout = read_vmec_wout(path)
    psi_a_hat = float(psi_a_hat_from_wout(wout))
    a_hat = float(wout.aminor_p)
    psi_n = effective_psi_n_wish(
        geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
    )
    # The VMEC half mesh carries psiN = (j - 0.5)/(ns - 1) for j = 1 .. ns-1;
    # outside that span there is nothing to interpolate between.
    half = 0.5 / max(1, int(wout.ns) - 1)
    return (
        path,
        "vmec",
        psi_a_hat,
        a_hat,
        float(psi_n),
        (math.sqrt(half), math.sqrt(1.0 - half)),
    )


def _portable_equilibrium_path(
    equilibrium: Path, base_dir: str | Path | None
) -> str:
    """The equilibrium path a case should record.

    Relative when the equilibrium sits beside or beneath the case file, so the
    pair can be moved or committed together; absolute otherwise. A relative path
    that climbs out of the case's directory (``../../../..`` into a download
    cache) is portable in name only and breaks the moment the case is copied.
    """
    if base_dir is None:
        return str(equilibrium)
    relative = os.path.relpath(equilibrium, Path(base_dir).expanduser().resolve())
    return str(equilibrium) if relative.startswith(os.pardir) else relative


def _refuse_unconvertible_physics(
    nml: Any, *, collision_operator: int, n_xi: int, field_is_zero: bool
) -> None:
    """Refuse every deck option the native execution route does not carry.

    Only options that change the converged answer are listed, matching the
    policy of :func:`dkx.inputs.check_supported_options`: Fortran-only I/O,
    PETSc plumbing, the preconditioner family and ``export_f`` are left alone
    because refusing them would make users edit working decks for nothing.
    """
    from .constants import DEFAULT_ALPHA, DEFAULT_DELTA, DEFAULT_NU_N  # noqa: PLC0415

    phys = "physicsParameters"
    other = "otherNumericalParameters"

    for key, pinned, why in (
        (
            "Delta",
            DEFAULT_DELTA,
            "Delta = mBar*vBar/(e*BBar*RBar) follows from the pinned reference set, "
            "which a case states in SI instead of restating as a ratio",
        ),
        (
            "alpha",
            DEFAULT_ALPHA,
            "alpha = e*phiBar/TBar follows from the pinned reference set, which fixes "
            "phiBar = TBar/e",
        ),
        (
            "nu_n",
            DEFAULT_NU_N,
            "nu_n = nuBar*RBar/vBar is the collisionality AT the reference parameters; "
            "a case gives densities and temperatures in SI and dkx.execution evaluates "
            "the collisionality from them with ln(Lambda) = 17",
        ),
    ):
        value = config_float(nml, (phys,), key, pinned)
        if value != pinned:
            _refuse(
                phys,
                key,
                value,
                f"{pinned!r}, the value the pinned SFINCS reference set implies",
                f"{why}. A case has no field for an overridden {key}, so the "
                "converted case would solve at a different normalization than the "
                "deck. Keep this deck on the SFINCS-compatibility path.",
            )

    krook = config_float(nml, (phys,), "Krook", 0.0)
    if krook != 0.0:
        _refuse(
            phys,
            "Krook",
            krook,
            "0.0",
            "A nonzero Krook adds a model relaxation term to the collision operator "
            "(dkx.collisions), which a case cannot declare.",
        )

    e_parallel = config_float(nml, (phys,), "EParallelHat", 0.0)
    if e_parallel != 0.0:
        _refuse(
            phys,
            "EParallelHat",
            e_parallel,
            "0.0",
            "An inductive parallel drive enters the kinetic equation as an extra "
            "source; dkx.execution builds the native operator with "
            "e_parallel_hat = 0 and a case has no field for it.",
        )
    e_parallel_spec = _deck_vector(nml, phys, "EParallelHatSpec")
    if e_parallel_spec is not None and any(value != 0.0 for value in e_parallel_spec):
        _refuse(
            phys,
            "EParallelHatSpec",
            list(e_parallel_spec),
            "all zeros",
            "A per-species inductive drive has no case field; dkx.execution builds "
            "the native operator with e_parallel_hat_spec = 0.",
        )

    if bool(config_bool(nml, (phys,), "includePhi1", False)):
        _refuse(
            phys,
            "includePhi1",
            True,
            ".false.",
            "physics.phi1 = 'kinetic'/'full' is in the case schema but "
            "dkx.execution.run_case refuses it: the native route solves the LINEAR "
            "drift-kinetic equation with no quasineutrality block. Converting would "
            "produce a case that cannot run.",
        )

    drift_scheme = config_int(nml, (phys,), "magneticDriftScheme", 0)
    if drift_scheme != 0:
        _refuse(
            phys,
            "magneticDriftScheme",
            drift_scheme,
            "0 (no tangential magnetic drifts)",
            "physics.magnetic_drifts = 'full' is in the case schema but "
            "dkx.execution.run_case implements only 'dkes': it assembles no "
            "tangential magnetic-drift terms. Converting would produce a case that "
            "cannot run.",
        )

    if not field_is_zero:
        for key, required, explanation in (
            (
                "useDKESExBDrift",
                True,
                "the native operator divides the ExB drift by <B^2> "
                "(use_dkes_exb=True in dkx.execution._make_operator); full "
                "trajectories divide by B^2 pointwise",
            ),
            (
                "includeXDotTerm",
                False,
                "the native operator carries no E_r speed-space acceleration "
                "(with_er_xdot=False)",
            ),
            (
                "includeElectricFieldTermInXiDot",
                False,
                "the native operator carries no E_r pitch-angle term "
                "(with_er_xidot=False)",
            ),
        ):
            value = bool(config_bool(nml, (phys,), key, key != "useDKESExBDrift"))
            if value != required:
                _refuse(
                    phys,
                    key,
                    value,
                    ".true." if required else ".false.",
                    "This deck runs full E_r trajectories. physics.magnetic_drifts "
                    "= 'dkes' is the only trajectory model dkx.execution.run_case "
                    f"builds, and {explanation}. With Er = 0 the three switches are "
                    "inert and the deck converts; with a finite E_r they change the "
                    "answer.",
                )

    if collision_operator not in _COLLISION_OPERATORS:
        _refuse(
            phys,
            "collisionOperator",
            collision_operator,
            "0 (linearized Fokker-Planck) or 1 (pitch-angle scattering)",
            "physics.collisions has no name for the improved Sugama model operator "
            "(collisionOperator=3); it is reachable only through the "
            "SFINCS-compatibility path.",
        )

    constraint_scheme = config_int(nml, (phys,), "constraintScheme", -1)
    expected_constraint = 1 if collision_operator == 0 else 2
    if constraint_scheme not in (-1, expected_constraint):
        _refuse(
            phys,
            "constraintScheme",
            constraint_scheme,
            f"-1 (auto) or {expected_constraint} for collisionOperator="
            f"{collision_operator}",
            "dkx.execution._make_operator derives the constraint scheme from the "
            "collision operator (1 for Fokker-Planck, 2 for pitch-angle scattering); "
            "a case cannot record a different null-space treatment.",
        )

    x_grid_scheme = config_int(nml, (other,), "xGridScheme", 5)
    if x_grid_scheme != 5:
        _refuse(
            other,
            "xGridScheme",
            x_grid_scheme,
            "5 (the Landreman-Ernst speed grid, the v3 default)",
            "dkx.execution._make_grids builds the native speed grid with scheme 5 "
            "and a case has no field for another speed grid.",
        )
    x_grid_k = config_float(nml, (other,), "xGrid_k", 0.0)
    if x_grid_k != 0.0:
        _refuse(
            other,
            "xGrid_k",
            x_grid_k,
            "0.0",
            "The speed-grid weight exponent changes the quadrature nodes; "
            "dkx.execution builds the native grid with xGrid_k = 0.",
        )
    for key in ("thetaDerivativeScheme", "zetaDerivativeScheme"):
        scheme = config_int(nml, (other,), key, 2)
        if scheme != 2:
            _refuse(
                other,
                key,
                scheme,
                "2 (the 4th-order centered difference, the v3 default)",
                "dkx.execution._make_grids builds both angular derivative matrices "
                "with scheme 2; a case has no field for the spectral or 2nd-order "
                "stencils.",
            )
    ramp = config_int(nml, (other,), "Nxi_for_x_option", 1)
    if ramp not in (0, 1, 2):
        _refuse(
            other,
            "Nxi_for_x_option",
            ramp,
            "0, 1, or 2",
            "resolution.pitch_speed_ramp carries options 0-2 only.",
        )

    if collision_operator == 0:
        n_l = config_int(nml, ("resolutionParameters",), "NL", 4)
        expected_n_l = min(4, int(n_xi))
        if n_l != expected_n_l:
            _refuse(
                "resolutionParameters",
                "NL",
                n_l,
                f"{expected_n_l} (= min(4, Nxi)) for collisionOperator=0",
                "NL is the Legendre depth of the Fokker-Planck field term; "
                "dkx.execution._make_grids fixes it at min(4, resolution.pitch) and "
                "a case has no field for it. It is inert for "
                "collisionOperator=1, so pitch-angle-scattering decks are free to "
                "set it.",
            )
        rosenbluth = lookup_config_value(nml, (other,), "RosenbluthMethod", None)
        if rosenbluth is not None and str(first_config_value(rosenbluth)).strip().strip(
            "\"'"
        ).lower() not in {"", "quadpack"}:
            _refuse(
                other,
                "RosenbluthMethod",
                rosenbluth,
                "'quadpack' (the Fortran-parity quadrature, the dkx default)",
                "The Rosenbluth-potential quadrature route changes the Fokker-Planck "
                "matrices; dkx.execution builds them with the default route and a "
                "case has no field for another.",
            )


def _electric_field_hat(nml: Any) -> float:
    """The deck's ``Er`` (kV/m in Hat units), refusing what a case cannot carry."""
    phys = "physicsParameters"
    er = config_float(nml, (phys,), "Er", 0.0)
    coordinate = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=nml.group("geometryParameters"),
        phys_params=nml.group(phys),
        default=4,
    )
    if coordinate not in _PHI_GRADIENT_KEYS and coordinate != 4:
        _refuse(
            "geometryParameters",
            "inputRadialCoordinateForGradients",
            coordinate,
            "0, 1, 2, 3, or 4",
            "radialCoordinates.F90 defines only these five gradient coordinates.",
        )
    if coordinate == 4:
        return er

    key = _PHI_GRADIENT_KEYS[coordinate]
    value = config_float(nml, (phys,), key, 0.0)
    if value == 0.0 and er == 0.0:
        return 0.0
    _refuse(
        phys,
        key if value != 0.0 else "Er",
        value if value != 0.0 else er,
        "the radial electric field given as Er with "
        "inputRadialCoordinateForGradients = 4",
        "electric_field.value_kV_m drives BOTH the radial-electric-field source "
        "and the ExB advection of the native operator, from one number. Under "
        f"inputRadialCoordinateForGradients={coordinate} the compatibility path "
        f"drives the source from {key} while leaving the ExB terms at the Er "
        "input, and a case cannot express that split.",
    )
    raise AssertionError("unreachable")  # pragma: no cover


def case_from_sfincs_namelist(
    source: str | Path,
    *,
    name: str | None = None,
    geometry_base_dir: str | Path | None = None,
) -> Any:
    """Build the native :class:`dkx.config.Case` a SFINCS deck describes.

    Args:
        source: path to a SFINCS v3 ``input.namelist``.
        name: case name; defaults to a name derived from ``source``.
        geometry_base_dir: directory the VMEC/Boozer ``geometry.file`` is written
            relative to (normally the destination case file's directory).  With
            ``None`` the resolved absolute path is recorded.

    Returns:
        A validated ``Case`` describing the same calculation as the deck.

    Raises:
        dkx.config.CaseValidationError: the deck asks for something the case
            schema or the native execution route cannot express.  The message
            names the namelist key and its value.
    """
    from .config import Case  # noqa: PLC0415
    from .constants import RadialCoordinates  # noqa: PLC0415
    from .inputs import (  # noqa: PLC0415
        check_supported_deck_options,
        read_sfincs_input as read_typed_sfincs_input,
    )
    from .units import (  # noqa: PLC0415
        density_m3_from_n_hat,
        electric_field_kv_m_from_er_hat,
        mass_amu_from_m_hat,
        temperature_kev_from_t_hat,
    )

    path = Path(source).expanduser().resolve()
    nml = read_typed_sfincs_input(path)
    # The guards that landed with the namelist reader own the options that are
    # unsupported everywhere in dkx; this module adds only the ones specific to
    # the native case route.
    check_supported_deck_options(nml)

    geom_params = nml.group("geometryParameters")

    rhs_mode = config_int(nml, ("general",), "RHSMode", 1)
    if rhs_mode != 1:
        _refuse(
            "general",
            "RHSMode",
            rhs_mode,
            "1 (a profile-gradient solve)",
            "run.workflow = 'transport_matrix' (RHSMode=2) and 'monoenergetic' "
            "(RHSMode=3) are in the case schema but dkx.execution.run_case "
            "implements neither, and RHSMode=3's nuPrime/EStar forcing has no case "
            "field at all. Run these decks with `dkx sfincs transport-matrix-v3` or "
            "`dkx sfincs monoenergetic-database`.",
        )

    scheme_value = lookup_config_value(nml, ("geometryParameters",), "geometryScheme", None)
    if scheme_value is None:
        _refuse(
            "geometryParameters",
            "geometryScheme",
            None,
            "an explicit geometryScheme",
            "The geometry family selects geometry.format, and assuming the Fortran "
            "default would silently pick an equilibrium the deck never named.",
        )
    scheme = int(first_config_value(scheme_value))
    equilibrium_path: Path | None = None
    # The analytic models are closed-form at every radius; a file only carries
    # the surfaces it stores, and the profile stencil must stay inside them.
    r_n_bounds = (0.0, 1.0)
    if scheme in _ANALYTIC_GEOMETRY_FILES:
        geometry_file, psi_a_hat, a_hat, psi_n = _analytic_geometry(nml, scheme)
        geometry_format = "analytic"
    elif scheme in {5, 11, 12}:
        (
            equilibrium_path,
            geometry_format,
            psi_a_hat,
            a_hat,
            psi_n,
            r_n_bounds,
        ) = _file_geometry(nml, scheme)
        geometry_file = _portable_equilibrium_path(equilibrium_path, geometry_base_dir)
    elif scheme == 13:
        _refuse(
            "geometryParameters",
            "geometryScheme",
            scheme,
            "1, 2, 3, 4 (analytic), 5 (VMEC), or 11/12 (Boozer .bc)",
            "geometryScheme=13 carries the Boozer |B| spectrum inline as "
            "boozer_bmnc(m,n)/boozer_bmns(m,n) namelist entries. geometry.format "
            "names a file ('vmec', 'boozer') or a built-in analytic model; there is "
            "no spectrum member, and geometry.file cannot hold a Fourier table. "
            "Write the spectrum to a Boozer .bc file first.",
        )
        raise AssertionError("unreachable")  # pragma: no cover
    else:
        _refuse(
            "geometryParameters",
            "geometryScheme",
            scheme,
            "1, 2, 3, 4 (analytic), 5 (VMEC), or 11/12 (Boozer .bc)",
            "geometry.format has three members (analytic, vmec, boozer) and no case "
            "field can name this equilibrium family.",
        )
        raise AssertionError("unreachable")  # pragma: no cover

    r_n = math.sqrt(float(psi_n))
    if not 0.0 < r_n <= 1.0:
        _refuse(
            "geometryParameters",
            "rN_wish",
            r_n,
            "a normalized radius in (0, 1]",
            "geometry.surfaces holds normalized toroidal flux in [0, 1], and the "
            "magnetic axis has a singular radial Jacobian.",
        )
    if not r_n_bounds[0] * (1.0 - 1.0e-9) <= r_n <= r_n_bounds[1] * (1.0 + 1.0e-9):
        _refuse(
            "geometryParameters",
            "rN_wish",
            r_n,
            f"a normalized radius inside the equilibrium's stored range "
            f"[{r_n_bounds[0]:.6g}, {r_n_bounds[1]:.6g}]",
            "The requested surface is outside the radial span this equilibrium "
            "file carries, so neither the deck nor the converted case has a "
            "geometry to interpolate there.",
        )
    radial = RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    # --- species: dimensionless deck ratios -> SI profiles ------------------
    z_s = _deck_vector(nml, "speciesParameters", "Zs") or (1.0,)
    n_species = len(z_s)
    m_hats = _species_vector(nml, "mHats", default=1.0, count=n_species)
    n_hats = _species_vector(nml, "nHats", default=1.0, count=n_species)
    t_hats = _species_vector(nml, "THats", default=1.0, count=n_species)

    gradient_coordinate = infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params,
        species_params=nml.group("speciesParameters"),
        default=4,
    )
    if gradient_coordinate not in _SPECIES_GRADIENT_SUFFIX:
        _refuse(
            "geometryParameters",
            "inputRadialCoordinateForGradients",
            gradient_coordinate,
            "0, 1, 2, 3, or 4",
            "radialCoordinates.F90 defines only these five gradient coordinates.",
        )
    suffix = _SPECIES_GRADIENT_SUFFIX[gradient_coordinate]
    dn_in = _species_vector(
        nml, f"dNHatd{suffix}", default=0.0, count=n_species, broadcast=True
    )
    dt_in = _species_vector(
        nml, f"dTHatd{suffix}", default=0.0, count=n_species, broadcast=True
    )
    # radialCoordinates.F90 lines 167-238, through the shared helpers: into
    # d/dpsiHat (what the deck means) and back out to d/drHat (what the native
    # profile carries).
    dn_dr_hat = tuple(
        radial.d_dpsi_hat_to_d_dr_hat
        * radial.to_d_dpsi_hat(value, coordinate=gradient_coordinate)
        for value in dn_in
    )
    dt_dr_hat = tuple(
        radial.d_dpsi_hat_to_d_dr_hat
        * radial.to_d_dpsi_hat(value, coordinate=gradient_coordinate)
        for value in dt_in
    )

    half_width = _stencil_half_width(
        r_n=r_n,
        a_hat=a_hat,
        r_n_bounds=r_n_bounds,
        profiles=((n_hats, dn_dr_hat), (t_hats, dt_dr_hat)),
    )
    if half_width < _MIN_STENCIL_HALF_WIDTH_R_N:
        _refuse(
            "speciesParameters",
            f"dNHatd{suffix}",
            list(dn_in),
            "gradients gentle enough for a resolvable radial profile",
            "A case states a profile and dkx.execution differentiates it. These "
            "gradients drive the density or temperature through zero within "
            f"{_MIN_STENCIL_HALF_WIDTH_R_N} in rN of the requested surface, so no "
            "profile with positive values everywhere carries them.",
        )
    # Centred when there is room above the deck's surface, one-sided when there
    # is not (an edge surface, or a file whose outermost stored surface is the
    # requested one). Either way the spacing is uniform and the profile linear,
    # which is what makes numpy.gradient exact at the deck's node.
    offsets = (
        (-half_width, 0.0, half_width)
        if r_n_bounds[1] - r_n >= half_width
        else (-half_width, -half_width / 2.0, 0.0)
    )
    surfaces = tuple((r_n + offset) ** 2 for offset in offsets)

    taken: set[str] = set()
    species = []
    for index in range(n_species):
        mass_amu = mass_amu_from_m_hat(m_hats[index])
        species.append(
            {
                "name": _species_name(
                    charge=z_s[index], mass_amu=mass_amu, index=index, taken=taken
                ),
                "charge": z_s[index],
                "mass_amu": mass_amu,
                "density_m3": [
                    density_m3_from_n_hat(
                        n_hats[index] + dn_dr_hat[index] * a_hat * offset
                    )
                    for offset in offsets
                ],
                "temperature_keV": [
                    temperature_kev_from_t_hat(
                        t_hats[index] + dt_dr_hat[index] * a_hat * offset
                    )
                    for offset in offsets
                ],
            }
        )

    # --- physics, field, resolution, solver --------------------------------
    collision_operator = config_int(nml, ("physicsParameters",), "collisionOperator", 0)
    n_xi = config_int(nml, ("resolutionParameters",), "Nxi", 16)
    er_hat = _electric_field_hat(nml)
    _refuse_unconvertible_physics(
        nml,
        collision_operator=collision_operator,
        n_xi=n_xi,
        field_is_zero=(er_hat == 0.0),
    )

    ambipolar = bool(config_bool(nml, ("general",), "ambipolarSolve", False))
    if ambipolar:
        er_min = config_float(nml, ("general",), "Er_min", -100.0)
        er_max = config_float(nml, ("general",), "Er_max", 100.0)
        if er_min >= er_max:
            _refuse(
                "general",
                "Er_min",
                er_min,
                "a value below Er_max",
                "electric_field.search_kV_m is an increasing bracket.",
            )
        electric_field: dict[str, Any] = {
            "mode": "ambipolar",
            "search_kV_m": [
                electric_field_kv_m_from_er_hat(er_min),
                electric_field_kv_m_from_er_hat(er_max),
            ],
            "search_points": max(
                3, config_int(nml, ("general",), "NEr_ambipolarSolve", 20)
            ),
            "root_tolerance_kV_m": abs(
                config_float(nml, ("general",), "Er_search_tolerance_dx", 1.0e-8)
            )
            or 1.0e-8,
        }
    else:
        electric_field = {
            "mode": "prescribed",
            "value_kV_m": electric_field_kv_m_from_er_hat(er_hat),
        }

    solver_tolerance = config_float(
        nml, ("resolutionParameters",), "solverTolerance", 1.0e-6
    )
    if not 0.0 < solver_tolerance <= 1.0:
        _refuse(
            "resolutionParameters",
            "solverTolerance",
            solver_tolerance,
            "a relative tolerance in (0, 1]",
            "solver.relative_tolerance is a relative residual tolerance.",
        )

    case_name = name or _case_name_from_source(path)
    mapping = {
        "schema": 1,
        "name": case_name,
        "run": {
            "workflow": "ambipolar_profile" if ambipolar else "profile",
            "precision": "float64",
            "device": "auto",
            "progress": True,
        },
        "geometry": {
            "format": geometry_format,
            "file": str(geometry_file),
            "surfaces": list(surfaces),
        },
        "species": species,
        "physics": {
            "model": "full_local",
            "collisions": _COLLISION_OPERATORS[collision_operator],
            # useDKESExBDrift/includeXDotTerm/includeElectricFieldTermInXiDot are
            # guarded above; magneticDriftScheme=0 plus those switches is exactly
            # what 'dkes' names.
            "magnetic_drifts": "dkes",
            "phi1": "off",
        },
        "electric_field": electric_field,
        "resolution": {
            "theta": config_int(nml, ("resolutionParameters",), "Ntheta", 15),
            "zeta": config_int(nml, ("resolutionParameters",), "Nzeta", 15),
            "pitch": n_xi,
            "speed": config_int(nml, ("resolutionParameters",), "Nx", 5),
            "pitch_speed_ramp": config_int(
                nml, ("otherNumericalParameters",), "Nxi_for_x_option", 1
            ),
        },
        "solver": {
            "method": "auto",
            "relative_tolerance": solver_tolerance,
            "memory_fraction": 0.75,
            "reuse": "auto",
        },
        "output": {"file": f"{case_name}.nc", "plots": True},
    }
    return Case.from_mapping(mapping)


def _case_name_from_source(path: Path) -> str:
    """A case name derived from the deck's filename."""
    stem = path.name
    for suffix in (".namelist", ".nml", ".input"):
        while stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    if not stem or stem.lower() == "input":
        stem = path.parent.name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "sfincs_case"


# --- case serialization -----------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string literal is a TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} to TOML")


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _emit_toml(
    data: Mapping[str, Any], *, prefix: tuple[str, ...], lines: list[str]
) -> None:
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in data.items():
        if value is None or isinstance(value, Mapping) or _is_table_array(value):
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append("")
            _emit_toml(value, prefix=(*prefix, key), lines=lines)
        elif _is_table_array(value):
            for item in value:
                lines.append("")
                lines.append(f"[[{'.'.join((*prefix, key))}]]")
                for item_key, item_value in item.items():
                    if item_value is None:
                        continue
                    lines.append(f"{item_key} = {_toml_value(item_value)}")


def _without_none(value: Any) -> Any:
    """Drop absent optional fields.

    TOML has no null, and the case parser rejects an explicit ``None`` where it
    expects a number, so both serializations must omit rather than spell them.
    """
    if isinstance(value, Mapping):
        return {
            key: _without_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_without_none(item) for item in value]
    return value


def write_case_file(case: Any, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Write a ``Case`` to ``destination``; the extension picks the format.

    Raises:
        dkx.config.CaseValidationError: the extension names neither format, or
            the file exists and ``overwrite`` is false.
    """
    from .config import CaseValidationError  # noqa: PLC0415

    target = Path(destination).expanduser()
    suffix = target.suffix.lower()
    if suffix not in {".toml", ".json"}:
        raise CaseValidationError(
            "$destination",
            target.name,
            "a .toml or .json case file",
            "TOML is the human-authored format and JSON the machine-authored one; "
            "name the destination with the extension you want.",
        )
    if target.exists() and not overwrite:
        raise CaseValidationError(
            "$destination",
            str(target),
            "a path that does not exist yet",
            "Choose another destination, or pass --force to overwrite it.",
        )
    data = _without_none(case.to_dict())
    if suffix == ".toml":
        lines: list[str] = ["# DKX case converted from a SFINCS input.namelist."]
        _emit_toml(data, prefix=(), lines=lines)
        text = "\n".join(lines) + "\n"
    else:
        text = json.dumps(data, indent=2) + "\n"
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def convert_sfincs_namelist(
    source: str | Path,
    destination: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> tuple[Any, Path]:
    """Convert a SFINCS ``input.namelist`` into a native case file.

    Returns:
        ``(case, path)`` -- the validated ``Case`` and where it was written.
    """
    from .config import CaseValidationError  # noqa: PLC0415

    target = Path(destination).expanduser()
    suffix = target.suffix.lower()
    if suffix not in {".toml", ".json"}:
        # Checked before the (possibly expensive) conversion so a mistyped
        # destination fails immediately rather than after reading an equilibrium.
        raise CaseValidationError(
            "$destination",
            target.name,
            "a .toml or .json case file",
            "TOML is the human-authored format and JSON the machine-authored one; "
            "name the destination with the extension you want.",
        )
    case = case_from_sfincs_namelist(
        source, name=name, geometry_base_dir=target.parent if target.parent else None
    )
    return case, write_case_file(case, target, overwrite=overwrite)
