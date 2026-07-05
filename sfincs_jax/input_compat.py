from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from .namelist import Namelist, read_sfincs_input
from .paths import resolve_existing_path


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
    repo_root = Path(__file__).resolve().parents[1]
    extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
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
    for both SFINCS_JAX and SFINCS Fortran v3 comparisons.
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
