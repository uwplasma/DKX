from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from .namelist import Namelist


def _group_get(group: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = group.get(key.upper(), None)
        if value is not None:
            return value
    return None


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
