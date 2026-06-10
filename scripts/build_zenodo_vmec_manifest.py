#!/usr/bin/env python3
"""Build a compact inventory of Zenodo VMEC/SFINCS benchmark inputs.

The QS-optimization Zenodo bundle contains many SFINCS v3 radial-surface
directories.  This script records only small metadata needed to plan and audit
SFINCS-JAX parity campaigns, avoiding large equilibrium/output blobs in git.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sfincs_jax.namelist import Namelist, read_sfincs_input

DEFAULT_ZENODO_ROOT = Path(
    os.environ.get(
        "SFINCS_JAX_QS_ZENODO_ROOT",
        "/Users/rogeriojorge/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current",
    )
)

SURFACE_RE = re.compile(r"^(?:psiN|rN|s|rho)[_=]?(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)$")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _first_value(nml: Namelist, key: str, default: Any = None) -> Any:
    """Return the first namelist value with uppercase key across all groups."""
    key = key.upper()
    for group in nml.groups.values():
        if key in group:
            value = group[key]
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value
    return default


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        return text[1:-1]
    return text


def _surface_from_path(path: Path) -> float | None:
    for part in reversed(path.parts):
        match = SURFACE_RE.match(part)
        if match:
            return float(match.group("value"))
    return None


def _case_parts(rel_input: Path) -> tuple[str, str, str]:
    """Return campaign/case/surface labels from a relative input path."""
    parts = rel_input.parts
    surface_label = rel_input.parent.name
    case_label = rel_input.parent.parent.name if len(parts) >= 3 else rel_input.parent.name
    campaign_label = rel_input.parent.parent.parent.name if len(parts) >= 4 else ""
    return campaign_label, case_label, surface_label


def _case_family(text: str) -> str:
    lower = text.lower()
    if "w7x" in lower or "w7-x" in lower:
        return "w7x"
    if "lhd" in lower:
        return "lhd"
    if "qh" in lower or "helical" in lower:
        return "qh"
    if "qi" in lower or "isodynamic" in lower:
        return "qi"
    if "qa" in lower or "axisym" in lower:
        return "qa"
    return "unknown"


def _h5_scalar(data: h5py.File, key: str) -> Any:
    if key not in data:
        return None
    value = data[key][...]
    if np.asarray(value).size != 1:
        return None
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item


def _h5_summary(path: Path, *, include_keys: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {"exists": path.exists()}
    if not path.exists():
        return summary
    summary["size_bytes"] = path.stat().st_size
    with h5py.File(path, "r") as data:
        keys: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                keys.append(name)

        data.visititems(visit)
        summary["dataset_count"] = len(keys)
        if include_keys:
            summary["datasets"] = sorted(keys)
        for key in ("FSABjHat", "FSABjHatOverRootFSAB2", "NIterations", "residual_norm", "RHSMode"):
            value = _h5_scalar(data, key)
            if value is not None:
                summary[key] = value
    return summary


def _manifest_entry(root: Path, input_path: Path, *, include_h5_keys: bool) -> dict[str, Any]:
    rel_input = input_path.relative_to(root)
    nml = read_sfincs_input(input_path)
    campaign_label, case_label, surface_label = _case_parts(rel_input)
    equilibrium_file = _normalize_string(_first_value(nml, "equilibriumFile"))
    surface_s = _to_float(_first_value(nml, "psiN_wish", _surface_from_path(input_path)))
    fortran_output = input_path.parent / "sfincsOutput.h5"

    ntheta = _to_int(_first_value(nml, "Ntheta"))
    nzeta = _to_int(_first_value(nml, "Nzeta"))
    nxi = _to_int(_first_value(nml, "Nxi"))
    nx = _to_int(_first_value(nml, "Nx"))
    resolution = {
        "Ntheta": ntheta,
        "Nzeta": nzeta,
        "Nxi": nxi,
        "Nx": nx,
        "label": "x".join(str(v) if v is not None else "?" for v in (ntheta, nzeta, nxi, nx)),
    }

    family_text = " ".join(str(v) for v in (rel_input, equilibrium_file, case_label) if v is not None)
    return {
        "input": str(rel_input),
        "campaign": campaign_label,
        "case": case_label,
        "surface_label": surface_label,
        "surface_s": surface_s,
        "family": _case_family(family_text),
        "rhs_mode": _to_int(_first_value(nml, "RHSMode", 1)),
        "geometry_scheme": _to_int(_first_value(nml, "geometryScheme")),
        "input_radial_coordinate": _to_int(_first_value(nml, "inputRadialCoordinate")),
        "collision_operator": _to_int(_first_value(nml, "collisionOperator")),
        "include_phi1": bool(_first_value(nml, "includePhi1", False)),
        "include_x_dot_term": bool(_first_value(nml, "includeXDotTerm", False)),
        "include_xi_dot_electric_field_term": bool(_first_value(nml, "includeElectricFieldTermInXiDot", False)),
        "use_dkes_exb_drift": bool(_first_value(nml, "useDKESExBDrift", False)),
        "solver_tolerance": _to_float(_first_value(nml, "solverTolerance")),
        "nu_n": _to_float(_first_value(nml, "nu_n")),
        "er": _to_float(_first_value(nml, "Er")),
        "equilibrium_file": equilibrium_file,
        "resolution": resolution,
        "fortran_output": str(fortran_output.relative_to(root)),
        "fortran_output_summary": _h5_summary(fortran_output, include_keys=include_h5_keys),
    }


def build_zenodo_vmec_manifest(root: Path, *, include_h5_keys: bool = False) -> dict[str, Any]:
    """Return a compact JSON-ready manifest for all Zenodo SFINCS inputs."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Zenodo root not found: {root}")

    entries = [_manifest_entry(root, path, include_h5_keys=include_h5_keys) for path in sorted(root.rglob("input.namelist"))]
    by_family = Counter(entry["family"] for entry in entries)
    by_rhs_mode = Counter(str(entry["rhs_mode"]) for entry in entries)
    by_resolution = Counter(entry["resolution"]["label"] for entry in entries)
    with_fortran_output = sum(bool(entry["fortran_output_summary"]["exists"]) for entry in entries)

    return {
        "schema_version": 1,
        "zenodo_root": str(root),
        "input_count": len(entries),
        "with_fortran_output_count": with_fortran_output,
        "family_counts": dict(sorted(by_family.items())),
        "rhs_mode_counts": dict(sorted(by_rhs_mode.items())),
        "resolution_counts": dict(sorted(by_resolution.items())),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo-root", type=Path, default=DEFAULT_ZENODO_ROOT)
    parser.add_argument("--out", type=Path, required=True, help="JSON manifest path to write")
    parser.add_argument("--include-h5-keys", action="store_true", help="Include all HDF5 dataset names in each output summary")
    args = parser.parse_args(argv)

    manifest = build_zenodo_vmec_manifest(args.zenodo_root, include_h5_keys=args.include_h5_keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(
        "Wrote Zenodo VMEC manifest: "
        f"{args.out} ({manifest['input_count']} inputs, {manifest['with_fortran_output_count']} with Fortran H5)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
