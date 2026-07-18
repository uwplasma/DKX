"""Optional adapters for JAX-native geometry producers.

The functions in this module intentionally avoid importing ``vmex`` or
``booz_xform_jax`` at module import time. They operate on structural "wout-like"
objects so users can pass objects produced by optional differentiable geometry
pipelines without making those packages mandatory dependencies of ``dkx``.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from dkx.magnetic_geometry import VmecWout


GEOMETRY_PROXY_WORKFLOW = "vmex_to_boozer_sfincs_geometry_proxy"
GEOMETRY_PROXY_CONTRACT_VERSION = 1
KINETIC_TRANSPORT_SCALAR_CONTRACT_VERSION = 1

_DIFFERENTIABILITY_LABELS = {
    "differentiated": "covered by JAX autodiff in this geometry-proxy graph",
    "differentiated_for_geometry_proxy_gate": (
        "differentiated only for the Boozer-spectrum proxy objective and its "
        "finite-difference gradient gate"
    ),
    "setup_only_not_differentiated": (
        "allowed as setup/provenance, but not included in the differentiated graph"
    ),
    "not_claimed_not_covered_by_this_lane": (
        "explicitly outside this public lane; do not present it as a gradient claim"
    ),
}

_DIFFERENTIATED_GRAPH = [
    "scaled VMEC-like spectral arrays",
    "booz_xform_jax",
    "dkx Boozer-spectrum proxy objective",
]

_OUTSIDE_DIFFERENTIATED_GRAPH = [
    "VMEC file I/O",
    "vmex example fixed-boundary setup",
    "dkx VMEC file adapters",
    "SFINCS kinetic transport solve",
]

_FORBIDDEN_GRADIENT_CLAIM = "full VMEC-boundary-to-SFINCS kinetic transport gradients"

_KINETIC_TRANSPORT_SCALAR_REQUIRED_STAGES = (
    "vmec_source",
    "vmec_equilibrium_or_wout",
    "boozer_transform",
    "sfincs_geometry_adapter",
    "kinetic_operator_assembly",
    "linear_kinetic_solve",
    "transport_scalar_reduction",
    "gradient_validation",
)


def _kinetic_transport_scalar_required_stages() -> list[dict[str, Any]]:
    """Return the staged contract for a future kinetic transport scalar.

    The stages are intentionally more detailed than the current proxy workflow.
    They make the missing pieces explicit so examples and docs can describe
    current differentiability without implying a full kinetic-transport gradient.
    """
    return [
        {
            "name": "vmec_source",
            "component": "VMEC boundary, VMEC input, or VMEC wout provenance",
            "current_public_role": "setup_or_file_provenance",
            "differentiability_boundary": "setup_only_not_differentiated",
            "current_status": "available_for_provenance",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["source identifier", "radial coordinate convention"],
        },
        {
            "name": "vmec_equilibrium_or_wout",
            "component": "vmex equilibrium solve or file-backed VMEC wout",
            "current_public_role": "setup_or_optional_in_memory_provenance",
            "differentiability_boundary": "setup_only_not_differentiated",
            "current_status": "optional_setup_not_default_ci",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["wout-like field arrays", "surface selection provenance"],
        },
        {
            "name": "boozer_transform",
            "component": "booz_xform_jax Boozer-spectrum transform",
            "current_public_role": "differentiated_geometry_proxy_stage",
            "differentiability_boundary": "differentiated_for_geometry_proxy_gate",
            "current_status": "optional_proxy_gate",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["mboz/nboz", "selected surface", "Boozer spectrum"],
        },
        {
            "name": "sfincs_geometry_adapter",
            "component": "dkx geometry objects used by kinetic operators",
            "current_public_role": "setup_only_adapter",
            "differentiability_boundary": "setup_only_not_differentiated",
            "current_status": "file_backed_adapter_available",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["geometry scheme", "theta/zeta grid", "normalization audit"],
        },
        {
            "name": "kinetic_operator_assembly",
            "component": "drift-kinetic matrix/operator assembly",
            "current_public_role": "not_run_by_vmec_boozer_proxy_lane",
            "differentiability_boundary": "not_claimed_not_covered_by_this_lane",
            "current_status": "deferred_for_full_kinetic_scalar",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["operator residual gate", "memory/runtime budget", "CPU/GPU parity"],
        },
        {
            "name": "linear_kinetic_solve",
            "component": "SFINCS kinetic linear solve and constraints",
            "current_public_role": "not_run_by_vmec_boozer_proxy_lane",
            "differentiability_boundary": "not_claimed_not_covered_by_this_lane",
            "current_status": "deferred_for_full_kinetic_scalar",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["solver residual history", "fallback/provenance log", "parity gate"],
        },
        {
            "name": "transport_scalar_reduction",
            "component": "transport coefficient, current, flux, or ambipolar scalar",
            "current_public_role": "proxy_scalar_only",
            "differentiability_boundary": "not_claimed_not_covered_by_this_lane",
            "current_status": "proxy_available_kinetic_scalar_deferred",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["observable normalization", "units", "comparison target"],
        },
        {
            "name": "gradient_validation",
            "component": "JVP/VJP, finite-difference, and physics-gate validation",
            "current_public_role": "proxy_gradient_validation_only",
            "differentiability_boundary": "not_claimed_not_covered_by_this_lane",
            "current_status": "proxy_gate_available_kinetic_gate_deferred",
            "required_for_future_kinetic_scalar": True,
            "required_evidence": ["finite-difference check", "shape derivative provenance", "physics gate"],
        },
    ]


def kinetic_transport_scalar_no_overclaim_gate(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate that the VMEC/Boozer contract does not overclaim kinetic gradients.

    This pure-Python gate is deliberately cheap enough for default CI. It checks
    schema completeness and verifies that the current public scalar remains a
    Boozer-spectrum proxy unless all future kinetic stages have explicit
    promotion evidence.
    """
    stages = list(contract.get("required_stages") or [])
    stage_names = [str(stage.get("name")) for stage in stages]
    missing_stages = [
        name for name in _KINETIC_TRANSPORT_SCALAR_REQUIRED_STAGES if name not in stage_names
    ]
    duplicate_stages = sorted(
        {name for name in stage_names if stage_names.count(name) > 1}
    )
    schema_errors: list[str] = []
    for stage in stages:
        for key in (
            "name",
            "component",
            "current_public_role",
            "differentiability_boundary",
            "current_status",
            "required_for_future_kinetic_scalar",
            "required_evidence",
        ):
            if key not in stage:
                schema_errors.append(f"{stage.get('name', '<unnamed>')} missing {key}")
        if not stage.get("required_evidence"):
            schema_errors.append(f"{stage.get('name', '<unnamed>')} has no required_evidence")

    current = dict(contract.get("current_public_scalar") or {})
    differentiated = set(current.get("differentiated_stage_names") or [])
    setup_only = set(current.get("setup_only_stage_names") or [])
    not_covered = set(current.get("not_covered_stage_names") or [])
    boundary_overlap = sorted(differentiated & (setup_only | not_covered))
    forbidden_claims = []
    if bool(current.get("kinetic_transport_scalar_claimed", True)):
        forbidden_claims.append("kinetic_transport_scalar_claimed")
    if bool(current.get("kinetic_solve_executed", True)):
        forbidden_claims.append("kinetic_solve_executed")
    if str(current.get("scalar_kind")) != "boozer_spectrum_proxy_not_kinetic":
        forbidden_claims.append("scalar_kind")

    ci_policy = dict(contract.get("ci_dependency_policy") or {})
    ci_violations = [
        key
        for key in (
            "default_ci_requires_vmex",
            "default_ci_requires_booz_xform_jax",
        )
        if bool(ci_policy.get(key, True))
    ]

    promotion = dict(contract.get("promotion_requirements") or {})
    promoted = bool(promotion.get("full_kinetic_scalar_promoted", False))
    unvalidated_future_stages = [
        str(stage.get("name"))
        for stage in stages
        if str(stage.get("current_status", "")).startswith("deferred")
        or str(stage.get("current_status", "")).endswith("_deferred")
    ]
    promotion_violations = unvalidated_future_stages if promoted else []
    passed = not (
        missing_stages
        or duplicate_stages
        or schema_errors
        or boundary_overlap
        or forbidden_claims
        or ci_violations
        or promotion_violations
    )
    return {
        "status": "pass" if passed else "fail",
        "claim_scope": "proxy_scalar_only_until_full_kinetic_contract_is_promoted",
        "required_stage_count": len(_KINETIC_TRANSPORT_SCALAR_REQUIRED_STAGES),
        "present_stage_count": len(stages),
        "missing_required_stages": missing_stages,
        "duplicate_stages": duplicate_stages,
        "schema_errors": schema_errors,
        "boundary_overlap": boundary_overlap,
        "forbidden_claims": forbidden_claims,
        "ci_dependency_violations": ci_violations,
        "promotion_violations": promotion_violations,
        "kinetic_solve_executed": bool(current.get("kinetic_solve_executed", True)),
        "kinetic_transport_scalar_claimed": bool(
            current.get("kinetic_transport_scalar_claimed", True)
        ),
    }


def vmec_boozer_kinetic_transport_scalar_contract(
    *,
    backend_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return the contract for a future VMEC/Boozer-to-kinetic scalar.

    The current public implementation only differentiates a Boozer-spectrum
    proxy scalar. This contract records the required stages for promoting a true
    kinetic transport scalar later, without importing optional geometry packages
    or changing default CI dependencies.
    """
    status = optional_jax_geometry_backend_status() if backend_status is None else dict(backend_status)
    contract: dict[str, Any] = {
        "workflow": GEOMETRY_PROXY_WORKFLOW,
        "contract_version": KINETIC_TRANSPORT_SCALAR_CONTRACT_VERSION,
        "scalar_target": "future_vmec_boozer_to_sfincs_kinetic_transport_scalar",
        "ci_dependency_policy": {
            "default_ci_requires_vmex": False,
            "default_ci_requires_booz_xform_jax": False,
            "backend_check_imports_optional_packages": False,
        },
        "optional_backends": {
            "vmex": bool(status.get("vmex", False)),
            "booz_xform_jax": bool(status.get("booz_xform_jax", False)),
        },
        "required_stages": _kinetic_transport_scalar_required_stages(),
        "current_public_scalar": {
            "name": "boozer_spectrum_proxy_transport_objective",
            "scalar_kind": "boozer_spectrum_proxy_not_kinetic",
            "kinetic_transport_scalar_claimed": False,
            "kinetic_solve_executed": False,
            "differentiated_stage_names": [
                "boozer_transform",
                "transport_scalar_reduction",
            ],
            "setup_only_stage_names": [
                "vmec_source",
                "vmec_equilibrium_or_wout",
                "sfincs_geometry_adapter",
            ],
            "not_covered_stage_names": [
                "kinetic_operator_assembly",
                "linear_kinetic_solve",
                "gradient_validation",
            ],
        },
        "promotion_requirements": {
            "full_kinetic_scalar_promoted": False,
            "required_before_promotion": [
                "pure-JAX or explicitly non-differentiable geometry path selected by user",
                "kinetic operator assembly residual and normalization audit",
                "linear solve residual history and CPU/GPU parity gate",
                "finite-difference/JVP/VJP validation for the promoted scalar",
                "physics-gate comparison against documented SFINCS or literature targets",
            ],
        },
    }
    contract["no_overclaim_gate"] = kinetic_transport_scalar_no_overclaim_gate(contract)
    return contract


def optional_jax_geometry_backend_status() -> dict[str, bool]:
    """Return whether optional JAX geometry backends are importable.

    The check is deliberately shallow: it reports importability without importing
    the packages, initializing devices, or changing JAX configuration.  This keeps
    CLI startup and normal CI independent of optional differentiable-geometry
    packages.
    """
    return {
        "vmex": find_spec("vmex") is not None,
        "booz_xform_jax": find_spec("booz_xform_jax") is not None,
    }


def geometry_proxy_workflow_contract(
    *,
    backend_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return the public contract for the optional JAX geometry-proxy lane.

    This is the machine-readable version of the prose in the docs and example
    CLI.  It is intentionally conservative: default CI must be able to evaluate
    the contract without importing optional geometry packages, and the only
    supported gradient claim is the Boozer-spectrum proxy gate.
    """
    status = optional_jax_geometry_backend_status() if backend_status is None else dict(backend_status)
    kinetic_scalar_contract = vmec_boozer_kinetic_transport_scalar_contract(
        backend_status=status
    )
    return {
        "workflow": GEOMETRY_PROXY_WORKFLOW,
        "contract_version": GEOMETRY_PROXY_CONTRACT_VERSION,
        "ci_dependency_policy": {
            "default_ci_requires_vmex": False,
            "default_ci_requires_booz_xform_jax": False,
            "backend_check_imports_optional_packages": False,
            "optional_integration_fixture_env": "DKX_VMEX_WOUT",
        },
        "optional_backends": {
            "vmex": bool(status.get("vmex", False)),
            "booz_xform_jax": bool(status.get("booz_xform_jax", False)),
        },
        "differentiability_labels": dict(_DIFFERENTIABILITY_LABELS),
        "differentiated_graph": list(_DIFFERENTIATED_GRAPH),
        "outside_differentiated_graph": list(_OUTSIDE_DIFFERENTIATED_GRAPH),
        "kinetic_transport_scalar_contract": kinetic_scalar_contract,
        "deferred_work": [
            "VMEC boundary-shape gradients through a production equilibrium solve",
            "pure-JAX scheme-5 VMEC geometry evaluation for kinetic operators",
            "SFINCS kinetic transport objective gradients through radial profile scans",
        ],
        "no_overclaim_gate": {
            "status": "pass",
            "claim_scope": "geometry_proxy_gradient_only",
            "full_transport_gradients_claimed": False,
            "forbidden_gradient_claim": _FORBIDDEN_GRADIENT_CLAIM,
            "kinetic_gradient_status": "deferred_not_covered_by_this_lane",
            "kinetic_transport_scalar_contract_gate": kinetic_scalar_contract[
                "no_overclaim_gate"
            ],
        },
    }


def optional_jax_geometry_backend_report() -> dict[str, Any]:
    """Return user-facing metadata for the optional JAX geometry lane.

    This is intentionally static apart from shallow backend importability checks.
    The report clarifies the differentiability boundary without importing optional
    packages or claiming end-to-end VMEC-boundary-to-transport gradients.
    """
    status = optional_jax_geometry_backend_status()
    contract = geometry_proxy_workflow_contract(backend_status=status)
    return {
        "backends": status,
        "workflow_contract": contract,
        "kinetic_transport_scalar_contract": contract["kinetic_transport_scalar_contract"],
        "runnable_paths": {
            "no_optional_dependencies": "--check-backends",
            "file_backed_setup": "--wout /path/to/wout.nc",
            "optional_in_memory_setup": "--vmec-case circular_tokamak",
        },
        "gradient_availability": {
            "spectral_scale_to_boozer_proxy": "available_when_optional_backends_installed",
            "vmec_file_io": "setup_only_not_differentiated",
            "vmec_fixed_boundary_solve": "setup_only_not_differentiated",
            "sfincs_vmec_file_adapter": "setup_only_not_differentiated",
            "sfincs_kinetic_transport_solve": "not_covered_by_this_lane",
        },
        "differentiability_labels": contract["differentiability_labels"],
        "differentiated_graph": contract["differentiated_graph"],
        "outside_differentiated_graph": contract["outside_differentiated_graph"],
        "no_overclaim_gate": contract["no_overclaim_gate"],
        "claim": "geometry_proxy_gradient_gate_not_full_transport_gradient",
    }


def geometry_proxy_workflow_summary(
    *,
    provenance: str | None = None,
    requested_surface: float | None = None,
    selected_surface: float | None = None,
    boozer_resolution: dict[str, int] | None = None,
    grid_shape: dict[str, int] | None = None,
    scale: float | None = None,
    proxy_objective: float | None = None,
    autodiff_gradient: float | None = None,
    finite_difference_gradient: float | None = None,
    finite_difference_step: float | None = None,
    gradient_rtol: float = 5.0e-3,
    gradient_atol: float = 1.0e-7,
    backend_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a reusable provenance summary for the geometry-proxy workflow.

    The summary is intentionally conservative: it records the stages that are
    differentiable in the public vmex/booz_xform_jax handoff, the optional
    packages needed to run that handoff, and a numerical gradient gate for the
    scalar geometry proxy when both autodiff and finite-difference gradients are
    supplied.  It never claims gradients through the SFINCS kinetic transport
    solve.
    """
    status = optional_jax_geometry_backend_status() if backend_status is None else dict(backend_status)
    contract = geometry_proxy_workflow_contract(backend_status=status)
    gate: dict[str, Any] = {
        "status": "not_run",
        "autodiff_gradient": None,
        "finite_difference_gradient": None,
        "finite_difference_step": finite_difference_step,
        "absolute_error": None,
        "rtol": float(gradient_rtol),
        "atol": float(gradient_atol),
        "claim": "geometry_proxy_gradient_gate_only",
    }
    if autodiff_gradient is not None and finite_difference_gradient is not None:
        autodiff_value = float(autodiff_gradient)
        fd_value = float(finite_difference_gradient)
        abs_error = abs(autodiff_value - fd_value)
        tolerance = float(gradient_atol) + float(gradient_rtol) * abs(fd_value)
        gate.update(
            {
                "status": "pass" if abs_error <= tolerance else "fail",
                "autodiff_gradient": autodiff_value,
                "finite_difference_gradient": fd_value,
                "absolute_error": abs_error,
                "tolerance": tolerance,
            }
        )

    return {
        "workflow": GEOMETRY_PROXY_WORKFLOW,
        "workflow_contract": contract,
        "kinetic_transport_scalar_contract": contract["kinetic_transport_scalar_contract"],
        "provenance": {
            "source": provenance,
            "requested_surface": requested_surface,
            "selected_surface": selected_surface,
            "boozer_resolution": boozer_resolution,
            "grid_shape": grid_shape,
            "scale": scale,
        },
        "required_optional_dependencies": {
            "vmex": {
                "importable": bool(status.get("vmex", False)),
                "used_for": "VMEC-like wout provenance and optional fixed-boundary setup",
            },
            "booz_xform_jax": {
                "importable": bool(status.get("booz_xform_jax", False)),
                "used_for": "Boozer transform in the differentiable geometry-proxy path",
            },
        },
        "differentiability_labels": contract["differentiability_labels"],
        "stages": [
            {
                "name": "vmec_provenance",
                "component": "vmex or VMEC wout input",
                "differentiability": "setup_only_not_differentiated",
            },
            {
                "name": "spectral_scale",
                "component": "in-memory VMEC-like magnetic spectrum",
                "differentiability": "differentiated",
            },
            {
                "name": "boozer_transform",
                "component": "booz_xform_jax",
                "differentiability": "differentiated_for_geometry_proxy_gate",
            },
            {
                "name": "sfincs_geometry_proxy",
                "component": "dkx Boozer-spectrum proxy objective",
                "differentiability": "differentiated",
            },
            {
                "name": "sfincs_kinetic_transport_solve",
                "component": "full kinetic transport solver",
                "differentiability": "not_claimed_not_covered_by_this_lane",
            },
        ],
        "numerical_gradient_gate": gate,
        "no_overclaim_gate": contract["no_overclaim_gate"],
        "results": {
            "proxy_objective": proxy_objective,
        },
        "claims": {
            "differentiable": (
                "scaled spectral arrays -> booz_xform_jax -> dkx "
                "Boozer-spectrum proxy objective"
            ),
            "not_claimed": _FORBIDDEN_GRADIENT_CLAIM,
        },
    }


def geometry_proxy_no_solve_provenance_gate(
    summary: dict[str, Any],
    *,
    require_file_provenance: bool = False,
) -> dict[str, Any]:
    """Validate that a workflow summary remains a no-solve proxy contract.

    The VMEC/Boozer public lane currently differentiates only a Boozer-spectrum
    proxy scalar.  This gate is shared by examples and tests so the no-solve
    provenance requirements cannot drift independently of the scalar contract.
    """
    provenance = dict(summary.get("provenance") or {})
    no_overclaim = dict(summary.get("no_overclaim_gate") or {})
    gradient_gate = dict(summary.get("numerical_gradient_gate") or {})
    kinetic_contract = dict(summary.get("kinetic_transport_scalar_contract") or {})
    kinetic_contract_gate = dict(kinetic_contract.get("no_overclaim_gate") or {})
    current_scalar = dict(kinetic_contract.get("current_public_scalar") or {})
    required_kinetic_stages = [
        str(stage.get("name"))
        for stage in list(kinetic_contract.get("required_stages") or [])
    ]

    required_fields = (
        "source",
        "selected_surface",
        "boozer_resolution",
        "grid_shape",
        "scale",
    )
    present_fields = [
        field
        for field in required_fields
        if provenance.get(field) not in (None, {}, [])
    ]
    missing_fields = [
        field
        for field in required_fields
        if require_file_provenance and provenance.get(field) in (None, {}, [])
    ]

    proxy_status = str(gradient_gate.get("status", "not_run"))
    proxy_ok = (
        proxy_status == "pass"
        if require_file_provenance
        else proxy_status in {"pass", "not_run"}
    )
    kinetic_claimed = bool(no_overclaim.get("full_transport_gradients_claimed", True))
    current_scalar_claimed = bool(
        current_scalar.get("kinetic_transport_scalar_claimed", True)
    )
    kinetic_solve_executed = bool(current_scalar.get("kinetic_solve_executed", True))
    scalar_kind = str(current_scalar.get("scalar_kind", ""))
    current_scalar_ok = (
        not current_scalar_claimed
        and not kinetic_solve_executed
        and scalar_kind == "boozer_spectrum_proxy_not_kinetic"
    )
    kinetic_contract_ok = kinetic_contract_gate.get("status") == "pass"
    passed = (
        (not kinetic_claimed)
        and proxy_ok
        and not missing_fields
        and kinetic_contract_ok
        and current_scalar_ok
    )
    return {
        "status": "pass" if passed else "fail",
        "claim_scope": "no_solve_boozer_spectrum_proxy_gradient",
        "kinetic_solve_executed": False,
        "kinetic_transport_scalar_claimed": current_scalar_claimed,
        "current_public_scalar_kind": scalar_kind,
        "full_vmec_boundary_to_sfincs_kinetic_gradients": "deferred_not_covered_by_this_lane",
        "kinetic_transport_scalar_contract_gate": kinetic_contract_gate,
        "required_kinetic_transport_scalar_stages": required_kinetic_stages,
        "differentiability_boundary": {
            "differentiated_stage_names": list(
                current_scalar.get("differentiated_stage_names") or []
            ),
            "setup_only_stage_names": list(
                current_scalar.get("setup_only_stage_names") or []
            ),
            "not_covered_stage_names": list(
                current_scalar.get("not_covered_stage_names") or []
            ),
        },
        "requires_file_provenance": bool(require_file_provenance),
        "required_file_provenance_fields": list(required_fields),
        "present_file_provenance_fields": present_fields,
        "missing_file_provenance_fields": missing_fields,
        "proxy_gradient_gate_status": proxy_status,
        "proxy_vs_kinetic": {
            "proxy": "differentiated Boozer-spectrum transport-like scalar",
            "kinetic": "not run and not differentiated by this workflow",
        },
    }


def _attr(obj: Any, *names: str, default: Any = None, required: bool = True) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if required:
        joined = ", ".join(names)
        raise AttributeError(f"wout-like object is missing required attribute: {joined}")
    return default


def _as_1d(obj: Any, *names: str, dtype: Any) -> np.ndarray:
    arr = np.asarray(_attr(obj, *names), dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{names[0]} must be 1D, got shape {arr.shape}")
    return arr


def _mode_radius_array(
    value: Any,
    *,
    modes: int,
    radius_options: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Normalize a VMEC coefficient table to internal ``(mode, radius)`` layout."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == int(modes) and arr.shape[1] in radius_options:
        return arr
    if arr.shape[1] == int(modes) and arr.shape[0] in radius_options:
        return arr.T
    raise ValueError(
        f"{name} must have shape (modes, radius) or (radius, modes); "
        f"got {arr.shape}, modes={int(modes)}, radius_options={radius_options}"
    )


def _mode_radius_attr(
    obj: Any,
    name: str,
    *,
    modes: int,
    radius_options: tuple[int, ...],
    default_like: np.ndarray | None = None,
) -> np.ndarray:
    """Read and normalize a coefficient table, optionally zero-filling one table.

    Some in-memory VMEC producers expose a minimal stellarator-symmetric subset.
    For those objects, covariant/contravariant magnetic-field coefficient tables
    that are absent from the producer can be represented as zeros.  Required field
    strength, metric/Jacobian, and shape tables remain strict and raise if absent.
    """
    if hasattr(obj, name):
        value = getattr(obj, name)
    elif default_like is not None:
        value = np.zeros_like(default_like)
    else:
        raise AttributeError(f"wout-like object is missing required attribute: {name}")
    return _mode_radius_array(value, modes=modes, radius_options=radius_options, name=name)


def vmec_wout_from_wout_like(wout_like: Any, *, path: str | Path | None = None) -> VmecWout:
    """Convert a VMEC-like in-memory object into :class:`dkx.magnetic_geometry.VmecWout`.

    This covers the field names used by ``vmex.wout.WoutData`` and by
    :class:`dkx.magnetic_geometry.VmecWout`. Arrays may be stored either as
    ``(radius, mode)`` or ``(mode, radius)``; the returned object always uses the
    ``dkx`` convention ``(mode, radius)``.  The optional ``path`` argument
    is metadata only and is useful when the source object is produced entirely in
    memory.
    """
    ns = int(_attr(wout_like, "ns"))
    mnmax = int(_attr(wout_like, "mnmax", default=np.asarray(_attr(wout_like, "xm")).size, required=False))
    mnmax_nyq = int(
        _attr(wout_like, "mnmax_nyq", default=np.asarray(_attr(wout_like, "xm_nyq")).size, required=False)
    )
    path_value = Path(path) if path is not None else Path(str(_attr(wout_like, "path", default="<in-memory-wout>", required=False)))

    xm = _as_1d(wout_like, "xm", dtype=np.int32)
    xn = _as_1d(wout_like, "xn", dtype=np.int32)
    xm_nyq = _as_1d(wout_like, "xm_nyq", dtype=np.int32)
    xn_nyq = _as_1d(wout_like, "xn_nyq", dtype=np.int32)

    bmnc = _mode_radius_attr(wout_like, "bmnc", modes=mnmax_nyq, radius_options=(ns,))
    gmnc = _mode_radius_attr(wout_like, "gmnc", modes=mnmax_nyq, radius_options=(ns,))
    bsubumnc = _mode_radius_attr(wout_like, "bsubumnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsubvmnc = _mode_radius_attr(wout_like, "bsubvmnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsubsmns = _mode_radius_attr(wout_like, "bsubsmns", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsupumnc = _mode_radius_attr(wout_like, "bsupumnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)
    bsupvmnc = _mode_radius_attr(wout_like, "bsupvmnc", modes=mnmax_nyq, radius_options=(ns,), default_like=gmnc)

    rmnc = _mode_radius_attr(wout_like, "rmnc", modes=mnmax, radius_options=(ns,))
    zmns = _mode_radius_attr(wout_like, "zmns", modes=mnmax, radius_options=(ns,))
    lmns = _mode_radius_attr(wout_like, "lmns", modes=mnmax, radius_options=(ns, ns - 1))

    return VmecWout(
        path=path_value,
        nfp=int(_attr(wout_like, "nfp")),
        ns=ns,
        mpol=int(_attr(wout_like, "mpol")),
        ntor=int(_attr(wout_like, "ntor")),
        mnmax=mnmax,
        mnmax_nyq=mnmax_nyq,
        lasym=bool(_attr(wout_like, "lasym", default=False, required=False)),
        aminor_p=float(_attr(wout_like, "aminor_p", "Aminor_p", default=0.0, required=False)),
        phi=np.asarray(_attr(wout_like, "phi"), dtype=np.float64),
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        bmnc=bmnc,
        gmnc=gmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        iotas=np.asarray(_attr(wout_like, "iotas"), dtype=np.float64),
        presf=np.asarray(_attr(wout_like, "presf", default=np.zeros((ns,), dtype=np.float64), required=False), dtype=np.float64),
    )


def boozer_bhat_from_spectrum(
    theta: Any,
    zeta: Any,
    *,
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    normalize: bool = True,
    eps: float = 1.0e-14,
) -> jnp.ndarray:
    r"""Evaluate normalized magnetic-field strength from a Boozer cosine spectrum.

    Parameters use the public ``booz_xform_jax`` output convention: ``bmnc_b`` is a
    one-dimensional Boozer :math:`|B|` cosine spectrum, while ``ixm_b`` and
    ``ixn_b`` are the corresponding integer mode numbers.  In ``booz_xform_jax``
    output, ``ixn_b`` already includes the field-period factor, so this function
    evaluates :math:`m\theta - n\zeta` directly.

    The helper is intentionally small and pure JAX.  It is used by optional
    vmex/booz_xform_jax workflow gates without making either package a
    required dependency of ``dkx``.
    """
    theta_arr = jnp.asarray(theta)
    zeta_arr = jnp.asarray(zeta)
    coeff = jnp.asarray(bmnc_b)
    m_mode = jnp.asarray(ixm_b)
    n_mode = jnp.asarray(ixn_b)

    if coeff.ndim != 1:
        raise ValueError(f"bmnc_b must be 1D, got shape {coeff.shape}")
    if m_mode.ndim != 1 or n_mode.ndim != 1:
        raise ValueError("ixm_b and ixn_b must be 1D")
    if coeff.shape[0] != m_mode.shape[0] or coeff.shape[0] != n_mode.shape[0]:
        raise ValueError(
            "bmnc_b, ixm_b, and ixn_b must have the same length; "
            f"got {coeff.shape[0]}, {m_mode.shape[0]}, {n_mode.shape[0]}"
        )

    angle = (
        m_mode[:, None, None] * theta_arr[None, :, None]
        - n_mode[:, None, None] * zeta_arr[None, None, :]
    )
    bmod = jnp.sum(coeff[:, None, None] * jnp.cos(angle), axis=0)
    if not normalize:
        return bmod

    b00 = jnp.sum(jnp.where((m_mode == 0) & (n_mode == 0), coeff, 0.0))
    fallback = jnp.mean(jnp.abs(bmod)) + eps
    denom = jnp.where(jnp.abs(b00) > eps, b00, fallback)
    return bmod / denom


def boozer_spectrum_geometry_proxy_objective(
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Return a differentiable scalar proxy from a Boozer :math:`|B|` spectrum.

    This is a geometry/transport proxy, not a kinetic solve.  It measures the
    normalized field-strength variation plus a small angular-roughness penalty,
    giving optional vmex/booz_xform_jax examples a bounded scalar that can be
    differentiated, finite-difference checked, and optimized quickly.
    """
    bhat = boozer_bhat_from_spectrum(
        theta,
        zeta,
        bmnc_b=bmnc_b,
        ixm_b=ixm_b,
        ixn_b=ixn_b,
        normalize=normalize,
    )
    centered = bhat - jnp.mean(bhat)
    theta_slope = jnp.mean((jnp.roll(bhat, -1, axis=0) - bhat) ** 2)
    zeta_slope = jnp.mean((jnp.roll(bhat, -1, axis=1) - bhat) ** 2)
    return jnp.mean(centered**2) + 0.05 * (theta_slope + zeta_slope)


def boozer_spectrum_proxy_transport_objective(
    bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    thermodynamic_forces: Any | None = None,
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Return a differentiable transport-like proxy from a Boozer spectrum.

    This is not a kinetic SFINCS solve.  It maps normalized field-strength
    variation and angular roughness into a tiny symmetric proxy transport matrix,
    then contracts that matrix with thermodynamic forces.  The objective is useful
    as a fast JAX differentiability gate for optional VMEC/Boozer handoffs because
    it exercises a spectrum-to-transport-objective shape without requiring
    ``vmex`` or ``booz_xform_jax`` at import or test time.
    """
    forces = (
        jnp.asarray([1.0, -0.35], dtype=jnp.result_type(jnp.asarray(bmnc_b), jnp.float64))
        if thermodynamic_forces is None
        else jnp.asarray(thermodynamic_forces)
    )
    if forces.shape != (2,):
        raise ValueError(f"thermodynamic_forces must have shape (2,), got {forces.shape}")

    bhat = boozer_bhat_from_spectrum(
        theta,
        zeta,
        bmnc_b=bmnc_b,
        ixm_b=ixm_b,
        ixn_b=ixn_b,
        normalize=normalize,
    )
    centered = bhat - jnp.mean(bhat)
    theta_delta = jnp.roll(bhat, -1, axis=0) - bhat
    zeta_delta = jnp.roll(bhat, -1, axis=1) - bhat
    variance = jnp.mean(centered**2)
    theta_roughness = jnp.mean(theta_delta**2)
    zeta_roughness = jnp.mean(zeta_delta**2)
    cross_coupling = jnp.mean(centered * (theta_delta - zeta_delta))

    transport_proxy = jnp.asarray(
        [
            [variance + 0.10 * theta_roughness, 0.15 * cross_coupling],
            [0.15 * cross_coupling, 0.65 * variance + 0.10 * zeta_roughness],
        ],
        dtype=bhat.dtype,
    )
    return forces @ transport_proxy @ forces


def boozer_spectrum_proxy_transport_gradient_gate(
    *,
    n_theta: int = 16,
    n_zeta: int = 12,
    finite_difference_step: float = 1.0e-5,
    rtol: float = 5.0e-4,
    atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Run a no-optional-dependency gradient gate for the proxy objective."""
    import jax

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(n_theta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, int(n_zeta), endpoint=False, dtype=jnp.float64)
    coeff0 = jnp.asarray([1.0, 0.045, -0.021, 0.012, 0.006, -0.004], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 1, 1, 2, 3, 4], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 0, 1, -1, 2, -2], dtype=jnp.int32)
    forces = jnp.asarray([1.0, -0.35], dtype=jnp.float64)
    direction = jnp.asarray([0.0, 0.40, -0.30, 0.20, -0.10, 0.05], dtype=jnp.float64)
    eps = float(finite_difference_step)

    def objective(coeff: jnp.ndarray) -> jnp.ndarray:
        return boozer_spectrum_proxy_transport_objective(
            coeff,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
            thermodynamic_forces=forces,
        )

    value, gradient = jax.value_and_grad(objective)(coeff0)
    basis = jnp.eye(int(coeff0.size), dtype=coeff0.dtype)
    finite_difference_gradient = jax.vmap(
        lambda basis_vector: (
            objective(coeff0 + eps * basis_vector) - objective(coeff0 - eps * basis_vector)
        )
        / (2.0 * eps)
    )(basis)
    _value, jvp = jax.jvp(objective, (coeff0,), (direction,))
    directional_from_gradient = jnp.vdot(gradient, direction)

    max_abs_error = float(jnp.max(jnp.abs(gradient - finite_difference_gradient)))
    gradient_scale = float(jnp.max(jnp.abs(finite_difference_gradient)))
    tolerance = float(atol) + float(rtol) * gradient_scale
    jvp_dot_abs_error = abs(float(jvp) - float(directional_from_gradient))
    jvp_tolerance = float(atol) + float(rtol) * max(abs(float(jvp)), 1.0e-300)
    gradient_norm = float(jnp.linalg.norm(gradient))
    objective_value = float(value)
    passed = (
        bool(jnp.isfinite(value))
        and bool(jnp.all(jnp.isfinite(gradient)))
        and gradient_norm > 1.0e-8
        and max_abs_error <= tolerance
        and jvp_dot_abs_error <= jvp_tolerance
    )

    return {
        "status": "pass" if passed else "fail",
        "optional_dependencies_required": False,
        "claim": "dkx_boozer_proxy_transport_objective_autodiff_only",
        "not_claimed": "vmex, booz_xform_jax, or kinetic SFINCS transport solve execution",
        "objective": objective_value,
        "gradient_norm": gradient_norm,
        "max_gradient_abs_error": max_abs_error,
        "gradient_tolerance": tolerance,
        "jvp_dot_abs_error": jvp_dot_abs_error,
        "jvp_tolerance": jvp_tolerance,
        "finite_difference_step": eps,
        "rtol": float(rtol),
        "atol": float(atol),
        "grid_shape": {"n_theta": int(theta.size), "n_zeta": int(zeta.size)},
        "spectrum_modes": int(coeff0.size),
        "transport_forces": [float(value) for value in forces],
    }


__all__ = [
    "boozer_bhat_from_spectrum",
    "boozer_spectrum_geometry_proxy_objective",
    "boozer_spectrum_proxy_transport_gradient_gate",
    "boozer_spectrum_proxy_transport_objective",
    "geometry_proxy_no_solve_provenance_gate",
    "geometry_proxy_workflow_contract",
    "geometry_proxy_workflow_summary",
    "kinetic_transport_scalar_no_overclaim_gate",
    "optional_jax_geometry_backend_report",
    "optional_jax_geometry_backend_status",
    "vmec_boozer_kinetic_transport_scalar_contract",
    "vmec_wout_from_wout_like",
]
