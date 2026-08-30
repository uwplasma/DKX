"""Immutable native DKX results and the version-1 NetCDF contract."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


RESULT_SCHEMA_VERSION = 1


def _frozen_arrays(values: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.array(value, copy=True)
        array.setflags(write=False)
        arrays[str(name)] = array
    return MappingProxyType(arrays)


def _frozen_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _frozen_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return _frozen_value(values)


@dataclass(frozen=True, slots=True)
class Result:
    """One native DKX calculation with named dimensions and arrays.

    Arrays are copied and made read-only at construction.  ``dimensions`` maps
    every array name to its dimension names; this keeps the in-memory contract
    usable without xarray while producing a naturally labelled NetCDF file.
    """

    case_id: str
    case_name: str
    workflow: str
    arrays: Mapping[str, np.ndarray]
    dimensions: Mapping[str, tuple[str, ...]]
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    output_path: Path | None = None
    schema_version: int = RESULT_SCHEMA_VERSION
    _runtime: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.schema_version) != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"result schema {self.schema_version} is not supported; "
                f"expected {RESULT_SCHEMA_VERSION}"
            )
        arrays = _frozen_arrays(self.arrays)
        dims = {str(key): tuple(value) for key, value in self.dimensions.items()}
        missing = set(arrays) - set(dims)
        if missing:
            raise ValueError(f"dimensions missing for result arrays: {sorted(missing)}")
        for name, array in arrays.items():
            if array.ndim != len(dims[name]):
                raise ValueError(
                    f"{name}: array rank {array.ndim} does not match dimensions {dims[name]!r}"
                )
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "dimensions", MappingProxyType(dims))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a documented array by name."""

        return self.arrays[name]

    def __getattr__(self, name: str) -> np.ndarray:
        """Allow concise direct array access such as ``result.particle_flux``."""

        arrays = object.__getattribute__(self, "arrays")
        try:
            return arrays[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def operator(self):
        """Temporary expert bridge to the solved operator, when retained."""

        runtime = object.__getattribute__(self, "_runtime")
        if runtime is None or "operator" not in runtime:
            raise AttributeError("operator is not retained in a loaded Result")
        return runtime["operator"]

    def to_dict(self) -> dict[str, Any]:
        """Return small metadata, dimensions, and array shape/dtype summaries."""

        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "workflow": self.workflow,
            "dimensions": {key: list(value) for key, value in self.dimensions.items()},
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in self.arrays.items()
            },
            "metadata": _jsonable(self.metadata),
            "warnings": list(self.warnings),
            "output_path": None if self.output_path is None else str(self.output_path),
        }

    def certificate(self) -> dict[str, Any]:
        """Return the compact numerical/provenance record used for review."""

        keys = (
            "converged",
            "solver_route",
            "route_reason",
            "residual_norm",
            "iterations",
            "ambipolar_all_surfaces_bracketed",
            "ambipolar_search",
            "ambipolar_selection",
            "ambipolar_refinement",
            "ambipolar_branch_continuation",
            "ambipolar_solver_attempts",
            "normalization",
            "geometry_sha256",
            "dkx_version",
            "python_version",
            "jax_version",
            "jaxlib_version",
            "platform",
            "precision",
            "device",
            "timings_s",
            "peak_host_memory_bytes",
        )
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            **{
                key: _jsonable(self.metadata[key])
                for key in keys
                if key in self.metadata
            },
            "warnings": list(self.warnings),
        }

    def print_summary(self) -> None:
        """Print a stable, plain-text summary suitable for logs."""

        converged = "yes" if self.metadata.get("converged") else "no"
        residual = self.metadata.get("residual_norm", "not measured")
        total = self.metadata.get("timings_s", {}).get("total", "not measured")
        print(f"DKX result: {self.case_name} ({self.case_id[:12]})")
        print(f"workflow: {self.workflow}")
        print(f"converged: {converged}; residual: {residual}")
        print(
            f"solver: {self.metadata.get('solver_route', 'unknown')}; total time: {total} s"
        )
        if "ambipolar_status" in self.arrays:
            statuses = [str(value) for value in self.arrays["ambipolar_status"]]
            bracketed = (
                int(np.count_nonzero(self.arrays["ambipolar_root_count"]))
                if "ambipolar_root_count" in self.arrays
                else sum(
                    value
                    in {"bracketed_root", "seeded_bracket_partial_failure"}
                    for value in statuses
                )
            )
            print(
                f"ambipolar roots: {bracketed}/{len(statuses)} surfaces bracketed; "
                "unbracketed surfaces retain the closest scanned point"
            )
        if "ambipolar_search_scope" in self.arrays:
            scopes = sorted(
                {str(value) for value in self.arrays["ambipolar_search_scope"]}
            )
            print(f"ambipolar search scope: {', '.join(scopes)}")
        if "ambipolar_refinement_status" in self.arrays:
            refinement_statuses = [
                str(value) for value in self.arrays["ambipolar_refinement_status"]
            ]
            resolved = sum(value == "resolved" for value in refinement_statuses)
            exhausted = sum(
                value == "refinement_exhausted" for value in refinement_statuses
            )
            no_bracket = sum(
                value == "no_bracket_observed" for value in refinement_statuses
            )
            not_requested = sum(
                value == "not_requested" for value in refinement_statuses
            )
            if not_requested == len(refinement_statuses):
                print("adaptive evidence: not requested")
            else:
                print(
                    "adaptive evidence: "
                    f"{resolved} resolved, {exhausted} refinement exhausted, "
                    f"{no_bracket} no bracket observed, {not_requested} not requested"
                )
        if "ambipolar_root_branch_id" in self.arrays:
            branch_ids = {
                str(value)
                for value in np.asarray(self.arrays["ambipolar_root_branch_id"]).flat
                if str(value)
            }
            event_kinds = [
                str(value)
                for value in np.asarray(
                    self.arrays.get("ambipolar_branch_event_kind", [])
                ).flat
                if str(value) and str(value) != "boundary_origin"
            ]
            print(
                f"ambipolar branches: {len(branch_ids)} identities; "
                f"{len(event_kinds)} interior events"
            )
        attempt_evidence = self.metadata.get("ambipolar_solver_attempts")
        if attempt_evidence:
            print(
                "ambipolar solver attempts: "
                f"{attempt_evidence['attempt_count']} total; "
                f"{attempt_evidence['automatic_true_residual_recovery_count']} "
                "automatic true-residual recoveries"
            )
        print("arrays: " + ", ".join(sorted(self.arrays)))
        for warning in self.warnings:
            print(f"warning: {warning}")

    def save(self, path: str | Path | None = None, *, overwrite: bool = True) -> Path:
        """Write the versioned native result as NetCDF4."""

        target = Path(path) if path is not None else self.output_path
        if target is None:
            raise ValueError(
                "save() needs a path because this Result has no output_path"
            )
        target = target.expanduser().resolve()
        if target.suffix.lower() != ".nc":
            raise ValueError("native Result.save() writes .nc files; use a .nc suffix")
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        from netCDF4 import Dataset  # noqa: PLC0415

        sizes: dict[str, int] = {}
        for name, array in self.arrays.items():
            for dim, size in zip(self.dimensions[name], array.shape):
                old = sizes.setdefault(dim, int(size))
                if old != int(size):
                    raise ValueError(
                        f"dimension {dim!r} has inconsistent sizes {old} and {size}"
                    )
        with Dataset(str(target), "w", format="NETCDF4") as handle:
            handle.setncattr("dkx_result_schema", int(self.schema_version))
            handle.setncattr("case_id", self.case_id)
            handle.setncattr("case_name", self.case_name)
            handle.setncattr("workflow", self.workflow)
            handle.setncattr(
                "metadata_json", json.dumps(_jsonable(self.metadata), sort_keys=True)
            )
            handle.setncattr("warnings_json", json.dumps(list(self.warnings)))
            for dim, size in sizes.items():
                handle.createDimension(dim, size)
            for name, array in self.arrays.items():
                if array.dtype.kind in {"U", "S", "O"}:
                    variable = handle.createVariable(name, str, self.dimensions[name])
                    variable[:] = array.astype(object)
                else:
                    variable = handle.createVariable(
                        name, array.dtype, self.dimensions[name], zlib=array.ndim > 0
                    )
                    variable[...] = array
        return target

    def plot(self, path: str | Path | None = None, *, panels: str = "auto"):
        """Plot the principal native observables against normalized flux.

        ``panels='auto'`` selects the arrays present in the result.  A path
        writes the figure and returns the resolved path; without one the
        Matplotlib figure is returned for notebook customization.
        """

        if panels != "auto":
            raise ValueError("Result.plot currently supports panels='auto'")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        choices = [
            ("particle_flux_m2_s", r"particle flux [m$^{-2}$ s$^{-1}$]"),
            ("heat_flux_W_m2", r"heat flux [W m$^{-2}$]"),
            ("parallel_current_A_T_m2", r"$\langle j\cdot B\rangle$ [A T m$^{-2}$]"),
            ("electric_field_kV_m", r"$E_r$ [kV m$^{-1}$]"),
        ]
        selected = [(name, label) for name, label in choices if name in self.arrays]
        if not selected:
            raise ValueError("Result has no auto-plot observables")
        surface = np.asarray(self.arrays.get("surface", self.arrays.get("r_N")))
        figure, axes = plt.subplots(
            len(selected),
            1,
            sharex=True,
            figsize=(7.2, 2.5 * len(selected)),
            squeeze=False,
        )
        species = [str(value) for value in self.arrays.get("species", [])]
        for axis, (name, label) in zip(axes[:, 0], selected):
            value = np.asarray(self.arrays[name])
            if value.ndim == 2:
                for index in range(value.shape[1]):
                    axis.plot(
                        surface, value[:, index], marker="o", label=species[index]
                    )
                axis.legend(frameon=False)
            else:
                axis.plot(
                    surface,
                    value,
                    marker="o",
                    label="selected branch" if name == "electric_field_kV_m" else None,
                )
                if (
                    name == "electric_field_kV_m"
                    and "ambipolar_root_kV_m" in self.arrays
                    and "ambipolar_root_branch_id" in self.arrays
                ):
                    root_fields = np.asarray(self.arrays["ambipolar_root_kV_m"])
                    root_branch_ids = np.asarray(
                        self.arrays["ambipolar_root_branch_id"]
                    ).astype(str)
                    branch_ids = sorted(
                        {branch for branch in root_branch_ids.flat if branch}
                    )
                    for branch_id in branch_ids:
                        branch_field = np.full(surface.shape, np.nan)
                        for surface_index in range(surface.size):
                            matches = np.flatnonzero(
                                root_branch_ids[surface_index] == branch_id
                            )
                            if matches.size:
                                branch_field[surface_index] = root_fields[
                                    surface_index, int(matches[0])
                                ]
                        axis.plot(
                            surface,
                            branch_field,
                            linestyle="--",
                            marker=".",
                            alpha=0.75,
                            label=branch_id,
                        )
                    if branch_ids:
                        axis.legend(frameon=False, ncol=2)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[-1, 0].set_xlabel(r"normalized toroidal flux $\psi_N$")
        title = self.case_name
        if "ambipolar_status" in self.arrays:
            statuses = np.asarray(self.arrays["ambipolar_status"]).astype(str)
            root_counts = (
                np.asarray(self.arrays["ambipolar_root_count"])
                if "ambipolar_root_count" in self.arrays
                else np.isin(
                    statuses,
                    ["bracketed_root", "seeded_bracket_partial_failure"],
                ).astype(np.int64)
            )
            missing = np.flatnonzero(root_counts == 0)
            if missing.size:
                locations = ", ".join(f"{surface[index]:.4g}" for index in missing)
                title += (
                    f"\nno bracketed root at $\\psi_N$={locations}; "
                    "showing closest scanned values"
                )
            partial = np.flatnonzero(statuses == "seeded_bracket_partial_failure")
            if partial.size:
                locations = ", ".join(f"{surface[index]:.4g}" for index in partial)
                title += (
                    "\none or more seeded brackets failed at "
                    f"$\\psi_N$={locations}"
                )
        if "ambipolar_refinement_status" in self.arrays:
            refinement_statuses = np.asarray(
                self.arrays["ambipolar_refinement_status"]
            ).astype(str)
            exhausted = np.flatnonzero(refinement_statuses == "refinement_exhausted")
            no_bracket = np.flatnonzero(refinement_statuses == "no_bracket_observed")
            notes = []
            if exhausted.size:
                notes.append(
                    "refinement exhausted at $\\psi_N$="
                    + ", ".join(f"{surface[index]:.4g}" for index in exhausted)
                )
            if no_bracket.size:
                notes.append(
                    "no bracket observed after finite refinement at $\\psi_N$="
                    + ", ".join(f"{surface[index]:.4g}" for index in no_bracket)
                )
            if notes:
                title += "\n" + "; ".join(notes)
        if "ambipolar_search_scope" in self.arrays and np.any(
            np.asarray(self.arrays["ambipolar_search_scope"]).astype(str)
            == "explicit_seeded_intervals_only"
        ):
            title += "\nseeded intervals only; unsampled crossings not excluded"
        if "ambipolar_nonsmooth_event" in self.arrays:
            nonsmooth = np.flatnonzero(
                np.asarray(self.arrays["ambipolar_nonsmooth_event"], dtype=bool)
            )
            if nonsmooth.size:
                title += "\nbranch event warning at $\\psi_N$=" + ", ".join(
                    f"{surface[index]:.4g}" for index in nonsmooth
                )
        figure.suptitle(title)
        figure.tight_layout()
        if path is None:
            return figure
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Result":
        """Load a version-1 native NetCDF result."""

        source = Path(path).expanduser().resolve()
        from netCDF4 import Dataset  # noqa: PLC0415

        with Dataset(str(source)) as handle:
            version = int(handle.getncattr("dkx_result_schema"))
            if version != RESULT_SCHEMA_VERSION:
                raise ValueError(
                    f"result schema {version} needs an explicit migration; "
                    f"this DKX reads schema {RESULT_SCHEMA_VERSION}"
                )
            arrays = {
                name: np.asarray(variable[:])
                for name, variable in handle.variables.items()
            }
            dimensions = {
                name: tuple(variable.dimensions)
                for name, variable in handle.variables.items()
            }
            metadata = json.loads(handle.getncattr("metadata_json"))
            warnings = tuple(json.loads(handle.getncattr("warnings_json")))
            return cls(
                schema_version=version,
                case_id=str(handle.getncattr("case_id")),
                case_name=str(handle.getncattr("case_name")),
                workflow=str(handle.getncattr("workflow")),
                arrays=arrays,
                dimensions=dimensions,
                metadata=metadata,
                warnings=warnings,
                output_path=source,
            )
