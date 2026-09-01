"""Versioned, immutable DKX case configuration.

TOML is the human-authored format and JSON is the machine-authored format.
Both enter through :meth:`Case.from_mapping`, which is the single validation
boundary.  The numerical normalizer will consume this model rather than
re-reading either serialization format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import tomllib


SCHEMA_VERSION = 1
DEFAULT_MAX_SCAN_CASES = 10_000


class CaseValidationError(ValueError):
    """A precise error at the native-case serialization boundary."""

    def __init__(self, path: str, value: Any, expected: str, correction: str) -> None:
        self.path = path
        self.value = value
        self.expected = expected
        self.correction = correction
        rendered = repr(value)
        super().__init__(
            f"{path}: supplied {rendered}; expected {expected}. {correction}"
        )


@dataclass(frozen=True)
class RunConfig:
    workflow: str
    precision: str = "float64"
    device: str = "auto"
    progress: bool = True


@dataclass(frozen=True)
class GeometryConfig:
    format: str
    file: Path
    surfaces: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", _semantic_path(self.file))
        object.__setattr__(self, "surfaces", tuple(self.surfaces))


@dataclass(frozen=True)
class SpeciesConfig:
    name: str
    charge: float
    mass_amu: float
    density_m3: tuple[float, ...]
    temperature_keV: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_m3", tuple(self.density_m3))
        object.__setattr__(self, "temperature_keV", tuple(self.temperature_keV))


@dataclass(frozen=True)
class PhysicsConfig:
    model: str = "full_local"
    collisions: str = "linearized_fokker_planck"
    magnetic_drifts: str = "full"
    phi1: str = "off"
    #: Coulomb logarithm. The reference set pins 17.0, and the normalized
    #: collisionality is proportional to it, so this is how a case says what a
    #: SFINCS deck says with a ``nu_n`` override. Without it a case could not
    #: express a different ln-Lambda at all, which is what blocked 28 of the
    #: 102 checked-in decks from converting.
    coulomb_logarithm: float = 17.0


@dataclass(frozen=True)
class ElectricFieldConfig:
    mode: str = "prescribed"
    value_kV_m: float | None = None
    search_kV_m: tuple[float, float] | None = None
    find_all_roots: bool = True
    continue_branches: bool = True
    search_points: int = 9
    root_tolerance_kV_m: float = 1.0e-3
    max_root_iterations: int = 20
    search_strategy: str = "uniform"
    seed_brackets_kV_m: tuple[tuple[tuple[float, float], ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.search_kV_m is not None:
            object.__setattr__(self, "search_kV_m", tuple(self.search_kV_m))
        if self.seed_brackets_kV_m is not None:
            object.__setattr__(
                self,
                "seed_brackets_kV_m",
                tuple(
                    tuple(tuple(bracket) for bracket in surface)
                    for surface in self.seed_brackets_kV_m
                ),
            )


@dataclass(frozen=True)
class ResolutionConfig:
    theta: int
    zeta: int
    pitch: int
    speed: int
    pitch_speed_ramp: int = 1
    pitch_modes_by_speed: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.pitch_modes_by_speed is not None:
            object.__setattr__(
                self, "pitch_modes_by_speed", tuple(self.pitch_modes_by_speed)
            )


@dataclass(frozen=True)
class SolverConfig:
    method: str = "auto"
    relative_tolerance: float = 1.0e-10
    memory_fraction: float = 0.75
    reuse: str = "auto"


@dataclass(frozen=True)
class ParallelConfig:
    strategy: str = "auto"
    shard: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard", tuple(self.shard))


@dataclass(frozen=True)
class ConvergenceConfig:
    enabled: bool = False
    observables: tuple[str, ...] = ()
    relative_tolerance: float = 0.02
    max_refinements: int = 3
    retain_legendre_tail: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "observables", tuple(self.observables))


@dataclass(frozen=True)
class OutputConfig:
    file: Path = Path("dkx_result.nc")
    plots: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", _semantic_path(self.file))


@dataclass(frozen=True)
class ScanAxis:
    path: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class ScanConfig:
    combine: str = "cartesian"
    resume: bool = True
    output: Path = Path("dkx_scan.nc")
    axes: tuple[ScanAxis, ...] = ()
    max_cases: int = DEFAULT_MAX_SCAN_CASES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _semantic_path(self.output))
        object.__setattr__(self, "axes", tuple(self.axes))

    @property
    def case_count(self) -> int:
        if not self.axes:
            return 0
        if self.combine == "cartesian":
            count = 1
            for axis in self.axes:
                count *= len(axis.values)
            return count
        return len(self.axes[0].values)


@dataclass(frozen=True)
class Case:
    """Canonical, immutable high-level DKX input.

    ``source_path`` is provenance used to resolve relative external files.  It
    is intentionally excluded from :attr:`case_id`, equality, and serialized
    semantic content, so moving a case file does not change the calculation it
    identifies.
    """

    schema: int
    name: str
    run: RunConfig
    geometry: GeometryConfig
    species: tuple[SpeciesConfig, ...]
    physics: PhysicsConfig
    electric_field: ElectricFieldConfig
    resolution: ResolutionConfig
    solver: SolverConfig
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scan: ScanConfig | None = None
    source_path: Path | None = field(default=None, compare=False, repr=False)
    _case_id: str = field(init=False, default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        if self.source_path is not None:
            object.__setattr__(
                self, "source_path", Path(self.source_path).expanduser().resolve()
            )
        _validate_case(self)
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        object.__setattr__(self, "_case_id", hashlib.sha256(canonical).hexdigest())

    @property
    def case_id(self) -> str:
        """SHA-256 of normalized semantic content, independent of key order."""

        return self._case_id

    @property
    def base_directory(self) -> Path:
        """Directory against which relative external paths are resolved."""

        return self.source_path.parent if self.source_path is not None else Path.cwd()

    @property
    def geometry_path(self) -> Path:
        path = self.geometry.file
        return path if path.is_absolute() else self.base_directory / path

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/TOML-compatible normalized semantic content."""

        data = asdict(self)
        data.pop("source_path", None)
        data.pop("_case_id", None)
        # Option 1 has always been the case execution default.  Omitting it
        # from canonical content preserves existing schema-v1 case IDs while
        # non-default ramp choices remain explicit semantic input.
        if data["resolution"]["pitch_speed_ramp"] == 1:
            data["resolution"].pop("pitch_speed_ramp")
        if data["resolution"]["pitch_modes_by_speed"] is None:
            data["resolution"].pop("pitch_modes_by_speed")
        # Tail reconstruction is opt-in. Omitting the false default preserves
        # every existing schema-v1 case ID.
        if not data["convergence"]["retain_legendre_tail"]:
            data["convergence"].pop("retain_legendre_tail")
        # Uniform discovery is the historical schema-v1 behavior. Omitting
        # both defaults preserves every existing case ID; seeded promotion is
        # explicit semantic content and therefore remains canonical.
        if data["electric_field"]["search_strategy"] == "uniform":
            data["electric_field"].pop("search_strategy")
        if data["electric_field"]["seed_brackets_kV_m"] is None:
            data["electric_field"].pop("seed_brackets_kV_m")
        if data["scan"] is not None:
            data["scan"]["axis"] = data["scan"].pop("axes")
        return _paths_to_strings(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Case":
        """Read and validate a schema-v1 TOML or JSON case."""

        source = Path(path).expanduser().resolve()
        suffix = source.suffix.lower()
        try:
            if suffix == ".toml":
                with source.open("rb") as stream:
                    raw = tomllib.load(stream)
            elif suffix == ".json":
                raw = json.loads(source.read_text(encoding="utf-8"))
            else:
                raise CaseValidationError(
                    "$file",
                    source.name,
                    "a .toml or .json DKX case",
                    "Rename or convert the input to a supported case format.",
                )
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise CaseValidationError(
                "$file",
                source.name,
                "valid syntax",
                f"Correct the parser error: {exc}",
            ) from exc
        if not isinstance(raw, Mapping):
            raise CaseValidationError(
                "$", raw, "a table/object", "Put the case fields in a top-level table."
            )
        return cls.from_mapping(raw, source_path=source)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source_path: str | Path | None = None
    ) -> "Case":
        """Validate one mapping and construct the canonical typed model."""

        data = migrate_case_data(raw)
        _reject_unknown(
            data,
            "$",
            {
                "schema",
                "name",
                "run",
                "geometry",
                "species",
                "physics",
                "electric_field",
                "resolution",
                "solver",
                "parallel",
                "convergence",
                "output",
                "scan",
            },
        )
        schema = _integer(data, "schema", "$", required=True)
        name = _string(data, "name", "$", required=True)
        run = _parse_run(_table(data, "run", "$", required=True))
        geometry = _parse_geometry(_table(data, "geometry", "$", required=True))
        species_raw = _sequence(data, "species", "$", required=True)
        species = tuple(
            _parse_species(_as_table(item, f"species[{index}]"), index)
            for index, item in enumerate(species_raw)
        )
        physics = _parse_physics(_table(data, "physics", "$", required=True))
        electric_field = _parse_electric_field(
            _table(data, "electric_field", "$", required=True)
        )
        resolution = _parse_resolution(_table(data, "resolution", "$", required=True))
        solver = _parse_solver(_table(data, "solver", "$", required=True))
        parallel = _parse_parallel(_table(data, "parallel", "$", default={}))
        convergence = _parse_convergence(_table(data, "convergence", "$", default={}))
        output = _parse_output(_table(data, "output", "$", default={}))
        scan_raw = data.get("scan")
        scan = None if scan_raw is None else _parse_scan(_as_table(scan_raw, "scan"))
        source = (
            None if source_path is None else Path(source_path).expanduser().resolve()
        )
        return cls(
            schema=schema,
            name=name,
            run=run,
            geometry=geometry,
            species=species,
            physics=physics,
            electric_field=electric_field,
            resolution=resolution,
            solver=solver,
            parallel=parallel,
            convergence=convergence,
            output=output,
            scan=scan,
            source_path=source,
        )


def migrate_case_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return schema-v1 data or reject versions without a defined migration."""

    data = dict(raw)
    supplied = data.get("schema")
    if supplied != SCHEMA_VERSION:
        raise CaseValidationError(
            "schema",
            supplied,
            f"integer {SCHEMA_VERSION}",
            "Set schema = 1; no migration from this version is defined yet.",
        )
    return data


def case_json_schema() -> dict[str, Any]:
    """Return the machine-readable JSON Schema for case schema version 1."""

    positive_numbers = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "number", "exclusiveMinimum": 0},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/uwplasma/DKX/schemas/case-v1.json",
        "title": "DKX case",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "name",
            "run",
            "geometry",
            "species",
            "physics",
            "electric_field",
            "resolution",
            "solver",
        ],
        "properties": {
            "schema": {"const": SCHEMA_VERSION},
            "name": {"type": "string", "minLength": 1},
            "run": _object_schema(
                ["workflow"],
                workflow={
                    "enum": [
                        "profile",
                        "ambipolar_profile",
                        "transport_matrix",
                        "monoenergetic",
                    ]
                },
                precision={"enum": ["float64"]},
                device={"type": "string", "minLength": 1},
                progress={"type": "boolean"},
            ),
            "geometry": _object_schema(
                ["format", "file", "surfaces"],
                format={"enum": ["vmec", "boozer", "analytic"]},
                file={"type": "string", "minLength": 1},
                surfaces={
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            ),
            "species": {
                "type": "array",
                "minItems": 1,
                "items": _object_schema(
                    ["name", "charge", "mass_amu", "density_m3", "temperature_keV"],
                    name={"type": "string", "minLength": 1},
                    charge={"type": "number", "not": {"const": 0}},
                    mass_amu={"type": "number", "exclusiveMinimum": 0},
                    density_m3=positive_numbers,
                    temperature_keV=positive_numbers,
                ),
            },
            "physics": _object_schema(
                [],
                model={"enum": ["full_local"]},
                collisions={
                    "enum": ["linearized_fokker_planck", "pitch_angle_scattering"]
                },
                magnetic_drifts={"enum": ["full", "dkes"]},
                phi1={"enum": ["off", "kinetic", "full"]},
                coulomb_logarithm={"type": "number", "minimum": 5.0, "maximum": 30.0},
            ),
            "electric_field": _object_schema(
                ["mode"],
                mode={"enum": ["prescribed", "ambipolar"]},
                value_kV_m={"type": "number"},
                search_kV_m={
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                find_all_roots={"type": "boolean"},
                continue_branches={"type": "boolean"},
                search_points={"type": "integer", "minimum": 3},
                root_tolerance_kV_m={"type": "number", "exclusiveMinimum": 0},
                max_root_iterations={"type": "integer", "minimum": 1},
                search_strategy={"enum": ["uniform", "seeded_brackets"]},
                seed_brackets_kV_m={
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                    },
                },
            ),
            "resolution": _object_schema(
                ["theta", "zeta", "pitch", "speed"],
                pitch_speed_ramp={"type": "integer", "enum": [0, 1, 2]},
                pitch_modes_by_speed={
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 1},
                },
                **{
                    key: {"type": "integer", "minimum": 1}
                    for key in ("theta", "zeta", "pitch", "speed")
                },
            ),
            "solver": _object_schema(
                [],
                method={
                    "enum": [
                        "auto",
                        "structured_direct",
                        "recycled_krylov",
                        "sparse_direct_referee",
                    ]
                },
                relative_tolerance={
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                memory_fraction={"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                reuse={"enum": ["auto", "on", "off"]},
            ),
            "parallel": _object_schema(
                [],
                strategy={"enum": ["auto", "serial", "batch"]},
                shard={
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"enum": ["surface", "electric_field", "species"]},
                },
            ),
            "convergence": _object_schema(
                [],
                enabled={"type": "boolean"},
                observables={
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                relative_tolerance={
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                max_refinements={"type": "integer", "minimum": 0},
                retain_legendre_tail={"type": "boolean"},
            ),
            "output": _object_schema(
                [], file={"type": "string", "minLength": 1}, plots={"type": "boolean"}
            ),
            "scan": _object_schema(
                ["axis"],
                combine={"enum": ["cartesian", "zipped"]},
                resume={"type": "boolean"},
                output={"type": "string", "minLength": 1},
                max_cases={"type": "integer", "minimum": 1},
                axis={
                    "type": "array",
                    "minItems": 1,
                    "items": _object_schema(
                        ["path", "values"],
                        path={"type": "string", "minLength": 1},
                        values={
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "number"},
                        },
                    ),
                },
            ),
        },
    }


COMMENTED_TOML_EXAMPLE = """# DKX case schema version 1.
schema = 1
name = "w7x_ambipolar_profile"

[run]
workflow = "ambipolar_profile" # profile, transport_matrix, or monoenergetic are also supported.
precision = "float64"
device = "auto"
progress = true

[geometry]
format = "vmec"
file = "wout_w7x.nc"
surfaces = [0.20, 0.35, 0.50, 0.65, 0.80] # Normalized toroidal flux.

[[species]]
name = "deuterium"
charge = 1
mass_amu = 2.014
density_m3 = [8.0e19, 7.3e19, 6.4e19, 5.2e19, 3.8e19]
temperature_keV = [1.2, 1.1, 0.95, 0.75, 0.50]

[[species]]
name = "electron"
charge = -1
mass_amu = 5.485799e-4
density_m3 = [8.0e19, 7.3e19, 6.4e19, 5.2e19, 3.8e19]
temperature_keV = [2.5, 2.3, 2.0, 1.6, 1.0]

[physics]
model = "full_local"
collisions = "linearized_fokker_planck"
magnetic_drifts = "full"
phi1 = "off"
# Coulomb logarithm. The collisionality is proportional to it, so this is how a
# case says what a SFINCS deck says with a nu_n override. 17.0 is the pinned
# reference value and the default; omit this line to use it.
coulomb_logarithm = 17.0

[electric_field]
mode = "ambipolar"
search_kV_m = [-40.0, 40.0]
find_all_roots = true
continue_branches = true
search_points = 9
root_tolerance_kV_m = 0.001
max_root_iterations = 20

[resolution]
theta = 31
zeta = 31
pitch = 24
speed = 8
# SFINCS Nxi_for_x_option: 0 uniform, 1 linear (default), 2 quadratic.
# pitch_speed_ramp = 1
# Advanced evidence control: one monotone active-mode count per speed node.
# pitch_modes_by_speed = [4, 8, 16, 24, 24, 24, 24, 24]

[solver]
method = "auto"
relative_tolerance = 1.0e-10
memory_fraction = 0.75
reuse = "auto"

[parallel]
strategy = "auto"
shard = ["surface", "electric_field"]

[convergence]
enabled = true
observables = ["particle_flux", "heat_flux", "bootstrap_current", "electric_field"]
relative_tolerance = 0.02
max_refinements = 3
# Opt-in exact selected-tail sweep and rigorous relative-L2 upper bound.
# retain_legendre_tail = false

[output]
file = "outputs/w7x_ambipolar.nc"
plots = true

# Uncomment to declare a resumable scan.
#[scan]
#combine = "cartesian"
#resume = true
#output = "outputs/er_scan.nc"
#max_cases = 10000
#
#[[scan.axis]]
#path = "electric_field.value_kV_m"
#values = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
"""


def _parse_run(data: Mapping[str, Any]) -> RunConfig:
    path = "run"
    _reject_unknown(data, path, {"workflow", "precision", "device", "progress"})
    return RunConfig(
        workflow=_string(data, "workflow", path, required=True),
        precision=_string(data, "precision", path, default="float64"),
        device=_string(data, "device", path, default="auto"),
        progress=_boolean(data, "progress", path, default=True),
    )


def _parse_geometry(data: Mapping[str, Any]) -> GeometryConfig:
    path = "geometry"
    _reject_unknown(data, path, {"format", "file", "surfaces"})
    return GeometryConfig(
        format=_string(data, "format", path, required=True),
        file=Path(_string(data, "file", path, required=True)),
        surfaces=_number_tuple(data, "surfaces", path, required=True),
    )


def _parse_species(data: Mapping[str, Any], index: int) -> SpeciesConfig:
    path = f"species[{index}]"
    _reject_unknown(
        data, path, {"name", "charge", "mass_amu", "density_m3", "temperature_keV"}
    )
    return SpeciesConfig(
        name=_string(data, "name", path, required=True),
        charge=_number(data, "charge", path, required=True),
        mass_amu=_number(data, "mass_amu", path, required=True),
        density_m3=_number_tuple(data, "density_m3", path, required=True),
        temperature_keV=_number_tuple(data, "temperature_keV", path, required=True),
    )


def _parse_physics(data: Mapping[str, Any]) -> PhysicsConfig:
    path = "physics"
    _reject_unknown(
        data,
        path,
        {"model", "collisions", "magnetic_drifts", "phi1", "coulomb_logarithm"},
    )
    return PhysicsConfig(
        model=_string(data, "model", path, default="full_local"),
        collisions=_string(
            data, "collisions", path, default="linearized_fokker_planck"
        ),
        magnetic_drifts=_string(data, "magnetic_drifts", path, default="full"),
        phi1=_string(data, "phi1", path, default="off"),
        coulomb_logarithm=_number(data, "coulomb_logarithm", path, default=17.0),
    )


def _parse_electric_field(data: Mapping[str, Any]) -> ElectricFieldConfig:
    path = "electric_field"
    _reject_unknown(
        data,
        path,
        {
            "mode",
            "value_kV_m",
            "search_kV_m",
            "find_all_roots",
            "continue_branches",
            "search_points",
            "root_tolerance_kV_m",
            "max_root_iterations",
            "search_strategy",
            "seed_brackets_kV_m",
        },
    )
    search = None
    if "search_kV_m" in data:
        values = _number_tuple(data, "search_kV_m", path, required=True)
        if len(values) != 2:
            _fail(
                f"{path}.search_kV_m",
                values,
                "exactly [minimum, maximum]",
                "Supply two finite bounds.",
            )
        search = (values[0], values[1])
    value = (
        None
        if "value_kV_m" not in data
        else _number(data, "value_kV_m", path, required=True)
    )
    seeds = None
    if "seed_brackets_kV_m" in data:
        parsed_surfaces = []
        for surface_index, surface in enumerate(
            _sequence(data, "seed_brackets_kV_m", path, required=True)
        ):
            if isinstance(surface, (str, bytes)) or not isinstance(surface, Sequence):
                _fail(
                    f"{path}.seed_brackets_kV_m[{surface_index}]",
                    surface,
                    "an array of [left, right] brackets",
                    "Provide one bracket array per geometry surface.",
                )
            parsed_brackets = []
            for bracket_index, bracket in enumerate(surface):
                where = f"{path}.seed_brackets_kV_m[{surface_index}][{bracket_index}]"
                if isinstance(bracket, (str, bytes)) or not isinstance(
                    bracket, Sequence
                ):
                    _fail(
                        where,
                        bracket,
                        "exactly [left, right]",
                        "Provide two finite bracket endpoints.",
                    )
                values = tuple(bracket)
                if len(values) != 2 or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in values
                ):
                    _fail(
                        where,
                        bracket,
                        "exactly two finite numbers",
                        "Provide a finite increasing bracket.",
                    )
                parsed_brackets.append((float(values[0]), float(values[1])))
            parsed_surfaces.append(tuple(parsed_brackets))
        seeds = tuple(parsed_surfaces)
    return ElectricFieldConfig(
        mode=_string(data, "mode", path, default="prescribed"),
        value_kV_m=value,
        search_kV_m=search,
        find_all_roots=_boolean(data, "find_all_roots", path, default=True),
        continue_branches=_boolean(data, "continue_branches", path, default=True),
        search_points=_integer(data, "search_points", path, default=9),
        root_tolerance_kV_m=_number(data, "root_tolerance_kV_m", path, default=1.0e-3),
        max_root_iterations=_integer(data, "max_root_iterations", path, default=20),
        search_strategy=_string(data, "search_strategy", path, default="uniform"),
        seed_brackets_kV_m=seeds,
    )


def _parse_resolution(data: Mapping[str, Any]) -> ResolutionConfig:
    path = "resolution"
    _reject_unknown(
        data,
        path,
        {
            "theta",
            "zeta",
            "pitch",
            "speed",
            "pitch_speed_ramp",
            "pitch_modes_by_speed",
        },
    )
    return ResolutionConfig(
        theta=_integer(data, "theta", path, required=True),
        zeta=_integer(data, "zeta", path, required=True),
        pitch=_integer(data, "pitch", path, required=True),
        speed=_integer(data, "speed", path, required=True),
        pitch_speed_ramp=_integer(data, "pitch_speed_ramp", path, default=1),
        pitch_modes_by_speed=(
            _integer_tuple(data, "pitch_modes_by_speed", path)
            if "pitch_modes_by_speed" in data
            else None
        ),
    )


def _parse_solver(data: Mapping[str, Any]) -> SolverConfig:
    path = "solver"
    _reject_unknown(
        data, path, {"method", "relative_tolerance", "memory_fraction", "reuse"}
    )
    return SolverConfig(
        method=_string(data, "method", path, default="auto"),
        relative_tolerance=_number(data, "relative_tolerance", path, default=1.0e-10),
        memory_fraction=_number(data, "memory_fraction", path, default=0.75),
        reuse=_string(data, "reuse", path, default="auto"),
    )


def _parse_parallel(data: Mapping[str, Any]) -> ParallelConfig:
    path = "parallel"
    _reject_unknown(data, path, {"strategy", "shard"})
    return ParallelConfig(
        strategy=_string(data, "strategy", path, default="auto"),
        shard=_string_tuple(data, "shard", path, default=()),
    )


def _parse_convergence(data: Mapping[str, Any]) -> ConvergenceConfig:
    path = "convergence"
    _reject_unknown(
        data,
        path,
        {
            "enabled",
            "observables",
            "relative_tolerance",
            "max_refinements",
            "retain_legendre_tail",
        },
    )
    return ConvergenceConfig(
        enabled=_boolean(data, "enabled", path, default=False),
        observables=_string_tuple(data, "observables", path, default=()),
        relative_tolerance=_number(data, "relative_tolerance", path, default=0.02),
        max_refinements=_integer(data, "max_refinements", path, default=3),
        retain_legendre_tail=_boolean(
            data, "retain_legendre_tail", path, default=False
        ),
    )


def _parse_output(data: Mapping[str, Any]) -> OutputConfig:
    path = "output"
    _reject_unknown(data, path, {"file", "plots"})
    return OutputConfig(
        file=Path(_string(data, "file", path, default="dkx_result.nc")),
        plots=_boolean(data, "plots", path, default=True),
    )


def _parse_scan(data: Mapping[str, Any]) -> ScanConfig:
    path = "scan"
    _reject_unknown(data, path, {"combine", "resume", "output", "axis", "max_cases"})
    axes_raw = _sequence(data, "axis", path, required=True)
    axes = []
    for index, item in enumerate(axes_raw):
        axis_path = f"scan.axis[{index}]"
        table = _as_table(item, axis_path)
        _reject_unknown(table, axis_path, {"path", "values"})
        axes.append(
            ScanAxis(
                path=_string(table, "path", axis_path, required=True),
                values=_number_tuple(table, "values", axis_path, required=True),
            )
        )
    return ScanConfig(
        combine=_string(data, "combine", path, default="cartesian"),
        resume=_boolean(data, "resume", path, default=True),
        output=Path(_string(data, "output", path, default="dkx_scan.nc")),
        axes=tuple(axes),
        max_cases=_integer(data, "max_cases", path, default=DEFAULT_MAX_SCAN_CASES),
    )


def _validate_case(case: Case) -> None:
    _choice("schema", case.schema, (SCHEMA_VERSION,))
    if not case.name.strip():
        _fail(
            "name", case.name, "a non-empty string", "Give the case a descriptive name."
        )
    _choice(
        "run.workflow",
        case.run.workflow,
        ("profile", "ambipolar_profile", "transport_matrix", "monoenergetic"),
    )
    _choice("run.precision", case.run.precision, ("float64",))
    if not case.run.device.strip():
        _fail(
            "run.device",
            case.run.device,
            "a device name or 'auto'",
            "Use device = 'auto' for portable cases.",
        )
    _is_boolean("run.progress", case.run.progress)
    _choice("geometry.format", case.geometry.format, ("vmec", "boozer", "analytic"))
    if not str(case.geometry.file):
        _fail(
            "geometry.file",
            case.geometry.file,
            "a non-empty path",
            "Supply the geometry source path.",
        )
    if not case.geometry.surfaces:
        _fail(
            "geometry.surfaces",
            case.geometry.surfaces,
            "one or more normalized surfaces",
            "Supply values from 0 through 1.",
        )
    for index, surface in enumerate(case.geometry.surfaces):
        _bounded(f"geometry.surfaces[{index}]", surface, 0.0, 1.0)
    if len(set(case.geometry.surfaces)) != len(case.geometry.surfaces):
        _fail(
            "geometry.surfaces",
            case.geometry.surfaces,
            "unique values",
            "Remove duplicate surfaces.",
        )
    if not case.species:
        _fail(
            "species",
            case.species,
            "one or more species",
            "Add at least one [[species]] table.",
        )
    names: set[str] = set()
    n_surfaces = len(case.geometry.surfaces)
    for index, species in enumerate(case.species):
        prefix = f"species[{index}]"
        if not species.name.strip() or species.name in names:
            _fail(
                f"{prefix}.name",
                species.name,
                "a unique non-empty name",
                "Use a distinct physical species name.",
            )
        names.add(species.name)
        if species.charge == 0:
            _fail(
                f"{prefix}.charge",
                species.charge,
                "a nonzero charge in elementary-charge units",
                "Use a signed ion charge or -1 for electrons.",
            )
        _positive(f"{prefix}.mass_amu", species.mass_amu)
        for field_name, values in (
            ("density_m3", species.density_m3),
            ("temperature_keV", species.temperature_keV),
        ):
            if len(values) != n_surfaces:
                _fail(
                    f"{prefix}.{field_name}",
                    values,
                    f"{n_surfaces} values (one per geometry surface)",
                    "Add or remove profile entries to match geometry.surfaces.",
                )
            for value_index, value in enumerate(values):
                _positive(f"{prefix}.{field_name}[{value_index}]", value)
    _choice("physics.model", case.physics.model, ("full_local",))
    _choice(
        "physics.collisions",
        case.physics.collisions,
        ("linearized_fokker_planck", "pitch_angle_scattering"),
    )
    _choice("physics.magnetic_drifts", case.physics.magnetic_drifts, ("full", "dkes"))
    _choice("physics.phi1", case.physics.phi1, ("off", "kinetic", "full"))
    # A ln-Lambda outside this band is not a plasma anyone is modelling here;
    # the bound catches a value entered in the wrong units far more often than
    # it refuses a real one.
    if not (5.0 <= float(case.physics.coulomb_logarithm) <= 30.0):
        _fail(
            "physics.coulomb_logarithm",
            case.physics.coulomb_logarithm,
            "a Coulomb logarithm between 5 and 30",
            "Fusion-relevant ln-Lambda is near 17, which is the default.",
        )
    _choice(
        "electric_field.mode", case.electric_field.mode, ("prescribed", "ambipolar")
    )
    if (
        case.electric_field.mode == "prescribed"
        and case.electric_field.value_kV_m is None
    ):
        _fail(
            "electric_field.value_kV_m",
            None,
            "a finite number when mode = 'prescribed'",
            "Add value_kV_m or select mode = 'ambipolar'.",
        )
    if case.electric_field.value_kV_m is not None and not math.isfinite(
        case.electric_field.value_kV_m
    ):
        _fail(
            "electric_field.value_kV_m",
            case.electric_field.value_kV_m,
            "a finite number",
            "Use a finite electric field in kV/m.",
        )
    if case.electric_field.mode == "ambipolar":
        bounds = case.electric_field.search_kV_m
        if (
            bounds is None
            or not all(math.isfinite(value) for value in bounds)
            or bounds[0] >= bounds[1]
        ):
            _fail(
                "electric_field.search_kV_m",
                bounds,
                "[minimum, maximum] with minimum < maximum",
                "Supply a finite ambipolar search bracket.",
            )
    _choice(
        "electric_field.search_strategy",
        case.electric_field.search_strategy,
        ("uniform", "seeded_brackets"),
    )
    seeds = case.electric_field.seed_brackets_kV_m
    if case.electric_field.search_strategy == "uniform" and seeds is not None:
        _fail(
            "electric_field.seed_brackets_kV_m",
            seeds,
            "omitted when search_strategy = 'uniform'",
            "Remove seed_brackets_kV_m or select search_strategy = 'seeded_brackets'.",
        )
    if case.electric_field.search_strategy == "seeded_brackets":
        if (
            case.run.workflow != "ambipolar_profile"
            or case.electric_field.mode != "ambipolar"
        ):
            _fail(
                "electric_field.search_strategy",
                case.electric_field.search_strategy,
                "seeded_brackets only for an ambipolar_profile workflow",
                "Select workflow = 'ambipolar_profile' and electric_field.mode = 'ambipolar'.",
            )
        if seeds is None or len(seeds) != len(case.geometry.surfaces):
            _fail(
                "electric_field.seed_brackets_kV_m",
                seeds,
                f"exactly {len(case.geometry.surfaces)} non-empty surface arrays",
                "Provide one bracket array per geometry surface.",
            )
        if case.convergence.enabled:
            _fail(
                "convergence.enabled",
                case.convergence.enabled,
                "false for seeded bracket promotion",
                "Converge the global discovery grid first; seeded promotion only refines explicit brackets.",
            )
        if not case.electric_field.find_all_roots:
            _fail(
                "electric_field.find_all_roots",
                case.electric_field.find_all_roots,
                "true for seeded bracket promotion",
                "Retain and refine every explicitly supplied branch bracket.",
            )
        assert seeds is not None
        bounds = case.electric_field.search_kV_m
        assert bounds is not None
        for surface_index, surface_brackets in enumerate(seeds):
            if not surface_brackets:
                _fail(
                    f"electric_field.seed_brackets_kV_m[{surface_index}]",
                    surface_brackets,
                    "at least one bracket",
                    "Provide every previously discovered branch bracket.",
                )
            previous_right = -math.inf
            for bracket_index, (left, right) in enumerate(surface_brackets):
                where = (
                    f"electric_field.seed_brackets_kV_m[{surface_index}]"
                    f"[{bracket_index}]"
                )
                if left >= right or left < bounds[0] or right > bounds[1]:
                    _fail(
                        where,
                        (left, right),
                        f"an increasing bracket inside search_kV_m={bounds}",
                        "Order the endpoints and keep them inside the declared search domain.",
                    )
                if left < previous_right:
                    _fail(
                        where,
                        (left, right),
                        "ordered non-overlapping brackets",
                        "Sort the brackets and remove overlaps.",
                    )
                previous_right = right
    _is_boolean("electric_field.find_all_roots", case.electric_field.find_all_roots)
    _is_boolean(
        "electric_field.continue_branches", case.electric_field.continue_branches
    )
    if (
        isinstance(case.electric_field.search_points, bool)
        or not isinstance(case.electric_field.search_points, int)
        or case.electric_field.search_points < 3
    ):
        _fail(
            "electric_field.search_points",
            case.electric_field.search_points,
            "an integer >= 3",
            "Use at least three coarse electric-field samples.",
        )
    _positive(
        "electric_field.root_tolerance_kV_m",
        case.electric_field.root_tolerance_kV_m,
    )
    if (
        isinstance(case.electric_field.max_root_iterations, bool)
        or not isinstance(case.electric_field.max_root_iterations, int)
        or case.electric_field.max_root_iterations < 1
    ):
        _fail(
            "electric_field.max_root_iterations",
            case.electric_field.max_root_iterations,
            "an integer >= 1",
            "Allow at least one bracket-refinement iteration.",
        )
    for name in ("theta", "zeta", "pitch", "speed"):
        value = getattr(case.resolution, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _fail(
                f"resolution.{name}",
                value,
                "an integer >= 1",
                "Choose a positive phase-space resolution.",
            )
    _choice("resolution.pitch_speed_ramp", case.resolution.pitch_speed_ramp, (0, 1, 2))
    modes = case.resolution.pitch_modes_by_speed
    if modes is not None:
        if case.resolution.pitch_speed_ramp != 1:
            _fail(
                "resolution.pitch_speed_ramp",
                case.resolution.pitch_speed_ramp,
                "the default value 1 when pitch_modes_by_speed is supplied",
                "Remove pitch_speed_ramp; the explicit allocation replaces the ramp.",
            )
        if len(modes) != case.resolution.speed:
            _fail(
                "resolution.pitch_modes_by_speed",
                modes,
                f"exactly {case.resolution.speed} entries",
                "Provide one active pitch-mode count for every speed node.",
            )
        for index, value in enumerate(modes):
            if isinstance(value, bool) or not isinstance(value, int):
                _fail(
                    f"resolution.pitch_modes_by_speed[{index}]",
                    value,
                    "an integer",
                    "Use whole active-mode counts without quotes.",
                )
            if value < 4 or value > case.resolution.pitch:
                _fail(
                    f"resolution.pitch_modes_by_speed[{index}]",
                    value,
                    f"an integer from 4 through {case.resolution.pitch}",
                    "Keep the collision-coupling floor and declared pitch maximum.",
                )
        if any(right < left for left, right in zip(modes, modes[1:])):
            _fail(
                "resolution.pitch_modes_by_speed",
                modes,
                "a nondecreasing speed-node allocation",
                "Increase or retain the active pitch count as speed increases.",
            )
        if modes and modes[-1] != case.resolution.pitch:
            _fail(
                "resolution.pitch_modes_by_speed[-1]",
                modes[-1],
                f"the declared pitch maximum {case.resolution.pitch}",
                "Retain the declared maximum at the highest-speed node.",
            )
    _choice(
        "solver.method",
        case.solver.method,
        ("auto", "structured_direct", "recycled_krylov", "sparse_direct_referee"),
    )
    _positive("solver.relative_tolerance", case.solver.relative_tolerance, upper=1.0)
    _positive("solver.memory_fraction", case.solver.memory_fraction, upper=1.0)
    _choice("solver.reuse", case.solver.reuse, ("auto", "on", "off"))
    _choice("parallel.strategy", case.parallel.strategy, ("auto", "serial", "batch"))
    for index, shard in enumerate(case.parallel.shard):
        _choice(
            f"parallel.shard[{index}]", shard, ("surface", "electric_field", "species")
        )
    if len(set(case.parallel.shard)) != len(case.parallel.shard):
        _fail(
            "parallel.shard",
            case.parallel.shard,
            "unique shard axes",
            "Remove duplicate axes.",
        )
    _is_boolean("convergence.enabled", case.convergence.enabled)
    _is_boolean(
        "convergence.retain_legendre_tail",
        case.convergence.retain_legendre_tail,
    )
    if len(set(case.convergence.observables)) != len(case.convergence.observables):
        _fail(
            "convergence.observables",
            case.convergence.observables,
            "unique observable names",
            "Remove duplicate observables.",
        )
    _positive(
        "convergence.relative_tolerance", case.convergence.relative_tolerance, upper=1.0
    )
    if (
        isinstance(case.convergence.max_refinements, bool)
        or not isinstance(case.convergence.max_refinements, int)
        or case.convergence.max_refinements < 0
    ):
        _fail(
            "convergence.max_refinements",
            case.convergence.max_refinements,
            "an integer >= 0",
            "Use zero to disable refinements.",
        )
    _is_boolean("output.plots", case.output.plots)
    if case.scan is not None:
        _choice("scan.combine", case.scan.combine, ("cartesian", "zipped"))
        _is_boolean("scan.resume", case.scan.resume)
        if not case.scan.axes:
            _fail(
                "scan.axis",
                case.scan.axes,
                "one or more scan axes",
                "Add at least one [[scan.axis]] table.",
            )
        paths: set[str] = set()
        lengths: set[int] = set()
        for index, axis in enumerate(case.scan.axes):
            if not axis.path.strip() or axis.path in paths:
                _fail(
                    f"scan.axis[{index}].path",
                    axis.path,
                    "a unique non-empty schema path",
                    "Use a distinct field path for every axis.",
                )
            if not _is_supported_scan_path(axis.path, names):
                _fail(
                    f"scan.axis[{index}].path",
                    axis.path,
                    "a supported numeric Case path",
                    "Use electric_field.value_kV_m, a resolution/solver field, or species[NAME].density_scale/temperature_scale.",
                )
            paths.add(axis.path)
            if not axis.values:
                _fail(
                    f"scan.axis[{index}].values",
                    axis.values,
                    "one or more finite values",
                    "Add explicit scan values.",
                )
            lengths.add(len(axis.values))
        if case.scan.combine == "zipped" and len(lengths) > 1:
            _fail(
                "scan.axis",
                tuple(sorted(lengths)),
                "equal value counts for a zipped scan",
                "Make all zipped axes the same length or use combine = 'cartesian'.",
            )
        if (
            isinstance(case.scan.max_cases, bool)
            or not isinstance(case.scan.max_cases, int)
            or case.scan.max_cases < 1
        ):
            _fail(
                "scan.max_cases",
                case.scan.max_cases,
                "an integer >= 1",
                "Set a positive launch bound.",
            )
        if case.scan.case_count > case.scan.max_cases:
            _fail(
                "scan.axis",
                case.scan.case_count,
                f"at most scan.max_cases ({case.scan.max_cases}) cases",
                "Reduce axis values or raise max_cases deliberately after checking memory and runtime.",
            )


def _object_schema(required: list[str], **properties: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value


def _semantic_path(value: str | Path) -> Path:
    """Normalize lexical path aliases without binding them to a machine."""

    return Path(os.path.normpath(os.fspath(value)))


def _is_supported_scan_path(path: str, species_names: set[str]) -> bool:
    if path in {
        "electric_field.value_kV_m",
        "resolution.theta",
        "resolution.zeta",
        "resolution.pitch",
        "resolution.speed",
        "solver.relative_tolerance",
        "solver.memory_fraction",
    }:
        return True
    match = re.fullmatch(
        r"species\[([^\]]+)\]\.(density_scale|temperature_scale)", path
    )
    return match is not None and match.group(1) in species_names


def _reject_unknown(data: Mapping[str, Any], path: str, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        key = unknown[0]
        where = key if path == "$" else f"{path}.{key}"
        _fail(
            where,
            data[key],
            f"one of {sorted(allowed)!r}",
            "Remove the field or correct its spelling.",
        )


def _as_table(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, value, "a table/object", "Use a TOML table or JSON object here.")
    return value


def _table(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    required: bool = False,
    default: Any = None,
) -> Mapping[str, Any]:
    if key not in data:
        if required:
            _fail(
                key if path == "$" else f"{path}.{key}",
                None,
                "a table/object",
                f"Add the [{key}] table.",
            )
        return default
    return _as_table(data[key], key if path == "$" else f"{path}.{key}")


def _sequence(
    data: Mapping[str, Any], key: str, path: str, *, required: bool = False
) -> Sequence[Any]:
    where = key if path == "$" else f"{path}.{key}"
    if key not in data:
        if required:
            _fail(where, None, "a non-empty array", f"Add {where}.")
        return ()
    value = data[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, value, "an array", "Use square brackets or an array of tables.")
    return value


def _string(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    required: bool = False,
    default: str = "",
) -> str:
    where = key if path == "$" else f"{path}.{key}"
    if key not in data:
        if required:
            _fail(where, None, "a non-empty string", f"Add {where}.")
        return default
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        _fail(where, value, "a non-empty string", "Use a quoted descriptive value.")
    return value


def _boolean(data: Mapping[str, Any], key: str, path: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        _fail(
            f"{path}.{key}",
            value,
            "true or false",
            "Use an unquoted TOML/JSON boolean.",
        )
    return value


def _number(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    required: bool = False,
    default: float = 0.0,
) -> float:
    where = f"{path}.{key}"
    if key not in data:
        if required:
            _fail(where, None, "a finite number", f"Add {where}.")
        return default
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(
            where,
            value,
            "a finite number",
            "Use a finite integer or floating-point value.",
        )
    return float(value)


def _integer(
    data: Mapping[str, Any],
    key: str,
    path: str,
    *,
    required: bool = False,
    default: int = 0,
) -> int:
    where = key if path == "$" else f"{path}.{key}"
    if key not in data:
        if required:
            _fail(where, None, "an integer", f"Add {where}.")
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, value, "an integer", "Use a whole number without quotes.")
    return value


def _number_tuple(
    data: Mapping[str, Any], key: str, path: str, *, required: bool = False
) -> tuple[float, ...]:
    values = _sequence(data, key, path, required=required)
    result = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            _fail(
                f"{path}.{key}[{index}]",
                value,
                "a finite number",
                "Replace it with a finite numeric value.",
            )
        result.append(float(value))
    return tuple(result)


def _integer_tuple(
    data: Mapping[str, Any], key: str, path: str, *, required: bool = False
) -> tuple[int, ...]:
    values = _sequence(data, key, path, required=required)
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(
                f"{path}.{key}[{index}]",
                value,
                "an integer",
                "Use whole numbers without quotes.",
            )
        result.append(value)
    return tuple(result)


def _string_tuple(
    data: Mapping[str, Any], key: str, path: str, *, default: tuple[str, ...]
) -> tuple[str, ...]:
    if key not in data:
        return default
    values = _sequence(data, key, path)
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            _fail(
                f"{path}.{key}[{index}]",
                value,
                "a non-empty string",
                "Use a quoted descriptive value.",
            )
        result.append(value)
    return tuple(result)


def _choice(path: str, value: Any, choices: tuple[Any, ...]) -> None:
    if value not in choices:
        _fail(
            path,
            value,
            f"one of {choices!r}",
            f"Choose {', '.join(map(repr, choices))}.",
        )


def _is_boolean(path: str, value: Any) -> None:
    if not isinstance(value, bool):
        _fail(path, value, "true or false", "Use a boolean value.")


def _positive(path: str, value: float, *, upper: float | None = None) -> None:
    if (
        not math.isfinite(float(value))
        or value <= 0
        or (upper is not None and value > upper)
    ):
        expected = (
            "a finite number > 0"
            if upper is None
            else f"a finite number in (0, {upper}]"
        )
        _fail(
            path,
            value,
            expected,
            "Choose a value in the stated physical or numerical range.",
        )


def _bounded(path: str, value: float, lower: float, upper: float) -> None:
    if not math.isfinite(float(value)) or value < lower or value > upper:
        _fail(
            path,
            value,
            f"a finite number in [{lower}, {upper}]",
            "Choose a normalized radial surface in range.",
        )


def _fail(path: str, value: Any, expected: str, correction: str) -> None:
    raise CaseValidationError(path, value, expected, correction)


__all__ = [
    "COMMENTED_TOML_EXAMPLE",
    "Case",
    "CaseValidationError",
    "ConvergenceConfig",
    "ElectricFieldConfig",
    "GeometryConfig",
    "OutputConfig",
    "ParallelConfig",
    "PhysicsConfig",
    "ResolutionConfig",
    "RunConfig",
    "SCHEMA_VERSION",
    "ScanAxis",
    "ScanConfig",
    "SolverConfig",
    "SpeciesConfig",
    "case_json_schema",
    "migrate_case_data",
]
