"""Tests for the declarative scan workflow behind ``[scan]``.

``[scan]`` was parsed and validated by the case schema from the start, then
refused by ``execution.run_case`` with a message naming a ``dkx.scan`` module
that did not exist. This is that module.

The expansion and bookkeeping are tested against a stub solve. What needs
pinning is which cases get built, what happens when one fails, and that resume
does not destroy the results it exists to preserve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from dkx.workflows import scan as sc


@dataclass(frozen=True)
class Axis:
    path: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class Scan:
    axes: tuple[Axis, ...]
    combine: str = "cartesian"
    resume: bool = True
    output: Path = Path("scan.nc")
    max_cases: int = 100


@dataclass(frozen=True)
class Field:
    value_kV_m: float = 0.0


@dataclass(frozen=True)
class Resolution:
    theta: int = 9
    zeta: int = 1
    pitch: int = 8
    speed: int = 4


@dataclass(frozen=True)
class Solver:
    relative_tolerance: float = 1e-8
    memory_fraction: float = 0.75


@dataclass(frozen=True)
class Species:
    name: str
    density_m3: tuple[float, ...] = (1.0, 2.0)
    temperature_keV: tuple[float, ...] = (3.0, 4.0)


@dataclass(frozen=True)
class Run:
    workflow: str = "profile"


@dataclass(frozen=True)
class Case:
    scan: Scan | None
    electric_field: Field = Field()
    resolution: Resolution = Resolution()
    solver: Solver = Solver()
    species: tuple[Species, ...] = (Species("deuterium"),)
    run: Run = Run()
    name: str = "case"

    @property
    def case_id(self) -> str:
        return (
            f"{self.electric_field.value_kV_m}|{self.resolution}|{self.solver}|"
            f"{self.species}"
        )


# --------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------


def test_cartesian_takes_every_combination() -> None:
    case = Case(Scan((Axis("resolution.theta", (9, 11)), Axis("resolution.pitch", (4, 6, 8)))))
    assert len(sc.expand_scan(case)) == 6


def test_zipped_walks_the_axes_together() -> None:
    case = Case(
        Scan(
            (Axis("resolution.theta", (9, 11, 13)), Axis("resolution.pitch", (4, 6, 8))),
            combine="zipped",
        )
    )
    points = [point for point, _ in sc.expand_scan(case)]
    assert points == [(9, 4), (11, 6), (13, 8)]


def test_derived_cases_carry_no_scan_of_their_own() -> None:
    """Otherwise running one would recurse into another scan."""
    case = Case(Scan((Axis("resolution.theta", (9, 11)),)))
    assert all(derived.scan is None for _, derived in sc.expand_scan(case))


def test_every_derived_case_has_a_distinct_identity() -> None:
    """Resume keys on case_id, so a collision would silently skip real work."""
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0, 1.0)),)))
    ids = {derived.case_id for _, derived in sc.expand_scan(case)}
    assert len(ids) == 3


@pytest.mark.parametrize(
    ("path", "check"),
    [
        ("electric_field.value_kV_m", lambda c: c.electric_field.value_kV_m == 2.0),
        ("resolution.theta", lambda c: c.resolution.theta == 2),
        ("solver.relative_tolerance", lambda c: c.solver.relative_tolerance == 2.0),
    ],
)
def test_each_supported_axis_path_reaches_its_field(path: str, check) -> None:
    case = Case(Scan((Axis(path, (2.0,)),)))
    _, derived = sc.expand_scan(case)[0]
    assert check(derived)


def test_a_species_scale_multiplies_that_species_only() -> None:
    """The axis is a *scale*, not a value: it must scale the whole profile."""
    case = Case(
        Scan((Axis("species[deuterium].density_scale", (3.0,)),)),
        species=(Species("deuterium"), Species("carbon")),
    )
    _, derived = sc.expand_scan(case)[0]
    assert derived.species[0].density_m3 == (3.0, 6.0)
    assert derived.species[1].density_m3 == (1.0, 2.0)


def test_a_temperature_scale_leaves_density_alone() -> None:
    case = Case(Scan((Axis("species[deuterium].temperature_scale", (2.0,)),)))
    _, derived = sc.expand_scan(case)[0]
    assert derived.species[0].temperature_keV == (6.0, 8.0)
    assert derived.species[0].density_m3 == (1.0, 2.0)


def test_a_scan_larger_than_its_limit_is_refused_before_solving() -> None:
    """The limit exists so a typo in an axis cannot start a thousand solves."""
    case = Case(Scan((Axis("resolution.theta", tuple(range(50))),), max_cases=10))
    with pytest.raises(ValueError, match="above the 10 limit"):
        sc.expand_scan(case)


def test_a_case_without_a_scan_table_is_refused() -> None:
    with pytest.raises(ValueError, match="no \\[scan\\] table"):
        sc.expand_scan(Case(None))


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


class FakeResult:
    def __init__(self, value: float):
        self.arrays = {"particle_flux_m2_s": np.array([value])}


def run(monkeypatch, tmp_path, case, *, fail_on=(), resume=None):
    calls: list = []

    def fake_run_case(derived, **_):
        calls.append(derived)
        value = derived.electric_field.value_kV_m
        if value in fail_on:
            raise ValueError(f"unsupported at {value}")
        return FakeResult(value * 10.0)

    monkeypatch.setattr("dkx.execution.run_case", fake_run_case)
    result, failures = sc.run_scan(case, out=tmp_path / "s.nc", resume=resume)
    return result, failures, calls


def test_a_scan_records_every_case(monkeypatch, tmp_path) -> None:
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0, 1.0)),)))
    result, failures, calls = run(monkeypatch, tmp_path, case)
    assert len(calls) == 3 and failures == 0
    assert list(np.asarray(result.arrays["particle_flux_m2_s"])) == [10.0, 0.0, 10.0]


def test_one_failing_case_does_not_discard_the_others(monkeypatch, tmp_path) -> None:
    """Losing completed solves because a later point failed is the wrong trade.

    A scan is run precisely because each point is expensive. The failure is
    recorded in the row and reported through the return value, so nothing is
    hidden, but the cases that did work are still written.
    """
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0, 1.0)),)))
    result, failures, _ = run(monkeypatch, tmp_path, case, fail_on=(0.0,))
    assert failures == 1
    statuses = [str(s) for s in np.asarray(result.arrays["status"])]
    assert statuses[0] == "ok" and statuses[2] == "ok"
    assert "unsupported at 0.0" in statuses[1]
    assert result.metadata["scan_succeeded"] == 2
    assert result.metadata["converged"] is False


def test_the_axis_values_are_written_beside_the_observables(monkeypatch, tmp_path) -> None:
    """The output must say what was varied without needing the case file too."""
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 1.0)),)))
    result, _, _ = run(monkeypatch, tmp_path, case)
    assert list(np.asarray(result.arrays["axis_electric_field_value_kV_m"])) == [-1.0, 1.0]


def test_resume_skips_finished_cases(monkeypatch, tmp_path) -> None:
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0, 1.0)),)))
    run(monkeypatch, tmp_path, case)
    _, _, calls = run(monkeypatch, tmp_path, case)
    assert calls == []


def test_resume_preserves_the_results_it_skipped(monkeypatch, tmp_path) -> None:
    """Regression: resuming a finished scan used to write an empty table.

    Carrying only the case *ids* forward meant the rewritten output contained
    just the freshly-run cases -- none, when everything was cached -- so
    resuming a completed scan destroyed exactly the results resume exists to
    preserve.
    """
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0, 1.0)),)))
    first, _, _ = run(monkeypatch, tmp_path, case)
    second, _, calls = run(monkeypatch, tmp_path, case)
    assert calls == []
    assert second.metadata["scan_cases"] == 3
    assert np.array_equal(
        np.asarray(first.arrays["particle_flux_m2_s"]),
        np.asarray(second.arrays["particle_flux_m2_s"]),
    )


def test_resume_off_reruns_everything(monkeypatch, tmp_path) -> None:
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0, 0.0)),)))
    run(monkeypatch, tmp_path, case)
    _, _, calls = run(monkeypatch, tmp_path, case, resume=False)
    assert len(calls) == 2


def test_an_unreadable_output_is_not_trusted_as_a_cache(monkeypatch, tmp_path) -> None:
    """A truncated file is what an interrupted scan leaves behind."""
    (tmp_path / "s.nc").write_bytes(b"not a netcdf file")
    case = Case(Scan((Axis("electric_field.value_kV_m", (-1.0,)),)))
    _, _, calls = run(monkeypatch, tmp_path, case)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Against a real solve
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTIC_CASE = REPO_ROOT / "examples" / "01_tokamak_profile" / "case.toml"


def test_a_real_two_point_scan_runs_and_resumes(tmp_path) -> None:
    """End to end on the shipped analytic tokamak deck.

    The stub tests above pin the bookkeeping; this one pins that a derived case
    is actually runnable -- that ``replace(case, scan=None)`` plus an axis
    substitution produces a Case the executor accepts, which no amount of
    stubbing can show.
    """
    from dkx.config import Case
    from dkx.result import Result
    from dkx.workflows.scan import run_scan

    source = ANALYTIC_CASE.read_text(encoding="utf-8")
    case_file = tmp_path / "scan.toml"
    case_file.write_text(
        source
        + '\n[scan]\ncombine = "cartesian"\nresume = true\noutput = "s.nc"\n'
        + '\n[[scan.axis]]\npath = "electric_field.value_kV_m"\nvalues = [0.0, 1.0]\n',
        encoding="utf-8",
    )
    case = Case.from_file(case_file)
    out = tmp_path / "s.nc"

    result, failures = run_scan(case, out=out)
    assert failures == 0
    assert result.metadata["scan_cases"] == 2
    fluxes = np.asarray(result.arrays["particle_flux_m2_s"])
    assert np.all(np.isfinite(fluxes))
    # A finite E_r must move the transport; identical rows would mean the axis
    # never reached the solve.
    assert fluxes[0] != fluxes[1]

    resumed, _ = run_scan(case, out=out)
    assert resumed.metadata["scan_cases"] == 2
    assert np.array_equal(np.asarray(Result.load(out).arrays["particle_flux_m2_s"]), fluxes)


# --------------------------------------------------------------------------
# The `dkx scan` command
# --------------------------------------------------------------------------


def scan_case_file(tmp_path, values="[0.0, 1.0]") -> Path:
    source = ANALYTIC_CASE.read_text(encoding="utf-8")
    path = tmp_path / "scan.toml"
    path.write_text(
        source
        + '\n[scan]\ncombine = "cartesian"\nresume = true\noutput = "s.nc"\n'
        + f'\n[[scan.axis]]\npath = "electric_field.value_kV_m"\nvalues = {values}\n',
        encoding="utf-8",
    )
    return path


def test_the_command_runs_a_scan_and_exits_zero(tmp_path, capsys) -> None:
    from dkx import cli

    out = tmp_path / "out.nc"
    assert cli.main(["scan", str(scan_case_file(tmp_path)), "--out", str(out), "--quiet"]) == 0
    assert out.exists()


def test_a_case_without_a_scan_table_is_refused_by_name(tmp_path, capsys) -> None:
    """The error must point at the alternative rather than just failing."""
    from dkx import cli

    assert cli.main(["scan", str(ANALYTIC_CASE)]) == 2
    err = capsys.readouterr().err
    assert "no [scan] table" in err and "dkx run" in err


def test_a_failing_point_exits_one_but_still_writes(monkeypatch, tmp_path, capsys) -> None:
    """Non-zero says something failed; the file says which, and keeps the rest."""
    from dkx import cli
    from dkx.result import Result

    calls = {"n": 0}

    def flaky(derived, **_):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("unsupported combination")
        return FakeResult(float(calls["n"]))

    monkeypatch.setattr("dkx.execution.run_case", flaky)
    out = tmp_path / "out.nc"
    assert cli.main(["scan", str(scan_case_file(tmp_path)), "--out", str(out), "--quiet"]) == 1
    assert "1 of 2 cases failed" in capsys.readouterr().err
    statuses = [str(s) for s in np.asarray(Result.load(out).arrays["status"])]
    assert statuses[0] == "ok"
    assert "unsupported combination" in statuses[1]


def test_no_resume_reruns_every_point(monkeypatch, tmp_path) -> None:
    from dkx import cli

    calls = {"n": 0}

    def counting(derived, **_):
        calls["n"] += 1
        return FakeResult(float(calls["n"]))

    monkeypatch.setattr("dkx.execution.run_case", counting)
    case_file = scan_case_file(tmp_path)
    out = tmp_path / "out.nc"
    cli.main(["scan", str(case_file), "--out", str(out), "--quiet"])
    before = calls["n"]
    cli.main(["scan", str(case_file), "--out", str(out), "--quiet"])
    assert calls["n"] == before, "the second run should have been fully cached"
    cli.main(["scan", str(case_file), "--out", str(out), "--quiet", "--no-resume"])
    assert calls["n"] == before * 2


def test_scan_is_a_registered_command_not_a_filename() -> None:
    from dkx import cli

    assert cli._normalize_default_argv(["scan"]) == ["scan"]
    assert "scan" in cli._USER_COMMANDS
