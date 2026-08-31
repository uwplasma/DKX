"""Tests for ``dkx compare``.

plan.md section 5.6 lists one ``compare`` rather than a command per format, so
the dispatch and the verdict are what need pinning: which differences count,
which are reported without counting, and which pairs are refused outright.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dkx import cli


class FakeResult:
    def __init__(self, arrays):
        self.arrays = arrays
        self.case_name = "fake"
        self.case_id = "0123456789ab"
        self.workflow = "profile"
        self.metadata = {}


def loads(monkeypatch, left, right) -> None:
    seq = iter([left, right])
    monkeypatch.setattr("dkx.result.Result.load", staticmethod(lambda path: next(seq)))


def base(**overrides):
    arrays = {
        "particle_flux_m2_s": np.array([1.0, 2.0]),
        "heat_flux_W_m2": np.array([3.0, 4.0]),
        "solve_time_s": np.array([1.0]),
    }
    arrays.update(overrides)
    return FakeResult(arrays)


def test_identical_results_agree(monkeypatch, capsys) -> None:
    loads(monkeypatch, base(), base())
    assert cli.main(["compare", "a.nc", "b.nc"]) == 0
    assert "agree" in capsys.readouterr().out


def test_a_physics_difference_fails(monkeypatch, capsys) -> None:
    loads(monkeypatch, base(), base(heat_flux_W_m2=np.array([3.0, 5.0])))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 2
    assert "heat_flux_W_m2" in capsys.readouterr().out


def test_wall_clock_time_is_reported_but_does_not_fail_the_run(monkeypatch, capsys) -> None:
    """Two runs of one case always differ in timing.

    Counting it would make ``dkx compare`` exit non-zero on a bit-identical
    re-run, which trains the reader to ignore the exit status -- the one thing
    the command exists to make trustworthy.
    """
    loads(monkeypatch, base(), base(solve_time_s=np.array([9.0])))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 0
    out = capsys.readouterr().out
    assert "solve_time_s" in out and "informational" in out


def test_iteration_counts_are_informational_too(monkeypatch) -> None:
    """They move with warm starts without the answer changing."""
    loads(monkeypatch,
          FakeResult({"x": np.array([1.0]), "solver_iterations": np.array([10])}),
          FakeResult({"x": np.array([1.0]), "solver_iterations": np.array([14])}))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 0


def test_an_array_present_in_only_one_result_fails(monkeypatch, capsys) -> None:
    """A run that stopped producing an output is not a run that agrees.

    Comparing the intersection alone would report agreement on the keys that
    remain, which is how a dropped output becomes invisible.
    """
    loads(monkeypatch, base(), FakeResult({"particle_flux_m2_s": np.array([1.0, 2.0])}))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 2
    assert "only in A" in capsys.readouterr().out


def test_a_shape_change_is_reported_rather_than_broadcast(monkeypatch, capsys) -> None:
    """numpy would happily broadcast (2,) against (1,); that would hide a regrid."""
    loads(monkeypatch, base(), base(particle_flux_m2_s=np.array([1.0])))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 2
    assert "shape" in capsys.readouterr().out


def test_matching_nans_count_as_agreement(monkeypatch) -> None:
    """A quantity undefined in both runs has not changed between them."""
    nan = np.array([np.nan, 1.0])
    loads(monkeypatch, FakeResult({"q": nan.copy()}), FakeResult({"q": nan.copy()}))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 0


def test_a_nan_appearing_on_one_side_only_fails(monkeypatch) -> None:
    loads(monkeypatch,
          FakeResult({"q": np.array([1.0, 1.0])}),
          FakeResult({"q": np.array([np.nan, 1.0])}))
    assert cli.main(["compare", "a.nc", "b.nc"]) == 2


def test_tolerance_is_honoured(monkeypatch) -> None:
    loads(monkeypatch, base(), base(heat_flux_W_m2=np.array([3.0, 4.0 + 1e-12])))
    assert cli.main(["compare", "a.nc", "b.nc", "--rtol", "1e-6"]) == 0
    loads(monkeypatch, base(), base(heat_flux_W_m2=np.array([3.0, 4.0 + 1e-12])))
    assert cli.main(["compare", "a.nc", "b.nc", "--rtol", "1e-15"]) == 2


def test_json_output_is_machine_readable(monkeypatch, capsys) -> None:
    loads(monkeypatch, base(), base())
    assert cli.main(["compare", "a.nc", "b.nc", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["agree"] is True
    assert {r["key"] for r in report["rows"]} == {
        "particle_flux_m2_s", "heat_flux_W_m2", "solve_time_s"
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), [
    ("out.h5", True), ("out.HDF5", True), ("result.nc", False), ("result.NC", False),
])
def test_format_is_decided_by_extension_not_by_sniffing(name: str, expected: bool) -> None:
    """NetCDF4 *is* HDF5, so an h5py open succeeds on both.

    Sniffing would route every dkx Result into the SFINCS comparison, where the
    variable names do not exist.
    """
    from pathlib import Path

    assert cli._looks_like_sfincs_h5(Path(name)) is expected


def test_a_mixed_pair_is_refused_with_a_reason(monkeypatch, capsys) -> None:
    """Unmatched keys everywhere would otherwise read as agreement."""
    assert cli.main(["compare", "a.nc", "b.h5"]) == 2
    err = capsys.readouterr().err
    assert "different variable names" in err


def test_compare_is_a_registered_command_not_a_filename() -> None:
    assert cli._normalize_default_argv(["compare"]) == ["compare"]
