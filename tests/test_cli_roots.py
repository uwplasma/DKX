"""Tests for ``dkx roots``.

The command reads a stored Result rather than recomputing, so these build the
arrays directly instead of paying for an ambipolar solve. What needs pinning is
the reading and the reporting -- especially the two cases where saying nothing
would be misleading: a result that has no roots because it was never an
ambipolar run, and a root table that spans a nonsmooth branch event.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dkx import cli


class FakeResult:
    def __init__(self, arrays, *, workflow="ambipolar_profile"):
        self.arrays = arrays
        self.workflow = workflow
        self.case_name = "fake_case"
        self.case_id = "0123456789abcdef"
        self.metadata = {"converged": True}


def ambipolar_arrays(*, counts=(2, 1), nonsmooth=(0, 0)):
    """Two surfaces: the first with an ion/electron pair, the second with one root."""
    return {
        "ambipolar_root_kV_m": np.array([[-1.8, 2.4], [-1.5, 0.0]]),
        "ambipolar_root_count": np.array(counts),
        "ambipolar_root_current_A_m2": np.array([[-0.46, 0.11], [0.01, 0.0]]),
        "ambipolar_root_slope_A_m2_per_kV_m": np.array([[1.0, 1.0], [1.0, 0.0]]),
        "ambipolar_root_type": np.array([["ion", "electron"], ["ion", ""]], dtype=object),
        "ambipolar_root_final_bracket_width_kV_m": np.array([[3.9e-2, 3.9e-2], [3.9e-2, 0.0]]),
        "selected_ambipolar_root": np.array([0, 0]),
        "ambipolar_nonsmooth_event": np.array(nonsmooth, dtype=np.int8),
    }


def load(monkeypatch, result) -> None:
    monkeypatch.setattr("dkx.result.Result.load", staticmethod(lambda path: result))


def test_every_root_on_every_surface_is_listed(monkeypatch, capsys) -> None:
    load(monkeypatch, FakeResult(ambipolar_arrays()))
    assert cli.main(["roots", "any.nc", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [(r["surface"], r["root"]) for r in report["roots"]] == [(0, 0), (0, 1), (1, 0)]
    assert report["roots"][0]["type"] == "ion"
    assert report["roots"][1]["type"] == "electron"


def test_only_the_admitted_roots_are_listed_not_the_padded_slots(monkeypatch, capsys) -> None:
    """The arrays are rectangular; ``ambipolar_root_count`` says how much is real.

    Surface 1 has one root in a width-2 array. Reading the array shape instead
    of the count would report a second root at exactly ``E_r = 0`` with zero
    current -- a fabricated ion root that looks entirely plausible.
    """
    load(monkeypatch, FakeResult(ambipolar_arrays(counts=(2, 1))))
    assert cli.main(["roots", "any.nc", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert sum(r["surface"] == 1 for r in report["roots"]) == 1


def test_the_selected_root_is_marked(monkeypatch, capsys) -> None:
    load(monkeypatch, FakeResult(ambipolar_arrays()))
    cli.main(["roots", "any.nc", "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert [r["selected"] for r in report["roots"]] == [True, False, True]


def test_a_nonsmooth_branch_event_is_reported(monkeypatch, capsys) -> None:
    """A branch appearing or vanishing makes the output non-differentiable there.

    jax.grad still returns a number across such an interval, so a gradient that
    is meaningless is indistinguishable from one that is not unless the event is
    surfaced. plan.md section 7.6 requires these be marked rather than
    differentiated through as if smooth.
    """
    load(monkeypatch, FakeResult(ambipolar_arrays(nonsmooth=(0, 1))))
    assert cli.main(["roots", "any.nc"]) == 0
    assert "onsmooth" in capsys.readouterr().out


def test_a_smooth_result_says_nothing_about_events(monkeypatch, capsys) -> None:
    load(monkeypatch, FakeResult(ambipolar_arrays(nonsmooth=(0, 0))))
    cli.main(["roots", "any.nc"])
    assert "onsmooth" not in capsys.readouterr().out


def test_a_non_ambipolar_result_is_refused_by_name(monkeypatch, capsys) -> None:
    """The error must say which workflow would produce roots.

    Printing an empty table here would read as "this configuration has no
    ambipolar roots", which is a physics claim, when the truth is that the run
    never looked for any.
    """
    load(monkeypatch, FakeResult({"particle_flux_m2_s": np.zeros(2)}, workflow="profile"))
    assert cli.main(["roots", "any.nc"]) == 2
    err = capsys.readouterr().err
    assert "ambipolar_profile" in err and "profile" in err


def test_an_ambipolar_run_that_found_nothing_says_why_that_is_not_proof(
    monkeypatch, capsys
) -> None:
    """Zero admitted roots is not evidence that none exist.

    Sign sampling cannot see a tangential root or an even number of crossings
    between samples -- both are proved in tests/test_numerics.py. The command
    has to carry that caveat or it silently converts "we did not find one" into
    "there is none".
    """
    arrays = ambipolar_arrays(counts=(0, 0))
    load(monkeypatch, FakeResult(arrays))
    assert cli.main(["roots", "any.nc"]) == 0
    out = capsys.readouterr().out
    assert "tangential" in out and "even number of crossings" in out


def test_bytes_valued_types_are_decoded(monkeypatch, capsys) -> None:
    """NetCDF round-trips string arrays as bytes; the table must not print b'ion'."""
    arrays = ambipolar_arrays()
    arrays["ambipolar_root_type"] = np.array([[b"ion", b"electron"], [b"ion", b""]], dtype=object)
    load(monkeypatch, FakeResult(arrays))
    cli.main(["roots", "any.nc", "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert report["roots"][0]["type"] == "ion"


def test_a_missing_file_fails_with_a_message_not_a_traceback(monkeypatch, capsys) -> None:
    def raise_oserror(path):
        raise OSError("no such file")

    monkeypatch.setattr("dkx.result.Result.load", staticmethod(raise_oserror))
    assert cli.main(["roots", "missing.nc"]) == 2
    assert "dkx roots failed" in capsys.readouterr().err


def test_roots_is_a_registered_command_not_a_filename() -> None:
    assert cli._normalize_default_argv(["roots"]) == ["roots"]
