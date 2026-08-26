"""One unsolvable scan point must not discard the rest of the scan.

Reported from a scanType=5 run: tier-2 stalled at the largest |Er| on a
66004-DOF deck, the exception propagated out of ``run_dkx``, and every
remaining Er point at that radius was lost.  One radius folder finished with
zero of a hundred outputs, another with three.  The loss was silent, and it
also stranded the process inside the failed run's directory, because the
scan loops ``chdir`` back only *after* the solve returns.

A point that will not solve is information, not a reason to throw away the
other ninety-nine.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "examples" / "sfincs_examples" / "utils"
SCAN_SCRIPTS = ("sfincsScan_1", "sfincsScan_2", "sfincsScan_3",
                "sfincsScan_4", "sfincsScan_21", "sfincsScan_22")


def _load_common(tmp_path, monkeypatch):
    """Exec sfincsScan_common the way sfincsScan does, in its own namespace."""
    monkeypatch.syspath_prepend(str(UTILS))
    namespace: dict = {"__file__": str(UTILS / "sfincsScan_common")}
    exec((UTILS / "sfincsScan_common").read_text(), namespace)  # noqa: S102
    return namespace


def test_a_failing_point_is_recorded_and_the_scan_continues(tmp_path, monkeypatch):
    ns = _load_common(tmp_path, monkeypatch)
    run_scan_point = ns["run_scan_point"]

    # Stand in for dkx_driver: fail only the third point, as a stalled solve
    # would, and succeed on the rest.
    calls = []

    class _FakeDriver:
        @staticmethod
        def run_dkx(*, input_namelist, output_path, **kwargs):
            calls.append(Path(output_path).parent.name)
            if Path(output_path).parent.name == "Er3":
                raise RuntimeError("the linear solve did not converge at total_size=66004")
            Path(output_path).write_text("ok")

    monkeypatch.setitem(sys.modules, "dkx_driver", _FakeDriver)

    ok = []
    for i in range(5):
        directory = tmp_path / f"Er{i + 1}"
        directory.mkdir()
        (directory / "input.namelist").write_text("&general\n/\n")
        ok.append(
            run_scan_point(
                input_namelist=directory / "input.namelist",
                output_path=directory / "sfincsOutput.h5",
                directory=directory.name,
            )
        )

    # Every point was attempted, not just the ones before the failure.
    assert len(calls) == 5
    assert ok == [True, True, False, True, True]

    # The four that solved have output; the one that did not is marked.
    assert (tmp_path / "Er4" / "sfincsOutput.h5").is_file()
    assert not (tmp_path / "Er3" / "sfincsOutput.h5").exists()
    marker = tmp_path / "Er3" / ns["FAILURE_MARKER"]
    assert marker.is_file()
    assert "did not converge" in marker.read_text()

    assert ns["report_scan_failures"]() == 1


def test_keyboard_interrupt_still_stops_the_scan(tmp_path, monkeypatch):
    """Catching solver failures must not swallow the user pressing Ctrl-C."""
    ns = _load_common(tmp_path, monkeypatch)

    class _Interrupting:
        @staticmethod
        def run_dkx(**kwargs):
            raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "dkx_driver", _Interrupting)
    directory = tmp_path / "Er1"
    directory.mkdir()
    with pytest.raises(KeyboardInterrupt):
        ns["run_scan_point"](
            input_namelist=directory / "input.namelist",
            output_path=directory / "sfincsOutput.h5",
            directory="Er1",
        )


@pytest.mark.parametrize("script", SCAN_SCRIPTS)
def test_every_scan_type_uses_the_resilient_helper(script: str) -> None:
    """A raw run_dkx call in a scan loop is the bug returning."""
    text = (UTILS / script).read_text()
    assert "run_scan_point(" in text, f"{script} does not use the resilient helper"
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "    run_dkx(" not in body and "   run_dkx(" not in body, (
        f"{script} still calls run_dkx directly, so one bad point aborts the scan"
    )
