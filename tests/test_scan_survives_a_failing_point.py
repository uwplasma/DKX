"""One unsolvable scan point must not discard the rest of the scan.

Reported from a scanType=5 run: recycled Krylov stalled at the largest |Er| on a
66004-DOF deck, the exception propagated out of ``run_dkx``, and every
remaining Er point at that radius was lost.  One radius folder finished with
zero of a hundred outputs, another with three.  The loss was silent, and it
also stranded the process inside the failed run's directory, because the
scan loops ``chdir`` back only *after* the solve returns.

A point that will not solve is information, not a reason to throw away the
other ninety-nine.
"""

import ast
import os
import shutil
import subprocess

import h5py
import numpy as np
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1] / "examples" / "sfincs_examples" / "utils"
SCAN_SCRIPTS = ("sfincsScan_1", "sfincsScan_2", "sfincsScan_3",
                "sfincsScan_4", "sfincsScan_21", "sfincsScan_22")


def _validator_source():
    text = (UTILS / "dkx_driver.py").read_text()
    return ("_EQUILIBRIUM_DIGESTS = {}\n"
            "from dkx.input_compat import effective_equilibrium_file, _resolve_equilibrium_file_from_namelist\n") + "\n\n".join(ast.get_source_segment(text, n) for n in ast.parse(text).body
                         if isinstance(n, ast.FunctionDef)
                         and n.name in {"output_is_complete", "output_matches_input", "scan_input_fingerprint"})


def _driver_stub():
    from types import SimpleNamespace

    ns = {"Path": Path}
    exec(_validator_source(), ns)
    return SimpleNamespace(scan_input_fingerprint=ns["scan_input_fingerprint"], output_is_complete=ns["output_is_complete"],
                           output_matches_input=ns["output_matches_input"])


def _write_complete(path, mode=1):
    with h5py.File(path, "w") as f:
        f["RHSMode"] = mode
        f["integerToRepresentTrue"] = 1
        f["finished"] = 1
        if mode == 1:
            for key in ["FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"]:
                f[key] = [[1.0]]
            f["FSABjHat"] = [1.0]
        else:
            f["transportMatrix"] = np.eye(3 if mode == 2 else 2)


def _load_common(tmp_path, monkeypatch):
    """Exec sfincsScan_common the way sfincsScan does, in its own namespace."""
    monkeypatch.syspath_prepend(str(UTILS))
    monkeypatch.setitem(sys.modules, "dkx_driver", _driver_stub())
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
            _write_complete(output_path)

    monkeypatch.setattr(sys.modules["dkx_driver"], "run_dkx", _FakeDriver.run_dkx, raising=False)

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

    monkeypatch.setattr(sys.modules["dkx_driver"], "run_dkx", _Interrupting.run_dkx, raising=False)
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


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("defect", [None, "partial", "truncated", "unfinished", "nan", "empty", "nonlinear"])
def test_scan_output_requires_complete_finite_finished_data(tmp_path, mode, defect):
    path = tmp_path / "sfincsOutput.h5"
    _write_complete(path, mode)
    key = "FSABFlow" if mode == 1 else "transportMatrix"
    if defect == "truncated":
        path.write_bytes(b"partial hdf5")
    elif defect:
        with h5py.File(path, "a") as f:
            if defect == "partial":
                del f[key]
            if defect == "unfinished":
                f["finished"][...] = 0
            if defect == "nan":
                f[key][...] = np.nan
            if defect == "empty":
                del f[key]
                f[key] = []
            if defect == "nonlinear":
                f["didNonlinearCalculationConverge"] = -1
    assert _driver_stub().output_is_complete(path) is (defect is None)


@pytest.fixture
def legacy_dispatcher(tmp_path):
    """Real dispatcher/point/radius loops, tiny fake solver, no kinetic solves."""
    utils = tmp_path / "utils"
    shutil.copytree(UTILS, utils)
    (utils / "dkx_driver.py").write_text(
        "from pathlib import Path\nimport os\nimport h5py\n" + _validator_source() + "\n" +
        "def run_dkx(*, input_namelist, output_path, **kwargs):\n"
        "    directory = Path(output_path).resolve().parent\n"
        "    with (directory / 'calls.txt').open('a') as log: log.write('called\\n')\n"
        "    with h5py.File(output_path, 'w') as f:\n"
        "        f['input.namelist'] = Path(input_namelist).read_text()\n"
        "        f['RHSMode'] = 1; f['integerToRepresentTrue'] = 1; f['finished'] = 1\n"
        "        if os.environ.get('SCAN_TEST_FAIL_POINT') == directory.name:\n"
        "            raise RuntimeError('synthetic point failure')\n"
        "        if os.environ.get('SCAN_TEST_PARTIAL_POINT') == directory.name: return output_path\n"
        "        for key in ['FSABFlow','FSABjHat','particleFlux_vm_psiHat','heatFlux_vm_psiHat']:\n"
        "            f[key] = [[1.0]]\n"
        "        f.attrs['dkx_scan_input_fingerprint'] = scan_input_fingerprint(Path(input_namelist).read_text(), Path(input_namelist))\n"
        "    return output_path\n"
    )
    # Supply two synthetic profiles; exercise real nested dispatch, not profile interpolation.
    (utils / "radialScans").write_text(
        "radii=[0.2,0.4]\ndirectories=['radius_a','radius_b']\n"
        "radiusName='rN'\nradiusNameForGradients='rHat'\ngeneralErName='Er'\n"
        "Nspecies=1\nnHats=THats=[[1.,1.]]\ndnHatdradii=dTHatdradii=[[0.,0.]]\n"
        "NErs=[3,3]\ngeneralEr_min=[-1.,-1.]\ngeneralEr_max=[1.,1.]\n"
    )
    def launch(directory, scan_type=2, **flags):
        directory.mkdir(exist_ok=True)
        deck = directory / "input.namelist"
        if not deck.exists():
            deck.write_text(f"!ss scanType = {scan_type}\n!ss NErs = 3\n!ss ErMin = -1\n!ss ErMax = 1\n"
                            "&geometryParameters\n inputRadialCoordinateForGradients = 4\n/\n"
                            "&speciesParameters\n/\n&physicsParameters\n Er = 0\n/\n")
        env = dict(os.environ, **flags)
        return subprocess.run([sys.executable, str(utils / "sfincsScan"), "--yes", "--input", str(deck)],
                              env=env, capture_output=True, text=True, timeout=20)
    return launch


def test_er_scan_retries_only_failed_or_incomplete_points(legacy_dispatcher, tmp_path):
    root = tmp_path / "scan"
    first = legacy_dispatcher(root, SCAN_TEST_FAIL_POINT="Er0")
    assert first.returncode == 1, first.stdout + first.stderr
    assert "FAILED Er0" in first.stdout and "done Er0" not in first.stdout
    assert (root / "Er-1/sfincsOutput.h5").exists()
    accepted = (root / "Er1/sfincsOutput.h5").read_bytes()
    old_input = (root / "Er0/input.namelist").read_bytes()
    partial = (root / "Er0/sfincsOutput.h5").read_bytes()
    # Also simulate a killed job which never got far enough to create a failure marker.
    (root / "Er-1/sfincsOutput.h5").write_bytes(b"truncated")
    (root / "Er0/dkx_state.npz").write_bytes(b"failed state")
    second = legacy_dispatcher(root)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (root / "Er1/sfincsOutput.h5").read_bytes() == accepted
    assert (root / "Er1/calls.txt").read_text().count("called") == 1
    assert (root / "Er0/calls.txt").read_text().count("called") == 2
    assert (root / "Er-1/calls.txt").read_text().count("called") == 2
    history = root / "Er0/.dkx-failed-attempts/1"
    assert (history / "input.namelist").read_bytes() == old_input
    assert (history / "sfincsOutput.h5").read_bytes() == partial
    assert (history / "dkx_FAILED.txt").exists()
    assert (history / "dkx_state.npz").read_bytes() == b"failed state"
    assert not (root / "Er0/dkx_FAILED.txt").exists()
    third = legacy_dispatcher(root)
    assert third.returncode == 0 and "start Er" not in third.stdout


def test_zero_exit_partial_output_is_a_failure(legacy_dispatcher, tmp_path):
    root = tmp_path / "scan"
    result = legacy_dispatcher(root, SCAN_TEST_PARTIAL_POINT="Er0")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "missing, partial or unsuccessful output" in result.stdout
    assert (root / "Er-1/sfincsOutput.h5").exists()


def test_nested_radius_failures_propagate_and_retry(legacy_dispatcher, tmp_path):
    root = tmp_path / "radial"
    first = legacy_dispatcher(root, scan_type=5, SCAN_TEST_FAIL_POINT="Er0")
    assert first.returncode == 1, first.stdout + first.stderr
    for radius in ["radius_a", "radius_b"]:
        assert (root / radius / "Er-1/sfincsOutput.h5").exists()
        assert (root / radius / "dkx_FAILED.txt").exists()
        assert f"FAILED {radius}" in first.stdout
    second = legacy_dispatcher(root, scan_type=5)
    assert second.returncode == 0, second.stdout + second.stderr
    for radius in ["radius_a", "radius_b"]:
        assert (root / radius / "Er1/calls.txt").read_text().count("called") == 1
        assert (root / radius / "Er0/calls.txt").read_text().count("called") == 2
        assert not (root / radius / "dkx_FAILED.txt").exists()


def test_child_launch_error_is_recorded_and_interrupt_propagates(tmp_path, monkeypatch):
    ns = _load_common(tmp_path, monkeypatch)
    assert not ns["run_scan_command"]([str(tmp_path / "missing-executable")], tmp_path)
    assert ns["report_scan_failures"]() == 1
    def interrupt(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        ns["run_scan_command"](["unused"], tmp_path)


@pytest.mark.parametrize("complete", [False, True])
def test_driver_never_reports_done_for_partial_output(tmp_path, monkeypatch, capsys, complete):
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("scan_driver_under_test", UTILS / "dkx_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    output = tmp_path / "sfincsOutput.h5"
    _write_complete(output)
    if not complete:
        with h5py.File(output, "a") as f:
            del f["FSABFlow"]
    (tmp_path / "input.namelist").write_text("&general\n/\n")
    with h5py.File(output, "a") as f:
        f["input.namelist"] = (tmp_path / "input.namelist").read_text()
    monkeypatch.setattr(driver, "read_sfincs_input", lambda _: SimpleNamespace(group=lambda _: {}))
    monkeypatch.setattr(driver, "write_output", lambda *a, **kw: output)
    if complete:
        assert driver.run_dkx(input_namelist=tmp_path / "input.namelist", ensure_equilibrium=False) == output
    else:
        with pytest.raises(RuntimeError, match="Incomplete"):
            driver.run_dkx(input_namelist=tmp_path / "input.namelist", ensure_equilibrium=False)
    assert ("dkx_driver: done" in capsys.readouterr().out) is complete
    with h5py.File(output, "r") as f:
        assert ("dkx_scan_input_fingerprint" in f.attrs) is complete
    if complete:
        assert driver.output_matches_input(output, (tmp_path / "input.namelist").read_text(),
                                           tmp_path / "input.namelist")


def test_success_retires_failure_marker_and_keeps_history(tmp_path, monkeypatch):
    ns = _load_common(tmp_path, monkeypatch)
    marker = tmp_path / ns["FAILURE_MARKER"]
    marker.write_text("previous failure")
    monkeypatch.setattr(sys.modules["dkx_driver"], "run_dkx",
                        lambda **kw: _write_complete(kw["output_path"]), raising=False)
    assert ns["run_scan_point"](input_namelist=tmp_path / "input.namelist",
                                output_path=tmp_path / "sfincsOutput.h5")
    assert not marker.exists()
    assert (tmp_path / ".dkx-failed-attempts/recovered-failures.txt").read_text() == "previous failure"


@pytest.mark.parametrize("code", [0, 2])
def test_solver_system_exit_is_a_failed_point_not_scan_success(tmp_path, monkeypatch, code):
    ns = _load_common(tmp_path, monkeypatch)
    def stop(**kwargs):
        raise SystemExit(code)
    monkeypatch.setattr(sys.modules["dkx_driver"], "run_dkx", stop, raising=False)
    assert not ns["run_scan_point"](input_namelist=tmp_path / "input.namelist",
                                    output_path=tmp_path / "sfincsOutput.h5")
    assert ns["report_scan_failures"]() == 1


def test_changed_requested_input_does_not_reuse_stale_output(legacy_dispatcher, tmp_path):
    root = tmp_path / "scan"
    assert legacy_dispatcher(root).returncode == 0
    accepted = (root / "Er1/sfincsOutput.h5").read_bytes()
    deck = root / "input.namelist"
    deck.write_text(deck.read_text().replace("&speciesParameters", "&speciesParameters\n nHats = 2"))
    # An edited point input must not make an OLD output look like the new request.
    child = root / "Er1/input.namelist"
    child.write_text(child.read_text().replace("&speciesParameters", "&speciesParameters\n nHats = 2"))
    result = legacy_dispatcher(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "Er1/calls.txt").read_text().count("called") == 2
    assert (root / "Er1/.dkx-failed-attempts/1/sfincsOutput.h5").read_bytes() == accepted
    assert legacy_dispatcher(root).returncode == 0
    assert (root / "Er1/calls.txt").read_text().count("called") == 2


def test_localized_equilibrium_and_comments_match_but_changed_content_does_not(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run = tmp_path / "point"
    run.mkdir()
    (source / "field.nc").write_bytes(b"same geometry")
    (run / "field.nc").write_bytes(b"same geometry")
    text = '&geometryParameters\n geometryScheme = 5\n equilibriumFile = "field.nc"\n/\n'
    output = run / "sfincsOutput.h5"
    with h5py.File(output, "w") as f:
        f["input.namelist"] = text
        f.attrs["dkx_scan_input_fingerprint"] = _driver_stub().scan_input_fingerprint(text, run / "input.namelist")
    requested = text.replace('"field.nc"', repr(str(source / "field.nc"))) + "! comment\n"
    match = _driver_stub().output_matches_input
    assert match(output, requested, source / "input.namelist")
    (source / "field.nc").write_bytes(b"changed geometry")
    assert not match(output, requested, source / "input.namelist")



def test_retry_refuses_conflicting_localized_geometry_without_overwriting(tmp_path):
    import importlib.util

    source = tmp_path / "original"
    source.mkdir()
    (source / "field.nc").write_bytes(b"new geometry")
    point = tmp_path / "point"
    point.mkdir()
    local = point / "field.nc"
    local.write_bytes(b"old geometry")
    deck = point / "input.namelist"
    deck.write_text(f'&geometryParameters\n geometryScheme = 5\n equilibriumFile = "{source / "field.nc"}"\n/\n')
    spec = importlib.util.spec_from_file_location("scan_driver_geometry_test", UTILS / "dkx_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with pytest.raises(RuntimeError, match="Conflicting localized equilibrium"):
        driver.run_dkx(input_namelist=deck)
    assert local.read_bytes() == b"old geometry"
    assert not (point / "sfincsOutput.h5").exists()



def test_colliding_er_labels_are_rejected_before_any_point_writes(legacy_dispatcher, tmp_path):
    root = tmp_path / "scan"
    root.mkdir()
    deck = root / "input.namelist"
    deck.write_text("!ss scanType = 2\n!ss NErs = 2\n!ss ErMin = 1.00001\n!ss ErMax = 1.00002\n"
                    "&geometryParameters\n inputRadialCoordinateForGradients = 4\n/\n"
                    "&physicsParameters\n Er = 0\n/\n")
    point = root / "Er1"
    point.mkdir()
    _write_complete(point / "sfincsOutput.h5")
    (point / "input.namelist").write_text("previous accepted input")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = legacy_dispatcher(root)
    assert result.returncode == 1
    assert "both map to directory 'Er1'" in result.stderr
    assert "reduce NErs" in result.stderr
    assert "No scan points were written" in result.stderr
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir()) == ["Er1"]


@pytest.mark.parametrize("change", ["missing_fingerprint", "same_path_equilibrium"])
def test_unbound_or_replaced_equilibrium_output_is_archived_before_retry(legacy_dispatcher, tmp_path, change):
    root = tmp_path / "scan"
    root.mkdir()
    equilibrium = root / "field.nc"
    equilibrium.write_bytes(b"old geometry")
    deck = root / "input.namelist"
    deck.write_text('!ss scanType = 2\n!ss NErs = 1\n!ss ErMin = 1\n!ss ErMax = 1\n'
                    '&geometryParameters\n inputRadialCoordinateForGradients = 4\n'
                    f' equilibriumFile = "{equilibrium}"\n/\n'
                    '&speciesParameters\n/\n&physicsParameters\n Er = 0\n/\n')
    first = legacy_dispatcher(root)
    assert first.returncode == 0, first.stdout + first.stderr
    output = root / "Er1/sfincsOutput.h5"
    with h5py.File(output, "r+") as f:
        original_fingerprint = f.attrs["dkx_scan_input_fingerprint"]
        if change == "missing_fingerprint":
            del f.attrs["dkx_scan_input_fingerprint"]
    accepted = output.read_bytes()
    if change == "same_path_equilibrium":
        equilibrium.write_bytes(b"new geometry")  # Same path and length, different bytes.
    retry = legacy_dispatcher(root)
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert (root / "Er1/.dkx-failed-attempts/1/sfincsOutput.h5").read_bytes() == accepted
    assert (root / "Er1/calls.txt").read_text().count("called") == 2
    with h5py.File(output, "r") as f:
        assert (f.attrs["dkx_scan_input_fingerprint"] != original_fingerprint) is (change == "same_path_equilibrium")
    assert legacy_dispatcher(root).returncode == 0
    assert (root / "Er1/calls.txt").read_text().count("called") == 2


def test_equilibrium_digest_cache_invalidates_same_path_replacement(tmp_path, monkeypatch):
    import hashlib

    equilibrium = tmp_path / "field.nc"
    equilibrium.write_bytes(b"old geometry")
    deck = '&geometryParameters\n equilibriumFile = "field.nc"\n/\n'
    fingerprint = _driver_stub().scan_input_fingerprint
    real_digest = hashlib.file_digest
    calls = []

    def counted(*args):
        calls.append(1)
        return real_digest(*args)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    before = fingerprint(deck, tmp_path / "input.namelist")
    assert fingerprint(deck + "! comment\n", tmp_path / "input.namelist") == before
    assert len(calls) == 1
    equilibrium.write_bytes(b"new geometry")
    assert fingerprint(deck, tmp_path / "input.namelist") != before
    assert len(calls) == 2


def test_driver_does_not_stamp_equilibrium_changed_during_execution(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("scan_driver_changed_geometry", UTILS / "dkx_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    equilibrium = tmp_path / "field.nc"
    equilibrium.write_bytes(b"old geometry")
    deck = tmp_path / "input.namelist"
    deck.write_text('&geometryParameters\n equilibriumFile = "field.nc"\n/\n')
    output = tmp_path / "sfincsOutput.h5"

    def fake_solve(*args, **kwargs):
        _write_complete(output)
        with h5py.File(output, "a") as f:
            f["input.namelist"] = deck.read_text()
        equilibrium.write_bytes(b"new geometry")
        return output

    monkeypatch.setattr(driver, "write_output", fake_solve)
    with pytest.raises(RuntimeError, match="changed during execution"):
        driver.run_dkx(input_namelist=deck, ensure_equilibrium=False)
    with h5py.File(output, "r") as f:
        assert "dkx_scan_input_fingerprint" not in f.attrs
