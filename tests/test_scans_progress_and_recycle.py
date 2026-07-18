from __future__ import annotations

import json
from pathlib import Path

import pytest

import dkx.workflows.scans as scans
from dkx.namelist import Namelist


def _minimal_scan_input(path: Path, *, gradient_coord: int = 4) -> Path:
    path.write_text(
        "&geometryParameters\n"
        f"  inputRadialCoordinateForGradients = {gradient_coord}\n"
        "/\n"
        "&physicsParameters\n"
        "  Er = 0.0\n"
        "/\n"
    )
    return path


def test_scan_format_duration_covers_long_running_user_estimates() -> None:
    assert scans._format_duration(-3.0) == "0.0s"
    assert scans._format_duration(4.25) == "4.2s"
    assert scans._format_duration(125.0) == "2m05s"
    assert scans._format_duration(3.0 * 3600.0 + 59.0) == "3h00m"
    assert scans._format_duration(2.0 * 86400.0 + 5.0 * 3600.0) == "2d05h"


def test_scan_progress_messages_report_eta_and_reused_outputs() -> None:
    messages: list[tuple[int, str]] = []

    scans._emit_scan_progress(
        emit=lambda level, msg: messages.append((level, msg)),
        completed=1,
        total=3,
        run_name="Er1",
        point_elapsed_s=2.0,
        total_elapsed_s=5.0,
        solved_elapsed_s=[2.0, 4.0],
        skipped_existing=False,
    )
    assert messages[-1] == (
        0,
        "scan-er: progress 1/3 Er1 point_elapsed=2.0s avg_point=3.0s "
        "elapsed=5.0s est_remaining=6.0s",
    )

    scans._emit_scan_progress(
        emit=lambda level, msg: messages.append((level, msg)),
        completed=2,
        total=3,
        run_name="Er0",
        point_elapsed_s=0.0,
        total_elapsed_s=7.0,
        solved_elapsed_s=[2.0, 4.0],
        skipped_existing=True,
    )
    assert messages[-1] == (
        0,
        "scan-er: progress 2/3 Er0 reused existing output total_elapsed=7.0s remaining_points=1",
    )

    scans._emit_scan_progress(
        emit=None,
        completed=1,
        total=1,
        run_name="noop",
        point_elapsed_s=0.0,
        total_elapsed_s=0.0,
        solved_elapsed_s=[],
        skipped_existing=False,
    )


@pytest.mark.parametrize(
    ("coord", "expected"),
    [
        (None, "Er"),
        (0, "dPhiHatdpsiHat"),
        (1, "dPhiHatdpsiN"),
        (2, "dPhiHatdrHat"),
        (3, "dPhiHatdrN"),
        (4, "Er"),
    ],
)
def test_er_scan_var_name_matches_gradient_coordinate_options(coord: int | None, expected: str) -> None:
    geom = {} if coord is None else {"INPUTRADIALCOORDINATEFORGRADIENTS": coord}
    nml = Namelist(groups={"geometryparameters": geom}, indexed={}, source_path=None, source_text=None)
    assert scans._er_scan_var_name(nml=nml) == expected


def test_er_scan_var_name_rejects_invalid_coordinate() -> None:
    nml = Namelist(
        groups={"geometryparameters": {"INPUTRADIALCOORDINATEFORGRADIENTS": 99}},
        indexed={},
        source_path=None,
        source_text=None,
    )
    with pytest.raises(ValueError, match="Invalid inputRadialCoordinateForGradients"):
        scans._er_scan_var_name(nml=nml)


def test_patch_scalar_in_group_appends_and_reports_malformed_namelists() -> None:
    patched = scans._patch_scalar_in_group(
        txt="&physicsParameters\n  Er = 1.0\n/\n",
        group="physicsParameters",
        key="dPhiHatdpsiN",
        value=-2.5,
    )
    assert "dPhiHatdpsiN = -2.5" in patched

    with pytest.raises(ValueError, match="Missing namelist group"):
        scans._patch_scalar_in_group(txt="&other\n/\n", group="physicsParameters", key="Er", value=0.0)
    with pytest.raises(ValueError, match="Missing '/' terminator"):
        scans._patch_scalar_in_group(txt="&physicsParameters\n", group="physicsParameters", key="Er", value=0.0)


def test_linspace_including_endpoints_matches_scan_grid_contract() -> None:
    values = scans.linspace_including_endpoints(-2.0, 1.0, 4)

    assert values.tolist() == pytest.approx([-2.0, -1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="n must be >= 2"):
        scans.linspace_including_endpoints(0.0, 1.0, 1)


def test_find_upstream_utils_dir_honors_override_env_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "override_utils"
    override.mkdir()
    env_utils = tmp_path / "env_utils"
    env_utils.mkdir()

    assert scans.find_upstream_utils_dir(override=override) == override
    with pytest.raises(FileNotFoundError, match="utils dir does not exist"):
        scans.find_upstream_utils_dir(override=tmp_path / "missing")

    monkeypatch.setenv("DKX_UPSTREAM_UTILS_DIR", str(env_utils))
    assert scans.find_upstream_utils_dir() == env_utils

    monkeypatch.setenv("DKX_UPSTREAM_UTILS_DIR", str(tmp_path / "missing_env"))
    with pytest.raises(FileNotFoundError, match="DKX_UPSTREAM_UTILS_DIR does not exist"):
        scans.find_upstream_utils_dir()

    monkeypatch.delenv("DKX_UPSTREAM_UTILS_DIR", raising=False)
    default_utils = scans.find_upstream_utils_dir()
    assert default_utils.name == "utils"
    assert (default_utils / "sfincsScanPlot_1").is_file()


def test_run_upstream_util_executes_noninteractive_helper(tmp_path: Path) -> None:
    utils_dir = tmp_path / "utils"
    utils_dir.mkdir()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    script = utils_dir / "plot_scan.py"
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "answer = input('prompt:')\n"
        "Path('result.json').write_text(json.dumps({\n"
        "    'answer': answer,\n"
        "    'argv': sys.argv[1:],\n"
        "    'backend': os.environ.get('MPLBACKEND'),\n"
        "}))\n",
        encoding="utf-8",
    )
    messages: list[str] = []

    scans.run_upstream_util(
        util="plot_scan.py",
        case_dir=case_dir,
        args=("--quantity", "Gamma"),
        utils_dir=utils_dir,
        emit=lambda _level, msg: messages.append(msg),
    )

    payload = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    assert payload == {
        "answer": "",
        "argv": ["--quantity", "Gamma"],
        "backend": "Agg",
    }
    assert messages == [f"postprocess-upstream: running {script.name} in {case_dir.resolve()}"]


def test_run_upstream_util_reports_missing_script_and_case_dir(tmp_path: Path) -> None:
    utils_dir = tmp_path / "utils"
    utils_dir.mkdir()
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Upstream util not found"):
        scans.run_upstream_util(util="missing.py", case_dir=case_dir, utils_dir=utils_dir)

    script = utils_dir / "noop.py"
    script.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="case_dir does not exist"):
        scans.run_upstream_util(util="noop.py", case_dir=tmp_path / "missing_case", utils_dir=utils_dir)


def test_run_er_scan_skip_existing_reuses_completed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = _minimal_scan_input(tmp_path / "input.namelist")
    existing_run_dir = tmp_path / "scan" / "Er0"
    existing_run_dir.mkdir(parents=True)
    existing_output = existing_run_dir / "sfincsOutput.h5"
    existing_output.write_bytes(b"done")

    def _unexpected_write(*_args, **_kwargs) -> None:
        raise AssertionError("skip_existing should not write a completed point")

    def _unexpected_localize(**_kwargs) -> None:
        raise AssertionError("skip_existing should not localize a completed point")

    monkeypatch.setattr(scans, "run_from_namelist", _unexpected_write)
    monkeypatch.setattr(scans, "localize_equilibrium_file_in_place", _unexpected_localize)
    messages: list[str] = []

    result = scans.run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.0],
        skip_existing=True,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.run_dirs == (existing_run_dir.resolve(),)
    assert result.outputs == (existing_output.resolve(),)
    assert result.values == (0.0,)
    assert any("reused existing output" in message for message in messages)


def test_run_er_scan_parallel_path_uses_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _minimal_scan_input(tmp_path / "input.namelist")
    writes: list[str] = []

    class _FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _FakePool:
        def __init__(self, max_workers: int):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def submit(self, fn, payload):
            return _FakeFuture(fn(payload))

    def _fake_run(_namelist_path, *, out_path, **_kwargs) -> None:
        out_path = Path(out_path)
        out_path.write_bytes(b"")
        writes.append(out_path.parent.name)

    monkeypatch.setattr(scans, "localize_equilibrium_file_in_place", lambda **_kwargs: None)
    monkeypatch.setattr(scans, "run_from_namelist", _fake_run)
    monkeypatch.setattr(scans.concurrent.futures, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(scans.concurrent.futures, "as_completed", lambda futures: list(futures))
    messages: list[tuple[int, str]] = []

    result = scans.run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.0, 1.0],
        jobs=2,
        emit=lambda level, msg: messages.append((level, msg)),
    )

    assert result.values == (1.0, 0.0)
    assert [path.name for path in result.run_dirs] == ["Er0", "Er1"]
    assert sorted(writes) == ["Er0", "Er1"]
    assert any("jobs=2 (parallel)" in message for _level, message in messages)
    assert sum("scan-er: progress" in message for _level, message in messages) == 2


def test_run_er_scan_subset_serial_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = _minimal_scan_input(tmp_path / "input.namelist")
    writes: list[tuple[str, Path]] = []

    def _fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        assert input_namelist.exists()
        assert overwrite is False

    def _fake_run(_namelist_path, *, out_path, solver_trace_path, **_kwargs) -> None:
        out_path = Path(out_path)
        out_path.write_bytes(b"")
        writes.append((out_path.parent.name, Path(solver_trace_path)))

    monkeypatch.setattr(scans, "localize_equilibrium_file_in_place", _fake_localize)
    monkeypatch.setattr(scans, "run_from_namelist", _fake_run)

    result = scans.run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.0, 1.0, 2.0, 3.0, 4.0],
        compute_solution=True,
        index=1,
        stride=2,
        emit=lambda _level, _msg: None,
    )

    assert result.values == (3.0, 1.0)
    assert [path.name for path in result.run_dirs] == ["Er3", "Er1"]
    assert writes[0][0] == "Er3"
    assert writes[0][1].name == "sfincsOutput.solver_trace.json"
    assert writes[1][0] == "Er1"
    assert writes[1][1].parent.name == "Er1"


def test_run_er_scan_rejects_invalid_subset_index(tmp_path: Path) -> None:
    template = _minimal_scan_input(tmp_path / "input.namelist")
    with pytest.raises(ValueError, match="index=2 out of range"):
        scans.run_er_scan(
            input_namelist=template,
            out_dir=tmp_path / "scan",
            values=[0.0, 1.0],
            index=2,
            stride=2,
        )
