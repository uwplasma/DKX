from __future__ import annotations

import os
from pathlib import Path

import pytest

import sfincs_jax.scans as scans
from sfincs_jax.namelist import Namelist


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


def test_run_er_scan_subset_and_serial_recycle_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = _minimal_scan_input(tmp_path / "input.namelist")
    writes: list[tuple[str, str | None, str | None]] = []

    def _fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        assert input_namelist.exists()
        assert overwrite is False

    def _fake_write(**kwargs) -> None:
        out_path = Path(kwargs["output_path"])
        out_path.write_bytes(b"")
        state_out = os.environ.get("SFINCS_JAX_STATE_OUT")
        if state_out:
            Path(state_out).write_bytes(b"state")
        writes.append((out_path.parent.name, os.environ.get("SFINCS_JAX_STATE_IN"), state_out))

    monkeypatch.setattr(scans, "localize_equilibrium_file_in_place", _fake_localize)
    monkeypatch.setattr(scans, "write_sfincs_jax_output_h5", _fake_write)
    monkeypatch.setenv("SFINCS_JAX_SCAN_RECYCLE", "1")

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
    assert writes[0][1] is None
    assert writes[0][2] is not None and writes[0][2].endswith("Er3/sfincs_jax_state.npz")
    assert writes[1][0] == "Er1"
    assert writes[1][1] is not None and writes[1][1].endswith("Er3/sfincs_jax_state.npz")
    assert writes[1][2] is not None and writes[1][2].endswith("Er1/sfincs_jax_state.npz")


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
