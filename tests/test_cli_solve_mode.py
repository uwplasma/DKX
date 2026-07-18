from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dkx import cli
from dkx.input_compat import effective_equilibrium_file
from dkx.namelist import parse_sfincs_input_text
from dkx.api import write_output
from dkx.io import read_sfincs_h5


class _FakeNamelist:
    def __init__(self, rhs_mode: int = 1) -> None:
        self.groups = {
            "general": {"RHSMODE": rhs_mode},
            "geometryparameters": {},
            "physicsparameters": {},
            "resolutionparameters": {},
        }
        self.indexed = {}
        self.source_path = None
        self.source_text = None

    def group(self, name: str):
        return self.groups.get(name.lower(), {})


def test_cmd_write_output_routes_solver_trace_through_canonical(monkeypatch, tmp_path: Path) -> None:
    """A supported RHSMode=1 --solver-trace deck now emits the sidecar from run_profile."""
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out=str(tmp_path / "sfincsOutput.h5"),
        equilibrium_file=None,
        wout_path=None,
        fortran_layout=True,
        overwrite=True,
        compute_transport_matrix=False,
        compute_solution=False,
        geometry_only=False,
        solver_trace=str(tmp_path / "solver_trace.json"),
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_write_output(args) == 0
    assert captured["solver_trace_path"] == Path(tmp_path / "solver_trace.json")
    assert captured["out_path"] == Path(tmp_path / "sfincsOutput.h5")


def test_cmd_write_output_accepts_extension_selected_formats(monkeypatch, tmp_path: Path) -> None:
    """--geometry-only routes through the canonical run_geometry; --out suffix selects the format."""
    captured: dict[str, object] = {}

    def _fake_run_geometry(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_geometry", _fake_run_geometry)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out=str(tmp_path / "sfincsOutput.nc"),
        equilibrium_file=None,
        wout_path=None,
        fortran_layout=True,
        overwrite=True,
        compute_transport_matrix=False,
        compute_solution=False,
        geometry_only=True,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_write_output(args) == 0
    assert Path(captured["out_path"]).suffix == ".nc"
    assert captured["overwrite"] is True
    assert captured["fortran_layout"] is True


def test_default_plot_output_path_uses_pdf() -> None:
    assert cli._default_plot_output_path(Path("sfincsOutput.h5")).name == "sfincsOutput_summary.pdf"


def test_cmd_solve_v3_routes_canonical_run_profile(monkeypatch, tmp_path: Path) -> None:
    """A supported RHSMode=1 deck routes through run_profile, not the legacy problems/ owner."""
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        return SimpleNamespace(
            state_vector=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            solve_result=SimpleNamespace(residual_norms=np.asarray([1.5e-11], dtype=np.float64)),
            output_path=None,
        )

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    out_state = tmp_path / "state.npy"
    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_state=str(out_state),
        equilibrium_file=None,
        wout_path=None,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        which_rhs=None,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_solve_v3(args) == 0
    assert captured["solve_method"] == "incremental"
    assert captured["tol"] == 1e-8
    assert captured["emit"] is None  # --quiet silences the Fortran-parity console
    np.testing.assert_array_equal(np.load(out_state), np.asarray([1.0, 2.0, 3.0]))


def test_cmd_solve_v3_transport_mode_selects_which_rhs_column(monkeypatch, tmp_path: Path) -> None:
    """RHSMode=2/3 decks route through run_transport_matrix; --which-rhs selects the column."""
    captured: dict[str, object] = {}

    def _fake_run_transport_matrix(namelist_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            state_vectors=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64),
            solve_result=SimpleNamespace(
                residual_norms=np.asarray([1e-11, 2e-11, 3e-11], dtype=np.float64)
            ),
            output_path=None,
        )

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("dkx.run.run_transport_matrix", _fake_run_transport_matrix)

    out_state = tmp_path / "state.npy"
    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_state=str(out_state),
        equilibrium_file=None,
        wout_path=None,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="auto",
        which_rhs=2,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_solve_v3(args) == 0
    # whichRHS=2 -> zero-based column index 1.
    np.testing.assert_array_equal(np.load(out_state), np.asarray([2.0, 2.0]))


def test_cmd_solve_v3_rhsmode4_is_a_validation_error(tmp_path: Path, capsys) -> None:
    """RHSMode=4 (the retired Fortran adjoint mode) is a clean namelist validation error."""
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n  RHSMode = 4\n/\n", encoding="utf-8")

    args = Namespace(
        input=str(input_path),
        out_state=str(tmp_path / "state.npy"),
        equilibrium_file=None,
        wout_path=None,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        which_rhs=None,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_solve_v3(args) == 2
    err = capsys.readouterr().err
    assert "dkx solve-v3 failed: RHSMode must be 1, 2, or 3 (got 4)" in err
    assert "jax.grad" in err
    assert not (tmp_path / "state.npy").exists()


def test_cmd_transport_matrix_v3_uses_canonical_run(monkeypatch, tmp_path: Path) -> None:
    """transport-matrix-v3 routes through the canonical dkx.run driver."""
    captured: dict[str, object] = {}

    def _fake_run_transport_matrix(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        return SimpleNamespace(
            transport_matrix=np.zeros((2, 2), dtype=np.float64),
            state_vectors=np.zeros((2, 4), dtype=np.float64),
            solve_result=SimpleNamespace(residual_norms=np.zeros((2,), dtype=np.float64)),
            output_path=None,
        )

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("dkx.run.run_transport_matrix", _fake_run_transport_matrix)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_matrix=str(tmp_path / "tm.npy"),
        out=None,
        out_state_prefix=str(tmp_path / "state"),
        equilibrium_file=None,
        wout_path=None,
        tol=1e-8,
        solve_method="auto",
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_transport_matrix_v3(args) == 0
    assert captured["namelist_path"] == Path(args.input)
    assert captured["tol"] == 1e-8
    assert captured["solve_method"] == "auto"
    assert captured["out_path"] is None
    assert captured["emit"] is None  # --quiet silences the Fortran-parity stdout
    assert (tmp_path / "tm.npy").exists()
    assert (tmp_path / "state.whichRHS1.npy").exists()
    assert (tmp_path / "state.whichRHS2.npy").exists()


def test_write_output_full_system_regression(tmp_path: Path, monkeypatch) -> None:
    """Full-system write-output should not reference transport-only distributed state."""
    input_path = (
        Path(__file__).parent / "reduced_inputs" / "inductiveE_noEr.input.namelist"
    )
    assert input_path.exists()

    monkeypatch.setenv("DKX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("DKX_SOLVER_ITER_STATS", "0")

    out_path = tmp_path / "sfincsOutput.h5"
    write_output(input_path, out_path)

    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"]).item()) == 1
    assert "classicalParticleFluxNoPhi1_psiHat" in data


def test_apply_runtime_env_defaults_disables_preallocation_by_default(monkeypatch) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    cli._apply_runtime_env_defaults()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_apply_runtime_env_defaults_respects_existing_preallocation(monkeypatch) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    cli._apply_runtime_env_defaults()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_normalize_default_argv_keeps_wout_path_override() -> None:
    argv = ["input.namelist", "--wout-path", "override.nc", "--out", "sfincsOutput.h5"]
    assert cli._normalize_default_argv(argv) == [
        "write-output",
        "--input",
        "input.namelist",
        "--wout-path",
        "override.nc",
        "--out",
        "sfincsOutput.h5",
    ]


def test_normalize_default_argv_keeps_parallel_flags() -> None:
    argv = [
        "--cores",
        "8",
        "--transport-workers",
        "3",
        "input.namelist",
        "--out",
        "sfincsOutput.h5",
    ]
    assert cli._normalize_default_argv(argv) == [
        "--cores",
        "8",
        "--transport-workers",
        "3",
        "write-output",
        "--input",
        "input.namelist",
        "--out",
        "sfincsOutput.h5",
    ]


def test_normalize_default_argv_maps_plot_shortcut() -> None:
    argv = ["--plot", "sfincsOutput.h5", "--out", "summary.png"]
    assert cli._normalize_default_argv(argv) == [
        "plot-output",
        "--input-h5",
        "sfincsOutput.h5",
        "--out",
        "summary.png",
    ]


def test_default_plot_output_path_handles_sfincsoutput_suffix() -> None:
    path = cli._default_plot_output_path(Path("sfincsOutput.h5"))
    assert path.name == "sfincsOutput_summary.pdf"


def test_apply_parallel_runtime_settings_sets_transport_workers(monkeypatch) -> None:
    monkeypatch.delenv("DKX_TRANSPORT_PARALLEL", raising=False)
    monkeypatch.delenv("DKX_TRANSPORT_PARALLEL_WORKERS", raising=False)

    cli._apply_parallel_runtime_settings(
        Namespace(
            transport_workers=4,
            distributed=False,
            process_id=None,
            process_count=None,
            coordinator_address=None,
            coordinator_port=None,
        )
    )

    assert os.environ["DKX_TRANSPORT_PARALLEL"] == "process"
    assert os.environ["DKX_TRANSPORT_PARALLEL_WORKERS"] == "4"


def test_apply_parallel_runtime_settings_initializes_distributed(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.delenv("DKX_DISTRIBUTED", raising=False)
    monkeypatch.delenv("DKX_PROCESS_ID", raising=False)
    monkeypatch.delenv("DKX_PROCESS_COUNT", raising=False)
    monkeypatch.delenv("DKX_COORDINATOR_ADDRESS", raising=False)
    monkeypatch.delenv("DKX_COORDINATOR_PORT", raising=False)
    monkeypatch.setattr(
        "dkx.initialize_distributed_runtime_from_env",
        lambda: calls.append("init") or True,
    )

    cli._apply_parallel_runtime_settings(
        Namespace(
            transport_workers=None,
            distributed=True,
            process_id=1,
            process_count=2,
            coordinator_address="worker0",
            coordinator_port=2345,
        )
    )

    assert os.environ["DKX_DISTRIBUTED"] == "1"
    assert os.environ["DKX_PROCESS_ID"] == "1"
    assert os.environ["DKX_PROCESS_COUNT"] == "2"
    assert os.environ["DKX_COORDINATOR_ADDRESS"] == "worker0"
    assert os.environ["DKX_COORDINATOR_PORT"] == "2345"
    assert calls == ["init"]


def test_emit_parallel_runtime_info_reports_active_parallel_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("DKX_CORES", "8")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("DKX_CPU_DEVICES", "8")
    monkeypatch.setenv("DKX_TRANSPORT_PARALLEL", "process")
    monkeypatch.setenv("DKX_TRANSPORT_PARALLEL_WORKERS", "4")
    monkeypatch.setenv("DKX_DISTRIBUTED", "1")
    monkeypatch.setenv("DKX_PROCESS_ID", "1")
    monkeypatch.setenv("DKX_PROCESS_COUNT", "2")
    monkeypatch.setenv("DKX_COORDINATOR_ADDRESS", "node0")
    monkeypatch.setenv("DKX_COORDINATOR_PORT", "1234")

    cli._emit_parallel_runtime_info(args=Namespace(verbose=1, quiet=False))
    out = capsys.readouterr().out

    assert "parallel: cores=8 threads=8 cpu_devices=8" in out
    assert "transport_parallel: mode=process workers=4" in out
    assert "multi_host: enabled=1 process_id=1 process_count=2 coordinator=node0 port=1234" in out


def test_emit_parallel_runtime_info_suppresses_empty_state(monkeypatch, capsys) -> None:
    for name in (
        "DKX_CORES",
        "DKX_CPU_DEVICES",
        "DKX_TRANSPORT_PARALLEL",
        "DKX_TRANSPORT_PARALLEL_WORKERS",
        "DKX_DISTRIBUTED",
        "DKX_PROCESS_ID",
        "DKX_PROCESS_COUNT",
        "DKX_COORDINATOR_ADDRESS",
        "DKX_COORDINATOR_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    # NPROC alone (e.g. the import-time default clamp) must not trigger the
    # parallel banner.
    monkeypatch.setenv("NPROC", "8")
    cli._emit_parallel_runtime_info(args=Namespace(verbose=1, quiet=False))
    assert capsys.readouterr().out == ""


def test_maybe_reexec_for_early_runtime_reexecs_for_cores(monkeypatch) -> None:
    monkeypatch.delenv("DKX_CORES", raising=False)
    monkeypatch.delenv("DKX_CPU_DEVICES", raising=False)
    captured: dict[str, object] = {}

    def _fake_execvpe(executable, argv, env):
        captured["executable"] = executable
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execvpe", _fake_execvpe)

    try:
        cli._maybe_reexec_for_early_runtime(["--cores", "4", "write-output", "--input", "input.namelist"])
    except SystemExit:
        pass

    assert captured["argv"] == [cli.sys.executable, "-m", "dkx", "--cores", "4", "write-output", "--input", "input.namelist"]
    env = captured["env"]
    assert env["DKX_CORES"] == "4"
    # --cores controls threads only; it must not force host devices.
    assert "DKX_CPU_DEVICES" not in env
    assert env["DKX_CLI_BOOTSTRAPPED"] == "1"


def test_maybe_reexec_for_early_runtime_skips_when_env_matches(monkeypatch) -> None:
    monkeypatch.setenv("DKX_CORES", "4")
    called = []
    monkeypatch.setattr(cli.os, "execvpe", lambda *args, **kwargs: called.append((args, kwargs)))
    cli._maybe_reexec_for_early_runtime(["--cores", "4", "write-output", "--input", "input.namelist"])
    assert called == []


def test_maybe_reexec_for_early_runtime_reexecs_for_distributed(monkeypatch) -> None:
    for name in (
        "DKX_DISTRIBUTED",
        "DKX_PROCESS_ID",
        "DKX_PROCESS_COUNT",
        "DKX_COORDINATOR_ADDRESS",
        "DKX_COORDINATOR_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    captured: dict[str, object] = {}

    def _fake_execvpe(executable, argv, env):
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execvpe", _fake_execvpe)

    try:
        cli._maybe_reexec_for_early_runtime(
            [
                "--distributed",
                "--process-id",
                "1",
                "--process-count",
                "2",
                "--coordinator-address",
                "node0",
                "--coordinator-port",
                "1234",
                "write-output",
                "--input",
                "input.namelist",
            ]
        )
    except SystemExit:
        pass

    env = captured["env"]
    assert env["DKX_DISTRIBUTED"] == "1"
    assert env["DKX_PROCESS_ID"] == "1"
    assert env["DKX_PROCESS_COUNT"] == "2"
    assert env["DKX_COORDINATOR_ADDRESS"] == "node0"
    assert env["DKX_COORDINATOR_PORT"] == "1234"


def test_cmd_solve_v3_applies_equilibrium_override(monkeypatch, tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        # The canonical driver reads the namelist file itself, so the CLI
        # equilibrium override is materialized into the namelist it is handed.
        captured["namelist_text"] = Path(namelist_path).read_text(encoding="utf-8")
        captured.update(kwargs)
        return SimpleNamespace(
            state_vector=np.zeros((2,), dtype=np.float64),
            solve_result=SimpleNamespace(residual_norms=np.zeros((1,), dtype=np.float64)),
            output_path=None,
        )

    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    args = Namespace(
        input=str(input_path),
        out_state=str(tmp_path / "state.npy"),
        equilibrium_file=None,
        wout_path="override_wout.nc",
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        which_rhs=None,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_solve_v3(args) == 0
    materialized = parse_sfincs_input_text(captured["namelist_text"])
    assert (
        effective_equilibrium_file(geom_params=materialized.group("geometryParameters"))
        == "override_wout.nc"
    )


def test_main_accepts_quiet_after_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["out_path"] == Path(tmp_path / "sfincsOutput.h5")
    assert captured["solve_method"] == "auto"


def test_main_bare_input_uses_public_auto_contract(monkeypatch, tmp_path: Path) -> None:
    """Bare-input runs route through the canonical run_profile with the auto contract."""
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_text"] = Path(namelist_path).read_text(encoding="utf-8")
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    def _fake_namelist_with_source(_path):
        nml = _FakeNamelist(rhs_mode=1)
        nml.source_text = "&general\n/\n&geometryParameters\n/\n"
        return nml

    monkeypatch.setattr("dkx.cli.read_sfincs_input", _fake_namelist_with_source)
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    rc = cli.main(
        [
            str(tmp_path / "input.namelist"),
            "--wout-path",
            str(tmp_path / "wout.nc"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "auto"
    assert captured["overwrite"] is True
    assert captured["fortran_layout"] is True
    assert captured["out_path"] == Path(tmp_path / "sfincsOutput.h5")
    # The --wout-path override is materialized into the namelist run_profile reads.
    assert str(tmp_path / "wout.nc") in str(captured["namelist_text"])


def test_write_output_help_presents_solver_override_as_advanced(capsys) -> None:
    try:
        cli.main(["write-output", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "Advanced RHSMode=1 solver override" in out
    assert "Default" in out
    assert "'auto'" in out
    assert "recommended" in out


def test_main_write_output_forwards_solver_trace_sidecar(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    trace_path = tmp_path / "solver_trace.json"
    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solver-trace",
            str(trace_path),
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solver_trace_path"] == trace_path


@pytest.mark.parametrize(
    "method",
    ["sparse_host", "sparse_pc_gmres", "xblock_sparse_pc_gmres", "sparse_lsmr", "petsc_compat"],
)
def test_main_write_output_removed_solve_methods_error_cleanly(
    monkeypatch, tmp_path: Path, capsys, method: str
) -> None:
    """Solve methods removed from the canonical stack error cleanly (no legacy fallback remains)."""

    def _canonical_refuses(*_args, **_kwargs):
        raise NotImplementedError(f"solve_method={method} was removed from the canonical stack")

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _canonical_refuses)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            method,
            "--quiet",
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert f"dkx write-output failed: solve_method={method} was removed" in err
    assert "Traceback" not in err


def test_main_write_output_forwards_sparse_host_safe_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    # sparse_host_safe survives on the canonical RHSMode=1 stack, so the CLI
    # forwards the override to run_profile rather than the legacy writer.
    def _fake_run_profile(namelist_path, **kwargs):
        captured["namelist_path"] = Path(namelist_path)
        captured.update(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "sparse_host_safe",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "sparse_host_safe"


def test_main_scan_er_forwards_structured_csr_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_er_scan(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("dkx.workflows.scans.run_er_scan", _fake_run_er_scan)

    rc = cli.main(
        [
            "scan-er",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out-dir",
            str(tmp_path / "scan"),
            "--values",
            "0.0",
            "--compute-solution",
            "--solve-method",
            "host_structured_csr",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["compute_solution"] is True
    assert captured["solve_method"] == "host_structured_csr"
    assert captured["differentiable"] is False


def test_main_write_output_reports_runtime_errors_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    def _fake_run_profile(*_args, **_kwargs):
        raise RuntimeError("host sparse factorization failed")

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_profile", _fake_run_profile)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "dkx write-output failed: host sparse factorization failed" in captured.err
    assert "Traceback" not in captured.err


def test_main_preserves_cores_before_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_geometry(namelist_path, **kwargs):
        captured["cores"] = os.environ.get("DKX_CORES")
        captured["threads"] = os.environ.get("NPROC")
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("dkx.run.run_geometry", _fake_run_geometry)
    monkeypatch.setenv("DKX_CORES", "4")

    rc = cli.main(
        [
            "--cores",
            "4",
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--geometry-only",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["cores"] == "4"
    assert captured["threads"] == "4"


def test_main_preserves_transport_workers_before_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_transport_matrix(namelist_path, **kwargs):
        captured["parallel_mode"] = os.environ.get("DKX_TRANSPORT_PARALLEL")
        captured["workers"] = os.environ.get("DKX_TRANSPORT_PARALLEL_WORKERS")
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return SimpleNamespace(output_path=out)

    monkeypatch.setattr("dkx.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("dkx.run.run_transport_matrix", _fake_run_transport_matrix)

    rc = cli.main(
        [
            "--transport-workers",
            "3",
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["parallel_mode"] == "process"
    assert captured["workers"] == "3"


def test_cmd_run_fortran_reports_output_path(monkeypatch, tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "run" / "sfincsOutput.h5"
    captured: dict[str, object] = {}

    def _fake_run_fortran(**kwargs):
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr("dkx.validation.fortran.run_sfincs_fortran", _fake_run_fortran)

    rc = cli._cmd_run_fortran(
        Namespace(
            input=str(tmp_path / "input.namelist"),
            exe=str(tmp_path / "sfincs"),
            workdir=str(tmp_path / "work"),
            quiet=False,
            verbose=1,
        )
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert captured["input_namelist"] == Path(tmp_path / "input.namelist")
    assert captured["exe"] == Path(tmp_path / "sfincs")
    assert captured["workdir"] == Path(tmp_path / "work")
    assert f"wrote sfincsOutput.h5 -> {output_path}" in out


def test_cmd_dump_h5_keys_and_json_payload(monkeypatch, tmp_path: Path, capsys) -> None:
    data = {"zeta": np.asarray([2.0]), "alpha": np.asarray([[1, 2]])}
    monkeypatch.setattr("dkx.io.read_sfincs_h5", lambda _path: data)

    rc = cli._cmd_dump_h5(
        Namespace(
            sfincs_output=str(tmp_path / "sfincsOutput.h5"),
            out_json=str(tmp_path / "unused.json"),
            keys_only=True,
        )
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == ["alpha", "zeta"]

    out_json = tmp_path / "dump.json"
    rc = cli._cmd_dump_h5(
        Namespace(
            sfincs_output=str(tmp_path / "sfincsOutput.h5"),
            out_json=str(out_json),
            keys_only=False,
        )
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload == {"alpha": [[1, 2]], "zeta": [2.0]}


def test_cmd_compare_h5_prints_failures_and_show_all(monkeypatch, tmp_path: Path, capsys) -> None:
    results = [
        SimpleNamespace(ok=True, key="ok_key", max_abs=0.0, max_rel=0.0),
        SimpleNamespace(ok=False, key="bad_key", max_abs=1.0e-3, max_rel=2.0e-2),
    ]
    captured: dict[str, object] = {}

    def _fake_compare(**kwargs):
        captured.update(kwargs)
        return results

    monkeypatch.setattr("dkx.compare.compare_sfincs_outputs", _fake_compare)

    tolerances = tmp_path / "tolerances.json"
    tolerances.write_text(json.dumps({"bad_key": {"rtol": 0.1}}))
    rc = cli._cmd_compare_h5(
        Namespace(
            a=str(tmp_path / "a.h5"),
            b=str(tmp_path / "b.h5"),
            rtol="1e-4",
            atol="1e-6",
            tolerances_json=str(tolerances),
            show_all=False,
        )
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "FAIL bad_key" in out
    assert captured["tolerances"] == {"bad_key": {"rtol": 0.1}}

    rc = cli._cmd_compare_h5(
        Namespace(
            a=str(tmp_path / "a.h5"),
            b=str(tmp_path / "b.h5"),
            rtol="1e-4",
            atol="1e-6",
            tolerances_json=None,
            show_all=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "OK ok_key" in out
    assert "FAIL bad_key" in out


def test_apply_cores_setting_pins_threads_and_zero_defers_to_xla(monkeypatch) -> None:
    for name in ("DKX_CORES", "NPROC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    cli._apply_cores_setting(4)
    assert os.environ["DKX_CORES"] == "4"
    assert os.environ["NPROC"] == "4"
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"

    # cores=0 records the "let XLA size the threadpool" preference without
    # pinning anything itself.
    monkeypatch.delenv("NPROC", raising=False)
    cli._apply_cores_setting(0)
    assert os.environ["DKX_CORES"] == "0"
    assert "NPROC" not in os.environ

    # None / negative / junk change nothing.
    cli._apply_cores_setting(None)
    cli._apply_cores_setting(-2)
    cli._apply_cores_setting("junk")  # type: ignore[arg-type]
    assert os.environ["DKX_CORES"] == "0"
    assert "NPROC" not in os.environ


def test_normalize_default_argv_edge_cases() -> None:
    assert cli._normalize_default_argv([]) == []
    assert cli._normalize_default_argv(["--help"]) == ["--help"]
    assert cli._normalize_default_argv(["--cores=4", "input.namelist"]) == [
        "--cores=4",
        "write-output",
        "--input",
        "input.namelist",
    ]
    assert cli._normalize_default_argv(["plot-output", "--input-h5", "out.h5"]) == [
        "plot-output",
        "--input-h5",
        "out.h5",
    ]


def test_postprocess_upstream_cli_strips_separator_and_forwards_flags(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_upstream_util(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("dkx.workflows.scans.run_upstream_util", _fake_run_upstream_util)

    rc = cli.main(
        [
            "postprocess-upstream",
            "--case-dir",
            str(tmp_path / "case"),
            "--util",
            "sfincsScanPlot_1",
            "--utils-dir",
            str(tmp_path / "utils"),
            "--",
            "pdf",
        ]
    )

    assert rc == 0
    assert captured["util"] == "sfincsScanPlot_1"
    assert captured["case_dir"] == Path(tmp_path / "case")
    assert captured["args"] == ["pdf"]
    assert captured["utils_dir"] == Path(tmp_path / "utils")
    assert captured["noninteractive"] is True


# ---------------------------------------------------------------------------
# Canonical --no-overwrite / --no-fortran-layout / --geometry-only coverage
# (the CLI output options previously routed to the legacy outputs writer).
# ---------------------------------------------------------------------------

_TINY_DECK = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"


def _h5_datasets(path: Path) -> dict[str, object]:
    import h5py

    with h5py.File(path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def test_write_output_no_overwrite_errors_when_file_exists(tmp_path: Path, capsys) -> None:
    """--no-overwrite refuses to clobber an existing output (canonical FileExistsError guard)."""
    out = tmp_path / "sfincsOutput.h5"
    assert cli.main(
        ["write-output", "--input", str(_TINY_DECK), "--out", str(out), "--geometry-only", "--quiet"]
    ) == 0
    assert out.exists()
    capsys.readouterr()

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(_TINY_DECK),
            "--out",
            str(out),
            "--geometry-only",
            "--no-overwrite",
            "--quiet",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "dkx write-output failed" in err
    assert str(out.resolve()) in err
    assert "Traceback" not in err

    # The library-level guard is the same FileExistsError the legacy writer raised.
    from dkx.run import run_geometry

    with pytest.raises(FileExistsError):
        run_geometry(_TINY_DECK, out_path=out, overwrite=False, emit=None)


def test_write_output_no_fortran_layout_reverses_axes(tmp_path: Path) -> None:
    """--no-fortran-layout stores every dataset as the reversed-axes view of the Fortran layout."""
    fortran_out = tmp_path / "fortran.h5"
    native_out = tmp_path / "native.h5"
    assert cli.main(
        ["write-output", "--input", str(_TINY_DECK), "--out", str(fortran_out), "--geometry-only", "--quiet"]
    ) == 0
    assert cli.main(
        [
            "write-output",
            "--input",
            str(_TINY_DECK),
            "--out",
            str(native_out),
            "--geometry-only",
            "--no-fortran-layout",
            "--quiet",
        ]
    ) == 0

    fortran = _h5_datasets(fortran_out)
    native = _h5_datasets(native_out)
    assert set(fortran) == set(native)
    multi_dim = 0
    for key, f_val in fortran.items():
        n_val = native[key]
        if isinstance(f_val, bytes) or (hasattr(f_val, "dtype") and f_val.dtype.kind in "OSU"):
            assert n_val == f_val
            continue
        f_arr = np.asarray(f_val)
        n_arr = np.asarray(n_val)
        if f_arr.ndim >= 2:
            multi_dim += 1
            assert n_arr.shape == tuple(reversed(f_arr.shape))
            np.testing.assert_array_equal(n_arr, np.transpose(f_arr, tuple(reversed(range(f_arr.ndim)))))
        else:
            np.testing.assert_array_equal(n_arr, f_arr)
    assert multi_dim > 0  # e.g. BHat (Nzeta, Ntheta) in the Fortran view


def test_write_output_no_fortran_layout_covers_solution_datasets(tmp_path: Path) -> None:
    """fortran_layout=False also reverses the per-iteration solution datasets (one solve)."""
    from dkx.drift_kinetic import _geometry_and_radial
    from dkx.run import _grids_from_input, run_profile
    from dkx.writer import write_profile_output

    fortran_out = tmp_path / "fortran.h5"
    native_out = tmp_path / "native.h5"
    run = run_profile(_TINY_DECK, out_path=fortran_out, emit=None)
    grids = _grids_from_input(run.input, run.input.raw)
    geom, radial = _geometry_and_radial(nml=run.input.raw, grids=grids)
    residual_norms = np.atleast_1d(np.asarray(run.solve_result.residual_norms, dtype=np.float64))
    write_profile_output(
        path=native_out,
        inp=run.input,
        op=run.operator,
        grids=grids,
        geom=geom,
        radial=radial,
        state_vector=run.state_vector,
        solver_method=run.solve_result.method,
        residual_norm=float(np.max(residual_norms)) if residual_norms.size else None,
        fortran_layout=False,
    )

    fortran = _h5_datasets(fortran_out)
    native = _h5_datasets(native_out)
    assert set(fortran) == set(native)
    for key in ("particleFlux_vm_psiHat", "FSABFlow", "sources"):
        f_arr = np.asarray(fortran[key])
        n_arr = np.asarray(native[key])
        assert f_arr.ndim >= 2
        np.testing.assert_array_equal(n_arr, np.transpose(f_arr, tuple(reversed(range(f_arr.ndim)))))


def test_write_output_geometry_only_emits_base_and_geometry_fields(tmp_path: Path) -> None:
    """Canonical --geometry-only writes the state-independent base/geometry key set."""
    from dkx.run import run_geometry

    canonical_out = tmp_path / "canonical.h5"
    run = run_geometry(_TINY_DECK, out_path=canonical_out, emit=None)
    assert run.output_path == canonical_out.resolve()

    canonical = _h5_datasets(canonical_out)
    assert int(np.asarray(canonical["NIterations"]).reshape(())) == 0
    for key in ("BHat", "DHat", "theta", "zeta", "x", "GHat", "IHat", "iota", "psiAHat", "input.namelist"):
        assert key in canonical, key
    # No solve ran: the solution/moment datasets must be absent.
    for key in ("particleFlux_vm_psiHat", "FSABFlow", "transportMatrix"):
        assert key not in canonical, key
