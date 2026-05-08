from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sfincs_jax import cli
from sfincs_jax.input_compat import effective_equilibrium_file
from sfincs_jax.io import (
    _select_rhsmode1_linear_solve_method,
    _select_phi1_newton_linear_solve_method,
    _select_phi1_use_frozen_linearization,
    read_sfincs_h5,
    write_sfincs_jax_output_h5,
)


class _FakeNamelist:
    def __init__(self, rhs_mode: int = 1) -> None:
        self._groups = {
            "general": {"RHSMODE": rhs_mode},
            "geometryParameters": {},
            "physicsParameters": {},
            "resolutionParameters": {},
        }

    def group(self, name: str):
        return self._groups.get(name, {})


def test_cmd_write_output_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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
    assert captured["differentiable"] is False
    assert captured["solver_trace_path"] == Path(tmp_path / "solver_trace.json")


def test_cmd_write_output_accepts_extension_selected_formats(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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
    assert Path(captured["output_path"]).suffix == ".nc"


def test_default_plot_output_path_uses_pdf() -> None:
    assert cli._default_plot_output_path(Path("sfincsOutput.h5")).name == "sfincsOutput_summary.pdf"


def test_cmd_solve_v3_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_solve(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(x=np.zeros((2,), dtype=np.float64), residual_norm=np.float64(0.0))

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.v3_driver.solve_v3_full_system_linear_gmres", _fake_solve)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
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
    assert cli._cmd_solve_v3(args) == 0
    assert captured["differentiable"] is False


def test_cmd_transport_matrix_v3_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_transport(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            transport_matrix=np.zeros((2, 2), dtype=np.float64),
            state_vectors_by_rhs={},
            residual_norms_by_rhs={1: np.float64(0.0)},
        )

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("sfincs_jax.v3_driver.solve_v3_transport_matrix_linear_gmres", _fake_transport)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_matrix=str(tmp_path / "tm.npy"),
        out_state_prefix=None,
        equilibrium_file=None,
        wout_path=None,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_transport_matrix_v3(args) == 0
    assert captured["differentiable"] is False


def test_write_output_full_system_regression(tmp_path: Path, monkeypatch) -> None:
    """Full-system write-output should not reference transport-only distributed state."""
    input_path = (
        Path(__file__).parent / "reduced_inputs" / "inductiveE_noEr.input.namelist"
    )
    assert input_path.exists()

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
    )

    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"]).item()) == 1
    assert "classicalParticleFluxNoPhi1_psiHat" in data


def test_phi1_newton_auto_method_uses_dense_on_cpu() -> None:
    msgs: list[str] = []

    method = _select_phi1_newton_linear_solve_method(
        active_total_size=1090,
        dense_cutoff=5000,
        default_method="incremental",
        fast_explicit=False,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "dense"
    assert any("using dense Newton step" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_accepts_lgmres() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="lgmres",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "lgmres"
    assert any("solve method forced by env -> lgmres" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_accepts_sparse_host() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="sparse_host",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "sparse_host"
    assert any("solve method forced by env -> sparse_host" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_accepts_sparse_pc_gmres() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="sparse_pc_gmres",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "sparse_pc_gmres"
    assert any("solve method forced by env -> sparse_pc_gmres" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_accepts_xblock_sparse_pc_gmres() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="xblock_sparse_pc_gmres",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "xblock_sparse_pc_gmres"
    assert any("solve method forced by env -> xblock_sparse_pc_gmres" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_accepts_sparse_lsmr() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="sparse_lsmr",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "sparse_lsmr"
    assert any("solve method forced by env -> sparse_lsmr" in msg for msg in msgs)


def test_rhsmode1_solve_method_env_ignores_unknown_override() -> None:
    msgs: list[str] = []

    method = _select_rhsmode1_linear_solve_method(
        default_method="incremental",
        env_override="not_a_method",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "incremental"
    assert msgs == []


def test_phi1_newton_auto_method_skips_dense_on_gpu() -> None:
    msgs: list[str] = []

    method = _select_phi1_newton_linear_solve_method(
        active_total_size=1090,
        dense_cutoff=5000,
        default_method="incremental",
        fast_explicit=False,
        dense_auto_ok=False,
        dense_auto_backend="gpu",
        env_override="",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "incremental"
    assert any("skipping dense auto mode on backend=gpu" in msg for msg in msgs)


def test_phi1_newton_fast_explicit_prefers_sparse_direct_on_large_cpu() -> None:
    msgs: list[str] = []

    method = _select_phi1_newton_linear_solve_method(
        active_total_size=68000,
        dense_cutoff=5000,
        default_method="batched",
        fast_explicit=True,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "sparse_direct"
    assert any("host sparse-direct Newton step" in msg for msg in msgs)


def test_phi1_newton_fast_explicit_prefers_sparse_direct_on_large_gpu() -> None:
    msgs: list[str] = []

    method = _select_phi1_newton_linear_solve_method(
        active_total_size=12753,
        dense_cutoff=5000,
        default_method="incremental",
        fast_explicit=True,
        dense_auto_ok=False,
        dense_auto_backend="gpu",
        env_override="",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )

    assert method == "sparse_direct"
    assert any("backend=gpu" in msg for msg in msgs)


def test_phi1_newton_fast_explicit_prefers_sparse_direct_on_moderate_cpu() -> None:
    method = _select_phi1_newton_linear_solve_method(
        active_total_size=5703,
        dense_cutoff=5000,
        default_method="incremental",
        fast_explicit=True,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="",
        emit=None,
    )

    assert method == "sparse_direct"


def test_phi1_frozen_linearization_policy_keeps_sparse_direct_full_newton() -> None:
    assert not _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="sparse_direct",
        env_value="",
    )
    assert _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="incremental",
        env_value="",
    )


def test_phi1_frozen_linearization_policy_respects_env_overrides() -> None:
    assert _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="sparse_direct",
        env_value="1",
    )
    assert not _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="incremental",
        env_value="0",
    )


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
        "--shard-axis",
        "theta",
        "--distributed-gmres",
        "auto",
        "--transport-workers",
        "3",
        "input.namelist",
        "--out",
        "sfincsOutput.h5",
    ]
    assert cli._normalize_default_argv(argv) == [
        "--cores",
        "8",
        "--shard-axis",
        "theta",
        "--distributed-gmres",
        "auto",
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


def test_apply_parallel_runtime_settings_sets_transport_and_sharding(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_MATVEC_SHARD_AXIS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_AUTO_SHARD", raising=False)
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED", raising=False)
    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", raising=False)
    monkeypatch.delenv("SFINCS_JAX_SHARD_PAD", raising=False)

    cli._apply_parallel_runtime_settings(
        Namespace(
            transport_workers=4,
            shard_axis="theta",
            distributed_gmres="auto",
            distributed_krylov="gmres",
            shard_pad=False,
            distributed=False,
            process_id=None,
            process_count=None,
            coordinator_address=None,
            coordinator_port=None,
        )
    )

    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] == "process"
    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] == "4"
    assert os.environ["SFINCS_JAX_MATVEC_SHARD_AXIS"] == "theta"
    assert os.environ["SFINCS_JAX_AUTO_SHARD"] == "0"
    assert os.environ["SFINCS_JAX_GMRES_DISTRIBUTED"] == "auto"
    assert os.environ["SFINCS_JAX_DISTRIBUTED_KRYLOV"] == "gmres"
    assert os.environ["SFINCS_JAX_SHARD_PAD"] == "0"


def test_apply_parallel_runtime_settings_initializes_distributed(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PROCESS_ID", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PROCESS_COUNT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_COORDINATOR_ADDRESS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_COORDINATOR_PORT", raising=False)
    monkeypatch.setattr(
        "sfincs_jax.initialize_distributed_runtime_from_env",
        lambda: calls.append("init") or True,
    )

    cli._apply_parallel_runtime_settings(
        Namespace(
            transport_workers=None,
            shard_axis=None,
            distributed_gmres=None,
            distributed_krylov=None,
            shard_pad=None,
            distributed=True,
            process_id=1,
            process_count=2,
            coordinator_address="worker0",
            coordinator_port=2345,
        )
    )

    assert os.environ["SFINCS_JAX_DISTRIBUTED"] == "1"
    assert os.environ["SFINCS_JAX_PROCESS_ID"] == "1"
    assert os.environ["SFINCS_JAX_PROCESS_COUNT"] == "2"
    assert os.environ["SFINCS_JAX_COORDINATOR_ADDRESS"] == "worker0"
    assert os.environ["SFINCS_JAX_COORDINATOR_PORT"] == "2345"
    assert calls == ["init"]


def test_emit_parallel_runtime_info_reports_active_parallel_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("SFINCS_JAX_CORES", "8")
    monkeypatch.setenv("SFINCS_JAX_CPU_DEVICES", "8")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "1")
    monkeypatch.setenv("SFINCS_JAX_SHARD_PAD", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "process")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "4")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", "bicgstab")
    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED", "1")
    monkeypatch.setenv("SFINCS_JAX_PROCESS_ID", "1")
    monkeypatch.setenv("SFINCS_JAX_PROCESS_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_COORDINATOR_ADDRESS", "node0")
    monkeypatch.setenv("SFINCS_JAX_COORDINATOR_PORT", "1234")

    cli._emit_parallel_runtime_info(args=Namespace(verbose=1, quiet=False))
    out = capsys.readouterr().out

    assert "parallel: cores=8 cpu_devices=8 shard_axis=theta auto_shard=1 shard_pad=1" in out
    assert "transport_parallel: mode=process workers=4" in out
    assert "distributed_solver: gmres=auto krylov=bicgstab" in out
    assert "multi_host: enabled=1 process_id=1 process_count=2 coordinator=node0 port=1234" in out


def test_emit_parallel_runtime_info_suppresses_empty_state(monkeypatch, capsys) -> None:
    for name in (
        "SFINCS_JAX_CORES",
        "SFINCS_JAX_CPU_DEVICES",
        "SFINCS_JAX_MATVEC_SHARD_AXIS",
        "SFINCS_JAX_AUTO_SHARD",
        "SFINCS_JAX_SHARD_PAD",
        "SFINCS_JAX_TRANSPORT_PARALLEL",
        "SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS",
        "SFINCS_JAX_GMRES_DISTRIBUTED",
        "SFINCS_JAX_DISTRIBUTED_KRYLOV",
        "SFINCS_JAX_DISTRIBUTED",
        "SFINCS_JAX_PROCESS_ID",
        "SFINCS_JAX_PROCESS_COUNT",
        "SFINCS_JAX_COORDINATOR_ADDRESS",
        "SFINCS_JAX_COORDINATOR_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    cli._emit_parallel_runtime_info(args=Namespace(verbose=1, quiet=False))
    assert capsys.readouterr().out == ""


def test_maybe_reexec_for_early_runtime_reexecs_for_cores(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_CORES", raising=False)
    monkeypatch.delenv("SFINCS_JAX_CPU_DEVICES", raising=False)
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

    assert captured["argv"] == [cli.sys.executable, "-m", "sfincs_jax", "--cores", "4", "write-output", "--input", "input.namelist"]
    env = captured["env"]
    assert env["SFINCS_JAX_CORES"] == "4"
    assert env["SFINCS_JAX_CPU_DEVICES"] == "4"
    assert env["SFINCS_JAX_CLI_BOOTSTRAPPED"] == "1"


def test_maybe_reexec_for_early_runtime_skips_when_env_matches(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_CORES", "4")
    monkeypatch.setenv("SFINCS_JAX_CPU_DEVICES", "4")
    called = []
    monkeypatch.setattr(cli.os, "execvpe", lambda *args, **kwargs: called.append((args, kwargs)))
    cli._maybe_reexec_for_early_runtime(["--cores", "4", "write-output", "--input", "input.namelist"])
    assert called == []


def test_maybe_reexec_for_early_runtime_reexecs_for_distributed(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_DISTRIBUTED",
        "SFINCS_JAX_PROCESS_ID",
        "SFINCS_JAX_PROCESS_COUNT",
        "SFINCS_JAX_COORDINATOR_ADDRESS",
        "SFINCS_JAX_COORDINATOR_PORT",
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
    assert env["SFINCS_JAX_DISTRIBUTED"] == "1"
    assert env["SFINCS_JAX_PROCESS_ID"] == "1"
    assert env["SFINCS_JAX_PROCESS_COUNT"] == "2"
    assert env["SFINCS_JAX_COORDINATOR_ADDRESS"] == "node0"
    assert env["SFINCS_JAX_COORDINATOR_PORT"] == "1234"


def test_cmd_solve_v3_applies_equilibrium_override(monkeypatch, tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    captured: dict[str, object] = {}

    def _fake_solve(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(x=np.zeros((2,), dtype=np.float64), residual_norm=np.float64(0.0))

    monkeypatch.setattr("sfincs_jax.v3_driver.solve_v3_full_system_linear_gmres", _fake_solve)

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
    nml = captured["nml"]
    assert effective_equilibrium_file(geom_params=nml.group("geometryParameters")) == "override_wout.nc"


def test_main_accepts_quiet_after_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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
    assert captured["output_path"] == Path(tmp_path / "sfincsOutput.h5")


def test_main_write_output_forwards_solver_trace_sidecar(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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


def test_main_write_output_forwards_sparse_host_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "sparse_host",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "sparse_host"


def test_main_write_output_forwards_sparse_pc_gmres_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "sparse_pc_gmres",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "sparse_pc_gmres"


def test_main_write_output_forwards_xblock_sparse_pc_gmres_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "xblock_sparse_pc_gmres",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "xblock_sparse_pc_gmres"


def test_main_write_output_forwards_sparse_lsmr_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "sparse_lsmr",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "sparse_lsmr"


def test_main_write_output_forwards_sparse_host_safe_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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


def test_main_write_output_forwards_petsc_compat_solve_method(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    rc = cli.main(
        [
            "write-output",
            "--input",
            str(tmp_path / "input.namelist"),
            "--out",
            str(tmp_path / "sfincsOutput.h5"),
            "--solve-method",
            "petsc_compat",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["solve_method"] == "petsc_compat"


def test_main_write_output_reports_runtime_errors_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    def _fake_write_output_h5(**_kwargs):
        raise RuntimeError("host sparse factorization failed")

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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
    assert "sfincs_jax write-output failed: host sparse factorization failed" in captured.err
    assert "Traceback" not in captured.err


def test_main_preserves_shard_axis_before_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured["shard_axis"] = os.environ.get("SFINCS_JAX_MATVEC_SHARD_AXIS")
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)
    monkeypatch.setenv("SFINCS_JAX_CORES", "4")
    monkeypatch.setenv("SFINCS_JAX_CPU_DEVICES", "4")

    rc = cli.main(
        [
            "--cores",
            "4",
            "--shard-axis",
            "theta",
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
    assert captured["shard_axis"] == "theta"


def test_main_preserves_transport_workers_before_subcommand(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured["parallel_mode"] = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL")
        captured["workers"] = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS")
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

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
