from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_reduced_upstream_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_reduced_upstream_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
suite = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suite
_SPEC.loader.exec_module(suite)


def test_executable_metadata_records_hash(tmp_path: Path) -> None:
    exe = tmp_path / "sfincs"
    exe.write_bytes(b"fake executable\n")

    metadata = suite._executable_metadata(exe)

    assert metadata["exists"] is True
    assert metadata["path"] == str(exe.resolve())
    assert metadata["size_bytes"] == exe.stat().st_size
    assert isinstance(metadata["sha256"], str)
    assert len(metadata["sha256"]) == 64


def test_fortran_final_residual_parser_and_quality_classifier(tmp_path: Path) -> None:
    log = tmp_path / "sfincs.log"
    log.write_text(
        "\n".join(
            [
                "--------- Residual function norm:  8.0376895E-03 -----------------------------",
                "--------- Residual function norm:  2.1662674E-03 -----------------------------",
            ]
        ),
        encoding="utf-8",
    )

    assert suite._parse_fortran_final_residual_norm_from_log(log) == pytest.approx(2.1662674e-3)
    assert (
        suite._classify_blocker(
            status="parity_mismatch",
            note="Fortran final residual=2.166e-03 exceeds solverTolerance=1.0e-07; reference-solve quality suspect.",
            mismatch_keys=["FSABFlow"],
            jax_log=None,
        )
        == "reference solver quality"
    )


def test_runtime_window_max_attempts_returns_last_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_input = tmp_path / "seed.namelist"
    reference_input = tmp_path / "reference.namelist"
    case_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 5\n  NZETA = 1\n  NX = 2\n  NXI = 4\n/\n",
        encoding="utf-8",
    )
    reference_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 11\n  NZETA = 7\n  NX = 5\n  NXI = 8\n/\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(suite, "localize_equilibrium_file_in_place", lambda *args, **kwargs: None)

    def fake_fortran(*, input_path: Path, exe: Path, timeout_s: float, log_path: Path):
        out = input_path.parent / "sfincsOutput.h5"
        out.write_text("fortran-h5", encoding="utf-8")
        log_path.write_text("Time to solve: 0.5 seconds\n", encoding="utf-8")
        return 0.5, out, 0, 100.0

    def fake_jax(
        *,
        input_path: Path,
        output_path: Path,
        timeout_s: float,
        log_path: Path,
        compute_solution: bool,
        compute_transport_matrix: bool,
        collect_iterations: bool = True,
        repeats: int = 1,
        cache_dir: Path | None = None,
        profile_mode: str = "off",
    ):
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.write_text("elapsed_s=0.5\n", encoding="utf-8")
        return 0.5, None, 200.0, 0.5

    monkeypatch.setattr(suite, "_run_fortran_direct", fake_fortran)
    monkeypatch.setattr(suite, "_run_jax_cli", fake_jax)
    monkeypatch.setattr(suite, "_compare_outputs", lambda *args, **kwargs: (10, 0, 0.0, []))
    monkeypatch.setattr(suite, "_compute_print_parity", lambda *args, **kwargs: (0, 0, []))
    monkeypatch.setattr(suite, "_parse_ksp_iterations", lambda *args, **kwargs: ([], []))

    result = suite._run_case(
        case_name="case",
        case_input=case_input,
        reference_input=reference_input,
        case_out_dir=tmp_path / "out",
        fortran_exe=tmp_path / "sfincs",
        timeout_s=1.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=1,
        target_runtime_s=1.0,
        target_runtime_max_s=20.0,
        target_runtime_max_iters=1,
        target_runtime_basis="fortran",
        use_seed_resolution=True,
        reuse_fortran=False,
        collect_iterations=False,
        jax_repeats=1,
        jax_cache_dir=None,
        jax_profile_mode="off",
    )

    assert result.status == "parity_ok"
    assert result.n_mismatch_common == 0
    assert result.fortran_runtime_s == pytest.approx(0.5)


def test_frozen_reference_reuse_aligns_to_staged_input_without_live_fortran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_input = tmp_path / "seed.namelist"
    reference_input = tmp_path / "reference.namelist"
    case_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 27\n  NZETA = 29\n  NX = 2\n  NXI = 70\n/\n",
        encoding="utf-8",
    )
    reference_input.write_text(case_input.read_text(encoding="utf-8"), encoding="utf-8")

    case_out_dir = tmp_path / "out"
    fortran_dir = case_out_dir / "fortran_run"
    fortran_dir.mkdir(parents=True)
    staged_input = (
        "&general\n/\n&resolutionParameters\n  NTHETA = 8\n  NZETA = 9\n  NX = 2\n  NXI = 23\n/\n"
    )
    (fortran_dir / "input.namelist").write_text(staged_input, encoding="utf-8")
    (fortran_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (fortran_dir / "sfincs.log").write_text("Time to solve: 0.5 seconds\n", encoding="utf-8")

    monkeypatch.setattr(suite, "localize_equilibrium_file_in_place", lambda *args, **kwargs: None)

    def fail_fortran(**kwargs):
        raise AssertionError("frozen reference lane should not rerun Fortran")

    def fake_jax(
        *,
        input_path: Path,
        output_path: Path,
        timeout_s: float,
        log_path: Path,
        compute_solution: bool,
        compute_transport_matrix: bool,
        collect_iterations: bool = True,
        repeats: int = 1,
        cache_dir: Path | None = None,
        profile_mode: str = "off",
    ):
        assert input_path.read_text(encoding="utf-8") == staged_input
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.write_text("elapsed_s=0.5\n", encoding="utf-8")
        return 0.5, None, 200.0, 0.5

    monkeypatch.setattr(suite, "_run_fortran_direct", fail_fortran)
    monkeypatch.setattr(suite, "_run_jax_cli", fake_jax)
    monkeypatch.setattr(suite, "_compare_outputs", lambda *args, **kwargs: (10, 0, 0.0, []))
    monkeypatch.setattr(suite, "_compute_print_parity", lambda *args, **kwargs: (0, 0, []))
    monkeypatch.setattr(suite, "_parse_ksp_iterations", lambda *args, **kwargs: ([], []))

    result = suite._run_case(
        case_name="mono",
        case_input=case_input,
        reference_input=reference_input,
        case_out_dir=case_out_dir,
        fortran_exe=tmp_path / "__unused_sfincs__",
        timeout_s=1.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=1,
        target_runtime_s=None,
        target_runtime_max_s=None,
        target_runtime_max_iters=0,
        target_runtime_basis="fortran",
        use_seed_resolution=True,
        reuse_fortran=True,
        collect_iterations=False,
        jax_repeats=1,
        jax_cache_dir=None,
        jax_profile_mode="off",
    )

    assert result.status == "parity_ok"
    assert result.final_resolution == {"NTHETA": 8, "NZETA": 9, "NX": 2, "NXI": 23}
    assert (case_out_dir / "input.namelist").read_text(encoding="utf-8") == staged_input


def test_frozen_reference_reuse_survives_jax_retry_without_live_fortran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_input = tmp_path / "seed.namelist"
    reference_input = tmp_path / "reference.namelist"
    case_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 27\n  NZETA = 29\n  NX = 2\n  NXI = 70\n/\n",
        encoding="utf-8",
    )
    reference_input.write_text(case_input.read_text(encoding="utf-8"), encoding="utf-8")

    case_out_dir = tmp_path / "out"
    fortran_dir = case_out_dir / "fortran_run"
    fortran_dir.mkdir(parents=True)
    staged_input = (
        "&general\n/\n&resolutionParameters\n  NTHETA = 8\n  NZETA = 9\n  NX = 2\n  NXI = 23\n/\n"
    )
    (fortran_dir / "input.namelist").write_text(staged_input, encoding="utf-8")
    (fortran_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (fortran_dir / "sfincs.log").write_text("Time to solve: 0.5 seconds\n", encoding="utf-8")

    monkeypatch.setattr(suite, "localize_equilibrium_file_in_place", lambda *args, **kwargs: None)

    def fail_fortran(**kwargs):
        raise AssertionError("frozen reference retry should not rerun Fortran")

    attempts = {"n": 0}

    def fake_jax(
        *,
        input_path: Path,
        output_path: Path,
        timeout_s: float,
        log_path: Path,
        compute_solution: bool,
        compute_transport_matrix: bool,
        collect_iterations: bool = True,
        repeats: int = 1,
        cache_dir: Path | None = None,
        profile_mode: str = "off",
    ):
        attempts["n"] += 1
        assert input_path.read_text(encoding="utf-8") == staged_input
        if attempts["n"] == 1:
            raise RuntimeError("transient GPU allocator failure")
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.write_text("elapsed_s=0.5\n", encoding="utf-8")
        return 0.5, None, 200.0, 0.5

    monkeypatch.setattr(suite, "_run_fortran_direct", fail_fortran)
    monkeypatch.setattr(suite, "_run_jax_cli", fake_jax)
    monkeypatch.setattr(suite, "_compare_outputs", lambda *args, **kwargs: (10, 0, 0.0, []))
    monkeypatch.setattr(suite, "_compute_print_parity", lambda *args, **kwargs: (0, 0, []))
    monkeypatch.setattr(suite, "_parse_ksp_iterations", lambda *args, **kwargs: ([], []))

    result = suite._run_case(
        case_name="mono",
        case_input=case_input,
        reference_input=reference_input,
        case_out_dir=case_out_dir,
        fortran_exe=tmp_path / "__unused_sfincs__",
        timeout_s=1.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=2,
        target_runtime_s=None,
        target_runtime_max_s=None,
        target_runtime_max_iters=0,
        target_runtime_basis="fortran",
        use_seed_resolution=True,
        reuse_fortran=True,
        collect_iterations=False,
        jax_repeats=1,
        jax_cache_dir=None,
        jax_profile_mode="off",
    )

    assert attempts["n"] == 2
    assert result.status == "parity_ok"
    assert result.final_resolution == {"NTHETA": 8, "NZETA": 9, "NX": 2, "NXI": 23}
    assert (case_out_dir / "input.namelist").read_text(encoding="utf-8") == staged_input


def test_run_jax_cli_defaults_to_profile_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    log_path = tmp_path / "sfincs_jax.log"
    seen_env: dict[str, str] = {}

    def fake_run(*, cmd, cwd, env, log_path, timeout_s, mode="w"):
        seen_env.update(env)
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.open(mode, encoding="utf-8").write("elapsed_s=0.5\n")
        return 0

    monkeypatch.setattr(suite, "_run_logged_subprocess", fake_run)

    cold, warm, rss_mb, logged = suite._run_jax_cli(
        input_path=input_path,
        output_path=output_path,
        timeout_s=1.0,
        log_path=log_path,
        compute_solution=False,
        compute_transport_matrix=False,
        collect_iterations=False,
        repeats=1,
        cache_dir=tmp_path / ".jax_cache",
    )

    assert cold == pytest.approx(0.5, abs=0.5)
    assert warm is None
    assert rss_mb is None
    assert logged == pytest.approx(0.5)
    assert seen_env["SFINCS_JAX_PRECOMPILE"] == "0"
    assert seen_env["SFINCS_JAX_PROFILE"] == "0"
    assert seen_env["SFINCS_JAX_PROFILE_DEVICE_MEM"] == "0"


def test_run_jax_cli_uses_solver_trace_sidecar_for_logged_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    log_path = tmp_path / "sfincs_jax.log"
    seen_cmd: list[str] = []

    def fake_run(*, cmd, cwd, env, log_path, timeout_s, mode="w"):
        seen_cmd[:] = list(cmd)
        output_path.write_text("jax-h5", encoding="utf-8")
        trace_path = Path(cmd[cmd.index("--solver-trace") + 1])
        trace_path.write_text(json.dumps({"schema_version": 1, "elapsed_s": 0.125}), encoding="utf-8")
        log_path.open(mode, encoding="utf-8").write("elapsed_s=9.0\n")
        return 0

    monkeypatch.setattr(suite, "_run_logged_subprocess", fake_run)

    _cold, _warm, _rss_mb, logged = suite._run_jax_cli(
        input_path=input_path,
        output_path=output_path,
        timeout_s=1.0,
        log_path=log_path,
        compute_solution=False,
        compute_transport_matrix=False,
        collect_iterations=False,
        repeats=1,
        cache_dir=None,
    )

    assert "--solver-trace" in seen_cmd
    assert logged == pytest.approx(0.125)


def test_run_jax_cli_uses_time_rss_when_profile_rss_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    log_path = tmp_path / "sfincs_jax.log"

    def fake_run(*, cmd, cwd, env, log_path, timeout_s, mode="w"):
        output_path.write_text("jax-h5", encoding="utf-8")
        trace_path = Path(cmd[cmd.index("--solver-trace") + 1])
        trace_path.write_text(json.dumps({"schema_version": 1, "elapsed_s": 0.25}), encoding="utf-8")
        log_path.open(mode, encoding="utf-8").write("Maximum resident set size (kbytes): 204800\n")
        return 0

    monkeypatch.setattr(suite, "_run_logged_subprocess", fake_run)

    _cold, _warm, rss_mb, logged = suite._run_jax_cli(
        input_path=input_path,
        output_path=output_path,
        timeout_s=1.0,
        log_path=log_path,
        compute_solution=False,
        compute_transport_matrix=False,
        collect_iterations=False,
        repeats=1,
        cache_dir=None,
    )

    assert rss_mb == pytest.approx(200.0)
    assert logged == pytest.approx(0.25)


def test_run_jax_cli_leaves_compilation_cache_unset_when_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    log_path = tmp_path / "sfincs_jax.log"
    seen_env: dict[str, str] = {}

    def fake_run(*, cmd, cwd, env, log_path, timeout_s, mode="w"):
        seen_env.update(env)
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.open(mode, encoding="utf-8").write("elapsed_s=0.5\n")
        return 0

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(suite, "_run_logged_subprocess", fake_run)

    suite._run_jax_cli(
        input_path=input_path,
        output_path=output_path,
        timeout_s=1.0,
        log_path=log_path,
        compute_solution=False,
        compute_transport_matrix=False,
        collect_iterations=False,
        repeats=1,
        cache_dir=None,
    )

    assert "JAX_COMPILATION_CACHE_DIR" not in seen_env


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_run_logged_subprocess_timeout_kills_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script_path = tmp_path / "spawn_child.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import subprocess
            import sys
            import time
            from pathlib import Path

            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
            try:
                time.sleep(30)
            finally:
                child.poll()
            """
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "run.log"

    with pytest.raises(subprocess.TimeoutExpired):
        suite._run_logged_subprocess(
            cmd=[sys.executable, str(script_path)],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=log_path,
            timeout_s=0.2,
        )

    deadline = time.time() + 3.0
    while not child_pid_path.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.time() + 3.0
    while _pid_is_alive(child_pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _pid_is_alive(child_pid)
