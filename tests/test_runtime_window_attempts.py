from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_reduced_upstream_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_reduced_upstream_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
suite = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suite
_SPEC.loader.exec_module(suite)


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

    def fake_run(cmd, cwd, check, timeout, stdout, stderr, env):
        seen_env.update(env)
        output_path.write_text("jax-h5", encoding="utf-8")
        stdout.write("elapsed_s=0.5\n")
        return 0

    monkeypatch.setattr(suite.subprocess, "run", fake_run)

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


def test_run_jax_cli_leaves_compilation_cache_unset_when_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    log_path = tmp_path / "sfincs_jax.log"
    seen_env: dict[str, str] = {}

    def fake_run(cmd, cwd, check, timeout, stdout, stderr, env):
        seen_env.update(env)
        output_path.write_text("jax-h5", encoding="utf-8")
        stdout.write("elapsed_s=0.5\n")
        return 0

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(suite.subprocess, "run", fake_run)

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
