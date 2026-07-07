from __future__ import annotations

import json
from pathlib import Path

from sfincs_jax.validation import write_output_trace


def _load_module():
    return write_output_trace


def test_profile_write_output_trace_main_runs_warmup_trace_and_memory_dump(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    localized = tmp_path / "localized"
    localized.mkdir()
    localized_input = localized / "input.namelist"
    localized_input.write_text("&general\n/\n", encoding="utf-8")

    trace_calls: list[tuple[str, bool]] = []
    device_mem_profiles: list[str] = []
    run_calls: list[dict[str, object]] = []
    prepare_calls: list[dict[str, object]] = []

    class _TraceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_prepare(path, **kwargs):
        prepare_calls.append({"path": path, **kwargs})
        return localized_input, localized

    monkeypatch.setattr(mod, "_prepare_input", _fake_prepare)
    monkeypatch.setattr(
        mod.jax_profiler,
        "trace",
        lambda log_dir, create_perfetto_trace=False: trace_calls.append((str(log_dir), bool(create_perfetto_trace))) or _TraceContext(),
    )
    monkeypatch.setattr(
        mod.jax_profiler,
        "save_device_memory_profile",
        lambda path: device_mem_profiles.append(str(path)),
    )
    monkeypatch.setattr(mod.jax, "block_until_ready", lambda x: x)

    def _fake_run(**kwargs):
        run_calls.append(dict(kwargs))
        Path(kwargs["output_path"]).write_bytes(b"")

    monkeypatch.setattr(mod, "_run_write_output", _fake_run)

    trace_dir = tmp_path / "trace"
    out_path = tmp_path / "out" / "sfincsOutput.h5"
    mem_path = tmp_path / "trace" / "device_memory.pb"
    solver_trace_path = tmp_path / "out" / "solver_trace.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--trace-dir",
            str(trace_dir),
            "--out",
            str(out_path),
            "--warmup",
            "1",
            "--perfetto",
            "--device-memory-profile",
            str(mem_path),
            "--compute-solution",
            "--wout-path",
            str(tmp_path / "wout_test.nc"),
            "--solver-trace",
            str(solver_trace_path),
        ]
    )

    assert rc == 0
    assert len(run_calls) == 2
    assert run_calls[0]["output_path"] == out_path.with_name("sfincsOutput.warmup.h5")
    assert run_calls[1]["output_path"] == out_path
    assert run_calls[0]["compute_solution"] is True
    assert run_calls[1]["compute_solution"] is True
    assert run_calls[0]["differentiable"] is False
    assert run_calls[1]["differentiable"] is False
    assert prepare_calls[0]["wout_path"] == str(tmp_path / "wout_test.nc")
    assert run_calls[1]["wout_path"] == str(tmp_path / "wout_test.nc")
    assert run_calls[0]["solver_trace_path"] is None
    assert run_calls[1]["solver_trace_path"] == solver_trace_path.resolve()
    assert trace_calls == [(str(trace_dir.resolve()), True)]
    assert device_mem_profiles == [str(mem_path)]
    phase_log = trace_dir / "profile_write_output_trace_phases.json"
    phases = json.loads(phase_log.read_text(encoding="utf-8"))
    assert phases["status"] == "completed"
    assert phases["differentiable"] is False
    assert phases["solver_trace"] == str(solver_trace_path.resolve())
    assert [phase["name"] for phase in phases["phases"]] == [
        "prepare_input",
        "warmup",
        "jax_trace",
        "write_output_solve",
        "block_until_ready",
        "device_memory_profile",
    ]


def test_profile_write_output_trace_keeps_solve_success_when_profiler_finalization_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    localized = tmp_path / "localized"
    localized.mkdir()
    localized_input = localized / "input.namelist"
    localized_input.write_text("&general\n/\n", encoding="utf-8")

    class _TraceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise RuntimeError("xplane finalization failed")
            return False

    monkeypatch.setattr(mod, "_prepare_input", lambda path, **kwargs: (localized_input, localized))
    monkeypatch.setattr(mod.jax_profiler, "trace", lambda *args, **kwargs: _TraceContext())
    monkeypatch.setattr(mod.jax, "block_until_ready", lambda x: x)

    def _fake_run(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"solved")

    monkeypatch.setattr(mod, "_run_write_output", _fake_run)

    trace_dir = tmp_path / "trace"
    out_path = tmp_path / "out" / "sfincsOutput.h5"
    phase_log = tmp_path / "phases.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--trace-dir",
            str(trace_dir),
            "--out",
            str(out_path),
            "--warmup",
            "0",
            "--phase-log",
            str(phase_log),
        ]
    )

    assert rc == 0
    assert out_path.read_bytes() == b"solved"
    phases = json.loads(phase_log.read_text(encoding="utf-8"))
    assert phases["status"] == "solve_completed_profile_incomplete"
    trace_phase = next(phase for phase in phases["phases"] if phase["name"] == "jax_trace")
    assert trace_phase["status"] == "failed"
    assert "xplane finalization failed" in trace_phase["exception"]


def test_profile_write_output_trace_strict_profiler_fails_on_finalization_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    localized = tmp_path / "localized"
    localized.mkdir()
    localized_input = localized / "input.namelist"
    localized_input.write_text("&general\n/\n", encoding="utf-8")

    class _TraceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise RuntimeError("perfetto flush failed")
            return False

    monkeypatch.setattr(mod, "_prepare_input", lambda path, **kwargs: (localized_input, localized))
    monkeypatch.setattr(mod.jax_profiler, "trace", lambda *args, **kwargs: _TraceContext())
    monkeypatch.setattr(mod.jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(mod, "_run_write_output", lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"solved"))

    trace_dir = tmp_path / "trace"
    out_path = tmp_path / "out" / "sfincsOutput.h5"
    phase_log = tmp_path / "phases.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--trace-dir",
            str(trace_dir),
            "--out",
            str(out_path),
            "--warmup",
            "0",
            "--phase-log",
            str(phase_log),
            "--strict-profiler",
        ]
    )

    assert rc == 1
    assert out_path.exists()
    phases = json.loads(phase_log.read_text(encoding="utf-8"))
    assert phases["status"] == "failed"


def test_profile_write_output_trace_keeps_solve_success_when_device_memory_snapshot_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    localized = tmp_path / "localized"
    localized.mkdir()
    localized_input = localized / "input.namelist"
    localized_input.write_text("&general\n/\n", encoding="utf-8")

    class _TraceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod, "_prepare_input", lambda path, **kwargs: (localized_input, localized))
    monkeypatch.setattr(mod.jax_profiler, "trace", lambda *args, **kwargs: _TraceContext())
    monkeypatch.setattr(
        mod.jax_profiler,
        "save_device_memory_profile",
        lambda path: (_ for _ in ()).throw(RuntimeError("tensorflow profiler hook unavailable")),
    )
    monkeypatch.setattr(mod.jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(mod, "_run_write_output", lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"solved"))

    trace_dir = tmp_path / "trace"
    out_path = tmp_path / "out" / "sfincsOutput.h5"
    phase_log = tmp_path / "phases.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--trace-dir",
            str(trace_dir),
            "--out",
            str(out_path),
            "--warmup",
            "0",
            "--phase-log",
            str(phase_log),
            "--device-memory-profile",
            str(trace_dir / "device_memory.prof"),
        ]
    )

    assert rc == 0
    phases = json.loads(phase_log.read_text(encoding="utf-8"))
    assert phases["status"] == "solve_completed_profile_incomplete"
    mem_phase = next(phase for phase in phases["phases"] if phase["name"] == "device_memory_profile")
    assert mem_phase["status"] == "failed"
    assert "tensorflow profiler hook unavailable" in mem_phase["exception"]


def test_profile_write_output_trace_no_jax_trace_skips_profiler_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    localized = tmp_path / "localized"
    localized.mkdir()
    localized_input = localized / "input.namelist"
    localized_input.write_text("&general\n/\n", encoding="utf-8")
    trace_calls: list[object] = []

    monkeypatch.setattr(mod, "_prepare_input", lambda path, **kwargs: (localized_input, localized))
    monkeypatch.setattr(mod.jax_profiler, "trace", lambda *args, **kwargs: trace_calls.append(args) or None)
    monkeypatch.setattr(mod.jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(mod, "_run_write_output", lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"solved"))

    trace_dir = tmp_path / "trace"
    out_path = tmp_path / "out" / "sfincsOutput.h5"
    phase_log = tmp_path / "phases.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--trace-dir",
            str(trace_dir),
            "--out",
            str(out_path),
            "--warmup",
            "0",
            "--phase-log",
            str(phase_log),
            "--no-jax-trace",
        ]
    )

    assert rc == 0
    assert trace_calls == []
    phases = json.loads(phase_log.read_text(encoding="utf-8"))
    assert phases["jax_trace"] is False
    trace_phase = next(phase for phase in phases["phases"] if phase["name"] == "jax_trace")
    assert trace_phase["enabled"] is False
    assert trace_phase["status"] == "ok"
