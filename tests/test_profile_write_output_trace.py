from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "profile_write_output_trace.py"
    spec = importlib.util.spec_from_file_location("profile_write_output_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    class _TraceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod, "_prepare_input", lambda path: (localized_input, localized))
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
        ]
    )

    assert rc == 0
    assert len(run_calls) == 2
    assert run_calls[0]["output_path"] == out_path.with_name("sfincsOutput.warmup.h5")
    assert run_calls[1]["output_path"] == out_path
    assert run_calls[0]["compute_solution"] is True
    assert run_calls[1]["compute_solution"] is True
    assert trace_calls == [(str(trace_dir.resolve()), True)]
    assert device_mem_profiles == [str(mem_path)]
