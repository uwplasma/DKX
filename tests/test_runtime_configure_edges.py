"""Branches of ``dkx.runtime.configure`` that the default environment never takes."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from dkx import runtime


@pytest.fixture
def env(monkeypatch):
    """A private environment and a recorded ``jax.config.update``; ``configure`` re-runs on each call."""
    import jax

    clean = {k: v for k, v in os.environ.items() if not k.startswith(("DKX_", "JAX_", "NPROC", "XDG_"))}
    monkeypatch.setattr(os, "environ", clean)
    updates: dict = {}
    monkeypatch.setattr(jax.config, "update", lambda name, value: updates.__setitem__(name, value))
    monkeypatch.setattr(runtime, "_configured", False)
    return clean, updates


def test_numpy_one_is_refused_with_the_fix(monkeypatch) -> None:
    monkeypatch.setattr(np, "__version__", "1.26.4")
    with pytest.raises(ImportError, match="numpy>=2.1"):
        runtime._check_numpy()


def test_thread_split_falls_back_to_cpu_count_without_affinity(monkeypatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    monkeypatch.delenv("DKX_CPU_DEVICES", raising=False)
    assert runtime._default_thread_split() == (3, 3)
    monkeypatch.setenv("DKX_CPU_DEVICES", "2")
    assert runtime._default_thread_split() == (3, 1)


def test_cores_and_cpu_device_knobs(env) -> None:
    clean, _ = env
    clean.update(DKX_CORES="3", DKX_CPU_DEVICES="2")
    runtime.configure()
    assert clean["NPROC"] == "3" and "--xla_force_host_platform_device_count=2" in clean["XLA_FLAGS"]
    for cores, nproc in (("0", None), ("junk", "9")):  # 0 undoes our own clamp; junk changes nothing
        clean.clear()
        clean.update(DKX_CORES=cores, DKX_CPU_DEVICES="junk", NPROC="9", _DKX_NPROC_DEFAULTED="1")
        runtime._configured = False
        runtime.configure()
        assert clean.get("NPROC") == nproc and "XLA_FLAGS" not in clean


def test_cache_directory_fallbacks(env, monkeypatch, tmp_path) -> None:
    clean, updates = env
    clean["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
    runtime.configure()
    assert clean["JAX_COMPILATION_CACHE_DIR"].startswith(str(tmp_path / "xdg" / "dkx"))
    assert updates["jax_compilation_cache_dir"] == clean["JAX_COMPILATION_CACHE_DIR"]

    real_makedirs = os.makedirs
    tmp = str(tmp_path / "tmp")
    monkeypatch.setattr("tempfile.gettempdir", lambda: tmp)

    def refuse_primary(path, exist_ok=False):
        if not str(path).startswith(tmp):
            raise OSError("read-only")
        real_makedirs(path, exist_ok=exist_ok)

    for stage, patch in (("primary refused", refuse_primary), ("everything refused", lambda p, exist_ok=False: (_ for _ in ()).throw(OSError()))):
        clean.clear()
        runtime._configured = False
        monkeypatch.setattr(os, "makedirs", patch)
        runtime.configure()
        assert ("JAX_COMPILATION_CACHE_DIR" in clean) == (stage == "primary refused"), stage
    monkeypatch.setattr(os, "makedirs", real_makedirs)

    # Created but unwritable: first the tempdir takes over, then, if it is unwritable too, no cache at all.
    for writable_after, expect in ((True, True), (False, False)):
        clean.clear()
        clean["DKX_COMPILATION_CACHE_DIR"] = str(tmp_path / "ro")
        runtime._configured = False
        real_open = open
        calls = []

        def picky_open(path, *a, **k):
            if str(path).endswith(".dkx_write_test"):
                calls.append(path)
                if not (writable_after and len(calls) > 1):
                    raise OSError("read-only")
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", picky_open)
        runtime.configure()
        monkeypatch.setattr("builtins.open", real_open)
        assert ("JAX_COMPILATION_CACHE_DIR" in clean) == expect


def test_precision_switch_bad_thresholds_and_a_missing_jax(env, monkeypatch) -> None:
    clean, updates = env
    clean.update(DKX_NO_X64_SETUP="1", JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="soon")
    runtime.configure()
    assert "jax_enable_x64" not in updates
    clean.clear()
    runtime._configured = False
    runtime.configure(jax_x64=True)
    assert updates["jax_enable_x64"] is True
    clean.clear()
    runtime._configured = False
    monkeypatch.setitem(sys.modules, "jax", None)  # configure on a machine without JAX
    runtime.configure()
    assert runtime.is_configured()
