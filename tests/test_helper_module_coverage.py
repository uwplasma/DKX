from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.ambipolar import (
    _fortran_bool_to_py,
    _infer_var_name_from_scan_input,
    _scanplot2_labels,
    _scanplot2_outputs_for_run,
    radial_current_from_output,
)
from sfincs_jax.diagnostics import b0_over_bbar, fsab_hat2, g_hat_i_hat, u_hat, u_hat_np, vprime_hat
from sfincs_jax.validation.fortran import default_fortran_exe, run_sfincs_fortran
from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import Namelist
from sfincs_jax.paths import _strip_quotes, resolve_existing_path
from sfincs_jax.profiling import SimpleProfiler, _device_mem_mb, _rss_mb, maybe_profiler
from sfincs_jax.scans import _er_scan_var_name, _patch_scalar_in_group, linspace_including_endpoints, run_er_scan
from sfincs_jax.problems.transport_matrix.parallel.worker import main as transport_worker_main
from sfincs_jax.profiling import Timer, make_emit


def test_fortran_bool_and_radial_current_helpers() -> None:
    assert _fortran_bool_to_py(True) is True
    assert _fortran_bool_to_py(np.bool_(False)) is False
    assert _fortran_bool_to_py(np.asarray([1])) is True
    assert _fortran_bool_to_py(np.asarray(0)) is False

    include_phi1 = {
        "includePhi1": 1,
        "Zs": np.asarray([1.0, 2.0]),
        "particleFlux_vm_rHat": np.asarray([[1.0, 100.0], [2.0, 200.0]]),
        "particleFlux_vd_rHat": np.asarray([[3.0, 10.0], [4.0, 20.0]]),
    }
    no_phi1 = {
        "includePhi1": 0,
        "Zs": np.asarray([1.0, -1.0]),
        "particleFlux_vm_rHat": np.asarray([[0.0, 8.0], [0.0, 3.0]]),
    }

    assert radial_current_from_output(include_phi1) == pytest.approx(10.0 + 2.0 * 20.0)
    assert radial_current_from_output(no_phi1) == pytest.approx(5.0)


def test_scanplot2_helpers_cover_single_and_multi_species() -> None:
    labels = _scanplot2_labels(n_species=1, include_phi1=False)
    assert labels[-1] == "radial current"
    assert "source 1" in labels

    labels_multi = _scanplot2_labels(n_species=2, include_phi1=True)
    assert labels_multi == [
        "FSABFlow (species 1)",
        "particleFlux rHat (species 1)",
        "heatFlux rHat (species 1)",
        "FSABFlow (species 2)",
        "particleFlux rHat (species 2)",
        "heatFlux rHat (species 2)",
        "FSABjHat",
        "radial current",
    ]

    single = {
        "Nspecies": 1,
        "includePhi1": 0,
        "FSABFlow": np.asarray([2.0]),
        "particleFlux_vm_rHat": np.asarray([3.0]),
        "heatFlux_vm_rHat": np.asarray([4.0]),
        "sources": np.asarray([5.0, 6.0]),
        "FSABjHat": np.asarray([7.0]),
        "Zs": np.asarray([2.0]),
    }
    out_single = _scanplot2_outputs_for_run(single)
    np.testing.assert_allclose(out_single, np.asarray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0]))

    multi = {
        "Nspecies": 2,
        "includePhi1": 1,
        "FSABFlow": np.asarray([[1.0], [2.0]]),
        "particleFlux_vd_rHat": np.asarray([[3.0], [4.0]]),
        "heatFlux_vd_rHat": np.asarray([[5.0], [6.0]]),
        "FSABjHat": np.asarray([7.0]),
        "Zs": np.asarray([1.0, -1.0]),
    }
    out_multi = _scanplot2_outputs_for_run(multi)
    np.testing.assert_allclose(out_multi, np.asarray([1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, -1.0]))


def test_diagnostics_flux_surface_averages_for_constant_geometry() -> None:
    grids = SimpleNamespace(
        theta_weights=np.asarray([1.0, 2.0]),
        zeta_weights=np.asarray([3.0, 4.0]),
        theta=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False),
        zeta=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False),
    )
    geom = SimpleNamespace(
        d_hat=jnp.full((2, 2), 2.0),
        b_hat=jnp.full((2, 2), 5.0),
        b_hat_sub_zeta=jnp.full((2, 2), 7.0),
        b_hat_sub_theta=jnp.full((2, 2), 11.0),
        iota=0.7,
        g_hat=1.2,
        i_hat=-0.5,
        n_periods=1,
    )
    weight_sum = np.sum(np.asarray(grids.theta_weights)[:, None] * np.asarray(grids.zeta_weights)[None, :])
    assert float(vprime_hat(grids=grids, geom=geom)) == pytest.approx(weight_sum / 2.0)
    assert float(fsab_hat2(grids=grids, geom=geom)) == pytest.approx(25.0)
    assert float(b0_over_bbar(grids=grids, geom=geom)) == pytest.approx(5.0)
    g_hat, i_hat = g_hat_i_hat(grids=grids, geom=geom)
    assert float(g_hat) == pytest.approx(7.0 * weight_sum / (4.0 * np.pi * np.pi))
    assert float(i_hat) == pytest.approx(11.0 * weight_sum / (4.0 * np.pi * np.pi))


def test_u_hat_is_zero_for_constant_bhat() -> None:
    grids = SimpleNamespace(
        theta=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        zeta=np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False),
    )
    geom = SimpleNamespace(
        b_hat=np.full((8, 6), 3.0),
        iota=0.4,
        g_hat=1.3,
        i_hat=-0.7,
        n_periods=2,
    )
    np.testing.assert_allclose(np.asarray(u_hat(grids=grids, geom=geom)), 0.0, atol=1e-12)
    np.testing.assert_allclose(u_hat_np(grids=grids, geom=geom), 0.0, atol=1e-12)


def test_indexing_and_paths_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    indexing = V3Indexing(
        n_species=2,
        n_x=2,
        n_theta=3,
        n_zeta=4,
        n_xi_max=5,
        n_xi_for_x=np.asarray([2, 1]),
    )
    assert indexing.dke_size == 36
    np.testing.assert_array_equal(indexing.first_index_for_x, np.asarray([0, 2]))
    assert indexing.f_index(i_species=1, i_x=1, i_xi=0, i_theta=2, i_zeta=3) == 71
    inv = indexing.build_inverse_f_map()
    assert len(inv) == 72
    assert inv[35] == (0, 1, 0, 2, 3)

    target = tmp_path / "equilibria" / "wout.nc"
    target.parent.mkdir()
    target.write_text("ok")
    monkeypatch.setenv("SFINCS_JAX_EQUILIBRIA_DIRS", f'"{target.parent}"')
    assert _strip_quotes("'abc'") == "abc"
    resolved = resolve_existing_path('"wout.nc"', base_dir=tmp_path)
    assert resolved.path == target
    assert target in resolved.tried
    stale_absolute = tmp_path / "missing" / "machine" / "path" / "wout.nc"
    resolved_stale = resolve_existing_path(stale_absolute, base_dir=tmp_path)
    assert resolved_stale.path == target
    assert stale_absolute in resolved_stale.tried


def test_profiling_and_verbose_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import sfincs_jax.profiling as profiling

    fake_psutil = SimpleNamespace(
        Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=12_500_000))
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _rss_mb() == pytest.approx(12.5)

    device_mem_calls: list[int] = []
    monkeypatch.setattr(
        "jax.devices",
        lambda: device_mem_calls.append(1) or [SimpleNamespace(memory_stats=lambda: {"bytes_active": 3_000_000})],
    )
    assert _device_mem_mb() == pytest.approx(3.0)

    monkeypatch.setattr(profiling.time, "perf_counter", lambda: 11.25)
    profiler = SimpleProfiler(emit=None, sample_device_mem=False, t0=10.0, last=10.5, rss0_mb=12.5)
    profiler.mark("phase")
    assert profiler.entries[0]["dt_s"] == pytest.approx(0.75)
    assert profiler.entries[0]["total_s"] == pytest.approx(1.25)
    assert profiler.entries[0]["device_mb"] is None
    assert device_mem_calls == [1]

    profiler_with_dev = SimpleProfiler(emit=None, sample_device_mem=True, t0=10.0, last=10.5, rss0_mb=12.5)
    profiler_with_dev.mark("phase_dev")
    assert profiler_with_dev.entries[0]["device_mb"] == pytest.approx(3.0)
    assert device_mem_calls == [1, 1]

    monkeypatch.setenv("SFINCS_JAX_PROFILE", "on")
    monkeypatch.delenv("SFINCS_JAX_PROFILE_DEVICE_MEM", raising=False)
    prof = maybe_profiler()
    assert prof is not None
    assert prof.sample_device_mem is False
    monkeypatch.setenv("SFINCS_JAX_PROFILE_DEVICE_MEM", "1")
    prof = maybe_profiler()
    assert prof is not None
    assert prof.sample_device_mem is True
    monkeypatch.delenv("SFINCS_JAX_PROFILE_DEVICE_MEM", raising=False)
    monkeypatch.setenv("SFINCS_JAX_PROFILE", "full")
    prof = maybe_profiler()
    assert prof is not None
    assert prof.sample_device_mem is True
    monkeypatch.setenv("SFINCS_JAX_PROFILE", "off")
    assert maybe_profiler() is None

    lines: list[str] = []
    emit = make_emit(verbose=1, stream=SimpleNamespace(write=lambda s: lines.append(s), flush=lambda: None))
    emit(0, "hello")
    emit(2, "skip")
    assert any("hello" in line for line in lines)
    assert not any("skip" in line for line in lines)

    import sfincs_jax.profiling as verbose

    t = iter([2.0, 2.4])
    monkeypatch.setattr(verbose.time, "perf_counter", lambda: next(t))
    timer = Timer()
    assert timer.elapsed_s() == pytest.approx(0.4)


def test_fortran_wrapper_and_entrypoint_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_FORTRAN_EXE", str(tmp_path / "sfincs"))
    assert default_fortran_exe() == tmp_path / "sfincs"

    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    exe = tmp_path / "sfincs"
    exe.write_text("")
    exe.chmod(0o755)
    workdir = tmp_path / "work"
    calls: list[tuple[list[str], str, float | None]] = []

    def _fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        assert input_namelist == workdir / "input.namelist"
        assert overwrite is False

    def _fake_run(cmd, cwd, stdout, stderr, env, check, timeout):
        calls.append((cmd, cwd, timeout))
        assert check is False
        assert env["EXTRA"] == "1"
        Path(cwd, "sfincsOutput.h5").write_bytes(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("sfincs_jax.io.localize_equilibrium_file_in_place", _fake_localize)
    monkeypatch.setattr("sfincs_jax.validation.fortran.subprocess.run", _fake_run)

    out = run_sfincs_fortran(
        input_namelist=input_path,
        exe=exe,
        workdir=workdir,
        env={"EXTRA": "1"},
        timeout_s=5.0,
    )
    assert out == workdir / "sfincsOutput.h5"
    assert calls == [([str(exe.resolve())], str(workdir.resolve()), 5.0)]

    def _fake_finalize_failure(cmd, cwd, stdout, stderr, env, check, timeout):
        Path(cwd, "sfincsOutput.h5").write_bytes(b"complete")
        stdout.write("Saving diagnostics to h5 file for iteration 1\n")
        stdout.write("Goodbye!\n")
        stdout.write("MPI_Finalize failed\n")
        return SimpleNamespace(returncode=143)

    monkeypatch.setattr("sfincs_jax.validation.fortran.subprocess.run", _fake_finalize_failure)
    out = run_sfincs_fortran(input_namelist=input_path, exe=exe, workdir=workdir)
    assert out == workdir / "sfincsOutput.h5"

    def _fake_real_failure(cmd, cwd, stdout, stderr, env, check, timeout):
        Path(cwd, "sfincsOutput.h5").write_bytes(b"incomplete")
        stdout.write("MPI_Finalize failed\n")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr("sfincs_jax.validation.fortran.subprocess.run", _fake_real_failure)
    with pytest.raises(subprocess.CalledProcessError):
        run_sfincs_fortran(input_namelist=input_path, exe=exe, workdir=workdir)

    monkeypatch.setattr("sfincs_jax.cli.main", lambda: 7)
    sys.modules.pop("sfincs_jax.__main__", None)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("sfincs_jax.__main__", run_name="__main__")
    assert excinfo.value.code == 7


def test_scan_helpers_and_run_er_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nml = Namelist(
        groups={"geometryparameters": {"INPUTRADIALCOORDINATEFORGRADIENTS": 2}},
        indexed={},
        source_path=None,
        source_text=None,
    )
    assert _er_scan_var_name(nml=nml) == "dPhiHatdrHat"
    assert linspace_including_endpoints(-2.0, 2.0, 5).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        linspace_including_endpoints(0.0, 1.0, 1)

    patched = _patch_scalar_in_group(
        txt="&physicsParameters\n  Er = 1.0\n/\n",
        group="physicsParameters",
        key="Er",
        value=3.5,
    )
    assert "Er = 3.5" in patched

    scan_input = tmp_path / "scan_input.namelist"
    scan_input.write_text("!ss dPhiHatdpsiNMin = -1\n&general\n/\n")
    assert _infer_var_name_from_scan_input(scan_input) == "dPhiHatdpsiN"

    template = tmp_path / "input.namelist"
    template.write_text(
        "&geometryParameters\n"
        "  inputRadialCoordinateForGradients = 4\n"
        "/\n"
        "&physicsParameters\n"
        "  Er = 0.0\n"
        "/\n"
    )

    calls: list[tuple[Path, Path, Path, str]] = []

    def _fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        assert overwrite is False
        assert input_namelist.exists()

    def _fake_write(**kwargs):
        calls.append(
            (
                Path(kwargs["input_namelist"]),
                Path(kwargs["output_path"]),
                Path(kwargs["solver_trace_path"]),
                str(kwargs["solve_method"]),
            )
        )
        Path(kwargs["output_path"]).write_bytes(b"")

    monkeypatch.setattr("sfincs_jax.scans.localize_equilibrium_file_in_place", _fake_localize)
    monkeypatch.setattr("sfincs_jax.scans.write_sfincs_jax_output_h5", _fake_write)
    emits: list[tuple[int, str]] = []

    result = run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.5, -0.25, 1.5],
        compute_solution=True,
        solve_method="host_structured_csr",
        jobs=1,
        emit=lambda level, msg: emits.append((level, msg)),
    )
    assert result.variable == "Er"
    assert result.values == (1.5, 0.5, -0.25)
    assert len(result.outputs) == 3
    assert all(path.exists() for path in result.outputs)
    assert [p.name for p in result.run_dirs] == ["Er1.5", "Er0.5", "Er-0.25"]
    assert len(calls) == 3
    assert all(trace.name == "sfincsOutput.solver_trace.json" for _, _, trace, _ in calls)
    assert {solve_method for _, _, _, solve_method in calls} == {"host_structured_csr"}
    assert any("ETA becomes available after the first completed point" in msg for _, msg in emits)
    assert any("scan-er: progress 3/3" in msg and "est_remaining=" in msg for _, msg in emits)

    calls.clear()
    emits.clear()
    result_resume = run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.5, -0.25, 1.5],
        compute_solution=True,
        jobs=1,
        skip_existing=True,
        emit=lambda level, msg: emits.append((level, msg)),
    )
    assert result_resume.values == (1.5, 0.5, -0.25)
    assert len(calls) == 0
    assert any("reused existing output" in msg for _, msg in emits)

    missing_output = (tmp_path / "scan" / "Er0.5" / "sfincsOutput.h5")
    missing_output.unlink()
    emits.clear()
    result_partial_resume = run_er_scan(
        input_namelist=template,
        out_dir=tmp_path / "scan",
        values=[0.5, -0.25, 1.5],
        compute_solution=True,
        jobs=1,
        skip_existing=True,
        emit=lambda level, msg: emits.append((level, msg)),
    )
    assert result_partial_resume.values == (1.5, 0.5, -0.25)
    assert len(calls) == 1
    assert calls[0][1] == missing_output
    assert calls[0][2] == missing_output.with_name("sfincsOutput.solver_trace.json")
    assert missing_output.exists()
    assert any("scan-er: progress 3/3" in msg for _, msg in emits)


def test_transport_parallel_worker_writes_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "worker.npz"
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    payload = {
        "input_path": str(input_path),
        "which_rhs_values": [3, 1],
        "tol": 1e-9,
    }
    payload_path.write_text(json.dumps(payload))

    monkeypatch.setattr("sfincs_jax.problems.transport_matrix.parallel.worker.read_sfincs_input", lambda _p: object())
    monkeypatch.setattr(
        "sfincs_jax.problems.transport_matrix.parallel.worker.solve_v3_transport_matrix_linear_gmres",
        lambda **_kwargs: SimpleNamespace(
            state_vectors_by_rhs={1: np.asarray([1.0, 2.0]), 3: np.asarray([3.0, 4.0])},
            residual_norms_by_rhs={1: np.float64(0.25), 3: np.float64(0.75)},
            elapsed_time_s=np.asarray([10.0, 20.0, 30.0]),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["transport_parallel_worker", "--payload", str(payload_path), "--output", str(output_path)],
    )

    assert transport_worker_main() == 0
    data = np.load(output_path)
    np.testing.assert_array_equal(data["which_rhs_values"], np.asarray([3, 1], dtype=np.int32))
    np.testing.assert_allclose(data["state_vectors"], np.asarray([[3.0, 4.0], [1.0, 2.0]]))
    np.testing.assert_allclose(data["residual_norms"], np.asarray([0.75, 0.25]))
    np.testing.assert_allclose(data["elapsed_time_s"], np.asarray([30.0, 10.0]))
