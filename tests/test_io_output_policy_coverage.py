from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sfincs_jax.input_compat import (
    dphi_hat_dpsi_hat_from_er_geometry_scheme4,
    scheme4_radial_constants,
    set_input_radial_coordinate_wish,
)
from sfincs_jax.geometry import boozer as geometry_boozer
from sfincs_jax.geometry import vmec_wout as geometry_vmec_wout
from sfincs_jax.geometry.boozer import evaluate_boozer_rzd_and_derivatives
from sfincs_jax.io import (
    _apply_export_f_maps,
    _as_1d_float,
    _export_f_config,
    _fortran_logical,
    _get_float,
    _get_int,
    _legendre_matrix,
    _output_geom_cache_key,
    localize_equilibrium_file_in_place,
    _phi1_fast_explicit_gmres_restart_default,
    _select_phi1_use_frozen_linearization,
    _select_phi1_newton_linear_solve_method,
    _select_rhsmode1_linear_solve_method,
    _should_precompile_v3_full_system,
    read_sfincs_h5,
    _resolve_equilibrium_file_from_namelist,
    sfincs_jax_output_dict,
    write_sfincs_h5,
)
from sfincs_jax.geometry.boozer import (
    BoozerBCHeader,
    BoozerBCSurface,
    read_boozer_bc_bracketing_surfaces,
    read_boozer_bc_header,
    selected_r_n_from_bc,
)
from sfincs_jax.namelist import Namelist, read_sfincs_input
from sfincs_jax.outputs import rhsmode1 as rhsmode1_output
from sfincs_jax.outputs import writer as output_writer
from sfincs_jax.outputs.formats import (
    output_cache_dir,
    output_cache_path,
    output_geom_cache_key,
    write_sfincs_netcdf,
    write_sfincs_npz,
)
from sfincs_jax.outputs.rhsmode1 import (
    RHSMode1SolveMethodSelectionContext,
    _maybe_align_pas_no_phi1_flow_diagnostics_to_fortran,
    _maybe_apply_constraint0_fortran_gauge,
    _maybe_apply_pas_no_phi1_output_scale,
    select_rhsmode1_solve_method,
    write_rhsmode1_classical_fluxes_to_data,
    write_rhsmode1_core_diagnostics_to_data,
    write_rhsmode1_electric_drift_diagnostics_to_data,
    write_rhsmode1_flux_coordinate_variants_to_data,
    write_rhsmode1_ntv_diagnostics_to_data,
    write_rhsmode1_phi1_diagnostics_to_data,
)
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.solvers.diagnostics import read_solver_trace_json


def test_output_cache_dir_prefers_xdg_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache_dir = output_cache_dir()
    assert cache_dir == tmp_path / "xdg" / "sfincs_jax" / "output_cache"
    assert cache_dir.is_dir()


def _minimal_writer_input(tmp_path: Path, *, rhs_mode: int = 1, geometry_scheme: int = 5) -> Path:
    input_path = tmp_path / f"writer_rhs{rhs_mode}_geom{geometry_scheme}.namelist"
    input_path.write_text(
        "&general\n"
        f"  RHSMode = {int(rhs_mode)}\n"
        "/\n"
        "&geometryParameters\n"
        f"  geometryScheme = {int(geometry_scheme)}\n"
        "  equilibriumFile = 'missing_equilibrium.nc'\n"
        "/\n"
        "&speciesParameters\n"
        "  Zs = 1.0\n"
        "/\n"
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "/\n"
        "&resolutionParameters\n"
        "  solverTolerance = 1.0e-7\n"
        "/\n"
        "&otherNumericalParameters\n"
        "/\n"
        "&preconditionerOptions\n"
        "/\n",
        encoding="utf-8",
    )
    return input_path


def _minimal_writer_grids() -> SimpleNamespace:
    return SimpleNamespace(
        theta=np.asarray([0.0, np.pi], dtype=np.float64),
        zeta=np.asarray([0.0, np.pi], dtype=np.float64),
        x=np.asarray([0.25, 0.75], dtype=np.float64),
        n_xi=3,
        n_l=3,
        n_xi_for_x=np.asarray([3, 2], dtype=np.int32),
    )


def test_write_output_geometry_only_trace_return_results_and_wout_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geometry-only writer orchestration should be cheap and metadata-complete."""

    input_path = _minimal_writer_input(tmp_path, rhs_mode=1, geometry_scheme=5)
    wout_path = tmp_path / "owned_wout.nc"
    wout_path.write_bytes(b"owned vmec fixture")
    trace_path = tmp_path / "solver_trace.json"
    output_path = tmp_path / "sfincsOutput.npz"
    captured: dict[str, object] = {}

    monkeypatch.setattr(output_writer, "grids_from_namelist", lambda _nml: _minimal_writer_grids())
    monkeypatch.setattr(output_writer, "geometry_from_namelist", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(output_writer, "_export_f_config", lambda **_kwargs: None)

    def _fake_output_dict(*, nml, grids, geom, export_cfg):
        captured["equilibrium_override"] = nml.group("geometryParameters")["EQUILIBRIUMFILE"]
        captured["grid_shape"] = (int(grids.theta.size), int(grids.zeta.size), int(grids.x.size))
        captured["geom"] = geom
        captured["export_cfg"] = export_cfg
        return {
            "RHSMode": np.asarray(1, dtype=np.int32),
            "geometryScheme": np.asarray(5, dtype=np.int32),
            "constraintScheme": np.asarray(1, dtype=np.int32),
            "psiAHat": np.asarray(1.0, dtype=np.float64),
            "aHat": np.asarray(1.0, dtype=np.float64),
        }

    def _fake_write_output_file(*, path, data, fortran_layout, overwrite):
        captured["write_path"] = Path(path)
        captured["write_data"] = dict(data)
        captured["fortran_layout"] = bool(fortran_layout)
        captured["overwrite"] = bool(overwrite)

    monkeypatch.setattr(output_writer, "sfincs_jax_output_dict", _fake_output_dict)
    monkeypatch.setattr(output_writer, "write_sfincs_output_file", _fake_write_output_file)

    resolved, data = output_writer.write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=output_path,
        wout_path=wout_path,
        verbose=False,
        solver_trace_path=trace_path,
        return_results=True,
    )

    assert resolved == output_path.resolve()
    assert int(np.asarray(data["RHSMode"]).reshape(())) == 1
    assert int(np.asarray(captured["write_data"]["RHSMode"]).reshape(())) == 1
    assert captured["write_path"] == output_path
    assert captured["fortran_layout"] is True
    assert captured["overwrite"] is True
    assert captured["grid_shape"] == (2, 2, 2)
    assert captured["equilibrium_override"] == str(wout_path)
    assert str(wout_path) in str(data["input.namelist"])

    trace = read_solver_trace_json(trace_path)
    assert trace.selected_path == "geometry_only"
    assert trace.rhs_mode == 1
    assert trace.geometry_scheme == 5
    assert trace.metadata["output_format"] == "npz"
    assert trace.metadata["compute_solution"] is False
    assert trace.metadata["compute_transport_matrix"] is False


def test_write_output_transport_streaming_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport streaming should avoid the full in-memory diagnostics writer."""

    input_path = _minimal_writer_input(tmp_path, rhs_mode=2, geometry_scheme=4)
    output_path = tmp_path / "sfincsOutput.h5"
    captured: dict[str, object] = {}

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_STREAM_H5", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_STORE_STATE", "old-store")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(output_writer, "grids_from_namelist", lambda _nml: _minimal_writer_grids())
    monkeypatch.setattr(output_writer, "geometry_from_namelist", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(output_writer, "_export_f_config", lambda **_kwargs: None)
    monkeypatch.setattr(
        output_writer,
        "sfincs_jax_output_dict",
        lambda **_kwargs: {
            "RHSMode": np.asarray(2, dtype=np.int32),
            "geometryScheme": np.asarray(4, dtype=np.int32),
            "constraintScheme": np.asarray(1, dtype=np.int32),
            "psiAHat": np.asarray(1.0, dtype=np.float64),
            "aHat": np.asarray(1.0, dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        output_writer,
        "write_sfincs_output_file",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("streaming path should write directly")),
    )

    import sfincs_jax.problems.transport_solve as transport_solve

    def _fake_transport_solve(**kwargs):
        captured["force_store_state"] = kwargs["force_store_state"]
        captured["store_state_during_solve"] = os.environ.get("SFINCS_JAX_TRANSPORT_STORE_STATE")
        captured["stream_diag_during_solve"] = os.environ.get("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS")
        return SimpleNamespace(
            op0=SimpleNamespace(),
            state_vectors_by_rhs={},
            transport_matrix=np.eye(3),
            elapsed_time_s=np.asarray([0.1, 0.2, 0.3]),
        )

    def _fake_stream_h5(**kwargs):
        captured["stream_output_path"] = Path(kwargs["output_path"])
        captured["stream_data"] = kwargs["data"]
        captured["stream_result"] = kwargs["result"]

    monkeypatch.setattr(transport_solve, "solve_v3_transport_matrix_linear_gmres", _fake_transport_solve)
    monkeypatch.setattr(output_writer, "_write_transport_h5_streaming", _fake_stream_h5)

    resolved = output_writer.write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=output_path,
        compute_transport_matrix=True,
        verbose=False,
    )

    assert resolved == output_path.resolve()
    assert captured["force_store_state"] is None
    assert captured["store_state_during_solve"] == "1"
    assert captured["stream_diag_during_solve"] == "0"
    assert captured["stream_output_path"] == output_path
    assert int(np.asarray(captured["stream_data"]["RHSMode"]).reshape(())) == 2
    assert os.environ.get("SFINCS_JAX_TRANSPORT_STORE_STATE") == "old-store"
    assert "SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS" not in os.environ


def test_output_cache_path_is_stable_and_key_sensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_DIR", str(tmp_path / "cache"))
    path1 = output_cache_path(("a", 1))
    path2 = output_cache_path(("a", 1))
    path3 = output_cache_path(("a", 2))
    assert path1 == path2
    assert path1 != path3
    assert path1 is not None and path1.name.startswith("output_geom_")


def test_output_geom_cache_key_direct_tracks_equilibrium_and_grid(tmp_path: Path) -> None:
    eq = tmp_path / "wout_direct.nc"
    eq.write_bytes(b"direct cache key equilibrium")
    nml_path = tmp_path / "input.namelist"
    nml_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 5\n"
        "  equilibriumFile = 'wout_direct.nc'\n"
        "/\n",
        encoding="utf-8",
    )
    nml = read_sfincs_input(nml_path)
    grids = SimpleNamespace(
        theta=np.asarray([0.0, 1.0], dtype=np.float64),
        zeta=np.asarray([0.0, 2.0], dtype=np.float64),
    )

    key = output_geom_cache_key(
        nml=nml,
        grids=grids,
        get_int=_get_int,
        resolve_equilibrium_file=lambda **_kwargs: eq,
    )

    assert key is not None
    assert any(item == 5 for item in key)
    assert any(isinstance(item, tuple) and item[0] == len(eq.read_bytes()) for item in key)


def test_output_geom_cache_key_uses_equilibrium_content_identity(tmp_path: Path) -> None:
    eq1 = tmp_path / "wout_a.nc"
    eq2 = tmp_path / "wout_b.nc"
    eq1.write_bytes(b"same vmec content")
    eq2.write_bytes(b"same vmec content")

    def _read_with_equilibrium(name: str) -> Namelist:
        input_path = tmp_path / f"input_{name}.namelist"
        input_path.write_text(
            "&geometryParameters\n"
            "  geometryScheme = 5\n"
            f"  equilibriumFile = '{name}'\n"
            "/\n",
            encoding="utf-8",
        )
        return read_sfincs_input(input_path)

    grids = SimpleNamespace(
        theta=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        zeta=np.asarray([0.0, 1.0], dtype=np.float64),
    )
    key1 = _output_geom_cache_key(nml=_read_with_equilibrium(eq1.name), grids=grids)
    key2 = _output_geom_cache_key(nml=_read_with_equilibrium(eq2.name), grids=grids)
    assert key1 == key2

    eq2.write_bytes(b"different vmec content")
    key3 = _output_geom_cache_key(nml=_read_with_equilibrium(eq2.name), grids=grids)
    assert key3 != key1


def test_direct_boozer_bc_readers_select_bracketing_and_effective_radius() -> None:
    bc_path = Path("tests/ref/nonStelSym_tiny_geometryScheme12.bc")

    header = read_boozer_bc_header(path=bc_path, geometry_scheme=12)
    header2, old, new = read_boozer_bc_bracketing_surfaces(
        path=bc_path,
        geometry_scheme=12,
        r_n_wish=0.5,
    )

    assert header2 == header
    assert header.n_periods == 1
    assert header.psi_a_hat == pytest.approx(-1.0 / (2.0 * np.pi))
    assert old.r_n == pytest.approx(0.4)
    assert new.r_n == pytest.approx(0.6)
    assert selected_r_n_from_bc(path=bc_path, geometry_scheme=12, r_n_wish=0.45) == pytest.approx(0.4)
    assert selected_r_n_from_bc(path=bc_path, geometry_scheme=12, r_n_wish=0.55) == pytest.approx(0.6)
    assert selected_r_n_from_bc(
        path=bc_path,
        geometry_scheme=12,
        r_n_wish=0.5,
        vmecradial_option=2,
    ) == pytest.approx(0.5)


def test_read_sfincs_h5_handles_nested_datasets_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("sub")
        grp.create_dataset("value", data=np.asarray([[1.0, 2.0]]))
        grp.create_dataset("label", data=np.asarray([b"abc"]))

    out = read_sfincs_h5(path)
    np.testing.assert_allclose(out["sub/value"], np.asarray([[1.0, 2.0]]))
    assert out["sub/label"] == "abc"

    with pytest.raises(FileNotFoundError):
        read_sfincs_h5(tmp_path / "missing.h5")


def test_direct_npz_and_netcdf_writers_preserve_tiny_payloads(tmp_path: Path) -> None:
    payload = {
        "scalar": np.asarray(2.5),
        "matrix": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "flag": np.asarray(True),
        "skipped": None,
    }

    npz_path = tmp_path / "out.npz"
    write_sfincs_npz(path=npz_path, data=payload, fortran_layout=False)
    with np.load(npz_path) as data:
        assert "skipped" not in data.files
        np.testing.assert_allclose(data["scalar"], np.asarray(2.5))
        np.testing.assert_allclose(data["matrix"], payload["matrix"])
        assert bool(data["flag"])
    with pytest.raises(FileExistsError):
        write_sfincs_npz(path=npz_path, data=payload, fortran_layout=False, overwrite=False)

    netcdf4 = pytest.importorskip("netCDF4")
    nc_path = tmp_path / "out.nc"
    write_sfincs_netcdf(path=nc_path, data=payload, fortran_layout=False)
    with netcdf4.Dataset(nc_path) as ds:
        assert ds.getncattr("sfincs_jax_format") == "netcdf"
        assert "skipped" not in ds.variables
        np.testing.assert_allclose(ds.variables["scalar"][...], np.asarray(2.5))
        np.testing.assert_allclose(ds.variables["matrix"][...], payload["matrix"])
        assert int(ds.variables["flag"][...]) == 1
    with pytest.raises(FileExistsError):
        write_sfincs_netcdf(path=nc_path, data=payload, fortran_layout=False, overwrite=False)


def test_write_sfincs_h5_respects_overwrite_guard(tmp_path: Path) -> None:
    path = tmp_path / "out.h5"
    write_sfincs_h5(path=path, data={"a": np.asarray([1.0])}, overwrite=True)
    with pytest.raises(FileExistsError):
        write_sfincs_h5(path=path, data={"a": np.asarray([2.0])}, overwrite=False)


def test_scalar_and_legendre_helpers_cover_defaults_and_errors() -> None:
    group = {"A": [3.5], "B": 7}
    np.testing.assert_allclose(_as_1d_float(group, "A"), np.asarray([3.5]))
    np.testing.assert_allclose(_as_1d_float(group, "MISSING", default=2.0), np.asarray([2.0]))
    with pytest.raises(KeyError):
        _as_1d_float(group, "MISSING")

    assert _get_float({"A": [2.5]}, "A", 0.0) == pytest.approx(2.5)
    assert _get_float({"A": []}, "A", 1.25) == pytest.approx(1.25)
    assert _get_int({"B": [4]}, "B", 0) == 4
    assert _get_int({"B": []}, "B", 6) == 6
    assert _get_int({}, "B", 6) == 6

    xi = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    p = _legendre_matrix(xi, n_l=4)
    np.testing.assert_allclose(p[:, 0], 1.0)
    np.testing.assert_allclose(p[:, 1], xi)
    np.testing.assert_allclose(p[:, 2], 0.5 * (3.0 * xi**2 - 1.0))
    np.testing.assert_allclose(p[:, 3], 0.5 * (5.0 * xi**3 - 3.0 * xi))
    with pytest.raises(ValueError):
        _legendre_matrix(xi, n_l=0)


def test_select_phi1_newton_linear_solve_method_handles_invalid_sparse_direct_min() -> None:
    msgs: list[str] = []
    method = _select_phi1_newton_linear_solve_method(
        active_total_size=25000,
        dense_cutoff=5000,
        default_method="batched",
        fast_explicit=True,
        dense_auto_ok=False,
        dense_auto_backend="cpu",
        env_override="",
        emit=lambda _lvl, msg: msgs.append(str(msg)),
    )
    assert method == "sparse_direct"
    assert any("host sparse-direct Newton step" in msg for msg in msgs)


def test_write_output_solver_policy_helpers_are_fail_closed() -> None:
    messages: list[tuple[int, str]] = []

    assert _should_precompile_v3_full_system(env_value=" yes ")
    assert not _should_precompile_v3_full_system(env_value="maybe")

    selected = _select_rhsmode1_linear_solve_method(
        default_method="AUTO",
        env_override=" sparse_host_lu ",
        emit=lambda level, message: messages.append((level, message)),
    )
    assert selected == "sparse_host_lu"
    assert messages == [(1, "write_sfincs_jax_output_h5: solve method forced by env -> sparse_host_lu")]

    assert (
        _select_rhsmode1_linear_solve_method(
            default_method="dense",
            env_override="not-a-method",
            emit=messages.append,
        )
        == "dense"
    )
    assert _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="sparse_direct",
        env_value="",
    ) is False
    assert _select_phi1_use_frozen_linearization(
        fast_explicit=False,
        solve_method="dense",
        env_value="true",
    ) is True
    assert _select_phi1_use_frozen_linearization(
        fast_explicit=True,
        solve_method="dense",
        env_value="off",
    ) is False


def _rhsmode1_selector_op(*, has_fp: bool, has_pas: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        constraint_scheme=2,
        include_phi1=False,
        n_theta=7,
        n_zeta=5,
        rhs_mode=1,
        total_size=120,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            pas=object() if has_pas else None,
        ),
    )


def _rhsmode1_selector_context(**updates) -> RHSMode1SolveMethodSelectionContext:
    values = {
        "active_total_size": 120,
        "dense_auto_accelerator_fp_window": False,
        "dense_auto_backend": "cpu",
        "dense_auto_ok": True,
        "dense_active_cutoff": 8000,
        "dense_fp_cutoff": 8000,
        "dense_pas_cutoff": 2500,
        "differentiable": False,
        "eparallel_abs": 0.0,
        "er_abs": 0.0,
        "force_krylov": False,
        "include_electric_field_xi": False,
        "include_phi1": False,
        "include_phi1_in_kinetic": False,
        "include_xdot": False,
        "op": _rhsmode1_selector_op(has_fp=True),
        "quasineutrality_option": 1,
        "solve_method": "auto",
        "solve_method_arg_forced": False,
        "solve_method_env": "",
        "use_dkes": False,
        "emit": None,
        "resolve_use_implicit": lambda *, differentiable: bool(differentiable),
        "rhsmode1_host_dense_shortcut_allowed": lambda **_kwargs: False,
    }
    values.update(updates)
    return RHSMode1SolveMethodSelectionContext(**values)


def _disable_rhsmode1_selector_auto_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "rhs1_structured_full_csr_auto_allowed",
        "rhs1_tokamak_pas_er_sparse_pc_auto_allowed",
        "rhs1_tokamak_er_dense_auto_allowed",
        "rhs1_tokamak_pas_noer_sparse_pc_auto_allowed",
        "rhs1_tokamak_fp_er_sparse_pc_auto_allowed",
        "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed",
        "rhs1_fp_3d_xblock_sparse_pc_auto_allowed",
        "rhs1_fp_3d_sparse_pc_auto_allowed",
        "rhs1_constrained_pas_sparse_pc_auto_allowed",
    ):
        monkeypatch.setattr(rhsmode1_output, name, lambda **_kwargs: False)


def test_rhsmode1_selector_promotes_small_fp_to_dense() -> None:
    method = select_rhsmode1_solve_method(_rhsmode1_selector_context())

    assert method == "dense"


def test_rhsmode1_selector_keeps_explicit_env_override() -> None:
    messages: list[str] = []

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            solve_method_env="sparse_pc_gmres",
            emit=lambda _level, message: messages.append(str(message)),
        )
    )

    assert method == "sparse_pc_gmres"
    assert any("forced by env" in message for message in messages)
    assert any("keeping explicit solve_method=sparse_pc_gmres" in message for message in messages)


def test_rhsmode1_selector_force_krylov_wins_over_small_fp_dense() -> None:
    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            force_krylov=True,
            op=_rhsmode1_selector_op(has_fp=False),
        )
    )

    assert method == "incremental"


def test_rhsmode1_selector_uses_tokamak_pas_er_sparse_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_tokamak_pas_er_sparse_pc_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            op=_rhsmode1_selector_op(has_fp=False, has_pas=True),
            active_total_size=50_000,
            dense_fp_cutoff=1,
            er_abs=0.2,
        )
    )

    assert method == "sparse_pc_gmres"


def test_rhsmode1_selector_uses_tokamak_er_dense_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_tokamak_er_dense_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            op=_rhsmode1_selector_op(has_fp=False),
            active_total_size=1024,
            dense_fp_cutoff=1,
            er_abs=0.1,
        )
    )

    assert method == "dense"


def test_rhsmode1_selector_uses_3d_fp_xblock_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_fp_3d_xblock_sparse_pc_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            active_total_size=50_000,
            dense_fp_cutoff=1,
            eparallel_abs=0.0,
        )
    )

    assert method == "xblock_sparse_pc_gmres"


def test_rhsmode1_selector_uses_tokamak_fp_noer_sparse_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            active_total_size=50_000,
            dense_fp_cutoff=1,
            er_abs=0.0,
            use_dkes=False,
        )
    )

    assert method == "xblock_sparse_pc_gmres"


def test_rhsmode1_selector_uses_3d_fp_sparse_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_fp_3d_sparse_pc_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            active_total_size=50_000,
            dense_fp_cutoff=1,
            eparallel_abs=0.0,
        )
    )

    assert method == "sparse_pc_gmres"


def test_rhsmode1_selector_records_host_dense_shortcut_when_accelerator_dense_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    messages: list[str] = []

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            dense_auto_ok=False,
            dense_auto_backend="gpu",
            rhsmode1_host_dense_shortcut_allowed=lambda **_kwargs: True,
            emit=lambda _level, message: messages.append(str(message)),
        )
    )

    assert method == "auto"
    assert any("host dense shortcut on backend=gpu" in message for message in messages)


def test_rhsmode1_selector_reports_dense_auto_skip_when_shortcut_is_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    messages: list[str] = []

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            dense_auto_ok=False,
            dense_auto_backend="gpu",
            rhsmode1_host_dense_shortcut_allowed=lambda **_kwargs: False,
            emit=lambda _level, message: messages.append(str(message)),
        )
    )

    assert method == "auto"
    assert any("skipping dense auto mode on backend=gpu" in message for message in messages)


def test_rhsmode1_selector_uses_bicgstab_for_eparallel_full_fp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            active_total_size=50_000,
            dense_fp_cutoff=1,
            eparallel_abs=0.25,
        )
    )

    assert method == "bicgstab"


def test_rhsmode1_selector_uses_constrained_pas_sparse_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)
    monkeypatch.setattr(
        rhsmode1_output,
        "rhs1_constrained_pas_sparse_pc_auto_allowed",
        lambda **_kwargs: True,
    )

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            op=_rhsmode1_selector_op(has_fp=False, has_pas=True),
            active_total_size=50_000,
            dense_pas_cutoff=1,
        )
    )

    assert method == "sparse_pc_gmres"


def test_rhsmode1_selector_uses_small_pas_incremental_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)

    method = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            op=_rhsmode1_selector_op(has_fp=False, has_pas=True),
            active_total_size=120,
            dense_pas_cutoff=2500,
        )
    )

    assert method == "incremental"


def test_rhsmode1_selector_include_phi1_linear_mode_chooses_dense_or_incremental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_rhsmode1_selector_auto_policy(monkeypatch)

    dense = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            include_phi1=True,
            include_phi1_in_kinetic=False,
            quasineutrality_option=2,
            active_total_size=128,
            dense_active_cutoff=256,
            dense_auto_ok=True,
        )
    )
    incremental = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            include_phi1=True,
            include_phi1_in_kinetic=False,
            quasineutrality_option=2,
            active_total_size=128,
            dense_active_cutoff=256,
            dense_auto_ok=False,
            dense_auto_backend="gpu",
        )
    )
    large = select_rhsmode1_solve_method(
        _rhsmode1_selector_context(
            include_phi1=True,
            include_phi1_in_kinetic=False,
            quasineutrality_option=2,
            active_total_size=1024,
            dense_active_cutoff=256,
            dense_auto_ok=True,
        )
    )

    assert dense == "dense"
    assert incremental == "incremental"
    assert large == "incremental"


def test_constraint0_fortran_gauge_helper_is_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", raising=False)
    x_list = [np.asarray([1.0, 2.0], dtype=np.float64)]

    unchanged = _maybe_apply_constraint0_fortran_gauge(
        x_list=x_list,
        op=SimpleNamespace(constraint_scheme=0),
    )

    assert unchanged is x_list

    monkeypatch.setenv("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "1")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_OUTPUT_H5", "/does/not/exist.h5")
    unchanged_missing_reference = _maybe_apply_constraint0_fortran_gauge(
        x_list=x_list,
        op=SimpleNamespace(constraint_scheme=0),
    )

    assert unchanged_missing_reference is x_list


def test_constraint0_fortran_gauge_reports_malformed_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "bad_sfincsOutput.h5"
    with h5py.File(ref_path, "w") as h5:
        h5.create_dataset("FSADensityPerturbation", data=np.asarray([1.0], dtype=np.float64))
    monkeypatch.setenv("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "1")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_OUTPUT_H5", str(ref_path))
    messages: list[str] = []
    x_list = [np.asarray([1.0, 2.0], dtype=np.float64)]

    unchanged = _maybe_apply_constraint0_fortran_gauge(
        x_list=x_list,
        op=SimpleNamespace(constraint_scheme=0),
        emit=lambda _level, message: messages.append(str(message)),
    )

    assert unchanged is x_list
    assert any("failed to read Fortran output" in message for message in messages)


def test_constraint0_fortran_gauge_applies_reference_moment_shift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "sfincsOutput.h5"
    with h5py.File(ref_path, "w") as h5:
        h5.create_dataset("FSADensityPerturbation", data=np.asarray([[0.10]], dtype=np.float64))
        h5.create_dataset("FSAPressurePerturbation", data=np.asarray([[0.04]], dtype=np.float64))
    monkeypatch.setenv("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "1")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_OUTPUT_H5", str(ref_path))

    f_shape = (1, 3, 1, 1, 1)
    op = SimpleNamespace(
        constraint_scheme=0,
        n_species=1,
        n_x=3,
        f_size=int(np.prod(f_shape)),
        fblock=SimpleNamespace(
            f_shape=f_shape,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1, 1, 1], dtype=np.int32)),
        ),
        x=np.asarray([0.2, 0.7, 1.2], dtype=np.float64),
        x_weights=np.asarray([0.3, 0.4, 0.5], dtype=np.float64),
        theta_weights=np.asarray([1.0], dtype=np.float64),
        zeta_weights=np.asarray([1.0], dtype=np.float64),
        d_hat=np.asarray([[1.0]], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([1.0], dtype=np.float64),
        point_at_x0=False,
    )
    x_full = np.concatenate([np.zeros((op.f_size,), dtype=np.float64), np.asarray([7.0], dtype=np.float64)])
    messages: list[str] = []

    adjusted = _maybe_apply_constraint0_fortran_gauge(
        x_list=[x_full],
        op=op,
        emit=lambda _level, message: messages.append(str(message)),
    )

    assert len(adjusted) == 1
    adjusted_np = np.asarray(adjusted[0])
    assert adjusted_np.shape == x_full.shape
    assert np.all(np.isfinite(adjusted_np))
    assert not np.allclose(adjusted_np[: op.f_size], 0.0)
    np.testing.assert_allclose(adjusted_np[op.f_size :], [7.0])
    assert any("using Fortran reference" in message for message in messages)


def test_pas_no_phi1_output_scale_helper_scales_only_distribution(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_NO_PHI1_OUTPUT_SCALE", "2.5")
    op = SimpleNamespace(
        f_size=3,
        rhs_mode=1,
        fblock=SimpleNamespace(pas=object()),
    )

    scaled = _maybe_apply_pas_no_phi1_output_scale(
        x_list=[np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)],
        op=op,
        include_phi1=False,
    )

    np.testing.assert_allclose(np.asarray(scaled[0]), [2.5, 5.0, 7.5, 4.0])

    unscaled = _maybe_apply_pas_no_phi1_output_scale(
        x_list=[np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)],
        op=op,
        include_phi1=True,
    )
    np.testing.assert_allclose(np.asarray(unscaled[0]), [1.0, 2.0, 3.0, 4.0])


def test_pas_no_phi1_flow_alignment_uses_bounded_fortran_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "sfincsOutput.h5"
    with h5py.File(ref_path, "w") as h5:
        h5.create_dataset("FSABFlow", data=np.asarray([[3.0]], dtype=np.float64))
    monkeypatch.setenv("SFINCS_JAX_ALLOW_FORTRAN_REFERENCE", "1")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_OUTPUT_H5", str(ref_path))
    op = SimpleNamespace(
        rhs_mode=1,
        n_species=1,
        total_size=200000,
        fblock=SimpleNamespace(pas=object(), fp=None),
    )
    nml = SimpleNamespace(
        source_path=None,
        group=lambda name: {"Er": 0.0} if name == "physicsParameters" else {},
    )
    arrays = {
        "FSABFlow": np.asarray([[2.0]], dtype=np.float64),
        "FSABjHat": np.asarray([4.0], dtype=np.float64),
        "densityPerturbation": np.asarray([5.0], dtype=np.float64),
    }

    aligned = _maybe_align_pas_no_phi1_flow_diagnostics_to_fortran(
        arrays=arrays,
        op=op,
        nml=nml,
        include_phi1=False,
    )

    np.testing.assert_allclose(aligned["FSABFlow"], [[3.0]])
    np.testing.assert_allclose(aligned["FSABjHat"], [6.0])
    np.testing.assert_allclose(aligned["densityPerturbation"], [5.0])


def test_write_rhsmode1_flux_coordinate_variants_to_data_scales_coordinates() -> None:
    data: dict[str, np.ndarray] = {}
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    conversion_factors = {
        "ddpsiN2ddpsiHat": 10.0,
        "ddrHat2ddpsiHat": 20.0,
        "ddrN2ddpsiHat": 30.0,
    }

    write_rhsmode1_flux_coordinate_variants_to_data(
        data=data,
        base="particleFlux_vm_psiHat",
        values_sN=values,
        conversion_factors=conversion_factors,
        fortran_h5_layout=lambda array: np.asarray(array, dtype=np.float64),
    )

    np.testing.assert_allclose(data["particleFlux_vm_psiHat"], values)
    np.testing.assert_allclose(data["particleFlux_vm_psiN"], values * 10.0)
    np.testing.assert_allclose(data["particleFlux_vm_rHat"], values * 20.0)
    np.testing.assert_allclose(data["particleFlux_vm_rN"], values * 30.0)


def test_write_rhsmode1_classical_fluxes_to_data_repeats_no_phi1_and_writes_coordinates() -> None:
    op = SimpleNamespace(
        theta_weights=np.asarray([0.4, 0.6], dtype=np.float64),
        zeta_weights=np.asarray([0.25, 0.35, 0.40], dtype=np.float64),
        d_hat=np.full((2, 3), 1.7, dtype=np.float64),
        b_hat=np.asarray([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float64),
    )
    data: dict[str, np.ndarray | float] = {
        "gpsiHatpsiHat": np.full((2, 3), 0.8, dtype=np.float64),
        "VPrimeHat": 1.9,
        "alpha": 0.7,
        "Delta": 0.02,
        "nu_n": 0.1,
        "Zs": np.asarray([1.0, -1.0], dtype=np.float64),
        "mHats": np.asarray([2.0, 1.0 / 1836.0], dtype=np.float64),
        "THats": np.asarray([1.2, 0.9], dtype=np.float64),
        "nHats": np.asarray([1.0, 1.1], dtype=np.float64),
        "dnHatdpsiHat": np.asarray([-0.2, -0.3], dtype=np.float64),
        "dTHatdpsiHat": np.asarray([-0.1, -0.15], dtype=np.float64),
    }
    conversion_factors = {
        "ddpsiN2ddpsiHat": 2.0,
        "ddrHat2ddpsiHat": 3.0,
        "ddrN2ddpsiHat": 4.0,
    }

    def identity_layout(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    particle_flux, heat_flux = write_rhsmode1_classical_fluxes_to_data(
        data=data,
        op=op,
        phi1_list=[],
        n_iter=3,
        conversion_factors=conversion_factors,
        fortran_h5_layout=identity_layout,
    )

    assert particle_flux.shape == (2, 3)
    assert heat_flux.shape == (2, 3)
    np.testing.assert_allclose(particle_flux[:, 0], particle_flux[:, 1])
    np.testing.assert_allclose(data["classicalParticleFlux_psiHat"], particle_flux)
    np.testing.assert_allclose(data["classicalParticleFlux_rN"], particle_flux * 4.0)
    assert np.all(np.isfinite(heat_flux))

    phi1_data = dict(data)
    particle_flux_phi1, heat_flux_phi1 = write_rhsmode1_classical_fluxes_to_data(
        data=phi1_data,
        op=op,
        phi1_list=[
            np.zeros((2, 3), dtype=np.float64),
            np.full((2, 3), 0.05, dtype=np.float64),
        ],
        n_iter=2,
        conversion_factors=conversion_factors,
        fortran_h5_layout=identity_layout,
    )

    assert particle_flux_phi1.shape == (2, 2)
    assert heat_flux_phi1.shape == (2, 2)
    np.testing.assert_allclose(phi1_data["classicalHeatFlux_psiN"], heat_flux_phi1 * 2.0)
    assert np.all(np.isfinite(particle_flux_phi1))


def test_write_rhsmode1_ntv_diagnostics_to_data_zeros_vmec_geometry_scheme() -> None:
    op = SimpleNamespace(
        theta_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        zeta_weights=np.asarray([0.25, 0.75], dtype=np.float64),
        d_hat=np.ones((2, 2), dtype=np.float64),
        x=np.asarray([0.2, 0.8], dtype=np.float64),
        x_weights=np.asarray([0.4, 0.6], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([2.0], dtype=np.float64),
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        f_size=24,
    )
    data: dict[str, np.ndarray | float | int] = {
        "geometryScheme": 5,
        "BHat": np.ones((2, 2), dtype=np.float64),
    }

    write_rhsmode1_ntv_diagnostics_to_data(
        data=data,
        op=op,
        xs=[np.ones(24, dtype=np.float64)],
        x_stack=None,
        n_iter=1,
        fortran_h5_layout=lambda array: np.asarray(array, dtype=np.float64),
    )

    np.testing.assert_allclose(data["NTVBeforeSurfaceIntegral"], np.zeros((2, 2, 1, 1)))
    np.testing.assert_allclose(data["NTV"], np.zeros((1, 1)))


def test_write_rhsmode1_ntv_diagnostics_to_data_recomputes_nonaxisymmetric_l2() -> None:
    f_shape = (1, 2, 3, 2, 2)
    f_delta = np.zeros(f_shape, dtype=np.float64)
    f_delta[:, :, 2, :, :] = 0.25
    op = SimpleNamespace(
        theta_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        zeta_weights=np.asarray([0.25, 0.75], dtype=np.float64),
        d_hat=np.ones((2, 2), dtype=np.float64),
        x=np.asarray([0.2, 0.8], dtype=np.float64),
        x_weights=np.asarray([0.4, 0.6], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([2.0], dtype=np.float64),
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        f_size=int(np.prod(f_shape)),
        fblock=SimpleNamespace(f_shape=f_shape),
    )
    data: dict[str, np.ndarray | float | int] = {
        "geometryScheme": 11,
        "BHat": np.asarray([[1.0, 1.1], [1.2, 1.3]], dtype=np.float64),
        "dBHatdtheta": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        "dBHatdzeta": np.asarray([[0.2, 0.1], [0.4, 0.3]], dtype=np.float64),
        "uHat": np.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64),
        "FSABHat2": 1.4,
        "GHat": 0.2,
        "IHat": 0.1,
        "iota": 0.8,
    }

    write_rhsmode1_ntv_diagnostics_to_data(
        data=data,
        op=op,
        xs=[f_delta.reshape((-1,))],
        x_stack=None,
        n_iter=1,
        fortran_h5_layout=lambda array: np.asarray(array, dtype=np.float64),
    )

    assert data["NTVBeforeSurfaceIntegral"].shape == (2, 2, 1, 1)
    assert data["NTV"].shape == (1, 1)
    assert np.all(np.isfinite(data["NTV"]))
    assert float(np.abs(data["NTV"]).max()) > 0.0


def test_write_rhsmode1_core_diagnostics_to_data_writes_schema_layouts() -> None:
    n_iter, n_species, n_theta, n_zeta, n_x = 2, 2, 3, 4, 5
    diag_arrays: dict[str, np.ndarray] = {}

    for offset, key in enumerate(rhsmode1_output._RHSMODE1_CORE_GRID_MOMENT_KEYS):
        diag_arrays[key] = np.arange(
            n_iter * n_species * n_theta * n_zeta,
            dtype=np.float64,
        ).reshape((n_iter, n_species, n_theta, n_zeta)) + 1000.0 * offset

    for offset, key in enumerate(rhsmode1_output._RHSMODE1_FLUX_SURFACE_AVERAGE_KEYS):
        diag_arrays[key] = (
            np.arange(n_iter * n_species, dtype=np.float64).reshape((n_iter, n_species))
            + 2000.0 * offset
        )

    for offset, key in enumerate(rhsmode1_output._RHSMODE1_VELOCITY_SPACE_KEYS):
        diag_arrays[key] = (
            np.arange(n_iter * n_x * n_species, dtype=np.float64).reshape((n_iter, n_x, n_species))
            + 3000.0 * offset
        )

    diag_arrays["sources"] = np.arange(
        n_iter * n_x * n_species,
        dtype=np.float64,
    ).reshape((n_iter, n_x, n_species)) + 4000.0
    diag_arrays["jHat"] = np.arange(
        n_iter * n_theta * n_zeta,
        dtype=np.float64,
    ).reshape((n_iter, n_theta, n_zeta)) + 5000.0

    for offset, key in enumerate(rhsmode1_output._RHSMODE1_CURRENT_KEYS):
        diag_arrays[key] = np.arange(n_iter, dtype=np.float64) + 6000.0 + 10.0 * offset

    for offset, key in enumerate(rhsmode1_output._RHSMODE1_VM_FLUX_KEYS):
        diag_arrays[key] = (
            np.arange(n_iter * n_species, dtype=np.float64).reshape((n_iter, n_species))
            + 7000.0 * offset
        )

    data: dict[str, np.ndarray] = {}
    conversion_factors = {
        "ddpsiN2ddpsiHat": 2.0,
        "ddrHat2ddpsiHat": 3.0,
        "ddrN2ddpsiHat": 4.0,
    }
    def identity_layout(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    write_rhsmode1_core_diagnostics_to_data(
        data=data,
        diag_arrays=diag_arrays,
        conversion_factors=conversion_factors,
        fortran_h5_layout=identity_layout,
    )

    np.testing.assert_allclose(
        data["densityPerturbation"],
        np.transpose(diag_arrays["densityPerturbation"], (3, 2, 1, 0)),
    )
    np.testing.assert_allclose(
        data["FSABFlow"],
        np.transpose(diag_arrays["FSABFlow"], (1, 0)),
    )
    np.testing.assert_allclose(
        data["jHat"],
        np.transpose(diag_arrays["jHat"], (2, 1, 0)),
    )
    np.testing.assert_allclose(
        data["sources"],
        np.transpose(diag_arrays["sources"], (1, 2, 0)),
    )
    np.testing.assert_allclose(data["FSABjHatOverRootFSAB2"], diag_arrays["FSABjHatOverRootFSAB2"])
    np.testing.assert_allclose(
        data["heatFlux_vm_psiHat"],
        np.transpose(diag_arrays["heatFlux_vm_psiHat"], (1, 0)),
    )
    np.testing.assert_allclose(
        data["heatFlux_vm_rN"],
        np.transpose(diag_arrays["heatFlux_vm_psiHat"], (1, 0)) * 4.0,
    )


def test_write_rhsmode1_phi1_diagnostics_to_data_writes_fields_and_qn_debug() -> None:
    phi1_list = [
        np.arange(6, dtype=np.float64).reshape((2, 3)),
        np.arange(6, 12, dtype=np.float64).reshape((2, 3)),
    ]
    dtheta_list = [array + 100.0 for array in phi1_list]
    dzeta_list = [array + 200.0 for array in phi1_list]
    qn_from_f_list = [array + 300.0 for array in phi1_list]
    qn_nonlin_list = [array + 400.0 for array in phi1_list]
    qn_diag_list = [array + 500.0 for array in phi1_list]
    data: dict[str, np.ndarray] = {}

    def identity_layout(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    write_rhsmode1_phi1_diagnostics_to_data(
        data=data,
        phi1_list=phi1_list,
        dphi1_dtheta_list=dtheta_list,
        dphi1_dzeta_list=dzeta_list,
        lambda_list=[1.25, 2.5],
        qn_from_f_list=qn_from_f_list,
        qn_nonlin_list=qn_nonlin_list,
        qn_diag_list=qn_diag_list,
        write_qn_debug=True,
        fortran_h5_layout=identity_layout,
    )

    expected_phi1 = np.stack([np.transpose(array, (1, 0)) for array in phi1_list], axis=-1)
    np.testing.assert_allclose(data["Phi1Hat"], expected_phi1)
    np.testing.assert_allclose(
        data["dPhi1Hatdtheta"],
        np.stack([np.transpose(array, (1, 0)) for array in dtheta_list], axis=-1),
    )
    np.testing.assert_allclose(
        data["QN_diag"],
        np.stack([np.transpose(array, (1, 0)) for array in qn_diag_list], axis=-1),
    )
    np.testing.assert_allclose(data["lambda"], [1.25, 2.5])


def test_write_rhsmode1_electric_drift_diagnostics_to_data_writes_derived_fluxes() -> None:
    n_species, n_theta, n_zeta, n_iter = 2, 2, 3, 2
    before = [
        np.arange(n_species * n_theta * n_zeta, dtype=np.float64).reshape((n_species, n_theta, n_zeta)),
        np.arange(n_species * n_theta * n_zeta, 2 * n_species * n_theta * n_zeta, dtype=np.float64).reshape(
            (n_species, n_theta, n_zeta)
        ),
    ]
    flux_series = [
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([3.0, 4.0], dtype=np.float64),
    ]
    data: dict[str, np.ndarray] = {}
    for flux in ("particleFlux", "heatFlux", "momentumFlux"):
        data[f"{flux}_vm_psiHat"] = np.ones((n_species, n_iter), dtype=np.float64)

    conversion_factors = {
        "ddpsiN2ddpsiHat": 2.0,
        "ddrHat2ddpsiHat": 3.0,
        "ddrN2ddpsiHat": 4.0,
    }

    def identity_layout(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    write_rhsmode1_electric_drift_diagnostics_to_data(
        data=data,
        before_surface_integral_stz={
            "particleFluxBeforeSurfaceIntegral_vE": before,
            "NTVBeforeSurfaceIntegral": before,
        },
        fluxes_s={
            "particleFlux_vE0_psiHat": flux_series,
            "particleFlux_vE_psiHat": [value + 10.0 for value in flux_series],
            "heatFlux_vE0_psiHat": flux_series,
            "heatFlux_vE_psiHat": [value + 20.0 for value in flux_series],
            "momentumFlux_vE0_psiHat": flux_series,
            "momentumFlux_vE_psiHat": [value + 30.0 for value in flux_series],
        },
        ntv_list=flux_series,
        conversion_factors=conversion_factors,
        fortran_h5_layout=identity_layout,
    )

    expected_before = np.stack([np.transpose(array, (2, 1, 0)) for array in before], axis=-1)
    expected_vE0 = np.stack(flux_series, axis=-1)
    np.testing.assert_allclose(data["particleFluxBeforeSurfaceIntegral_vE"], expected_before)
    np.testing.assert_allclose(data["particleFlux_vE0_psiHat"], expected_vE0)
    np.testing.assert_allclose(data["particleFlux_vE0_rN"], expected_vE0 * 4.0)
    np.testing.assert_allclose(data["NTV"], expected_vE0)
    np.testing.assert_allclose(data["particleFlux_vd1_psiHat"], np.ones((n_species, n_iter)) + expected_vE0)
    np.testing.assert_allclose(
        data["heatFlux_withoutPhi1_psiHat"],
        np.ones((n_species, n_iter)) + (5.0 / 3.0) * expected_vE0,
    )


def test_select_phi1_newton_linear_solve_method_env_override_wins() -> None:
    method = _select_phi1_newton_linear_solve_method(
        active_total_size=25000,
        dense_cutoff=5000,
        default_method="incremental",
        fast_explicit=True,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="batched",
        emit=None,
    )
    assert method == "batched"


def test_select_phi1_newton_linear_solve_method_prefers_incremental_when_sparse_direct_not_allowed() -> None:
    messages: list[str] = []
    method = _select_phi1_newton_linear_solve_method(
        active_total_size=25000,
        dense_cutoff=5000,
        default_method="batched",
        fast_explicit=True,
        dense_auto_ok=False,
        dense_auto_backend="tpu",
        env_override="",
        emit=lambda _level, message: messages.append(str(message)),
    )

    assert method == "incremental"
    assert any("preferring incremental Newton step" in message for message in messages)


def test_select_phi1_newton_linear_solve_method_dense_auto_branches() -> None:
    dense_messages: list[str] = []
    dense_method = _select_phi1_newton_linear_solve_method(
        active_total_size=128,
        dense_cutoff=512,
        default_method="incremental",
        fast_explicit=False,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="",
        emit=lambda _level, message: dense_messages.append(str(message)),
    )
    assert dense_method == "dense"
    assert any("using dense Newton step" in message for message in dense_messages)

    skip_messages: list[str] = []
    skipped_method = _select_phi1_newton_linear_solve_method(
        active_total_size=128,
        dense_cutoff=512,
        default_method="incremental",
        fast_explicit=False,
        dense_auto_ok=False,
        dense_auto_backend="gpu",
        env_override="",
        emit=lambda _level, message: skip_messages.append(str(message)),
    )
    assert skipped_method == "incremental"
    assert any("skipping dense auto mode" in message for message in skip_messages)


def test_select_phi1_newton_linear_solve_method_sparse_direct_env_override() -> None:
    method = _select_phi1_newton_linear_solve_method(
        active_total_size=128,
        dense_cutoff=512,
        default_method="dense",
        fast_explicit=False,
        dense_auto_ok=True,
        dense_auto_backend="cpu",
        env_override="sparse_direct",
        emit=None,
    )

    assert method == "sparse_direct"


def test_phi1_fast_explicit_gmres_restart_default_targets_production_size() -> None:
    assert _phi1_fast_explicit_gmres_restart_default(7999) == 80
    assert _phi1_fast_explicit_gmres_restart_default(8000) == 120
    assert _phi1_fast_explicit_gmres_restart_default(12753) == 120
    assert _phi1_fast_explicit_gmres_restart_default(25000) == 120


def test_phi1_history_alignment_preserves_accepted_iterates() -> None:
    from sfincs_jax.outputs.writer import _align_phi1_history_for_output

    history = [np.asarray([1.0]), np.asarray([2.0])]
    aligned = _align_phi1_history_for_output(
        history=history,
        result_x=np.asarray([3.0]),
        x0_state=np.asarray([0.0]),
        use_frozen_linearization=True,
        min_iters=0,
        n_newton=2,
    )
    np.testing.assert_allclose(aligned[0], np.asarray([1.0]))
    np.testing.assert_allclose(aligned[1], np.asarray([2.0]))

    padded = _align_phi1_history_for_output(
        history=[],
        result_x=np.asarray([3.0]),
        x0_state=None,
        use_frozen_linearization=False,
        min_iters=3,
        n_newton=4,
    )
    assert len(padded) == 4
    for item in padded:
        np.testing.assert_allclose(item, np.asarray([3.0]))


def test_phi1_history_alignment_trims_frozen_initial_guess_before_output() -> None:
    from sfincs_jax.outputs.writer import _align_phi1_history_for_output

    aligned = _align_phi1_history_for_output(
        history=[np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])],
        result_x=np.asarray([4.0]),
        x0_state=np.asarray([0.0]),
        use_frozen_linearization=True,
        min_iters=0,
        n_newton=2,
    )

    assert len(aligned) == 2
    np.testing.assert_allclose(aligned[0], np.asarray([2.0]))
    np.testing.assert_allclose(aligned[1], np.asarray([3.0]))


def test_geometry_scheme4_radial_helpers_match_v3_conventions() -> None:
    psi_a_hat, a_hat = scheme4_radial_constants()
    assert psi_a_hat == pytest.approx(-0.384935)
    assert a_hat == pytest.approx(0.5109)

    target_psi_n = 0.25
    target_r_n = np.sqrt(target_psi_n)
    target_r_hat = target_r_n * a_hat
    input_cases = {
        # These four modes mirror v3 radialCoordinates.F90: psiHat, psiN,
        # rHat, and rN are equivalent ways to name the same flux surface.
        0: {"psi_hat_wish_in": target_psi_n * psi_a_hat},
        1: {"psi_n_wish_in": target_psi_n},
        2: {"r_hat_wish_in": target_r_hat},
        3: {"r_n_wish_in": target_r_n},
    }
    for input_radial_coordinate, overrides in input_cases.items():
        base = {
            "psi_hat_wish_in": 0.0,
            "psi_n_wish_in": 0.0,
            "r_hat_wish_in": 0.0,
            "r_n_wish_in": 0.0,
        }
        base.update(overrides)
        psi_hat, psi_n, r_hat, r_n = set_input_radial_coordinate_wish(
            input_radial_coordinate=input_radial_coordinate,
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
            **base,
        )
        assert psi_hat == pytest.approx(target_psi_n * psi_a_hat)
        assert psi_n == pytest.approx(target_psi_n)
        assert r_hat == pytest.approx(target_r_hat)
        assert r_n == pytest.approx(target_r_n)

    dphi = dphi_hat_dpsi_hat_from_er_geometry_scheme4(2.0)
    expected = a_hat / (2.0 * psi_a_hat * np.sqrt(0.25)) * (-2.0)
    assert dphi == pytest.approx(expected)

    assert _fortran_logical(True) == np.int32(1)
    assert _fortran_logical(False) == np.int32(-1)
    with pytest.raises(ValueError, match="Invalid inputRadialCoordinate"):
        set_input_radial_coordinate_wish(
            input_radial_coordinate=99,
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
            psi_hat_wish_in=0.0,
            psi_n_wish_in=0.0,
            r_hat_wish_in=0.0,
            r_n_wish_in=0.5,
        )


def test_boozer_fourier_derivative_evaluator_matches_analytic_modes() -> None:
    theta = np.asarray([0.0, np.pi / 2.0], dtype=np.float64)
    zeta = np.asarray([0.0, np.pi / 3.0], dtype=np.float64)
    m = np.asarray([1], dtype=np.int32)
    n = np.asarray([1], dtype=np.int32)
    parity = np.asarray([True])

    r, dr_dtheta, dr_dzeta, z, dz_dtheta, dz_dzeta, dzeta, ddz_dtheta, ddz_dzeta = (
        evaluate_boozer_rzd_and_derivatives(
            theta=theta,
            zeta=zeta,
            n_periods=2,
            m=m,
            n=n,
            parity=parity,
            r0=3.0,
            r_amp=np.asarray([0.25]),
            z_amp=np.asarray([0.5]),
            dz_amp=np.asarray([0.75]),
            dz_scale=2.0,
            chunk=1,
        )
    )
    angle = theta[:, None] - 2.0 * zeta[None, :]
    np.testing.assert_allclose(r, 3.0 + 0.25 * np.cos(angle))
    np.testing.assert_allclose(dr_dtheta, -0.25 * np.sin(angle))
    np.testing.assert_allclose(dr_dzeta, 0.5 * np.sin(angle))
    np.testing.assert_allclose(z, 0.5 * np.sin(angle))
    np.testing.assert_allclose(dz_dtheta, 0.5 * np.cos(angle))
    np.testing.assert_allclose(dz_dzeta, -1.0 * np.cos(angle))
    np.testing.assert_allclose(dzeta, 1.5 * np.sin(angle))
    np.testing.assert_allclose(ddz_dtheta, 1.5 * np.cos(angle))
    np.testing.assert_allclose(ddz_dzeta, -3.0 * np.cos(angle))


def test_boozer_fourier_evaluator_handles_nonstellarator_symmetric_modes() -> None:
    theta = np.linspace(0.0, 0.8 * np.pi, 5, dtype=np.float64)
    zeta = np.linspace(0.0, 0.8 * np.pi, 5, dtype=np.float64)
    m = np.asarray([1], dtype=np.int32)
    n = np.asarray([1], dtype=np.int32)
    parity = np.asarray([False])

    r, dr_dtheta, dr_dzeta, z, dz_dtheta, dz_dzeta, dzeta, ddz_dtheta, ddz_dzeta = (
        evaluate_boozer_rzd_and_derivatives(
            theta=theta,
            zeta=zeta,
            n_periods=3,
            m=m,
            n=n,
            parity=parity,
            r0=2.0,
            r_amp=np.asarray([0.20]),
            z_amp=np.asarray([0.30]),
            dz_amp=np.asarray([0.40]),
            dz_scale=-0.5,
            chunk=1,
        )
    )

    angle = theta[:, None] - 3.0 * zeta[None, :]
    np.testing.assert_allclose(r, 2.0 + 0.20 * np.sin(angle))
    np.testing.assert_allclose(dr_dtheta, 0.20 * np.cos(angle))
    np.testing.assert_allclose(dr_dzeta, -0.60 * np.cos(angle))
    np.testing.assert_allclose(z, 0.30 * np.cos(angle))
    np.testing.assert_allclose(dz_dtheta, -0.30 * np.sin(angle))
    np.testing.assert_allclose(dz_dzeta, 0.90 * np.sin(angle))
    np.testing.assert_allclose(dzeta, -0.20 * np.cos(angle))
    np.testing.assert_allclose(ddz_dtheta, 0.20 * np.sin(angle))
    np.testing.assert_allclose(ddz_dzeta, -0.60 * np.sin(angle))


def test_boozer_fourier_evaluator_drops_sine_nyquist_modes_like_v3() -> None:
    theta = np.asarray([0.0, np.pi], dtype=np.float64)
    zeta = np.asarray([0.0, np.pi], dtype=np.float64)
    r, dr_dtheta, dr_dzeta, z, dz_dtheta, dz_dzeta, dzeta, ddz_dtheta, ddz_dzeta = (
        evaluate_boozer_rzd_and_derivatives(
            theta=theta,
            zeta=zeta,
            n_periods=1,
            m=np.asarray([0, 1], dtype=np.int32),
            n=np.asarray([0, 1], dtype=np.int32),
            parity=np.asarray([False, False]),
            r0=4.0,
            r_amp=np.asarray([10.0, 20.0]),
            z_amp=np.asarray([30.0, 40.0]),
            dz_amp=np.asarray([50.0, 60.0]),
            dz_scale=1.0,
            chunk=1,
        )
    )

    expected_shape = (theta.size, zeta.size)
    np.testing.assert_allclose(r, np.full(expected_shape, 4.0))
    for derivative in (dr_dtheta, dr_dzeta, z, dz_dtheta, dz_dzeta, dzeta, ddz_dtheta, ddz_dzeta):
        np.testing.assert_allclose(derivative, np.zeros(expected_shape))


def test_export_f_config_returns_none_without_export_request() -> None:
    input_path = Path(__file__).parent.parent / "examples" / "getting_started" / "input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    assert _export_f_config(nml=nml, grids=grids, geom=geom) is None


def test_export_f_config_builds_scheme4_mapping_from_real_fixture() -> None:
    input_path = (
        Path(__file__).parent.parent
        / "examples"
        / "upstream"
        / "fortran_v3"
        / "geometryScheme4_2species_PAS_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    cfg = _export_f_config(nml=nml, grids=grids, geom=geom)
    assert cfg is not None
    assert cfg.export_full_f is True
    assert cfg.theta_option == 1
    assert cfg.zeta_option == 1
    assert cfg.x_option == 1
    assert cfg.xi_option == 1
    assert cfg.map_theta.shape[1] == grids.theta.size
    assert cfg.map_zeta.shape[1] == grids.zeta.size
    assert cfg.map_x.shape[1] == grids.x.size
    assert cfg.map_xi.shape[1] == grids.n_xi
    np.testing.assert_allclose(np.sum(cfg.map_theta, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(cfg.map_zeta, axis=1), 1.0, atol=1e-12)


def test_export_f_config_rejects_invalid_options() -> None:
    input_path = (
        Path(__file__).parent.parent
        / "examples"
        / "upstream"
        / "fortran_v3"
        / "geometryScheme4_2species_PAS_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    nml.group("export_f")["EXPORT_F_THETA_OPTION"] = 9
    with pytest.raises(ValueError, match="theta_option"):
        _export_f_config(nml=nml, grids=grids, geom=geom)


def _toy_export_namelist(export_f: dict[str, object]) -> Namelist:
    return Namelist(
        groups={
            "export_f": {
                "EXPORT_FULL_F": True,
                "EXPORT_DELTA_F": False,
                **export_f,
            },
            "othernumericalparameters": {"XGRIDSCHEME": 5, "XGRID_K": 0.0},
        },
        indexed={},
    )


def _toy_export_grid_and_geometry() -> tuple[SimpleNamespace, SimpleNamespace]:
    grids = SimpleNamespace(
        theta=np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float64),
        zeta=np.asarray([0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi], dtype=np.float64),
        x=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
        n_xi=3,
    )
    geom = SimpleNamespace(n_periods=2)
    return grids, geom


def test_export_f_config_option_zero_exports_native_grid() -> None:
    grids, geom = _toy_export_grid_and_geometry()
    cfg = _export_f_config(
        nml=_toy_export_namelist(
            {
                "EXPORT_F_THETA_OPTION": 0,
                "EXPORT_F_ZETA_OPTION": 0,
                "EXPORT_F_X_OPTION": 0,
                "EXPORT_F_XI_OPTION": 0,
            }
        ),
        grids=grids,
        geom=geom,
    )

    assert cfg is not None
    np.testing.assert_allclose(cfg.export_theta, grids.theta)
    np.testing.assert_allclose(cfg.export_zeta, grids.zeta)
    np.testing.assert_allclose(cfg.export_x, grids.x)
    assert cfg.export_xi is None
    np.testing.assert_allclose(cfg.map_theta, np.eye(grids.theta.size))
    np.testing.assert_allclose(cfg.map_zeta, np.eye(grids.zeta.size))
    np.testing.assert_allclose(cfg.map_x, np.eye(grids.x.size))
    np.testing.assert_allclose(cfg.map_xi, np.eye(grids.n_xi))


def test_export_f_config_option_two_snaps_to_nearest_native_points() -> None:
    grids, geom = _toy_export_grid_and_geometry()
    cfg = _export_f_config(
        nml=_toy_export_namelist(
            {
                "EXPORT_F_THETA_OPTION": 2,
                "EXPORT_F_ZETA_OPTION": 2,
                "EXPORT_F_X_OPTION": 2,
                "EXPORT_F_XI_OPTION": 1,
                "EXPORT_F_THETA": [0.1, 3.1],
                "EXPORT_F_ZETA": [0.2, 1.7],
                "EXPORT_F_X": [0.6, 2.4],
                "EXPORT_F_XI": [-1.0, 0.0, 1.0],
            }
        ),
        grids=grids,
        geom=geom,
    )

    assert cfg is not None
    np.testing.assert_allclose(cfg.export_theta, grids.theta[[0, 2]])
    np.testing.assert_allclose(cfg.export_zeta, grids.zeta[[0, 2]])
    np.testing.assert_allclose(cfg.export_x, grids.x[[0, 2]])
    np.testing.assert_allclose(np.sum(cfg.map_theta, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(cfg.map_zeta, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(cfg.map_x, axis=1), 1.0)
    np.testing.assert_allclose(cfg.map_xi[:, 0], 1.0)


def test_export_f_config_option_one_interpolates_all_requested_axes() -> None:
    grids, geom = _toy_export_grid_and_geometry()
    cfg = _export_f_config(
        nml=_toy_export_namelist(
            {
                "EXPORT_F_THETA_OPTION": 1,
                "EXPORT_F_ZETA_OPTION": 1,
                "EXPORT_F_X_OPTION": 1,
                "EXPORT_F_XI_OPTION": 1,
                "EXPORT_F_THETA": [0.25 * np.pi],
                "EXPORT_F_ZETA": [0.125 * np.pi],
                "EXPORT_F_X": [1.0, 2.0],
                "EXPORT_F_XI": [-0.5, 0.5],
            }
        ),
        grids=grids,
        geom=geom,
    )

    assert cfg is not None
    assert cfg.theta_option == 1
    assert cfg.zeta_option == 1
    assert cfg.x_option == 1
    assert cfg.xi_option == 1
    np.testing.assert_allclose(cfg.export_theta, [0.25 * np.pi])
    np.testing.assert_allclose(cfg.export_zeta, [0.125 * np.pi])
    np.testing.assert_allclose(cfg.export_x, [1.0, 2.0])
    np.testing.assert_allclose(cfg.export_xi, [-0.5, 0.5])
    np.testing.assert_allclose(cfg.map_theta, [[0.5, 0.5, 0.0, 0.0]])
    np.testing.assert_allclose(cfg.map_zeta, [[0.5, 0.5, 0.0, 0.0]])
    assert cfg.map_x.shape == (2, grids.x.size)
    assert cfg.map_xi.shape == (2, grids.n_xi)
    assert np.all(np.isfinite(cfg.map_x))
    assert np.all(np.isfinite(cfg.map_xi))


def test_export_f_config_single_zeta_plane_keeps_fortran_axis_contract() -> None:
    grids, geom = _toy_export_grid_and_geometry()
    grids = SimpleNamespace(**{**vars(grids), "zeta": np.asarray([0.0], dtype=np.float64)})
    cfg = _export_f_config(
        nml=_toy_export_namelist(
            {
                "EXPORT_F_THETA_OPTION": 0,
                "EXPORT_F_ZETA_OPTION": 2,
                "EXPORT_F_X_OPTION": 0,
                "EXPORT_F_XI_OPTION": 0,
                "EXPORT_F_ZETA": [0.25, 0.5],
            }
        ),
        grids=grids,
        geom=geom,
    )

    assert cfg is not None
    np.testing.assert_allclose(cfg.export_zeta, [0.0])
    np.testing.assert_allclose(cfg.map_zeta, [[1.0]])
    assert cfg.n_export_zeta == 1


def test_export_f_config_rejects_interpolated_x_for_unsupported_grid_scheme() -> None:
    grids, geom = _toy_export_grid_and_geometry()
    nml = _toy_export_namelist(
        {
            "EXPORT_F_THETA_OPTION": 0,
            "EXPORT_F_ZETA_OPTION": 0,
            "EXPORT_F_X_OPTION": 1,
            "EXPORT_F_XI_OPTION": 0,
            "EXPORT_F_X": [1.0],
        }
    )
    nml.group("otherNumericalParameters")["XGRIDSCHEME"] = 3
    with pytest.raises(NotImplementedError, match="xGridScheme"):
        _export_f_config(
            nml=nml,
            grids=grids,
            geom=geom,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"EXPORT_F_ZETA_OPTION": 9}, "zeta_option"),
        ({"EXPORT_F_X_OPTION": 9}, "x_option"),
        ({"EXPORT_F_XI_OPTION": 9}, "xi_option"),
    ],
)
def test_export_f_config_rejects_invalid_zeta_x_and_xi_options(
    updates: dict[str, object],
    message: str,
) -> None:
    grids, geom = _toy_export_grid_and_geometry()
    base = {
        "EXPORT_F_THETA_OPTION": 0,
        "EXPORT_F_ZETA_OPTION": 0,
        "EXPORT_F_X_OPTION": 0,
        "EXPORT_F_XI_OPTION": 0,
    }
    base.update(updates)
    with pytest.raises(ValueError, match=message):
        _export_f_config(nml=_toy_export_namelist(base), grids=grids, geom=geom)


def test_apply_export_f_maps_identity_preserves_distribution() -> None:
    class _Cfg:
        map_x = np.eye(2)
        map_xi = np.eye(3)
        map_theta = np.eye(2)
        map_zeta = np.eye(2)

    f = np.arange(1 * 2 * 3 * 2 * 2, dtype=np.float64).reshape(1, 2, 3, 2, 2)
    mapped = _apply_export_f_maps(f, _Cfg())
    np.testing.assert_allclose(mapped, f)


def test_apply_export_f_maps_contracts_each_export_axis() -> None:
    class _Cfg:
        map_x = np.asarray([[0.25, 0.75]], dtype=np.float64)
        map_xi = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        map_theta = np.asarray([[0.5, 0.5]], dtype=np.float64)
        map_zeta = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    f = np.arange(1 * 2 * 2 * 2 * 2, dtype=np.float64).reshape(1, 2, 2, 2, 2)
    mapped = _apply_export_f_maps(f, _Cfg())
    expected = np.einsum(
        "dz,ct,bl,ax,sxltz->sabcd",
        _Cfg.map_zeta,
        _Cfg.map_theta,
        _Cfg.map_xi,
        _Cfg.map_x,
        f,
        optimize=True,
    )

    assert mapped.shape == (1, 1, 2, 1, 2)
    np.testing.assert_allclose(mapped, expected)


def test_localize_equilibrium_file_returns_none_without_equilibrium(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&geometryParameters\n  geometryScheme = 1\n/\n")

    assert localize_equilibrium_file_in_place(input_namelist=input_path) is None


def test_localize_equilibrium_file_copies_boozer_alias_and_patches_input(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_dir.mkdir()
    run_dir.mkdir()
    source = source_dir / "toy.bc"
    source.write_text("toy boozer content\n")
    input_path = run_dir / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 10\n"
        "  fort996boozer_file = '../source/toy.bc'\n"
        "/\n"
    )

    localized = localize_equilibrium_file_in_place(input_namelist=input_path)

    assert localized == run_dir / "toy.bc"
    assert localized.read_text() == "toy boozer content\n"
    assert "fort996boozer_file = 'toy.bc'" in input_path.read_text()


def test_localize_equilibrium_file_patches_unquoted_vmec_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_dir.mkdir()
    run_dir.mkdir()
    source = source_dir / "wout_toy.nc"
    source.write_bytes(b"toy vmec netcdf bytes")
    input_path = run_dir / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 5\n"
        "  equilibriumFile = ../source/wout_toy.nc\n"
        "/\n",
        encoding="utf-8",
    )

    localized = localize_equilibrium_file_in_place(input_namelist=input_path)

    assert localized == run_dir / "wout_toy.nc"
    assert localized.read_bytes() == b"toy vmec netcdf bytes"
    assert 'equilibriumFile = "wout_toy.nc"' in input_path.read_text(encoding="utf-8")


def test_localize_equilibrium_file_handles_nonstellarator_symmetric_boozer_alias(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_dir.mkdir()
    run_dir.mkdir()
    source = source_dir / "toy_nonstel.bc"
    source.write_text("non-stellarator-symmetric boozer content\n", encoding="utf-8")
    input_path = run_dir / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 12\n"
        "  JGboozer_file_NonStelSym = '../source/toy_nonstel.bc'\n"
        "/\n",
        encoding="utf-8",
    )

    localized = localize_equilibrium_file_in_place(input_namelist=input_path)

    assert localized == run_dir / "toy_nonstel.bc"
    assert localized.read_text(encoding="utf-8") == "non-stellarator-symmetric boozer content\n"
    assert "JGboozer_file_NonStelSym = 'toy_nonstel.bc'" in input_path.read_text(encoding="utf-8")


def test_resolve_equilibrium_file_contracts_cover_missing_vmec_fallback_and_nonvmec_ascii(
    tmp_path: Path,
) -> None:
    missing_input = tmp_path / "missing_input.namelist"
    missing_input.write_text("&geometryParameters\n  geometryScheme = 5\n/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing geometryParameters.equilibriumFile"):
        _resolve_equilibrium_file_from_namelist(nml=read_sfincs_input(missing_input))

    vmec_ascii = tmp_path / "vmec_ascii.txt"
    vmec_ascii.write_text("ascii vmec placeholder\n", encoding="utf-8")
    vmec_input = tmp_path / "vmec_input.namelist"
    vmec_input.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 5\n"
        '  equilibriumFile = "vmec_ascii.txt"\n'
        "/\n",
        encoding="utf-8",
    )
    assert _resolve_equilibrium_file_from_namelist(nml=read_sfincs_input(vmec_input)) == vmec_ascii.resolve()

    boozer_ascii = tmp_path / "boozer_ascii.txt"
    boozer_netcdf_sibling = tmp_path / "boozer_ascii.nc"
    boozer_ascii.write_text("boozer ascii placeholder\n", encoding="utf-8")
    boozer_netcdf_sibling.write_text("not the requested boozer path\n", encoding="utf-8")
    boozer_input = tmp_path / "boozer_input.namelist"
    boozer_input.write_text(
        "&geometryParameters\n"
        "  geometryScheme = 11\n"
        '  equilibriumFile = "boozer_ascii.txt"\n'
        "/\n",
        encoding="utf-8",
    )
    assert _resolve_equilibrium_file_from_namelist(nml=read_sfincs_input(boozer_input)) == boozer_ascii.resolve()


def test_localize_equilibrium_file_overwrite_flag_controls_existing_local_copy(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    run_dir = tmp_path / "run"
    source_dir.mkdir()
    run_dir.mkdir()
    source = source_dir / "toy.bc"
    source.write_text("fresh source\n", encoding="utf-8")
    local_copy = run_dir / "toy.bc"
    local_copy.write_text("existing local copy\n", encoding="utf-8")
    input_path = run_dir / "input.namelist"
    input_text = (
        "&geometryParameters\n"
        "  geometryScheme = 11\n"
        "  JGboozer_file = '../source/toy.bc'\n"
        "/\n"
    )
    input_path.write_text(input_text, encoding="utf-8")

    localized = localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)

    assert localized == local_copy
    assert local_copy.read_text(encoding="utf-8") == "existing local copy\n"
    assert "JGboozer_file = 'toy.bc'" in input_path.read_text(encoding="utf-8")

    input_path.write_text(input_text, encoding="utf-8")
    localized_overwrite = localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=True)

    assert localized_overwrite == local_copy
    assert local_copy.read_text(encoding="utf-8") == "fresh source\n"
    assert "JGboozer_file = 'toy.bc'" in input_path.read_text(encoding="utf-8")


def test_bc_metric_output_branch_matches_positive_boozer_surface_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the v3 Boozer ``gpsiHatpsiHat`` output path on a tiny analytic surface."""

    header = BoozerBCHeader(n_periods=1, psi_a_hat=0.5, a_hat=1.0, turkin_sign=1)
    old = BoozerBCSurface(
        r_n=0.25,
        iota=0.4,
        g_hat=1.0,
        i_hat=0.1,
        p_prime_hat=0.0,
        b0_over_bbar=1.0,
        r0=1.6,
        m=np.asarray([1], dtype=np.int32),
        n=np.asarray([0], dtype=np.int32),
        parity=np.asarray([True]),
        b_amp=np.asarray([0.0]),
        r_amp=np.asarray([0.10]),
        z_amp=np.asarray([0.12]),
        dz_amp=np.asarray([0.03]),
    )
    new = BoozerBCSurface(
        r_n=0.75,
        iota=0.5,
        g_hat=1.0,
        i_hat=0.1,
        p_prime_hat=0.0,
        b0_over_bbar=1.0,
        r0=1.8,
        m=old.m,
        n=old.n,
        parity=old.parity,
        b_amp=old.b_amp,
        r_amp=np.asarray([0.14]),
        z_amp=np.asarray([0.16]),
        dz_amp=np.asarray([0.04]),
    )

    monkeypatch.setattr(output_writer, "_resolve_equilibrium_file_from_namelist", lambda *, nml: Path("toy.bc"))
    monkeypatch.setattr(
        geometry_boozer,
        "read_boozer_bc_bracketing_surfaces",
        lambda **_kwargs: (header, old, new),
    )

    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64),
        zeta=np.asarray([0.0, np.pi], dtype=np.float64),
    )
    geom = SimpleNamespace(d_hat=np.asarray(0.7, dtype=np.float64))
    gpsipsi = output_writer._gpsipsi_from_bc_file(
        nml=Namelist(groups={}, indexed={}),
        grids=grids,
        geom=geom,
        r_n_wish=0.5,
        vmecradial_option=0,
        geometry_scheme=11,
    )

    assert gpsipsi.shape == (3, 2)
    assert np.all(np.isfinite(gpsipsi))
    assert np.all(gpsipsi > 0.0)


def test_vmec_metric_output_branch_filters_modes_and_produces_finite_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the VMEC ``gpsiHatpsiHat`` output formula without a large wout fixture."""

    w = SimpleNamespace(
        nfp=1,
        ns=3,
        mpol=2,
        ntor=0,
        phi=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        xm=np.asarray([0, 1], dtype=np.int32),
        xn=np.asarray([0, 0], dtype=np.int32),
        xm_nyq=np.asarray([0, 1], dtype=np.int32),
        xn_nyq=np.asarray([0, 0], dtype=np.int32),
        bmnc=np.asarray([[1.0, 1.0, 1.0], [0.05, 0.05, 0.05]], dtype=np.float64),
        rmnc=np.asarray([[1.5, 1.6, 1.7], [0.10, 0.12, 0.14]], dtype=np.float64),
        zmns=np.asarray([[0.0, 0.0, 0.0], [0.08, 0.10, 0.12]], dtype=np.float64),
    )
    interp = SimpleNamespace(
        index_full=(1, 2),
        weight_full=(0.75, 0.25),
        index_half=(1, 2),
        weight_half=(0.75, 0.25),
    )

    monkeypatch.setattr(output_writer, "_resolve_equilibrium_file_from_namelist", lambda *, nml: Path("wout_toy.nc"))
    monkeypatch.setattr(geometry_vmec_wout, "read_vmec_wout", lambda _path: w)
    monkeypatch.setattr(geometry_vmec_wout, "vmec_interpolation", lambda **_kwargs: interp)

    nml = Namelist(
        groups={
            "geometryparameters": {
                "EQUILIBRIUMFILE": "wout_toy.nc",
                "MIN_BMN_TO_LOAD": 0.02,
                "RIPPLESCALE": 0.5,
                "HELICITY_N": 0,
                "HELICITY_L": 0,
                "VMEC_NYQUIST_OPTION": 1,
            }
        },
        indexed={},
    )
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64),
        zeta=np.asarray([0.0, np.pi], dtype=np.float64),
    )

    gpsipsi = output_writer._gpsipsi_from_wout_file(
        nml=nml,
        grids=grids,
        psi_n_wish=0.5,
        vmec_radial_option=0,
    )

    assert gpsipsi.shape == (3, 2)
    assert np.all(np.isfinite(gpsipsi))
    assert np.all(gpsipsi > 0.0)


def _toy_output_grids_and_geometry() -> tuple[SimpleNamespace, SimpleNamespace]:
    shape = (2, 2)
    zeros = np.zeros(shape, dtype=np.float64)
    ones = np.ones(shape, dtype=np.float64)
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi], dtype=np.float64),
        zeta=np.asarray([0.0, np.pi / 2.0], dtype=np.float64),
        theta_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        zeta_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        x=np.asarray([0.25, 0.75], dtype=np.float64),
        n_xi=3,
        n_l=3,
        n_xi_for_x=np.asarray([3, 2], dtype=np.int32),
    )
    geom = SimpleNamespace(
        n_periods=10,
        b0_over_bbar=1.2,
        iota=0.45,
        g_hat=1.1,
        i_hat=0.2,
        d_hat=ones,
        b_hat=1.0 + 0.05 * np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        db_hat_dpsi_hat=zeros,
        db_hat_dtheta=zeros,
        db_hat_dzeta=zeros,
        b_hat_sub_psi=zeros,
        db_hat_sub_psi_dtheta=zeros,
        db_hat_sub_psi_dzeta=zeros,
        b_hat_sub_theta=ones,
        db_hat_sub_theta_dpsi_hat=zeros,
        db_hat_sub_theta_dzeta=zeros,
        b_hat_sub_zeta=0.5 * ones,
        db_hat_sub_zeta_dpsi_hat=zeros,
        db_hat_sub_zeta_dtheta=zeros,
        b_hat_sup_theta=0.25 * ones,
        db_hat_sup_theta_dpsi_hat=zeros,
        db_hat_sup_theta_dzeta=zeros,
        b_hat_sup_zeta=0.125 * ones,
        db_hat_sup_zeta_dpsi_hat=zeros,
        db_hat_sup_zeta_dtheta=zeros,
    )
    return grids, geom


def _minimal_scheme2_output_namelist(
    *,
    physics: dict[str, object] | None = None,
    species: dict[str, object] | None = None,
) -> Namelist:
    return Namelist(
        groups={
            "geometryparameters": {
                "GEOMETRYSCHEME": 2,
                "INPUTRADIALCOORDINATE": 3,
                "RN_WISH": 0.5,
            },
            "physicsparameters": {
                "COLLISIONOPERATOR": 1,
                "INCLUDEPHI1": False,
                **(physics or {}),
            },
            "speciesparameters": {
                "ZS": [1.0, -1.0],
                "MHATS": [2.0, 5.446170214e-4],
                "THATS": [1.0, 1.1],
                "NHATS": [0.9, 0.8],
                **(species or {}),
            },
            "othernumericalparameters": {"XGRIDSCHEME": 5},
            "resolutionparameters": {},
            "preconditioneroptions": {},
            "general": {"RHSMODE": 1},
        },
        indexed={},
    )


def test_output_dict_scheme2_writes_v3_flags_species_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "0")
    monkeypatch.setattr(output_writer, "vprime_hat_jax", lambda *, grids, geom: 4.0)
    monkeypatch.setattr(output_writer, "fsab_hat2_jax", lambda *, grids, geom: 1.25)
    monkeypatch.setattr(output_writer, "u_hat_np", lambda *, grids, geom: np.full((2, 2), 0.3))
    grids, geom = _toy_output_grids_and_geometry()
    nml = Namelist(
        groups={
            "geometryparameters": {
                "GEOMETRYSCHEME": 2,
                "INPUTRADIALCOORDINATE": 3,
                "RN_WISH": 0.5,
            },
            "physicsparameters": {
                "COLLISIONOPERATOR": 1,
                "ER": 2.0,
                "INCLUDEPHI1": True,
                "INCLUDEPHI1INKINETICEQUATION": False,
                "QUASINEUTRALITYOPTION": 2,
            },
            "speciesparameters": {
                "ZS": [1.0, -1.0],
                "MHATS": [2.0, 5.446170214e-4],
                "THATS": [1.2, 0.8],
                "NHATS": [1.0, 0.9],
                "WITHADIABATIC": True,
                "ADIABATICZ": -1.0,
                "ADIABATICNHAT": 0.8,
                "ADIABATICTHAT": 0.7,
            },
            "othernumericalparameters": {"XGRIDSCHEME": 5, "USEITERATIVELINEARSOLVER": False},
            "resolutionparameters": {"SOLVERTOLERANCE": 1.0e-9},
            "preconditioneroptions": {},
            "general": {"RHSMODE": 1},
        },
        indexed={},
    )

    out = sfincs_jax_output_dict(nml=nml, grids=grids, geom=geom)

    assert int(out["geometryScheme"]) == 2
    assert int(out["Nspecies"]) == 2
    assert int(out["constraintScheme"]) == 2
    assert int(out["includePhi1"]) == 1
    assert int(out["includePhi1InKineticEquation"]) == -1
    assert int(out["quasineutralityOption"]) == 2
    assert int(out["withAdiabatic"]) == 1
    assert float(out["adiabaticNHat"]) == pytest.approx(0.8)
    assert int(out["useIterativeLinearSolver"]) == -1
    np.testing.assert_allclose(out["VPrimeHat"], 4.0)
    np.testing.assert_allclose(out["FSABHat2"], 1.25)
    np.testing.assert_allclose(out["gpsiHatpsiHat"], np.zeros((2, 2)))
    np.testing.assert_allclose(out["uHat"], np.full((2, 2), 0.3))
    np.testing.assert_allclose(out["BHat"], geom.b_hat)
    assert out["classicalParticleFluxNoPhi1_psiHat"].shape == (2,)


@pytest.mark.parametrize(
    ("physics_key", "output_key"),
    [
        ("DPHIHATDPSIHAT", "dPhiHatdpsiHat"),
        ("DPHIHATDPSIN", "dPhiHatdpsiN"),
        ("DPHIHATDRHAT", "dPhiHatdrHat"),
        ("DPHIHATDRN", "dPhiHatdrN"),
    ],
)
def test_output_dict_preserves_user_phi_gradient_coordinate(
    physics_key: str,
    output_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "0")
    monkeypatch.setattr(output_writer, "vprime_hat_jax", lambda *, grids, geom: 4.0)
    monkeypatch.setattr(output_writer, "fsab_hat2_jax", lambda *, grids, geom: 1.25)
    monkeypatch.setattr(output_writer, "u_hat_np", lambda *, grids, geom: np.full((2, 2), 0.3))
    grids, geom = _toy_output_grids_and_geometry()

    out = sfincs_jax_output_dict(
        nml=_minimal_scheme2_output_namelist(physics={physics_key: 1.75}),
        grids=grids,
        geom=geom,
    )

    assert float(np.asarray(out[output_key]).reshape(())) == pytest.approx(1.75)


@pytest.mark.parametrize(
    ("species_update", "density_key", "temperature_key"),
    [
        ({"DNHATDPSIHATS": [0.1, 0.2], "DTHATDPSIHATS": [0.3, 0.4]}, "dnHatdpsiHat", "dTHatdpsiHat"),
        ({"DNHATDPSINS": [0.1, 0.2], "DTHATDPSINS": [0.3, 0.4]}, "dnHatdpsiN", "dTHatdpsiN"),
        ({"DNHATDRHATS": [0.1, 0.2], "DTHATDRHATS": [0.3, 0.4]}, "dnHatdrHat", "dTHatdrHat"),
        ({"DNHATDRNS": [0.1, 0.2], "DTHATDRNS": [0.3, 0.4]}, "dnHatdrN", "dTHatdrN"),
    ],
)
def test_output_dict_preserves_user_species_gradient_coordinate(
    species_update: dict[str, object],
    density_key: str,
    temperature_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "0")
    monkeypatch.setattr(output_writer, "vprime_hat_jax", lambda *, grids, geom: 4.0)
    monkeypatch.setattr(output_writer, "fsab_hat2_jax", lambda *, grids, geom: 1.25)
    monkeypatch.setattr(output_writer, "u_hat_np", lambda *, grids, geom: np.full((2, 2), 0.3))
    grids, geom = _toy_output_grids_and_geometry()

    out = sfincs_jax_output_dict(
        nml=_minimal_scheme2_output_namelist(species=species_update),
        grids=grids,
        geom=geom,
    )

    np.testing.assert_allclose(out[density_key], [0.1, 0.2])
    np.testing.assert_allclose(out[temperature_key], [0.3, 0.4])


def test_output_dict_rejects_unsupported_gradient_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "0")
    grids, geom = _toy_output_grids_and_geometry()
    nml = _minimal_scheme2_output_namelist()
    nml.group("geometryParameters")["INPUTRADIALCOORDINATEFORGRADIENTS"] = 99

    with pytest.raises(NotImplementedError, match="inputRadialCoordinateForGradients"):
        sfincs_jax_output_dict(nml=nml, grids=grids, geom=geom)


def test_output_dict_reuses_and_completes_geometry_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "1")
    monkeypatch.setattr(output_writer, "_OUTPUT_GEOM_CACHE", {})
    monkeypatch.setattr(output_writer, "_output_cache_enabled", lambda: True)
    monkeypatch.setattr(output_writer, "_output_geom_cache_key", lambda *, nml, grids: ("toy-output-cache",))

    shape = (2, 2)
    cached_payload = {
        "VPrimeHat": np.asarray(9.0),
        "FSABHat2": np.asarray(8.0),
        "gpsiHatpsiHat": np.full(shape, 0.7),
        "BDotCurlB": np.full(shape, 0.6),
        "diotadpsiHat": np.asarray(0.125),
        "classicalParticleFluxNoPhi1_psiHat": np.asarray([0.1, 0.2]),
        "classicalParticleFluxNoPhi1_psiN": np.asarray([0.3, 0.4]),
        "classicalParticleFluxNoPhi1_rHat": np.asarray([0.5, 0.6]),
        "classicalParticleFluxNoPhi1_rN": np.asarray([0.7, 0.8]),
        "classicalHeatFluxNoPhi1_psiHat": np.asarray([1.1, 1.2]),
        "classicalHeatFluxNoPhi1_psiN": np.asarray([1.3, 1.4]),
        "classicalHeatFluxNoPhi1_rHat": np.asarray([1.5, 1.6]),
        "classicalHeatFluxNoPhi1_rN": np.asarray([1.7, 1.8]),
    }
    saved_payloads: list[dict[str, np.ndarray]] = []
    monkeypatch.setattr(output_writer, "_load_output_cache", lambda key: dict(cached_payload))
    monkeypatch.setattr(
        output_writer,
        "_save_output_cache",
        lambda key, payload: saved_payloads.append(dict(payload)),
    )
    monkeypatch.setattr(
        output_writer,
        "vprime_hat_jax",
        lambda *, grids, geom: (_ for _ in ()).throw(AssertionError("cached VPrimeHat not used")),
    )
    monkeypatch.setattr(
        output_writer,
        "fsab_hat2_jax",
        lambda *, grids, geom: (_ for _ in ()).throw(AssertionError("cached FSABHat2 not used")),
    )
    monkeypatch.setattr(output_writer, "u_hat_np", lambda *, grids, geom: np.full(shape, 0.33))

    grids, geom = _toy_output_grids_and_geometry()
    nml = Namelist(
        groups={
            "geometryparameters": {"GEOMETRYSCHEME": 2, "INPUTRADIALCOORDINATE": 3, "RN_WISH": 0.5},
            "physicsparameters": {"COLLISIONOPERATOR": 1, "ER": 0.0, "INCLUDEPHI1": False},
            "speciesparameters": {"ZS": [1.0, -1.0], "MHATS": [2.0, 5.4e-4], "THATS": [1.0, 1.0], "NHATS": [1.0, 1.0]},
            "othernumericalparameters": {"XGRIDSCHEME": 5},
            "resolutionparameters": {},
            "preconditioneroptions": {},
            "general": {"RHSMODE": 1},
        },
        indexed={},
    )

    out = sfincs_jax_output_dict(nml=nml, grids=grids, geom=geom)

    np.testing.assert_allclose(out["VPrimeHat"], 9.0)
    np.testing.assert_allclose(out["FSABHat2"], 8.0)
    np.testing.assert_allclose(out["gpsiHatpsiHat"], np.full(shape, 0.7))
    np.testing.assert_allclose(out["BDotCurlB"], np.full(shape, 0.6))
    np.testing.assert_allclose(out["diotadpsiHat"], 0.125)
    np.testing.assert_allclose(out["uHat"], np.full(shape, 0.33))
    np.testing.assert_allclose(out["classicalParticleFluxNoPhi1_psiHat"], [0.1, 0.2])
    assert saved_payloads
    np.testing.assert_allclose(saved_payloads[-1]["uHat"], np.full(shape, 0.33))
    assert output_writer._OUTPUT_GEOM_CACHE[("toy-output-cache",)]["uHat"].shape == shape


def test_output_dict_rejects_unsupported_geometry_and_singular_monoenergetic() -> None:
    grids, geom = _toy_output_grids_and_geometry()
    with pytest.raises(NotImplementedError, match="geometryScheme"):
        sfincs_jax_output_dict(
            nml=Namelist(groups={"geometryparameters": {"GEOMETRYSCHEME": 99}}, indexed={}),
            grids=grids,
            geom=geom,
        )

    singular_geom = SimpleNamespace(**vars(geom))
    singular_geom.g_hat = 0.0
    singular_geom.i_hat = 0.0
    mono_nml = Namelist(
        groups={
            "geometryparameters": {"GEOMETRYSCHEME": 2, "INPUTRADIALCOORDINATE": 3, "RN_WISH": 0.5},
            "physicsparameters": {"NUPRIME": 0.1, "ESTAR": 0.2},
            "speciesparameters": {"ZS": [1.0]},
            "othernumericalparameters": {},
            "resolutionparameters": {},
            "preconditioneroptions": {},
            "general": {"RHSMODE": 3},
        },
        indexed={},
    )
    with pytest.raises(ZeroDivisionError, match="GHat"):
        sfincs_jax_output_dict(nml=mono_nml, grids=grids, geom=singular_geom)
