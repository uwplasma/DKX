from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from sfincs_jax.io import (
    _apply_export_f_maps,
    _as_1d_float,
    _export_f_config,
    _get_float,
    _get_int,
    _legendre_matrix,
    _output_cache_dir,
    _output_cache_path,
    _select_phi1_newton_linear_solve_method,
    read_sfincs_h5,
    write_sfincs_h5,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def test_output_cache_dir_prefers_xdg_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache_dir = _output_cache_dir()
    assert cache_dir == tmp_path / "xdg" / "sfincs_jax" / "output_cache"
    assert cache_dir.is_dir()


def test_output_cache_path_is_stable_and_key_sensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_DIR", str(tmp_path / "cache"))
    path1 = _output_cache_path(("a", 1))
    path2 = _output_cache_path(("a", 1))
    path3 = _output_cache_path(("a", 2))
    assert path1 == path2
    assert path1 != path3
    assert path1 is not None and path1.name.startswith("output_geom_")


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
    assert _get_int({"B": [4]}, "B", 0) == 4
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


def test_apply_export_f_maps_identity_preserves_distribution() -> None:
    class _Cfg:
        map_x = np.eye(2)
        map_xi = np.eye(3)
        map_theta = np.eye(2)
        map_zeta = np.eye(2)

    f = np.arange(1 * 2 * 3 * 2 * 2, dtype=np.float64).reshape(1, 2, 3, 2, 2)
    mapped = _apply_export_f_maps(f, _Cfg())
    np.testing.assert_allclose(mapped, f)
