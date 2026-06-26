from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sfincs_jax.io import (
    _apply_export_f_maps,
    _as_1d_float,
    _dphi_hat_dpsi_hat_from_er_geometry_scheme4,
    _evaluate_boozer_rzd_and_derivatives,
    _export_f_config,
    _fortran_logical,
    _get_float,
    _get_int,
    _legendre_matrix,
    localize_equilibrium_file_in_place,
    _phi1_fast_explicit_gmres_restart_default,
    _scheme4_radial_constants,
    _select_phi1_use_frozen_linearization,
    _select_phi1_newton_linear_solve_method,
    _select_rhsmode1_linear_solve_method,
    _set_input_radial_coordinate_wish,
    _should_precompile_v3_full_system,
    read_sfincs_h5,
    write_sfincs_h5,
)
from sfincs_jax.namelist import Namelist, read_sfincs_input
from sfincs_jax.outputs.cache import output_cache_dir, output_cache_path
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist


def test_output_cache_dir_prefers_xdg_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache_dir = output_cache_dir()
    assert cache_dir == tmp_path / "xdg" / "sfincs_jax" / "output_cache"
    assert cache_dir.is_dir()


def test_output_cache_path_is_stable_and_key_sensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_DIR", str(tmp_path / "cache"))
    path1 = output_cache_path(("a", 1))
    path2 = output_cache_path(("a", 1))
    path3 = output_cache_path(("a", 2))
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


def test_geometry_scheme4_radial_helpers_match_v3_conventions() -> None:
    psi_a_hat, a_hat = _scheme4_radial_constants()
    assert psi_a_hat == pytest.approx(-0.384935)
    assert a_hat == pytest.approx(0.5109)

    psi_hat, psi_n, r_hat, r_n = _set_input_radial_coordinate_wish(
        input_radial_coordinate=3,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        psi_hat_wish_in=0.0,
        psi_n_wish_in=0.0,
        r_hat_wish_in=0.0,
        r_n_wish_in=0.5,
    )
    assert psi_hat == pytest.approx(0.25 * psi_a_hat)
    assert psi_n == pytest.approx(0.25)
    assert r_hat == pytest.approx(0.5 * a_hat)
    assert r_n == pytest.approx(0.5)

    dphi = _dphi_hat_dpsi_hat_from_er_geometry_scheme4(2.0)
    expected = a_hat / (2.0 * psi_a_hat * np.sqrt(0.25)) * (-2.0)
    assert dphi == pytest.approx(expected)

    assert _fortran_logical(True) == np.int32(1)
    assert _fortran_logical(False) == np.int32(-1)
    with pytest.raises(ValueError, match="Invalid inputRadialCoordinate"):
        _set_input_radial_coordinate_wish(
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
        _evaluate_boozer_rzd_and_derivatives(
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
