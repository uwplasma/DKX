from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.io import (
    _add_rhsmode1_solver_diagnostics,
    _apply_export_f_maps,
    _as_1d_float,
    _export_f_config,
    _legendre_matrix,
    _raise_for_nonconverged_rhsmode1_output,
    _rhsmode1_result_residual_and_target,
    _should_fail_nonconverged_rhsmode1_output,
    read_sfincs_h5,
    read_sfincs_output_file,
    write_sfincs_h5,
    write_sfincs_output_file,
)
from sfincs_jax.namelist import Namelist


def _minimal_namelist(groups: dict[str, dict]) -> Namelist:
    return Namelist(groups=groups, indexed={}, source_path=None, source_text=None)


def test_write_sfincs_h5_roundtrip_and_overwrite_guard(tmp_path: Path) -> None:
    out = tmp_path / "mini.h5"
    data = {
        "scalar": np.asarray(3.0),
        "vector": np.asarray([1.0, 2.0, 3.0]),
        "matrix": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    }

    write_sfincs_h5(path=out, data=data, fortran_layout=False)
    loaded = read_sfincs_h5(out)

    assert np.asarray(loaded["scalar"]).shape == ()
    np.testing.assert_allclose(loaded["scalar"], data["scalar"])
    np.testing.assert_allclose(loaded["vector"], data["vector"])
    np.testing.assert_allclose(loaded["matrix"], data["matrix"])

    with pytest.raises(FileExistsError):
        write_sfincs_h5(path=out, data=data, overwrite=False)


def test_nonconverged_rhsmode1_production_output_gate(monkeypatch) -> None:
    result = SimpleNamespace(
        residual_norm=np.asarray(1.0e-2),
        rhs=np.asarray([2.0, 0.0], dtype=np.float64),
    )
    residual_norm, residual_target = _rhsmode1_result_residual_and_target(result, solver_tol=1.0e-7)

    assert residual_norm == 1.0e-2
    assert residual_target == 2.0e-7
    assert _should_fail_nonconverged_rhsmode1_output(
        active_total_size=100_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=100_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
        accepted_converged=True,
    )
    with pytest.raises(RuntimeError, match="Refusing to write nonconverged RHSMode=1"):
        _raise_for_nonconverged_rhsmode1_output(
            active_total_size=100_000,
            residual_norm=residual_norm,
            residual_target=residual_target,
            solve_method="incremental",
        )

    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=100,
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    monkeypatch.setenv("SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT", "1")
    assert not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=100_000,
        residual_norm=residual_norm,
        residual_target=residual_target,
    )


def test_rhsmode1_solver_diagnostics_are_output_visible() -> None:
    data: dict[str, object] = {}

    _add_rhsmode1_solver_diagnostics(
        data,
        residual_norm=2.0e-8,
        residual_target=1.0e-7,
        solve_method="sparse_host",
        solver_metadata={
            "accepted_converged": True,
            "acceptance_criterion": "true_residual",
            "iterations": 7,
            "matvecs": 11,
            "setup_s": 0.25,
            "solve_s": 0.75,
            "elapsed_s": 1.0,
            "sparse_pattern_nnz": 1234,
            "sparse_pattern_avg_row_nnz": 5.5,
            "sparse_pattern_max_row_nnz": 9,
            "sparse_pattern_build_s": 0.1,
            "sparse_pc_factor_s": 0.15,
            "sparse_pc_factor_nbytes_estimate": 4567,
            "sparse_pc_factor_nnz_estimate": 321,
        },
    )

    assert data["linearSolverMethod"] == "sparse_host"
    assert float(np.asarray(data["linearSolverResidualNorm"])) == 2.0e-8
    assert float(np.asarray(data["linearSolverResidualTarget"])) == 1.0e-7
    assert int(np.asarray(data["linearSolverConverged"])) == 1
    assert int(np.asarray(data["linearSolverAccepted"])) == 1
    assert data["linearSolverAcceptanceCriterion"] == "true_residual"
    assert int(np.asarray(data["linearSolverIterations"])) == 7
    assert int(np.asarray(data["linearSolverMatvecs"])) == 11
    assert float(np.asarray(data["linearSolverSetupTime"])) == pytest.approx(0.25)
    assert float(np.asarray(data["linearSolverSolveTime"])) == pytest.approx(0.75)
    assert float(np.asarray(data["linearSolverElapsedTime"])) == pytest.approx(1.0)
    assert int(np.asarray(data["linearSolverSparsePatternNnz"])) == 1234
    assert float(np.asarray(data["linearSolverSparsePatternAvgRowNnz"])) == pytest.approx(5.5)
    assert int(np.asarray(data["linearSolverSparsePatternMaxRowNnz"])) == 9
    assert float(np.asarray(data["linearSolverSparsePatternBuildTime"])) == pytest.approx(0.1)
    assert float(np.asarray(data["linearSolverSparsePCFactorTime"])) == pytest.approx(0.15)
    assert int(np.asarray(data["linearSolverSparsePCFactorNbytesEstimate"])) == 4567
    assert int(np.asarray(data["linearSolverSparsePCFactorNnzEstimate"])) == 321
    assert float(np.asarray(data["linearSolverResidualTargetRatio"])) == pytest.approx(0.2)


def test_rhsmode1_solver_diagnostics_label_petsc_compatible_acceptance() -> None:
    data: dict[str, object] = {}

    _add_rhsmode1_solver_diagnostics(
        data,
        residual_norm=1.0e-2,
        residual_target=1.0e-9,
        solve_method="petsc_compat",
        solver_metadata={
            "accepted_converged": True,
            "acceptance_criterion": "petsc_compatible_minimum_norm",
            "least_squares_converged": True,
            "reported_residual_norm": 4.0e-8,
            "iterations": 25,
            "info_code": 1,
        },
    )

    assert int(np.asarray(data["linearSolverConverged"])) == -1
    assert int(np.asarray(data["linearSolverTrueResidualConverged"])) == -1
    assert int(np.asarray(data["linearSolverAccepted"])) == 1
    assert int(np.asarray(data["linearSolverLeastSquaresConverged"])) == 1
    assert data["linearSolverAcceptanceCriterion"] == "petsc_compatible_minimum_norm"
    assert float(np.asarray(data["linearSolverReportedResidualNorm"])) == 4.0e-8
    assert int(np.asarray(data["linearSolverIterations"])) == 25
    assert int(np.asarray(data["linearSolverInfoCode"])) == 1


@pytest.mark.parametrize("suffix", [".npz", ".nc"])
def test_write_sfincs_output_file_roundtrips_npz_and_netcdf(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"mini{suffix}"
    data = {
        "scalar": np.asarray(3.0),
        "vector": np.asarray([1.0, 2.0, 3.0]),
        "matrix with spaces": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "input.namelist": "example = true",
    }

    write_sfincs_output_file(path=out, data=data, fortran_layout=False)
    loaded = read_sfincs_output_file(out)

    assert np.asarray(loaded["scalar"]).shape == ()
    np.testing.assert_allclose(loaded["scalar"], data["scalar"])
    np.testing.assert_allclose(loaded["vector"], data["vector"])
    np.testing.assert_allclose(loaded["matrix with spaces"], data["matrix with spaces"])
    assert str(loaded["input.namelist"]) == "example = true"

    with pytest.raises(FileExistsError):
        write_sfincs_output_file(path=out, data=data, fortran_layout=False, overwrite=False)


def test_write_sfincs_h5_fortran_layout_reverses_axes_for_python_readback(tmp_path: Path) -> None:
    out = tmp_path / "fortran_layout.h5"
    arr = np.arange(24.0).reshape(2, 3, 4)

    write_sfincs_h5(path=out, data={"cube": arr}, fortran_layout=True)
    loaded = read_sfincs_h5(out)

    np.testing.assert_allclose(loaded["cube"], np.transpose(arr, axes=(2, 1, 0)))


def test_as_1d_float_and_legendre_matrix_behave_on_boundary_cases() -> None:
    group = {"VALUE": 2.5, "LISTED": [1.0, 2.0]}

    np.testing.assert_allclose(_as_1d_float(group, "value"), np.asarray([2.5]))
    np.testing.assert_allclose(_as_1d_float(group, "listed"), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(_as_1d_float(group, "missing", default=7.0), np.asarray([7.0]))
    with pytest.raises(KeyError):
        _as_1d_float(group, "missing")

    xi = np.asarray([-1.0, 0.0, 1.0])
    p = _legendre_matrix(xi, n_l=3)
    expected = np.asarray(
        [
            [1.0, -1.0, 1.0],
            [1.0, 0.0, -0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(p, expected, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="n_l must be >= 1"):
        _legendre_matrix(xi, n_l=0)


def test_export_f_config_builds_identity_like_maps_and_preserves_constant_distribution() -> None:
    nml = _minimal_namelist(
        {
            "export_f": {
                "EXPORT_FULL_F": True,
                "EXPORT_F_THETA_OPTION": 0,
                "EXPORT_F_ZETA_OPTION": 0,
                "EXPORT_F_X_OPTION": 0,
                "EXPORT_F_XI_OPTION": 1,
                "EXPORT_F_XI": [-1.0, 0.0, 1.0],
            },
            "otherNumericalParameters": {},
        }
    )
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi]),
        zeta=np.asarray([0.0, np.pi / 5.0]),
        x=np.asarray([0.4, 1.2]),
        n_xi=3,
    )
    geom = SimpleNamespace(n_periods=5)

    cfg = _export_f_config(nml=nml, grids=grids, geom=geom)
    assert cfg is not None
    assert cfg.n_export_theta == 2
    assert cfg.n_export_zeta == 2
    assert cfg.n_export_x == 2
    assert cfg.n_export_xi == 3
    np.testing.assert_allclose(cfg.map_theta, np.eye(2))
    np.testing.assert_allclose(cfg.map_zeta, np.eye(2))
    np.testing.assert_allclose(cfg.map_x, np.eye(2))

    f = np.zeros((2, 2, 3, 2, 2), dtype=np.float64)
    f[:, :, 0, :, :] = 1.0
    mapped = _apply_export_f_maps(f, cfg)
    assert mapped.shape == (2, 2, 3, 2, 2)
    np.testing.assert_allclose(mapped, 1.0)


def test_export_f_config_nearest_neighbor_x_and_invalid_theta_option() -> None:
    base_groups = {
        "export_f": {
            "EXPORT_DELTA_F": True,
            "EXPORT_F_THETA_OPTION": 2,
            "EXPORT_F_THETA": [0.1, 6.25],
            "EXPORT_F_ZETA_OPTION": 2,
            "EXPORT_F_ZETA": [0.02],
            "EXPORT_F_X_OPTION": 2,
            "EXPORT_F_X": [0.18, 1.7],
            "EXPORT_F_XI_OPTION": 0,
        },
        "otherNumericalParameters": {},
    }
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi, 1.5 * np.pi]),
        zeta=np.asarray([0.0, 0.2]),
        x=np.asarray([0.2, 0.8, 1.6]),
        n_xi=2,
    )
    geom = SimpleNamespace(n_periods=5)

    cfg = _export_f_config(nml=_minimal_namelist(base_groups), grids=grids, geom=geom)
    assert cfg is not None
    np.testing.assert_allclose(cfg.export_x, np.asarray([0.2, 1.6]))
    np.testing.assert_allclose(cfg.map_x.sum(axis=1), 1.0)

    bad_groups = {
        **base_groups,
        "export_f": {**base_groups["export_f"], "EXPORT_F_THETA_OPTION": 99},
    }
    with pytest.raises(ValueError, match="Invalid export_f_theta_option"):
        _export_f_config(nml=_minimal_namelist(bad_groups), grids=grids, geom=geom)


def test_export_f_config_periodic_linear_maps_wrap_theta_and_zeta() -> None:
    nml = _minimal_namelist(
        {
            "export_f": {
                "EXPORT_DELTA_F": True,
                "EXPORT_F_THETA_OPTION": 1,
                "EXPORT_F_THETA": [7.0 * np.pi / 4.0],
                "EXPORT_F_ZETA_OPTION": 1,
                "EXPORT_F_ZETA": [3.0 * np.pi / 10.0],
                "EXPORT_F_X_OPTION": 0,
                "EXPORT_F_XI_OPTION": 0,
            },
            "otherNumericalParameters": {},
        }
    )
    grids = SimpleNamespace(
        theta=np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]),
        zeta=np.asarray([0.0, np.pi / 5.0]),
        x=np.asarray([0.4, 1.2]),
        n_xi=3,
    )
    geom = SimpleNamespace(n_periods=5)

    cfg = _export_f_config(nml=nml, grids=grids, geom=geom)
    assert cfg is not None
    np.testing.assert_allclose(cfg.map_theta, np.asarray([[0.5, 0.0, 0.0, 0.5]]), atol=1e-15)
    np.testing.assert_allclose(cfg.map_zeta, np.asarray([[0.5, 0.5]]), atol=1e-15)
    np.testing.assert_allclose(cfg.map_x, np.eye(2))
    np.testing.assert_allclose(cfg.map_xi, np.eye(3))


def test_export_f_config_single_zeta_and_invalid_non_theta_options() -> None:
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi]),
        zeta=np.asarray([0.0]),
        x=np.asarray([0.4, 1.2]),
        n_xi=2,
    )
    geom = SimpleNamespace(n_periods=5)
    base = {
        "export_f": {
            "EXPORT_FULL_F": True,
            "EXPORT_F_THETA_OPTION": 0,
            "EXPORT_F_ZETA_OPTION": 99,
            "EXPORT_F_X_OPTION": 0,
            "EXPORT_F_XI_OPTION": 0,
        },
        "otherNumericalParameters": {},
    }

    cfg = _export_f_config(nml=_minimal_namelist(base), grids=grids, geom=geom)
    assert cfg is not None
    np.testing.assert_allclose(cfg.map_zeta, np.ones((1, 1)))

    bad_x = {"export_f": {**base["export_f"], "EXPORT_F_X_OPTION": 99}, "otherNumericalParameters": {}}
    with pytest.raises(ValueError, match="Invalid export_f_x_option"):
        _export_f_config(nml=_minimal_namelist(bad_x), grids=grids, geom=geom)

    bad_xi = {"export_f": {**base["export_f"], "EXPORT_F_XI_OPTION": 99}, "otherNumericalParameters": {}}
    with pytest.raises(ValueError, match="Invalid export_f_xi_option"):
        _export_f_config(nml=_minimal_namelist(bad_xi), grids=grids, geom=geom)

    grids_two_zeta = SimpleNamespace(theta=grids.theta, zeta=np.asarray([0.0, np.pi / 5.0]), x=grids.x, n_xi=2)
    with pytest.raises(ValueError, match="Invalid export_f_zeta_option"):
        _export_f_config(nml=_minimal_namelist(base), grids=grids_two_zeta, geom=geom)
