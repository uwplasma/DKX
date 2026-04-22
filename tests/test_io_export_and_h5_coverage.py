from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.io import (
    _apply_export_f_maps,
    _as_1d_float,
    _export_f_config,
    _legendre_matrix,
    read_sfincs_h5,
    write_sfincs_h5,
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

    np.testing.assert_allclose(loaded["scalar"], data["scalar"])
    np.testing.assert_allclose(loaded["vector"], data["vector"])
    np.testing.assert_allclose(loaded["matrix"], data["matrix"])

    with pytest.raises(FileExistsError):
        write_sfincs_h5(path=out, data=data, overwrite=False)


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
