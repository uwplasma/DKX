from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.discretization import v3
from sfincs_jax.discretization.xgrid import (
    make_x_polynomial_diff_matrices,
    x_weight_d1_over_weight_np,
    x_weight_d2_over_weight_np,
    x_weight_np,
)
from sfincs_jax.geometry import boozer_geometry_scheme1
from sfincs_jax.namelist import Namelist


@pytest.fixture(autouse=True)
def _clear_discretization_caches() -> None:
    v3._GRIDS_CACHE.clear()
    v3._GEOMETRY_CACHE.clear()
    v3._equilibrium_file_identity.cache_clear()
    yield
    v3._GRIDS_CACHE.clear()
    v3._GEOMETRY_CACHE.clear()
    v3._equilibrium_file_identity.cache_clear()


def _nml(
    *,
    resolution: dict | None = None,
    other: dict | None = None,
    geometry: dict | None = None,
    general: dict | None = None,
    source_path: Path | None = None,
) -> Namelist:
    return Namelist(
        groups={
            "resolutionparameters": {
                "NTHETA": 5,
                "NZETA": 5,
                "NX": 4,
                "NXI": 8,
                "NL": 4,
                **(resolution or {}),
            },
            "othernumericalparameters": {
                "THETADERIVATIVESCHEME": 2,
                "ZETADERIVATIVESCHEME": 2,
                "MAGNETICDRIFTDERIVATIVESCHEME": 3,
                "XGRIDSCHEME": 5,
                "NXI_FOR_X_OPTION": 0,
                "XDOTDERIVATIVESCHEME": 0,
                **(other or {}),
            },
            "geometryparameters": {
                "GEOMETRYSCHEME": 1,
                "HELICITY_N": 4,
                **(geometry or {}),
            },
            "general": {
                "RHSMODE": 1,
                **(general or {}),
            },
        },
        indexed={},
        source_path=source_path,
        source_text=None,
    )


def test_bc_header_period_parser_uses_run_directory_and_skips_column_header(tmp_path: Path) -> None:
    bc_path = tmp_path / "synthetic.bc"
    bc_path.write_text(
        "\n".join(
            [
                "CC synthetic Boozer header",
                " m0b  n0b nsurf nper  flux/[Tm^2]     a/[m]",
                "   2    0    1    7  1.000000E+00   1.00000",
            ]
        ),
        encoding="utf-8",
    )

    assert v3._n_periods_from_bc_file("synthetic.bc", base_dir=tmp_path) == 7


def test_xgrid_weight_formulas_and_polynomial_diff_matrix_shape_contract() -> None:
    x = np.asarray([0.0, 0.5, 1.5, 2.0], dtype=np.float64)
    k = 2.0

    weights = x_weight_np(x, k)
    d1_over_w = x_weight_d1_over_weight_np(x, k)
    d2_over_w = x_weight_d2_over_weight_np(x, k)
    x_matrix = np.asarray([0.2, 0.5, 1.5, 2.0], dtype=np.float64)
    ddx, d2dx2 = make_x_polynomial_diff_matrices(x_matrix, k=k)

    np.testing.assert_allclose(weights, np.exp(-(x * x)) * (x**k), rtol=1.0e-15, atol=1.0e-15)
    assert d1_over_w[0] == 0.0
    np.testing.assert_allclose(d1_over_w[1:], k / x[1:] - 2.0 * x[1:], rtol=1.0e-15, atol=1.0e-15)
    assert d2_over_w[0] == -2.0
    np.testing.assert_allclose(
        d2_over_w[1:],
        k * (k - 1.0) / (x[1:] * x[1:]) - 2.0 * (2.0 * k + 1.0) + 4.0 * x[1:] * x[1:],
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert ddx.shape == (x_matrix.size, x_matrix.size)
    assert d2dx2.shape == (x_matrix.size, x_matrix.size)
    assert np.isfinite(ddx).all()
    assert np.isfinite(d2dx2).all()


def test_bc_header_period_parser_fails_closed_on_malformed_files(tmp_path: Path) -> None:
    short_header = tmp_path / "short.bc"
    short_header.write_text("not enough\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected .bc header line"):
        v3._n_periods_from_bc_file("short.bc", base_dir=tmp_path)

    comments_only = tmp_path / "comments_only.bc"
    comments_only.write_text("CC no data\nCC still no data\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to find header line"):
        v3._n_periods_from_bc_file("comments_only.bc", base_dir=tmp_path)


def test_vmec_resolution_supports_ascii_to_netcdf_sibling_fallback(tmp_path: Path) -> None:
    netcdf_sibling = tmp_path / "wout_demo.nc"
    netcdf_sibling.write_bytes(b"not a real netcdf file, only a path-resolution fixture")

    resolved = v3._resolve_vmec_equilibrium_file(
        "wout_demo.txt",
        base_dir=tmp_path,
        extra_search_dirs=(),
    )

    assert resolved == netcdf_sibling


def test_vmec_resolution_keeps_non_ascii_missing_paths_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OFFLINE", "1")
    with pytest.raises(FileNotFoundError):
        v3._resolve_vmec_equilibrium_file(
            "missing_equilibrium_for_unit_test.nc",
            base_dir=tmp_path,
            extra_search_dirs=(),
        )


def test_equilibrium_file_key_is_content_based_for_copied_vmec_like_files(tmp_path: Path) -> None:
    file_a = tmp_path / "wout_a.nc"
    file_b = tmp_path / "wout_b.nc"
    file_c = tmp_path / "wout_c.nc"
    file_a.write_bytes(b"same equilibrium bytes")
    file_b.write_bytes(b"same equilibrium bytes")
    file_c.write_bytes(b"different equilibrium bytes with another size")

    def key_for(path: Path) -> tuple[int, str] | None:
        nml = _nml(geometry={"GEOMETRYSCHEME": 5, "EQUILIBRIUMFILE": path.name}, source_path=tmp_path / "input.namelist")
        return v3._equilibrium_file_key(
            nml=nml,
            geometry_scheme=5,
            geom_group=nml.group("geometryParameters"),
        )

    assert key_for(file_a) == key_for(file_b)
    assert key_for(file_a) != key_for(file_c)


def test_geometry_cache_payload_roundtrips_all_operator_fields() -> None:
    theta = jnp.linspace(0.0, 2.0 * math.pi, 5, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * math.pi / 4.0, 3, endpoint=False)
    geom = boozer_geometry_scheme1(
        theta=theta,
        zeta=zeta,
        epsilon_t=-0.07,
        epsilon_h=0.05,
        epsilon_antisymm=0.01,
        iota=0.45,
        g_hat=3.7,
        i_hat=0.1,
        b0_over_bbar=1.02,
        helicity_l=2,
        helicity_n=4,
        helicity_antisymm_l=1,
        helicity_antisymm_n=4,
    )

    payload = v3._geometry_to_cache_payload(geom, ("unit", "cache"))
    restored = v3._geometry_from_cache_payload(payload)

    assert restored is not None
    assert restored.n_periods == geom.n_periods
    assert restored.b0_over_bbar == pytest.approx(geom.b0_over_bbar)
    for field in v3._GEOMETRY_CACHE_FIELDS:
        np.testing.assert_allclose(np.asarray(getattr(restored, field)), np.asarray(getattr(geom, field)))


def test_geometry_cache_payload_rejects_stale_incomplete_or_corrupt_payloads() -> None:
    theta = jnp.asarray([0.0, 1.0])
    zeta = jnp.asarray([0.0])
    geom = boozer_geometry_scheme1(
        theta=theta,
        zeta=zeta,
        epsilon_t=0.0,
        epsilon_h=0.0,
        epsilon_antisymm=0.0,
        iota=0.4,
        g_hat=1.0,
        i_hat=0.0,
        b0_over_bbar=1.0,
        helicity_l=1,
        helicity_n=1,
        helicity_antisymm_l=1,
        helicity_antisymm_n=0,
    )
    payload = v3._geometry_to_cache_payload(geom, ("reject",))

    stale = dict(payload)
    stale["cache_version"] = np.asarray(-1, dtype=np.int32)
    assert v3._geometry_from_cache_payload(stale) is None

    missing_array = dict(payload)
    missing_array.pop(v3._GEOMETRY_CACHE_FIELDS[0])
    assert v3._geometry_from_cache_payload(missing_array) is None

    missing_scalar = dict(payload)
    missing_scalar.pop("iota")
    assert v3._geometry_from_cache_payload(missing_scalar) is None


def test_persistent_geometry_cache_save_load_and_corrupt_file_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GEOMETRY_CACHE", "1")
    monkeypatch.setenv("SFINCS_JAX_GEOMETRY_CACHE_PERSIST", "1")
    monkeypatch.setenv("SFINCS_JAX_GEOMETRY_CACHE_DIR", str(tmp_path))

    theta = jnp.asarray([0.0, 1.0])
    zeta = jnp.asarray([0.0, 0.5])
    geom = boozer_geometry_scheme1(
        theta=theta,
        zeta=zeta,
        epsilon_t=0.01,
        epsilon_h=0.02,
        epsilon_antisymm=0.0,
        iota=0.5,
        g_hat=2.0,
        i_hat=0.0,
        b0_over_bbar=1.0,
        helicity_l=1,
        helicity_n=2,
        helicity_antisymm_l=1,
        helicity_antisymm_n=0,
    )

    cache_key = ("geometry", "roundtrip")
    v3._save_geometry_cache(cache_key, geom)
    restored = v3._load_geometry_cache(cache_key)
    assert restored is not None
    np.testing.assert_allclose(np.asarray(restored.b_hat), np.asarray(geom.b_hat))

    corrupt_key = ("geometry", "corrupt")
    corrupt_path = v3._geometry_cache_path(corrupt_key)
    assert corrupt_path is not None
    corrupt_path.write_bytes(b"not an npz")
    assert v3._load_geometry_cache(corrupt_key) is None

    monkeypatch.setenv("SFINCS_JAX_GEOMETRY_CACHE", "0")
    assert v3._load_geometry_cache(cache_key) is None


def test_grids_from_namelist_uses_fortran_odd_grid_and_weight_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_GRIDS_CACHE", "0")
    grids = v3.grids_from_namelist(
        _nml(
            resolution={"NTHETA": 4, "NZETA": 4, "NX": 3, "NXI": 9, "NL": 4},
            geometry={"GEOMETRYSCHEME": 1, "HELICITY_N": 5},
        )
    )

    assert grids.theta.size == 5
    assert grids.zeta.size == 5
    np.testing.assert_allclose(float(jnp.sum(grids.theta_weights)), 2.0 * math.pi, rtol=1.0e-14)
    np.testing.assert_allclose(float(jnp.sum(grids.zeta_weights)), 2.0 * math.pi, rtol=1.0e-14)
    assert grids.ddtheta.shape == (5, 5)
    assert grids.ddzeta.shape == (5, 5)
    assert grids.ddtheta_magdrift_plus.shape == (5, 5)
    assert grids.ddzeta_magdrift_minus.shape == (5, 5)


@pytest.mark.parametrize("option", [0, 1, 2, 3])
def test_nxi_for_x_options_match_bounded_fortran_style_contract(option: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_GRIDS_CACHE", "0")
    grids = v3.grids_from_namelist(
        _nml(
            resolution={"NX": 5, "NXI": 12, "NL": 4},
            other={"NXI_FOR_X_OPTION": option},
        )
    )

    nxi_for_x = np.asarray(grids.n_xi_for_x)
    assert nxi_for_x.shape == (5,)
    if option == 0:
        np.testing.assert_array_equal(nxi_for_x, np.full((5,), 12))
    else:
        assert np.all(nxi_for_x >= 4)
        if option in {1, 2}:
            assert np.all(nxi_for_x <= 12)
        else:
            assert nxi_for_x[-1] > 12
        assert nxi_for_x[-1] >= nxi_for_x[0]


def test_grids_cache_can_be_disabled_for_diagnostic_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    nml = _nml()

    monkeypatch.setenv("SFINCS_JAX_GRIDS_CACHE", "1")
    cached_first = v3.grids_from_namelist(nml)
    cached_second = v3.grids_from_namelist(nml)
    assert cached_first is cached_second

    monkeypatch.setenv("SFINCS_JAX_GRIDS_CACHE", "0")
    uncached = v3.grids_from_namelist(nml)
    assert uncached is not cached_first


def test_grids_from_namelist_keeps_rhsmode3_monoenergetic_velocity_contract() -> None:
    grids = v3.grids_from_namelist(
        _nml(
            resolution={"NX": 7, "NXI": 9, "NL": 4},
            general={"RHSMODE": 3},
            other={"NXI_FOR_X_OPTION": 3, "XGRIDSCHEME": 6},
        )
    )

    np.testing.assert_allclose(np.asarray(grids.x), np.asarray([1.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(grids.x_weights), np.asarray([math.e]), rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_array_equal(np.asarray(grids.n_xi_for_x), np.asarray([9]))


@pytest.mark.parametrize(
    ("other", "error_type", "message"),
    [
        ({"THETADERIVATIVESCHEME": 9}, ValueError, "thetaDerivativeScheme"),
        ({"ZETADERIVATIVESCHEME": 9}, ValueError, "zetaDerivativeScheme"),
        ({"MAGNETICDRIFTDERIVATIVESCHEME": 99}, ValueError, "magneticDriftDerivativeScheme"),
        ({"XGRIDSCHEME": 7}, NotImplementedError, "xGridScheme"),
        ({"NXI_FOR_X_OPTION": 99}, ValueError, "Nxi_for_x_option"),
        ({"XDOTDERIVATIVESCHEME": 1}, NotImplementedError, "xDotDerivativeScheme"),
    ],
)
def test_grids_from_namelist_rejects_unsupported_numerical_knobs(
    other: dict,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        v3.grids_from_namelist(_nml(other=other))


@pytest.mark.parametrize("geometry_scheme", [5, 11, 12])
def test_grids_from_namelist_requires_equilibrium_file_for_file_backed_geometry(geometry_scheme: int) -> None:
    with pytest.raises(ValueError, match="requires equilibriumFile"):
        v3.grids_from_namelist(_nml(geometry={"GEOMETRYSCHEME": geometry_scheme}))


def test_grids_and_geometry_reject_unsupported_geometry_schemes() -> None:
    with pytest.raises(NotImplementedError, match="grid construction"):
        v3.grids_from_namelist(_nml(geometry={"GEOMETRYSCHEME": 99}))

    grids = v3.grids_from_namelist(_nml(geometry={"GEOMETRYSCHEME": 3}))
    with pytest.raises(NotImplementedError, match="implemented so far"):
        v3.geometry_from_namelist(nml=_nml(geometry={"GEOMETRYSCHEME": 3}), grids=grids)
