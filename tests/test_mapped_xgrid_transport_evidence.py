from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.mapped_xgrid_transport_evidence import (
    copy_namelist_with_mapped_xgrid,
    copy_namelist_with_resolution,
    run_rational_tail_transport_comparison,
    transport_matrix_error,
    transport_solve_summary,
)
from sfincs_jax.namelist import Namelist, read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist


def _nml() -> Namelist:
    return Namelist(
        groups={
            "resolutionparameters": {
                "NX": 5,
                "NTHETA": 3,
                "NZETA": 1,
                "NXI": 4,
                "NL": 3,
            },
            "othernumericalparameters": {
                "XGRIDSCHEME": 2,
                "NXI_FOR_X_OPTION": 0,
            },
            "geometryparameters": {
                "GEOMETRYSCHEME": 1,
            },
            "general": {
                "RHSMODE": 2,
            },
        },
        indexed={},
        source_path=Path("input.namelist"),
        source_text="&general\n/\n",
    )


def _get(mapping: dict, key: str, default):
    for existing_key, value in mapping.items():
        if existing_key.lower() == key.lower():
            return value
    return default


def _fake_transport_solve(*, nml: Namelist, **_kwargs):
    other = nml.group("otherNumericalParameters")
    resolution = nml.group("resolutionParameters")
    x_scheme = int(_get(other, "XGRIDSCHEME", 0))
    log_length = float(_get(other, "MAPPEDXGRIDLOGLENGTH", 0.0))
    scale = 1.0 if x_scheme != 50 else 1.0 + 0.2 * (log_length + 0.5) ** 2
    base = np.asarray([[1.0, 0.2], [0.1, 2.0]], dtype=np.float64)
    return SimpleNamespace(
        op0=SimpleNamespace(total_size=12, n_x=int(_get(resolution, "NX", 0))),
        transport_matrix=scale * base,
        state_vectors_by_rhs={1: np.ones(2), 2: np.ones(2)},
        residual_norms_by_rhs={1: 1.0e-10 * scale, 2: 2.0e-10 * scale},
        rhs_norms_by_rhs={1: 1.0, 2: 2.0},
        fsab_flow=np.zeros((1, 2)),
        particle_flux_vm_psi_hat=np.zeros((1, 2)),
        heat_flux_vm_psi_hat=np.zeros((1, 2)),
        elapsed_time_s=np.asarray([0.01, 0.02]) * scale,
        transport_output_fields=None,
        active_size=8,
        use_active_dof_mode=True,
        solver_kinds_by_rhs={1: "sparse_lu", 2: "gmres"},
        solve_methods_by_rhs={1: "sparse_lu", 2: "incremental"},
    )


def test_copy_namelist_with_mapped_xgrid_sets_opt_in_keys_without_mutating_source():
    original = _nml()
    mapped = copy_namelist_with_mapped_xgrid(
        original,
        log_length=-0.25,
        eta_kind="uniform",
        derivative="chain-rule",
        extra_options={"MAPPEDXGRIDCOMMENT": "test"},
    )

    assert int(original.group("otherNumericalParameters")["XGRIDSCHEME"]) == 2
    other = mapped.group("otherNumericalParameters")
    assert int(other["XGRIDSCHEME"]) == 50
    assert other["MAPPEDXGRIDFAMILY"] == "rational_tail"
    np.testing.assert_allclose(float(other["MAPPEDXGRIDLOGLENGTH"]), -0.25)
    assert other["MAPPEDXGRIDETAKIND"] == "uniform"
    assert other["MAPPEDXGRIDDERIVATIVE"] == "chain-rule"
    assert other["MAPPEDXGRIDCOMMENT"] == "test"
    assert mapped.source_path == original.source_path
    assert mapped.source_text == original.source_text

    no_eps = copy_namelist_with_mapped_xgrid(original, log_length=0.0, eps=None)
    assert "MAPPEDXGRIDEPS" not in no_eps.group("otherNumericalParameters")


def test_copy_namelist_with_resolution_updates_only_requested_resolution_keys():
    original = _nml()
    resized = copy_namelist_with_resolution(original, nx=7, nxi=9)

    original_resolution = original.group("resolutionParameters")
    resized_resolution = resized.group("resolutionParameters")
    assert int(original_resolution["NX"]) == 5
    assert int(original_resolution["NXI"]) == 4
    assert int(resized_resolution["NX"]) == 7
    assert int(resized_resolution["NXI"]) == 9
    assert int(resized_resolution["NL"]) == int(original_resolution["NL"])


def test_copy_namelist_mapped_options_build_v3_speed_grid():
    fixture = Path(__file__).parent / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    nml = read_sfincs_input(fixture)
    mapped = copy_namelist_with_mapped_xgrid(nml, log_length=-0.3)
    grids = grids_from_namelist(mapped)

    x = np.asarray(grids.x)
    weights = np.asarray(grids.x_weights)
    assert x.shape == (int(nml.group("resolutionParameters")["NX"]),)
    assert np.all(np.diff(x) > 0.0)
    assert np.all(weights > 0.0)


def test_transport_solve_summary_handles_absolute_and_relative_residuals():
    result = _fake_transport_solve(nml=_nml())
    summary = transport_solve_summary(result)

    np.testing.assert_allclose(summary.max_residual_norm, 2.0e-10)
    np.testing.assert_allclose(summary.max_relative_residual_norm, 1.0e-10)
    np.testing.assert_allclose(summary.total_elapsed_time_s, 0.03)
    assert summary.total_size == 12
    assert summary.active_size == 8
    np.testing.assert_allclose(summary.active_fraction, 8.0 / 12.0)
    assert summary.n_x == 5
    assert summary.use_active_dof_mode is True
    assert summary.solver_kinds == ("gmres", "sparse_lu")
    assert summary.solve_methods == ("incremental", "sparse_lu")


def test_transport_solve_summary_without_rhs_norms_reports_nan_relative_residual():
    result = _fake_transport_solve(nml=_nml())
    delattr(result, "rhs_norms_by_rhs")
    result.residual_norms_by_rhs = {}
    summary = transport_solve_summary(result)

    assert summary.max_residual_norm == 0.0
    assert np.isnan(summary.max_relative_residual_norm)
    assert summary.active_size == 8


def test_transport_matrix_error_reports_frobenius_and_shape_mismatch():
    reference = _fake_transport_solve(nml=_nml())
    candidate = _fake_transport_solve(
        nml=copy_namelist_with_mapped_xgrid(_nml(), log_length=0.0),
    )

    error = transport_matrix_error(candidate, reference)
    assert error.relative_frobenius > 0.0
    assert error.max_abs > 0.0
    assert error.reference_norm > 0.0

    bad = SimpleNamespace(transport_matrix=np.ones((3, 3)))
    with pytest.raises(ValueError, match="same shape"):
        transport_matrix_error(bad, reference)


def test_run_rational_tail_transport_comparison_with_fake_solver():
    reference_nml = copy_namelist_with_resolution(_nml(), nx=7)
    report = run_rational_tail_transport_comparison(
        _nml(),
        log_length_values=(-0.5, 0.0),
        reference_nml=reference_nml,
        solve_fn=_fake_transport_solve,
        solve_kwargs={"tol": 1.0e-8},
    )

    assert len(report.rows) == 2
    np.testing.assert_allclose(report.reference_summary.max_residual_norm, 2.0e-10)
    assert report.reference_summary.n_x == 7
    assert report.best_by_transport_error.log_length == -0.5
    assert report.best_by_moment in report.rows
    for row in report.rows:
        assert row.total_size == 12
        assert row.active_size == 8
        np.testing.assert_allclose(row.active_fraction, 8.0 / 12.0)
        assert row.n_x == 5
        assert row.use_active_dof_mode is True
        assert row.solver_kinds == ("gmres", "sparse_lu")
        assert row.min_dx > 0.0
        assert row.width_ratio > 1.0
        assert row.moment_objective >= 0.0


def test_run_rational_tail_transport_comparison_can_reuse_reference_result():
    reference = _fake_transport_solve(nml=copy_namelist_with_resolution(_nml(), nx=9))
    seen_nx: list[int] = []

    def _recording_solve(*, nml: Namelist, **kwargs):
        seen_nx.append(int(_get(nml.group("resolutionParameters"), "NX", 0)))
        return _fake_transport_solve(nml=nml, **kwargs)

    report = run_rational_tail_transport_comparison(
        _nml(),
        log_length_values=(-0.5,),
        reference_result=reference,
        solve_fn=_recording_solve,
    )

    assert report.reference_summary.n_x == 9
    assert seen_nx == [5]


def test_run_rational_tail_transport_comparison_rejects_empty_scan():
    with pytest.raises(ValueError, match="at least one"):
        run_rational_tail_transport_comparison(
            _nml(),
            log_length_values=(),
            solve_fn=_fake_transport_solve,
        )
