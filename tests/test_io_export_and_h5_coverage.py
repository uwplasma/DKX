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
    read_sfincs_output_file,
    write_sfincs_h5,
    write_sfincs_output_file,
)
from sfincs_jax.namelist import Namelist
from sfincs_jax.outputs.formats import write_export_f_state_vectors_to_data
from sfincs_jax.outputs.rhsmode1 import (
    _add_rhsmode1_solver_diagnostics,
    _compact_json_metadata,
    _metadata_float,
    _metadata_int,
    _raise_for_nonconverged_rhsmode1_output,
    _rhs1_active_size_for_trace,
    _rhsmode1_result_residual_and_target,
    _should_fail_nonconverged_rhsmode1_output,
    _solver_metadata_dict,
    _write_nonconverged_rhsmode1_solver_trace_json,
)
from sfincs_jax.solvers.diagnostics import read_solver_trace_json


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
    assert _should_fail_nonconverged_rhsmode1_output(
        active_total_size=12_725,
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


def test_nonconverged_rhsmode1_solver_trace_sidecar_is_written(tmp_path: Path) -> None:
    trace_path = tmp_path / "solver_trace.json"
    input_path = tmp_path / "input.namelist"
    output_path = tmp_path / "sfincsOutput.h5"
    input_path.write_text("&general\n/\n")
    op = SimpleNamespace(total_size=20_284, active_size=11_496, collision_operator=0)
    result = SimpleNamespace(
        op=op,
        residual_norm=np.asarray(1.334334e-5),
        rhs=np.asarray([1.466182e-5], dtype=np.float64),
        metadata={
            "accepted_converged": False,
            "acceptance_criterion": "true_residual",
            "setup_s": 0.4,
            "solve_s": 193.8,
            "matvecs": 803,
            "solver_kind": "xblock_sparse_pc_gmres",
            "sparse_pc_xblock_preconditioner_built": False,
        },
    )

    _write_nonconverged_rhsmode1_solver_trace_json(
        solver_trace_path=trace_path,
        input_namelist=input_path,
        output_path=output_path,
        output_format="h5",
        rhs_mode=1,
        geom_scheme_hint=5,
        compute_solution=True,
        compute_transport_matrix=False,
        differentiable=None,
        result=result,
        op_fallback=op,
        solver_tol=1.0e-6,
        solve_method="xblock_sparse_pc_gmres",
        residual_norm=1.334334e-5,
        residual_target=1.466182e-11,
        active_total_size=11_496,
        run_t0=0.0,
        profiler=None,
    )

    trace = read_solver_trace_json(trace_path)
    assert trace.rhs_mode == 1
    assert trace.selected_path == "rhsmode1_solution"
    assert trace.solve_method == "xblock_sparse_pc_gmres"
    assert trace.active_size == 11_496
    assert trace.total_size == 20_284
    assert trace.residual_norm == pytest.approx(1.334334e-5)
    assert trace.residual_target == pytest.approx(1.466182e-11)
    assert trace.converged is False
    assert trace.matvec_count == 803
    assert trace.metadata["output_refused"] is True
    assert trace.metadata["failure_reason"] == "nonconverged_rhsmode1_output"
    solver_metadata = trace.metadata["solver_metadata"]
    assert solver_metadata["sparse_pc_xblock_preconditioner_built"] is False


def test_rhsmode1_active_trace_size_uses_pas_projection_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    collisionless = SimpleNamespace(n_xi_for_x=np.asarray([3, 2, 1], dtype=np.int32))
    op = SimpleNamespace(
        n_species=2,
        n_theta=3,
        n_zeta=4,
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=2,
        phi1_size=0,
        extra_size=2,
        total_size=5000,
        fblock=SimpleNamespace(collisionless=collisionless, pas=SimpleNamespace()),
    )

    active_f = 2 * int(np.sum(collisionless.n_xi_for_x)) * 3 * 4
    assert _rhs1_active_size_for_trace(op) == active_f

    monkeypatch.setenv("SFINCS_JAX_PAS_PROJECT_MIN", "99999")
    assert _rhs1_active_size_for_trace(op) == active_f + 2

    op.include_phi1 = True
    op.phi1_size = 7
    assert _rhs1_active_size_for_trace(op) == active_f + 7 + 2
    assert _rhs1_active_size_for_trace(SimpleNamespace()) is None


def test_rhsmode1_solver_metadata_helpers_are_fail_closed_and_bounded() -> None:
    metadata = {
        "good_int": "12",
        "bad_int": "not-an-int",
        "negative_int": -1,
        "good_float": "1.25",
        "bad_float": "nan",
        "inf_float": float("inf"),
    }

    assert _metadata_int(metadata, "good_int") == 12
    assert _metadata_int(metadata, "bad_int") is None
    assert _metadata_int(metadata, "negative_int") is None
    assert _metadata_int(metadata, "missing") is None
    assert _metadata_float(metadata, "good_float") == pytest.approx(1.25)
    assert _metadata_float(metadata, "bad_float") is None
    assert _metadata_float(metadata, "inf_float") is None
    assert _metadata_float(metadata, "missing") is None

    compact = _compact_json_metadata({"letters": "abcdef"}, max_chars=12)
    assert compact is not None
    assert compact.endswith("...<truncated>")
    assert len(compact) <= len("...<truncated>")
    assert _compact_json_metadata({"ok": (1, 2)}, max_chars=128) == '{"ok": [1, 2]}'

    source = {"metadata": {"accepted_converged": True}}
    copied = _solver_metadata_dict(SimpleNamespace(**source))
    copied["accepted_converged"] = False
    assert source["metadata"]["accepted_converged"] is True
    assert _solver_metadata_dict(SimpleNamespace(metadata=("not", "dict"))) == {}


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
            "solve_method_requested": "host_structured_csr",
            "solver_path": "structured_full_csr_host_gmres",
            "solver_kind": "structured_full_csr",
            "preconditioner_kind": "xblock_tz_low_l_schur",
            "iterations": 7,
            "matvecs": 11,
            "setup_s": 0.25,
            "solve_s": 0.75,
            "elapsed_s": 1.0,
            "sparse_pattern_nnz": 1234,
            "sparse_pattern_avg_row_nnz": 5.5,
            "sparse_pattern_max_row_nnz": 9,
            "csr_nnz": 4321,
            "csr_operator_nbytes": 876543,
            "sparse_pattern_build_s": 0.1,
            "sparse_pc_factor_s": 0.15,
            "sparse_pc_factor_nbytes_estimate": None,
            "sparse_pc_factor_nnz_estimate": None,
            "sparse_pc_xblock_preconditioner_xi": 1,
            "sparse_pc_xblock_assembled_host": True,
            "xblock_initial_seed_used": True,
            "xblock_initial_seed_residual_norm": 3.0e-8,
            "xblock_initial_seed_residual_ratio": 0.3,
            "xblock_post_minres_steps_requested": 3,
            "xblock_post_minres_steps_accepted": 2,
            "xblock_post_minres_residual_before": 4.0e-8,
            "xblock_post_minres_residual_after": 2.5e-8,
            "xblock_post_coarse_steps_requested": 1,
            "xblock_post_coarse_steps_accepted": 1,
            "xblock_post_coarse_direction_count": 6,
            "xblock_post_coarse_residual_before": 2.5e-8,
            "xblock_post_coarse_residual_after": 1.5e-8,
            "xblock_post_residual_equation_steps_requested": 1,
            "xblock_post_residual_equation_steps_accepted": 1,
            "xblock_post_residual_equation_direction_count": 9,
            "xblock_post_residual_equation_residual_before": 1.5e-8,
            "xblock_post_residual_equation_residual_after": 9.0e-9,
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb": 1024.0,
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto": True,
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata": {
                "selected": True,
                "kind": "active_fortran_v3_reduced_lu",
                "reason": "complete",
                "setup_s": 0.4,
                "metadata": {
                    "factor_kind": "lu",
                    "permc_spec": "NATURAL",
                    "permc_spec_requested": "AUTO",
                    "permc_spec_candidates": ("NATURAL", "COLAMD"),
                    "permc_failures": (
                        {
                            "permc_spec": "MMD_ATA",
                            "error_type": "RuntimeError",
                            "error": "not selected",
                        },
                    ),
                },
            },
        },
    )

    assert data["linearSolverMethod"] == "sparse_host"
    assert float(np.asarray(data["linearSolverResidualNorm"])) == 2.0e-8
    assert float(np.asarray(data["linearSolverResidualTarget"])) == 1.0e-7
    assert int(np.asarray(data["linearSolverConverged"])) == 1
    assert int(np.asarray(data["linearSolverAccepted"])) == 1
    assert data["linearSolverAcceptanceCriterion"] == "true_residual"
    assert data["linearSolverRequestedMethod"] == "host_structured_csr"
    assert data["linearSolverPath"] == "structured_full_csr_host_gmres"
    assert data["linearSolverKind"] == "structured_full_csr"
    assert data["linearSolverPreconditionerKind"] == "xblock_tz_low_l_schur"
    assert int(np.asarray(data["linearSolverIterations"])) == 7
    assert int(np.asarray(data["linearSolverMatvecs"])) == 11
    assert float(np.asarray(data["linearSolverSetupTime"])) == pytest.approx(0.25)
    assert float(np.asarray(data["linearSolverSolveTime"])) == pytest.approx(0.75)
    assert float(np.asarray(data["linearSolverElapsedTime"])) == pytest.approx(1.0)
    assert int(np.asarray(data["linearSolverSparsePatternNnz"])) == 1234
    assert float(np.asarray(data["linearSolverSparsePatternAvgRowNnz"])) == pytest.approx(5.5)
    assert int(np.asarray(data["linearSolverSparsePatternMaxRowNnz"])) == 9
    assert int(np.asarray(data["linearSolverCsrNnz"])) == 4321
    assert int(np.asarray(data["linearSolverCsrOperatorNbytes"])) == 876543
    assert float(np.asarray(data["linearSolverSparsePatternBuildTime"])) == pytest.approx(0.1)
    assert float(np.asarray(data["linearSolverSparsePCFactorTime"])) == pytest.approx(0.15)
    assert "linearSolverSparsePCFactorNbytesEstimate" not in data
    assert "linearSolverSparsePCFactorNnzEstimate" not in data
    assert int(np.asarray(data["linearSolverSparsePCXBlockPreconditionerXi"])) == 1
    assert int(np.asarray(data["linearSolverSparsePCXBlockAssembledHost"])) == 1
    assert data["linearSolverSparsePCSelectedKind"] == "active_fortran_v3_reduced_lu"
    assert data["linearSolverSparsePCFactorKind"] == "lu"
    assert data["linearSolverSparsePCPermcSpec"] == "NATURAL"
    assert data["linearSolverSparsePCPermcSpecRequested"] == "AUTO"
    assert '"NATURAL"' in data["linearSolverSparsePCPermcSpecCandidatesJson"]
    assert '"MMD_ATA"' in data["linearSolverSparsePCPermcFailuresJson"]
    assert int(np.asarray(data["linearSolverXBlockInitialSeedUsed"])) == 1
    assert float(np.asarray(data["linearSolverXBlockInitialSeedResidualNorm"])) == pytest.approx(3.0e-8)
    assert float(np.asarray(data["linearSolverXBlockInitialSeedResidualRatio"])) == pytest.approx(0.3)
    assert int(np.asarray(data["linearSolverXBlockPostMinresStepsRequested"])) == 3
    assert int(np.asarray(data["linearSolverXBlockPostMinresStepsAccepted"])) == 2
    assert float(np.asarray(data["linearSolverXBlockPostMinresResidualBefore"])) == pytest.approx(4.0e-8)
    assert float(np.asarray(data["linearSolverXBlockPostMinresResidualAfter"])) == pytest.approx(2.5e-8)
    assert int(np.asarray(data["linearSolverXBlockPostCoarseStepsRequested"])) == 1
    assert int(np.asarray(data["linearSolverXBlockPostCoarseStepsAccepted"])) == 1
    assert int(np.asarray(data["linearSolverXBlockPostCoarseDirectionCount"])) == 6
    assert float(np.asarray(data["linearSolverXBlockPostCoarseResidualBefore"])) == pytest.approx(2.5e-8)
    assert float(np.asarray(data["linearSolverXBlockPostCoarseResidualAfter"])) == pytest.approx(1.5e-8)
    assert int(np.asarray(data["linearSolverXBlockPostResidualEquationStepsRequested"])) == 1
    assert int(np.asarray(data["linearSolverXBlockPostResidualEquationStepsAccepted"])) == 1
    assert int(np.asarray(data["linearSolverXBlockPostResidualEquationDirectionCount"])) == 9
    assert float(np.asarray(data["linearSolverXBlockPostResidualEquationResidualBefore"])) == pytest.approx(1.5e-8)
    assert float(np.asarray(data["linearSolverXBlockPostResidualEquationResidualAfter"])) == pytest.approx(9.0e-9)
    assert float(np.asarray(data["linearSolverDirectTailStructuredPCMaxMB"])) == pytest.approx(1024.0)
    assert int(np.asarray(data["linearSolverDirectTailStructuredPCMaxMBAuto"])) == 1
    assert not any("DirectTailSupportMode" in key for key in data)
    assert not any("TrueCoupledCoarse" in key for key in data)
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


def test_write_export_f_state_vectors_to_data_writes_delta_and_full_iterations() -> None:
    nml = _minimal_namelist(
        {
            "export_f": {
                "EXPORT_FULL_F": True,
                "EXPORT_DELTA_F": True,
                "EXPORT_F_THETA_OPTION": 0,
                "EXPORT_F_ZETA_OPTION": 0,
                "EXPORT_F_X_OPTION": 0,
                "EXPORT_F_XI_OPTION": 0,
            },
            "otherNumericalParameters": {},
        }
    )
    grids = SimpleNamespace(
        theta=np.asarray([0.0, np.pi]),
        zeta=np.asarray([0.0]),
        x=np.asarray([0.4]),
        n_xi=2,
    )
    geom = SimpleNamespace(n_periods=5)
    cfg = _export_f_config(nml=nml, grids=grids, geom=geom)
    assert cfg is not None

    f_shape = (1, 1, 2, 2, 1)
    f0_l0 = np.asarray([[[[10.0], [20.0]]]])
    state0 = np.arange(np.prod(f_shape), dtype=np.float64)
    state1 = state0 + 100.0
    data = {"export_full_f": np.int32(1), "export_delta_f": np.int32(1)}

    write_export_f_state_vectors_to_data(
        data=data,
        state_vectors=[state0, state1],
        f_size=state0.size,
        f_shape=f_shape,
        f0_l0=f0_l0,
        export_cfg=cfg,
        fortran_h5_layout_fn=lambda x: x,
    )

    assert data["delta_f"].shape == (1, 2, 1, 2, 1, 2)
    assert data["full_f"].shape == data["delta_f"].shape
    np.testing.assert_allclose(data["delta_f"][0, 0, 0, :, 0, 0], np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(data["full_f"][0, 0, 0, :, 0, 0], np.asarray([10.0, 21.0]))
    np.testing.assert_allclose(data["full_f"][0, 0, 0, :, 0, 1], np.asarray([110.0, 121.0]))


def test_write_export_f_state_vectors_to_data_noops_without_enabled_export() -> None:
    cfg = SimpleNamespace(
        map_x=np.eye(1),
        map_xi=np.eye(1),
        map_theta=np.eye(1),
        map_zeta=np.eye(1),
    )
    data = {"export_full_f": np.int32(-1), "export_delta_f": np.int32(-1)}
    write_export_f_state_vectors_to_data(
        data=data,
        state_vectors=[np.asarray([1.0])],
        f_size=1,
        f_shape=(1, 1, 1, 1, 1),
        f0_l0=np.ones((1, 1, 1, 1)),
        export_cfg=cfg,
    )

    assert set(data) == {"export_full_f", "export_delta_f"}


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
