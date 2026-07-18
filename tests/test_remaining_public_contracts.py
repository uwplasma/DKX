from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dkx.validation.artifacts import (
    ARTIFACT_CLASS_RELEASE_BLOCKING,
    BenchmarkArtifactIndex,
    BenchmarkArtifactIndexEntry,
    BenchmarkArtifactPolicyError,
    CollisionalityLike,
    CollisionalityRecord,
    ErSweepRecord,
    PhaseRecord,
    SuiteCaseMetric,
    validate_benchmark_artifact_file,
)
from dkx.validation.fortran import PetscCSRMatrix, PetscVec, parse_fortran_v3_profile_file
from dkx.workflows import optimization as opt
from dkx.workflows.optimization import (
    AmbipolarRoot,
    AmbipolarRootSummary,
    CandidateScanPlan,
    NeoclassicalObjectiveWeights,
    PromotionEvidenceLane,
    PromotionEvidencePlan,
    ScanPromotionRun,
    ScanPromotionSummary,
    load_ladder_config,
    load_proxy_summary,
    run_fortran_er_scan,
    write_promotion_evidence_plan,
)


def _valid_pas_benchmark_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "pas_tz_memory_fallback_benchmark",
        "plan": {
            "variants": ["zeta"],
            "variant_methods": [
                {"variant": "zeta", "realized_solve_method": "incremental"},
            ],
        },
        "results": [
            {
                "variant": "zeta",
                "status": "ok",
                "variant_provenance": {"variant": "zeta"},
                "solver_provenance": {
                    "requested_solve_method": "incremental",
                    "realized_solve_method": "incremental",
                },
                "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 0.1}],
                "tail_metadata": {"messages_tail_limit": 20, "messages_tail_count": 2},
            },
        ],
    }


def test_optimization_evidence_and_fortran_scan_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = AmbipolarRoot(er=1.25, root_type="electron", bracket=(1.0, 1.5), slope=2.0)
    summary = AmbipolarRootSummary(
        roots=(root,),
        er_min=-1.0,
        er_max=2.0,
        radial_current_min=-0.5,
        radial_current_max=0.5,
        bracketed=True,
    )
    weights = NeoclassicalObjectiveWeights(bootstrap=0.5, electron_root=2.0)
    assert summary.has_electron_root
    assert summary.as_dict()["roots"][0]["root_type"] == "electron"
    assert weights.bootstrap == 0.5

    run = ScanPromotionRun(
        path=tmp_path / "sfincsOutput.h5",
        er=1.25,
        radial_current=0.0,
        bootstrap_current=0.01,
        particle_flux=(1.0, -0.2),
        heat_flux=(0.5, 0.6),
        residual_norm=1.0e-12,
        residual_target=1.0e-10,
        residual_gate={"status": "pass", "failures": []},
    )
    promotion = ScanPromotionSummary(
        scan_dir=tmp_path,
        runs=(run,),
        selected_root=root,
        bootstrap_objective=1.0e-4,
        flux_objective={"total": 0.2},
        electron_root_penalty=0.0,
        gate_status="pass",
        failures=(),
    )
    assert promotion.as_dict()["gate_status"] == "pass"

    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text(json.dumps({"workflow": "proxy", "objective_preset": "qa"}) + "\n")
    assert load_proxy_summary(proxy_path)["workflow"] == "proxy"

    candidate = CandidateScanPlan(
        proxy_summary=proxy_path,
        input_namelist=tmp_path / "input.namelist",
        out_dir=tmp_path / "scan",
        er_values=(-1.0, 0.0, 1.0),
        compute_solution=True,
        compute_transport_matrix=False,
        jobs=2,
        skip_existing=True,
        scan_command=("dkx", "scan-er"),
        promotion_command=("python", "promote.py"),
    )
    assert candidate.as_dict()["jobs"] == 2

    lane = PromotionEvidenceLane(
        label="cpu",
        backend="jax_cpu",
        scan_dir=tmp_path / "cpu_scan",
        promotion_dir=tmp_path / "cpu_promotion",
        scan_command=("dkx", "scan-er"),
        promotion_command=("python", "promote.py"),
        env={"JAX_PLATFORM_NAME": "cpu"},
    )
    evidence = PromotionEvidencePlan(
        input_namelist=tmp_path / "input.namelist",
        out_dir=tmp_path / "evidence",
        er_values=(-1.0, 1.0),
        lanes=(lane,),
        comparison_command=("python", "compare.py"),
    )
    evidence_path = write_promotion_evidence_plan(tmp_path / "evidence.json", evidence)
    payload = json.loads(evidence_path.read_text())
    assert payload["lanes"][0]["env"] == {"JAX_PLATFORM_NAME": "cpu"}

    ladder_path = tmp_path / "ladder.json"
    ladder_path.write_text(json.dumps({"surfaces": [0.25, 0.5]}) + "\n")
    assert load_ladder_config(ladder_path)["surfaces"] == [0.25, 0.5]

    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&geometryParameters\n"
        " inputRadialCoordinateForGradients = 4\n"
        "/\n"
        "&physicsParameters\n"
        " Er = 0.0\n"
        "/\n"
    )

    def fake_run_sfincs_fortran(*, input_namelist, exe, workdir, timeout_s):
        del input_namelist, exe, timeout_s
        Path(workdir, "sfincsOutput.h5").write_bytes(b"placeholder")

    messages: list[str] = []
    monkeypatch.setattr(opt, "run_sfincs_fortran", fake_run_sfincs_fortran)
    result = run_fortran_er_scan(
        input_namelist=input_path,
        out_dir=tmp_path / "fortran_scan",
        values=(-1.0, 1.0),
        timeout_s=0.1,
        skip_existing=False,
        emit=lambda _level, message: messages.append(message),
    )
    assert result.variable == "Er"
    assert result.values == (1.0, -1.0)
    assert all(path.exists() for path in result.outputs)
    assert any("fortran-scan: progress" in message for message in messages)


def test_validation_artifact_and_profile_file_contracts(tmp_path: Path) -> None:
    assert CollisionalityLike.__name__ == "CollisionalityLike"
    collision = CollisionalityRecord(
        label="high-nu",
        nuprime=10.0,
        transport_matrix=np.eye(2),
    )
    sweep = ErSweepRecord(
        model="dkes",
        label="W7-X",
        er=0.0,
        er_over_eres=None,
        particle_flux_vm_psi_hat=1.0,
        heat_flux_vm_psi_hat=2.0,
        fsab_flow=0.1,
        fsab_jhat=0.2,
        output_path="sfincsOutput.h5",
    )
    phase = PhaseRecord(name="solve", elapsed_s=0.25, metadata={"rhs_mode": 1})
    assert collision.transport_matrix.shape == (2, 2)
    assert sweep.model == "dkes"
    assert phase.to_json()["metadata"] == {"rhs_mode": 1}

    metric = SuiteCaseMetric(
        case="qa_bootstrap",
        status="parity_ok",
        blocker_type="none",
        fortran_runtime_s=10.0,
        jax_runtime_s=5.0,
        jax_runtime_s_cold=6.0,
        jax_runtime_s_warm=4.0,
        jax_logged_elapsed_s=4.5,
        fortran_max_rss_mb=100.0,
        jax_max_rss_mb=50.0,
        jax_incremental_max_rss_mb=25.0,
        jax_rss_baseline_mb=20.0,
        jax_memory_metric_source="incremental",
        practical_mismatches=0,
        strict_mismatches=0,
    )
    assert metric.runtime_ratio == 0.5
    assert metric.warm_or_logged_runtime_source == "jax_runtime_s_warm"
    assert metric.active_memory_ratio == 0.25

    blocking_entry = BenchmarkArtifactIndexEntry(
        path=tmp_path / "bad.json",
        classification=ARTIFACT_CLASS_RELEASE_BLOCKING,
        errors=("missing schema_version",),
    )
    index = BenchmarkArtifactIndex(entries=(blocking_entry,))
    assert index.counts[ARTIFACT_CLASS_RELEASE_BLOCKING] == 1
    assert index.release_blocking == (blocking_entry,)

    valid_artifact = tmp_path / "pas_benchmark.json"
    valid_artifact.write_text(json.dumps(_valid_pas_benchmark_payload()) + "\n")
    validate_benchmark_artifact_file(valid_artifact)
    invalid_artifact = tmp_path / "invalid.json"
    invalid_artifact.write_text("{}\n")
    with pytest.raises(BenchmarkArtifactPolicyError, match="schema_version"):
        validate_benchmark_artifact_file(invalid_artifact)

    log_path = tmp_path / "fortran_profile.log"
    log_path.write_text(
        "Parallel job ( 2 processes) detected\n"
        "Ntheta = 25\n"
        "Nzeta = 51\n"
        "Nxi = 100\n"
        "NL = 4\n"
        "Nx = 4\n"
        "solverTolerance = 1.0e-9\n"
        "Solver package which will be used:\n"
        "mumps\n"
        "The matrix is 4 x 4 elements\n"
        "# of nonzeros in Jacobian matrix: 8, allocated: 16\n"
        "# of nonzeros in Jacobian preconditioner matrix: 6, allocated: 12\n"
        "Entering DMUMPS\n"
        "INFOG(21) = 33\n"
        "INFOG(22) = 44\n"
        "INFOG(29) = 55\n"
        " 0 KSP Residual norm 1.0e-3\n"
        " 1 KSP Residual norm 1.0e-9\n"
        "KSPConvergedReason = 2\n"
        "Elapsed time in analysis driver= 0.11\n"
        "Elapsed time in factorization driver= 0.22\n"
    )
    profile = parse_fortran_v3_profile_file(log_path)
    assert profile["solver_package"] == "mumps"
    assert profile["matrix_nnz"] == {"nnz": 8, "allocated_nnz": 16}
    assert profile["ksp"]["final_residual"] == 1.0e-9

    vec = PetscVec(values=np.asarray([1.0, 2.0]))
    matrix = PetscCSRMatrix(
        shape=(2, 2),
        row_ptr=np.asarray([0, 1, 2]),
        col_ind=np.asarray([0, 1]),
        data=np.asarray([3.0, 4.0]),
    )
    assert vec.size == 2
    assert matrix.get(1, 1) == 4.0
    assert matrix.get(0, 1) == 0.0


