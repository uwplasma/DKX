from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfincs_jax.solvers.preconditioner_pas_policy import (
    ConstrainedPASBranchRecord,
    summarize_constrained_pas_branches,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "reference_solver_path_artifacts"
    / "constrained_pas_branch_probe_2026-05-02.json"
)


def _records(case_name: str) -> list[ConstrainedPASBranchRecord]:
    payload = json.loads(FIXTURE.read_text())
    case = payload["cases"][case_name]
    return [ConstrainedPASBranchRecord(**row) for row in case["records"]]


def test_empty_constrained_pas_branch_summary_is_explicit() -> None:
    summary = summarize_constrained_pas_branches([])

    assert summary.reference_label is None
    assert not summary.branch_sensitive
    assert summary.recommendation == "no_branch_records"


def test_constrained_pas_branch_summary_accepts_consistent_converged_records() -> None:
    records = [
        ConstrainedPASBranchRecord(
            label="exact",
            observable=1.0,
            residual_norm=1.0e-13,
            residual_target=1.0e-10,
            criterion="true_residual",
        ),
        ConstrainedPASBranchRecord(
            label="gmres",
            observable=1.0 + 2.0e-6,
            residual_norm=2.0e-13,
            residual_target=1.0e-10,
            criterion="true_residual",
        ),
    ]

    summary = summarize_constrained_pas_branches(records, branch_relative_gate=1.0e-4)

    assert summary.reference_label == "exact"
    assert not summary.branch_sensitive
    assert summary.weak_reference_labels == ()
    assert summary.recommendation == "converged_branch_consistent"


def test_constrained_pas_branch_probe_blocks_weak_reference_parity_claim() -> None:
    records = _records("finite_beta_profile_current_25x31x17_nx11")

    summary = summarize_constrained_pas_branches(records)

    assert summary.reference_label == "jax_sparse_pc_gmres"
    assert summary.branch_sensitive
    assert summary.max_relative_spread == pytest.approx(0.99911791, rel=1.0e-5)
    assert summary.has_reference_quality_blocker
    assert set(summary.weak_reference_labels) == {
        "jax_petsc_compat_minimum_norm",
        "fortran_v3_preconditioned_branch_plain",
        "fortran_v3_preconditioned_branch_binary_dump",
    }
    assert summary.recommendation == "pin_gauge_before_parity_claim"


def test_constrained_pas_branch_probe_keeps_diagnostic_lsmr_out_of_true_residual_claim() -> None:
    records = _records("finite_beta_profile_current_25x31x17_nx11")
    diagnostic = next(row for row in records if row.label == "jax_petsc_compat_minimum_norm")

    assert diagnostic.accepted
    assert diagnostic.criterion == "petsc_compatible_minimum_norm_diagnostic"
    assert not diagnostic.true_residual_converged()
    assert diagnostic.residual_ratio > 1.0e6
