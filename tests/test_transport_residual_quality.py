from __future__ import annotations

import numpy as np

from sfincs_jax.problems.transport_policies import (
    transport_residual_gate_failure,
    transport_residual_gate_failures_from_arrays,
    transport_residual_gate_thresholds_from_env,
)


def test_transport_residual_gate_thresholds_parse_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL", "-1e-8")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL", "bad")

    max_abs, max_rel = transport_residual_gate_thresholds_from_env()

    assert max_abs == 0.0
    assert max_rel == 0.0


def test_transport_residual_gate_thresholds_support_custom_env_names(monkeypatch) -> None:
    monkeypatch.setenv("ABS_GATE", "2.5e-6")
    monkeypatch.setenv("REL_GATE", "3.5e-3")

    max_abs, max_rel = transport_residual_gate_thresholds_from_env(
        abs_env="ABS_GATE",
        rel_env="REL_GATE",
    )

    assert max_abs == 2.5e-6
    assert max_rel == 3.5e-3


def test_transport_residual_gate_failure_checks_absolute_and_relative() -> None:
    assert (
        transport_residual_gate_failure(
            which_rhs=2,
            residual_norm=1.0e-8,
            rhs_norm=1.0,
            max_abs=1.0e-6,
            max_relative=1.0e-6,
        )
        is None
    )

    failure = transport_residual_gate_failure(
        which_rhs=3,
        residual_norm=1.0e-7,
        rhs_norm=1.0e-10,
        max_abs=1.0e-6,
        max_relative=1.0e-6,
    )

    assert failure is not None
    assert "whichRHS=3" in failure
    assert "relative_residual=1.000000e+03" in failure


def test_transport_residual_gate_failure_reports_nonfinite_absolute_failure() -> None:
    failure = transport_residual_gate_failure(
        which_rhs=4,
        residual_norm=np.inf,
        rhs_norm=0.0,
        max_abs=1.0e-6,
        max_relative=0.0,
    )

    assert failure is not None
    assert "whichRHS=4" in failure
    assert "residual_norm=inf" in failure
    assert "relative_residual=nan" in failure


def test_transport_residual_gate_failures_from_arrays_handles_nan() -> None:
    failures = transport_residual_gate_failures_from_arrays(
        which_rhs_values=np.asarray([1, 2, 3], dtype=np.int32),
        residual_norms=np.asarray([1.0e-12, np.nan, 1.0e-8], dtype=np.float64),
        rhs_norms=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        max_abs=1.0e-6,
        max_relative=1.0e-6,
    )

    assert len(failures) == 1
    assert "whichRHS=2" in failures[0]
