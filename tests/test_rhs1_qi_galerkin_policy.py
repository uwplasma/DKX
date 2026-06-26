from __future__ import annotations

import pytest

from sfincs_jax.solvers.preconditioner_qi_corrections import (
    RHS1QIGalerkinProbeCandidate,
    parse_rhs1_qi_galerkin_dampings,
    parse_rhs1_qi_galerkin_modes,
    select_rhs1_qi_galerkin_probe_candidate,
)


def test_parse_qi_galerkin_modes_supports_auto_lists_and_invalid_tokens() -> None:
    assert parse_rhs1_qi_galerkin_modes("") == ("additive", "multiplicative")
    assert parse_rhs1_qi_galerkin_modes("auto") == ("additive", "multiplicative")
    assert parse_rhs1_qi_galerkin_modes("multiplicative,additive,multiplicative") == (
        "multiplicative",
        "additive",
    )
    assert parse_rhs1_qi_galerkin_modes("bad") == ("additive",)


def test_parse_qi_galerkin_dampings_is_fail_safe() -> None:
    assert parse_rhs1_qi_galerkin_dampings("", default=1.0) == (1.0, 0.5, 0.25)
    assert parse_rhs1_qi_galerkin_dampings("", default=0.3) == (0.3,)
    assert parse_rhs1_qi_galerkin_dampings("1,0.5,bad,-1,0.5", default=1.0) == (1.0, 0.5)
    assert parse_rhs1_qi_galerkin_dampings("bad,-1", default=0.2) == (0.2,)


def test_qi_galerkin_probe_selection_accepts_best_residual_reducer() -> None:
    selection = select_rhs1_qi_galerkin_probe_candidate(
        10.0,
        [
            {"mode": "additive", "damping": 1.0, "residual_norm": 12.0},
            {"mode": "multiplicative", "damping": 0.5, "residual_norm": 3.0},
            {"mode": "additive", "damping": 0.25, "residual_norm": 4.0},
        ],
    )

    assert selection.accepted is True
    assert selection.reason == "probe_reduced"
    assert selection.selected_index == 1
    assert selection.selected_mode == "multiplicative"
    assert selection.selected_damping == pytest.approx(0.5)
    assert selection.residual_after_norm == pytest.approx(3.0)
    assert selection.improvement_ratio == pytest.approx(0.3)
    assert selection.to_dict()["candidates"][1]["reduced"] is True


def test_qi_galerkin_probe_selection_rejects_non_reducers_and_nonfinite_candidates() -> None:
    selection = select_rhs1_qi_galerkin_probe_candidate(
        10.0,
        (
            RHS1QIGalerkinProbeCandidate(
                mode="additive",
                damping=1.0,
                residual_norm=10.1,
                improvement_ratio=1.01,
                reduced=False,
            ),
            RHS1QIGalerkinProbeCandidate(
                mode="multiplicative",
                damping=1.0,
                residual_norm=float("inf"),
                improvement_ratio=None,
                reduced=False,
            ),
        ),
    )

    assert selection.accepted is False
    assert selection.reason == "probe_not_reduced"
    assert selection.selected_index is None
    assert selection.residual_after_norm == pytest.approx(10.1)
    assert selection.improvement_ratio == pytest.approx(1.01)


def test_qi_galerkin_probe_selection_honors_required_drop() -> None:
    selection = select_rhs1_qi_galerkin_probe_candidate(
        10.0,
        [{"mode": "additive", "damping": 1.0, "residual_norm": 9.95}],
        min_relative_improvement=0.01,
    )

    assert selection.accepted is False
    assert selection.reason == "probe_not_reduced"
    assert selection.residual_after_norm == pytest.approx(9.95)
