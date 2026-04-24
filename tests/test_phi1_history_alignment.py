from __future__ import annotations

import numpy as np

from sfincs_jax.io import _align_phi1_history_for_output


def test_align_phi1_history_keeps_accepted_iterate_for_one_step_frozen_run() -> None:
    x0 = np.asarray([0.0, 0.0], dtype=np.float64)
    x1 = np.asarray([1.0, 2.0], dtype=np.float64)

    xs = _align_phi1_history_for_output(
        history=[x1],
        result_x=x1,
        x0_state=x0,
        use_frozen_linearization=True,
        min_iters=1,
        n_newton=1,
    )

    assert len(xs) == 1
    np.testing.assert_allclose(xs[0], x1, rtol=0.0, atol=0.0)


def test_align_phi1_history_returns_recent_accepted_tail_for_multi_step_run() -> None:
    x0 = np.asarray([0.0], dtype=np.float64)
    x1 = np.asarray([1.0], dtype=np.float64)
    x2 = np.asarray([2.0], dtype=np.float64)

    xs = _align_phi1_history_for_output(
        history=[x1, x2],
        result_x=x2,
        x0_state=x0,
        use_frozen_linearization=True,
        min_iters=1,
        n_newton=2,
    )

    assert len(xs) == 2
    np.testing.assert_allclose(xs[0], x1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xs[1], x2, rtol=0.0, atol=0.0)


def test_align_phi1_history_pads_empty_history_from_result_state() -> None:
    result = np.asarray([3.0, 4.0], dtype=np.float64)

    xs = _align_phi1_history_for_output(
        history=[],
        result_x=result,
        x0_state=None,
        use_frozen_linearization=False,
        min_iters=3,
        n_newton=4,
    )

    assert len(xs) == 4
    for x in xs:
        np.testing.assert_allclose(x, result, rtol=0.0, atol=0.0)


def test_align_phi1_history_pads_short_nonfrozen_history_with_last_iterate() -> None:
    x1 = np.asarray([1.0], dtype=np.float64)
    x2 = np.asarray([2.0], dtype=np.float64)

    xs = _align_phi1_history_for_output(
        history=[x1, x2],
        result_x=x2,
        x0_state=np.asarray([-1.0], dtype=np.float64),
        use_frozen_linearization=False,
        min_iters=0,
        n_newton=4,
    )

    assert len(xs) == 4
    np.testing.assert_allclose(xs[0], x1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xs[1], x2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xs[2], x2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xs[3], x2, rtol=0.0, atol=0.0)
