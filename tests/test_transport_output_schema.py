from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.outputs.transport import transport_solver_diagnostic_arrays


def test_transport_solver_diagnostic_arrays_keep_missing_rhs_explicit() -> None:
    result = SimpleNamespace(
        residual_norms_by_rhs={1: 1.0e-8, 3: np.asarray(3.0e-8)},
        rhs_norms_by_rhs={1: 2.0, 2: 4.0, 3: np.asarray(6.0)},
    )

    arrays = transport_solver_diagnostic_arrays(result, n_rhs=3)

    np.testing.assert_allclose(
        arrays["transportResidualNorms"],
        np.asarray([1.0e-8, np.nan, 3.0e-8]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        arrays["transportRhsNorms"],
        np.asarray([2.0, 4.0, 6.0]),
    )
    np.testing.assert_allclose(
        arrays["transportRelativeResidualNorms"],
        np.asarray([5.0e-9, np.nan, 5.0e-9]),
        equal_nan=True,
    )
    assert float(arrays["transportMaxResidualNorm"]) == 3.0e-8
    assert float(arrays["transportMaxRelativeResidualNorm"]) == 5.0e-9
