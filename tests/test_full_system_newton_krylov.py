from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.problems.profile_response import phi1_newton
from sfincs_jax.v3_driver import solve_v3_full_system_newton_krylov


def test_v3_driver_newton_krylov_name_is_profile_response_facade() -> None:
    assert solve_v3_full_system_newton_krylov is phi1_newton.solve_v3_full_system_newton_krylov


def test_newton_krylov_converges_for_pas_tiny_phi1_in_kinetic_fixture() -> None:
    """End-to-end nonlinear solve smoke/parity test for includePhi1InKineticEquation=true.

    This uses a very small fixture so the Newton–Krylov iteration stays fast and robust in CI.
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    x_ref = read_petsc_vec(vec_path).values

    result = phi1_newton.solve_v3_full_system_newton_krylov(
        nml=nml,
        x0=None,
        tol=1e-9,
        max_newton=10,
        gmres_tol=1e-10,
        gmres_restart=80,
        gmres_maxiter=300,
    )
    x = np.asarray(result.x)

    # Converged residual and solution parity.
    assert float(result.residual_norm) < 1e-9
    np.testing.assert_allclose(x, x_ref, rtol=0, atol=5e-8)
