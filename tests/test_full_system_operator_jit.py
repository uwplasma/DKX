from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.petsc_binary import read_petsc_vec
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres
from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator, apply_v3_full_system_operator_jit, full_system_operator_from_namelist


def test_full_system_operator_can_jit_compile() -> None:
    """Regression test: V3FullSystemOperator must be a JAX PyTree usable under jit."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x = jnp.asarray(read_petsc_vec(vec_path).values)

    y = np.asarray(apply_v3_full_system_operator(op, x))
    y_jit = np.asarray(apply_v3_full_system_operator_jit(op, x))
    np.testing.assert_allclose(y_jit, y, rtol=0, atol=1e-15)


def test_full_system_operator_reuses_prebuilt_grids_and_geometry() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op_default = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op_reused = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)
    x = jnp.asarray(read_petsc_vec(vec_path).values)

    y_default = np.asarray(apply_v3_full_system_operator(op_default, x))
    y_reused = np.asarray(apply_v3_full_system_operator(op_reused, x))

    np.testing.assert_allclose(y_reused, y_default, rtol=0, atol=1e-15)


def test_full_system_linear_gmres_reuses_prebuilt_operator() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)

    res_default = solve_v3_full_system_linear_gmres(nml=nml, tol=1e-10, emit=None)
    res_reused = solve_v3_full_system_linear_gmres(nml=nml, op=op, tol=1e-10, emit=None)

    np.testing.assert_allclose(np.asarray(res_reused.x), np.asarray(res_default.x), rtol=0, atol=1e-11)
    assert float(res_reused.residual_norm) <= float(res_default.residual_norm) * 10.0 + 1e-12
