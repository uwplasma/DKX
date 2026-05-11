from __future__ import annotations

from sfincs_jax.rhs1_xblock_sparse_host_policy import (
    rhs1_fp_xblock_species_decoupled_for_host_assembly,
    rhs1_xblock_sparse_lu_default_max,
)


def test_xblock_sparse_lu_default_max_only_expands_pure_fp_host_path() -> None:
    assert rhs1_xblock_sparse_lu_default_max(
        has_fp=True,
        has_pas=False,
        build_jax_factors=False,
    ) == 3000
    assert rhs1_xblock_sparse_lu_default_max(
        has_fp=True,
        has_pas=False,
        build_jax_factors=True,
    ) == 2000
    assert rhs1_xblock_sparse_lu_default_max(
        has_fp=False,
        has_pas=True,
        build_jax_factors=False,
    ) == 2000
    assert rhs1_xblock_sparse_lu_default_max(
        has_fp=False,
        has_pas=False,
        build_jax_factors=False,
    ) == 2000


def test_fp_xblock_host_species_decoupling_matches_driver_equivalence() -> None:
    assert rhs1_fp_xblock_species_decoupled_for_host_assembly(
        n_species=1,
        preconditioner_species=0,
    )
    assert rhs1_fp_xblock_species_decoupled_for_host_assembly(
        n_species=2,
        preconditioner_species=1,
    )
    assert not rhs1_fp_xblock_species_decoupled_for_host_assembly(
        n_species=2,
        preconditioner_species=0,
    )
